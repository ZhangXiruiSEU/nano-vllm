import pickle
import torch
import torch.distributed as dist
from time import perf_counter
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from typing import Any

class ModelRunner:

    def __init__(self, config: Config, device:int, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        if self.world_size > 1 :
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(device + rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        print(config.model, config.gpu_memory_utilization, config.num_kvcache_blocks)
        print(self.kv_cache.numel() * self.kv_cache.element_size() / 1024**3, "GB")
        if not self.enforce_eager:
            self.capture_cudagraph()
            if self.config.num_spec_tokens > 0:
                self.capture_verify_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "verify_graphs"):
                del self.verify_graphs
        torch.cuda.synchronize()
        if self.world_size > 1:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            token_pos = len(seq) - 1
            block_idx = token_pos // self.block_size
            block_offset = token_pos % self.block_size
            input_ids.append(seq.last_token)
            positions.append(token_pos)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[block_idx] * self.block_size + block_offset)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions
    


    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
    
    def run_with_probs(self, seqs: list[Sequence], is_prefill: bool):
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        if self.rank == 0:
            token_ids, selected_probs, full_probs = self.sampler.forward_with_probs(logits, temperatures)
            token_ids = token_ids.tolist()
        else:
            token_ids, selected_probs, full_probs = None, None, None

        reset_context()
        return token_ids, selected_probs, full_probs # probs 仍然是GPU Tensor

    @torch.inference_mode()
    def run_extend_with_probs(self, seqs, tokens_per_seq, start_offsets=None):
        input_ids, positions, extend_lens = self.prepare_extend(
            seqs,
            tokens_per_seq,
            start_offsets,
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits_all(hidden_states)

        if self.rank != 0:
            reset_context()
            return None, None, None

        last_logits = []
        offset = 0
        for extend_len in extend_lens:
            last_logits.append(logits[offset + extend_len - 1])
            offset += extend_len
        last_logits = torch.stack(last_logits, dim=0)

        token_ids, selected_probs, full_probs = self.sampler.forward_with_probs(last_logits, temperatures)
        token_ids = token_ids.tolist()

        reset_context()
        return token_ids, selected_probs, full_probs

    @torch.inference_mode() 
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_verify_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        verify_len = config.num_spec_tokens + 1
        if verify_len <= 1:
            return

        # First version: capture only batch size 1 and keep eager fallback for other sizes.
        max_bs = 1
        max_num_blocks = (config.max_model_len + verify_len + self.block_size - 1) // self.block_size
        max_seqlen_k = config.max_model_len + verify_len
        input_ids = torch.zeros(max_bs * verify_len, dtype=torch.int64)
        positions = torch.zeros(max_bs * verify_len, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs * verify_len, dtype=torch.int32)
        cu_seqlens_q = torch.arange(0, (max_bs + 1) * verify_len, verify_len, dtype=torch.int32)
        cu_seqlens_k = torch.arange(0, (max_bs + 1) * verify_len, verify_len, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs * verify_len, hf_config.hidden_size)
        self.verify_graph_bs = [1]
        self.verify_graphs = {}

        for bs in reversed(self.verify_graph_bs):
            n = bs * verify_len
            graph = torch.cuda.CUDAGraph()
            set_context(
                True,
                cu_seqlens_q[:bs + 1],
                cu_seqlens_k[:bs + 1],
                verify_len,
                max_seqlen_k,
                slot_mapping[:n],
                None,
                block_tables[:bs],
            )
            outputs[:n] = self.model(input_ids[:n], positions[:n])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:n] = self.model(input_ids[:n], positions[:n])
            self.verify_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.verify_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
            verify_len=verify_len,
            max_seqlen_k=max_seqlen_k,
        )

    @torch.inference_mode()
    def run_verify_model_graph(self, input_ids: torch.Tensor, positions: torch.Tensor, bs: int):
        if not hasattr(self, "verify_graphs") or bs not in self.verify_graphs:
            return None
        graph_vars = self.verify_graph_vars
        verify_len = graph_vars["verify_len"]
        if input_ids.numel() != bs * verify_len:
            return None

        context = get_context()
        if context.block_tables is None:
            return None
        if context.block_tables.size(1) > graph_vars["block_tables"].size(1):
            return None
        if context.max_seqlen_k > graph_vars["max_seqlen_k"]:
            return None

        n = bs * verify_len
        graph_vars["input_ids"][:n] = input_ids
        graph_vars["positions"][:n] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:n] = context.slot_mapping
        graph_vars["cu_seqlens_q"][:bs + 1] = context.cu_seqlens_q
        graph_vars["cu_seqlens_k"][:bs + 1] = context.cu_seqlens_k
        graph_vars["block_tables"].zero_()
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        set_context(
            True,
            graph_vars["cu_seqlens_q"][:bs + 1],
            graph_vars["cu_seqlens_k"][:bs + 1],
            verify_len,
            graph_vars["max_seqlen_k"],
            graph_vars["slot_mapping"][:n],
            None,
            graph_vars["block_tables"][:bs],
        )
        self.verify_graphs[bs].replay()
        return graph_vars["outputs"][:n]

    # 有时候是会送进去比如 k 个 token 如验证，或者 2 个 token 如完全验证通过
    def prepare_prefill_mini(self):
        pass

    # 让draft 模型自回归生成 k 个新 token
    def draft_propose_k(self):
        pass

    # 让target 模型去验证 k 个新 token 能不能采用
    def target_varify_k(self):
        pass

    def prepare_extend(
        self,
        seqs: list[Sequence],
        tokens_per_seq: list[list[int]],
        start_offsets: list[int] | None = None,
    ):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        extend_lens = []

        if start_offsets is None:
            start_offsets = [0] * len(seqs)

        for seq, token_ids, start_offset in zip(seqs, tokens_per_seq, start_offsets):
            start = len(seq) + start_offset
            seqlen_q = len(token_ids)
            end = start + seqlen_q
            seqlen_k = end

            input_ids.extend(token_ids)
            positions.extend(range(start, end))
            extend_lens.append(seqlen_q)

            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size

            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size

                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size

                slot_mapping.extend(range(slot_start, slot_end))

        assert len(input_ids) == len(slot_mapping)

        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions, extend_lens
    
    def prepare_verify(self, seqs: list[Sequence], draft_token_ids: list[list[int]]):
        tokens_per_seq = [
            [seq.last_token] + list(draft_ids)
            for seq, draft_ids in zip(seqs, draft_token_ids)
        ]
        start_offsets = [-1] * len(seqs)
        return self.prepare_extend(seqs, tokens_per_seq, start_offsets)
    
    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence], draft_token_ids: list[list[int]]):
        timings = {
            "prepare": 0.0,
            "model": 0.0,
            "logits": 0.0,
            "probs": 0.0,
            "graph": 0.0,
        }
        torch.cuda.synchronize()
        t = perf_counter()
        input_ids, positions, verify_lens = self.prepare_verify(seqs, draft_token_ids)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        torch.cuda.synchronize()
        timings["prepare"] = perf_counter() - t

        t = perf_counter()
        hidden_states = None
        if not self.enforce_eager and all(x == self.config.num_spec_tokens + 1 for x in verify_lens):
            hidden_states = self.run_verify_model_graph(input_ids, positions, len(seqs))
            timings["graph"] = 1.0 if hidden_states is not None else 0.0
        if hidden_states is None:
            hidden_states = self.model(input_ids, positions)
        torch.cuda.synchronize()
        timings["model"] = perf_counter() - t

        t = perf_counter()
        logits = self.model.compute_logits_all(hidden_states)
        torch.cuda.synchronize()
        timings["logits"] = perf_counter() - t

        if self.rank != 0:
            reset_context()
            return None, None, None

        t = perf_counter()
        target_token_probs = []
        target_full_probs = []
        bonus_logits = []
        offset = 0

        for i, draft_ids in enumerate(draft_token_ids):
            q_len = verify_lens[i]
            seq_logits = logits[offset: offset + q_len]
            offset += q_len

            # verify 输入是 [last_token] + draft_ids
            # seq_logits[:-1] 对应每个 draft token 的预测分布
            verify_logits = seq_logits[:-1].float().div(temperatures[i])
            probs = torch.softmax(verify_logits, dim=-1)  # [K, V]

            draft_tensor = torch.tensor(draft_ids, dtype=torch.long, device=probs.device)
            selected_probs = probs.gather(1, draft_tensor.unsqueeze(1)).squeeze(1)  # [K]

            target_token_probs.append(selected_probs)
            target_full_probs.append(probs)

            # seq_logits[-1] 是全接受时 bonus token 的预测分布
            bonus_logits.append(seq_logits[-1])

        target_token_probs = torch.stack(target_token_probs, dim=0)  # [B, K]
        target_full_probs = torch.stack(target_full_probs, dim=0)    # [B, K, V]

        bonus_logits = torch.stack(bonus_logits, dim=0).float()
        bonus_logits = bonus_logits.div(temperatures.unsqueeze(dim=1))
        bonus_full_probs = torch.softmax(bonus_logits, dim=-1)       # [B, V]
        torch.cuda.synchronize()
        timings["probs"] = perf_counter() - t
        self.last_verify_timings = timings

        reset_context()
        return target_token_probs, target_full_probs, bonus_full_probs

        '''  draft:
            draft_token_ids
            draft_token_probs  [B, K]
            draft_full_probs   [B, K, V]

        target:
            target_token_probs [B, K]
            target_full_probs  [B, K, V]
            bonus_full_probs   [B, V]
        '''


    @torch.inference_mode()
    def extend_kv(self, seqs, tokens_per_seq, start_offsets=None):
        input_ids, positions, _ = self.prepare_extend(
            seqs,
            tokens_per_seq,
            start_offsets,
        )
        self.model(input_ids, positions)
        reset_context()
