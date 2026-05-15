import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import SpConfig
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence, SequenceStatus
import torch
import copy
class SpLLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(SpConfig)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = SpConfig(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config=config, device=0, rank=0, event=self.events)
        # 整理一下 Config 送进 draft 的 runner 里面
        draft_config = copy.copy(config)
        draft_config.model = draft_config.draft_model
        draft_config.hf_config = draft_config.draft_hf_config
        # modelrunner 目前只是用 rank 编号给相同的 GPU 所以 直接加上 config.tensor_parallel_size 即可
        # 其实目前这个实验也不涉及 TP 这样写是为了维护方便吧
        self.draft_model_runner = ModelRunner(config=draft_config,device=config.tensor_parallel_size, rank=0, event=self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 默认分词表是完全一致，EOS也是一致的
        # self.tokenizer_draft = AutoTokenizer.from_pretrained(draft_config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 两个模型共同控制一个 scheduler，控制一份 sequence 所以只能有一个
        # 但是控制的block 块的数量必须一致，
        # 因为 scheduler 只有一个；里面 sequence 对应的块的列表只有一个，
        # 假设块的列表指向了块的数量多的，那小的就报错了
        config.num_kvcache_blocks = min (config.num_kvcache_blocks,draft_config.num_kvcache_blocks)
        self.scheduler = Scheduler(config)

        self.num_spec_tokens = config.num_spec_tokens
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        self.draft_model_runner.call("exit")
        del self.draft_model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 如果是str 的话就转换成prompt 
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 转为我们的sequence对象
        seq = Sequence(prompt, sampling_params)
        # 送进scheduler
        self.scheduler.add(seq)

    def step_old(self):
        # scheduler 先去 schedule  这个实现中 PD 是分开的，所以一次 schedule 返回 PD 之一
        seqs, is_prefill = self.scheduler.schedule()
        # 
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    # # 总体控制投机采样的流程
    def step(self):
        seqs, is_prefill = self.scheduler.schedule(self.num_spec_tokens + 1)

        if is_prefill:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)

            token_ids = self.model_runner.call("run", seqs, True)
            _ = self.draft_model_runner.call("run", seqs, True)

            self.scheduler.postprocess(seqs, token_ids, True)

        else:
            draft_token_ids, draft_token_probs, draft_full_probs, draft_lens = self._draft_propose(seqs)

            # 第一版先要求每条 seq 都 propose 满 K 个，避免 ragged verify/stack 问题。
            assert all(x == self.num_spec_tokens for x in draft_lens)

            self._reserve_verify_slots(seqs, draft_token_ids)

            target_token_probs, target_full_probs, bonus_full_probs = self.model_runner.call(
                "verify",
                seqs,
                draft_token_ids,
            )

            # 如果 draft/target 在不同 GPU，这里先搬到 target 分布所在 device。
            device = target_full_probs.device
            draft_token_probs = draft_token_probs.to(device)
            draft_full_probs = draft_full_probs.to(device)

            accepted_token_ids = self._accept_reject(
                draft_token_ids,
                draft_token_probs,
                draft_full_probs,
                target_token_probs,
                target_full_probs,
                bonus_full_probs,
                draft_lens,
            )

            self.scheduler.postprocess_speculative(seqs, accepted_token_ids)
            num_tokens = -sum(len(x) for x in accepted_token_ids)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
    
    def _draft_propose(self, seqs):
        snapshots = []
        for seq in seqs:
            snapshots.append((
                list(seq.token_ids),
                seq.num_tokens,
                seq.num_cached_tokens,
                seq.num_scheduled_tokens,
                seq.last_token,
            ))

        batch_size = len(seqs)
        draft_token_ids = [[] for _ in range(batch_size)]
        draft_token_probs_steps = []
        draft_full_probs_steps = []

        active = [True] * batch_size

        for _ in range(self.num_spec_tokens):
            active_indices = [i for i, flag in enumerate(active) if flag]
            if not active_indices:
                break

            active_seqs = [seqs[i] for i in active_indices]

            # 每一步 decode 前，给 active seq 当前 last_token 的 KV 写入位置确保 block。
            for i in active_indices:
                seq = seqs[i]
                if not self.scheduler.block_manager.can_append(seq):
                    active[i] = False
                    continue
                self.scheduler.block_manager.may_append(seq)

            active_indices = [i for i, flag in enumerate(active) if flag]
            if not active_indices:
                break

            active_seqs = [seqs[i] for i in active_indices]

            token_ids, selected_probs, full_probs = self.draft_model_runner.call(
                "run_with_probs",
                active_seqs,
                False,
            )

            # 对齐回原 batch 维度；inactive 的位置用 0 占位，后面靠长度/active mask 忽略。
            step_token_probs = torch.zeros(
                batch_size,
                dtype=selected_probs.dtype,
                device=selected_probs.device,
            )
            step_full_probs = torch.zeros(
                batch_size,
                full_probs.size(-1),
                dtype=full_probs.dtype,
                device=full_probs.device,
            )

            for local_idx, seq_idx in enumerate(active_indices):
                token_id = token_ids[local_idx]
                seq = seqs[seq_idx]

                draft_token_ids[seq_idx].append(token_id)
                step_token_probs[seq_idx] = selected_probs[local_idx]
                step_full_probs[seq_idx] = full_probs[local_idx]

                # 临时推进，让下一步 draft decode 基于刚采出的 token
                seq.append_token(token_id)
                seq.num_cached_tokens += 1
                seq.num_scheduled_tokens = 1

            draft_token_probs_steps.append(step_token_probs)
            draft_full_probs_steps.append(step_full_probs)

        # 回滚 canonical Sequence；block_table 不回滚，预留 block 留给后续 verify/commit。
        for seq, snapshot in zip(seqs, snapshots):
            token_ids0, num_tokens, num_cached, num_scheduled, last_token = snapshot
            seq.token_ids = token_ids0
            seq.num_tokens = num_tokens
            seq.num_cached_tokens = num_cached
            seq.num_scheduled_tokens = num_scheduled
            seq.last_token = last_token

        if draft_token_probs_steps:
            # [B, K]
            draft_token_probs = torch.stack(draft_token_probs_steps, dim=1)
            # [B, K, V]
            draft_full_probs = torch.stack(draft_full_probs_steps, dim=1)
        else:
            device = self.draft_model_runner.kv_cache.device
            draft_token_probs = torch.empty((batch_size, 0), device=device)
            draft_full_probs = torch.empty((batch_size, 0, 0), device=device)
        draft_lens = [len(x) for x in draft_token_ids]
        return draft_token_ids, draft_token_probs, draft_full_probs, draft_lens
        


    def _reserve_verify_slots(self, seqs, draft_token_ids):
        for seq, draft_ids in zip(seqs, draft_token_ids):
            num_verify_tokens = len(draft_ids) + 1

            # scheduler 已经按 self.num_spec_tokens + 1 预留过。
            # 这里仅做防御性检查，避免 prepare_verify 访问越界。
            start = len(seq) - 1
            end = start + num_verify_tokens
            required_num_blocks = (end + self.scheduler.block_size - 1) // self.scheduler.block_size

            if len(seq.block_table) < required_num_blocks:
                raise RuntimeError(
                    f"not enough KV blocks for target verify: "
                    f"required={required_num_blocks}, actual={len(seq.block_table)}, "
                    f"num_verify_tokens={num_verify_tokens}"
                )

    import random
    def _accept_reject(
        self,
        draft_token_ids: list[list[int]],
        draft_token_probs: torch.Tensor,   # [B, K]
        draft_full_probs: torch.Tensor,    # [B, K, V]
        target_token_probs: torch.Tensor,  # [B, K]
        target_full_probs: torch.Tensor,   # [B, K, V]
        bonus_full_probs: torch.Tensor,    # [B, V]
        draft_lens: list[int],
    ) -> list[list[int]]:
        accepted_token_ids = []

        accept_ratio = torch.clamp(
            target_token_probs / draft_token_probs.clamp_min(1e-12),
            max=1.0,
        )
        accept_mask = torch.rand_like(accept_ratio) <= accept_ratio
        accept_mask = accept_mask.cpu().tolist()

        for b, draft_ids in enumerate(draft_token_ids):
            seq_accepted = []
            k = draft_lens[b]
            rejected = False

            for i in range(k):
                draft_token_id = draft_ids[i]

                if accept_mask[b][i]:
                    seq_accepted.append(draft_token_id)
                    continue

                residual = torch.clamp(
                    target_full_probs[b, i] - draft_full_probs[b, i],
                    min=0,
                )
                residual_sum = residual.sum()

                if residual_sum <= 0:
                    residual = target_full_probs[b, i]
                else:
                    residual = residual / residual_sum

                seq_accepted.append(int(self.model_runner.sampler.sample_from_probs(residual).item()))
                rejected = True
                break

            if not rejected:
                seq_accepted.append(int(self.model_runner.sampler.sample_from_probs(bonus_full_probs[b]).item()))

            accepted_token_ids.append(seq_accepted)

        return accepted_token_ids