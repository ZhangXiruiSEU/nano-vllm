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
        draft_config.gpu_memory_utilization = draft_config.draft_gpu_memory_utilization
        draft_config.num_spec_tokens = 0
        # modelrunner 目前只是用 rank 编号给相同的 GPU 所以 直接加上 config.tensor_parallel_size 即可
        # 其实目前这个实验也不涉及 TP 这样写是为了维护方便吧
        self.draft_model_runner = ModelRunner(config=draft_config,device=draft_config.draft_device_id, rank=0, event=self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 默认分词表是完全一致，EOS也是一致的
        # self.tokenizer_draft = AutoTokenizer.from_pretrained(draft_config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 两个模型共同控制一个 scheduler，控制一份 sequence 所以只能有一个
        # 但是控制的block 块的数量必须一致，
        # 因为 scheduler 只有一个；里面 sequence 对应的块的列表只有一个，
        # 假设块的列表指向了块的数量多的，那小的就报错了
        pass
        config.num_kvcache_blocks = min (config.num_kvcache_blocks,draft_config.num_kvcache_blocks)
        self.scheduler = Scheduler(config)

        self.num_spec_tokens = config.num_spec_tokens
        self.draft_catchup_tokens: dict[int, list[int]] = {}
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
        return_stats: bool = False,
    ) -> list[dict] | dict:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        stats = self._empty_generation_stats()
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, step_stats = self.step()
            self._add_generation_stats(stats, step_stats)
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
                "Accepted": stats["verified_tokens"],
                "Bonus": stats["bonus_tokens"],
                "Resample": stats["resampled_tokens"],
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        self.last_generation_stats = stats
        if return_stats:
            return {"outputs": outputs, "stats": stats}
        return outputs

    # # 总体控制投机采样的流程
    def step(self):
        stats = self._empty_generation_stats()
        step_start = perf_counter()
        schedule_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule(self.num_spec_tokens + 1)
        stats["schedule_time"] += perf_counter() - schedule_start

        if is_prefill:
            self._sync_cuda()
            prefill_start = perf_counter()
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            self._clear_draft_catchup(seqs)
            before_completion_lens = [seq.num_completion_tokens for seq in seqs]

            token_ids = self.model_runner.call("run", seqs, True)
            _ = self.draft_model_runner.call("run", seqs, True)

            self.scheduler.postprocess(seqs, token_ids, True)
            self._sync_cuda()
            stats["prefill_time"] += perf_counter() - prefill_start
            generated_lens = [
                seq.num_completion_tokens - before_len
                for seq, before_len in zip(seqs, before_completion_lens)
            ]
            prefill_generated_tokens = sum(generated_lens)
            stats["total_tokens"] += prefill_generated_tokens
            stats["bonus_tokens"] += prefill_generated_tokens

        else:
            stats["decode_steps"] += 1
            self._sync_cuda()
            draft_start = perf_counter()
            draft_token_ids, draft_token_probs, draft_full_probs, draft_lens = self._draft_propose(seqs)
            self._sync_cuda()
            stats["draft_propose_time"] += perf_counter() - draft_start
            stats["draft_proposed_tokens"] += sum(draft_lens)

            # 第一版先要求每条 seq 都 propose 满 K 个，避免 ragged verify/stack 问题。
            assert all(x == self.num_spec_tokens for x in draft_lens)
            self._check_reserved_verify_slots(seqs, draft_token_ids)
            self._sync_cuda()
            target_start = perf_counter()
            target_token_probs, target_full_probs, bonus_full_probs = self.model_runner.call(
                "run_verify",
                seqs,
                draft_token_ids,
            )
            self._sync_cuda()
            stats["target_verify_time"] += perf_counter() - target_start
            verify_timings = getattr(self.model_runner, "last_verify_timings", None)
            if verify_timings is not None:
                stats["verify_prepare_time"] += verify_timings["prepare"]
                stats["verify_model_time"] += verify_timings["model"]
                stats["verify_logits_time"] += verify_timings["logits"]
                stats["verify_probs_time"] += verify_timings["probs"]
                stats["verify_graph_steps"] += verify_timings["graph"]

            # 如果 draft/target 在不同 GPU，这里先搬到 target 分布所在 device。
            self._sync_cuda()
            transfer_start = perf_counter()
            device = target_full_probs.device
            draft_token_probs = draft_token_probs.to(device)
            draft_full_probs = draft_full_probs.to(device)
            self._sync_cuda()
            stats["prob_transfer_time"] += perf_counter() - transfer_start
            self._sync_cuda()
            accept_start = perf_counter()
            (
                accepted_token_ids,
                accepted_draft_lens,
                target_cached_advance_lens,
                resampled_lens,
                bonus_lens,
                trace_entries,
            ) = self._accept_reject(
                draft_token_ids,
                draft_token_probs,
                draft_full_probs,
                target_token_probs,
                target_full_probs,
                bonus_full_probs,
                draft_lens,
            )
            stats["spec_trace"].extend(trace_entries)
            self._sync_cuda()
            stats["accept_reject_time"] += perf_counter() - accept_start

            postprocess_start = perf_counter()
            committed_token_ids = self.scheduler.postprocess_speculative(
                seqs,
                accepted_token_ids,
                target_cached_advance_lens,
            )
            self._record_draft_catchup(seqs, committed_token_ids, accepted_draft_lens)
            self._update_decode_stats(stats, committed_token_ids, accepted_draft_lens, resampled_lens, bonus_lens)
            num_tokens = -sum(len(x) for x in committed_token_ids)
            stats["committed_decode_tokens"] += -num_tokens
            stats["postprocess_time"] += perf_counter() - postprocess_start
            self._sync_cuda()
            stats["decode_time"] += perf_counter() - step_start

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, stats
    
    def _draft_propose(self, seqs):
        # 先保存每条 canonical Sequence 的状态，因为 draft propose 会临时 append token。
        snapshots = []
        # 遍历当前这一批正在 decode 的 Sequence。
        for seq in seqs:
            # 记录 token 序列、长度、KV 已缓存长度、本轮调度 token 数和最后一个 token。
            snapshots.append((
                # token_ids 是 list，可变对象，必须复制一份，避免后面 append 影响快照。
                list(seq.token_ids),
                # num_tokens 表示当前 canonical 序列长度。
                seq.num_tokens,
                # num_cached_tokens 表示 target/canonical 视角下已经有 KV 的 token 数。
                seq.num_cached_tokens,
                # num_scheduled_tokens 表示 scheduler 本轮准备让 runner 处理多少 token。
                seq.num_scheduled_tokens,
                # last_token 是下一步 decode 的输入 token。
                seq.last_token,
            ))

        # 当前 batch 里有多少条 seq。
        batch_size = len(seqs)
        # 每条 seq 最终 draft 出来的 token id 列表，当前要求每条都是 B x K。
        draft_token_ids = [[] for _ in range(batch_size)]
        # 按 step 保存 draft 采样 token 的概率；最后会 stack 成 [B, K]。
        draft_token_probs_steps = []
        # 按 step 保存 draft 的完整 vocab 概率分布；最后会 stack 成 [B, K, V]。
        draft_full_probs_steps = []

        # 最多提出 self.num_spec_tokens 个 draft token。
        for step_idx in range(self.num_spec_tokens):
            # 每一步 decode 前，只检查 scheduler 是否已经为当前写入位置预留了 block。
            for i, seq in enumerate(seqs):
                # 当前 decode 会写 catch-up token 加 seq.last_token 这一段。
                catchup_len = len(self.draft_catchup_tokens.get(seq.seq_id, [])) if step_idx == 0 else 0
                write_start = len(seq) - 1 - catchup_len
                write_end = len(seq)
                # 根据写入区间计算它需要覆盖到哪个 KV block。
                required_num_blocks = (write_end + self.scheduler.block_size - 1) // self.scheduler.block_size
                # 如果这里越界，说明 scheduler 没有正确预留 lookahead block。
                if required_num_blocks > len(seq.block_table):
                    raise RuntimeError(
                        f"scheduler did not reserve draft KV block: "
                        f"seq_id={seq.seq_id}, write_start={write_start}, write_end={write_end}, "
                        f"required={required_num_blocks}, actual={len(seq.block_table)}"
                    )
                # draft 每一步只做单 token decode，所以临时把 scheduled token 数设成 1。
                seq.num_scheduled_tokens = 1

            if step_idx == 0 and any(seq.seq_id in self.draft_catchup_tokens for seq in seqs):
                tokens_per_seq = []
                start_offsets = []
                for seq in seqs:
                    catchup_tokens = self.draft_catchup_tokens.pop(seq.seq_id, [])
                    tokens_per_seq.append(catchup_tokens + [seq.last_token])
                    start_offsets.append(-len(tokens_per_seq[-1]))

                token_ids, selected_probs, full_probs = self.draft_model_runner.call(
                    "run_extend_with_probs",
                    seqs,
                    tokens_per_seq,
                    start_offsets,
                )
            else:
                # 调用 draft runner 做一次 decode，并返回采样 token、该 token 概率、完整 vocab 概率。
                token_ids, selected_probs, full_probs = self.draft_model_runner.call(
                    # run_with_probs 是 draft 专用路径：既采样 token，也保留概率用于 accept/reject。
                    "run_with_probs",
                    # scheduler 已经筛掉了空间不足的 seq，这里直接对整批 scheduled seq propose。
                    seqs,
                    # False 表示 decode 阶段，不是 prefill。
                    False,
                )

            # 保存这一 draft step 中每条 seq 的采样 token 概率。
            step_token_probs = torch.zeros(
                # 每个原 batch 位置放一个 draft token 的概率。
                batch_size,
                # dtype 跟 draft runner 返回的 selected_probs 一致。
                dtype=selected_probs.dtype,
                # device 跟 selected_probs 一致，避免跨设备 stack/copy。
                device=selected_probs.device,
            )
            step_full_probs = torch.zeros(
                # 每个原 batch 位置放一份完整 vocab 概率。
                batch_size,
                # vocab 大小取 full_probs 的最后一维。
                full_probs.size(-1),
                # dtype 跟 full_probs 一致。
                dtype=full_probs.dtype,
                # device 跟 full_probs 一致。
                device=full_probs.device,
            )

            # 把 draft runner 返回的 batch 结果写回每条 seq 的 proposal 缓冲。
            for seq_idx, seq in enumerate(seqs):
                # seq_idx 同时是原 batch 下标和 runner 返回的 batch 下标。
                token_id = token_ids[seq_idx]

                # 记录这条 seq 在当前 step 采出的 draft token。
                draft_token_ids[seq_idx].append(token_id)
                # 保存 draft 对自己采样 token 的概率 p_draft(draft_i)。
                step_token_probs[seq_idx] = selected_probs[seq_idx]
                # 保存 draft 在该位置的完整分布 q_i(x)，拒绝时 residual sampling 要用。
                step_full_probs[seq_idx] = full_probs[seq_idx]

                # 临时推进，让下一步 draft decode 基于刚采出的 token
                # append_token 会更新 token_ids、num_tokens 和 last_token。
                seq.append_token(token_id)
                # draft runner 已经为当前输入位置写了 KV，所以这里临时推进 cached 计数。
                seq.num_cached_tokens += 1
                # 下一轮 draft decode 仍然是每条 seq 只 decode 1 个 token。
                seq.num_scheduled_tokens = 1

            # 保存当前 step 的 [B] token 概率。
            draft_token_probs_steps.append(step_token_probs)
            # 保存当前 step 的 [B, V] 完整概率。
            draft_full_probs_steps.append(step_full_probs)

        # 回滚 canonical Sequence；block_table 不回滚，预留 block 留给后续 verify/commit。
        for seq, snapshot in zip(seqs, snapshots):
            # 拆出进入 draft propose 前保存的原始状态。
            token_ids0, num_tokens, num_cached, num_scheduled, last_token = snapshot
            # 恢复 canonical token 序列，丢弃临时 append 的 draft token。
            seq.token_ids = token_ids0
            # 恢复 canonical 序列长度。
            seq.num_tokens = num_tokens
            # 恢复 canonical KV 已缓存 token 数。
            seq.num_cached_tokens = num_cached
            # 恢复 scheduler 原本记录的本轮调度 token 数。
            seq.num_scheduled_tokens = num_scheduled
            # 恢复进入 draft propose 前的 last_token。
            seq.last_token = last_token

        # 如果至少 propose 过一步，就把按 step 存的概率拼成张量。
        if draft_token_probs_steps:
            # [B, K]
            draft_token_probs = torch.stack(draft_token_probs_steps, dim=1)
            # [B, K, V]
            draft_full_probs = torch.stack(draft_full_probs_steps, dim=1)
        else:
            # 没有 propose 出 token 时，仍然返回合法的空张量。
            device = self.draft_model_runner.kv_cache.device
            # token 概率为空，形状是 [B, 0]。
            draft_token_probs = torch.empty((batch_size, 0), device=device)
            # 完整分布也为空；这里 vocab 维度没有可用 full_probs，所以用 0。
            draft_full_probs = torch.empty((batch_size, 0, 0), device=device)
        # 每条 seq 实际 propose 了多少 token；后续 verify/accept 用它处理 ragged 情况。
        draft_lens = [len(x) for x in draft_token_ids]
        # 返回 draft token、draft token 概率、draft 完整分布、每条 seq 的 draft 长度。
        return draft_token_ids, draft_token_probs, draft_full_probs, draft_lens
        
    def _record_draft_catchup(self, seqs, committed_token_ids, accepted_draft_lens):
        for seq, token_ids, accepted_draft_len in zip(seqs, committed_token_ids, accepted_draft_lens):
            self.draft_catchup_tokens.pop(seq.seq_id, None)
            if seq.is_finished:
                continue
            if accepted_draft_len == self.num_spec_tokens and len(token_ids) == self.num_spec_tokens + 1:
                self.draft_catchup_tokens[seq.seq_id] = [int(token_ids[self.num_spec_tokens - 1])]

    def _clear_draft_catchup(self, seqs):
        for seq in seqs:
            self.draft_catchup_tokens.pop(seq.seq_id, None)

    def _empty_generation_stats(self):
        return {
            "total_tokens": 0,
            "verified_tokens": 0,
            "bonus_tokens": 0,
            "resampled_tokens": 0,
            "draft_proposed_tokens": 0,
            "committed_decode_tokens": 0,
            "decode_steps": 0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "schedule_time": 0.0,
            "draft_propose_time": 0.0,
            "target_verify_time": 0.0,
            "verify_prepare_time": 0.0,
            "verify_model_time": 0.0,
            "verify_logits_time": 0.0,
            "verify_probs_time": 0.0,
            "verify_graph_steps": 0.0,
            "prob_transfer_time": 0.0,
            "accept_reject_time": 0.0,
            "postprocess_time": 0.0,
            "spec_trace": [],
        }

    def _add_generation_stats(self, total_stats, step_stats):
        for key in total_stats:
            if key == "spec_trace":
                total_stats[key].extend(step_stats[key])
            else:
                total_stats[key] += step_stats[key]

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _update_decode_stats(
        self,
        stats,
        committed_token_ids,
        accepted_draft_lens,
        resampled_lens,
        bonus_lens,
    ):
        for token_ids, accepted_draft_len, resampled_len, bonus_len in zip(
            committed_token_ids,
            accepted_draft_lens,
            resampled_lens,
            bonus_lens,
        ):
            committed_len = len(token_ids)
            verified_len = min(accepted_draft_len, committed_len)
            stats["total_tokens"] += committed_len
            stats["verified_tokens"] += verified_len

            extra_committed_len = max(0, committed_len - verified_len)
            if resampled_len:
                stats["resampled_tokens"] += min(resampled_len, extra_committed_len)
            elif bonus_len:
                stats["bonus_tokens"] += min(bonus_len, extra_committed_len)



    def _check_reserved_verify_slots(self, seqs, draft_token_ids):
        for seq, draft_ids in zip(seqs, draft_token_ids):
            num_verify_tokens = len(draft_ids) + 1

            # scheduler 已经按 speculative lookahead 预留过。
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
    ) -> tuple[list[list[int]], list[int], list[int], list[int], list[int], list[dict]]:
        accepted_token_ids = []
        accepted_draft_lens = []
        target_cached_advance_lens = []
        resampled_lens = []
        bonus_lens = []
        trace_entries = []

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
            num_accepted_draft_tokens = 0
            rejected_index = None
            rejected_token_id = None
            resampled_token_id = None
            bonus_token_id = None

            for i in range(k):
                draft_token_id = draft_ids[i]

                if accept_mask[b][i]:
                    seq_accepted.append(draft_token_id)
                    num_accepted_draft_tokens += 1
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

                rejected_index = i
                rejected_token_id = int(draft_token_id)
                resampled_token_id = int(self.model_runner.sampler.sample_from_probs(residual).item())
                seq_accepted.append(resampled_token_id)
                rejected = True
                break

            if not rejected:
                bonus_token_id = int(self.model_runner.sampler.sample_from_probs(bonus_full_probs[b]).item())
                seq_accepted.append(bonus_token_id)

            accepted_token_ids.append(seq_accepted)
            accepted_draft_lens.append(num_accepted_draft_tokens)
            # Target verify runs old last_token plus the accepted draft prefix.
            target_cached_advance_lens.append(1 + num_accepted_draft_tokens)
            resampled_lens.append(1 if rejected else 0)
            bonus_lens.append(0 if rejected else 1)
            trace_entries.append({
                "batch_index": b,
                "draft_token_ids": [int(x) for x in draft_ids],
                "accepted_draft_len": num_accepted_draft_tokens,
                "rejected_index": rejected_index,
                "rejected_token_id": rejected_token_id,
                "resampled_token_id": resampled_token_id,
                "bonus_token_id": bonus_token_id,
                "committed_token_ids": [int(x) for x in seq_accepted],
            })

        return accepted_token_ids, accepted_draft_lens, target_cached_advance_lens, resampled_lens, bonus_lens, trace_entries
