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
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
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

    # 总体控制投机采样的流程
    def speculative_decoding(self):
        # scheduler 先去 schedule  这一步两个模型都一样吧
        seqs, is_prefill = self.scheduler.schedule()
        
        if is_prefill:
        # prompt prefill
        else:
        # scheduler 已经为 1 token 做 may_append
        # K=1 speculative 先跑通

        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
    
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

        if is_prefill:
            # target prefill: 写 target KV，并采样 canonical token
            token_ids = self.model_runner.call("run", seqs, True)

            # draft prefill: 写 draft KV，但返回 token 丢掉
            _ = self.draft_model_runner.call("run", seqs, True)

            # 只有 target token 能进入 canonical Sequence
            self.scheduler.postprocess(seqs, token_ids, True)
        else:
            token_ids = self.model_runner.call("run", seqs, False)
            self.scheduler.postprocess(seqs, token_ids, False)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens