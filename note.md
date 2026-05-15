# nano-vLLM Speculative Decoding 改造记录

## 目标

在 nano-vLLM 里实现 speculative decoding：

- 一个 engine 管理两个模型：draft model 和 target model。
- 只保留一个 scheduler 和一份 canonical `Sequence`。
- draft 只提出候选 token。
- target verify 后，只有 accepted token 才正式进入 `Sequence`。
- draft / target 各自维护自己的 KV cache，不共享 KV tensor。

## 当前架构理解

原始结构：

```text
LLMEngine
  ├── Scheduler
  │   ├── waiting sequences
  │   └── running sequences
  └── ModelRunner
      ├── model
      ├── sampler
      └── KV cache
```

关键职责：

- Scheduler：管理 waiting/running、block 分配、preempt、postprocess。
- Sequence：保存 canonical token 序列、长度、block table、cached token 数等。
- ModelRunner：准备 input tensor、positions、slot mapping、attention metadata，执行模型 forward。
- BlockManager：管理 physical KV block。
- Sampler：从 logits 中采样 token。

## 关键结论

### 1. 一个 engine 可以管理两个模型

不需要两个完整 LLMEngine。应该是：

SpLLMEngine
  ├── Scheduler
  ├── canonical Sequence
  ├── target_model_runner
  └── draft_model_runner

两个 runner 各自有独立 KV cache：

target_runner.kv_cache != draft_runner.kv_cache

但可以共享同一套 Sequence.block_table，前提是 scheduler 的 block 数取两边 KV cache block 数的最小值。

### 2. KV cache 不共享，但 block id 语义共享

seq.block_table 只是 logical block 到 physical block id 的映射。

同一个 block id：

target_runner.kv_cache[:, :, block_id, ...]
draft_runner.kv_cache[:, :, block_id, ...]

对应的是两个不同 tensor，只是逻辑位置一致。

### 3. Scheduler 不需要知道 draft/verify 细节

Scheduler 不应该变成：

prefill / draft_decode / target_verify / truncate

更合理的是：

prefill
extend/decode

speculative 内部流程由 engine 控制：

schedule
-> draft propose
-> target verify
-> accept/reject
-> commit accepted tokens

### 4. 原始 postprocess() 不能用于 speculative decode

原始 postprocess() 默认：

每条 seq 本轮只 append 1 token
num_cached_tokens 前进 num_scheduled_tokens
本轮 scheduled token 全部有效

speculative 不满足这些假设，因为 verify 写了 K+1 个 KV，但最终只接受 N 个 token。

因此需要 postprocess_speculative()：

- 只提交 accepted tokens。
- 按 EOS / max_tokens 截断。
- 只让 num_cached_tokens 前进 accepted token 数。
- rejected suffix 不进入 canonical Sequence。

## 已补的核心改动

### Sampler

补了 forward_with_probs()：

def forward_with_probs(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    sample_tokens = (probs / noise).argmax(dim=-1)
    selected_probs = probs.gather(1, sample_tokens.unsqueeze(1)).squeeze(1)
    return sample_tokens, selected_probs, probs

用于返回：

sampled token
sampled token probability
full vocab probability distribution

还补了 sample_from_probs()，用于 residual distribution 采样。

### BlockManager

补了：

can_append_n(seq, num_tokens)
may_append_n(seq, num_tokens)

用于 speculative verify 一次性预留 K+1 个 KV slot。

### Scheduler

Scheduler.schedule() 改成：

schedule(num_lookahead_tokens: int = 1)

普通 decode 默认还是 1 token。

Speculative decode 传：

self.num_spec_tokens + 1

因为 target verify 输入：

[last_token] + K draft tokens

一共写 K+1 个 KV slot。

### ModelRunner

补了通用 extend 路径：

prepare_extend(seqs, tokens_per_seq, start_offsets=None)

用于：

- target verify
- draft KV sync
- 小 prefill / multi-token append

prepare_verify() 变成 prepare_extend() 的薄封装：

tokens_per_seq = [[seq.last_token] + draft_ids]
start_offsets = [-1]

补了 verify()，返回：

target_token_probs: [B, K]
target_full_probs:  [B, K, V]
bonus_full_probs:   [B, V]

其中：

target_token_probs 用于 accept/reject 判断
target_full_probs 用于 rejected 后 residual sampling
bonus_full_probs 用于全接受后采 bonus token

### ParallelLMHead / Qwen3

补了返回所有位置 logits 的接口：

ParallelLMHead.forward_all()
Qwen3ForCausalLM.compute_logits_all()

因为 verify 需要 K+1 个位置的 logits，而不是只取最后一个。

### SpLLMEngine

当前 speculative step 结构：

prefill:
  target prefill
  draft prefill
  scheduler.postprocess 只提交 target token

decode:
  scheduler.schedule(K+1)
  draft propose K
  target verify K
  accept/reject
  scheduler.postprocess_speculative

_draft_propose() 支持 batch，返回：

draft_token_ids:    list[list[int]]
draft_token_probs:  [B, K]
draft_full_probs:   [B, K, V]
draft_lens:         list[int]

_accept_reject() 使用标准 speculative sampling 逻辑：

accept_prob = min(1, p_target(draft_i) / p_draft(draft_i))

如果拒绝：

residual = normalize(max(target_full_probs - draft_full_probs, 0))
sample from residual

如果全接受：

sample from bonus_full_probs

## 当前重要判断

### verify 为什么要返回完整分布

只返回 draft token 的概率：

p_target(draft_i)
p_draft(draft_i)

只能判断是否接受。

拒绝后需要从 residual distribution 采样：

residual(x) = max(p_target(x) - p_draft(x), 0)

所以必须有完整 vocab 分布：

target_full_probs[i]: [V]
draft_full_probs[i]:  [V]

### CPU/GPU 分工

当前项目本来就是：

CPU:
  Sequence
  Scheduler
  BlockManager metadata
  token_ids
  block_table

GPU:
  model weights
  KV cache
  logits
  probs
  attention

full probs 应尽量留在 GPU，不要 .tolist()。

只把很小的控制结果拿回 CPU，例如 accept mask 和最终 token id。

### draft KV sync 问题

当前最大未完成点是 draft KV 同步。

一轮 draft propose K 个时，draft 实际写过：

old last_token
draft_1
...
draft_{K-1}

它采出了 draft_K，但还没写 draft_K 的 KV。

如果全接受并追加 bonus：

commit: draft_1 ... draft_K bonus

则 draft KV 缺：

draft_K

bonus 的 KV 可以留给下一轮 draft decode 写，因为下一轮会输入 current last_token。

因此 commit 后需要一个“小 prefill / extend”把 draft KV 补到 canonical 的倒数第二个 token。


## 待办

### 1. 补 draft KV sync

在 ModelRunner 里补：

@torch.inference_mode()
def extend_kv(self, seqs, tokens_per_seq, start_offsets=None):
    input_ids, positions, _ = self.prepare_extend(
        seqs,
        tokens_per_seq,
        start_offsets,
    )
    self.model(input_ids, positions)
    reset_context()

在 SpLLMEngine 里补：

def _sync_draft_after_commit(self, seqs, base_lens, draft_lens):
    ...

逻辑：

draft_written_until = base_len + draft_len - 1
sync_until = len(seq) - 1

sync tokens:
  seq.token_ids[draft_written_until : sync_until]

调用位置：

base_lens = [len(seq) for seq in seqs]
draft propose
verify
accept/reject
committed = postprocess_speculative(...)
sync draft after commit

### 2. 让 postprocess_speculative() 返回实际 committed tokens

因为它会按 EOS / max_tokens 截断，所以需要返回最终提交的 token 列表：

return committed_token_ids

否则 draft sync 不知道最终 canonical 实际提交了多少。

### 3. 给 verify() 加 @torch.inference_mode()

避免 autograd graph。

### 4. 清理 TP>1 假设

当前先默认 TP=1。

TP>1 目前还有问题：

- ModelRunner.__init__ 签名改了，但 spawn 参数未同步。
- draft/target 多 runner + TP 的 rank/device 逻辑还没设计完整。

### 5. 检查单 GPU / 多 GPU 设备配置

当前 draft runner 用：

device=config.tensor_parallel_size

TP=1 时 draft 在 GPU1。如果机器只有一张 GPU 会失败。

需要明确：

- 是否要求两张 GPU。
- 或者允许 draft/target 同卡但显存要够。

### 6. 小规模运行测试

建议测试顺序：

1. py_compile
2. 单 prompt, max_tokens=4, num_spec_tokens=1
3. 单 prompt, max_tokens=8, num_spec_tokens=2
4. batch=2
5. num_spec_tokens=4

重点观察：

seq.num_tokens
seq.num_cached_tokens
seq.block_table
accepted_token_ids
draft_lens
finished/deallocate

## 当前状态总结

当前已经完成：

speculative decoding 的主骨架
full distribution verify
标准 accept/reject residual sampling
scheduler lookahead slot 预留
speculative postprocess

尚未完成：

draft KV sync
实际运行验证
TP>1
性能优化

当前可以进入小规模调试，但还不能认为是完整正确的最终版。