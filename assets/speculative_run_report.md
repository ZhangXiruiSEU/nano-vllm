# Speculative Decoding Run Report

Data source: [`assets/speculative_run_metrics.json`](speculative_run_metrics.json)

## Run Setup

| Item | Value |
|---|---:|
| Target model | `/root/autodl-tmp/Qwen3-8B/` |
| Draft model | `/root/autodl-tmp/Qwen3-0.6B/` |
| Prompt | `introduce yourself` |
| Temperature | `0.6` |
| Max tokens | `256` |
| `num_spec_tokens` | `4` |
| Target GPU memory utilization | `0.68` |
| Draft GPU memory utilization | `0.9` |

## Top-Level Results

| Metric | Value |
|---|---:|
| Init time | `92.635 s` |
| Generate wall time | `4.379 s` |
| Output tokens | `256` |
| Wall output throughput | `58.464 tok/s` |
| Speculative decode throughput | `74.650 tok/s` |
| Acceptance rate | `51.19%` |
| Avg committed tokens / step | `3.036` |

`spec_decode_tok_s` is the most relevant number for comparing against target-only decode throughput. It uses committed decode tokens divided by accumulated speculative decode time, excluding initialization and prefill.

## Decode Stats

| Metric | Value |
|---|---:|
| Decode steps | `84` |
| Draft proposed tokens | `336` |
| Accepted draft tokens | `172` |
| Bonus tokens | `28` |
| Resampled tokens | `56` |
| Total committed tokens | `256` |
| Committed decode tokens | `255` |
| Full-accept steps | `27` |
| Reject/resample steps | `57` |

Acceptance is token-level, not character-level. For Chinese text, one visible character can map to one token, multiple tokens, or share a token with neighboring text.

## Stage Timing

| Stage | Total Time | Per Step | Decode Time Share |
|---|---:|---:|---:|
| Draft propose | `1.732268 s` | `20.622 ms` | `50.7%` |
| Target verify | `1.515419 s` | `18.041 ms` | `44.4%` |
| Accept/reject | `0.160776 s` | `1.914 ms` | `4.7%` |
| Probability transfer | `0.001350 s` | `0.016 ms` | `0.0%` |
| Scheduler | `0.000480 s` | `0.006 ms` | `0.0%` |
| Postprocess | `0.001404 s` | `0.017 ms` | `0.0%` |

The two dominant costs are draft propose and target verify. Scheduler, probability transfer, and postprocess are negligible in this run.

## Target Verify Breakdown

| Verify Substage | Total Time | Per Step | Decode Time Share |
|---|---:|---:|---:|
| Prepare tensors/context | `0.011682 s` | `0.139 ms` | `0.3%` |
| Model forward | `1.341515 s` | `15.970 ms` | `39.3%` |
| Logits projection | `0.111010 s` | `1.322 ms` | `3.2%` |
| Probability / softmax | `0.047950 s` | `0.571 ms` | `1.4%` |
| Verify graph hits | `84/84` | - | - |

The CUDA graph optimization is active for every verify step in this run. The main verify cost is still target model forward, not logits projection or softmax.

## Acceptance Distribution

| Accepted Draft Tokens In Step | Step Count |
|---:|---:|
| `0` | `23` |
| `1` | `11` |
| `2` | `16` |
| `3` | `7` |
| `4` | `27` |

Reject positions are zero-based within the proposed draft tokens.

| Reject Position | Count |
|---:|---:|
| `0` | `23` |
| `1` | `11` |
| `2` | `16` |
| `3` | `7` |


## Trace Preview

The trace records token ids and a decoded preview. Each row is one speculative step for batch item 0.

| Step | Accepted Draft Len | Outcome | Draft Tokens Decoded Individually | Committed Tokens Decoded Individually |
|---:|---:|---|---|---|
| 0 | `4` | full accept + bonus | `'\n'`<br>`'Okay'`<br>`','`<br>`' the'` | `'\n'`<br>`'Okay'`<br>`','`<br>`' the'`<br>`' user'` |
| 1 | `0` | reject at `0`, resample `' asked'` | `' wants'`<br>`' me'`<br>`' to'`<br>`' introduce'` | `' asked'` |
| 2 | `4` | full accept + bonus | `' me'`<br>`' to'`<br>`' introduce'`<br>`' myself'` | `' me'`<br>`' to'`<br>`' introduce'`<br>`' myself'`<br>`'.'` |
| 3 | `3` | reject at `3`, resample `' provide'` | `' I'`<br>`' need'`<br>`' to'`<br>`' be'` | `' I'`<br>`' need'`<br>`' to'`<br>`' provide'` |
| 4 | `1` | reject at `1`, resample `' clear'` | `' a'`<br>`' friendly'`<br>`' and'`<br>`' concise'` | `' a'`<br>`' clear'` |
| 5 | `2` | reject at `2`, resample `' introduction'` | `' and'`<br>`' friendly'`<br>`' response'`<br>`'.'` | `' and'`<br>`' friendly'`<br>`' introduction'` |
| 6 | `1` | reject at `1`, resample `' Let'` | `'.'`<br>`' I'`<br>`' should'`<br>`' mention'` | `'.'`<br>`' Let'` |
| 7 | `4` | full accept + bonus | `' me'`<br>`' start'`<br>`' with'`<br>`' my'` | `' me'`<br>`' start'`<br>`' with'`<br>`' my'`<br>`' name'` |
## Interpretation

This run demonstrates a working speculative decoding pipeline with target verify CUDA graph enabled. The measured speculative decode throughput is `74.65 tok/s`, which is above the previously measured target-only decode baseline of roughly `59 tok/s`.

The speedup is modest because token acceptance is only `51.19%`. With `K=4`, the draft proposes four tokens per speculative step, but the system commits only `3.04` tokens per step on average. The rest of the potential speedup is spent on draft propose, target verify, and accept/reject overhead.

The verify graph optimization is important: verify graph hits are `84/84`. Without this graph, target verify model forward was previously the dominant bottleneck. In this run, target verify remains significant, but draft propose is also a major cost.

## Notes For Discussion

- Speculative decoding operates on tokenizer tokens, not characters or words.
- Rejection and resampling happen at token granularity.
- `accepted_draft_len` tells how many draft tokens were accepted before the first rejection.
- If `rejected_index` is `null`, all draft tokens were accepted and `bonus_token_id` was committed.
- If `rejected_index` is not `null`, `resampled_token_id` is committed at the rejection position.
- The KV cache invariant checked during the run is: `seq.num_cached_tokens == len(seq) - 1` after speculative postprocess for unfinished sequences.
