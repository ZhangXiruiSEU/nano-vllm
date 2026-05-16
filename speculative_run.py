import os
from nanovllm import LLM, SamplingParams ,SpLLM
from transformers import AutoTokenizer


def main():
    path_target = os.path.expanduser("/root/autodl-tmp/Qwen3-8B/")
    path_draft = os.path.expanduser("/root/autodl-tmp/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path_target)
    tokenizer_draft = AutoTokenizer.from_pretrained(path_draft)
    # llm_target = LLM(path_target, enforce_eager=False, tensor_parallel_size=1, is_target = True)
    spllm = SpLLM(path_target, enforce_eager=False, tensor_parallel_size=1, is_target = True, draft_model = path_draft)
    pass
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    #logits = llm_target.model_runner.run_verify()
    outputs = spllm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()