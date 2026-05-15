import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch

@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


@dataclass(slots=True, kw_only=True)
class SpConfig(Config):
    draft_model:str = ""
    # draft_tensor_parallel_size: int = 1
    draft_hf_config: AutoConfig | None = None
    num_spec_tokens:int = 4
    def __post_init__(self):
        super(SpConfig,self).__post_init__()
        # 如果父类有 post_init 记得 super().__post_init__()
        assert self.draft_model != "", "没给出 draft 模型的路径"
        self.hf_config = AutoConfig.from_pretrained(self.draft_model)
