import torch
from torch import nn


class Sampler(nn.Module):


    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # 先把 logits 转成 float32，再按每条序列自己的 temperature 做缩放。
        probs = torch.softmax(logits, dim=-1)  # 对词表维做 softmax，把缩放后的 logits 变成概率分布。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)  

        # 用 Gumbel-max 风格的扰动采样技巧，从每条序列的概率分布中随机采一个 token。
        #  argmax(log p_i + g_i)
        #  让g_i 服从 -ln(-ln(U_i))  let -ln(U_i) = e_i 服从 指数分布 参数为 1
        # argmax(ln p_i -  ln e_i) 所以就是二者相除即可，ln 单增也可忽略
        # torch.empty_like(probs) .exponential_(1) 造一个和 probs 同形状的随机噪声张量，里面每个元素都是指数分布样本。
        #  .clamp_min_(1e-10) 把张量里所有小于 1e-10 的值，原地抬到 1e-10。
        
        return sample_tokens  # 返回当前批次每条序列采样得到的 token id。
    

    @torch.compile
    def forward_with_probs(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 温度处理
        probs = torch.softmax(logits, dim=-1) # 变为概率
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10) # 加噪声
        sample_tokens = (probs / noise).argmax(dim=-1) # 取最大值为采样结果
        selected_probs = probs.gather(1, sample_tokens.unsqueeze(1)).squeeze(1)
        return sample_tokens, selected_probs, probs # 

    @torch.compile
    def sample_from_probs(self, probs: torch.Tensor):
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return (probs / noise).argmax(dim=-1)