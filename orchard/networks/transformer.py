# transformer.py
# This is a standard Transformer network. This network could be used to load and use
# llama2 checkpoint from Meta.
# This code is derived from gpt-fast code snippets.

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..cache import KVCache
from ..model_args import ModelArgs

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        local_intermediate_size = config.intermediate_size // config.world_size
        self.w1 = nn.Linear(config.dim, local_intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, local_intermediate_size, bias=False)
        self.w2 = nn.Linear(local_intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, index: int, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        self.index = index
        self.config = config
        self.n_head = config.n_head // config.world_size
        self.dim = config.dim // config.world_size
        self.head_dim = self.dim // self.n_head
        self.n_local_heads = config.n_local_heads // config.world_size

        total_head_dim = (self.n_head + 2 * self.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(self.dim, config.dim, bias=False)
        self.kv_cache = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    @staticmethod
    def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        return x_out2.type_as(x)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = Attention.apply_rotary_emb(q, freqs_cis)
        k = Attention.apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, index: int, config: ModelArgs) -> None:
        super().__init__()
        self.index = index
        self.attention = Attention(index, config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        # local_dim = config.dim // config.world_size
        local_dim = config.dim

        self.tok_embeddings = nn.Embedding(config.vocab_size, local_dim)
        self.layers = nn.ModuleList(TransformerBlock(i, config) for i in range(config.n_local_layers))
        self.norm = RMSNorm(local_dim, eps=config.norm_eps)
        self.output = nn.Linear(local_dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))

    @staticmethod
    def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
        return cache.to(dtype=torch.bfloat16)

    @staticmethod
    def find_multiple(n: int, k: int) -> int:
        return n if n % k == 0 else n + k - (n % k)

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return

        local_dim = self.config.dim //self.config.world_size
        local_n_head = self.config.n_head // self.config.world_size
        n_local_heads = self.config.n_local_heads // self.config.world_size

        head_dim = local_dim // local_n_head
        max_seq_length = Transformer.find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, n_local_heads, head_dim)

        self.freqs_cis = Transformer.precompute_freqs_cis(self.config.block_size, head_dim, self.config.rope_base)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for _, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)

        x = self.norm(x)
        logits = self.output(x)
        return logits

    @staticmethod
    def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    @staticmethod
    def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @staticmethod
    def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
        probs = Transformer.logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = Transformer.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def prefill(self, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
        # input_pos: [B, S]
        logits = self(x, input_pos)
        return Transformer.sample(logits, **sampling_kwargs)[0]

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        **sampling_kwargs,
    ) -> torch.Tensor:
        T = prompt.size(0)
        T_new = T + max_new_tokens
        max_seq_length = min(T_new, self.config.block_size)

        device, dtype = prompt.device, prompt.dtype
        with torch.device(device):
            self.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        input_pos = torch.arange(0, T, device=device)
        next_token = self.prefill(prompt.view(1, -1), input_pos, **sampling_kwargs)

        # create an empty tensor of the expected final shape and fill in the current tokens
        seq = torch.empty(T_new, dtype=dtype, device=device)
        seq[:T] = prompt
        seq[T] = next_token

        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        new_tokens = []
        cur_token = next_token.view(1, -1)
        for i in range(max_new_tokens - 1):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                logits = self(cur_token, input_pos)
                next_token, _ = self.sample(logits, **sampling_kwargs)
                input_pos += 1
                new_tokens.append(next_token.clone())
                cur_token = next_token.view(1, -1)

        seq[T + 1:] = torch.cat(new_tokens)
        return seq
