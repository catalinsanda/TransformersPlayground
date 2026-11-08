import torch
from torch import nn
import math

from kv_cache import HeadKVCache, LayerKVCache
from rope import RotaryPositionalEmbedding


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_head: int, rope: RotaryPositionalEmbedding):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head

        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self._rope = rope

    def forward(
        self, x: torch.Tensor, head_cache: HeadKVCache | None = None
    ) -> torch.Tensor:

        position_offset = head_cache.key.shape[1] if head_cache is not None else None

        q = self._rope(self.W_Q(x), position_offset)
        k = self._rope(self.W_K(x), position_offset)
        v = self.W_V(x)

        if head_cache is not None:
            k = torch.cat((head_cache.key, k), dim=1)
            v = torch.cat((head_cache.value, v), dim=1)

        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        t = x.shape[1]

        mask = ~torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        if head_cache is not None and position_offset is not None:
            mask = torch.cat(
                (
                    torch.zeros(t, position_offset, device=x.device, dtype=torch.bool),
                    mask,
                ),
                dim=1,
            )
        s = s.masked_fill(mask, -float("inf"))

        a = torch.softmax(s, -1)

        if head_cache:
            head_cache.key = k
            head_cache.value = v

        return torch.matmul(a, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding):
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads

        assert d_model % num_heads == 0

        self._d_head = d_model // num_heads

        assert self._d_head % 2 == 0

        self._module_list = nn.ModuleList(
            [
                AttentionHead(d_model, int(d_model // num_heads), rope)
                for _ in range(num_heads)
            ]
        )

        self._linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, layer_cache: LayerKVCache | None = None) -> torch.Tensor:
        outputs = [
            head(x, layer_cache.heads[index] if layer_cache else None)
            for index, head in enumerate(self._module_list)
        ]
        combined = torch.cat(outputs, dim=-1)
        return self._linear(combined)
