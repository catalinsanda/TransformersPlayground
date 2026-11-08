import torch
from torch import nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from kv_cache import LayerKVCache, ModelKVCache
from rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding):
        super().__init__()

        self._ln1 = nn.LayerNorm(d_model)
        self._mha = MultiHeadAttention(d_model, num_heads, rope)

        self._ln2 = nn.LayerNorm(d_model)
        self._mlp = FeedForward(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor, layer_cache: LayerKVCache | None = None) -> torch.Tensor:
        H = x + self._mha(self._ln1(x), layer_cache)
        Y = H + self._mlp(self._ln2(H))

        return Y


class TransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        rope: RotaryPositionalEmbedding,
    ):
        super().__init__()
        self._t_stack = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, rope) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, model_cache: ModelKVCache | None = None) -> torch.Tensor:
        for index, block in enumerate(self._t_stack):
            x = block(x, model_cache.layers[index] if model_cache is not None else None)

        return x
