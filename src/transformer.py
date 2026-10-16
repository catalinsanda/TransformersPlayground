import torch
from torch import nn

from attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self._ln1 = nn.LayerNorm(d_model)
        self._mha = MultiHeadAttention(d_model, num_heads)

        self._ln2 = nn.LayerNorm(d_model)
        self._mlp = FeedForward(d_model, 4 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = x + self._mha(self._ln1(x))
        Y = H + self._mlp(self._ln2(H))

        return Y
