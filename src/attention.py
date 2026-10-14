import torch
from torch import nn
import math


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head

        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        s = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        T = x.shape[1]
        mask = ~torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        s = s.masked_fill(mask, -float("inf"))

        A = torch.softmax(s, -1)

        return torch.matmul(A, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self._d_model = d_model
        self._num_heads = num_heads

        assert d_model % num_heads == 0
        
        self._module_list = nn.ModuleList(
            [AttentionHead(d_model, int(d_model // num_heads)) for _ in range(num_heads)]
        )
        
        self._linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [head(x) for head in self._module_list]
        combined = torch.cat(outputs, dim=-1)
        return self._linear(combined)
