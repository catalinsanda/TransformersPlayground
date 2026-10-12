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

        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        A = torch.softmax(S, -1)

        return torch.matmul(A, V)
