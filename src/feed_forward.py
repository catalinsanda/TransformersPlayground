import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        
        self._lin1 = nn.Linear(d_model, d_ff)
        self._lin2 = nn.Linear(d_ff, d_model)
        self._gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._lin2(self._gelu(self._lin1(x)))