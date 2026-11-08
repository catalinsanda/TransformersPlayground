import torch
from torch import nn

BASE = 10000


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_head: int):
        super().__init__()
        self._d_head = d_head
        assert d_head > 0 and d_head % 2 == 0

        self.register_buffer(
            "_rotations",
            torch.empty(0),
            persistent=False,
        )

    def build_rotation_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(self._d_head // 2, device=device, dtype=torch.float32)
        freq = BASE ** (-2 * i / self._d_head)

        p = torch.arange(seq_len, device=device, dtype=torch.float32)
        theta = p[:, None] * freq[None, :]

        return torch.stack(
            (
                torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1),
                torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1),
            ),
            dim=-2,
        )

    def forward(self, x: torch.Tensor, position_offset: int | None = None) -> torch.Tensor:
        assert x.dim() == 3
        assert x.size(-1) == self._d_head
        original_dtype = x.dtype

        x = x.to(torch.float32)
        a = torch.unflatten(x, -1, (-1, 2))
        a = a.unsqueeze(-1)

        seq_len = a.size(1) + (position_offset if position_offset else 0)
        if self._rotations.size(0) < seq_len or self._rotations.device != x.device:
            self._rotations = self.build_rotation_matrix(seq_len, x.device)
        rotations = self._rotations[position_offset:seq_len]

        a = torch.matmul(rotations, a)

        a = a.squeeze(-1)
        a = torch.flatten(a, -2)
        return a.to(original_dtype)
