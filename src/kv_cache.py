import torch


class HeadKVCache:
    def __init__(
        self, batch_size: int, d_head: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.key = torch.empty(batch_size, 0, d_head, device=device, dtype=dtype)
        self.value = torch.empty(batch_size, 0, d_head, device=device, dtype=dtype)


class LayerKVCache:
    def __init__(
        self,
        num_heads: int,
        batch_size: int,
        d_head: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.heads = [
            HeadKVCache(batch_size, d_head, device, dtype) for _ in range(num_heads)
        ]


class ModelKVCache:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        batch_size: int,
        d_head: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.layers = [
            LayerKVCache(num_heads, batch_size, d_head, device, dtype)
            for _ in range(num_layers)
        ]
