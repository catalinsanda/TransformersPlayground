from typing import List

import torch
from torch import nn
from statistics import mean

from data_loader import DataLoader
from decoder_transformer import DecoderOnlyTransformer


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    logits_reshaped = logits.reshape(-1, logits.shape[-1])
    targets_reshaped = targets.reshape(-1)

    return nn.functional.cross_entropy(logits_reshaped, targets_reshaped)


def estimate_loss(
    model: DecoderOnlyTransformer,
    data_loader: DataLoader,
    device: torch.device,
    num_batches: int = 20,
) -> float:

    if num_batches < 1:
        raise ValueError("num_batches must be positive")

    was_training = model.training

    try:
        model.eval()

        losses: List[float] = []

        with torch.inference_mode():
            for _ in range(num_batches):
                x, y = data_loader.next_batch()
                x, y = x.to(device), y.to(device)

                logits = model(x)
                losses.append(compute_loss(logits, y).item())

        return mean(losses)

    finally:
        model.train(was_training)
