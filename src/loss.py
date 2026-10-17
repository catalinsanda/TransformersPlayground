import torch
from torch import nn


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    logits_reshaped = torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
    targets_reshaped = torch.reshape(targets, (targets.shape[0] * targets.shape[1],))

    return nn.functional.cross_entropy(logits_reshaped, targets_reshaped)
