import random

import pytest
import torch
from data_loader import DataLoader


@pytest.mark.parametrize("run", range(1000))
def test_data_loader(run: int) -> None:
    tokens_size = 1000
    input_tokens = [random.randint(0, 1000) for _ in range(tokens_size)]
    seq_len = random.randint(1, tokens_size - 2)
    batch_size = random.randint(1, 30)
    data_loader = DataLoader(input_tokens, batch_size, seq_len)
    x, y = data_loader.next_batch()

    assert x.dtype == torch.long
    assert y.dtype == torch.long

    assert x.dim() == 2
    assert x.shape == (batch_size, seq_len)
    assert x.shape == y.shape

    assert torch.equal(x[:, 1:], y[:, :-1])
