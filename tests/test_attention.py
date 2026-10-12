
import math

import pytest
import torch
import torch.nn.functional as F

from attention import AttentionHead


D_MODEL = 128
D_HEAD = 32


@pytest.fixture
def attention_head() -> AttentionHead:
    return AttentionHead(d_model=D_MODEL, d_head=D_HEAD)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 1),
        (1, 10),
        (4, 10),
        (8, 100),
    ],
)
def test_output_shape(
    attention_head: AttentionHead,
    batch_size: int,
    seq_len: int,
) -> None:
    x = torch.randn(batch_size, seq_len, D_MODEL)

    output = attention_head(x)

    assert output.shape == (batch_size, seq_len, D_HEAD)


def test_single_token_attention(attention_head: AttentionHead) -> None:
    x = torch.randn(2, 1, D_MODEL)

    output = attention_head(x)
    expected = attention_head.W_V(x)

    torch.testing.assert_close(output, expected)


def test_attention_weights_sum_to_one(attention_head: AttentionHead) -> None:
    x = torch.randn(2, 5, D_MODEL)

    Q = attention_head.W_Q(x)
    K = attention_head.W_K(x)

    scores = Q @ K.transpose(-2, -1) / math.sqrt(D_HEAD)
    weights = torch.softmax(scores, dim=-1)

    sums = weights.sum(dim=-1)

    torch.testing.assert_close(sums, torch.ones_like(sums))


def test_matches_pytorch_reference(attention_head: AttentionHead) -> None:
    x = torch.randn(2, 5, D_MODEL)

    actual = attention_head(x)

    Q = attention_head.W_Q(x)
    K = attention_head.W_K(x)
    V = attention_head.W_V(x)

    expected = F.scaled_dot_product_attention(
        Q, K, V, dropout_p=0.0, is_causal=False
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_gradients_flow(attention_head: AttentionHead) -> None:
    x = torch.randn(2, 5, D_MODEL)

    output = attention_head(x)
    loss = output.square().mean()
    loss.backward()

    for projection in (
        attention_head.W_Q,
        attention_head.W_K,
        attention_head.W_V,
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0


def test_batch_independence(attention_head: AttentionHead) -> None:
    x = torch.randn(3, 5, D_MODEL)

    batched = attention_head(x)
    individual = torch.cat(
        [attention_head(x[i : i + 1]) for i in range(3)],
        dim=0,
    )

    torch.testing.assert_close(batched, individual)
