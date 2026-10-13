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

    expected = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=True)

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


def test_causal_attention(attention_head: AttentionHead) -> None:
    x = torch.randn(2, 5, D_MODEL)

    original = attention_head(x)

    modified_x = x.clone()
    modified_x[:, -1, :] += 10.0

    modified = attention_head(modified_x)

    torch.testing.assert_close(
        original[:, :-1, :],
        modified[:, :-1, :],
    )


@pytest.mark.parametrize("position", [0, 1, 2, 3, 4])
def test_causal_attention_all_positions(
    attention_head: AttentionHead,
    position: int,
) -> None:
    x = torch.randn(2, 5, D_MODEL)

    original = attention_head(x)

    modified_x = x.clone()
    modified_x[:, position, :] += 10.0

    modified = attention_head(modified_x)

    torch.testing.assert_close(
        original[:, :position, :],
        modified[:, :position, :],
    )


def test_first_token_only_attends_to_itself(
    attention_head: AttentionHead,
) -> None:
    x = torch.randn(2, 5, D_MODEL)

    output = attention_head(x)
    first_value = attention_head.W_V(x)[:, 0, :]

    torch.testing.assert_close(
        output[:, 0, :],
        first_value,
    )


def test_mask_preserves_zero_scores() -> None:
    head = AttentionHead(d_model=4, d_head=2)

    with torch.no_grad():
        head.W_Q.weight.zero_()
        head.W_K.weight.zero_()

    x = torch.randn(1, 3, 4)

    output = head(x)
    values = head.W_V(x)

    # All scores are zero, so allowed positions
    # receive equal attention weights.
    expected = torch.stack(
        [
            values[:, :1, :].mean(dim=1),
            values[:, :2, :].mean(dim=1),
            values[:, :3, :].mean(dim=1),
        ],
        dim=1,
    )

    torch.testing.assert_close(output, expected)
