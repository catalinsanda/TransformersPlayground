import pytest
import torch
import torch.nn.functional as F

from feed_forward import FeedForward


D_MODEL = 128
D_FF = 512


@pytest.fixture
def feed_forward() -> FeedForward:
    return FeedForward(d_model=D_MODEL, d_ff=D_FF)


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
    feed_forward: FeedForward,
    batch_size: int,
    seq_len: int,
) -> None:
    x = torch.randn(batch_size, seq_len, D_MODEL)

    output = feed_forward(x)

    assert output.shape == (batch_size, seq_len, D_MODEL)


def test_matches_manual_computation(feed_forward: FeedForward) -> None:
    x = torch.randn(2, 5, D_MODEL)

    actual = feed_forward(x)

    hidden = F.linear(
        x,
        feed_forward._lin1.weight,
        feed_forward._lin1.bias,
    )
    activated = F.gelu(hidden)
    expected = F.linear(
        activated,
        feed_forward._lin2.weight,
        feed_forward._lin2.bias,
    )

    torch.testing.assert_close(actual, expected)


def test_gradients_flow(feed_forward: FeedForward) -> None:
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)

    output = feed_forward(x)
    loss = output.square().mean()
    loss.backward()

    for layer in (feed_forward._lin1, feed_forward._lin2):
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

        assert torch.isfinite(layer.weight.grad).all()
        assert torch.isfinite(layer.bias.grad).all()

        assert layer.weight.grad.abs().sum() > 0

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_token_independence(feed_forward: FeedForward) -> None:
    x = torch.randn(2, 5, D_MODEL)

    original = feed_forward(x)

    modified_x = x.clone()
    modified_x[:, 2, :] += 10.0

    modified = feed_forward(modified_x)

    torch.testing.assert_close(
        original[:, :2, :],
        modified[:, :2, :],
    )
    torch.testing.assert_close(
        original[:, 3:, :],
        modified[:, 3:, :],
    )


def test_batch_independence(feed_forward: FeedForward) -> None:
    x = torch.randn(3, 5, D_MODEL)

    batched = feed_forward(x)

    individual = torch.cat(
        [feed_forward(x[i : i + 1]) for i in range(3)],
        dim=0,
    )

    torch.testing.assert_close(batched, individual)


def test_tokens_processed_independently(feed_forward: FeedForward) -> None:
    x = torch.randn(2, 5, D_MODEL)

    actual = feed_forward(x)

    individual = torch.cat(
        [feed_forward(x[:, i : i + 1, :]) for i in range(x.shape[1])],
        dim=1,
    )

    torch.testing.assert_close(actual, individual)
