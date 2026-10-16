import pytest
import torch

from transformer import TransformerBlock


D_MODEL = 32
NUM_HEADS = 4
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def block() -> TransformerBlock:
    torch.manual_seed(42)
    return TransformerBlock(D_MODEL, NUM_HEADS)


def test_output_shape(block: TransformerBlock) -> None:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    y = block(x)

    assert y.shape == x.shape


def test_gradients_reach_both_sublayers(
    block: TransformerBlock,
) -> None:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)

    y = block(x)
    loss = y.square().mean()
    loss.backward()

    # Input receives gradients
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Every trainable parameter receives a gradient
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"

    # Both sublayers have nonzero gradients
    for name, module in [
        ("attention", block._mha),
        ("feed_forward", block._mlp),
    ]:
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad).item() > 0
            for p in module.parameters()
        ), f"No nonzero gradients in {name}"


@pytest.mark.parametrize("position", [0, 1, 3, 6])
def test_causal_behavior(block: TransformerBlock, position: int) -> None:
    x = torch.randn(1, SEQ_LEN, D_MODEL)
    x_modified = x.clone()

    # Modify only tokens after the selected position
    x_modified[:, position + 1 :, :] = (
        torch.randn_like(x_modified[:, position + 1 :, :]) * 10
    )

    with torch.no_grad():
        y_original = block(x)
        y_modified = block(x_modified)

    # Earlier tokens and the selected token must be unaffected
    torch.testing.assert_close(
        y_original[:, : position + 1, :],
        y_modified[:, : position + 1, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_identity_when_sublayers_are_zero(
    block: TransformerBlock,
) -> None:
    # Zero only attention and MLP parameters.
    # Leave LayerNorm parameters unchanged.
    with torch.no_grad():
        for module in [block._mha, block._mlp]:
            for param in module.parameters():
                param.zero_()

    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    y = block(x)

    # Residual connections should preserve the input exactly
    torch.testing.assert_close(y, x, rtol=0, atol=0)


def test_batch_independence(block: TransformerBlock) -> None:
    x1 = torch.randn(1, SEQ_LEN, D_MODEL)
    x2 = torch.randn(1, SEQ_LEN, D_MODEL)
    batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        y_batch = block(batch)
        y1 = block(x1)
        y2 = block(x2)

    torch.testing.assert_close(y_batch[0:1], y1)
    torch.testing.assert_close(y_batch[1:2], y2)


def test_pre_ln_computation(block: TransformerBlock) -> None:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    with torch.no_grad():
        # Explicitly compute the expected Pre-LN architecture
        h = x + block._mha(block._ln1(x))
        expected = h + block._mlp(block._ln2(h))

        actual = block(x)

    torch.testing.assert_close(actual, expected)
