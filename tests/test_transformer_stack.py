
import pytest
import torch
from torch import nn

from transformer import TransformerStack


D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 3
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def stack() -> TransformerStack:
    torch.manual_seed(42)
    return TransformerStack(D_MODEL, NUM_HEADS, NUM_LAYERS)


def test_output_shape(stack: TransformerStack) -> None:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    y = stack(x)

    assert y.shape == x.shape


@pytest.mark.parametrize("num_layers", [1, 2, 4, 8])
def test_number_of_layers(num_layers: int) -> None:
    stack = TransformerStack(D_MODEL, NUM_HEADS, num_layers)

    assert isinstance(stack._t_stack, nn.ModuleList)
    assert len(stack._t_stack) == num_layers


def test_independent_parameters(stack: TransformerStack) -> None:
    blocks = stack._t_stack

    # Each block must be a separate module instance.
    assert len({id(block) for block in blocks}) == NUM_LAYERS

    # Corresponding parameters must not share storage.
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            params_i = dict(blocks[i].named_parameters())
            params_j = dict(blocks[j].named_parameters())

            assert params_i.keys() == params_j.keys()

            for name in params_i:
                assert (
                    params_i[name].data_ptr()
                    != params_j[name].data_ptr()
                ), f"Parameter {name} is shared by blocks {i} and {j}"


def test_gradients_reach_all_layers(
    stack: TransformerStack,
) -> None:
    x = torch.randn(
        BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True
    )

    y = stack(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for i, block in enumerate(stack._t_stack):
        for name, param in block.named_parameters():
            assert param.grad is not None, (
                f"No gradient for layer {i}, parameter {name}"
            )
            assert torch.isfinite(param.grad).all(), (
                f"Invalid gradient for layer {i}, parameter {name}"
            )

        assert any(
            param.grad is not None
            and torch.count_nonzero(param.grad).item() > 0
            for param in block.parameters()
        ), f"No nonzero gradients in layer {i}"


@pytest.mark.parametrize("position", [0, 1, 3, 6])
def test_causal_behavior(
    stack: TransformerStack, position: int
) -> None:
    x = torch.randn(1, SEQ_LEN, D_MODEL)
    x_modified = x.clone()

    # Change every token strictly after the selected position.
    x_modified[:, position + 1 :, :] = torch.randn_like(
        x_modified[:, position + 1 :, :]
    ) * 10

    with torch.no_grad():
        y_original = stack(x)
        y_modified = stack(x_modified)

    # Outputs at and before this position must stay unchanged.
    torch.testing.assert_close(
        y_original[:, : position + 1, :],
        y_modified[:, : position + 1, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_batch_independence(stack: TransformerStack) -> None:
    x1 = torch.randn(1, SEQ_LEN, D_MODEL)
    x2 = torch.randn(1, SEQ_LEN, D_MODEL)

    batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        y_batch = stack(batch)
        y1 = stack(x1)
        y2 = stack(x2)

    torch.testing.assert_close(y_batch[0:1], y1)
    torch.testing.assert_close(y_batch[1:2], y2)


def test_identity_when_sublayers_are_zero(
    stack: TransformerStack,
) -> None:
    # Zero attention and MLP parameters in every block.
    # Leave LayerNorm parameters untouched.
    with torch.no_grad():
        for block in stack._t_stack:
            for module in [block._mha, block._mlp]:
                for param in module.parameters():
                    param.zero_()

    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    y = stack(x)

    # With no final LayerNorm, the entire stack is identity.
    torch.testing.assert_close(y, x, rtol=0, atol=0)


def test_sequential_execution(stack: TransformerStack) -> None:
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    with torch.no_grad():
        # Explicitly apply each block in order.
        expected = x
        for block in stack._t_stack:
            expected = block(expected)

        actual = stack(x)

    torch.testing.assert_close(actual, expected)


def test_parameters_are_registered(
    stack: TransformerStack,
) -> None:
    registered_ids = {id(p) for p in stack.parameters()}

    for block in stack._t_stack:
        for param in block.parameters():
            assert id(param) in registered_ids
