import pytest
import torch

from rope import RotaryPositionalEmbedding


@pytest.mark.parametrize(
    "batch_size,seq_len,d_head",
    [
        (1, 1, 2),
        (1, 10, 16),
        (2, 128, 64),
        (4, 32, 128),
    ],
)
def test_output_shape(batch_size: int, seq_len: int, d_head: int) -> None:
    rope = RotaryPositionalEmbedding(d_head)
    x = torch.randn(batch_size, seq_len, d_head)

    y = rope(x)

    assert y.shape == x.shape


@pytest.mark.parametrize("d_head", [2, 16, 64, 128])
def test_position_zero_is_identity(d_head: int):
    rope = RotaryPositionalEmbedding(d_head)
    x = torch.randn(2, 1, d_head)

    y = rope(x)

    torch.testing.assert_close(y, x)


def test_known_rotation():
    rope = RotaryPositionalEmbedding(d_head=2)

    # Identical vector at positions 0, 1, and 2.
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])

    y = rope(x)

    positions = torch.arange(3, dtype=torch.float32)
    expected = torch.stack(
        [torch.cos(positions), torch.sin(positions)],
        dim=-1,
    ).unsqueeze(0)

    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("d_head", [2, 16, 64])
def test_norm_preservation(d_head: int):
    rope = RotaryPositionalEmbedding(d_head)
    x = torch.randn(2, 32, d_head)

    y = rope(x)

    original_norms = torch.linalg.vector_norm(x, dim=-1)
    rotated_norms = torch.linalg.vector_norm(y, dim=-1)

    torch.testing.assert_close(rotated_norms, original_norms, rtol=1e-5, atol=1e-5)


def test_relative_position_invariance():
    rope = RotaryPositionalEmbedding(d_head=16)

    torch.manual_seed(42)
    q = torch.randn(16)
    k = torch.randn(16)

    # Same vectors at different absolute positions.
    q_seq = q.repeat(1, 12, 1)
    k_seq = k.repeat(1, 12, 1)

    q_rot = rope(q_seq)
    k_rot = rope(k_seq)

    # Both pairs are separated by 3 positions.
    score_1 = torch.dot(q_rot[0, 2], k_rot[0, 5])
    score_2 = torch.dot(q_rot[0, 7], k_rot[0, 10])

    torch.testing.assert_close(score_1, score_2)


def test_batch_independence():
    rope = RotaryPositionalEmbedding(d_head=32)

    x = torch.randn(1, 20, 32)
    batch = x.repeat(4, 1, 1)

    y = rope(batch)

    for b in range(1, 4):
        torch.testing.assert_close(y[0], y[b])


def test_gradient_propagation():
    rope = RotaryPositionalEmbedding(d_head=16)

    x = torch.randn(2, 10, 16, requires_grad=True)
    y = rope(x)

    loss = y.square().sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Rotation preserves squared norm, so the gradient
    # of sum(||R(x)||²) should equal 2*x.
    torch.testing.assert_close(x.grad, 2 * x, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_preservation(dtype: torch.dtype):
    rope = RotaryPositionalEmbedding(d_head=16)

    x = torch.randn(2, 10, 16, dtype=dtype)
    y = rope(x)

    assert y.dtype == dtype
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d_head", [0, 1, 3, 15, 63])
def test_invalid_head_dimension(d_head: int):
    with pytest.raises(AssertionError):
        RotaryPositionalEmbedding(d_head)


def test_different_positions_produce_different_rotations():
    rope = RotaryPositionalEmbedding(d_head=2)

    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])

    y = rope(x)

    assert not torch.allclose(y[0, 0], y[0, 1])
    assert not torch.allclose(y[0, 1], y[0, 2])


def test_position_offset_matches_full_sequence() -> None:
    rope = RotaryPositionalEmbedding(d_head=32)
    x = torch.randn(2, 10, 32)

    expected = rope(x, 0)

    first = rope(x[:, :3], 0)
    second = rope(x[:, 3:7], 3)
    third = rope(x[:, 7:], 7)

    actual = torch.cat((first, second, third), dim=1)

    torch.testing.assert_close(actual, expected)
