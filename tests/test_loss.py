
import math

import pytest
import torch

from loss import compute_loss


def test_loss_is_scalar() -> None:
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))

    loss = compute_loss(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_manual_cross_entropy() -> None:
    # One batch, two positions, three vocabulary tokens.
    logits = torch.tensor([
        [
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0],
        ]
    ])

    targets = torch.tensor([[0, 1]])

    loss = compute_loss(logits, targets)

    # Compute cross-entropy manually using log-softmax.
    log_probs = torch.log_softmax(logits, dim=-1)

    expected = -(
        log_probs[0, 0, 0] +
        log_probs[0, 1, 1]
    ) / 2

    torch.testing.assert_close(loss, expected)


def test_gradients_propagate() -> None:
    logits = torch.randn(
        2, 4, 10, requires_grad=True
    )
    targets = torch.randint(0, 10, (2, 4))

    loss = compute_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_correct_prediction_has_lower_loss() -> None:
    targets = torch.tensor([[1]])

    # Correct token (index 1) receives highest logit.
    correct_logits = torch.tensor([[
        [0.0, 10.0, 0.0]
    ]])

    # Incorrect token (index 0) receives highest logit.
    incorrect_logits = torch.tensor([[
        [10.0, 0.0, 0.0]
    ]])

    correct_loss = compute_loss(correct_logits, targets)
    incorrect_loss = compute_loss(incorrect_logits, targets)

    assert correct_loss < incorrect_loss
    assert correct_loss.item() < 0.001
    assert incorrect_loss.item() > 9.0


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size",
    [
        (1, 1, 5),
        (1, 8, 10),
        (4, 16, 65),
        (8, 32, 100),
    ],
)
def test_batch_and_sequence_dimensions(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> None:
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(
        0, vocab_size, (batch_size, seq_len)
    )

    actual = compute_loss(logits, targets)

    # Calculate loss independently for every token position.
    individual_losses = []

    for b in range(batch_size):
        for t in range(seq_len):
            individual_losses.append(
                torch.nn.functional.cross_entropy(
                    logits[b, t].unsqueeze(0),
                    targets[b, t].unsqueeze(0),
                )
            )

    expected = torch.stack(individual_losses).mean()

    torch.testing.assert_close(actual, expected)


def test_uniform_predictions() -> None:
    vocab_size = 100

    # Equal logits produce uniform probabilities.
    logits = torch.zeros(2, 5, vocab_size)
    targets = torch.randint(0, vocab_size, (2, 5))

    loss = compute_loss(logits, targets)

    expected = torch.tensor(math.log(vocab_size))

    torch.testing.assert_close(loss, expected)


def test_noncontiguous_logits() -> None:
    # Transposing creates a non-contiguous tensor.
    logits = torch.randn(4, 2, 10).transpose(0, 1)
    targets = torch.randint(0, 10, (2, 4))

    assert not logits.is_contiguous()

    loss = compute_loss(logits, targets)

    expected = torch.nn.functional.cross_entropy(
        logits.contiguous().view(-1, 10),
        targets.reshape(-1),
    )

    torch.testing.assert_close(loss, expected)
