
import pytest
import torch

from decoder_transformer import DecoderOnlyTransformer


VOCAB_SIZE = 65
D_MODEL = 32
NUM_HEADS = 4
NUM_LAYERS = 3
MAX_SEQ_LEN = 64
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def model() -> DecoderOnlyTransformer:
    torch.manual_seed(42)
    return DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    )


def test_output_shape(model: DecoderOnlyTransformer) -> None:
    x = torch.randint(
        0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)
    )

    logits = model(x)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("seq_len", [1, 2, 8, 32, MAX_SEQ_LEN])
def test_variable_sequence_lengths(
    model: DecoderOnlyTransformer, seq_len: int
) -> None:
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len))

    logits = model(x)

    assert logits.shape == (BATCH_SIZE, seq_len, VOCAB_SIZE)


def test_gradients_reach_all_components(
    model: DecoderOnlyTransformer,
) -> None:
    x = torch.randint(
        0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)
    )

    logits = model(x)
    loss = logits.square().mean()
    loss.backward()

    components = {
        "embeddings": model._embedings,
        "transformer_stack": model._transformer_stack,
        "final_layer_norm": model._layer_norm,
        "token_projection": model._token_projection,
    }

    for component_name, component in components.items():
        parameters = list(component.named_parameters())

        assert parameters, f"No parameters in {component_name}"

        for name, param in parameters:
            assert param.grad is not None, (
                f"No gradient for {component_name}.{name}"
            )
            assert torch.isfinite(param.grad).all(), (
                f"Invalid gradient for {component_name}.{name}"
            )

        assert any(
            torch.count_nonzero(param.grad).item() > 0
            for _, param in parameters
        ), f"All gradients are zero in {component_name}"


@pytest.mark.parametrize("position", [0, 1, 3, 6])
def test_causal_behavior(
    model: DecoderOnlyTransformer, position: int
) -> None:
    x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    x_modified = x.clone()

    # Change every future token, ensuring it differs.
    x_modified[:, position + 1 :] = (
        x_modified[:, position + 1 :] + 1
    ) % VOCAB_SIZE

    with torch.no_grad():
        original_logits = model(x)
        modified_logits = model(x_modified)

    # Predictions at and before position must be unaffected.
    torch.testing.assert_close(
        original_logits[:, : position + 1, :],
        modified_logits[:, : position + 1, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_batch_independence(
    model: DecoderOnlyTransformer,
) -> None:
    x1 = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    x2 = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))

    batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        batch_logits = model(batch)
        logits1 = model(x1)
        logits2 = model(x2)

    torch.testing.assert_close(
        batch_logits[0:1], logits1, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        batch_logits[1:2], logits2, rtol=1e-5, atol=1e-6
    )


def test_output_contains_raw_logits(
    model: DecoderOnlyTransformer,
) -> None:
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    with torch.no_grad():
        logits = model(x)

    # Softmax probabilities sum to 1 along the vocabulary axis.
    # Raw logits generally do not.
    sums = logits.sum(dim=-1)

    assert not torch.allclose(
        sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6
    )

    # The caller can convert logits to probabilities.
    probabilities = torch.softmax(logits, dim=-1)

    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(BATCH_SIZE, SEQ_LEN),
        rtol=1e-5,
        atol=1e-6,
    )


def test_forward_matches_explicit_pipeline(
    model: DecoderOnlyTransformer,
) -> None:
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    with torch.no_grad():
        # Reconstruct the complete forward pass manually.
        embeddings = model._embedings(x)

        contextual = model._transformer_stack(embeddings)

        normalized = model._layer_norm(contextual)

        expected_logits = model._token_projection(normalized)

        actual_logits = model(x)

    assert embeddings.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert contextual.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert normalized.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    assert expected_logits.shape == (
        BATCH_SIZE, SEQ_LEN, VOCAB_SIZE
    )

    torch.testing.assert_close(actual_logits, expected_logits)


def test_parameters_are_registered(
    model: DecoderOnlyTransformer,
) -> None:
    components = [
        model._embedings,
        model._transformer_stack,
        model._layer_norm,
        model._token_projection,
    ]

    registered_ids = {id(p) for p in model.parameters()}

    for component in components:
        for param in component.parameters():
            assert id(param) in registered_ids


def test_rejects_invalid_token_ids(
    model: DecoderOnlyTransformer,
) -> None:
    x = torch.tensor([[0, 1, VOCAB_SIZE]], dtype=torch.long)

    with pytest.raises((IndexError, RuntimeError)):
        model(x)
