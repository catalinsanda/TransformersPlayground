from typing import cast

import pytest
import torch
from embeddings import EmbeddingLayer

EMBEDDING_DIM = 100
MAX_SEQ_LEN = 1000
VOCAB_SIZE = 26


@pytest.fixture
def embedding_layer() -> EmbeddingLayer:
    return EmbeddingLayer(
        vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_seq_len=MAX_SEQ_LEN
    )


def test_embedding_shape(embedding_layer: EmbeddingLayer) -> None:
    x = torch.zeros((2, 3), dtype=torch.long)
    x = cast(torch.Tensor, embedding_layer(x))

    assert x.dim() == 3
    assert x.shape == (2, 3, EMBEDDING_DIM)


def test_embed_pos_is_different(embedding_layer: EmbeddingLayer) -> None:
    x = torch.tensor([[5, 5, 5]], dtype=torch.long)
    x = cast(torch.Tensor, embedding_layer(x))

    assert not torch.equal(x[0, 0], x[0, 1])
    assert not torch.equal(x[0, 1], x[0, 2])
    assert not torch.equal(x[0, 0], x[0, 2])


def test_same_token_same_position_across_batch_is_equal(
    embedding_layer: EmbeddingLayer,
) -> None:
    x = torch.tensor(
        [
            [7, 1, 2],
            [7, 3, 4],
        ],
        dtype=torch.long,
    )
    x = cast(torch.Tensor, embedding_layer(x))

    assert torch.equal(x[0, 0], x[1, 0])
