
from pathlib import Path

import pytest

from tokenizer import HFTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> HFTokenizer:
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "tokenizers"
        / "bpe_4096.json"
    )
    return HFTokenizer(str(path))


def test_vocab_size(tokenizer: HFTokenizer) -> None:
    assert tokenizer.vocab_size == 4096
    assert len(tokenizer) == 4096


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Multiple   spaces and\ttabs.",
        "First line\nSecond line\n\nFourth line.",
        "Trailing spaces   ",
        "\n\nLeading newlines",
        "Romanian: ă â î ș ț",
        "Japanese: こんにちは世界",
        "Chinese: 你好世界",
        "Emojis: 🐱 🚀 ❤️",
        "Math: ∑ ∞ √ π ≠",
        "Mixed Unicode: café naïve résumé",
    ],
)
def test_encode_decode_roundtrip(
    tokenizer: HFTokenizer, text: str
) -> None:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


@pytest.mark.parametrize(
    ("special_token", "expected_id"),
    [
        ("[UNK]", 0),
        ("[PAD]", 1),
        ("[BOS]", 2),
        ("[EOS]", 3),
        ("[IM_START]", 4),
        ("[IM_END]", 5),
        ("[END_TURN]", 6),
    ],
)
def test_special_tokens(
    tokenizer: HFTokenizer,
    special_token: str,
    expected_id: int,
) -> None:
    assert tokenizer.encode(special_token) == [expected_id]
    assert tokenizer.decode([expected_id]) == special_token


def test_special_tokens_not_added_automatically(
    tokenizer: HFTokenizer,
) -> None:
    encoded = tokenizer.encode("Hello, world!")

    assert not set(encoded).intersection(range(7))


def test_deterministic_encoding(tokenizer: HFTokenizer) -> None:
    text = "Transformers use self-attention mechanisms."

    first = tokenizer.encode(text)
    second = tokenizer.encode(text)

    assert first == second


def test_no_unknown_tokens(tokenizer: HFTokenizer) -> None:
    text = "Unusual Unicode: 🦄 你好 𐍈 ∑"

    encoded = tokenizer.encode(text)

    assert 0 not in encoded
    assert tokenizer.decode(encoded) == text


def test_token_ids_within_vocabulary(tokenizer: HFTokenizer) -> None:
    text = "Testing token IDs for a decoder-only transformer."

    encoded = tokenizer.encode(text)

    assert all(0 <= token_id < tokenizer.vocab_size for token_id in encoded)
