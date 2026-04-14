from random import choice
from pytest import FixtureRequest

import pytest
import string

from tokenizer.tokenizer import SimpleCharacterTokenizer


@pytest.fixture
def random_string(request: FixtureRequest) -> str:
    return "".join(choice(string.printable) for _ in range(request.param))


@pytest.fixture
def tokenizer() -> SimpleCharacterTokenizer:
    return SimpleCharacterTokenizer()


@pytest.mark.parametrize(
    "random_string",
    [0, 1, 10, 50, 100, 500, 1000],
    indirect=True,
)
def test_encode_decode(tokenizer: SimpleCharacterTokenizer, random_string: str) -> None:
    tokenizer.build_vocab(random_string)
    assert tokenizer.decode(tokenizer.encode(random_string)) == random_string


def test_known_text(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "abca"

    tokenizer.build_vocab(test_string)
    assert tokenizer.encode(test_string) == [0, 1, 2, 0]


def test_vocab_size_matches_unique_characters(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "abca"

    tokenizer.build_vocab(test_string)

    assert tokenizer.vocab_size == 3


def test_encoding_is_deterministic(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "hello world"

    tokenizer.build_vocab(test_string)

    first = tokenizer.encode(test_string)
    second = tokenizer.encode(test_string)

    assert first == second


def test_encoded_ids_are_within_bounds(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "some random text"

    tokenizer.build_vocab(test_string)
    encoded = tokenizer.encode(test_string)

    assert all(0 <= token_id < tokenizer.vocab_size for token_id in encoded)


def test_unknown_character_raises(tokenizer: SimpleCharacterTokenizer) -> None:
    tokenizer.build_vocab("abc")

    with pytest.raises(KeyError):
        tokenizer.encode("abd")


def test_encode_empty_string(tokenizer: SimpleCharacterTokenizer) -> None:
    tokenizer.build_vocab("abc")

    assert tokenizer.encode("") == []


def test_decode_empty_list(tokenizer: SimpleCharacterTokenizer) -> None:
    tokenizer.build_vocab("abc")

    assert tokenizer.decode([]) == ""


def test_space_and_newline_are_handled(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "hi there\nok"

    tokenizer.build_vocab(test_string)

    encoded = tokenizer.encode(test_string)
    decoded = tokenizer.decode(encoded)

    assert decoded == test_string


def test_single_character(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "aaaaa"

    tokenizer.build_vocab(test_string)

    assert tokenizer.vocab_size == 1
    assert tokenizer.encode(test_string) == [0, 0, 0, 0, 0]
    assert tokenizer.decode([0, 0, 0, 0, 0]) == test_string


def test_decode_reconstructs_known_sequence(tokenizer: SimpleCharacterTokenizer) -> None:
    test_string = "abc"

    tokenizer.build_vocab(test_string)

    assert tokenizer.decode([0, 1, 2]) == "abc"