from abc import ABC, abstractmethod
from typing import Sequence
from tokenizers import Tokenizer as HuggingFaceTokenizer


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    def __len__(self) -> int:
        return self.vocab_size


class SimpleCharacterTokenizer(Tokenizer):
    def __init__(self) -> None:
        self._vocab = dict[str, int]()
        self._reverse_vocab = dict[int, str]()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> list[int]:
        return [self._vocab[t] for t in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self._reverse_vocab[t] for t in tokens])

    def build_vocab(self, corpus: str) -> None:
        for c in corpus:
            if c not in self._vocab:
                next_index = len(self._vocab)
                self._vocab[c] = next_index
                self._reverse_vocab[next_index] = c


class HFTokenizer(Tokenizer):
    def __init__(self, serialized_tokenizer_path: str) -> None:
        self._tokenizer = HuggingFaceTokenizer.from_file(serialized_tokenizer_path)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def encode_batch(self, batch: Sequence[str]) -> list[list[int]]:
        return [batch.ids for batch in self._tokenizer.encode_batch(batch)]

    def token_to_id(self, token_id: str) -> int | None:
        return self._tokenizer.token_to_id(token_id)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False)
