from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    
    def __init__(self):
        self._vocab = dict[str, int]()
        self._reverse_vocab = dict[int, str]()
        self.vocab_size = 0
            
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def build_vocab(self, corpus:str):
        pass
    
    def __len__(self) -> int:
        return len(self._vocab)

class SimpleCharacterTokenizer(Tokenizer):
    def encode(self, text:str) -> List[int]:
        return [self._vocab[t] for t in text]
            
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self._reverse_vocab[t] for t in tokens])

    def build_vocab(self, corpus: str):
        for c in corpus:
            if c not in self._vocab:
                next_index = len(self._vocab)
                self._vocab[c] = next_index
                self._reverse_vocab[next_index] = c
                self.vocab_size += 1

    
    



