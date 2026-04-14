from typing import List, Tuple
import torch
import random


class DataLoader:
    def __init__(self, tokens: List[int], batch_size: int, seq_len: int) -> None:
        self._tokens = tokens
        self._batch_size = batch_size
        self._seq_len = seq_len

        if len(tokens) < seq_len + 2:
            raise Exception(
                "Length of tokens should be larger than the sequence length + 1"
            )

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = list[torch.Tensor]()
        y = list[torch.Tensor]()
        for _ in range(self._batch_size):
            index = random.randint(0, len(self._tokens) - self._seq_len - 2)
            x.append(
                torch.tensor(
                    self._tokens[index : index + self._seq_len], dtype=torch.long
                )
            )
            y.append(
                torch.tensor(
                    self._tokens[index + 1 : index + self._seq_len + 1],
                    dtype=torch.long,
                )
            )

        return (torch.stack(x), torch.stack(y))
