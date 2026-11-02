import torch
from torch import nn

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.dim() == 2
        assert x.dtype == torch.long
        _, T = x.shape
        assert T <= self.max_seq_len
        
        return self.token_embeddings(x)   