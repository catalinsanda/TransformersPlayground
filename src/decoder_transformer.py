import torch
from torch import nn

from embeddings import EmbeddingLayer
from transformer import TransformerStack

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, num_heads:int, num_layers:int, max_seq_len:int):
        super().__init__()
        
        self._embedings = EmbeddingLayer(vocab_size, d_model, max_seq_len)
        self._transformer_stack = TransformerStack(d_model, num_heads, num_layers)
        self._layer_norm = nn.LayerNorm(d_model)
        self._token_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self._embedings(x)
        x = self._transformer_stack(x)
        x = self._layer_norm(x)

        return self._token_projection(x)
