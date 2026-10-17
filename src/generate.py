import string
from typing import List, cast

import torch

from decoder_transformer import DecoderOnlyTransformer
from tokenizer import SimpleCharacterTokenizer


def generate(
    model: DecoderOnlyTransformer,
    tokens: torch.Tensor,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    model.eval()

    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = tokens[:, -max_seq_len:]
            logits = model(context)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), 1)
    
    return tokens

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = string.printable
    tokenizer = SimpleCharacterTokenizer()
    tokenizer.build_vocab(vocab)
    input = tokenizer.encode("Say something...")
    model = DecoderOnlyTransformer(len(vocab), 256, 8, 16, 1024).to(device)
    
    output = generate(model, torch.tensor([input], dtype=torch.long, device=device), 64, 1024, 1.0)
    
    decoded_input: List[int] = cast(List[int], output[0].int().tolist()) # type: ignore
    text = tokenizer.decode(decoded_input)
    
    print(text)
    

if __name__ == "__main__":
    main()