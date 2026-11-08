import torch
import string
from functools import partial

from decoder_transformer import DecoderOnlyTransformer
from generate import generate
from kv_cache import ModelKVCache
from loss import compute_loss
from tokenizer import SimpleCharacterTokenizer
from typing import List


def main():
    TRAIN_STEPS = 1000
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    MAX_SEQ_LENGTH = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = string.printable
    tokenizer = SimpleCharacterTokenizer()
    tokenizer.build_vocab(vocab)
    input = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    x = torch.tensor([input[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([input[1:]], dtype=torch.long, device=device)

    model = DecoderOnlyTransformer(
        len(vocab), D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LENGTH
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    loss = torch.tensor([])

    model.train()

    for step in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = compute_loss(logits, y)

        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    print(f"Final loss: {loss.item():.4f}")

    prompt = tokenizer.encode("The quick ")

    kv_cache = ModelKVCache(
        NUM_LAYERS,
        NUM_HEADS,
        1,
        D_MODEL // NUM_HEADS,
        device=device,
        dtype=torch.float32,
    )

    output = generate(
        model,
        torch.tensor([prompt], dtype=torch.long, device=device),
        len(input) - len(prompt),
        64,
        kv_cache=kv_cache,
        temperature=1.0,
    )

    decoded_input: List[int] = output[0].cpu().tolist()  # type: ignore
    text = tokenizer.decode(decoded_input)

    print(f'Output: "{text}"')


if __name__ == "__main__":
    main()
