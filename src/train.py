import torch
import string

from decoder_transformer import DecoderOnlyTransformer
from generate import generate
from loss import compute_loss
from tokenizer import SimpleCharacterTokenizer
from typing import List


def main():
    TRAIN_STEPS = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = string.printable
    tokenizer = SimpleCharacterTokenizer()
    tokenizer.build_vocab(vocab)
    input = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    x = torch.tensor([input[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([input[1:]], dtype=torch.long, device=device)

    model = DecoderOnlyTransformer(len(vocab), 64, 4, 2, 64).to(device)

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

        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.4f}")

    prompt = tokenizer.encode("The quick ")    
    
    output = generate(
        model, torch.tensor([prompt], dtype=torch.long, device=device), len(input) - len(prompt), 64, 1.0
    )

    decoded_input: List[int] = output[0].cpu().tolist() # type: ignore
    text = tokenizer.decode(decoded_input)

    print(f"Output: \"{text}\"")


if __name__ == "__main__":
    main()
