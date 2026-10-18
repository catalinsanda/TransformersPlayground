
from pathlib import Path
from typing import List

import torch

from data_loader import DataLoader
from decoder_transformer import DecoderOnlyTransformer
from generate import generate
from loss import compute_loss, estimate_loss
from tokenizer import SimpleCharacterTokenizer


# ============================================================
# Hyperparameters
# ============================================================

# Model architecture
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_SEQ_LEN = 256

# Training
TRAIN_STEPS = 15_000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.9

# Evaluation and early stopping
EVAL_INTERVAL = 250
EVAL_BATCHES = 20
EARLY_STOP_PATIENCE = 8
MIN_DELTA = 0.001

# Checkpoint
CHECKPOINT_FILE = "shakespeare_checkpoint.pt"

# Generation
GENERATION_PROMPT = "ROMEO:\n"
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8

# Dataset
DATASET_PATH = "data/tinyshakespeare.txt"


# ============================================================
# Training
# ============================================================


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Tokenizer
    tokenizer = SimpleCharacterTokenizer()
    tokenizer.build_vocab(full_text)

    print(f"Found {tokenizer.vocab_size} characters in the dataset")

    tokens = tokenizer.encode(full_text)

    # Train / validation split
    split_idx = int(len(tokens) * TRAIN_SPLIT)

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_loader = DataLoader(
        train_tokens,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
    )

    val_loader = DataLoader(
        val_tokens,
        batch_size=BATCH_SIZE,
        seq_len=MAX_SEQ_LEN,
    )

    # Model
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_path = Path(CHECKPOINT_FILE)

    start_step = 0
    best_val_loss = float("inf")
    evaluations_without_improvement = 0

    if checkpoint_path.is_file():
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])

        start_step = checkpoint_data["step"] + 1
        best_val_loss = checkpoint_data["best_val_loss"]
        evaluations_without_improvement = checkpoint_data.get(
            "evaluations_without_improvement", 0
        )

        print(
            f"Loaded checkpoint at step {start_step - 1}, "
            f"best validation loss: {best_val_loss:.4f}"
        )

    model.train()

    last_loss: float | None = None

    # Training loop
    for step in range(start_step, TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = compute_loss(logits, y)

        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        if step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS - 1:
            train_loss = estimate_loss(
                model, train_loader, device, EVAL_BATCHES
            )
            val_loss = estimate_loss(
                model, val_loader, device, EVAL_BATCHES
            )

            last_loss = loss.item()

            print(
                f"Step {step:5d} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"Best: {best_val_loss:.4f}"
            )

            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                evaluations_without_improvement = 0

                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "evaluations_without_improvement": 0,
                    },
                    checkpoint_path,
                )

                print(
                    f"Saved checkpoint at step {step}, "
                    f"val_loss={best_val_loss:.4f}"
                )

            else:
                evaluations_without_improvement += 1

                print(
                    f"No improvement: "
                    f"{evaluations_without_improvement}/"
                    f"{EARLY_STOP_PATIENCE}"
                )

                if (
                    evaluations_without_improvement
                    >= EARLY_STOP_PATIENCE
                ):
                    print(
                        f"Early stopping at step {step}. "
                        f"Best validation loss: {best_val_loss:.4f}"
                    )
                    break

    if last_loss is not None:
        print(f"Last training loss: {last_loss:.4f}")

    # Restore the best model for generation
    if checkpoint_path.is_file():
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint_data["model"])

        print(
            f"Restored best model from step "
            f"{checkpoint_data['step']}, "
            f"validation loss: "
            f"{checkpoint_data['best_val_loss']:.4f}"
        )

    # Generate text
    prompt = tokenizer.encode(GENERATION_PROMPT)

    output = generate(
        model,
        torch.tensor(
            [prompt],
            dtype=torch.long,
            device=device,
        ),
        MAX_NEW_TOKENS,
        MAX_SEQ_LEN,
        TEMPERATURE,
    )

    decoded: List[int] = output[0].cpu().tolist()  # type: ignore
    text = tokenizer.decode(decoded)

    print(f'Output: "{text}"')


if __name__ == "__main__":
    main()
