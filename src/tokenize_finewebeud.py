from array import array
from io import BufferedWriter
from pathlib import Path
from typing import cast

from datasets import load_dataset
import typer
import hashlib

from tokenizer import HFTokenizer

BATCH_SIZE = 100
TRAIN_VALIDATION_SPLIT = 95  # 95% goes to training, 5% to validation

app = typer.Typer()


def append_batch(
    file: BufferedWriter, tokenizer: HFTokenizer, text_batch: list[str]
) -> int:
    eos = tokenizer.token_to_id("[EOS]")
    assert eos is not None

    size = 0

    for tokens in tokenizer.encode_batch(text_batch):
        file.write(array("H", tokens + [eos]).tobytes())
        size += len(tokens) + 1

    return size


@app.command()
def main() -> None:
    train_data = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    path = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "bpe_4096.json"
    tokenizer = HFTokenizer(str(path))

    with (
        open("train_tokens.bin", "wb") as train_file,
        open("validation_tokens.bin", "wb") as validation_file,
    ):
        size = 0
        train_batch: list[str] = []
        validation_batch: list[str] = []

        for batch in train_data.batch(BATCH_SIZE):  # pyright: ignore[reportUnknownVariableType]
            for index, document_id in enumerate(batch["id"]):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                bucket = (
                    int.from_bytes(
                        hashlib.sha1(cast(str, document_id).encode()).digest(), "big"
                    )
                    % 100
                )  # pyright: ignore[reportUnknownArgumentType]

                text: str = batch["text"][index]  # pyright: ignore[reportUnusedVariable, reportIndexIssue, reportUnknownVariableType]

                if bucket < TRAIN_VALIDATION_SPLIT:
                    train_batch.append(text)  # pyright: ignore[reportUnknownArgumentType]
                else:
                    validation_batch.append(text)  # pyright: ignore[reportUnknownArgumentType]

                if len(train_batch) >= BATCH_SIZE:
                    size += append_batch(train_file, tokenizer, train_batch)
                    train_batch = []

                if len(validation_batch) >= BATCH_SIZE:
                    size += append_batch(validation_file, tokenizer, validation_batch)
                    validation_batch = []

            if size > 100 * 1024 * 1024:
                if train_batch:
                    append_batch(train_file, tokenizer, train_batch)
                if validation_batch:
                    append_batch(validation_file, tokenizer, validation_batch)
                break


if __name__ == "__main__":
    app()
