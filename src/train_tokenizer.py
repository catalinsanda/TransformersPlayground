from typing import Any, Iterable, Iterator, List, Tuple

from datasets import IterableDataset, load_dataset

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def extract_text(iterable: Iterable[Any], path: List[str]) -> Iterable[Any]:
    for object in iterable:
        if len(path) == 1:
            text = object[path[0]]
            if isinstance(text, str) and text.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
                yield text
        else:
            for text in extract_text(object[path[0]], path[1:]):
                yield text


def build_data_iterator(
    iterator_size_tuples: List[Tuple[IterableDataset, int, List[str]]],
) -> Iterator[str]:
    for dataset, max_size, text_key_path in iterator_size_tuples:
        output_bytes = 0
        for text in extract_text(dataset, text_key_path):
            text_len = len(text.encode("utf-8"))
            if output_bytes + text_len <= max_size:
                yield text
                output_bytes += text_len
            else:
                break


def main() -> None:
    pretrain_data = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    chat_data = load_dataset(
        "HuggingFaceTB/smol-smoltalk",
        split="train",
        streaming=True,
    )

    pretrain_data = pretrain_data.shuffle(
        seed=42,
        buffer_size=1000,
    )

    chat_data = chat_data.shuffle(
        seed=42,
        buffer_size=1000,
    )

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=4096,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[BOS]",
            "[EOS]",
            "[IM_START]",
            "[IM_END]",
            "[END_TURN]",
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        build_data_iterator(
            [
                (pretrain_data, 80 * 1024 * 1024, ["text"]),
                (chat_data, 20 * 1024 * 1024, ["messages", "content"]),
            ]
        ),
        trainer=trainer,
    )

    tokenizer.save("data/tokenizers/bpe_4096.json")


if __name__ == "__main__":
    main()
