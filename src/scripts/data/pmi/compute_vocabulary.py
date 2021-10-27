from typing import List
import argparse
import re
from src.utils.wsd import read_from_raganato, expand_raganato_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raganato-paths",
        nargs="+",
    )
    parser.add_argument("--output-path")
    return parser.parse_args()


def compute_vocabulary(raganato_paths: List[str], output_path: str):
    vocabulary_store = set()
    mws_splitter = re.compile("[_ ]")
    for rag_path in raganato_paths:
        for _, _, wsd_sentence in read_from_raganato(*expand_raganato_path(rag_path)):
            for wsd_instance in wsd_sentence:
                if wsd_instance.labels is None:
                    continue
                word_parts = mws_splitter.split(wsd_instance.annotated_token.text)
                vocabulary_store.update(word_parts)
                if len(word_parts) > 1:
                    vocabulary_store.add("_".join(word_parts))

    with open(output_path, "w") as f:
        for word in vocabulary_store:
            if len(word.strip()) > 0:
                f.write(f"{word}\n")


def main():
    args = parse_args()
    compute_vocabulary(args.raganato_paths, args.output_path)


if __name__ == "__main__":
    main()
