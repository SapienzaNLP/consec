import collections
from typing import List
import argparse

from src.utils.wsd import read_from_raganato, expand_raganato_path


def compute_sense_count(raganato_paths: List[str], output_path: str) -> None:
    sense_counter = collections.Counter()
    for rag_path in raganato_paths:
        for _, _, wsd_sentence in read_from_raganato(*expand_raganato_path(rag_path)):
            for wsd_instance in wsd_sentence:
                if wsd_instance.labels is None:
                    continue
                sense_counter.update(wsd_instance.labels)

    with open(output_path, "w") as f:
        for sense, count in sense_counter.items():
            f.write(f"{sense}\t{count}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raganato-paths", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    compute_sense_count(args.raganato_paths, args.output_path)


if __name__ == "__main__":
    main()
