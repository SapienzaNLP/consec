import argparse
from collections import Counter

from src.utils.wsd import read_from_raganato, expand_raganato_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-raganato-path")
    parser.add_argument("--output-path")
    return parser.parse_args()


def compute_annotation_ratio(raganato_path: str, output_path: str) -> None:

    annotated_count = Counter()
    unannotated_count = Counter()

    for _, _, wsd_sentence in read_from_raganato(*expand_raganato_path(raganato_path)):

        for wsd_instance in wsd_sentence:

            if wsd_instance.labels is None:
                unannotated_count[(wsd_instance.annotated_token.lemma, wsd_instance.annotated_token.pos)] += 1
            else:
                annotated_count[(wsd_instance.annotated_token.lemma, wsd_instance.annotated_token.pos)] += 1

    output_file = open(output_path, "w")

    for (lemma, pos), uc in unannotated_count.items():

        ac = annotated_count[(lemma, pos)]

        output_file.write(f"{lemma}#{pos}\t{ac}\t{uc}\t{ac/uc}\n")

    output_file.close()


def main():
    args = parse_args()
    compute_annotation_ratio(args.input_raganato_path, args.output_path)


if __name__ == "__main__":
    main()
