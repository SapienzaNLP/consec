import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raganato-path")
    parser.add_argument("--vocabulary-path")
    return parser.parse_args()


def compute_coverage(raganato_path: str, vocabulary_path: str) -> None:

    with open(vocabulary_path) as f:
        vocabulary = set()
        for line in f:
            vocabulary.add(line.strip())

    total_counter = []
    word_level_counter = dict()

    for _, _, wsd_sentence in read_from_raganato(*expand_raganato_path(raganato_path)):
        for wsd_instance in wsd_sentence:
            if wsd_instance.text not in vocabulary:
                total_counter.append(0)
                word_level_counter[wsd_instance.text] = 0
            else:
                total_counter.append(1)
                word_level_counter[wsd_instance.text] = 1

    print(f"Instance level coverage: {sum(total_counter)/len(total_counter)*100:.2f}%")
    print(f"Word level coverage: {sum(word_level_counter.values())/len(word_level_counter)*100:.2f}%")


def main():
    args = parse_args()
    compute_coverage(args.raganato_path, args.vocabulary_path)


if __name__ == "__main__":
    main()
