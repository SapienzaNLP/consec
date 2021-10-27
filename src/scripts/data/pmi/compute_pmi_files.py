from typing import Dict, List, Tuple, Set
import argparse
import collections
import itertools
import re
from tqdm import tqdm


def index_vocabulary(vocabulary_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    word2index, index2word = dict(), dict()
    with open(vocabulary_path) as f:
        for i, line in enumerate(f):
            word = line.strip()
            word2index[word] = i
            index2word[i] = word
    return word2index, index2word


def index_corpus(word2index: Dict[str, int], corpus_path: str) -> Dict[int, Dict[int, List[int]]]:
    word_corpus_index = collections.defaultdict(dict)
    with open(corpus_path) as f:
        for doc_id, line in enumerate(f):
            for word_position, word in enumerate(line.strip().split(" ")):
                if word in word2index:
                    wi = word2index[word]
                    if wi not in word_corpus_index:
                        word_corpus_index[wi] = collections.defaultdict(list)
                    word_corpus_index[wi][doc_id].append(word_position)
    return word_corpus_index


def compute_corpus_occurrences(
    word2index: Dict[str, int], word_corpus_index: Dict[int, Dict[int, List[int]]]
) -> Dict[int, Set[int]]:
    corpus_occurrences = collections.defaultdict(set)
    mws_splitter = re.compile("[_ ]")

    for word, index in word2index.items():
        word_parts = mws_splitter.split(word)

        if len(word_parts) > 1:

            if not all(len(wp.strip()) > 0 for wp in word_parts):
                print(f"Found multiword with a blank component ({word_parts}). Skipping.")
                continue

            word_parts_indices = [word2index[wp] for wp in word_parts]
            wpi_docs = [set(word_corpus_index[wpi].keys()) for wpi in word_parts_indices]
            final_docs = wpi_docs[0]
            for wpid in wpi_docs[1:]:
                final_docs = final_docs.intersection(wpid)

            # the word parts does not appear altogether
            if len(final_docs) == 0:
                corpus_occurrences[index] = set()
                continue

            for doc_id in final_docs:
                wpis_positions = [word_corpus_index[wpi][doc_id] for wpi in word_parts_indices]
                for wpis_position in zip(*wpis_positions):
                    diffs = [wpis_position[i + 1] - wpis_position[i] for i in range(len(wpis_position) - 1)]
                    if set(diffs) == {1}:
                        corpus_occurrences[index].add(doc_id)
                        break
        else:
            corpus_occurrences[index] = set(word_corpus_index[index].keys())

    return corpus_occurrences


def compute_cooccurrences(words_occurrences: Dict[int, Set[int]]) -> List[Tuple[str, str, int]]:
    cooccurrences = collections.Counter()

    doc2words = collections.defaultdict(list)
    for wid, doc_ids in words_occurrences.items():
        for doc_id in doc_ids:
            doc2words[doc_id].append(wid)

    for doc, words in tqdm(doc2words.items(), total=len(doc2words)):
        cooccurrences.update(itertools.combinations(words, 2))

    return [(w1, w2, coocs) for (w1, w2), coocs in cooccurrences.items()]


def compute_pmi_files(
    vocabulary_path: str,
    corpus_path: str,
    output_dir: str,
    min_coocs_support: int,
) -> None:
    word2index, index2word = index_vocabulary(vocabulary_path)
    word_corpus_index = index_corpus(word2index, corpus_path)
    word_corpus_occurrences = compute_corpus_occurrences(word2index, word_corpus_index)

    words_occurrences = [(word, len(doc_ids)) for word, doc_ids in word_corpus_occurrences.items()]
    words_cooccurrences = compute_cooccurrences(word_corpus_occurrences)

    with open(f"{output_dir}/words_counter.tsv", "w") as f:
        for word, nocc in sorted(words_occurrences, key=lambda x: x[1], reverse=True):
            f.write(f"{index2word[word]}\t{nocc}\n")

    with open(f"{output_dir}/word_pairs_counter.tsv", "w") as f:
        for word1, word2, nocc in sorted(words_cooccurrences, key=lambda x: x[2], reverse=True):
            if nocc >= min_coocs_support:
                f.write(f"{index2word[word1]}\t{index2word[word2]}\t{nocc}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary")
    parser.add_argument("--corpus")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-coocs-support", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    compute_pmi_files(args.vocabulary, args.corpus, args.output_dir, args.min_coocs_support)


if __name__ == "__main__":
    main()
