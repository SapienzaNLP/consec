from typing import Tuple, Dict, List

from sacremoses import MosesDetokenizer

_md = MosesDetokenizer(lang="en")


def detokenize(sentence: str) -> Tuple[str, Dict[int, int]]:

    offset_map = {}
    detokenized_sentence = _md.detokenize(sentence.split(" "), return_str=True)

    it = iter(enumerate(sentence))
    for j, nc in enumerate(detokenized_sentence):
        while True:
            i, oc = next(it)
            offset_map[i] = j
            if oc == nc:
                break
    offset_map[len(sentence)] = len(detokenized_sentence)

    return detokenized_sentence, offset_map


def detokenize_text(text_tokens: List[str], ignore_idxs: List[int]) -> str:
    ignore_idxs = sorted(ignore_idxs)
    text, start = "", 0
    for ignore_idx in ignore_idxs:
        text += detokenize(" ".join(text_tokens[start:ignore_idx]))[0]
        text += " " + text_tokens[ignore_idx] + " "
        start = ignore_idx + 1
    if start != len(text_tokens):
        text += detokenize(" ".join(text_tokens[start:]))[0]
    return text.strip()


if __name__ == "__main__":

    def old_detokenize_text(text_tokens: List[str], ignore_idx: int) -> str:
        pre_instance = detokenize(" ".join(text_tokens[:ignore_idx]))[0]
        post_instance = detokenize(" ".join(text_tokens[ignore_idx + 1 :]))[0]
        return pre_instance + f" {text_tokens[ignore_idx]} " + post_instance

    tokens = ["I", "have", "a", "{{{ dog }}}", "."]
    print(old_detokenize_text(tokens, ignore_idx=3))
    print(detokenize_text(tokens, ignore_idxs=[3]))
    print(detokenize_text(tokens, ignore_idxs=[]))
    print()

    tokens = ["I", "have", "a  ", "{{{ dog }}}"]
    print(old_detokenize_text(tokens, ignore_idx=3))
    print(detokenize_text(tokens, ignore_idxs=[3]))
    print(detokenize_text(tokens, ignore_idxs=[]))
    print()

    tokens = ["{{{ I }}}", "have", "a", "dog"]
    print(old_detokenize_text(tokens, ignore_idx=0))
    print(detokenize_text(tokens, ignore_idxs=[0]))
    print(detokenize_text(tokens, ignore_idxs=[]))
    print()
