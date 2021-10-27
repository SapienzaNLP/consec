import argparse

import hydra
import pytorch_lightning as pl
from typing import Iterator, Tuple, List, Optional

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.consec_dataset import ConsecDataset, ConsecSample, ConsecDefinition
from src.disambiguation_corpora import DisambiguationInstance
from src.pl_modules import ConsecPLModule
from src.consec_tokenizer import DeBERTaTokenizer, ConsecTokenizer


def predict(
    module: pl.LightningModule,
    tokenizer: ConsecTokenizer,
    samples: Iterator[ConsecSample],
    text_encoding_strategy: str,
    token_batch_size: int = 1024,
    progress_bar: bool = False,
) -> Iterator[Tuple[ConsecSample, List[float]]]:

    # todo only works on single gpu
    device = next(module.parameters()).device

    # todo hardcoded dataset
    dataset = ConsecDataset.from_samples(
        samples,
        tokenizer=tokenizer,
        use_definition_start=True,
        text_encoding_strategy=text_encoding_strategy,
        tokens_per_batch=token_batch_size,
        max_batch_size=128,
        section_size=2_000,
        prebatch=True,
        shuffle=False,
        max_length=tokenizer.model_max_length,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # predict

    iterator = dataloader
    progress_bar = tqdm() if progress_bar else None

    for batch in iterator:

        batch_samples = batch["original_sample"]
        batch_definitions_positions = batch["definitions_positions"]

        with autocast(enabled=True):
            with torch.no_grad():
                batch_out = module(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
                batch_predictions = batch_out["pred_probs"]

        for sample, dp, probs in zip(batch_samples, batch_definitions_positions, batch_predictions):
            definition_probs = []
            for start in dp:
                definition_probs.append(probs[start].item())
            yield sample, definition_probs
            if progress_bar is not None:
                progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()


def interactive_main(
    model_checkpoint_path: str,
    device: int,
):
    def read_ld_pairs() -> List[Tuple[str, str, Optional[str]]]:
        pairs = []
        while True:
            line = input(" * ").strip()
            if line == "":
                break
            parts = line.split(" --- ")
            if len(parts) == 3:
                l, d, p = parts
                p = int(p)
            elif len(parts) == 2:
                l, d = parts
                p = None
            else:
                raise ValueError
            pairs.append((l, d, p))
        return pairs

    # load model
    # todo decouple BasecPLModule
    module = ConsecPLModule.load_from_checkpoint(model_checkpoint_path)
    module.to(torch.device(device if device != -1 else "cpu"))
    module.freeze()
    module.sense_extractor.evaluation_mode = True

    # load tokenizer
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    while True:

        # read marked text
        text = input("Enter space-separated text: ").strip()
        tokens = text.split(" ")
        target_position = int(input("Target position: ").strip())

        # read candidates definitions
        print('Enter candidate lemma-def pairs. " --- " separated. Enter to stop')
        candidate_definitions = read_ld_pairs()
        candidate_definitions = [ConsecDefinition(d, l) for l, d, _ in candidate_definitions]

        # read context definitions
        print(
            'Enter context lemma-def-position tuples. " --- " separated. Position should be token position in space-separated input. Enter to stop'
        )
        context_definitions = read_ld_pairs()
        context_definitions = [(ConsecDefinition(d, l), p) for l, d, p in context_definitions]

        # predict
        _, probs = next(
            predict(
                module,
                tokenizer,
                [
                    ConsecSample(
                        sample_id="interactive-d0",
                        position=target_position,
                        disambiguation_context=[
                            DisambiguationInstance("d0", "s0", "i0", t, None, None, None) for t in tokens
                        ],
                        candidate_definitions=candidate_definitions,
                        gold_definitions=None,
                        context_definitions=context_definitions,
                        in_context_sample_id2position={'interactive-d0': target_position},
                        disambiguation_instance=None,
                        kwargs={},
                    )
                ],
                text_encoding_strategy="simple-with-linker",  # todo hardcoded core param
            )
        )

        idxs = torch.tensor(probs).argsort(descending=True)
        print(f"\t# predictions")
        for idx in idxs:
            idx = idx.item()
            print(
                f"\t\t * {probs[idx]:.4f} \t {candidate_definitions[idx].linker} \t {candidate_definitions[idx].text} "
            )


def file_main(
    model_checkpoint_path: str,
    input_path: str,
    output_path: str,
    device: int,
    token_batch_size: int,
):
    raise NotImplementedError


def main():
    args = parse_args()
    if args.t:
        interactive_main(
            args.model_checkpoint,
            device=args.device,
        )
    else:
        file_main(
            args.model_checkpoint,
            args.f,
            args.o,
            device=args.device,
            token_batch_size=args.token_batch_size,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="Path to pl_modules checkpoint")
    parser.add_argument("--device", type=int, default=-1, help="Device")
    # interactive params
    parser.add_argument("-t", action="store_true", help="Interactive mode")
    # generation params
    parser.add_argument("-f", type=str, default=None, help="Input file")
    parser.add_argument("-o", type=str, default=None, help="Output file")
    parser.add_argument("--token-batch-size", type=int, default=128, help="Token batch size")
    # return
    return parser.parse_args()


if __name__ == "__main__":
    main()
