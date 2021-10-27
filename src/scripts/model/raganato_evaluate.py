import tempfile
from typing import Tuple, Any, Dict, Optional, List

import hydra
import torch
from omegaconf import omegaconf, DictConfig

from src.consec_dataset import ConsecSample
from src.dependency_finder import EmptyDependencyFinder
from src.pl_modules import ConsecPLModule
from src.scripts.model.continuous_predict import Predictor
from src.sense_inventories import SenseInventory, WordNetSenseInventory
from src.utils.commons import execute_bash_command
from src.utils.hydra import fix
from src.utils.wsd import expand_raganato_path



def framework_evaluate(framework_folder: str, gold_file_path: str, pred_file_path: str) -> Tuple[float, float, float]:
    scorer_folder = f"{framework_folder}/Evaluation_Datasets"
    command_output = execute_bash_command(
        f"[ ! -e {scorer_folder}/Scorer.class ] && javac -d {scorer_folder} {scorer_folder}/Scorer.java; java -cp {scorer_folder} Scorer {gold_file_path} {pred_file_path}"
    )
    command_output = command_output.split("\n")
    p, r, f1 = [float(command_output[i].split("=")[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


def sample_prediction2sense(sample: ConsecSample, prediction: int, sense_inventory: SenseInventory) -> str:
    sample_senses = sense_inventory.get_possible_senses(
        sample.disambiguation_instance.lemma, sample.disambiguation_instance.pos
    )
    sample_definitions = [sense_inventory.get_definition(s) for s in sample_senses]

    for s, d in zip(sample_senses, sample_definitions):
        if d == sample.candidate_definitions[prediction].text:
            return s

    raise ValueError


def raganato_evaluate(
    raganato_path: str,
    wsd_framework_dir: str,
    module: ConsecPLModule,
    predictor: Predictor,
    wordnet_sense_inventory: WordNetSenseInventory,
    samples_generator: DictConfig,
    prediction_params: Dict[Any, Any],
    fine_grained_evals: Optional[List[str]] = None,
    reporting_folder: Optional[str] = None,
) -> Tuple[float, float, float, Optional[List[Tuple[str, float, float, float]]]]:

    # load tokenizer
    tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)

    # instantiate samples
    consec_samples = list(hydra.utils.instantiate(samples_generator, dependency_finder=EmptyDependencyFinder())())

    # predict
    disambiguated_samples = predictor.predict(
        consec_samples,
        already_kwown_predictions=None,
        reporting_folder=reporting_folder,
        **dict(module=module, tokenizer=tokenizer, **prediction_params),
    )

    # write predictions and evaluate
    with tempfile.TemporaryDirectory() as tmp_dir:

        # write predictions to tmp file
        with open(f"{tmp_dir}/predictions.gold.key.txt", "w") as f:
            for sample, idx in disambiguated_samples:
                f.write(f"{sample.sample_id} {sample_prediction2sense(sample, idx, wordnet_sense_inventory)}\n")

        # compute metrics
        p, r, f1 = framework_evaluate(
            wsd_framework_dir,
            gold_file_path=expand_raganato_path(raganato_path)[1],
            pred_file_path=f"{tmp_dir}/predictions.gold.key.txt",
        )

        # fine grained eval

        fge_scores = None

        if fine_grained_evals is not None:
            fge_scores = []
            for fge in fine_grained_evals:
                _p, _r, _f1 = framework_evaluate(
                    wsd_framework_dir,
                    gold_file_path=expand_raganato_path(fge)[1],
                    pred_file_path=f"{tmp_dir}/predictions.gold.key.txt",
                )
                fge_scores.append((fge, _p, _r, _f1))

        return p, r, f1, fge_scores


@hydra.main(config_path="../../../conf/test", config_name="raganato")
def main(conf: omegaconf.DictConfig) -> None:

    fix(conf)

    # load module
    # todo decouple ConsecPLModule
    module = ConsecPLModule.load_from_checkpoint(conf.model.model_checkpoint)
    module.to(torch.device(conf.model.device if conf.model.device != -1 else "cpu"))
    module.eval()
    module.freeze()
    module.sense_extractor.evaluation_mode = True  # no loss will be computed even if labels are passed

    # instantiate sense inventory
    sense_inventory = hydra.utils.instantiate(conf.sense_inventory)

    # instantiate predictor
    predictor = hydra.utils.instantiate(conf.predictor)

    # evaluate
    p, r, f1, fge_scores = raganato_evaluate(
        raganato_path=conf.test_raganato_path,
        wsd_framework_dir=conf.wsd_framework_dir,
        module=module,
        predictor=predictor,
        wordnet_sense_inventory=sense_inventory,
        samples_generator=conf.samples_generator,
        prediction_params=conf.model.prediction_params,
        fine_grained_evals=conf.fine_grained_evals,
        reporting_folder=".",  # hydra will handle it
    )
    print(f"# p: {p}")
    print(f"# r: {r}")
    print(f"# f1: {f1}")

    if fge_scores:
        for fge, p, r, f1 in fge_scores:
            print(f'# {fge}: ({p:.1f}, {r:.1f}, {f1:.1f})')


if __name__ == "__main__":
    main()
