import os
from logging import getLogger
from typing import Dict, Any

import hydra
import pytorch_lightning as pl
import shutil
from pathlib import Path

from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint

from src.scripts.model.raganato_evaluate import raganato_evaluate
from src.sense_inventories import WordNetSenseInventory


logger = getLogger(__name__)


class ModelCheckpointWithBest(ModelCheckpoint):

    CHECKPOINT_NAME_BEST = "best.ckpt"

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path == "":
            return
        orig_best = Path(self.best_model_path)
        shutil.copyfile(orig_best, orig_best.parent / self.CHECKPOINT_NAME_BEST)


class PredictorsRaganatoEvaluateCallback(pl.Callback):
    def __init__(
        self,
        raganato_path: str,
        wsd_framework_dir: str,
        samples_generator: DictConfig,
        predictors: Dict[str, DictConfig],
        wordnet_sense_inventory: DictConfig,
        prediction_params: Dict[Any, Any],
    ):
        self.raganato_path = raganato_path
        self.wsd_framework_dir = wsd_framework_dir
        self.samples_generator = samples_generator
        self.predictors = {k: hydra.utils.instantiate(v) for k, v in predictors.items()}
        self.wordnet_sense_inventory = hydra.utils.instantiate(wordnet_sense_inventory)
        self.prediction_params = prediction_params

    def on_validation_epoch_start(self, trainer, pl_module):

        logger.info("PredictorsRaganatoEvaluateCallback started")

        pl_module.sense_extractor.evaluation_mode = True

        for predictor_name, predictor in self.predictors.items():

            logger.info(f"Doing {predictor_name}")

            # evaluate and log
            _, _, f1, _ = raganato_evaluate(
                raganato_path=self.raganato_path,
                wsd_framework_dir=self.wsd_framework_dir,
                module=pl_module,
                predictor=predictor,
                wordnet_sense_inventory=self.wordnet_sense_inventory,
                samples_generator=self.samples_generator,
                prediction_params=self.prediction_params,
            )
            pl_module.log(f"{predictor_name}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
            logger.info(f"{predictor_name}: {f1} f1")

        pl_module.sense_extractor.evaluation_mode = False


if __name__ == "__main__":

    import omegaconf
    from src.utils.hydra import fix

    # setup logging
    import logging

    logging.basicConfig(level=logging.INFO)

    @hydra.main(config_path="../conf", config_name="root")
    def main(conf: omegaconf.DictConfig) -> None:

        fix(conf)
        logger.info("log check")

        # load module
        from src.pl_modules import BasecPLModule

        module = BasecPLModule.load_from_checkpoint(
            fix("experiments/basec_deberta_large_directed/2021-04-05/09-05-39/checkpoints/best.ckpt")
        )
        module.to(0)
        module.eval()
        module.freeze()

        for callback in conf.callbacks.callbacks:
            callback = hydra.utils.instantiate(callback, _recursive_=False)
            callback.on_validation_epoch_start(None, module)

    main()
