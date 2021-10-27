import omegaconf
import hydra

from comet_ml import Experiment  # must be here for comet documentation
from pytorch_lightning.loggers import CometLogger

import pytorch_lightning as pl

from src.pl_data_modules import ConsecDataModule
from src.pl_modules import ConsecPLModule
from src.utils.hydra import fix


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = ConsecDataModule(conf)

    # main module declaration
    pl_module = ConsecPLModule(conf)

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            filename="{epoch:02d}-{" + conf.train.callbacks_monitor + ":.2f}",
        )
        callbacks_store.append(model_checkpoint)

    for callback in conf.callbacks.callbacks:
        callbacks_store.append(hydra.utils.instantiate(callback, _recursive_=False))

    # trainer
    trainer = hydra.utils.instantiate(
        conf.train.pl_trainer,
        callbacks=callbacks_store,
        logger=False,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../../../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    fix(conf)
    train(conf)


if __name__ == "__main__":
    main()
