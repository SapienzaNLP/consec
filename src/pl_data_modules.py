from typing import Union, List, Optional

from omegaconf import DictConfig

import hydra

from torch.utils.data import DataLoader
import pytorch_lightning as pl


class ConsecDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":

            # SENSE INVENTORY
            train_sense_inventory = hydra.utils.instantiate(self.conf.data.train_sense_inventory)
            dev_sense_inventory = (
                hydra.utils.instantiate(self.conf.data.dev_sense_inventory)
                if self.conf.data.dev_sense_inventory is not None
                else train_sense_inventory
            )

            # TOKENIZER
            consec_tokenizer = hydra.utils.instantiate(self.conf.tokenizer.consec_tokenizer)

            # DEPENDENCY FINDER
            dependency_finder = hydra.utils.instantiate(self.conf.data.dependency_finder)

            # train dataset
            self.train_dataset = hydra.utils.instantiate(
                self.conf.data.train_dataset,
                sense_inventory=train_sense_inventory,
                tokenizer=consec_tokenizer,
                dependency_finder=dependency_finder,
                max_length=consec_tokenizer.model_max_length,
            )

            # validation dataset
            self.validation_dataset = hydra.utils.instantiate(
                self.conf.data.validation_dataset,
                sense_inventory=dev_sense_inventory,
                tokenizer=consec_tokenizer,
                dependency_finder=dependency_finder,
                max_length=consec_tokenizer.model_max_length,
            )

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.conf.data.num_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.validation_dataset, batch_size=None, num_workers=self.conf.data.num_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=None, num_workers=self.conf.data.num_workers)
