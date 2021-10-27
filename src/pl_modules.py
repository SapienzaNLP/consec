from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from transformers import get_linear_schedule_with_warmup

from src.sense_extractors import SenseExtractor
from src.utils.optimizers import RAdam


class ConsecPLModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.sense_extractor: SenseExtractor = hydra.utils.instantiate(self.hparams.model.sense_extractor)

        new_embedding_size = self.sense_extractor.model.config.vocab_size + 203
        self.sense_extractor.resize_token_embeddings(new_embedding_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        relative_positions: Optional[torch.Tensor] = None,
        definitions_mask: Optional[torch.Tensor] = None,
        gold_markers: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> dict:

        sense_extractor_output = self.sense_extractor.extract(
            input_ids, attention_mask, token_type_ids, relative_positions, definitions_mask, gold_markers
        )

        output_dict = {
            "pred_logits": sense_extractor_output.prediction_logits,
            "pred_probs": sense_extractor_output.prediction_probs,
            "pred_markers": sense_extractor_output.prediction_markers,
            "loss": sense_extractor_output.loss,
        }

        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output["loss"], on_step=False, on_epoch=True)
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log(f"val_loss", forward_output["loss"], prog_bar=True)

    def get_optimizer_and_scheduler(self):

        no_decay = self.hparams.train.no_decay_params

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.train.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.train.optimizer == "adamw":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, self.hparams.train.learning_rate)
        elif self.hparams.train.optimizer == "radam":
            optimizer = RAdam(optimizer_grouped_parameters, self.hparams.train.learning_rate)
            return optimizer, None
        else:
            raise NotImplementedError

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.train.num_warmup_steps,
            num_training_steps=self.hparams.train.num_training_steps,
        )

        return optimizer, lr_scheduler

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.get_optimizer_and_scheduler()
        if lr_scheduler is None:
            return optimizer
        return [optimizer], [{"interval": "step", "scheduler": lr_scheduler}]
