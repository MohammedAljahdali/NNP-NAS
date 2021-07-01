# This file replaces experiment/train.py

from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from hydra.utils import instantiate
import torchvision
from src.models.modules.head import mark_classifier
import torch.nn.functional as F


class ClassificationTraining(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            module,
            optim,
            training,
            pruning,
            run_id=None,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        print(self.hparams)

        self.test_counter = 0

        self.module = instantiate(next(iter(self.hparams.module.values())))

        if hasattr(torchvision.models, next(iter(self.hparams.module.values()))._target_.split('.')[-1]):
            print(f"\n\n\n---------{self.hparams.module._target_.split('.')[-1]}-------\n\n\n")
            # https://pytorch.org/docs/stable/torchvision/models.html
            mark_classifier(self.module)  # add is_classifier attribute
        # Todo: Check if the model is compatible

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.train_accuracy5 = Accuracy(top_k=5)
        self.val_accuracy5 = Accuracy(top_k=5)
        self.test_accuracy5 = Accuracy(top_k=5)

    def forward(self, x: torch.Tensor):
        return self.module(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        logits = F.softmax(logits)
        return loss, logits, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, preds, targets = self.step(batch)

        run_id = self.hparams.run_id + '/train' if self.hparams.run_id else 'train'
        # log train metrics
        acc = self.train_accuracy(logits, targets)
        acc5 = self.train_accuracy5(logits, targets)
        self.log(f"{run_id}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{run_id}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{run_id}/acc5", acc5, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, preds, targets = self.step(batch)

        run_id = self.hparams.run_id + '/val' if self.hparams.run_id else 'val'
        # log val metrics
        acc = self.val_accuracy(logits, targets)
        acc5 = self.val_accuracy5(logits, targets)
        self.log(f"{run_id}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{run_id}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{run_id}/acc5", acc5, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(logits, targets)
        acc5 = self.test_accuracy5(logits, targets)
        state = "before_pruning" if self.test_counter % 2 == 0 else "after_pruning"

        self.log(f"{self.hparams.run_id}/test-{state}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{self.hparams.run_id}/test-{state}/acc", acc, on_step=False, on_epoch=True)
        self.log(f"{self.hparams.run_id}/test-{state}/acc5", acc5, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        next(iter(self.hparams.optim.values()))
        return instantiate(next(iter(self.hparams.optim.values())), params=self.module.parameters(), _convert_="partial")

    def on_fit_end(self) -> None:
        expr = self.logger.experiment[0]
        expr.summary[f"{self.hparams.run_id}/best_score"] = self.trainer.checkpoint_callback.best_model_score
