# This file replaces experiment/train.py

from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from hydra.utils import instantiate
import torchvision
from src.models.modules.head import mark_classifier
import torch.nn.functional as F
from .classification_training import ClassificationTraining
from hydra.utils import instantiate


class ClassificationPruning(ClassificationTraining):
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
            strategy,
            **kwargs
    ):
        super().__init__(module=module, optim=optim, strategy=strategy)


    def on_fit_start(self):
        x, y = next(iter(self.trainer.datamodule.train_dataloader()))
        self.pruning = instantiate(self.hparams.strategy, model=self.module, inputs=x, outputs=y)
        self.pruning.apply()
        # TODO log pruning metrics

