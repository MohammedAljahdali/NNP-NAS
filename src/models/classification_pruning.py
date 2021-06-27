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
from src.metrics import model_size, flops


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
        if self.training:
            return
        x, y = next(iter(self.trainer.datamodule.train_dataloader()))
        self.pruning = instantiate(self.hparams.strategy, model=self.module, inputs=x, outputs=y)
        self.pruning.apply()
        # This assumes that the logger is wandb
        # TODO: Make it general to support multiple loggers
        expr = self.logger.experiment[0]
        print(expr)
        metrics = {}
        # Model Size
        size, size_nz = model_size(self.module)
        expr.summary['prune/size'] = size
        expr.summary['prune/size_nz'] = size_nz
        expr.summary['prune/compression_ratio'] = size / size_nz

        # FLOPS
        ops, ops_nz = flops(self.module, x)
        expr.summary['prune/flops'] = ops
        expr.summary['prune/flops_nz'] = ops_nz
        expr.summary['prune/theoretical_speedup'] = ops / ops_nz
        results = self.trainer.validate(model=self,datamodule=self.trainer.datamodule)
        print(f"\n\n\n------\n{results}\n-------\n\n\n")
        for k, v in results[0].items():
            expr.summary['prune/'+k.split('/')[1]] = v
