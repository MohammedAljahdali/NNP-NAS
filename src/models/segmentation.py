import pytorch_lightning as pl
from detectron2.modeling import build_model
from detectron2.solver import WarmupMultiStepLR
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import EventStorage
from detectron2.solver import build_lr_scheduler, build_optimizer

import detectron2
import torch


from hydra.utils import instantiate
from typing import Any


class Segmentation(pl.LightningModule):

    def __init__(
        self, run_id=None, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.cfg = detectron2.config.get_cfg()
        self.cfg.MODEL.BACKBONE.FREEZE_AT = 0
        self.cfg.MODEL.RESNETS.NORM = "BN"
        self.module = build_model(self.cfg)
        self.module.roi_heads.box_predictor.cls_score.is_classifier = True
        self.module.roi_heads.box_predictor.bbox_pred.is_classifier = True

        self.test_counter = 0

        # self.val_evaluator = COCOEvaluator("val")
        # self.test_evaluator = COCOEvaluator("test")
        # self.test_evaluator.reset()

    def on_fit_start(self) -> None:
        self.val_evaluator = COCOEvaluator("val")

    def on_test_start(self) -> None:
        self.test_evaluator = COCOEvaluator("test")
        self.test_evaluator.reset()

    def training_step(self, batch: Any, batch_idx: int):
        with EventStorage() as storage:
            loss_dict = self.module(batch)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
        else:
            losses = sum(loss_dict.values())
        self.log(f"{self.hparams.run_id}/train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": losses}

    def on_validation_epoch_start(self):
        self.val_evaluator.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        if isinstance(batch, dict):
            batch = [batch]
        elif not isinstance(batch, list):
            raise ValueError("Batch should be either a list or dict")
        outputs = self.module(batch)
        self.val_evaluator.process(batch, outputs)


    def validation_epoch_end(self, predictions) -> None:
        results = self.val_evaluator.evaluate()
        for task_name, task_results in results.items():
            for metric, score in task_results.items():
                self.log(f"{self.hparams.run_id}/val/{task_name}_{metric}", score, on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=True)
        # self.log(f"{run_id}/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self.module(batch)
        self.test_evaluator.process(batch, outputs)

    def test_epoch_end(self, predictions) -> None:
        results = self.test_evaluator.evaluate()
        if self.trainer.is_global_zero:
            for task_name, task_results in results.items():
                for metric, score in task_results.items():
                    self.log(f"{self.hparams.run_id}/val/{task_name}_{metric}", score, on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=True)
        return results

    def configure_optimizers(self):
        # optim_cfg = next(iter(self.hparams.optim.values()))
        # optimizer = instantiate(optim_cfg, params=self.module.parameters(), _convert_="partial")
        #
        #
        # scheduler = WarmupMultiStepLR(
        #         optimizer,
        #         self.hparams.sch.n_steps,
        #         self.hparams.sch.GAMMA,
        #         warmup_factor=self.hparams.sch.WARMUP_FACTOR,
        #         warmup_iters=self.hparams.sch.WARMUP_ITERS,
        #         warmup_method=self.hparams.sch.WARMUP_METHOD,
        # )
        optimizer = build_optimizer(self.cfg, self.module)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        sch_dict = {
            # REQUIRED: The scheduler instance
            'scheduler': scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            'interval': 'step',
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            'frequency': 10,
        }
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}