from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
import pytorch_lightning
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelPruning, EarlyStopping, ModelCheckpoint
import functools
from src.utils import utils
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch import nn
from src.metrics import model_size, flops
_PARAM_TUPLE = Tuple[nn.Module, str]
_PARAM_LIST = Union[List[_PARAM_TUPLE], Tuple[_PARAM_TUPLE]]

log = utils.get_logger(__name__)


class MyModelPruning(ModelPruning):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.level = ''
        self.logger: Optional[LightningLoggerBase] = None
        self.module: Optional[nn.Module] = None
        self.x: Optional[torch.T] = None

    def filter_parameters_to_prune(self, parameters_to_prune: Optional[_PARAM_LIST] = None):
        log.debug(parameters_to_prune)
        return [module for module in parameters_to_prune if self.can_prune(module[0])]

    def can_prune(self, module):
        if hasattr(module, 'is_classifier'):
            return not module.is_classifier
        return True

    def on_train_epoch_end(self, trainer, pl_module: LightningModule):
        pass

    def on_fit_end(self, trainer, pl_module):

        current_epoch = trainer.current_epoch


        prune = self._apply_pruning(current_epoch) if isinstance(self._apply_pruning, Callable) else self._apply_pruning
        amount = self.amount(current_epoch) if isinstance(self.amount, Callable) else self.amount
        if not prune or not amount:
            return
        self.level = pl_module.hparams.run_id.split('-')[-1]
        log.info(f'\n\n\n---- on fit end ----- \n\n\n')
        # amount = 1
        log.info(f"Pruning Level f{self.level}")
        # for _ in range(int(self.level)+1):
        #     amount *= 0.5

        # TODO: Make it general to support multiple loggers
        self.logger = pl_module.logger
        self.module = pl_module.module
        self.x, y = next(iter(trainer.datamodule.train_dataloader()))
        log.debug("BEFORE")
        log.debug([self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune])
        log.debug("END OF BEFORE")
        self.apply_pruning(amount)
        log.debug("AFTER")
        log.debug([self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune])
        log.debug("END OF AFTER")

        if (
                self._use_lottery_ticket_hypothesis(current_epoch)
                if isinstance(self._use_lottery_ticket_hypothesis, Callable) else self._use_lottery_ticket_hypothesis
        ):
            log.info("Rested the Parameters")
            self.apply_lottery_ticket_hypothesis()

    @rank_zero_only
    def _log_sparsity_stats(
            self, prev: List[Tuple[int, int]], curr: List[Tuple[int, int]], amount: Union[int, float] = 0
    ):
        total_params = sum(p.numel() for layer, _ in self._parameters_to_prune for p in layer.parameters())
        prev_total_zeros = sum(zeros for zeros, _ in prev)
        curr_total_zeros = sum(zeros for zeros, _ in curr)
        log.info(
            f"Applied `{self._pruning_fn_name}`. Pruned:"
            f" {prev_total_zeros}/{total_params} ({prev_total_zeros / total_params:.2%}) ->"
            f" {curr_total_zeros}/{total_params} ({curr_total_zeros / total_params:.2%})"
        )
        self.logger.experiment[0].summary[f"level-{self.level}/prev_size"] = total_params - prev_total_zeros
        self.logger.experiment[0].summary[f"level-{self.level}/size"] = total_params - curr_total_zeros
        self.logger.experiment[0].summary[f"level-{self.level}/pruned"] = curr_total_zeros / total_params

        self.logger.experiment[0].summary[f"level-{self.level}/compression_ratio"] = total_params / (total_params - curr_total_zeros)

        # FLOPS
        ops, ops_nz = flops(self.module, self.x)
        self.logger.experiment[0].summary[f"level-{self.level}/flops"] = ops
        self.logger.experiment[0].summary[f"level-{self.level}/flops_nz"] = ops_nz
        self.logger.experiment[0].summary[f"level-{self.level}/theoretical_speedup"] = ops / ops_nz

        if self._verbose == 2:
            for i, (module, name) in enumerate(self._parameters_to_prune):
                prev_mask_zeros, prev_mask_size = prev[i]
                curr_mask_zeros, curr_mask_size = curr[i]
                log.info(
                    f"Applied `{self._pruning_fn_name}` to `{module!r}.{name}` with amount={amount}. Pruned:"
                    f" {prev_mask_zeros} ({prev_mask_zeros / prev_mask_size:.2%}) ->"
                    f" {curr_mask_zeros} ({curr_mask_zeros / curr_mask_size:.2%})"
                )


def lth(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    To perform iterative pruning
    The process is as follows:
        1. Train a network
        2. Prune it
        3. Reset the weights & Re-train
        4. Repeat 2 & 3
    In pytorch lightning terms:
        1. Initialize a model and a trainer
        2. Prune when early stopping decides the stoping epoch.
        3. Apply the lth reset function
        4. reinitialize the trainer, early stopping, and checkpoint callback, and increase iter_n in the model
            then start a new trainer.fit
        5. Repeat steps 2, 3 & 4 until a certain compression rate or something idk

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, dataset=config.dataset,
                                                              _recursive_=False, )

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}> with <{config.optim._target_}> optimizer")
    model: LightningModule = hydra.utils.instantiate(
        config.model, optim=config.optim, module=config.module, _recursive_=False,
    )
    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                if cb_conf._target_ == "pytorch_lightning.callbacks.EarlyStopping":
                    early_stopping_callback = hydra.utils.instantiate(cb_conf)
                    callbacks.append(early_stopping_callback)
                else:
                    callbacks.append(hydra.utils.instantiate(cb_conf))
    # Change monitor value
    model.hparams.run_id = "level-0"
    # Update the monitored value name
    for callback in callbacks:
        if isinstance(callback, EarlyStopping) or isinstance(callback, ModelCheckpoint):
            callback.monitor = f"{model.hparams.run_id}/{callback.monitor}"
        if isinstance(callback, ModelCheckpoint):
            callback.dirpath = callback.dirpath + f'/{model.hparams.run_id}/'

    # Init the pruning callback
    def stop_training_check(current_epoch, early_stopping_callback: EarlyStopping, max_epochs: int):
        print(early_stopping_callback)
        print(f'\n\n\n----{early_stopping_callback.stopped_epoch} ---- {current_epoch} ----- \n\n\n')
        return (current_epoch == early_stopping_callback.stopped_epoch and current_epoch != 0) \
               or current_epoch == max_epochs - 1

    # This code assumes you will always have early stopping callback
    pruning_callable = functools.partial(
        stop_training_check,
        early_stopping_callback=early_stopping_callback,
        max_epochs=config.trainer.max_epochs
    )

    pruning_callback = MyModelPruning(
        apply_pruning=True, use_lottery_ticket_hypothesis=True,
        pruning_fn='l1_unstructured', use_global_unstructured=True, verbose=1, make_pruning_permanent=False,
        amount=config.lth.amount
    )
    callbacks.append(pruning_callback)

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                if lg_conf._target_ == "pytorch_lightning.loggers.wandb.WandbLogger":
                    lg_conf.job_type = "prune" if not config.model.training else "train"
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer for training
    log.info(f"Instantiating  for training level 0   <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.debug("MODEL PARAMETERS LEVEL 0")
    log.debug(list(model.module.parameters())[5:7])

    log.info("Starting training level 0!")
    trainer.fit(model=model, datamodule=datamodule)
    print(f'\n\n\n----{early_stopping_callback.stopped_epoch}----- \n\n\n')
    # Now do the loop
    # TODO: modify the checkpoint callback save dir, to save models of different iterations on different folders
    for i in range(1, config.lth.N):
        # Init Lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
        callbacks.append(pruning_callback)
        # Change monitor value
        model.hparams.run_id = f"level-{i}"
        # Update the monitored value name
        for callback in callbacks:
            if isinstance(callback, EarlyStopping) or isinstance(callback, ModelCheckpoint):
                callback.monitor = f"{model.hparams.run_id}/{callback.monitor}"
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = callback.dirpath + f'/{model.hparams.run_id}/'

        # Init Lightning trainer for training
        log.info(f"Instantiating  for training level {i} <{config.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )

        log.debug(f"MODEL PARAMETERS LEVEL {i}")
        log.debug(list(model.module.parameters())[5:7])

        log.info(f"Starting training level {i}!")
        trainer.fit(model=model, datamodule=datamodule)

    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
