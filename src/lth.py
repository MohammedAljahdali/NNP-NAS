from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
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

log = utils.get_logger(__name__)


class MyModelPruning(ModelPruning):

    def filter_parameters_to_prune(self, parameters_to_prune):
        return [module for module in parameters_to_prune if self.can_prune(module[0])]

    def can_prune(self, module):
        if hasattr(module, 'is_classifier'):
            return not module.is_classifier
        return True

    def on_fit_end(self, trainer, pl_module):
        print(f'\n\n\n---- On Fit End Here ----- \n\n\n')
        current_epoch = trainer.current_epoch
        prune = self._apply_pruning(current_epoch) if isinstance(self._apply_pruning, Callable) else self._apply_pruning
        amount = self.amount(current_epoch) if isinstance(self.amount, Callable) else self.amount
        if not prune or not amount:
            return
        self.apply_pruning(amount)

        if (
            self._use_lottery_ticket_hypothesis(current_epoch)
            if isinstance(self._use_lottery_ticket_hypothesis, Callable) else self._use_lottery_ticket_hypothesis
        ):
            self.apply_lottery_ticket_hypothesis()


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
        apply_pruning=pruning_callable, use_lottery_ticket_hypothesis=pruning_callable,
        pruning_fn='l1_unstructured', use_global_unstructured=True, verbose=1
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
