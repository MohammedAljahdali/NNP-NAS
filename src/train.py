from typing import List, Optional

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

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

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
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, dataset=config.dataset, _recursive_=False,)

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
                callbacks.append(hydra.utils.instantiate(cb_conf))

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
    # TODO: Find a way to include all experiments runs in one expr
    if config.model.training:
        log.info(f"Instantiating  for training <{config.trainer._target_}>")
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

        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        # Make sure everything closed properly
        log.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

    # Init Lightning trainer for pruning
    if config.model.pruning:
        # Re-init loggers for the prune run
        if config.model.training:
            # Init Lightning loggers
            logger: List[LightningLoggerBase] = []
            if "logger" in config:
                for _, lg_conf in config["logger"].items():
                    if "_target_" in lg_conf:
                        log.info(f"Instantiating logger <{lg_conf._target_}>")
                        if lg_conf._target_ == "pytorch_lightning.loggers.wandb.WandbLogger":
                            lg_conf.job_type = "prune"
                        logger.append(hydra.utils.instantiate(lg_conf))
            log.info(f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}")
            model = utils.import_string(config.model._target_).load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
            model.training = False

        log.info(f"Instantiating for pruning and fine-tuning <{config.trainer._target_}>")
        if config.model.strategy.compression <= 1:
            config.trainer.max_epochs = 0
        trainer: Trainer = hydra.utils.instantiate(
            config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
        )
        if not config.model.training:
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
        log.info("Start pruning!")
        trainer.fit(model=model, datamodule=datamodule)


    # # Evaluate model on test set after training
    # if not config.trainer.get("fast_dev_run"):
    #     log.info("Starting testing!")
    #     trainer.test()

    # Make sure everything closed properly
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
