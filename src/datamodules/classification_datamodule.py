from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from hydra.utils import instantiate
from torchvision.transforms import transforms

from src.datamodules.datasets import dataset_utils

from src.utils import utils

log = utils.get_logger(__name__)


class ClassificationDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.dataset = kwargs['dataset']
        print(self.dataset)

        self.trainset: Optional[Dataset] = None
        self.valset: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        log.info("Preparing the Data!")
        if "download" in self.dataset:
            instantiate(self.dataset.train_dataset)
            instantiate(self.dataset.val_dataset)

            self.dataset.train_dataset.download = False
            self.dataset.val_dataset.download = False


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        log.info("Data Setup!")
        self.trainset = instantiate(self.dataset.train_dataset)
        self.valset = instantiate(self.dataset.val_dataset)
        # dataset = ConcatDataset(datasets=[trainset, testset])
        # self.data_train, self.data_val, self.data_test = random_split(
        #     dataset, self.train_val_test_split
        # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         shuffle=False,
    #     )