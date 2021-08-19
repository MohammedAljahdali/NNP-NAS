from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from hydra.utils import instantiate
from torchvision.transforms import transforms

from src.datamodules.datasets import dataset_utils
from src.datamodules.classification_datamodule import ClassificationDataModule

class DartDataModule(ClassificationDataModule):

    def train_dataloader(self):
        return {
            "train": DataLoader(
                dataset=self.trainset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                shuffle=True,
            ),
            "val": DataLoader(
                dataset=self.valset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                shuffle=True,
            )
        }
