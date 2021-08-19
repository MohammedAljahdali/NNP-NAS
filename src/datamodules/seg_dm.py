from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST
from hydra.utils import instantiate, get_original_cwd
from torchvision.transforms import transforms
from torch.utils.data import DistributedSampler, RandomSampler
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
import detectron2
import detectron2.utils.comm as comm
import torch



from src.utils import utils

log = utils.get_logger(__name__)


class SegmentationDataModule(LightningDataModule):
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
        dir: str,
        instance: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dir = dir
        self.instance = instance

        cfg = detectron2.config.get_cfg()
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ()
        cfg.SOLVER.IMS_PER_BATCH = self.batch_size
        cfg.DATALOADER.NUM_WORKERS = self.num_workers

        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None


    @property
    def num_classes(self) -> int:
        if self.dataset.target == "torchvision.datasets.CIFAR10":
            return 10
        elif self.dataset.target == "torchvision.datasets.CIFAR100":
            return 100
        elif self.dataset.target == "torchvision.datasets.ImageFolder":
            return 1000
        elif self.dataset.target == "torchvision.datasets.MNIST":
            return 10
        else:
            raise ValueError()

    @property
    def channels(self) -> int:
        if self.dataset.target == "torchvision.datasets.CIFAR10":
            return 3
        elif self.dataset.target == "torchvision.datasets.CIFAR100":
            return 3
        elif self.dataset.target == "torchvision.datasets.ImageFolder":
            return 3
        elif self.dataset.target == "torchvision.datasets.MNIST":
            return 1
        else:
            raise ValueError()


    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        import subprocess
        # Download coco data set into dir specified by config then /data/coco
        subprocess.call([f"{get_original_cwd()}/bin/fetch_dataset.sh", f"{self.dir}/data/coco", f"{get_original_cwd()}"])
        # subprocess.call([f"bin/fetch_dataset.sh", f"{self.dir}/data/coco"])
        task = "instances" if self.instance else "person_keypoints"
        register_coco_instances("train", {}, f"{self.dir}/data/coco/{task}_train2014.json",
                                f"{self.dir}/data/coco/train2014")
        register_coco_instances("val", {}, f"{self.dir}/data/coco/{task}_minival2014.json",
                                f"{self.dir}/data/coco/val2014")
        register_coco_instances("test", {}, f"{self.dir}/data/coco/{task}_valminusminival2014.json",
                                f"{self.dir}/data/coco/val2014")


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        log.info("Data Setup!")
        self.train_loader = DefaultTrainer.build_train_loader(self.cfg)
        ds = self.train_loader.dataset.dataset
        self.train_loader = DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=trivial_batch_collator
        )
        self.val_loader = detectron2.data.build_detection_test_loader(self.cfg, "val")
        self.test_loader = detectron2.data.build_detection_test_loader(self.cfg, "test")



    def train_dataloader(self):
        # from torch import distributed
        # if distributed.is_initialized():
        #     # Change Sampler only if in ddp
        #     sampler = DistributedSampler(dataset=ds)
        # else:
        #     sampler = RandomSampler(data_source=ds)
        # del ds
        # self.train_loader = detectron2.data.build_detection_train_loader(self.cfg, sampler=sampler)
        # self.train_loader = AspectRatioGroupedDataset(self.train_loader.dataset, self.train_loader.batch_size)
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        if self.test_loader:
            return self.test_loader

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

class AspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

    def __len__(self):
        addone = 1 if len(self.dataset) % self.batch_size != 0 else 0
        return len(self.dataset) // self.batch_size + addone