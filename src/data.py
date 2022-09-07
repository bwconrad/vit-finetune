import os
import shutil
from functools import partial
from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import (CIFAR10, CIFAR100, STL10, Flowers102,
                                  Food101, OxfordIIITPet)

DATASET_DICT = {
    "cifar10": [
        partial(CIFAR10, train=True, download=True),
        partial(CIFAR10, train=False, download=True),
        partial(CIFAR10, train=False, download=True),
        10,
    ],
    "cifar100": [
        partial(CIFAR100, train=True, download=True),
        partial(CIFAR100, train=False, download=True),
        partial(CIFAR100, train=False, download=True),
        100,
    ],
    "flowers102": [
        partial(Flowers102, split="train", download=True),
        partial(Flowers102, split="val", download=True),
        partial(Flowers102, split="test", download=True),
        102,
    ],
    "food101": [
        partial(Food101, split="train", download=True),
        partial(Food101, split="test", download=True),
        partial(Food101, split="test", download=True),
        101,
    ],
    "pets37": [
        partial(OxfordIIITPet, split="trainval", download=True),
        partial(OxfordIIITPet, split="test", download=True),
        partial(OxfordIIITPet, split="test", download=True),
        37,
    ],
    "stl10": [
        partial(STL10, split="train", download=True),
        partial(STL10, split="test", download=True),
        partial(STL10, split="test", download=True),
        10,
    ],
}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        root: str = "data/",
        size: int = 224,
        batch_size: int = 32,
        workers: int = 4,
        randaug_n: int = 0,
        randaug_m: int = 9,
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset
            root: Path to data directory
            size: Image size
            batch_size: Number of batch samples
            workers: Number of data loader workers
            randaug_n: RandAugment number of augmentations
            randaug_m: RandAugment magnitude of augmentations
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root
        self.size = size
        self.batch_size = batch_size
        self.workers = workers
        self.randaug_n = randaug_n
        self.randaug_m = randaug_m

        try:
            (
                self.train_dataset_fn,
                self.val_dataset_fn,
                self.test_dataset_fn,
                self.n_classes,
            ) = DATASET_DICT[self.dataset]
            print(f"Using the {self.dataset} dataset")
        except:
            raise ValueError(
                f"{dataset} is not an available dataset. Should be one of {[k for k in DATASET_DICT.keys()]}"
            )

        self.transforms_train = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.size, self.size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(self.randaug_n, self.randaug_m),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transforms_test = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def prepare_data(self):
        self.train_dataset_fn(self.root)
        self.val_dataset_fn(self.root)
        self.test_dataset_fn(self.root)

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = self.train_dataset_fn(
                self.root, transform=self.transforms_train, download=False
            )
            self.val_dataset = self.val_dataset_fn(
                self.root, transform=self.transforms_test, download=False
            )
        elif stage == "test":
            self.test_dataset = self.test_dataset_fn(
                self.root, transform=self.transforms_test, download=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )


class DAFRe(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__()

        self.split = split
        self.root = root
        self.transform = transform

        if download:
            self.download()

        self.process()

    def download(self) -> None:
        # if self._check_integrity():
        #     print("Files already downloaded and verified")
        #     return
        # download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        return

    def process(self) -> None:
        # Create directories
        split_dir = os.path.join(self.root, "dafre", self.split)
        for i in range(3263):
            os.makedirs(os.path.join(split_dir, str(i)), exist_ok=True)

        # Load image labels
        anns = pd.read_csv(
            os.path.join(
                self.root,
                "dafre",
                "labels",
                f"{self.split}.csv",
            ),
            names=["label", "path"],
        )

        # Move images to correct directories
        image_dir = os.path.join(self.root, "dafre", "faces")
        for label, path in zip(anns.label.to_list(), anns.path.to_list()):
            shutil.copy(
                os.path.join(image_dir, path), os.path.join(split_dir, str(label))
            )


if __name__ == "__main__":
    # dm = DataModule()
    # dm.setup()
    # dl = dm.train_dataloader()

    # for x, y in dl:
    #     print(x.size())
    #     print(x.min())
    #     print(x.max())
    #     print(x.dtype)
    #     print(y.size())
    #     print(y.dtype)
    #     break
    d = DAFRe("data/", download=True)
