from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import (CIFAR10, CIFAR100, PCAM, Flowers102, Food101,
                                  OxfordIIITPet)

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
    "pcam": [
        partial(PCAM, split="train", download=True),
        partial(PCAM, split="val", download=True),
        partial(PCAM, split="test", download=True),
        2,
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
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset
            root: Path to data directory
            size: Image size
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root
        self.size = size
        self.batch_size = batch_size
        self.workers = workers

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
                f"{dataset} is not an available dataset. Should be one of [...]"
            )

        self.transforms_train = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.size, self.size)),
                transforms.RandomHorizontalFlip(),
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


if __name__ == "__main__":
    dm = CIFAR10DataModule()
    dm.setup()
    dl = dm.train_dataloader()

    for x, y in dl:
        print(x.size())
        print(x.min())
        print(x.max())
        print(x.dtype)
        print(y.size())
        print(y.dtype)
        break
