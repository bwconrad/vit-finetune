import os
from functools import partial
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import (CIFAR10, CIFAR100, DTD, STL10, FGVCAircraft,
                                  Flowers102, Food101, ImageFolder,
                                  OxfordIIITPet, StanfordCars)

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
    "dtd": [
        partial(DTD, split="train", download=True),
        partial(DTD, split="val", download=True),
        partial(DTD, split="test", download=True),
        47,
    ],
    "aircraft": [
        partial(FGVCAircraft, split="train", download=True),
        partial(FGVCAircraft, split="val", download=True),
        partial(FGVCAircraft, split="test", download=True),
        100,
    ],
    "cars": [
        partial(StanfordCars, split="train", download=True),
        partial(StanfordCars, split="test", download=True),
        partial(StanfordCars, split="test", download=True),
        196,
    ],
}


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        root: str = "data/",
        n_classes: Optional[int] = None,
        size: int = 224,
        min_scale: float = 0.08,
        batch_size: int = 32,
        workers: int = 4,
        rand_aug_n: int = 0,
        rand_aug_m: int = 9,
        erase_prob: float = 0.0,
        use_trivial_aug: bool = False,
    ):
        """Classification Datamodule

        Args:
            dataset: Name of dataset [custom, cifar10, cifar100, flowers102
                     food101, pets37, stl10, dtd, aircraft, cars]
            root: Download path for built-in datasets or path to dataset directory for custom datasets
            n_classes: Number of classes when using a custom dataset
            size: Image size
            min_scale: Min crop scale
            batch_size: Number of batch samples
            workers: Number of data loader workers
            rand_aug_n: RandAugment number of augmentations
            rand_aug_m: RandAugment magnitude of augmentations
            erase_prob: Probability of applying random erasing
            use_trivial_aug: Apply TrivialAugment instead of RandAugment
        """
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.root = root
        self.size = size
        self.min_scale = min_scale
        self.batch_size = batch_size
        self.workers = workers
        self.rand_aug_n = rand_aug_n
        self.rand_aug_m = rand_aug_m
        self.erase_prob = erase_prob
        self.use_trivial_aug = use_trivial_aug

        # Define dataset
        if self.dataset == "custom":
            # Custom dataset
            assert n_classes is not None
            self.n_classes = n_classes

            self.train_dataset_fn = partial(
                ImageFolder, root=os.path.join(self.root, "train")
            )
            self.val_dataset_fn = partial(
                ImageFolder, root=os.path.join(self.root, "val")
            )
            self.test_dataset_fn = partial(
                ImageFolder, root=os.path.join(self.root, "test")
            )
        else:
            # Built-in dataset
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
                transforms.RandomResizedCrop(
                    (self.size, self.size), scale=(self.min_scale, 1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide()
                if self.use_trivial_aug
                else transforms.RandAugment(self.rand_aug_n, self.rand_aug_m),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=self.erase_prob),
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
        if self.dataset != "custom":
            self.train_dataset_fn(self.root)
            self.val_dataset_fn(self.root)
            self.test_dataset_fn(self.root)

    def setup(self, stage="fit"):
        if self.dataset == "custom":
            if stage == "fit":
                self.train_dataset = self.train_dataset_fn(
                    transform=self.transforms_train
                )
                self.val_dataset = self.val_dataset_fn(transform=self.transforms_test)
            elif stage == "test":
                self.test_dataset = self.test_dataset_fn(transform=self.transforms_test)
        else:
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
            drop_last=True,
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
    dm = DataModule(dataset="custom", root="data/dafre", n_classes=3263)
    dm.setup()
    dl = dm.train_dataloader()
    print(len(dm.train_dataset))
    print(len(dm.val_dataset))

    for x, y in dl:
        print(x.size())
        print(x.min())
        print(x.max())
        print(x.dtype)
        print(y.size())
        print(y.dtype)
        break
