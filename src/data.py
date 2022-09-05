import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "data/",
        size: int = 224,
        batch_size: int = 32,
        workers: int = 4,
    ):
        """CIFAR 10 Classification Datamodule

        Args:
            root: Path to data directory
            size: Image size
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.size = size
        self.batch_size = batch_size
        self.workers = workers

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
        CIFAR10(self.root, train=True, download=True)
        CIFAR10(self.root, train=False, download=True)

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_dataset = CIFAR10(
                self.root, train=True, transform=self.transforms_train
            )
            self.val_dataset = CIFAR10(
                self.root, train=False, transform=self.transforms_test
            )
        elif stage == "test":
            self.test_dataset = CIFAR10(
                self.root, train=False, transform=self.transforms_test
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
