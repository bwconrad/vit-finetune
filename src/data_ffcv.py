import pytorch_lightning as pl
import torch
from ffcv.fields.decoders import (CenterCropRGBImageDecoder, IntDecoder,
                                  SimpleRGBImageDecoder)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, RandomHorizontalFlip, RandomTranslate,
                             ToTensor, ToTorchImage)
from ffcv.transforms.common import Squeeze
from torchvision import transforms


class CIFAR10DataModuleFFCV(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        size: int = 32,
        padding: int = 4,
        batch_size: int = 128,
        workers: int = 4,
    ):
        """CIFAR 10 Classification Datamodule using FFCV

        Args:
            train_path: Path to training dataset beton file
            test_path: Path to test dataset beton file
            size: Image size
            padding: Amount of padding
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_path = train_path
        self.test_path = test_path
        self.size = size
        self.padding = padding
        self.batch_size = batch_size
        self.workers = workers

        mean = [125.307, 122.961, 113.8575]
        std = [51.5865, 50.847, 51.255]

        self.transforms_label = [IntDecoder(), ToTensor(), Squeeze()]
        self.transforms_train = [
            SimpleRGBImageDecoder(),
            RandomTranslate(padding=padding),
            RandomHorizontalFlip(),
            ToTensor(),
            ToTorchImage(channels_last=False, convert_back_int16=False),
            Convert(torch.float32),
            transforms.Normalize(mean, std),
        ]
        self.transforms_test = [
            CenterCropRGBImageDecoder((self.size, self.size), ratio=1),
            ToTensor(),
            ToTorchImage(channels_last=False, convert_back_int16=False),
            Convert(torch.float32),
            transforms.Normalize(mean, std),
        ]

    def train_dataloader(self):
        return Loader(
            self.train_path,
            batch_size=self.batch_size,
            num_workers=self.workers,
            order=OrderOption.RANDOM,
            pipelines={"image": self.transforms_train, "label": self.transforms_label},
            drop_last=True,
        )

    def val_dataloader(self):
        return Loader(
            self.test_path,
            batch_size=self.batch_size,
            num_workers=self.workers,
            order=OrderOption.RANDOM,
            pipelines={"image": self.transforms_test, "label": self.transforms_label},
            drop_last=False,
        )

    def test_dataloader(self):
        return Loader(
            self.test_path,
            batch_size=self.batch_size,
            num_workers=self.workers,
            order=OrderOption.RANDOM,
            pipelines={"image": self.transforms_test, "label": self.transforms_label},
            drop_last=False,
        )


if __name__ == "__main__":
    dm = CIFAR10DataModuleFFCV(
        train_path="data/cifar_train.beton", test_path="data/cifar_test.beton"
    )
    dm.setup()
    dl = dm.val_dataloader()

    for x, y in dl:
        print(x.size())
        print(x.min())
        print(x.max())
        print(x.dtype)
        print(y.size())
        print(y.dtype)
        break
