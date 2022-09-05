import pytorch_lightning as pl
import torch
from ffcv.fields.decoders import (CenterCropRGBImageDecoder, IntDecoder,
                                  RandomResizedCropRGBImageDecoder)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (Convert, RandomHorizontalFlip, ToTensor,
                             ToTorchImage)
from ffcv.transforms.common import Squeeze
from torchvision import transforms


class FFCVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: str,
        size: int = 224,
        batch_size: int = 32,
        workers: int = 4,
        os_cache: bool = False,
    ):
        """FFCV Classification Datamodule

        Args:
            train_path: Path to training dataset beton file
            test_path: Path to test dataset beton file
            size: Image size
            batch_size: Number of batch samples
            workers: Number of data loader workers
            os_cache: Whether to cache data to memory
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_path = train_path
        self.test_path = test_path
        self.size = size
        self.batch_size = batch_size
        self.workers = workers
        self.os_cache = os_cache

        mean = [128, 128, 128]
        std = [128, 128, 128]

        self.transforms_label = [IntDecoder(), ToTensor(), Squeeze()]
        self.transforms_train = [
            RandomResizedCropRGBImageDecoder((self.size, self.size)),
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
            os_cache=self.os_cache,
        )

    def val_dataloader(self):
        return Loader(
            self.test_path,
            batch_size=self.batch_size,
            num_workers=self.workers,
            order=OrderOption.RANDOM,
            pipelines={"image": self.transforms_test, "label": self.transforms_label},
            drop_last=False,
            os_cache=self.os_cache,
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
    dm = FFCVDataModule(
        train_path="data/cifar_train.beton",
        test_path="data/cifar_test.beton",
        batch_size=4,
    )
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
