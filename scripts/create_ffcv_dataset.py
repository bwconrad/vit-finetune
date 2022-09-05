import os
from typing import Optional

import ffcv
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def write_ffcv_dataset(
    dataset: Dataset,
    write_path: str,
    max_resolution: Optional[int] = None,
    num_workers: int = 4,
    write_mode: str = "raw",
    compress_probability: float = 0.50,
    jpeg_quality: float = 90,
    chunk_size: int = 100,
):
    """Converts PyTorch compatible ``dataset`` into FFCV format at filepath ``write_path``.
    Args:
        dataset (Iterable[Sample]): A PyTorch dataset. Default: ``None``.
        write_path (str): Write results to this file. Default: ``"/tmp/dataset.ffcv"``.
        max_resolution (int): Limit resolution if provided. Default: ``None``.
        num_workers (int): Numbers of workers to use. Default: ``16``.
        write_mode (str): Write mode for the dataset. Default: ``'raw'``.
        compress_probability (float): Probability with which image is JPEG-compressed. Default: ``0.5``.
        jpeg_quality (float): Quality to use for jpeg compression. Default: ``90``.
        chunk_size (int): Size of chunks processed by each worker during conversion. Default: ``100``.
    """

    writer = ffcv.writer.DatasetWriter(
        write_path,
        {
            "image": ffcv.fields.RGBImageField(
                write_mode=write_mode,
                max_resolution=max_resolution,
                compress_probability=compress_probability,
                jpeg_quality=jpeg_quality,
            ),
            "label": ffcv.fields.IntField(),
        },
        num_workers=num_workers,
    )
    writer.from_indexed_dataset(dataset, chunksize=chunk_size)


# Train dataset
ds = CIFAR10(root="data/", train=True, download=True)
write_ffcv_dataset(dataset=ds, write_path=os.path.join("data", "cifar_train.beton"))

# validation dataset
ds = CIFAR10(root="data/", train=False, download=True)
write_ffcv_dataset(dataset=ds, write_path=os.path.join("data/" + "/cifar_val.beton"))
