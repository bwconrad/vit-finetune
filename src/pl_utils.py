from argparse import Namespace
from typing import Any, Optional

from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger


class MyLightningArgumentParser(LightningArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_logger_args()

    def add_logger_args(self) -> None:
        # Shared args
        self.add_argument(
            "--logger_type",
            type=str,
            help="Name of logger",
            default="csv",
            choices=["csv", "wandb", "tensorboard"],
        )
        self.add_argument(
            "--save_path",
            type=str,
            help="Save path of outputs",
            default="output/",
        )
        self.add_argument(
            "--name", type=str, help="Name of experiment", default="default"
        )

        # Wandb args
        self.add_argument(
            "--project", type=str, help="Name of wandb project", default="default"
        )


def init_logger(args: Namespace) -> Optional[Logger]:
    """Initialize logger from arguments

    Args:
        args: parsed argument namespace

    Returns:
        logger: initialized logger object
    """
    if args.logger_type == "wandb":
        return WandbLogger(
            project=args.project,
            name=args.name,
            save_dir=args.save_path,
        )
    elif args.logger_type == "tensorboard":
        return TensorBoardLogger(name=args.name, save_dir=args.save_path)
    elif args.logger_type == "csv":
        return CSVLogger(name=args.name, save_dir=args.save_path)
    else:
        ValueError(
            f"{args.logger_type} is not an available logger. Should be one of ['cvs', 'wandb', 'tensorboard']"
        )
