from argparse import Namespace
from typing import Any, Callable, List, Optional, Type, Union

from jsonargparse import ActionConfigFile, class_from_function
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MyLightningArgumentParser(LightningArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_logger_args()
        self.add_argument("--config", action=ActionConfigFile)

    def add_logger_args(self) -> None:
        # Shared args
        self.add_argument(
            "--logger_type",
            type=str,
            help="Name of logger",
            default="csv",
            choices=["csv", "wandb"],
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

    def add_lightning_class_args(
        self,
        lightning_class: Union[
            Callable[
                ..., Union[Trainer, LightningModule, LightningDataModule, Callback]
            ],
            Type[Trainer],
            Type[LightningModule],
            Type[LightningDataModule],
            Type[Callback],
        ],
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
        skip: List = [],
    ) -> List[str]:
        """Adds arguments from a lightning class to a nested key of the parser.

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.
            skip: Names of arguments that should be skipped.

        Returns:
            A list with the names of the class arguments added.
        """
        if callable(lightning_class) and not isinstance(lightning_class, type):
            lightning_class = class_from_function(lightning_class)

        if isinstance(lightning_class, type) and issubclass(
            lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)
        ):
            if issubclass(lightning_class, Callback):
                self.callback_keys.append(nested_key)
            if subclass_mode:
                return self.add_subclass_arguments(  # type: ignore
                    lightning_class, nested_key, fail_untyped=False, required=required
                )
            return self.add_class_arguments(
                lightning_class,
                nested_key,
                fail_untyped=False,
                instantiate=not issubclass(lightning_class, Trainer),
                sub_configs=True,
                skip=set(skip),
            )
        raise MisconfigurationException(
            f"Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
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
    elif args.logger_type == "csv":
        return CSVLogger(name=args.name, save_dir=args.save_path)
    else:
        ValueError(
            f"{args.logger_type} is not an available logger. Should be one of ['cvs', 'wandb']"
        )
