from pytorch_lightning.callbacks import ModelCheckpoint
import os
import torch
import torch.backends.cuda
import torch.backends.cudnn
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger

from jsonargparse import lazy_instance
from src.data import DataModule
from src.model import ClassificationModel

model_class = ClassificationModel
dm_class = DataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir="output", name="default"
                ),
                "model_checkpoint.monitor": "val_acc",
                "model_checkpoint.mode": "max",
                "model_checkpoint.filename": "best-step-{step}-{val_acc:.4f}",
                "model_checkpoint.save_last": True,
            }
        )
        parser.link_arguments("data.size", "model.image_size")
        parser.link_arguments(
            "data.num_classes", "model.n_classes", apply_on="instantiate"
        )


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = MyLightningCLI(
    ClassificationModel,
    DataModule,
    save_config_kwargs={"overwrite": True},
    trainer_defaults={"check_val_every_n_epoch": None},
)


# Copy the config into the experiment directory
# Fix for https://github.com/Lightning-AI/lightning/issues/17168
os.rename(
    os.path.join(cli.trainer.logger.save_dir, "config.yaml"),  # type:ignore
    os.path.join(cli.trainer.logger.experiment.log_dir, "config.yaml"),  # type:ignore
)
