from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger

from jsonargparse import lazy_instance
from src.data import DataModule
from src.model import ClassificationModel

model_class = ClassificationModel
dm_class = DataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir="output", name="default"
                )
            }
        )
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
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
        parser.add_argument(
            "--test_at_end",
            action="store_true",
            help="Evaluate on test set after training",
        )


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = MyLightningCLI(
    ClassificationModel,
    DataModule,
    save_config_kwargs={"overwrite": True},
    trainer_defaults={"check_val_every_n_epoch": None},
    run=False,
)

# Train
cli.trainer.fit(cli.model, cli.datamodule)

# Test
if cli.config.test_at_end:
    cli.trainer.test(cli.model, cli.datamodule)
