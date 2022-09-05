import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import DataModule
from src.model import ClassifcationModel
from src.pl_utils import MyLightningArgumentParser, init_logger

model_class = ClassifcationModel
dm_class = DataModule

# Parse arguments
parser = MyLightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data")
parser.add_lightning_class_args(model_class, "model")
parser.add_argument(
    "--test_at_end", action="store_true", help="Evaluate on test set after training"
)
args = parser.parse_args()

# Setup trainer
logger = init_logger(args)
checkpoint_callback = ModelCheckpoint(
    filename="best-{epoch}-{val_acc:.4f}",
    monitor="val_acc",
    mode="max",
    save_last=True,
)
dm = dm_class(**args["data"])
args["model"]["n_classes"] = dm.n_classes
model = model_class(**args["model"])
trainer = pl.Trainer.from_argparse_args(
    args, logger=logger, callbacks=[checkpoint_callback]
)

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)

# Test
if args["test_at_end"]:
    trainer.test()
