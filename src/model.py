from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification.accuracy import Accuracy
from transformers.models.auto.modeling_auto import \
    AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup

from .loss import SoftTargetCrossEntropy
from .mixup import Mixup

MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit-b16-224-in21k",
        optimizer: str = "sgd",
        lr: float = 3e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        n_classes: int = 10,
        channels_last: bool = False,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        linear_prob: bool = False,
        image_size: int = 224,
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint [vit-b16-224-in21k]
            optimizer: Name of optimizer [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler [cosine, none]
            warmup_steps: Number of warmup epochs
            n_classes: Number of target class. AUTOMATICALLY SET FOR DATASET
            channels_last: Change to channels last memory format for possible training speed up
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix
            label_smoothing: Amount of label smoothing
            linear_prob: Only train the classifier and keep other layers frozen
            image_size: Size of input images
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes
        self.channels_last = channels_last
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.linear_prob = linear_prob
        self.image_size = image_size

        # Initialize network
        try:
            model_path = MODEL_DICT[model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available dataset. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )
        self.net = AutoModelForImageClassification.from_pretrained(
            model_path,
            num_labels=self.n_classes,
            ignore_mismatched_sizes=True,
            image_size=self.image_size,
        )

        # Freeze transformer layers if linear prob
        if self.linear_prob:
            for name, param in self.net.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        # Define metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        self.mixup = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=self.n_classes,
        )

        # Change to channel last memory format
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        if self.channels_last:
            print("Using channel last memory format")
            self = self.to(memory_format=torch.channels_last)

    def forward(self, x):
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        return self.net(pixel_values=x).logits

    def shared_step(self, batch, mode="train"):
        x, y = batch

        if mode == "train":
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()

        # if mode == "train":
        #     from torchvision.utils import save_image

        #     save_image(x[:16], "0.png", normalize=True)
        #     exit()

        # Pass through network
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Get accuracy
        pred = logits.argmax(1)
        acc = getattr(self, f"{mode}_acc")(pred, y.argmax(1))

        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True)
        self.log(f"{mode}_acc", acc, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
