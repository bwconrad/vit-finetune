from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from transformers.models.auto.modeling_auto import \
    AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup

from .loss import SoftTargetCrossEntropy
from .mixup import Mixup

MODEL_DICT = {
    "hf-vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "hf-vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "hf-vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "hf-vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "hf-vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "hf-vit-b16-224": "google/vit-base-patch16-224",
    "hf-vit-l16-224": "google/vit-large-patch16-224",
    "hf-vit-b16-384": "google/vit-base-patch16-384",
    "hf-vit-b32-384": "google/vit-base-patch32-384",
    "hf-vit-l16-384": "google/vit-large-patch16-384",
    "hf-vit-l32-384": "google/vit-large-patch32-384",
    "hf-vit-b16-224-dino": "facebook/dino-vitb16",
    "hf-vit-b8-224-dino": "facebook/dino-vitb8",
    "hf-vit-s16-224-dino": "facebook/dino-vits16",
    "hf-vit-s8-224-dino": "facebook/dino-vits8",
    "hf-beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "hf-beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
    "timm-beitv2-b16-224-in21k": "beitv2_base_patch16_224_in22k",
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "hf-vit-b16-224-in21k",
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
        linear_probe: bool = False,
        image_size: int = 224,
        weights: Optional[str] = None,
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
            n_classes: Number of target class. AUTOMATICALLY SET BY DATAMODULE
            channels_last: Change to channels last memory format for possible training speed up
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix
            label_smoothing: Amount of label smoothing
            linear_probe: Only train the classifier and keep other layers frozen
            image_size: Size of input images
            weights: Path of checkpoint to load weights from. E.g when resuming after linear probing
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
        self.linear_probe = linear_probe
        self.image_size = image_size
        self.weights = weights

        # Initialize network
        try:
            model_path = MODEL_DICT[self.model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available dataset. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )
        # Huggingface model
        if self.model_name.startswith("hf-"):
            self.model_type = "hf"
            self.net = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=self.n_classes,
                ignore_mismatched_sizes=True,
                image_size=self.image_size,
            )
        # Timm model
        else:
            assert self.image_size == 224
            self.model_type = "timm"
            self.net = timm.create_model(
                model_path,
                pretrained=True,
                num_classes=self.n_classes,
            )

        # Load checkpoint weights
        if self.weights:
            print(f"Loaded weights from {self.weights}")
            ckpt = torch.load(self.weights)["state_dict"]

            # Remove prefix from key names
            new_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("net"):
                    k = k.replace("net" + ".", "")
                    new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)

        # Freeze transformer layers if linear probing
        if self.linear_probe:
            for name, param in self.net.named_parameters():
                if "classifier" not in name and "head" not in name:
                    param.requires_grad = False

        # Define metrics
        self.train_metrics = MetricCollection(
            {"acc": Accuracy(top_k=1), "acc_top5": Accuracy(top_k=5)}
        )
        self.val_metrics = MetricCollection(
            {"acc": Accuracy(top_k=1), "acc_top5": Accuracy(top_k=5)}
        )
        self.test_metrics = MetricCollection(
            {
                "acc": Accuracy(top_k=1),
                "acc_top5": Accuracy(top_k=5),
                "stats": StatScores(reduce="macro", num_classes=self.n_classes),
            }
        )

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define regularizers
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

        if self.model_type == "hf":
            return self.net(pixel_values=x).logits
        else:
            return self.net(x)

    def shared_step(self, batch, mode="train"):
        x, y = batch

        if mode == "train":
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()

        # if mode == "train":
        #     from torchvision.utils import save_image

        #     save_image(x[:16], "things/0.png", normalize=True)
        #     exit()

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        metrics = getattr(self, f"{mode}_metrics")(pred, y.argmax(1))

        # Log
        self.log(f"{mode}_loss", loss, on_epoch=True)
        for k, v in metrics.items():
            if len(v.size()) == 0:
                self.log(f"{mode}_{k.lower()}", v, on_epoch=True)

        if mode == "test":
            return metrics["stats"]
        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs: List[torch.Tensor]):
        """Save per-class accuracies to csv"""
        # Aggregate all batch stats
        combined_stats = torch.sum(torch.stack(outputs, dim=-1), dim=-1)

        # Calculate accuracy per class
        per_class_acc = []
        for tp, fp, tn, fn, sup in combined_stats:
            acc = (tp + tn) / (tp + tn + fp + fn)
            per_class_acc.append((acc.item(), sup.item()))

        # Save to csv
        df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
        df.to_csv("per-class-acc-test.csv")

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
