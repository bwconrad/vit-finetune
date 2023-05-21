# Fine-tuning Vision Transformers
Code for fine-tuning ViT models on various classification datasets. Includes options for full model, LoRA and linear fine-tuning procedures.


## Available Datasets

| Dataset            | `--data.dataset` |
|:------------------:|:-----------:|
|[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)| `cifar10`|
|[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)| `cifar100`|
|[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)|  `pets37`|
|[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|  `flowers102`|
|[Food-101](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|  `food101`|
|[STL-10](https://cs.stanford.edu/~acoates/stl10/)|  `stl10`|
|[Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | `dtd`|
|[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | `cars`|
|[FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | `aircraft`|
|[Image Folder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) | `custom`|


## Requirements
- Python 3.8+
- `pip install -r requirements.txt`


## Usage
### Training
- To fine-tune a ViT-B/16 model on CIFAR-100 run:
```
python main.py fit --trainer.accelerator gpu --trainer.devices 1 --trainer.precision 16-mixed
--trainer.max_steps 5000 --model.warmup_steps 500 --model.lr 0.01
--trainer.val_check_interval 500 --data.batch_size 128 --data.dataset cifar100
```
- [`config/`](configs/) contains example configuration files which can be run with:
```
python main.py fit --config path/to/config
```
- To get a list of all arguments run `python train.py --help`

#### Training on a Custom Dataset
To train on a custom dataset first organize the images into 
[Image Folder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) 
format. Then set `--data.dataset custom`, `--data.root path/to/custom/dataset` and `--data.num_classes <num-dataset-classes>`.

### Evaluate
To evaluate a trained model on its test set, find the path of the saved config file for the checkpoint (eg. `output/cifar10/version_0/config.yaml`) and run:
```
python main.py test --ckpt_path path/to/checkpoint --config path/to/config
```
- __Note__: Make sure the `--trainer.precision` argument is set to the same level as used during training.


## Results
All results are from fine-tuned ViT-B/16 models which were pretrained on ImageNet-21k (`--model.model_name vit-b16-224-in21k`).

#### Full Fine-tuning

| Dataset            | Steps          | Warm Up Steps     | Learning Rate      | Test Accuracy | Config                              | 
|:------------------:|:--------------:|:-----------------:|:------------------:|:-------------:|:-----------------------------------:|
| CIFAR-10           | 5000           | 500               | 0.01               | 99.00         | [Link](configs/full/cifar10.yaml)   |
| CIFAR-100          | 5000           | 500               | 0.01               | 92.89         | [Link](configs/full/cifar100.yaml)  |
| Oxford Flowers-102 | 1000           | 100               | 0.03               | 99.02         | [Link](configs/full/flowers102.yaml)|
| Oxford-IIIT Pets   | 2000           | 200               | 0.01               | 93.68         | [Link](configs/full/pets37.yaml)    |
| Food-101           | 5000           | 500               | 0.03               | 90.67         | [Link](configs/full/food101.yaml)   |

#### LoRA

| Dataset            | r  | Alpha | Bias | Steps | Warm Up Steps | Learning Rate | Test Accuracy | Config                                   | 
|:------------------:|:--:|:-----:|:----:|:-----:|:-------------:|:-------------:|:-------------:|:----------------------------------------:|
| CIFAR-100          | 8  | 8     | None | 5000  | 500           | 0.05          | 92.40         | [Link](configs/lora/cifar100-r8.yaml)    |
| Oxford-IIIT Pets   | 1  | 16    | None | 3000  | 100           | 0.05          | 93.30         | [Link](configs/lora/pets37-r1.yaml)      |
| Oxford-IIIT Pets   | 8  | 8     | None | 3000  | 100           | 0.05          | 93.79         | [Link](configs/lora/pets37-r8.yaml)      |
| Oxford-IIIT Pets   | 8  | 8     | All  | 3000  | 300           | 0.05          | 93.76         | [Link](configs/lora/pets37-r8-bias.yaml) |

#### Linear Probe

| Dataset            | Steps          | Warm Up Steps     | Learning Rate      | Test Accuracy | Config                                | 
|:------------------:|:--------------:|:-----------------:|:------------------:|:-------------:|:-------------------------------------:|
| Oxford Flowers-102 | 2000           | 100               | 1.0                | 99.02         | [Link](configs/linear/flowers102.yaml)|
| Oxford-IIIT Pets   | 2000           | 100               | 0.5                | 92.64         | [Link](configs/linear/pets37.yaml)    |
