# Unlearning and Reconstruction App

## Version: 0.1.0 Summary
This unlearning application allows a user to specify a `config.yaml` file to perform unlearning on image classification models using benchmark datasets.

## Usage
1. Clone this repository.
2. Create a virtual environment in the root directory.
3. Run `pip install -e .` from root.
4. Run `pip install -r requirements.txt`.
5. An example `config.yaml` file is provided. You can run ` python src/unlearning/main.py --config_path src/experiments/simple_exp.yaml` to perform unlearning for MobileNetV2 on 3 samples from CIFAR5.

## Supported methods
### Models
Any `torchvision` or `timm` image classification model. The user must specify the name `str` appropriately.

### Datasets
CIFAR5, CIFAR10, CIFAR100.

### Unlearning Algorithms
SCRUB, NegGrad, NegGrad+, Finetuning, K-unlearn. 

## Important next features
1. Custom forget dataset loading, so the user can forget a particular image from very large datasets (e.g. ImageNet).
2. WandB training logging.
3. Optuna hyperparameter search.
