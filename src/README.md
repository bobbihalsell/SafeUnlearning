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

<!-- 
## read me notes by bobbi 
1. Datasets 
To remove instances: (use a list)
python src/datasets/main.py forget=instance dataset=cifar10 forget.forget_idx="[216, 32245, 27206, 10863, 2190, 31849, 25408, 33504]" forget.retain_size=15 dataset.init_dir=./raw dataset.save_dir=./data

To remove classes: (use a list)
python src/datasets/main.py forget=class dataset=cifar10 forget.forget_idx="[5]" dataset.init_dir=./raw dataset.save_dir=./data

To remove a specific number of classes: (use a dict)
python src/datasets/main.py forget=classnum dataset=cifar10 forget.forget_idx="{1:10, 5:10}" dataset.init_dir=./raw dataset.save_dir=./data

1. (b) Train a model optional
to load 
<!-- python src/import_model/main.py load_method=torch init_path=chenyaofo/pytorch-cifar-models model_name=cifar10_resnet20 pretrained=true dataset=cifar10 save_dir=./artifacts/models -->

<!-- 
2. unlearn eg
python src/unlearning/main.py dataset=cifar10 model=classloaded unlearner=neggrad dataset.save_path=./data model.num_classes=10



run
python src/datasets/main.py forget=classnum dataset=cifar10 forget.forget_idx="{5:500}" dataset.init_dir=./raw dataset.save_dir=./data

python src/import_model/main.py load_method=torch init_path=chenyaofo/pytorch-cifar-models model_name=cifar10_resnet20 pretrained=true dataset=cifar10 save_dir=./artifacts/models

 --> 
