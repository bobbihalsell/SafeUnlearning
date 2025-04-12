#!/bin/bash

# Test basic configuration MODEL LOADS ACC 0 
# python src/import_model/main/main.py load_method=torch repo=pytorch/vision model=resnet18 dataset=cifar10 save_dir=./artifacts/models

# Test torch hub loading with different models 
# python src/import_model/main.py load_method=torch repo=chenyaofo/pytorch-cifar-models model=cifar10_resnet20 dataset=cifar10 save_dir=./artifacts/models


# # Test with a different save directory
# python main.py load_method=torch repo=pytorch/vision model=resnet18 dataset=cifar10 save_dir=./different_output

python src/import_model/main.py load_method=file path=./artifacts/models/cifar10_resnet20_cifar10.pt dataset=cifar10 save_dir=./artifacts/models/reloaded/cifar10_resnet20_cifar10.pt