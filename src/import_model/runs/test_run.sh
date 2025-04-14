#!/bin/bash

# Test basic configuration MODEL LOADS ACC 0 
python src/import_model/main.py load_method=torch init_path=pytorch/vision init_name=resnet18 dataset=cifar10 save_dir=./artifacts/models
# Test torch hub loading with different models 
python src/import_model/main.py load_method=torch init_path=chenyaofo/pytorch-cifar-models init_name=cifar10_resnet20 pretrained=true dataset=cifar10 save_dir=./artifacts/models
# # Test with a different save directory
python src/import_model/main.py load_method=torch init_path=chenyaofo/pytorch-cifar-models init_name=cifar10_resnet20 weight_path=./artifacts/models/cifar10_resnet20_cifar10.pt dataset=cifar10 save_dir=./artifacts/models save_name=loadedmodel

# with class test.
python src/import_model/main.py load_method=function init_path=./modelloadtest init_name=SimpleCIFAR10Model weight_path=./saved_models/cifar10_model_1744618737.pth dataset=cifar10 save_dir=./artifacts/models save_name=functionmodel
