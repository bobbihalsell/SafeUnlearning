#!/bin/bash

python src/unlearning/main.py dataset=cifar10 model=timm unlearner=neggrad \
    experiment_name=neggrad_test \
    model.model_name=resnet18 \
    model.original_model_ckpt_path=./artifacts/models/cifar10/resnet18_42_original.pt \
    dataset.save_path=./datatest

python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=neggradplus \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data \
    unlearner.cfg.beta=0.97 \
    unlearner.cfg.momentum=0.9 \
    id='001' \
    experiment_name=torchhub_test \
    verbose=false

python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=neggradplus \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_mobilenetv2_x0_5 \
    dataset.save_path=./data \
    output_dir=./artifacts/modelstest \
    +wandb=default \
    unlearner.cfg.beta=0.995 \
    unlearner.cfg.momentum=0.9 \
    unlearner.cfg.epochs=5 \
    verbose=false \
    experiment_name=test

python src/unlearning/main.py model=torchvision unlearner=finetune dataset=cifar10 \
    model.model_name=resnet18 \
    dataset.save_path=./data \
    output_dir=artifacts/models \
    +wandb=default \
    unlearner.cfg.momentum=0.9 \
    experiment_name=tests \
    verbose=false