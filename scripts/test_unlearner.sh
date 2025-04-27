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
    experiment_name=torchhub_test

python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=neggradplus \
    experiment_name=models_test \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_mobilenetv2_x0_5 \
    dataset.save_path=./data \
    +wandb=default \
    wandb.run_id=cifar10_mobilenetv2_x0_5 \
    unlearner.cfg.beta=0.97 \
    unlearner.cfg.momentum=0.9

python src/unlearning/main.py model=torchvision unlearner=neggrad dataset=cifar10 \
    experiment_name=unl_test \
    model.model_name=resnet18 \
    dataset.save_path=./data \
    output_dir=artifacts/models \
    +wandb=default \
    wandb.run_id=resnet18-cifar10-neggrad \
    unlearner.cfg.momentum=0.9