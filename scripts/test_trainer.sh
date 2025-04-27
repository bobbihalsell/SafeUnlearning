#!/bin/bash

python src/train/main.py dataset=imagenet model=torchvision trainer=trainer \
    model.model_name=resnet18 \
    model.pretrained=true \
    dataset.save_path=./data \
    trainer.model_save_dir=./artifacts/models/imagenet_test \
    trainer.cfg.epochs=10 \
    +wandb=default \
    experiment_name=test

python src/train/main.py dataset=cifar10 model=torchvision trainer=trainer \
    model.model_name=resnet18 \
    model.pretrained=true \
    dataset.save_path=./data \
    trainer.model_save_dir=./artifacts/models/cifar10 \
    trainer.cfg.epochs=3 \
    trainer.cfg.lr=0.0005 \
    experiment_name=test \
    +wandb=default

python src/train/main.py dataset=cifar10 model=torchhub trainer=trainer \
    dataset.save_path=./data \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    trainer.model_save_dir=./artifacts/models/cifar_test \
    trainer.cfg.epochs=2 \
    +wandb=default \
    experiment_name=test