#!/bin/bash

python src/attacks/lira/main.py dataset=cifar10 attack=lira model=torchvision trainer=trainer unlearner=neggradplus \
    experiment_name=lira_test \
    model.model_name=resnet18 \
    model.pretrained=true \
    trainer.cfg.epochs=10 \
    seed=100

python src/attacks/lira/main.py dataset=cifar10 attack=lira model=torchhub trainer=trainer unlearner=naive \
    experiment_name=lira_test_v2 \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20