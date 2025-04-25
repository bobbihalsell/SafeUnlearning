#!/bin/bash

# Exact unlearning for baseline
python src/attacks/lira/main.py dataset=cifar10 attack=lira model=torchvision trainer=trainer unlearner=naive \
    experiment_name=lira_test \
    model.model_name=resnet18 \
    trainer.cfg.epochs=50 \
    unlearner.cfg.epochs=50 \
    seed=10

echo "Running attack for unlearner: neggradplus"
python src/attacks/lira/main.py dataset=cifar10 attack=lira model=torchvision trainer=trainer unlearner=neggradplus \
    experiment_name=lira_test \
    model.model_name=resnet18 \
    unlearner.cfg.epochs=5 \
    unlearner.cfg.momentum=0.9 \
    unlearner.cfg.beta=0.95 \
    seed=10

echo "Running attack for unlearner: scrub"
python src/attacks/lira/main.py dataset=cifar10 attack=lira model=torchvision trainer=trainer unlearner=scrub \
    experiment_name=lira_test \
    model.model_name=resnet18 \
    unlearner.cfg.epochs=5 \
    unlearner.cfg.epochs_per_lr_decay=3 \
    unlearner.cfg.min_epochs=3 \
    unlearner.cfg.max_epochs=2 \
    unlearner.cfg.alpha=0.5 \
    unlearner.cfg.gamma=1.0 \
    unlearner.cfg.sep_epochs=true \
    seed=10

echo "Running attack for unlearner: sgru"
python src/attacks/lira/main.py dataset=cifar10 attack=lira model=torchvision trainer=trainer unlearner=sgru \
    experiment_name=lira_test \
    model.model_name=resnet18 \
    unlearner.cfg.epochs=5 \
    unlearner.cfg.epochs_per_lr_decay=3 \
    unlearner.cfg.weight_decay=0.0005 \
    unlearner.cfg.num_components=10 \
    unlearner.cfg.redirection_strength=1.3 \
    seed=10

# Generate plots
python src/attacks/lira/generate_plots.py --experiment_name lira_test