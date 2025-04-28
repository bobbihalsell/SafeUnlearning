#!/bin/bash

# This script requires manual setting of model checkpoint paths.

echo You must specify the model checkpoint path before running this test

python src/attacks/main_reconstructor.py dataset=cifar10 attack=invertgrad model=torchvision +wandb=reconstruction \
    experiment_name=invert_grad_test \
    dataset.save_path=./data \
    attack.unlearned_labels=[0] attack.cfg.recon_iterations=500 \
    model.original_model_ckpt_path=  \
    model.unlearned_model_ckpt_path= \
    model.model_name=resnet18 \
    wandb.project_name=testrun