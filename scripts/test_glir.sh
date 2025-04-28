#!/bin/bash

python src/attacks/GLiR/main.py dataset=cifar10 model=torchhub attack=glir \
    experiment_name=glir_test1 \
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    dataset.save_path=./data \
    model.original_model_ckpt_path=artifacts/models/unlearn/neggradplus/cifar10_resnet20_42_original.pt \
    model.unlearned_model_ckpt_path=artifacts/models/unlearn/neggradplus/cifar10_resnet20_42_unlearned.pt \
    +combined_roc_dir=artifacts/attacks/ +plot_combined_roc=true