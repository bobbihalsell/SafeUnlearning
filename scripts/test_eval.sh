#!/bin/bash

python src/evaluation/main.py \
    dataset=cifar10 model=torchhub \                                                  
    model.repo_path=chenyaofo/pytorch-cifar-models \
    model.model_name=cifar10_resnet20 \
    +model.original_ckpt_path=artifacts/unlearn/torchhub_test/unlearn/sgru/cifar10_resnet20_42_original_001.pt \
    +model.unlearned1_ckpt_path=artifacts/unlearn/torchhub_test/unlearn/sgru/cifar10_resnet20_42_unlearned_001.pt \
    +model.unlearned2_ckpt_path=artifacts/unlearn/torchhub_test/unlearn/sgru/cifar10_resnet20_42_unlearned_002.pt \
    dataset.save_path=./unldata \
    experiment_name=torchhub_test