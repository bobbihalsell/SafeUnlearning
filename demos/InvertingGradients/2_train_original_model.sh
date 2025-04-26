#!/bin/bash
DIR=$(dirname "$(realpath "$0")")

python src/train/main.py dataset=cifar10 model=torchvision trainer=trainer \
    dataset.save_path=./data \
    model.pretrained=false model.model_name=resnet18 \
    trainer.model_save_dir=$DIR/models \
    trainer.cfg.lr=0.01 trainer.cfg.weight_decay=0 \
    trainer.cfg.optimizer=adam \
    trainer.cfg.epochs=15 \
    experiment_name=InvertGradExp \