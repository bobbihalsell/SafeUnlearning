#!/bin/bash

DIR=$(dirname "$(realpath "$0")")

python src/train/main.py dataset=cifar10 model=resnet18 trainer=default  \
    dataset.save_path=./data \
    model.pretrained=false model.save_dir=./model \
    trainer.lr=0.01 trainer.weight_decay=0 \
    trainer.epochs=15

