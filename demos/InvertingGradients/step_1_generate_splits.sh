#!/bin/bash

# This script is used to generate the splits for the CIFAR-10 dataset

DIR=$(dirname "$(realpath "$0")")

source $DIR/samples.sh

# Run data splitting app
python src/datasets/main.py forget=filename dataset=cifar10 \
    dataset.load_method=torchvision dataset.val_ratio=0.1 \
    forget.forget_filenames="${unlearn32}" forget.retain_filenames="${retain50}" \
    dataset.init_path=./cifar10 dataset.save_path=./data experiment_name=InvertGradExp \
    dataset.binaries_download_dir=./artifacts 
