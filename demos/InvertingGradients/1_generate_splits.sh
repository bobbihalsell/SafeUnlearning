#!/bin/bash

# This script is used to generate the splits for the CIFAR-10 dataset

DIR=$(dirname "$(realpath "$0")")


source $DIR/samples.sh

# Defaults
DEFAULT_UNLEARN="$unlearn1"
DEFAULT_RETAIN="$retain5"

UNLEARN_SET=$1
RETAIN_SET=$2

# # Check if the user provided arguments
# : "${UNLEARN_SET:=$DEFAULT_UNLEARN}"
# : "${RETAIN_SET:=$DEFAULT_RETAIN}"

# Run data splitting app
python src/datasets/main.py forget=filename dataset=cifar10 \
    dataset.load_method=torchvision dataset.val_ratio=0.1 \
    forget.forget_filenames="${UNLEARN_SET}" forget.retain_filenames="${RETAIN_SET}" \
    dataset.init_path=./cifar10 dataset.save_path=./data experiment_name=InvertGradExp \
    dataset.binaries_download_dir=./artifacts 
