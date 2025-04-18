#!/bin/bash

# Get current working directory
DIR=$(dirname "$(realpath "$0")")

# Read forget indices from file
INDICES=$(cat "${DIR}/forget_indices.json" | tr -d ' \n')

# Run data splitting app
python src/datasets/main.py \
    dataset=cifar10 forget=instance \
    dataset.init_dir=./raw \
    dataset.save_dir=./data \
    forget.forget_idx="${INDICES}" \
    forget.retain_size=50

echo "Data split done"

