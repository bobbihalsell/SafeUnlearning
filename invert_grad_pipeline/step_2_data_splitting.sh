#!/bin/bash

DIR=$(dirname "$(realpath "$0")")

source $DIR/settings.sh

# Get current working directory

# Read forget indices from file
INDICES=$(cat "${DIR}/forget_indices.json" | tr -d ' \n')

# Run data splitting app
python src/datasets/main.py \
    dataset=cifar10 forget=instance \
    dataset.init_path=./raw \
    dataset.save_dir=$DATASET_SAVE_PATH \
    forget.forget_idx="${INDICES}" \
    forget.retain_size=50

echo "Data split done"

