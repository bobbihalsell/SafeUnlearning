#!/bin/bash

python src/datasets/main.py forget=class dataset=cifar5 \
    forget.forget_idx="[5, 6]" \
    dataset.name=cifar10 \
    dataset.load_method=local \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test

python src/datasets/main.py forget=classnum dataset=cifar10 \
    dataset.load_method=local \
    forget.forget_idx="{0:10, 2:5}" \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test

python src/datasets/main.py forget=filename dataset=cifar10 \
    forget.forget_filenames="[08443.png, 41333.png, 25574.png, 31402i041.png]" \
    forget.retain_filenames="[]" \
    dataset.load_method=local \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test

python src/datasets/main.py forget=random_n dataset=cifar10 \
    forget.forget_size=10 \
    dataset.load_method=local \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test