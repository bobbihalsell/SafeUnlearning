#!/bin/bash

python src/datasets/main.py forget=class dataset=cifar5 \
    forget.forget_idx="[5, 6]"  \
    dataset.name=cifar10 \
    dataset.load_method=torchvision \
    dataset.init_path=./cifar10 \
    dataset.binaries_download_dir=./artifacts \
    dataset.save_path=./data \
    experiment_name=test

python src/datasets/main.py forget=classnum dataset=cifar10 \
    dataset.load_method=torchvision \
    forget.forget_idx="{0:10, 2:5}" \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test \
    dataset.binaries_download_dir=./artifacts

python src/datasets/main.py forget=filename dataset=cifar10 \
    dataset.load_method=torchvision \
    forget.forget_filenames="[00001.png, 00002.png]" \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test \
    dataset.binaries_download_dir=./artifacts

python src/datasets/main.py forget=random_n dataset=cifar10 \
    dataset.load_method=torchvision \
    forget.forget_size=500 \
    dataset.init_path=./cifar10 \
    dataset.save_path=./data \
    experiment_name=test \
    dataset.binaries_download_dir=./artifacts