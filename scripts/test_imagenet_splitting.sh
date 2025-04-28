#!/bin/bash

python src/datasets/main.py forget=filename dataset=imagenet \
    forget.forget_filenames="[n01443537_0.JPEG, n01443537_1.JPEG, n01641577_4.JPEG, rubbish.JPEG]" \
    dataset.load_method=local \
    dataset.init_path=./trial-local_2 \
    dataset.save_path=./imagenet \
    experiment_name=test

python src/datasets/main.py forget=random_n dataset=imagenet \
    forget.retain_size=15 \
    forget.forget_size=10 \
    dataset.load_method=local \
    dataset.init_path=./trial-local_2 \
    dataset.save_path=./imagenet \
    experiment_name=test