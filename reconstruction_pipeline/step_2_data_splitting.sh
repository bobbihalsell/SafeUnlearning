INDICES=$(cat reconstruction_pipeline/forget_indices.json | tr -d ' \n')
python src/datasets/main.py \
    dataset=cifar10 forget=instance \
    dataset.init_dir=./raw \
    dataset.save_dir=./data \
    forget.forget_idx="${INDICES}" \
    forget.retain_size=30

echo "Data split done"