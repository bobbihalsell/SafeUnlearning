echo Beginning training...

# Load the data from directory used in previous step
# And override default config options
python src/train/main.py \
    model=resnet18 dataset=cifar10 train_cfg=default wandb_cfg=default \
    dataset.load_dir=./data \
    model.pretrained=False model.num_classes=10 model.save_dir=./model \
    train_cfg.epochs=15 train_cfg.weight_decay=0 train_cfg.lr=0.01 \
