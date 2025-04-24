echo Beginning training...

DIR=$(dirname "$(realpath "$0")")

source $DIR/settings.sh

# Load the data from directory used in previous step
# And override default config options
python src/train/main.py \
    model=resnet18 dataset=$DATASET train_cfg=default wandb=default \
    dataset.load_dir=$DATASET_SAVE_PATH seed=$SEED \
    model.pretrained=False model.num_classes=$NUM_CLASSES model.save_dir=$MODEL_SAVE_PATH \
    train_cfg.epochs=$TRAINING_EPOCHS train_cfg.weight_decay=0 train_cfg.lr=$TRAINING_LR \
