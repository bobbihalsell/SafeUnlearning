# settings.sh

# Constants
SEED=42
TOTAL_FORGET_SAMPLES=32

# Paths
FORGET_ROOT="./data/forget"
FORGET_POOL="./forget_pool"
RETAIN_ROOT="./data/retain"
RETAIN_POOL="./retain_pool"
DATASET_SAVE_PATH="./data"

# Unlearning params
LR=0.01
MOMENTUM=0

# Model/data info
TRAINING_EPOCHS=15
MODEL_NAME="resnet18"
MODEL="torchvision"
DATASET="cifar10"
NUM_CLASSES=10
TRAINING_LR=0.01
MODEL_SAVE_PATH="./model"