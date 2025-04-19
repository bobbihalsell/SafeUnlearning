import os

SEED = 42
TOTAL_FORGET_SAMPLES = 32
DIR = os.path.dirname(os.path.abspath(__file__))

FORGET_ROOT = "./data/forget"
FORGET_POOL = "./forget_pool"
RETAIN_ROOT = "./data/retain"
RETAIN_POOL = "./retain_pool"

LR = 0.01
MOMENTUM = 0

MODEL_NAME = "resnet18"
MODEL = "torchvision"
DATASET = "cifar10"
DATASET_SAVE_PATH = "./data"
NUM_CLASSES = 10
TRAINING_EPOCHS = 15
TRAINING_LR = 0.01
MODEL_SAVE_PATH = "./model"
