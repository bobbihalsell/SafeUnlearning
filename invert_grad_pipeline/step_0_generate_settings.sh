#!/bin/bash
# Load variables from settings.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/settings.sh"

# Generate settings.py
cat <<EOF > $SCRIPT_DIR/settings.py
import os

TOTAL_FORGET_SAMPLES = $TOTAL_FORGET_SAMPLES
DIR = os.path.dirname(os.path.abspath(__file__))

FORGET_ROOT = "$FORGET_ROOT"
FORGET_POOL = "$FORGET_POOL"
RETAIN_ROOT = "$RETAIN_ROOT"
RETAIN_POOL = "$RETAIN_POOL"

LR = $LR
MOMENTUM = $MOMENTUM

MODEL_NAME = "$MODEL_NAME"
MODEL = "$MODEL"
DATASET = "$DATASET"
DATASET_SAVE_PATH = "$DATASET_SAVE_PATH"
NUM_CLASSES = $NUM_CLASSES
TRAINING_EPOCHS = $TRAINING_EPOCHS
TRAINING_LR = $TRAINING_LR
MODEL_SAVE_PATH = "$MODEL_SAVE_PATH"
EOF

echo "settings.py generated from settings.sh"
