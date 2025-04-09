#!/bin/bash

# Check if user provided the path to the data directory
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/dataset_root"
  exit 1
fi

# Resolve absolute path
DATASET_DIR="$(cd "$1" && pwd)"
SOURCE_DIR="$DATASET_DIR/train"

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: $SOURCE_DIR does not exist. Make sure 'train/' is inside the dataset directory."
  exit 1
fi

# Create symlinks for both retain and forget
for SPLIT in retain forget; do
  TARGET_DIR="$DATASET_DIR/$SPLIT"
  echo "Creating symlinks in $TARGET_DIR"

  rm -rf "$TARGET_DIR"/*
  mkdir -p "$TARGET_DIR"

  for class_dir in "$SOURCE_DIR"/*; do
    class_name=$(basename "$class_dir")
    ln -s "$class_dir" "$TARGET_DIR/$class_name"
  done
done

echo "Symlinks created in 'retain' and 'forget' under: $DATASET_DIR"
