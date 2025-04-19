#!/bin/bash

# Get the directory]
DIR=$(dirname "$(realpath "$0")")

# Load settings
source "$DIR/settings.sh"

# Create labels directory if it doesn't exist
if [ ! -d "$DIR/labels" ]; then
    mkdir -p "$DIR/labels"
fi

# Move forget data to forget_pool
if [ ! -d "$FORGET_POOL" ]; then
    mv "$FORGET_ROOT" "$FORGET_POOL"
fi

# Move retain data to retain_pool
if [ ! -d "$RETAIN_POOL" ]; then
    mv "$RETAIN_ROOT" "$RETAIN_POOL"
fi
