#!/usr/bin/env bash

# ================================
# Configuration
# ================================

ENV_NAME="ml-env"
VENV_BASE="$HOME/.venvs"
ENV_PATH="$VENV_BASE/$ENV_NAME"

# ================================
# Resolve requirements.txt path
# ================================

REQ_PATH="$1"

if [ -z "$REQ_PATH" ]; then
    REQ_PATH="./requirements.txt"
fi

if [ ! -f "$REQ_PATH" ]; then
    echo "ERROR: requirements.txt not found at: $REQ_PATH"
    exit 1
fi

REQ_PATH="$(cd "$(dirname "$REQ_PATH")" && pwd)/$(basename "$REQ_PATH")"
echo "Using requirements file: $REQ_PATH"

# ================================
# Ensure base directory exists
# ================================

mkdir -p "$VENV_BASE"

# ================================
# Create virtual environment
# ================================

if [ ! -d "$ENV_PATH" ]; then
    python3 -m venv "$ENV_PATH"
fi

# ================================
# Activate environment
# ================================

source "$ENV_PATH/bin/activate"

# ================================
# Install dependencies
# ================================

python -m pip install --upgrade pip
pip install -r "$REQ_PATH"

echo "Environment '$ENV_NAME' is ready."