#!/bin/bash
# install.sh - BertGCN Installation Script

set -e  # Exit on error

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install it from https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install framework
echo "Installing BertGCN Framework..."
poetry install --no-interaction
echo "Installation complete."