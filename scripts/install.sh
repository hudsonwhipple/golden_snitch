#!/bin/bash

echo "Installing dependencies..."

pip install -r requirements.txt

echo "Initializing submodules..."

git submodule update --init --recursive

echo "Done."