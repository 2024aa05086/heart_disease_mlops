#!/bin/bash
# Script to download the Heart Disease UCI dataset
# The original UCI datasets are restricted and may return 403 errors when 
# downloading directly. This script downloads the processed Cleveland heart
# disease dataset from a public GitHub repository containing the same
# 14‑feature structure. If the network blocks direct downloads, use the
# dataset provided in the `data/` directory instead.

set -e

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

URL="https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
OUTPUT="$DATA_DIR/heart.csv"

echo "Downloading heart disease dataset from $URL"
curl -L "$URL" -o "$OUTPUT"
echo "Dataset saved to $OUTPUT"