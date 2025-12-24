#!/bin/bash
set -e

# Change to repo root
cd "$(dirname "$0")/.."
export PYTHONPATH=$PYTHONPATH:$(pwd)

CONFIG="configs/baseline.yaml"

# 1) data prep (one-time)
echo "### 1) Data Prep ###"
python3 -m src.data.download_wikitext2
python3 -m src.data.train_tokenizer --config "$CONFIG"
python3 -m src.data.pretokenize_and_pack --config "$CONFIG"

# 2) train
echo ""
echo "### 2) Train ###"
python3 -m src.train --config "$CONFIG"

# 3) eval (optional)
# Usage: ./scripts/run_baseline.sh artifacts/runs/.../best_model.pt
if [ ! -z "$1" ]; then
    echo ""
    echo "### 3) Eval ###"
    python3 -m src.eval --config "$CONFIG" --ckpt "$1"
fi

echo "Done."
