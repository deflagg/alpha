#!/bin/bash
set -e

# Change to project root
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PYTHONPATH:$(pwd)

CONFIG="configs/falcon.yaml"

echo ""
echo "# 1) data prep (one-time)"
python -m src.data.download_tinystories --config $CONFIG

python -m src.data.train_tokenizer --config $CONFIG

python -m src.data.pretokenize_and_pack --config $CONFIG

echo ""
echo "# 2) train"
python -m src.train_falcon --config $CONFIG

echo ""
echo "# 3) eval (optional standalone)"
CKPT=$1
if [ -z "$CKPT" ]; then
    echo "Done."
    exit 0
fi

python -m src.eval_falcon --config $CONFIG --ckpt $CKPT
echo "Done."
