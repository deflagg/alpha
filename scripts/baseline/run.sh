#!/bin/bash
set -e

# Change to project root
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PYTHONPATH:$(pwd)

CONFIG="configs/baseline.yaml"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --force          Force rebuild of all data artifacts"
    echo "  --verify         Perform slow SHA256 verification of data artifacts"
    echo "  --ckpt PATH      Path to checkpoint for evaluation (runs eval if provided)"
    echo "  -h, --help       Show this help message"
    echo ""
}

PREP_ARGS=()
CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            PREP_ARGS+=("--force")
            shift
            ;;
        --verify)
            PREP_ARGS+=("--verify")
            shift
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1"
            usage
            exit 1
            ;;
    esac
done

echo ""
echo "# 1) data prep (one-time)"
python -m src.data.download_tinystories --config $CONFIG "${PREP_ARGS[@]}"
python -m src.data.train_tokenizer      --config $CONFIG "${PREP_ARGS[@]}"
python -m src.data.pretokenize_and_pack --config $CONFIG "${PREP_ARGS[@]}"

echo ""
echo "# 2) train"
python -m src.train_baseline --config $CONFIG

echo ""
echo "# 3) eval (optional standalone)"
if [ -n "$CKPT" ]; then
    python -m src.eval_baseline --config $CONFIG --ckpt "$CKPT"
else
    echo "No checkpoint provided via --ckpt. Skipping evaluation."
fi

echo "Done."
