# Tiny Baseline LM on WikiText-2

A clean, reproducible baseline GPT-style language model trained on the WikiText-2 (raw) corpus.

## Overview

This project implements a decoder-only Transformer with a focus on reproducibility and baseline purity. It features:
- **Decoder-only Transformer**: GPT-style architecture.
- **Pre-normalization (Pre-LN)**: For stable training.
- **No Dropout**: Set to 0.0 everywhere to maintain baseline results.
- **Full Quadratic Causal Attention**: Manual mask implemented to ensure no future leakage.
- **Custom BPE Tokenizer**: Byte-level BPE with 8192 vocab size trained only on the training split.
- **Fixed Seq-Len Packing**: Tokenized stream packed into blocks of 129 tokens (`seq_len=128` + 1 target).
- **Single-run Reproducibility**: Seeded components and checksum-verified dataset/tokenizer artifacts.

## Tech Stack

- **Framework**: PyTorch
- **Data**: Hugging Face `datasets` (WikiText-2 raw v1)
- **Tokenizer**: Hugging Face `tokenizers` (Byte-level BPE)
- **Logging**: Weights & Biases (W&B)
- **Utilities**: `numpy`, `pyyaml`, `python-dotenv`, `tqdm`

## Project Structure

```
.
├─ configs/
│  └─ baseline.yaml             # Single source of truth config
├─ data/
│  ├─ raw/                      # Downloaded raw text
│  └─ processed/                # Packed token blocks (memmap)
├─ artifacts/
│  ├─ tokenizers/               # Saved tokenizer.json & meta
│  └─ runs/                     # Checkpoints and logs
├─ scripts/
│  ├─ run_baseline.bat          # Windows runner script
│  └─ run_baseline.sh           # Linux/SSH runner script
├─ src/
│  ├─ data/                     # Data pipeline (download, train tokenizer, pack)
│  ├─ models/                   # Model architecture (layers, GPT wrapper)
│  ├─ utils/                    # Common utils (config, seed, logging, tests)
│  ├─ train.py                  # Main training loop
│  └─ eval.py                   # Evaluation and sampling harness
├─ .env.example                 # W&B credentials template
├─ requirements.txt             # Dependencies
└─ README.md
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**:
   Copy `.env.example` to `.env` and add your `WANDB_API_KEY`.

## RunPod / Linux SSH Setup

If you are running on a RunPod instance or any Linux SSH terminal:

```bash
# Get the code
git clone https://github.com/deflagg/alpha.git
cd alpha

# Pull latest (if already cloned)
git pull origin main

# Standard setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
chmod +x scripts/run_baseline.sh
./scripts/run_baseline.sh
```

```bash
# Get the code
git clone https://github.com/deflagg/alpha.git
cd alpha

# Pull latest (if already cloned)
git pull origin main

# Standard setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
chmod +x scripts/run_experimental.sh
./scripts/run_experimental.sh
```

## Usage

### 1. Full Pipeline (Windows)
Run the entire data prep and training pipeline with one command:
```bat
scripts\run_baseline.bat
```

### 2. Manual Execution

**Data Preparation**:
```bash
python -m src.data.download_wikitext2
python -m src.data.train_tokenizer --config configs/baseline.yaml
python -m src.data.pretokenize_and_pack --config configs/baseline.yaml
```

**Training**:
```bash
python -m src.train --config configs/baseline.yaml
```

**Evaluation**:
```bash
python -m src.eval --config configs/baseline.yaml --ckpt artifacts/runs/baseline_wt2_bpe8k/checkpoints/best_model.pt
```

## Verification

The project includes unit tests to ensure architectural correctness:
```bash
python -m src.utils.tests --config configs/baseline.yaml
```
Tests passed include:
- **Causal Mask Test**: No information leakage from future tokens.
- **Shape Test**: Model outputs match expected tensor shapes.
- **Shift Test**: Dataset correctly returns targets shifted by 1.

## Results

Metrics are logged to W&B. The training loop saves the `best_model.pt` based on validation loss. Perplexity (PPL) is computed on the test split during the final evaluation.
