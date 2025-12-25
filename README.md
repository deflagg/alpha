# Tiny Baseline LM on TinyStories

A clean, reproducible baseline GPT-style language model trained on the TinyStories dataset.

## Overview

This project implements a decoder-only Transformer with a focus on reproducibility and baseline purity. It features:
- **Decoder-only Transformer**: GPT-style architecture.
- **Pre-normalization (Pre-LN)**: For stable training.
- **No Dropout**: Set to 0.0 everywhere to maintain baseline results.
- **Full Quadratic Causal Attention**: Manual mask implemented to ensure no future leakage.
- **Custom BPE Tokenizer**: Byte-level BPE with 8192 vocab size trained only on the training split.
- **Fixed Seq-Len Packing**: Tokenized stream packed into blocks of 129 tokens (`seq_len=128` + 1 target).
- **Single-run Reproducibility**: Seeded components and checksum-verified dataset/tokenizer artifacts.
- **GPU-Only**: Explicitly requires CUDA for performance and consistency.

## Tech Stack

- **Framework**: PyTorch
- **Data**: Hugging Face `roneneldan/TinyStories`
- **Tokenizer**: Hugging Face `tokenizers` (Byte-level BPE)
- **Logging**: Weights & Biases (W&B)
- **Utilities**: `numpy`, `pyyaml`, `python-dotenv`, `tqdm`

## Project Structure

```
.
├─ configs/
│  ├─ baseline.yaml             # Baseline model config
│  └─ falcon.yaml               # Falcon model config
├─ data/
│  ├─ raw/                      # Downloaded raw text
│  └─ processed/                # Packed token blocks (memmap)
├─ artifacts/
│  ├─ tokenizers/               # Saved tokenizer.json & meta
│  └─ runs/                     # Checkpoints and logs
├─ scripts/
│  ├─ run_baseline.bat          # Windows baseline runner script
│  ├─ run_baseline.sh           # Linux/SSH baseline runner script
│  ├─ run_falcon.bat            # Windows falcon runner script
│  └─ run_falcon.sh             # Linux/SSH falcon runner script
├─ src/
│  ├─ data/                     # Data pipeline (download, train tokenizer, pack)
│  ├─ models/                   # Model architecture and factory
│  ├─ utils/                    # Common utils (config, seed, logging, tests)
│  ├─ train.py                  # Main training loop
│  └─ eval.py                   # Evaluation and sampling harness
├─ .env.example                 # W&B credentials template
├─ requirements.txt             # Dependencies
└─ README.md
```

## Setup

1. **Baseline model setup**:
   ```bash
   git clone https://github.com/deflagg/alpha.git
   cd alpha

   git pull origin main

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   chmod +x scripts/run_baseline.sh
   ./scripts/run_baseline.sh
   ```

2. **Falcon model setup**:
   ```bash
   git clone https://github.com/deflagg/alpha.git
   cd alpha

   git pull origin main

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   chmod +x scripts/run_falcon.sh
   ./scripts/run_falcon.sh
   ```

## Usage

### 1. Full Pipeline (Windows)
Run the entire data prep and training pipeline with one command:
```bat
scripts\run_baseline.bat
```
Or for the falcon model:
```bat
scripts\run_falcon.bat
```

### 2. Full Pipeline (Linux)
```bash
chmod +x scripts/*.sh
./scripts/run_baseline.sh
```

### 3. Evaluation
Pass a checkpoint path to the runner scripts to run evaluation:
```bash
python -m src.eval --config configs/baseline.yaml --ckpt artifacts/runs/baseline_ts_bpe8k/checkpoints/best_model.pt
```

## Verification

The project includes unit tests to ensure architectural correctness:
```bash
python -m src.utils.tests --config configs/baseline.yaml
```
Tests include:
- **Causal Mask Test**: No information leakage from future tokens.
- **Shape Test**: Model outputs match expected tensor shapes.
- **Shift Test**: Dataset correctly returns targets shifted by 1.

## Results

Metrics are logged to W&B. The training loop saves the `best_model.pt` based on validation loss. Perplexity (PPL) is computed on the validation/test splits.
