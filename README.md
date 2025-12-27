# Tiny Baseline LM on TinyStories

A clean, reproducible baseline GPT-style language model trained on the TinyStories dataset.

## Overview

This project implements a decoder-only Transformer with a focus on reproducibility and baseline purity. It features:
- **Decoder-only Transformer**: GPT-style architecture (Baseline, Falcon MoE, Condor).
- **Pre-normalization (Pre-LN)**: For stable training.
- **No Dropout**: Set to 0.0 everywhere to maintain baseline results.
- **Full Quadratic Causal Attention**: Manual mask implemented to ensure no future leakage.
- **Custom BPE Tokenizer**: Byte-level BPE with 8192 vocab size trained only on the training split.
- **Optimized Data Pipeline**: 
    - **Skip-if-valid**: Scripts skip work by comparing `meta.json` with current config and verifying file existence/size.
    - **Hashing**: Use `--verify` to trigger full SHA256 validation (slower).
    - **int32 Storage**: Packed tokens use `int32` dtype for large vocab support.
    - **Zero-Copy Loading**: Data is loaded directly from memmap without per-sample `astype(np.int64)`.
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
│  ├─ condor.yaml               # Condor model config
│  └─ falcon.yaml               # Falcon model config
├─ data/
│  ├─ raw/                      # Downloaded raw text
│  └─ processed/                # Packed token blocks (memmap)
├─ artifacts/
│  ├─ tokenizers/               # Saved tokenizer.json & meta
│  └─ runs/                     # Checkpoints and logs
├─ scripts/
│  ├─ baseline/                 # Baseline runner scripts
│  ├─ condor/                   # Condor runner scripts
│  └─ falcon/                   # Falcon runner scripts
├─ src/
│  ├─ data/                     # Data pipeline (download, tokenize, pack)
│  │  └─ cache_utils.py         # Shared cache validation logic
│  ├─ models/                   # Model architectures
│  │  ├─ baseline/              # Baseline GPT implementation
│  │  ├─ condor/                # Condor GPT implementation
│  │  └─ falcon/                # Falcon GPT implementation
│  ├─ utils/                    # Common utils
│  ├─ train_*.py                # Training loops per model type
│  └─ eval_*.py                 # Evaluation harnesses per model type
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

   chmod +x scripts/baseline/run.sh
   ./scripts/baseline/run.sh
   ```

2. **Condor model setup**:
   ```bash
   git clone https://github.com/deflagg/alpha.git
   cd alpha

   git pull origin main

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   chmod +x scripts/condor/run.sh
   ./scripts/condor/run.sh
   ```

3. **Falcon model setup**:
   ```bash
   git clone https://github.com/deflagg/alpha.git
   cd alpha

   git pull origin main

   #python3 -m venv .venv
   #source .venv/bin/activate
   pip install -r requirements.txt

   chmod +x scripts/falcon/run.sh
   # Create .env and add WANDB_API_KEY if logging to W&B
   ./scripts/falcon/run.sh
   ```

## Usage

### 1. Training & Data Prep
The runner scripts handle data downloading, tokenizer training, and token packing before starting the training loop. They use a fast cache check by default.

```bash
chmod +x scripts/**/*.sh

# 1. Standard run (cached prep + training, skips eval)
./scripts/condor/run.sh

# 2. Force rebuild artifacts (ignores cache)
./scripts/condor/run.sh --force

# 3. Verify artifact integrity (slow SHA256 check)
./scripts/condor/run.sh --verify

# 4. Run training and THEN evaluation
./scripts/condor/run.sh --ckpt artifacts/runs/condor_ts_bpe8k/checkpoints/best_model.pt

# 5. Force rebuild everything and evaluate
./scripts/condor/run.sh --force --ckpt artifacts/runs/condor_ts_bpe8k/checkpoints/best_model.pt
```

### 2. Standalone Evaluation
Evaluation is only triggered in `run.sh` if `--ckpt` is provided. You can also run it directly:

```bash
python -m src.eval_condor --config configs/condor.yaml --ckpt artifacts/runs/condor_ts_bpe8k/checkpoints/best_model.pt
```

For all scripts, use `-h` or `--help` to see all available options.

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
