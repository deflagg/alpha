# Semantic Compression Research

Reproducible experiment to compare Traditional LM (BPE-32K) vs Semantic LM (compressed semantic tokens).

## Structure

- `configs/`: YAML configurations.
- `src/`:
  - `toy_wiki.py`: Data generator logic.
  - `tokenizer_train.py`: Trains BPE tokenizer on generated text.
  - `dataset_generate.py`: Pre-generates the shared dataset.
  - `dataset.py`: Dataset loader.
  - `model.py`: Tiny Transformer Decoder.
  - `eval_harness.py`: Evaluation metrics (Accuracy, Loss).
  - `train.py`: Main training script.
- `artifacts/`: Generated dataset and tokenizer (created during setup).

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .\.venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   Copy `.env.example` to `.env` and fill in your `WANDB_API_KEY`.

## Execution Flow

### 1. Generate Data
Everything depends on the pre-generated dataset to ensure consistency between runs.

```bash
# Train tokenizer
python -m src.tokenizer_train --out artifacts/bpe32k.json --samples 200000 --seed 1337

# Generate dataset.pt
python -m src.dataset_generate --config configs/base.yaml
```

### 2. Run Experiments

Run both models. They will use the same deterministic batch order and evaluation set.

```bash
# Traditional LM
python -m src.train --config configs/base.yaml --model_type traditional --device cuda:0

# Semantic LM
python -m src.train --config configs/base.yaml --model_type semantic --device cuda:1
```

## Reproducibility Notes
- Both runs load from `artifacts/dataset.pt`.
- Both runs use `train_seed` from `base.yaml` to ensure identical batch ordering.
- Evaluation is performed on the same `eval_rel_completion` set stored in the dataset file.
