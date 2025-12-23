# Setup
python -m venv .venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Generate Data
python -m src.tokenizer_train --out artifacts/bpe32k.json --samples 200000 --seed 1337
python -m src.dataset_generate --config configs/base.yaml
# Train Traditional
python -m src.train --config configs/base.yaml --model_type traditional