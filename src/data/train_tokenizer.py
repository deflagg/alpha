import os
import json
import hashlib
import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from src.utils.config import load_config

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to baseline.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    train_txt = Path(cfg.data.raw_dir) / "train.txt"
    if not train_txt.exists():
        raise FileNotFoundError(f"Training file not found: {train_txt}. Run data downloader first.")
    
    tokenizer_dir = Path(cfg.tokenizer.out_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_size = cfg.tokenizer.vocab_size
    special_tokens = cfg.tokenizer.special_tokens
    
    print(f"Training Byte-level BPE tokenizer on {train_txt}...")
    print(f"Vocab size: {vocab_size}, Special tokens: {special_tokens}")
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(train_txt)],
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens
    )
    
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Generate meta
    tokenizer_checksum = get_sha256(tokenizer_path)
    train_txt_checksum = get_sha256(train_txt)
    
    # Get special token IDs
    special_token_ids = {token: tokenizer.token_to_id(token) for token in special_tokens}
    
    meta = {
        "vocab_size": vocab_size,
        "special_tokens": special_token_ids,
        "tokenizer_sha256": tokenizer_checksum,
        "train_corpus_sha256": train_txt_checksum
    }
    
    meta_path = tokenizer_dir / "tokenizer_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Tokenizer meta saved to {meta_path}")
    
    # Quick sanity check
    print("\nSanity Check:")
    sample_text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(sample_text)
    print(f"Input: {sample_text}")
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")

if __name__ == "__main__":
    main()
