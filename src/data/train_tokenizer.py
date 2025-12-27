import os
import json
import hashlib
import argparse
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from src.utils.config import load_config


from src.data.cache_utils import get_sha256, is_cache_valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to baseline.yaml")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--verify", action="store_true", help="Perform full SHA256 verification")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    raw_dir = Path(cfg.data.raw_dir)
    train_txt = raw_dir / "train.txt"
    raw_meta_path = raw_dir / "meta.json"
    
    if not train_txt.exists():
        raise FileNotFoundError(f"Training file not found: {train_txt}. Run data downloader first.")
    
    # Get train corpus sha256 from raw meta if possible (fast)
    train_txt_checksum = None
    if raw_meta_path.exists():
        with open(raw_meta_path, "r") as f:
            raw_meta = json.load(f)
            if "splits" in raw_meta and "train" in raw_meta["splits"]:
                train_txt_checksum = raw_meta["splits"]["train"].get("sha256")
    
    if train_txt_checksum is None:
        train_txt_checksum = get_sha256(train_txt)

    tokenizer_dir = Path(cfg.tokenizer.out_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    vocab_size = cfg.tokenizer.vocab_size
    special_tokens = cfg.tokenizer.special_tokens
    
    # Cache check
    meta_path = tokenizer_dir / "tokenizer_meta.json"
    expected_config = {
        "vocab_size": vocab_size,
        "train_corpus_sha256": train_txt_checksum
    }
    # Special tokens are also important, but we'll check them manually or extend is_cache_valid
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    
    if not args.force and is_cache_valid(meta_path, expected_config, [tokenizer_path], verify=args.verify):
        # Also check special tokens in meta
        with open(meta_path, "r") as f:
            meta = json.load(f)
            if list(meta.get("special_tokens", {}).keys()) == special_tokens:
                print(f"Skipping tokenizer training (cache valid): {tokenizer_dir}")
                return

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
    
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Generate meta
    tokenizer_checksum = get_sha256(tokenizer_path)
    size_bytes = tokenizer_path.stat().st_size
    
    # Get special token IDs
    special_token_ids = {token: tokenizer.token_to_id(token) for token in special_tokens}
    
    meta = {
        "vocab_size": vocab_size,
        "special_tokens": special_token_ids,
        "tokenizer_sha256": tokenizer_checksum,
        "train_corpus_sha256": train_txt_checksum,
        "size_bytes": size_bytes
    }
    
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
