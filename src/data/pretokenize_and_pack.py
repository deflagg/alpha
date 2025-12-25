import os
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
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
    
    # Load tokenizer
    tokenizer_path = Path(cfg.tokenizer.out_dir) / "tokenizer.json"
    tokenizer_meta_path = Path(cfg.tokenizer.out_dir) / "tokenizer_meta.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train_tokenizer first.")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(tokenizer_meta_path, "r") as f:
        tokenizer_meta = json.load(f)
    
    bos_id = tokenizer_meta["special_tokens"]["<bos>"]
    eos_id = tokenizer_meta["special_tokens"]["<eos>"]
    
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size >= 2**16:
        raise RuntimeError(
            f"Tokenizer vocab_size={vocab_size} exceeds uint16 capacity. "
            "Either reduce vocab_size or change packing dtype to uint32."
        )

    
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    seq_len = cfg.data.seq_len
    block_size = seq_len + 1
    
    meta = {
        "seq_len": seq_len,
        "dtype": "uint16",
        "vocab_size": vocab_size,
        "tokenizer_sha256": tokenizer_meta["tokenizer_sha256"],
        "splits": {}
    }
    
    for split in ["train", "validation", "test"]:
        input_txt = raw_dir / f"{split}.txt"
        if not input_txt.exists():
            print(f"Warning: {input_txt} not found. Skipping {split}.")
            continue
            
        print(f"Tokenizing and packing {split} split...")
        all_tokens = []
        
        with open(input_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Processing {split}"):
                text = line.strip()
                if text:
                    encoded = tokenizer.encode(text)
                    tokens = [bos_id] + encoded.ids + [eos_id]
                    all_tokens.extend(tokens)
        
        # Pack into blocks
        total_tokens = len(all_tokens)
        num_blocks = total_tokens // block_size
        
        if num_blocks == 0:
            print(f"Warning: {split} split is too short to form a single block.")
            continue
            
        packed_tokens = np.array(all_tokens[:num_blocks * block_size], dtype=np.uint16)
        packed_tokens = packed_tokens.reshape(num_blocks, block_size)
        
        output_bin = processed_dir / f"{split}.bin"
        # Save as binary file (for memmap compatibility)
        packed_tokens.tofile(output_bin)
        
        checksum = get_sha256(output_bin)
        meta["splits"][split] = {
            "file": f"{split}.bin",
            "shape": [num_blocks, block_size],
            "sha256": checksum
        }
        print(f"  Saved {num_blocks} blocks to {output_bin} (sha256: {checksum[:8]}...)")
    
    meta_path = processed_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Pretokenization and packing complete. Meta saved to {meta_path}")

if __name__ == "__main__":
    main()
