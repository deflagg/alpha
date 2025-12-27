import os
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
from src.utils.config import load_config


from src.data.cache_utils import get_sha256, is_cache_valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to baseline.yaml")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--verify", action="store_true", help="Perform full SHA256 verification")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Load tokenizer
    tokenizer_dir = Path(cfg.tokenizer.out_dir)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_meta_path = tokenizer_dir / "tokenizer_meta.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train_tokenizer first.")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with open(tokenizer_meta_path, "r") as f:
        tokenizer_meta = json.load(f)
    
    bos_id = tokenizer_meta["special_tokens"]["<bos>"]
    eos_id = tokenizer_meta["special_tokens"]["<eos>"]
    
    vocab_size = tokenizer.get_vocab_size()
    # We are switching to int32, so no uint16 overflow check needed (beyond what int32 can hold)
    
    raw_dir = Path(cfg.data.raw_dir)
    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    seq_len = cfg.data.seq_len
    block_size = seq_len + 1
    
    # Cache check
    meta_path = processed_dir / "meta.json"
    expected_config = {
        "seq_len": seq_len,
        "dtype": "int32",
        "tokenizer_sha256": tokenizer_meta["tokenizer_sha256"]
    }
    required_files = [processed_dir / f"{s}.bin" for s in ["train", "validation", "test"] if (raw_dir / f"{s}.txt").exists()]
    
    if not args.force and is_cache_valid(meta_path, expected_config, required_files, verify=args.verify):
        print(f"Skipping pretokenize/pack (cache valid): {processed_dir}")
        return

    meta = {
        "seq_len": seq_len,
        "dtype": "int32",
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
            
        packed_tokens = np.array(all_tokens[:num_blocks * block_size], dtype=np.int32)
        packed_tokens = packed_tokens.reshape(num_blocks, block_size)
        
        output_bin = processed_dir / f"{split}.bin"
        # Save as binary file (for memmap compatibility)
        packed_tokens.tofile(output_bin)
        
        checksum = get_sha256(output_bin)
        size_bytes = output_bin.stat().st_size
        meta["splits"][split] = {
            "file": f"{split}.bin",
            "shape": [num_blocks, block_size],
            "sha256": checksum,
            "size_bytes": size_bytes
        }
        print(f"  Saved {num_blocks} blocks to {output_bin} (sha256: {checksum[:8]}...)")
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Pretokenization and packing complete. Meta saved to {meta_path}")

if __name__ == "__main__":
    main()
