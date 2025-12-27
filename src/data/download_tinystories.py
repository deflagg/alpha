import os
import json
import hashlib
import argparse
import random
from pathlib import Path
from datasets import load_dataset
from src.utils.config import load_config


from src.data.cache_utils import get_sha256, is_cache_valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    parser.add_argument("--verify", action="store_true", help="Perform full SHA256 verification")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    dataset_name = cfg.data.hf_dataset
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    sub_cfg = cfg.data.subsample
    seed = sub_cfg.seed
    train_n = sub_cfg.train_n
    test_n = sub_cfg.test_n
    val_n = sub_cfg.validation_n
    
    # Cache check
    meta_path = raw_dir / "meta.json"
    expected_config = {
        "dataset": dataset_name,
        "seed": seed,
        "train_n": train_n,
        "test_n": test_n,
        "validation_n": val_n
    }
    required_files = [raw_dir / f"{s}.txt" for s in ["train", "test", "validation"]]
    
    if not args.force and is_cache_valid(meta_path, expected_config, required_files, verify=args.verify):
        print(f"Skipping download (cache valid): {raw_dir}")
        return

    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)
    
    # TinyStories only has train and validation splits
    full_train = dataset["train"]
    
    # Deterministic shuffling of indices
    indices = list(range(len(full_train)))
    random.Random(seed).shuffle(indices)
    
    test_indices = indices[:test_n]
    train_indices = indices[test_n : test_n + train_n]
    
    test_ds = full_train.select(test_indices)
    train_ds = full_train.select(train_indices)
    
    # Validation split
    val_ds = dataset["validation"]
    if val_n is not None:
        val_ds = val_ds.shuffle(seed=seed).select(range(min(val_n, len(val_ds))))
    
    splits = {
        "train": train_ds,
        "test": test_ds,
        "validation": val_ds
    }
    
    meta = {
        "dataset": dataset_name,
        "subsample": {
            "seed": seed,
            "train_n": train_n,
            "test_n": test_n,
            "validation_n": val_n
        },
        "splits": {}
    }
    
    for split_name, ds in splits.items():
        output_file = raw_dir / f"{split_name}.txt"
        print(f"Processing {split_name} split...")
        
        line_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                text = item["text"].strip()
                if text:
                    f.write(text + "\n")
                    line_count += 1
        
        checksum = get_sha256(output_file)
        size_bytes = output_file.stat().st_size
        meta["splits"][split_name] = {
            "file": f"{split_name}.txt",
            "line_count": line_count,
            "sha256": checksum,
            "size_bytes": size_bytes,
            "num_examples": len(ds)
        }
        
        # Save indices for train and test
        if split_name == "train":
            with open(raw_dir / "train_indices.json", "w") as f:
                json.dump(train_indices, f)
        elif split_name == "test":
            with open(raw_dir / "test_indices.json", "w") as f:
                json.dump(test_indices, f)
        
        print(f"  Saved to {output_file} ({line_count} lines, sha256: {checksum[:8]}...)")
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Data download and processing complete. Meta saved to {meta_path}")

if __name__ == "__main__":
    main()
