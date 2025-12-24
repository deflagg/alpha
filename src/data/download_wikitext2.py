import os
import json
import hashlib
from pathlib import Path
from datasets import load_dataset
from src.utils.config import load_config

def get_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    # Load config (hardcoded path or use environment variable if needed, but here we assume root/configs/baseline.yaml)
    # Since this is a data prep script, we can point it to the baseline config
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "baseline.yaml"
    cfg = load_config(str(config_path))
    
    dataset_name = cfg.data.hf_dataset
    dataset_config = cfg.data.hf_config
    raw_dir = Path(cfg.data.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset {dataset_name}:{dataset_config}...")
    dataset = load_dataset(dataset_name, dataset_config)
    
    meta = {
        "dataset": dataset_name,
        "config": dataset_config,
        "splits": {}
    }
    
    for split in ["train", "validation", "test"]:
        output_file = raw_dir / f"{split}.txt"
        print(f"Processing {split} split...")
        
        line_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset[split]:
                text = item["text"].strip()
                if text:
                    f.write(text + "\n")
                    line_count += 1
        
        checksum = get_sha256(output_file)
        meta["splits"][split] = {
            "file": f"{split}.txt",
            "line_count": line_count,
            "sha256": checksum
        }
        print(f"  Saved to {output_file} ({line_count} lines, sha256: {checksum[:8]}...)")
    
    meta_path = raw_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Data download and processing complete. Meta saved to {meta_path}")

if __name__ == "__main__":
    main()
