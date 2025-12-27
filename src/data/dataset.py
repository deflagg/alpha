import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PackedMemmapDataset(Dataset):
    def __init__(self, split: str, processed_dir: str):
        self.split = split
        self.processed_dir = Path(processed_dir)
        
        # Load meta
        meta_path = self.processed_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Processed data meta not found at {meta_path}. Run pretokenize_and_pack first.")
            
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
            
        if split not in self.meta["splits"]:
            raise ValueError(f"Split {split} not found in processed data meta.")
            
        self.bin_file = self.processed_dir / self.meta["splits"][split]["file"]
        self.shape = tuple(self.meta["splits"][split]["shape"])
        self.dtype = self.meta["dtype"]
        self.seq_len = self.meta["seq_len"]
        
        # Open memmap
        # Using mode 'c' for copy-on-write to avoid accidental writes
        self.data = np.memmap(self.bin_file, dtype=self.dtype, mode='c', shape=self.shape)
        
        # Quick assertion on initialization for a random row (if data exists)
        if len(self.data) > 0:
            idx = np.random.randint(0, len(self.data))
            row = self.data[idx]
            x = row[:self.seq_len]
            y = row[1:]
            assert np.array_equal(y[:-1], x[1:]), f"Shift check failed in {split} split at index {idx}"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Convert to torch tensors (zero-copy from numpy)
        # x and y will be torch.int32
        x = torch.from_numpy(row[:self.seq_len])
        y = torch.from_numpy(row[1:])
        
        return x, y

if __name__ == "__main__":
    # Test loader with a mock setup or check if data exists
    # This is mainly for manual sanity testing
    processed_dir = "data/processed/ts_bpe8k_seq128"
    if os.path.exists(processed_dir):
        try:
            ds = PackedMemmapDataset("validation", processed_dir)
            print(f"Dataset length: {len(ds)}")
            x, y = ds[0]
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            print(f"x: {x[:5]}")
            print(f"y: {y[:5]}")
        except Exception as e:
            print(f"Test failed or data not found: {e}")
    else:
        print("Processed data not found, skipping test.")
