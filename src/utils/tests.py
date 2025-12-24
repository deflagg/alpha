import torch
import numpy as np
import argparse
from src.models.baseline_gpt import BaselineGPT
from src.utils.config import load_config

def test_causal_mask(cfg):
    print("Testing Causal Mask...")
    model = BaselineGPT(cfg)
    model.eval()
    
    B, T = 1, 10
    x = torch.randint(0, cfg.tokenizer.vocab_size, (B, T))
    
    # Forward pass 1
    with torch.no_grad():
        logits1 = model(x)
        
    # Modify future tokens
    x_mod = x.clone()
    x_mod[0, 5:] = torch.randint(0, cfg.tokenizer.vocab_size, (5,))
    
    # Forward pass 2
    with torch.no_grad():
        logits2 = model(x_mod)
        
    # Confirm earlier logits (up to index 4) don't change
    # Note: logits at position i use tokens up to i. 
    # So modifying index 5 should not affect logits at indices 0, 1, 2, 3, 4.
    diff = (logits1[0, :5, :] - logits2[0, :5, :]).abs().max()
    print(f"Max diff in earlier logits: {diff.item():.2e}")
    assert diff < 1e-5, "Causal mask failed! Future tokens affected past logits."
    print("Causal Mask Test Passed.")

def test_shapes(cfg):
    print("\nTesting Shapes...")
    model = BaselineGPT(cfg)
    B, T = 4, 128
    x = torch.randint(0, cfg.tokenizer.vocab_size, (B, T))
    
    with torch.no_grad():
        logits = model(x)
        
    expected_shape = (B, T, cfg.tokenizer.vocab_size)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == expected_shape, f"Unexpected output shape: {logits.shape}, expected {expected_shape}"
    print("Shape Test Passed.")

def test_shift():
    print("\nTesting Dataset Shift (Manual Check)...")
    # This test is partially covered by the assertion in PackedMemmapDataset.__init__
    # But let's re-verify the logic here for a dummy row.
    seq_len = 128
    row = np.arange(seq_len + 1)
    x = row[:seq_len]
    y = row[1:]
    
    assert len(x) == seq_len, f"x length {len(x)} != {seq_len}"
    assert len(y) == seq_len, f"y length {len(y)} != {seq_len}"
    assert np.array_equal(y[:-1], x[1:]), "Target is not input shifted by 1!"
    print("Shift Test Logic Passed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    try:
        test_causal_mask(cfg)
        test_shapes(cfg)
        test_shift()
        print("\nAll automated tests passed successfully.")
    except Exception as e:
        print(f"\nTests failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
