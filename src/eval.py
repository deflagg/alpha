import argparse
import math
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from pathlib import Path

from src.models.factory import build_model
from src.data.dataset import PackedMemmapDataset
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.checkpointing import load_checkpoint

def require_cuda(device_str: str) -> None:
    if not device_str.startswith("cuda"):
        raise RuntimeError(
            f"This repo is GPU-only. Got run.device={device_str!r}. "
            "Set run.device to 'cuda' or 'cuda:0'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This repo is GPU-only.")


@torch.no_grad()
def evaluate_test(model, loader, device, amp=False):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    return mean_loss, ppl

def sample_greedy(model, tokenizer, prompt, max_new_tokens, device):
    model.eval()
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # We use greedy for the baseline, so temperature=1.0 and top_k=None is fine for pure greedy if we use argmax
    # But BaselineGPT.generate uses multinomial. For true greedy, we can just use argmax.
    # Let's adjust BaselineGPT.generate or just do it here.
    # Actually, if temperature is very low, it's basically greedy.
    # Or I can just implement a simple greedy loop here.
    
    generated_idx = model.generate(idx, max_new_tokens, temperature=0.01) # Near-greedy
    decoded = tokenizer.decode(generated_idx[0].cpu().numpy())
    return decoded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(args.seed)
    
    require_cuda(cfg.run.device)
    device = torch.device(cfg.run.device)
    print(f"Using device: {device}")
    
    # Load dataset
    test_ds = PackedMemmapDataset("test", cfg.data.processed_dir)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)
    
    # Load tokenizer for sampling
    tokenizer_path = Path(cfg.tokenizer.out_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Build model and load state
    model = build_model(cfg).to(device)
    load_checkpoint(args.ckpt, model)
    
    print("\n--- Evaluation on Test Split ---")
    loss, ppl = evaluate_test(model, test_loader, device, amp=cfg.train.amp)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Perplexity: {ppl:.2f}")
    
    print("\n--- Sample Generation ---")
    prompts = [
        "The",
        "Once",
        "The girl",
        "Once Upon a time",
        'The day is fun and',
        'The boy and his dog'
    ]
    
    for p in prompts:
        print(f"\nPrompt: {p}")
        output = sample_greedy(model, tokenizer, p, max_new_tokens=32, device=device)
        print(f"Output: {output}")

if __name__ == "__main__":
    main()
