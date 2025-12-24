import os
import time
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.models.baseline_gpt import BaselineGPT
from src.models.experimental_gpt import ExperimentalGPT
from src.data.dataset import PackedMemmapDataset
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, log_metrics, finish_logging
from src.utils.checkpointing import save_checkpoint

def get_lr(step, max_steps, warmup_steps, lr, min_lr):
    # 1) Linear warmup for warmup_steps steps
    if step < warmup_steps:
        return lr * step / warmup_steps
    # 2) If step > max_steps, return min_lr
    if step > max_steps:
        return min_lr
    # 3) In between, use cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)

@torch.no_grad()
def evaluate(model, dataloader, device, amp=False):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', enabled=amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    
    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    model.train()
    return mean_loss, ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to baseline.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg.run.seed)
    setup_logging(cfg)
    
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    train_ds = PackedMemmapDataset("train", cfg.data.processed_dir)
    val_ds = PackedMemmapDataset("validation", cfg.data.processed_dir)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    
    # Model
    model_type = cfg.model.get("type", "baseline")
    if model_type == "baseline":
        model = BaselineGPT(cfg).to(device)
    elif model_type == "experimental":
        model = ExperimentalGPT(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay
    )
    
    # Training state
    start_step = 0
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.train.amp)
    
    # Resolve run directory
    run_dir = Path(cfg.run.out_dir) / cfg.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training for {cfg.train.max_steps} steps...")
    
    # Training loop using an iterator to handle max_steps better
    train_iter = iter(train_loader)
    
    t0 = time.time()
    for step in range(start_step, cfg.train.max_steps):
        # Handle iterator reset if we exhaust the dataset before max_steps
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        # Determine and set learning rate
        curr_lr = get_lr(step, cfg.train.max_steps, cfg.sched.warmup_steps, cfg.optim.lr, cfg.sched.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
            
        # Optimization step
        with torch.amp.autocast('cuda', enabled=cfg.train.amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
        scaler.scale(loss).backward()
        
        if cfg.train.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Logging
        if step % cfg.logging.log_every == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (cfg.train.batch_size * cfg.data.seq_len * cfg.logging.log_every) / dt
            
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            
            metrics = {
                "train/loss": loss.item(),
                "train/ppl": ppl,
                "train/lr": curr_lr,
                "train/tokens_per_sec": tokens_per_sec,
            }
            log_metrics(metrics, step)
            print(f"Step {step}: loss {loss.item():.4f}, ppl {ppl:.2f}, lr {curr_lr:.2e}, tok/s {tokens_per_sec:.0f}")
            
        # Evaluation
        if step > 0 and step % cfg.train.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device, amp=cfg.train.amp)
            print(f"Evaluation at step {step}: val_loss {val_loss:.4f}, val_ppl {val_ppl:.2f}")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            log_metrics({
                "val/loss": val_loss,
                "val/ppl": val_ppl,
                "val/best_loss": best_val_loss
            }, step)
            
            # Save periodic/best checkpoint
            if step % cfg.train.save_every == 0 or is_best:
                save_checkpoint(
                    model, optimizer, None, step, str(run_dir), cfg, is_best=is_best
                )

    # Final save
    save_checkpoint(model, optimizer, None, cfg.train.max_steps, str(run_dir), cfg, is_best=False)
    finish_logging()
    print("Training finished.")

if __name__ == "__main__":
    main()
