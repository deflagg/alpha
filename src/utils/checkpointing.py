import torch
import os
import yaml
from pathlib import Path
from typing import Any, Dict

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    out_dir: str,
    config: Any,
    is_best: bool = False
):
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
    }
    
    ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(state, ckpt_path)
    
    # Save/update best model link/copy
    if is_best:
        best_path = ckpt_dir / "best_model.pt"
        torch.save(state, best_path)
        print(f"Best model saved at step {step}")
        
    # Save resolved config once
    if step == 0 or not (Path(out_dir) / "resolved_config.yaml").exists():
        with open(Path(out_dir) / "resolved_config.yaml", "w") as f:
            yaml.dump(config.to_dict(), f)

def load_checkpoint(ckpt_path: str, model: torch.nn.Module, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["step"]
