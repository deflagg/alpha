import os
import wandb
from dotenv import load_dotenv
from typing import Any, Dict

def setup_logging(cfg):
    load_dotenv() # Load .env if present
    
    if cfg.logging.wandb:
        # Use WANDB_API_KEY if in env, else it will prompt or use config
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.run.name,
            config=cfg.to_dict()
        )
        print(f"W&B logging initialized for project: {cfg.logging.wandb_project}")
    else:
        print("W&B logging is disabled.")

def log_metrics(metrics: Dict[str, Any], step: int):
    if wandb.run is not None:
        wandb.log(metrics, step=step)

def finish_logging():
    if wandb.run is not None:
        wandb.finish()
