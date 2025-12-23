import argparse
import os
import yaml
import torch
import torch.optim as optim
import wandb
from dotenv import load_dotenv
from tokenizers import Tokenizer
import math

from src.dataset import load_dataset, get_split
from src.model import TinyDecoderLM, lm_loss
from src.eval_harness import evaluate_lm_loss, evaluate_rel_completion_semantic, evaluate_rel_completion_traditional

def train(args):
    load_dotenv()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Overrides from CLI
    if args.steps: config['train']['steps'] = args.steps
    if args.log_every: config['train']['log_every'] = args.log_every
    if args.eval_every: config['train']['eval_every'] = args.eval_every
    if args.save_every: config['train']['save_every'] = args.save_every

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    ds_path = config['dataset']['paths']['dataset_path']
    dataset = load_dataset(ds_path)
    
    modality = "text" if args.model_type == "traditional" else "sem"
    train_ids = get_split(dataset, "train", modality)
    val_ids = get_split(dataset, "val", modality)
    
    if args.model_type == "traditional":
        tokenizer = Tokenizer.from_file(config['dataset']['paths']['tokenizer_path'])
        vocab_size = tokenizer.get_vocab_size()
        pad_id = tokenizer.token_to_id("[PAD]")
        seq_len = config['dataset']['sequences']['seq_len_text']
    else:
        tokenizer = None
        # Semantic vocab size: 4 + num_entities + num_relations
        spec = dataset['meta']['spec']
        vocab_size = 4 + spec['num_entities'] + spec['num_relations']
        pad_id = 0 # SEM_PAD
        seq_len = config['dataset']['sequences']['seq_len_sem']

    # Model
    model_config = config['model']
    model = TinyDecoderLM(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        max_len=seq_len,
        tie_weights=model_config['tie_weights']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    
    # Deterministic batch order
    train_seed = config['train']['train_seed']
    n_train = len(train_ids)
    g = torch.Generator().manual_seed(train_seed)
    indices = torch.randperm(n_train, generator=g)
    
    # W&B
    group = config['wandb']['group_template'].format(
        dataset_seed=dataset['meta']['seeds']['dataset'],
        train_seed=train_seed
    )
    run_name = f"{os.getenv('RUN_NAME_PREFIX', 'exp')}-{args.model_type}"
    
    wandb_key = os.getenv("WANDB_API_KEY")
    wandb_mode = "online" if wandb_key else "disabled"
    if not wandb_key:
        print("WANDB_API_KEY not found. Running in disabled mode.")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", config['wandb']['project']),
        entity=os.getenv("WANDB_ENTITY"),
        group=group,
        name=run_name,
        config=config,
        mode=wandb_mode
    )

    # Training loop
    batch_size = config['train']['batch_size']
    steps = config['train']['steps']
    idx = 0
    
    print(f"Starting training for {steps} steps...")
    for step in range(1, steps + 1):
        model.train()
        
        # Get batch
        batch_indices = indices[idx:idx+batch_size]
        if len(batch_indices) < batch_size:
            # Reshuffle or just wrap around
            indices = torch.randperm(n_train, generator=g)
            idx = 0
            batch_indices = indices[idx:idx+batch_size]
            
        x = train_ids[batch_indices].to(device)
        idx += batch_size
        
        optimizer.zero_grad()
        logits = model(x)
        loss = lm_loss(logits, x, pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
        optimizer.step()
        
        if step % config['train']['log_every'] == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/ppl": math.exp(loss.item()) if loss.item() < 100 else 1e9,
                "step": step
            })
            
        if step % config['train']['eval_every'] == 0:
            val_loss = evaluate_lm_loss(model, val_ids, pad_id, batch_size)
            
            if args.model_type == "traditional":
                acc = evaluate_rel_completion_traditional(
                    model, tokenizer, dataset['eval_rel_completion'], 
                    dataset['entity_names'], dataset['relation_names'], device
                )
            else:
                acc = evaluate_rel_completion_semantic(
                    model, dataset['eval_rel_completion'], 
                    dataset['entity_names'], dataset['relation_names'], device
                )
                
            wandb.log({
                "val/loss": val_loss,
                "val/ppl": math.exp(val_loss) if val_loss < 100 else 1e9,
                "eval/rel_completion_acc": acc,
                "step": step
            })
            print(f"Step {step}: val_loss={val_loss:.4f}, acc={acc:.4f}")

    # Final save
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/{run_name}.pt")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["traditional", "semantic"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    args = parser.parse_args()
    train(args)
