import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TransformerBlock, MoEStats
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class AggregatedMoEStats:
    """Aggregated MoE statistics across all layers."""
    total_dropped_assignments: int = 0
    total_dropped_frac: float = 0.0
    avg_balance_loss: float = 0.0
    avg_z_loss: float = 0.0
    per_layer_stats: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dropped_assignments": self.total_dropped_assignments,
            "total_dropped_frac": self.total_dropped_frac,
            "avg_balance_loss": self.avg_balance_loss,
            "avg_z_loss": self.avg_z_loss,
        }

class FalconGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_seq_len = cfg.model.max_seq_len
        
        # Get MoE config if present
        moe_cfg = getattr(cfg.model, 'moe', None)
        self.use_moe = moe_cfg is not None and getattr(moe_cfg, 'enabled', False)
        
        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(cfg.tokenizer.vocab_size, cfg.model.d_model),
            pos_emb = nn.Embedding(cfg.model.max_seq_len, cfg.model.d_model),
            blocks = nn.ModuleList([
                TransformerBlock(
                    cfg.model.d_model, 
                    cfg.model.n_heads, 
                    cfg.model.d_ff, 
                    cfg.model.max_seq_len,
                    moe_cfg=moe_cfg,
                ) for _ in range(cfg.model.n_layers)
            ]),
            ln_f = nn.LayerNorm(cfg.model.d_model)
        ))
        
        self.lm_head = nn.Linear(cfg.model.d_model, cfg.tokenizer.vocab_size, bias=False)
        
        # Weight tying
        if cfg.model.tie_weights:
            self.lm_head.weight = self.transformer.tok_emb.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
        model_type = "MoE" if self.use_moe else "Dense"
        print(f"Falcon Model ({model_type}) initialized with {self.get_n_params()/1e6:.2f}M parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[AggregatedMoEStats]]:
        """
        Returns:
            logits: [B, T, V]
            aux_loss: scalar tensor (sum of aux losses across layers, 0.0 if not using MoE)
            stats: AggregatedMoEStats or None
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max_seq_len is {self.max_seq_len}"
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # [1, T]
        tok_emb = self.transformer.tok_emb(idx) # [B, T, C]
        pos_emb = self.transformer.pos_emb(pos) # [1, T, C]
        
        x = tok_emb + pos_emb
        
        # Transformer blocks with aux loss collection
        total_aux_loss = torch.tensor(0.0, device=device)
        layer_stats = []
        
        for block in self.transformer.blocks:
            x, aux_loss, stats = block(x)
            total_aux_loss = total_aux_loss + aux_loss
            if stats is not None:
                layer_stats.append(stats)
            
        # Final norm
        x = self.transformer.ln_f(x)
        
        # LM Head
        logits = self.lm_head(x) # [B, T, V]
        
        # Aggregate stats
        if self.use_moe and layer_stats:
            n_layers = len(layer_stats)
            agg_stats = AggregatedMoEStats(
                total_dropped_assignments=sum(s.dropped_assignments for s in layer_stats),
                total_dropped_frac=sum(s.dropped_frac for s in layer_stats) / n_layers,
                avg_balance_loss=sum(s.balance_loss for s in layer_stats) / n_layers,
                avg_z_loss=sum(s.z_loss for s in layer_stats) / n_layers,
                per_layer_stats=layer_stats,
            )
        else:
            agg_stats = None
        
        return logits, total_aux_loss, agg_stats

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Greedy/Simple sampling. For the baseline, we prioritize greedy.
        idx is [B, T] array of indices in the current context.
        """
        for _ in range(max_new_tokens):
            # If the sequence context becomes too long we must crop it at max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # Forward the model to get the logits for the index in the sequence
            logits, _, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

