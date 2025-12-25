import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import TransformerBlock

class FalconGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_seq_len = cfg.model.max_seq_len
        
        self.transformer = nn.ModuleDict(dict(
            tok_emb = nn.Embedding(cfg.tokenizer.vocab_size, cfg.model.d_model),
            pos_emb = nn.Embedding(cfg.model.max_seq_len, cfg.model.d_model),
            blocks = nn.ModuleList([
                TransformerBlock(
                    cfg.model.d_model, 
                    cfg.model.n_heads, 
                    cfg.model.d_ff, 
                    cfg.model.max_seq_len
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
        
        print(f"Experimental Model initialized with {self.get_n_params()/1e6:.2f}M parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Cannot forward sequence of length {T}, max_seq_len is {self.max_seq_len}"
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # [1, T]
        tok_emb = self.transformer.tok_emb(idx) # [B, T, C]
        pos_emb = self.transformer.pos_emb(pos) # [1, T, C]
        
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.transformer.blocks:
            x = block(x)
            
        # Final norm
        x = self.transformer.ln_f(x)
        
        # LM Head
        logits = self.lm_head(x) # [B, T, V]
        
        return logits

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
            logits = self(idx_cond)
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
