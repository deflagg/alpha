import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=128, max_len=128, tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        # We use TransformerEncoderLayer but with causal mask for decoder behavior
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embedding.weight
            
        self.max_len = max_len

    def forward(self, x):
        b, t = x.size()
        if t > self.max_len:
            x = x[:, :self.max_len]
            t = self.max_len
            
        x = self.embedding(x) + self.pos_embedding[:, :t, :]
        
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        
        x = self.layer(x, src_key_padding_mask=None, src_mask=mask)
        logits = self.lm_head(x)
        return logits

def lm_loss(logits, x, pad_id):
    # Shift logits and targets
    # logits: [B, T, V]
    # x: [B, T]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = x[:, 1:].contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1), 
                           ignore_index=pad_id)
    return loss
