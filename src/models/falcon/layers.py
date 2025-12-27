import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class MoEStats:
    """Statistics returned by the MoE layer."""
    expert_load: torch.Tensor  # [n_experts]
    expert_importance: torch.Tensor  # [n_experts]
    dropped_assignments: int
    dropped_frac: float
    balance_loss: float
    z_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_load": self.expert_load.detach().cpu().tolist(),
            "expert_importance": self.expert_importance.detach().cpu().tolist(),
            "dropped_assignments": self.dropped_assignments,
            "dropped_frac": self.dropped_frac,
            "balance_loss": self.balance_loss,
            "z_loss": self.z_loss,
        }

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # QKV split
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head: [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # [B, n_heads, T, T]
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply mask
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # Attention output
        # [B, n_heads, T, head_dim] -> [B, T, C]
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.out_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class MoE(nn.Module):
    """Top-K Mixture of Experts layer."""
    
    def __init__(
        self,
        d_model: int,
        d_ff_expert: int,
        n_experts: int,
        top_k: int,
        capacity_factor: float,
        drop_tokens: bool,
        router_aux_coef: float,
        router_zloss_coef: float,
    ):
        super().__init__()
        assert n_experts >= top_k, f"n_experts ({n_experts}) must be >= top_k ({top_k})"
        assert top_k >= 1, f"top_k ({top_k}) must be >= 1"
        assert capacity_factor > 0, f"capacity_factor ({capacity_factor}) must be > 0"
        
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.router_aux_coef = router_aux_coef
        self.router_zloss_coef = router_zloss_coef
        
        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            MLP(d_model, d_ff_expert) for _ in range(n_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, MoEStats]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            y: [B, T, d_model]
            aux_loss: scalar tensor
            stats: MoEStats
        """
        B, T, D = x.shape
        N = B * T  # total tokens
        eps = 1e-9
        
        # Flatten to [N, D]
        x_flat = x.view(N, D)
        
        # Router logits and probs
        router_logits = self.router(x_flat)  # [N, E]
        router_probs = F.softmax(router_logits, dim=-1)  # [N, E]
        
        # Top-K selection
        topk_vals, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)  # [N, K]
        
        # Renormalize weights
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + eps)  # [N, K]
        
        # Compute importance (sum of router_probs for each expert)
        importance = router_probs.sum(dim=0)  # [E]
        
        # Capacity calculation
        total_assignments = N * self.top_k
        capacity = int(math.ceil(self.capacity_factor * total_assignments / self.n_experts))
        
        # Count assignments per expert and handle capacity
        # [N, K] -> flatten to get all assignments
        flat_expert_idx = topk_idx.view(-1)  # [N*K]
        flat_token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)  # [N*K]
        flat_k_idx = torch.arange(self.top_k, device=x.device).unsqueeze(0).expand(N, -1).reshape(-1)  # [N*K]
        
        # Initialize output
        y_flat = torch.zeros_like(x_flat)  # [N, D]
        
        # Track load and drops
        expert_load = torch.zeros(self.n_experts, device=x.device, dtype=torch.long)
        dropped_assignments = 0
        
        # Dispatch to each expert
        for e in range(self.n_experts):
            # Find assignments to this expert
            mask_e = (flat_expert_idx == e)
            indices_e = mask_e.nonzero(as_tuple=True)[0]
            
            n_assigned = indices_e.numel()
            
            if n_assigned == 0:
                continue
            
            # Apply capacity limit if drop_tokens is enabled
            if self.drop_tokens and n_assigned > capacity:
                # Keep only first `capacity` assignments (deterministic)
                dropped_assignments += n_assigned - capacity
                indices_e = indices_e[:capacity]
                n_assigned = capacity
            
            expert_load[e] = n_assigned
            
            # Get token indices and weights for this expert's assignments
            token_indices = flat_token_idx[indices_e]  # [n_assigned]
            k_indices = flat_k_idx[indices_e]  # [n_assigned]
            
            # Gather tokens
            tokens_e = x_flat[token_indices]  # [n_assigned, D]
            
            # Run expert
            expert_out = self.experts[e](tokens_e)  # [n_assigned, D]
            
            # Get weights for these assignments
            weights_e = topk_weights[token_indices, k_indices].unsqueeze(-1)  # [n_assigned, 1]
            
            # Scatter-add to output (weighted)
            y_flat.index_add_(0, token_indices, expert_out * weights_e)
        
        # Reshape output
        y = y_flat.view(B, T, D)
        
        # Compute auxiliary losses
        # Load balancing loss: L_balance = n_experts * sum(importance_norm * load_norm)
        importance_norm = importance / (importance.sum() + eps)
        load_float = expert_load.float()
        load_norm = load_float / (load_float.sum() + eps)
        balance_loss = self.n_experts * (importance_norm * load_norm).sum()
        
        # Router z-loss: L_z = mean((logsumexp(router_logits, dim=-1))^2)
        logsumexp_logits = torch.logsumexp(router_logits, dim=-1)  # [N]
        z_loss = (logsumexp_logits ** 2).mean()
        
        # Total aux loss
        aux_loss = self.router_aux_coef * balance_loss + self.router_zloss_coef * z_loss
        
        # Stats
        dropped_frac = dropped_assignments / (total_assignments + eps)
        stats = MoEStats(
            expert_load=expert_load,
            expert_importance=importance,
            dropped_assignments=dropped_assignments,
            dropped_frac=dropped_frac,
            balance_loss=balance_loss.item(),
            z_loss=z_loss.item(),
        )
        
        return y, aux_loss, stats


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        moe_cfg: Any,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Compute d_ff_expert
        d_ff_expert = moe_cfg.d_ff_expert
        if d_ff_expert is None:
            d_ff_expert = d_ff // moe_cfg.top_k
        assert d_ff_expert >= 1, f"d_ff_expert must be >= 1, got {d_ff_expert}"
        
        self.ffn = MoE(
            d_model=d_model,
            d_ff_expert=d_ff_expert,
            n_experts=moe_cfg.n_experts,
            top_k=moe_cfg.top_k,
            capacity_factor=moe_cfg.capacity_factor,
            drop_tokens=moe_cfg.drop_tokens,
            router_aux_coef=moe_cfg.router_aux_coef,
            router_zloss_coef=moe_cfg.router_zloss_coef,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, MoEStats]:
        """
        Returns:
            x: [B, T, d_model]
            aux_loss: scalar tensor
            stats: MoEStats
        """
        # Pre-Norm architecture
        x = x + self.attn(self.ln1(x))
        
        ffn_out, aux_loss, stats = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, aux_loss, stats
