"""Small GPT-2 style transformer with intermediate representation access."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    # q, k: (B, H, T, D_head)  cos, sin: (T, D_head)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

    def forward(self, x, return_weights=False):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D_head)
        q = q.transpose(1, 2)  # (B, H, T, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # scaled dot-product attention with causal mask
        if return_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            out = attn_weights @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
            )
            attn_weights = None

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, attn_weights


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, return_weights=False):
        attn_out, attn_weights = self.attn(self.ln1(x), return_weights=return_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    layer_representations: list  # list of (B, T, D) tensors per layer
    attention_weights: list      # list of (B, H, T, T) or None per layer


class SmallTransformer(nn.Module):
    """Small GPT-2-style decoder-only transformer.

    Exposes per-layer representations and attention weights for MCR2
    local loss and representation quality metrics.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # GPT-2 style residual scaling
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.cfg.n_layers),
            )
            nn.init.normal_(
                block.mlp.fc2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.cfg.n_layers),
            )

    def forward(self, idx, collect_intermediates=False, collect_layer_reps=False):
        """Forward pass.

        Args:
            idx: (B, T) token indices
            collect_intermediates: if True, store per-layer representations
                AND attention weights (expensive). Use only during evaluation.
            collect_layer_reps: if True, store per-layer representations but
                use efficient SDPA (no attention weights). Use during training
                when MCR2 local loss is needed.

        Returns:
            TransformerOutput with logits and optional intermediates.
        """
        x = self.drop(self.tok_emb(idx))

        layer_reps = []
        attn_weights_list = []

        store_reps = collect_intermediates or collect_layer_reps

        for block in self.blocks:
            x, attn_w = block(x, return_weights=collect_intermediates)
            if store_reps:
                layer_reps.append(x)
            if collect_intermediates:
                attn_weights_list.append(attn_w)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return TransformerOutput(
            logits=logits,
            layer_representations=layer_reps,
            attention_weights=attn_weights_list,
        )

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
