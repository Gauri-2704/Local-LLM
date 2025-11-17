# local_llm/models/bert.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BertConfig",
    "BertEmbeddings",
    "BertSelfAttention",
    "BertSelfOutput",
    "BertAttention",
    "BertIntermediate",
    "BertOutput",
    "BertLayer",
    "BertEncoder",
    "BertPooler",
    "BertModel",
    "masked_mean_pool",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BertConfig:
    """
    Minimal configuration for a BERT encoder.

    Fields are aligned with Google's `bert_config.json` schema so we can
    round-trip TF checkpoints without using HuggingFace.
    """
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "new_gelu"  # "gelu" | "relu" | "tanh" | "new_gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "new_gelu": torch.nn.GELU,  # left as-is to match your original implementation
}

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings="
                f"{self.position_embeddings.num_embeddings}."
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(bsz, seq_len)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale = self.attention_head_size ** -0.5

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        return x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._shape(self.query(hidden_states))
        k = self._shape(self.key(hidden_states))
        v = self._shape(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        return ctx.view(hidden_states.size(0), -1, self.all_head_size)


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.dropout(x)
        return self.LayerNorm(x + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn = self.self(x, attention_mask)
        return self.output(attn, x)


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        return self.LayerNorm(x + residual)


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(x, attention_mask)
        inter = self.intermediate(attn_out)
        return self.output(inter, attn_out)


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(x[:, 0]))


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _extend_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        extended = attention_mask[:, None, None, :].to(dtype=dtype)
        extended = (1.0 - extended) * -10000.0
        return extended

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.embeddings(input_ids, token_type_ids, position_ids)

        if attention_mask is not None:
            extended_mask = self._extend_attention_mask(attention_mask, dtype=x.dtype)
        else:
            extended_mask = None

        seq = self.encoder(x, extended_mask)
        pooled = self.pooler(seq)

        return {"last_hidden_state": seq, "pooled_output": pooled}

    # Convenience utilities

    def freeze_encoder(self) -> None:
        for p in self.embeddings.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int = 2) -> None:
        if n <= 0:
            return
        n = min(n, len(self.encoder.layer))
        for layer in self.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


def masked_mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom
