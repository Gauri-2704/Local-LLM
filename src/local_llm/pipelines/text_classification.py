# local_llm/pipelines/text_classification.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..models.bert import BertConfig, BertModel, masked_mean_pool
from ..tokenization.bert_wordpiece import (
    load_vocab,
    BasicTokenizer,
    BertInputEncoder,
    EncodeOutput,
)

@dataclass
class ClassifierHeadConfig:
    """
    Simple, serializable description of the classifier head.

    You can change this per run without touching package code.
    """
    hidden_sizes: Sequence[int] = (768,)   # MLP hidden layer sizes
    dropouts: Sequence[float] = (0.15, 0.20)
    use_layer_norm: bool = True
    activation: str = "gelu"              # "gelu", "relu", "tanh"

class BertClassifierHead(nn.Module):
    """
    MLP classifier head for BERT pooled outputs, driven by a small config.
    """
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        cfg: ClassifierHeadConfig | None = None,
    ):
        super().__init__()
        cfg = cfg or ClassifierHeadConfig()

        act_map = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        act_cls = act_map.get(cfg.activation.lower(), nn.GELU)

        layers: list[nn.Module] = []
        in_dim = hidden_size

        # First dropout before the head
        if cfg.dropouts:
            layers.append(nn.Dropout(cfg.dropouts[0]))

        # Hidden layers
        for i, hdim in enumerate(cfg.hidden_sizes):
            layers.append(nn.Linear(in_dim, hdim))
            if cfg.use_layer_norm:
                layers.append(nn.LayerNorm(hdim))
            layers.append(act_cls())
            # Optional extra dropout(s)
            if i + 1 < len(cfg.dropouts):
                layers.append(nn.Dropout(cfg.dropouts[i + 1]))
            in_dim = hdim

        # Final classifier layer
        layers.append(nn.Linear(in_dim, num_labels))

        self.net = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)

class BertTextClassifier(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        num_labels: int,
        pooling: str = "cls",
        head: nn.Module | None = None,
        head_config: ClassifierHeadConfig | None = None,
    ):
        super().__init__()
        pooling = pooling.lower().strip()
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.bert = bert
        self.num_labels = num_labels
        self.pooling = pooling

        if head is not None:
            self.classifier = head
        else:
            self.classifier = BertClassifierHead(
                hidden_size=bert.config.hidden_size,
                num_labels=num_labels,
                cfg=head_config,
            )

    @classmethod
    def from_pretrained(
        cls,
        assets_dir: str | Path,
        num_labels: int,
        pooling: str = "cls",
        map_location: str | torch.device = "cpu",
        head: nn.Module | None = None,
        head_config: ClassifierHeadConfig | None = None,
    ) -> "BertTextClassifier":
        assets_dir = Path(assets_dir)
        cfg_path = assets_dir / "config.json"
        weights_path = assets_dir / "pytorch_model.bin"

        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Missing BERT assets at {assets_dir}. "
                "Expected config.json and pytorch_model.bin."
            )

        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        allowed = set(BertConfig.__annotations__.keys())
        cfg = BertConfig(**{k: v for k, v in raw.items() if k in allowed})

        bert = BertModel(cfg)
        sd = torch.load(weights_path, map_location=map_location)
        bert.load_state_dict(sd, strict=True)

        model = cls(
            bert=bert,
            num_labels=num_labels,
            pooling=pooling,
            head=head, 
            head_config=head_config,
        )
        return model

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "classifier_full.pt")
        # also dump config snippet for reloading
        meta = {
            "num_labels": self.num_labels,
            "pooling": self.pooling,
            "bert_config": self.bert.config.__dict__,
        }
        with open(output_dir / "classifier_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _pool(
        self,
        bert_out: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.pooling == "cls":
            pooled = bert_out.get("pooled_output")
            if pooled is None:
                pooled = bert_out["last_hidden_state"][:, 0, :]
            return pooled
        else:
            return masked_mean_pool(bert_out["last_hidden_state"], attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        pooled = self._pool(out, attention_mask)
        logits = self.classifier(pooled)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss_f = nn.CrossEntropyLoss()
            loss = loss_f(logits, labels)
            result["loss"] = loss
        return result


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def build_bert_input_encoder(
    assets_dir: str | Path,
    max_len: int = 256,
    lowercase: bool = True,
) -> BertInputEncoder:
    """
    Convenience factory:
    - loads vocab.txt from `assets_dir`
    - returns a `BertInputEncoder` ready to encode single sentences
    """
    assets_dir = Path(assets_dir)
    vocab_path = assets_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.txt not found at {vocab_path}")
    vocab = load_vocab(vocab_path)
    basic = BasicTokenizer(lower=lowercase)
    return BertInputEncoder(vocab=vocab, max_len=max_len, basic_tokenizer=basic)
