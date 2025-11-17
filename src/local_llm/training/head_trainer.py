# local_llm/training/head_trainer.py
from __future__ import annotations

import json, math, copy
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from ..pipelines.text_classification import (
    BertTextClassifier,
    ClassifierHeadConfig,
)
from ..models.bert import masked_mean_pool


def _build_loader(split: Dict[str, torch.Tensor], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    g = torch.Generator(); g.manual_seed(seed)
    token_type_ids = split.get("token_type_ids", torch.zeros_like(split["input_ids"]))
    ds = TensorDataset(
        split["input_ids"],
        token_type_ids,
        split["attention_mask"],
        split["labels"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def train_classifier_head(
    assets_dir: str | Path,
    num_labels: int,
    cfg: TrainConfig,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Train a classifier head (and optionally some encoder layers) on preprocessed tensors.

    Returns (best_head_state, best_encoder_state_or_None).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    head_cfg = ClassifierHeadConfig(
        hidden_sizes=cfg.head_hidden_sizes,
        dropouts=cfg.head_dropouts,
        use_layer_norm=cfg.head_use_layer_norm,
        activation=cfg.head_activation,
    )

    model = BertTextClassifier.from_pretrained(
        assets_dir=assets_dir,
        num_labels=num_labels,
        pooling=cfg.pooling,
        head_config=head_cfg,
        map_location="cpu",
    ).to(device)

    # Apply finetune policy
    model.set_finetune_policy(
        policy=cfg.finetune_policy,
        last_n=cfg.finetune_last_n,
        train_embeddings=False,
    )
    any_unfrozen = any(p.requires_grad for p in model.bert.parameters())

    # Save original encoder snapshot if you want a pristine copy
    original_encoder_state = copy.deepcopy(model.bert.state_dict())

    # Build optimizer with different lrs for head vs encoder
    head_params, enc_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier."):
            head_params.append(p)
        else:
            enc_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"name": "head", "params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
            {"name": "encoder", "params": enc_params, "lr": cfg.lr_encoder, "weight_decay": cfg.weight_decay},
        ]
    )

    # Load data
    train = torch.load(cfg.train_pt, map_location="cpu")
    val = torch.load(cfg.val_pt, map_location="cpu")

    train_loader = _build_loader(train, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader   = _build_loader(val,   cfg.batch_size * 2, shuffle=False, seed=cfg.seed)

    # Simple schedule (you can paste in your warmup+cosine implementation here)
    total_steps = max(1, cfg.epochs * len(train_loader))
    warmup_steps = cfg.warmup_steps or int(cfg.warmup_ratio * total_steps)

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        t = min(max(0, step - warmup_steps), max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * t / max(1, total_steps - warmup_steps)))
        return cfg.eta_min_ratio + (1.0 - cfg.eta_min_ratio) * cos

    criterion = nn.CrossEntropyLoss()

    best_head_state: Dict[str, torch.Tensor] | None = None
    best_encoder_state: Dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            input_ids, token_type_ids, attention_mask, labels = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = out["loss"]
            scale = lr_scale(global_step)
            for pg in optimizer.param_groups:
                base_lr = pg.get("base_lr", pg["lr"])
                pg["lr"] = base_lr * scale
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            global_step += 1

        # simple val loop
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for batch in val_loader:
                input_ids, token_type_ids, attention_mask, labels = [
                    x.to(device) for x in batch
                ]
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )["logits"]
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(1, total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_head_state = {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()}
            if any_unfrozen:
                best_encoder_state = {k: v.detach().cpu().clone() for k, v in model.bert.state_dict().items()}

    if best_head_state is None:
        best_head_state = {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()}

    return best_head_state, best_encoder_state
