# src/local_llm/training/text_finetune.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Tuple, Sequence, List, Literal, Optional

import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import text

from ..models.bert import BertConfig, BertModel
from ..pipelines.text_classification import (
    BertTextClassifier,
    ClassifierHeadConfig,
    build_bert_input_encoder,
)
from ..tokenization.bert_wordpiece import EncodeOutput


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class FineTuneConfig:
    # Data
    text_cols: Sequence[str]
    label_col: str

    # Splits
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42

    # Tokenizer / model paths
    assets_dir: Path = Path("./assets/bert-base-local")
    max_len: int = 256
    lowercase: bool = True

    # Training hyperparams
    batch_size: int = 32
    epochs: int = 5
    base_lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Finetune policy
    finetune_policy: Literal["none", "last_n", "full"] = "last_n"
    finetune_last_n: int = 2
    train_embeddings: bool = False

    # Pooling + head
    pooling: Literal["cls", "mean"] = "cls"
    head_config: ClassifierHeadConfig = field(default_factory=ClassifierHeadConfig)


    # Device / output
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: Path = Path("./artifacts/finetune_bert")

    # Logging
    run_name: str = "finetune_run"

    def resolved(self) -> "FineTuneConfig":
        """Resolve paths to Path objects, ensure output dir exists."""
        cfg = FineTuneConfig(**asdict(self))
        if not isinstance(cfg.assets_dir, Path):
            cfg.assets_dir = Path(cfg.assets_dir)
        if not isinstance(cfg.output_dir, Path):
            cfg.output_dir = Path(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: convert nested dict back to ClassifierHeadConfig
        if isinstance(cfg.head_config, dict):
            cfg.head_config = ClassifierHeadConfig(**cfg.head_config)
        return cfg


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# 1–3: Label mapping + simple text concatenation
# ---------------------------------------------------------------------

def prepare_label_mapping(
    df: pd.DataFrame,
    label_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Convert label column to categorical codes:
    returns (df_with_label_id, label_to_id, id_to_label).
    """
    if label_col not in df.columns:
        raise KeyError(f"label_col not found: {label_col}")
    df = df.copy()
    df[label_col] = df[label_col].astype("category")
    cats = list(df[label_col].cat.categories)
    label_to_id = {lab: i for i, lab in enumerate(cats)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    # Ensure label_id is a plain integer dtype, not categorical
    df["label_id"] = df[label_col].map(label_to_id).astype("int64")
    return df, label_to_id, id_to_label


def concat_text(row: pd.Series, text_cols: Sequence[str]) -> str:
    parts: List[str] = []
    for col in text_cols:
        val = row.get(col, "")
        if isinstance(val, str):
            v = val.strip()
        else:
            v = "" if pd.isna(val) else str(val)
        if v:
            parts.append(v)
    return " [SEP] ".join(parts) if parts else ""


# ---------------------------------------------------------------------
# 4: Stratified split
# ---------------------------------------------------------------------

def stratified_split_indices(
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays of indices (train_idx, val_idx, test_idx)
    with approximate label proportions preserved.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    all_indices = np.arange(len(labels))

    train_idx, val_idx, test_idx = [], [], []

    for lab in np.unique(labels):
        idx = all_indices[labels == lab]
        rng.shuffle(idx)
        n = len(idx)

        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # guard rails so every label ends up somewhere reasonable
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        n_train = max(n_train, 1 if n > 1 else 0)
        n_val = max(n_val, 0)

        tr = idx[:n_train]
        va = idx[n_train:n_train + n_val]
        te = idx[n_train + n_val:]

        train_idx.append(tr)
        val_idx.append(va)
        test_idx.append(te)

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------
# 5: Encode splits with BERT input encoder
# ---------------------------------------------------------------------

def build_input_encoder(cfg: FineTuneConfig):
    enc = build_bert_input_encoder(
        assets_dir=cfg.assets_dir,
        max_len=cfg.max_len,
        lowercase=cfg.lowercase,
    )
    return enc


def encode_dataframe(
    df: pd.DataFrame,
    text_cols: Sequence[str],
    encoder,
) -> Dict[str, torch.Tensor]:
    """
    Encode a split into tensor dict:
    - input_ids
    - token_type_ids
    - attention_mask
    - labels
    """
    input_ids_list: List[List[int]] = []
    token_type_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    labels_list: List[int] = []

    for _, row in df.iterrows():
        text = concat_text(row, text_cols)
        enc: EncodeOutput = encoder.encode(text)
        input_ids_list.append(enc.input_ids)
        token_type_ids_list.append(enc.token_type_ids)
        attention_mask_list.append(enc.attention_mask)
        labels_list.append(int(row["label_id"]))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def encode_splits(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: FineTuneConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Convenience wrapper:
    - builds encoder
    - encodes train/val/test splits
    """
    enc = build_input_encoder(cfg)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    train_tensors = encode_dataframe(train_df, cfg.text_cols, enc)
    val_tensors   = encode_dataframe(val_df, cfg.text_cols, enc)
    test_tensors  = encode_dataframe(test_df, cfg.text_cols, enc)

    return {
        "train": train_tensors,
        "val": val_tensors,
        "test": test_tensors,
    }


# ---------------------------------------------------------------------
# 6: Dataset / DataLoader
# ---------------------------------------------------------------------

class TensorDictDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        lens = {k: v.size(0) for k, v in data.items()}
        n = next(iter(lens.values()))
        if not all(L == n for L in lens.values()):
            raise ValueError("All tensors in data must have the same first dimension.")
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            self.data["input_ids"][idx],
            self.data["token_type_ids"][idx],
            self.data["attention_mask"][idx],
            self.data["labels"][idx],
        )


def build_dataloaders(
    splits: Dict[str, Dict[str, torch.Tensor]],
    cfg: FineTuneConfig,
) -> Dict[str, DataLoader]:
    train_ds = TensorDictDataset(splits["train"])
    val_ds   = TensorDictDataset(splits["val"])
    test_ds  = TensorDictDataset(splits["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch_size * 2),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, cfg.batch_size * 2),
        shuffle=False,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ---------------------------------------------------------------------
# 7: Load BERT + classifier from assets
# ---------------------------------------------------------------------

def build_bert_text_classifier_from_assets(
    cfg: FineTuneConfig,
    num_labels: int,
) -> BertTextClassifier:
    cfg_resolved = cfg.resolved()
    assets_dir = cfg_resolved.assets_dir
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
    bert_cfg = BertConfig(**{k: v for k, v in raw.items() if k in allowed})

    bert = BertModel(bert_cfg)
    sd = torch.load(weights_path, map_location="cpu")
    bert.load_state_dict(sd, strict=True)

    # finetune policy
    bert.set_finetune_policy(
        policy=cfg.finetune_policy,
        last_n=cfg.finetune_last_n,
        train_embeddings=cfg.train_embeddings,
    )

    model = BertTextClassifier(
        bert=bert,
        num_labels=num_labels,
        pooling=cfg.pooling,
        head_config=cfg.head_config,
    )
    device = torch.device(cfg.device)
    model.to(device)
    return model


# ---------------------------------------------------------------------
# 8: Training loop
# ---------------------------------------------------------------------

def run_epoch(
    model: BertTextClassifier,
    loader: DataLoader,
    cfg: FineTuneConfig,
    train: bool,
) -> Tuple[float, float]:
    """
    Returns (avg_loss, accuracy) for one epoch.
    """
    device = torch.device(cfg.device)
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    if train:
        # optimizer should be attached externally
        raise RuntimeError("run_epoch is internal; use train_text_classifier().")


def train_text_classifier(
    model: BertTextClassifier,
    loaders: Dict[str, DataLoader],
    cfg: FineTuneConfig,
) -> Tuple[List[Dict[str, float]], Dict[str, torch.Tensor]]:
    """
    Full training loop:
    - builds optimizer
    - runs epochs over train/val
    - returns (history, best_state_dict)
      where history is list of dicts with epoch metrics.
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device)

    # Build optimizer on trainable params only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.base_lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    def _run_phase(phase: str, epoch: int) -> Tuple[float, float]:
        is_train = (phase == "train")
        model.train(is_train)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        loader = loaders[phase]
        for batch in loader:
            input_ids, token_type_ids, attention_mask, labels = [
                b.to(device) for b in batch
            ]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = out["loss"]
            logits = out["logits"]
            preds = logits.argmax(dim=-1)

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        if total_seen == 0:
            return 0.0, 0.0
        avg_loss = total_loss / total_seen
        acc = total_correct / total_seen
        return avg_loss, acc

    for epoch in range(cfg.epochs):
        train_loss, train_acc = _run_phase("train", epoch)
        val_loss, val_acc = _run_phase("val", epoch)

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return history, best_state


# ---------------------------------------------------------------------
# 10: Inference on a split
# ---------------------------------------------------------------------

def evaluate_on_split(
    model: BertTextClassifier,
    split_tensors: Dict[str, torch.Tensor],
    cfg: FineTuneConfig,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Run inference on a split tensor dict.
    Returns:
        metrics: {"loss": float, "acc": float}
        preds_df: DataFrame with columns:
            - label_id (true)
            - pred_label_id
            - pred_confidence
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device)
    ds = TensorDictDataset(split_tensors)
    loader = DataLoader(ds, batch_size=max(1, cfg.batch_size * 2), shuffle=False)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids, token_type_ids, attention_mask, labels = [
                b.to(device) for b in batch
            ]
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = out["loss"]
            logits = out["logits"]
            probs = logits.softmax(dim=-1)
            max_probs, preds = probs.max(dim=-1)

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_conf.extend(max_probs.cpu().tolist())

    if total_seen == 0:
        metrics = {"loss": 0.0, "acc": 0.0}
    else:
        metrics = {"loss": total_loss / total_seen, "acc": total_correct / total_seen}

    preds_df = pd.DataFrame(
        {
            "label_id": all_true,
            "pred_label_id": all_pred,
            "pred_confidence": all_conf,
        }
    )
    return metrics, preds_df


# ---------------------------------------------------------------------
# 9 & 11: Save models and export predictions
# ---------------------------------------------------------------------

def save_finetuned_classifier(
    model: BertTextClassifier,
    best_state: Dict[str, torch.Tensor],
    cfg: FineTuneConfig,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> None:
    cfg = cfg.resolved()
    out_dir = cfg.output_dir

    # restore best_state into model before saving
    model.load_state_dict(best_state)

    # full classifier
    classifier_full_path = out_dir / "classifier_full.pt"
    torch.save(model.state_dict(), classifier_full_path)

    # finetuned encoder with same layout as base BERT
    finetuned_bert_path = out_dir / "pytorch_model_finetuned.bin"
    torch.save(model.bert.state_dict(), finetuned_bert_path)

    # copy vocab.txt into the finetuned directory so it is self-contained
    vocab_src = Path(cfg.assets_dir) / "vocab.txt"
    vocab_dst = out_dir / "vocab.txt"
    if not vocab_src.exists():
        raise FileNotFoundError(f"Expected vocab.txt at {vocab_src}")
    shutil.copy2(vocab_src, vocab_dst)

    # Append BERT config + head config + tokenizer setting into meta
    bert_cfg_dict = model.bert.config.__dict__

    # robust extraction of head_config
    hc = cfg.head_config
    if isinstance(hc, ClassifierHeadConfig):
        head_cfg_dict = asdict(hc)
    elif isinstance(hc, dict):
        head_cfg_dict = hc
    else:
        head_cfg_dict = dict(hc.__dict__)

    meta = {
        "num_labels": len(label_to_id),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "pooling": model.pooling,
        "finetune_policy": cfg.finetune_policy,
        "finetune_last_n": cfg.finetune_last_n,
        "train_embeddings": cfg.train_embeddings,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "base_lr": cfg.base_lr,
        "weight_decay": cfg.weight_decay,
        "max_len": cfg.max_len,
        "run_name": cfg.run_name,
        "lowercase": cfg.lowercase,
        "bert_config": bert_cfg_dict,
        "head_config": head_cfg_dict,
    }
    with open(out_dir / "finetune_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def export_predictions_csv(
    preds_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    id_to_label: Dict[int, str],
    cfg: FineTuneConfig,
    split_name: str = "test",
) -> Path:
    """
    Merge predictions with original rows (by index) and export to CSV.
    Assumes preds_df rows align with raw_df rows in order.
    """
    cfg = cfg.resolved()
    out = raw_df.copy().reset_index(drop=True)
    out["label_id"] = preds_df["label_id"]
    out["pred_label_id"] = preds_df["pred_label_id"]
    out["pred_label"] = out["pred_label_id"].map(id_to_label)
    out["pred_confidence"] = preds_df["pred_confidence"]
    path = cfg.output_dir / f"{split_name}_predictions.csv"
    out.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------
# Inference helpers: load metadata + config + model
# ---------------------------------------------------------------------

def load_finetune_meta(
    output_dir: Path | str,
) -> Tuple[Dict, Dict[str, int], Dict[int, str]]:
    """
    Load finetune_meta.json and return:
        meta, label_to_id, id_to_label

    This normalizes JSON string-keys back to Python types.
    """
    out_dir = Path(output_dir)
    meta_path = out_dir / "finetune_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"finetune_meta.json not found at: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw_l2i = meta.get("label_to_id", {})
    raw_i2l = meta.get("id_to_label", {})

    # labels are strings in JSON, ids are ints
    label_to_id: Dict[str, int] = {str(k): int(v) for k, v in raw_l2i.items()}
    id_to_label: Dict[int, str] = {int(k): str(v) for k, v in raw_i2l.items()}

    return meta, label_to_id, id_to_label


def build_inference_config_from_meta(
    meta: Dict,
    text_cols: Sequence[str],
    output_dir: Path | str,
) -> FineTuneConfig:
    """
    Construct a FineTuneConfig suitable for inference, using:
    - text_cols (provided explicitly)
    - hyperparameters stored in finetune_meta.json

    for inference, 'assets_dir' is the finetuned model directory itself, which contains vocab.txt

    Note: label_col is irrelevant for unlabeled inference, so we use a dummy name.
    """
    output_dir = Path(output_dir)
    assets_dir = output_dir               # vocab.txt is here now

    max_len = int(meta.get("max_len", 256))
    pooling = meta.get("pooling", "cls")
    finetune_policy = meta.get("finetune_policy", "last_n")
    finetune_last_n = int(meta.get("finetune_last_n", 2))
    lowercase = bool(meta.get("lowercase", True))

    cfg = FineTuneConfig(
        text_cols=tuple(text_cols),
        label_col="__inference_only__",  # unused
        assets_dir=output_dir,
        output_dir=output_dir,
        max_len=max_len,
        lowercase=lowercase,
        pooling=pooling,
        finetune_policy=finetune_policy,
        finetune_last_n=finetune_last_n,
    )
    return cfg


def load_finetuned_classifier_for_inference(
    output_dir: Path | str,
    text_cols: Sequence[str],
    device: str | torch.device | None = None,
) -> Tuple[BertTextClassifier, FineTuneConfig, Dict[str, int], Dict[int, str], Dict]:
    """
    High-level helper with a *self-contained* finetuned bundle for inference:
    
    Expects in `output_dir`:
        - finetune_meta.json
        - classifier_full.pt
        - vocab.txt

    - loads finetune_meta.json (labels, config)
    - builds a FineTuneConfig for inference (using meta + text_cols)
    - loads fine-tuned weights from classifier_full.pt

    Returns:
        model, cfg, label_to_id, id_to_label, meta
    """
    output_dir = Path(output_dir)

    # 1) load meta + label mappings
    meta, label_to_id, id_to_label = load_finetune_meta(output_dir)

    # 2) Build a FineTuneConfig that points assets_dir to the finetuned dir
    cfg = build_inference_config_from_meta(
        meta=meta,
        text_cols=text_cols,
        output_dir=output_dir,
    )

    if device is not None:
        cfg.device = device

    # 3) Rebuild BERT config and model from saved config dict
    bert_cfg_dict = meta["bert_config"]
    bert_cfg = BertConfig(**bert_cfg_dict)
    bert = BertModel(bert_cfg)


    # 4) Rebuild classifier head config
    head_cfg_dict = meta.get("head_config", {})
    head_cfg = ClassifierHeadConfig(**head_cfg_dict)

    num_labels = len(label_to_id)
    pooling = meta.get("pooling", "cls")

    # 5) Build the classifier skeleton
    model = BertTextClassifier(
        bert=bert, 
        num_labels=num_labels,
        pooling=pooling,
        head_config=head_cfg,
    )

    # 6) load fine-tuned weights (full classifier)
    classifier_full_path = output_dir / "classifier_full.pt"
    if not classifier_full_path.exists():
        raise FileNotFoundError(f"classifier_full.pt not found at: {classifier_full_path}")

    map_location = torch.device(cfg.device)
    state_dict = torch.load(classifier_full_path, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)

    model.to(map_location)
    model.eval()

    return model, cfg, label_to_id, id_to_label, meta


def encode_unlabeled_dataframe(
    df: pd.DataFrame,
    cfg: FineTuneConfig,
) -> Dict[str, torch.Tensor]:
    """
    Encode an unlabeled DataFrame into tensors suitable for inference:

    Returned dict has:
        - "input_ids"
        - "token_type_ids"
        - "attention_mask"

    Labels are not required or produced.
    """
    cfg_resolved = cfg.resolved()
    encoder = build_input_encoder(cfg_resolved)

    input_ids_list: List[List[int]] = []
    token_type_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []

    for _, row in df.iterrows():
        text = concat_text(row, cfg.text_cols)
        enc: EncodeOutput = encoder.encode(text)
        input_ids_list.append(enc.input_ids)
        token_type_ids_list.append(enc.token_type_ids)
        attention_mask_list.append(enc.attention_mask)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }


class UnlabeledTensorDataset(Dataset):
    """
    Dataset for unlabeled inference:
    yields (input_ids, token_type_ids, attention_mask).
    """
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.input_ids = data["input_ids"]
        self.token_type_ids = data["token_type_ids"]
        self.attention_mask = data["attention_mask"]

        n = self.input_ids.size(0)
        if self.token_type_ids.size(0) != n or self.attention_mask.size(0) != n:
            raise ValueError("All tensors must have the same first dimension.")
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return (
            self.input_ids[idx],
            self.token_type_ids[idx],
            self.attention_mask[idx],
        )


def predict_unlabeled_tensors(
    model: BertTextClassifier,
    unlabeled_tensors: Dict[str, torch.Tensor],
    cfg: FineTuneConfig,
    id_to_label: Dict[int, str],
) -> pd.DataFrame:
    """
    Run inference on unlabeled tensor dict and return a predictions DataFrame with:
        - pred_label_id
        - pred_label
        - pred_confidence
    """
    cfg = cfg.resolved()
    device = torch.device(cfg.device)

    ds = UnlabeledTensorDataset(unlabeled_tensors)
    loader = DataLoader(
        ds,
        batch_size=max(1, cfg.batch_size * 2),
        shuffle=False,
    )

    model.eval()

    all_pred_ids: List[int] = []
    all_labels: List[str] = []
    all_conf: List[float] = []

    with torch.no_grad():
        for batch in loader:
            input_ids, token_type_ids, attention_mask = [
                b.to(device) for b in batch
            ]
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,
            )
            logits = out["logits"]
            probs = logits.softmax(dim=-1)
            max_probs, pred_ids = probs.max(dim=-1)

            for pid, conf in zip(pred_ids.cpu().tolist(), max_probs.cpu().tolist()):
                all_pred_ids.append(int(pid))
                all_labels.append(id_to_label[int(pid)])
                all_conf.append(float(conf))

    preds_df = pd.DataFrame(
        {
            "pred_label_id": all_pred_ids,
            "pred_label": all_labels,
            "pred_confidence": all_conf,
        }
    )
    return preds_df


def merge_unlabeled_with_predictions(
    raw_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge raw unlabeled rows with predictions.
    Assumes row order is aligned.
    """
    out = raw_df.copy().reset_index(drop=True)
    if len(out) != len(preds_df):
        raise ValueError("raw_df and preds_df must have the same length.")
    out["pred_label_id"] = preds_df["pred_label_id"]
    out["pred_label"] = preds_df["pred_label"]
    out["pred_confidence"] = preds_df["pred_confidence"]
    return out


def export_unlabeled_predictions_csv(
    merged_df: pd.DataFrame,
    cfg: FineTuneConfig,
    filename: str = "unlabeled_predictions.csv",
) -> Path:
    """
    Save merged predictions for unlabeled data to CSV.
    """
    cfg = cfg.resolved()
    path = cfg.output_dir / filename
    merged_df.to_csv(path, index=False)
    return path

def is_connection_alive(engine) -> bool:
    """
    Check if database connection is alive and callable
    SQLalchemy and psycopg2
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print("Connection failed:", e)
        return False

def split_for_review(
    merged_df: pd.DataFrame, 
    conf_col: str, 
    threshold: float = 0.8
) -> pd.DataFrame:
    """
    Split merged predictions by prediction confidence and store
    in separate dataframe: auto, and for review. 
    """
    auto_df = merged_df[merged_df[conf_col] >= threshold].copy()
    review_df = merged_df[merged_df[conf_col] < threshold].copy()
    return auto_df, review_df