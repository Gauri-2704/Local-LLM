# local_llm/training/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

@dataclass
class TrainConfig:
    artifacts_root: Path = Path("./artifacts")

    batch_size: int = 128
    epochs: int = 30
    lr_head: float = 2e-4
    lr_encoder: float = 1e-5
    weight_decay: float = 0.01
    seed: int = 42

    scheduler: str = "cosine"
    warmup_ratio: float = 0.02
    warmup_steps: int = 0
    eta_min_ratio: float = 0.01
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    pooling: str = "cls"           # "cls" or "mean"
    finetune_policy: str = "none"  # "none", "last_n", "full"
    finetune_last_n: int = 0

    save_full_model: bool = False
    run_name: str = "run"

    # Data tensors (preprocessed)
    train_pt: Path = Path("./artifacts/train.pt")
    val_pt: Path   = Path("./artifacts/val.pt")
    label_map: Optional[Path] = Path("./artifacts/label_map.json")

    # Head configuration
    head_hidden_sizes: Sequence[int] = (768,)
    head_dropouts: Sequence[float] = (0.15, 0.20)
    head_use_layer_norm: bool = True
    head_activation: str = "gelu"
