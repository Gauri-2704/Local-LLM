# local_llm/convert.py
from __future__ import annotations

import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import tensorflow as tf

from .models.bert import BertConfig, BertModel

# ---------------------------------------------------------------------------
# TF noise handling
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", message="Protobuf gencode version .* older than the runtime version .*")
tf.compat.v1.disable_eager_execution()


# ---------------------------------------------------------------------------
# Low-level helpers (unchanged core converter)
# ---------------------------------------------------------------------------
def _assert_checkpoint_files_exist(prefix: str | Path) -> None:
    """
    Guard that a TF1 checkpoint prefix is valid.

    Expects:
        <prefix>.index
        <prefix>.data-00000-of-00001   or other *.data-* shard(s)
    """
    prefix = str(prefix)
    index = prefix + ".index"
    data = prefix + ".data-00000-of-00001"

    has_index = os.path.isfile(index)
    has_data = os.path.isfile(data) or any(
        f.startswith(os.path.basename(prefix) + ".data-")
        for f in os.listdir(os.path.dirname(prefix) or ".")
    )
    if not (has_index and has_data):
        raise FileNotFoundError(
            "TensorFlow checkpoint files not found for prefix:\n"
            f"  prefix: {prefix}\n"
            f"  looked for: {index} and {data} (or any data shard)\n"
            "Double-check the checkpoint prefix (no extension)."
        )


def _load_bert_config(path: str | Path) -> BertConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return BertConfig(
        vocab_size=d.get("vocab_size", 30522),
        hidden_size=d["hidden_size"],
        num_hidden_layers=d["num_hidden_layers"],
        num_attention_heads=d["num_attention_heads"],
        intermediate_size=d["intermediate_size"],
        hidden_act=d.get("hidden_act", "gelu"),
        hidden_dropout_prob=d.get("hidden_dropout_prob", 0.1),
        attention_probs_dropout_prob=d.get("attention_probs_dropout_prob", 0.1),
        max_position_embeddings=d.get("max_position_embeddings", 512),
        type_vocab_size=d.get("type_vocab_size", 2),
        layer_norm_eps=d.get("layer_norm_eps", 1e-12),
    )


def _map_name(tf_name: str) -> Tuple[Optional[str], bool]:
    # Embeddings
    if tf_name == "bert/embeddings/word_embeddings":
        return "embeddings.word_embeddings.weight", False
    if tf_name == "bert/embeddings/token_type_embeddings":
        return "embeddings.token_type_embeddings.weight", False
    if tf_name == "bert/embeddings/position_embeddings":
        return "embeddings.position_embeddings.weight", False
    if tf_name == "bert/embeddings/LayerNorm/gamma":
        return "embeddings.LayerNorm.weight", False
    if tf_name == "bert/embeddings/LayerNorm/beta":
        return "embeddings.LayerNorm.bias", False

    # Encoder layers
    m = re.match(r"bert/encoder/layer_(\d+)/(.*)", tf_name)
    if m:
        i = int(m.group(1))
        rest = m.group(2)
        p = f"encoder.layer.{i}"
        mp = {
            "attention/self/query/kernel": (f"{p}.attention.self.query.weight", True),
            "attention/self/query/bias":   (f"{p}.attention.self.query.bias", False),
            "attention/self/key/kernel":   (f"{p}.attention.self.key.weight", True),
            "attention/self/key/bias":     (f"{p}.attention.self.key.bias", False),
            "attention/self/value/kernel": (f"{p}.attention.self.value.weight", True),
            "attention/self/value/bias":   (f"{p}.attention.self.value.bias", False),

            "attention/output/dense/kernel": (f"{p}.attention.output.dense.weight", True),
            "attention/output/dense/bias":   (f"{p}.attention.output.dense.bias", False),
            "attention/output/LayerNorm/gamma": (f"{p}.attention.output.LayerNorm.weight", False),
            "attention/output/LayerNorm/beta":  (f"{p}.attention.output.LayerNorm.bias", False),

            "intermediate/dense/kernel": (f"{p}.intermediate.dense.weight", True),
            "intermediate/dense/bias":   (f"{p}.intermediate.dense.bias", False),

            "output/dense/kernel":       (f"{p}.output.dense.weight", True),
            "output/dense/bias":         (f"{p}.output.dense.bias", False),
            "output/LayerNorm/gamma":    (f"{p}.output.LayerNorm.weight", False),
            "output/LayerNorm/beta":     (f"{p}.output.LayerNorm.bias", False),
        }
        if rest in mp:
            return mp[rest]

    # Pooler
    if tf_name == "bert/pooler/dense/kernel":
        return "pooler.dense.weight", True
    if tf_name == "bert/pooler/dense/bias":
        return "pooler.dense.bias", False

    # everything else (pretraining heads, optimizer slots) is ignored
    return None, False


def convert_tf_bert_to_torch(
    tf_checkpoint_prefix: str | Path,
    bert_config_json: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Convert a TF1 BERT checkpoint into a local PyTorch `pytorch_model.bin` +
    `config.json` pair under `output_dir`. No internet, no HuggingFace.
    """
    tf_checkpoint_prefix = Path(tf_checkpoint_prefix)
    bert_config_json = Path(bert_config_json)
    output_dir = Path(output_dir)

    _assert_checkpoint_files_exist(tf_checkpoint_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_bert_config(bert_config_json)
    model = BertModel(cfg)
    sd = model.state_dict()

    reader = tf.compat.v1.train.NewCheckpointReader(str(tf_checkpoint_prefix))
    varmap = reader.get_variable_to_shape_map()

    loaded = 0
    skipped = 0

    for name in sorted(varmap.keys()):
        if name.startswith("global_step"):
            continue

        torch_name, needs_t = _map_name(name)
        if torch_name is None:
            if name.startswith("bert/encoder/layer_"):
                print("UNMAPPED:", name)
            skipped += 1
            continue

        arr = reader.get_tensor(name)
        pt = torch.from_numpy(arr)
        if needs_t and pt.ndim == 2:
            pt = pt.t()

        if torch_name not in sd:
            skipped += 1
            continue

        if tuple(sd[torch_name].shape) != tuple(pt.shape):
            raise ValueError(
                f'Shape mismatch for "{torch_name}": expected '
                f"{tuple(sd[torch_name].shape)} but got {tuple(pt.shape)} "
                f'from TF variable "{name}".'
            )

        sd[torch_name] = pt
        loaded += 1

    if loaded < 150:
        raise RuntimeError(
            f"Too few tensors loaded ({loaded}). Name mapping likely broken."
        )

    model.load_state_dict(sd, strict=True)

    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"[convert] Loaded tensors: {loaded} | Skipped: {skipped}")
    print(f"[convert] Wrote: {output_dir / 'pytorch_model.bin'}")

    return output_dir


# ---------------------------------------------------------------------------
# High-level setup API (what you actually want)
# ---------------------------------------------------------------------------
def _derive_tf_prefix_from_dir(checkpoints_dir: Path) -> Path:
    """
    Given a directory containing TF checkpoints (e.g. uncased_L-12_H-768_A-12),
    find a usable checkpoint *prefix* (no extension).

    Priority:
        1) <dir>/bert_model.ckpt if it exists
        2) first *.ckpt.index found, drop the .index suffix
    """
    checkpoints_dir = checkpoints_dir.resolve()
    if not checkpoints_dir.is_dir():
        raise NotADirectoryError(f"checkpoints path is not a directory: {checkpoints_dir}")

    # Common Google BERT layout
    candidate = checkpoints_dir / "bert_model.ckpt"
    if (checkpoints_dir / "bert_model.ckpt.index").is_file():
        return candidate

    # Generic fallback: first *.ckpt.index
    for idx_file in checkpoints_dir.glob("*.ckpt.index"):
        # "something.ckpt.index" -> "something.ckpt"
        return idx_file.with_suffix("")

    raise FileNotFoundError(
        f"No *.ckpt.index checkpoint found in directory: {checkpoints_dir}"
    )


def setup_bert_base(
    *,
    checkpoints: str | Path | None = None,
    model_params: str | Path | None = None,
    vocab: str | Path,
    config: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Canonical setup function for a *local* BERT base model.

    Exactly one of `checkpoints` or `model_params` must be provided::

        # OPTION 1: start from Google's TF checkpoint folder
        setup_bert_base(
            checkpoints=".../uncased_L-12_H-768_A-12/",
            vocab=".../uncased_L-12_H-768_A-12/vocab.txt",
            config=".../uncased_L-12_H-768_A-12/bert_config.json",
        )

        # OPTION 2: start from an existing PyTorch .bin
        setup_bert_base(
            model_params=".../bert-base-local/pytorch_model.bin",
            vocab=".../bert-base-local/vocab.txt",
            config=".../bert-base-local/config.json",
        )

    Returns
    -------
    Path
        Directory containing the three canonical assets:

            assets_dir / "pytorch_model.bin"
            assets_dir / "config.json"
            assets_dir / "vocab.txt"
    """
    vocab_path = Path(vocab).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()

    if not vocab_path.is_file():
        raise FileNotFoundError(f"vocab file not found: {vocab_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config json not found: {config_path}")

    mode_from_tf = checkpoints is not None
    mode_from_bin = model_params is not None

    if mode_from_tf == mode_from_bin:
        raise ValueError(
            "Exactly one of `checkpoints` or `model_params` must be set "
            "(checkpoints=... for TF → Torch; model_params=... for existing .bin)."
        )

    if output_dir is not None:
        assets_dir = Path(output_dir).expanduser().resolve()
    else:
        if mode_from_tf:
            cp = Path(checkpoints).expanduser().resolve()
            # If you pass the folder, we default to sibling 'bert-base-local'
            base_dir = cp if cp.is_dir() else cp.parent
            assets_dir = base_dir.parent / "bert-base-local"
        else:
            mp = Path(model_params).expanduser().resolve()
            assets_dir = mp.parent

    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1) Handle model_params / TF conversion
    if mode_from_tf:
        cp = Path(checkpoints).expanduser().resolve()
        if cp.is_dir():
            tf_prefix = _derive_tf_prefix_from_dir(cp)
        else:
            # allow passing a prefix directly; strip any known extension if needed
            tf_prefix = cp
            for ext in (".index", ".data-00000-of-00001", ".meta"):
                if tf_prefix.name.endswith(ext):
                    tf_prefix = tf_prefix.with_name(tf_prefix.name[: -len(ext)])
                    break

        print(f"[setup] Using TF checkpoint prefix: {tf_prefix}")
        # convert_tf_bert_to_torch writes pytorch_model.bin + config.json into assets_dir
        convert_tf_bert_to_torch(tf_prefix, config_path, assets_dir)
    else:
        mp = Path(model_params).expanduser().resolve()
        if not mp.is_file():
            raise FileNotFoundError(f"model_params file not found: {mp}")
        target_bin = assets_dir / "pytorch_model.bin"
        if target_bin.exists() and not overwrite and target_bin.resolve() != mp:
            raise FileExistsError(
                f"Target model file already exists: {target_bin}\n"
                "Use overwrite=True if you intend to replace it."
            )
        if target_bin.resolve() != mp:
            shutil.copy2(mp, target_bin)
            print(f"[setup] Copied model params → {target_bin}")
        else:
            print(f"[setup] Reusing existing model params at {target_bin}")

    # 2) Ensure config.json is in assets_dir
    target_cfg = assets_dir / "config.json"
    if target_cfg.exists():
        # Already there
        if target_cfg.resolve() != config_path:
            # Different physical file
            if mode_from_tf or overwrite:
                # In TF mode, we always treat `config` as source-of-truth
                # and overwrite whatever the converter may have written.
                # In bin mode, allow overwrite=True to replace.
                shutil.copy2(config_path, target_cfg)
                print(f"[setup] Copied config → {target_cfg}")
            else:
                # Bin mode, overwrite=False, different file → refuse
                raise FileExistsError(
                    f"Target config.json already exists: {target_cfg}\n"
                    "Use overwrite=True if you intend to replace it."
                )
        else:
            # Same path, nothing to do
            print(f"[setup] Reusing existing config at {target_cfg}")
    else:
        # No config yet, just copy
        shutil.copy2(config_path, target_cfg)
        print(f"[setup] Copied config → {target_cfg}")


    # 3) Ensure vocab.txt is in assets_dir
    target_vocab = assets_dir / "vocab.txt"
    if target_vocab.exists() and not overwrite and target_vocab.resolve() != vocab_path:
        raise FileExistsError(
            f"Target vocab.txt already exists: {target_vocab}\n"
            "Use overwrite=True if you intend to replace it."
        )
    if target_vocab.resolve() != vocab_path:
        shutil.copy2(vocab_path, target_vocab)
        print(f"[setup] Copied vocab → {target_vocab}")
    else:
        print(f"[setup] Reusing existing vocab at {target_vocab}")

    return assets_dir


# ---------------------------------------------------------------------------
# Backwards-compatible interactive helper
# ---------------------------------------------------------------------------
def interactive_setup_bert_base(
    output_dir: str | Path = "./assets/bert-base-local",
) -> Path:
    """
    Interactive wrapper around `setup_bert_base`, mainly for quick notebook use.

    Prompts for:
        - TF checkpoint directory or prefix
        - bert_config.json path
        - vocab.txt path
    """
    output_dir = Path(output_dir)

    print("=== local-llm: BERT base setup (TF → PyTorch, fully offline) ===")
    cp_raw = input("Path to TF checkpoint directory or prefix (e.g. .../uncased_L-12_H-768_A-12 or .../bert_model.ckpt): ").strip()
    cfg_path = input("Path to bert_config.json: ").strip()
    vocab_path = input("Path to vocab.txt: ").strip()

    return setup_bert_base(
        checkpoints=cp_raw,
        vocab=vocab_path,
        config=cfg_path,
        output_dir=output_dir,
        overwrite=False,
    )
