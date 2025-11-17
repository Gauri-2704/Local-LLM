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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", message="Protobuf gencode version .* older than the runtime version .*")
tf.compat.v1.disable_eager_execution()


def _assert_checkpoint_files_exist(prefix: str | Path) -> None:
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


def interactive_setup_bert_base(
    output_dir: str | Path = "./assets/bert-base-local",
    copy_vocab: bool = True,
) -> Path:
    """
    Small helper intended for Jupyter:

    - Prompts for:
        * TF checkpoint prefix      (e.g. /path/to/bert_model.ckpt)
        * bert_config.json path
        * vocab.txt path
    - Runs conversion
    - Copies vocab.txt into output_dir (if copy_vocab=True)
    """
    output_dir = Path(output_dir)

    print("=== local-llm: BERT base setup (TF → PyTorch, fully offline) ===")
    tf_prefix = input("Path prefix to TF checkpoint (no extension, e.g. /.../bert_model.ckpt): ").strip()
    cfg_path = input("Path to bert_config.json: ").strip()
    vocab_path = input("Path to vocab.txt: ").strip()

    out = convert_tf_bert_to_torch(tf_prefix, cfg_path, output_dir)

    if copy_vocab:
        vp = Path(vocab_path)
        if not vp.exists():
            raise FileNotFoundError(f"vocab.txt not found at: {vp}")
        shutil.copy2(vp, out / "vocab.txt")
        print(f"[convert] Copied vocab.txt to {out / 'vocab.txt'}")

    print(f"[convert] local-llm BERT base assets ready at: {out}")
    return out
