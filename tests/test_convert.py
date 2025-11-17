# tests/test_convert.py
"""
Tests for local_llm.convert

Covers:
- _assert_checkpoint_files_exist
- _load_bert_config
- _map_name
- convert_tf_bert_to_torch
- _derive_tf_prefix_from_dir
- setup_bert_base
- interactive_setup_bert_base
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import local_llm.convert as conv
from local_llm.models.bert import BertConfig, BertModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTFReader:
    """
    Minimal stand-in for tf.compat.v1.train.NewCheckpointReader used by tests.
    """
    def __init__(self, tensors: dict[str, np.ndarray]):
        self._tensors = tensors

    def get_variable_to_shape_map(self):
        return {name: list(arr.shape) for name, arr in self._tensors.items()}

    def get_tensor(self, name: str):
        return self._tensors[name]


# ---------------------------------------------------------------------------
# _assert_checkpoint_files_exist
# ---------------------------------------------------------------------------

def test_assert_checkpoint_files_exist_happy_path(tmp_path: Path):
    prefix = tmp_path / "bert_model.ckpt"
    index = prefix.with_suffix(".ckpt.index")  # careful: with_suffix replaces the suffix; we want ".index"
    # Fix: construct explicitly
    index = tmp_path / "bert_model.ckpt.index"
    data = tmp_path / "bert_model.ckpt.data-00000-of-00001"

    index.touch()
    data.touch()

    conv._assert_checkpoint_files_exist(prefix)  # should not raise


def test_assert_checkpoint_files_exist_with_alt_data_shard(tmp_path: Path):
    prefix = tmp_path / "bert_model.ckpt"
    index = tmp_path / "bert_model.ckpt.index"
    alt_data = tmp_path / "bert_model.ckpt.data-00001-of-00002"

    index.touch()
    alt_data.touch()

    conv._assert_checkpoint_files_exist(prefix)  # should not raise


def test_assert_checkpoint_files_exist_missing_index(tmp_path: Path):
    prefix = tmp_path / "bert_model.ckpt"
    data = tmp_path / "bert_model.ckpt.data-00000-of-00001"
    data.touch()

    with pytest.raises(FileNotFoundError):
        conv._assert_checkpoint_files_exist(prefix)


def test_assert_checkpoint_files_exist_missing_data(tmp_path: Path):
    prefix = tmp_path / "bert_model.ckpt"
    index = tmp_path / "bert_model.ckpt.index"
    index.touch()

    with pytest.raises(FileNotFoundError):
        conv._assert_checkpoint_files_exist(prefix)


# ---------------------------------------------------------------------------
# _load_bert_config
# ---------------------------------------------------------------------------

def test_load_bert_config_full_spec(tmp_path: Path):
    cfg_path = tmp_path / "bert_config.json"
    cfg_json = {
        "vocab_size": 12345,
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 1024,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.2,
        "attention_probs_dropout_prob": 0.3,
        "max_position_embeddings": 128,
        "type_vocab_size": 4,
        "layer_norm_eps": 1e-5,
    }
    cfg_path.write_text(json.dumps(cfg_json), encoding="utf-8")

    cfg = conv._load_bert_config(cfg_path)

    assert isinstance(cfg, BertConfig)
    assert cfg.vocab_size == 12345
    assert cfg.hidden_size == 256
    assert cfg.num_hidden_layers == 6
    assert cfg.num_attention_heads == 8
    assert cfg.intermediate_size == 1024
    assert cfg.hidden_act == "relu"
    assert cfg.hidden_dropout_prob == 0.2
    assert cfg.attention_probs_dropout_prob == 0.3
    assert cfg.max_position_embeddings == 128
    assert cfg.type_vocab_size == 4
    assert cfg.layer_norm_eps == 1e-5


def test_load_bert_config_defaults_for_optional_fields(tmp_path: Path):
    cfg_path = tmp_path / "bert_config.json"
    # omit vocab_size, hidden_act, dropout fields
    cfg_json = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
    }
    cfg_path.write_text(json.dumps(cfg_json), encoding="utf-8")

    cfg = conv._load_bert_config(cfg_path)

    assert cfg.vocab_size == 30522  # default
    assert cfg.hidden_act == "gelu"  # default
    assert cfg.hidden_dropout_prob == 0.1
    assert cfg.attention_probs_dropout_prob == 0.1
    assert cfg.layer_norm_eps == 1e-12


# ---------------------------------------------------------------------------
# _map_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tf_name, expected_torch, needs_t",
    [
        ("bert/embeddings/word_embeddings", "embeddings.word_embeddings.weight", False),
        ("bert/embeddings/token_type_embeddings", "embeddings.token_type_embeddings.weight", False),
        ("bert/embeddings/position_embeddings", "embeddings.position_embeddings.weight", False),
        ("bert/embeddings/LayerNorm/gamma", "embeddings.LayerNorm.weight", False),
        ("bert/embeddings/LayerNorm/beta", "embeddings.LayerNorm.bias", False),
        ("bert/pooler/dense/kernel", "pooler.dense.weight", True),
        ("bert/pooler/dense/bias", "pooler.dense.bias", False),
        # One attention example
        ("bert/encoder/layer_0/attention/self/query/kernel", "encoder.layer.0.attention.self.query.weight", True),
        ("bert/encoder/layer_11/output/LayerNorm/beta", "encoder.layer.11.output.LayerNorm.bias", False),
    ],
)
def test_map_name_known_mappings(tf_name, expected_torch, needs_t):
    torch_name, t = conv._map_name(tf_name)
    assert torch_name == expected_torch
    assert t is needs_t


def test_map_name_unmapped_returns_none():
    torch_name, t = conv._map_name("cls/predictions/output_bias")
    assert torch_name is None
    assert t is False


# ---------------------------------------------------------------------------
# convert_tf_bert_to_torch (with TF mocked)
# ---------------------------------------------------------------------------

def _build_fake_tf_reader_for_config(config: BertConfig) -> FakeTFReader:
    """
    Build a FakeTFReader with enough TF names to satisfy the 150-tensor check.
    Uses a real BertModel to get correct shapes.
    """
    model = BertModel(config)
    sd = model.state_dict()
    tensors: dict[str, np.ndarray] = {}

    # Embedding + pooler names
    tf_names = [
        "bert/embeddings/word_embeddings",
        "bert/embeddings/token_type_embeddings",
        "bert/embeddings/position_embeddings",
        "bert/embeddings/LayerNorm/gamma",
        "bert/embeddings/LayerNorm/beta",
        "bert/pooler/dense/kernel",
        "bert/pooler/dense/bias",
    ]

    layer_patterns = [
        "attention/self/query/kernel",
        "attention/self/query/bias",
        "attention/self/key/kernel",
        "attention/self/key/bias",
        "attention/self/value/kernel",
        "attention/self/value/bias",
        "attention/output/dense/kernel",
        "attention/output/dense/bias",
        "attention/output/LayerNorm/gamma",
        "attention/output/LayerNorm/beta",
        "intermediate/dense/kernel",
        "intermediate/dense/bias",
        "output/dense/kernel",
        "output/dense/bias",
        "output/LayerNorm/gamma",
        "output/LayerNorm/beta",
    ]

    for i in range(config.num_hidden_layers):
        for pat in layer_patterns:
            tf_names.append(f"bert/encoder/layer_{i}/{pat}")

    for tf_name in tf_names:
        torch_name, needs_t = conv._map_name(tf_name)
        if torch_name is None:
            continue
        assert torch_name in sd, f"Expected {torch_name} in state_dict"
        shape = sd[torch_name].shape

        # Important: for TF kernels (needs_t=True), TF stores [in, out],
        # while PyTorch expects [out, in]. Our fake TF tensor must be [in, out]
        # so that after transpose it matches the PyTorch shape.
        if needs_t and len(shape) == 2:
            shape = (shape[1], shape[0])

        tensors[tf_name] = np.random.randn(*shape).astype("float32")

    assert len(tensors) >= 150
    return FakeTFReader(tensors)


def test_convert_tf_bert_to_torch_happy_path(monkeypatch, tmp_path: Path):
    # small-ish config but still 12 layers so threshold 150 is satisfied
    fake_cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=12,
        num_attention_heads=4,
        intermediate_size=64,
    )

    # monkeypatch _load_bert_config to return our small config
    monkeypatch.setattr(conv, "_load_bert_config", lambda path: fake_cfg)

    # build fake reader with tensors consistent with that config
    fake_reader = _build_fake_tf_reader_for_config(fake_cfg)

    # monkeypatch TF reader + checkpoint existence check
    monkeypatch.setattr(conv.tf.compat.v1.train, "NewCheckpointReader", lambda prefix: fake_reader)
    monkeypatch.setattr(conv, "_assert_checkpoint_files_exist", lambda prefix: None)

    out_dir = tmp_path / "out"
    cfg_json_path = tmp_path / "dummy_config.json"
    cfg_json_path.write_text("{}", encoding="utf-8")  # unused because _load_bert_config is patched

    result_dir = conv.convert_tf_bert_to_torch(
        tf_checkpoint_prefix=tmp_path / "dummy_prefix",
        bert_config_json=cfg_json_path,
        output_dir=out_dir,
    )

    assert result_dir == out_dir
    model_bin = out_dir / "pytorch_model.bin"
    config_json = out_dir / "config.json"
    assert model_bin.is_file()
    assert config_json.is_file()

    # state_dict should load cleanly
    sd = torch.load(model_bin, map_location="cpu")
    assert isinstance(sd, dict)
    with open(config_json, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    assert "hidden_size" in cfg_data
    assert cfg_data["hidden_size"] == fake_cfg.hidden_size


def test_convert_tf_bert_to_torch_too_few_tensors(monkeypatch, tmp_path: Path):
    """
    Ensure we raise RuntimeError when fewer than 150 tensors load.
    """
    # config: only 1 layer so we can get <150 easily
    fake_cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
    )
    monkeypatch.setattr(conv, "_load_bert_config", lambda path: fake_cfg)

    model = BertModel(fake_cfg)
    sd = model.state_dict()

    # only provide embeddings (far less than 150)
    tensors = {}
    for tf_name in [
        "bert/embeddings/word_embeddings",
        "bert/embeddings/token_type_embeddings",
        "bert/embeddings/position_embeddings",
        "bert/embeddings/LayerNorm/gamma",
        "bert/embeddings/LayerNorm/beta",
    ]:
        torch_name, _ = conv._map_name(tf_name)
        shape = sd[torch_name].shape
        tensors[tf_name] = np.random.randn(*shape).astype("float32")

    fake_reader = FakeTFReader(tensors)
    monkeypatch.setattr(conv.tf.compat.v1.train, "NewCheckpointReader", lambda prefix: fake_reader)
    monkeypatch.setattr(conv, "_assert_checkpoint_files_exist", lambda prefix: None)

    out_dir = tmp_path / "out"
    cfg_json_path = tmp_path / "dummy_config.json"
    cfg_json_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError):
        conv.convert_tf_bert_to_torch(
            tf_checkpoint_prefix=tmp_path / "dummy_prefix",
            bert_config_json=cfg_json_path,
            output_dir=out_dir,
        )


# ---------------------------------------------------------------------------
# _derive_tf_prefix_from_dir
# ---------------------------------------------------------------------------

def test_derive_tf_prefix_prefers_bert_model(tmp_path: Path):
    dir_path = tmp_path / "uncased_L-12_H-768_A-12"
    dir_path.mkdir()
    (dir_path / "bert_model.ckpt.index").touch()
    (dir_path / "bert_model.ckpt.data-00000-of-00001").touch()

    prefix = conv._derive_tf_prefix_from_dir(dir_path)
    # Should be dir_path / "bert_model.ckpt" (no extension)
    assert prefix == dir_path / "bert_model.ckpt"


def test_derive_tf_prefix_first_ckpt_index_when_no_bert_model(tmp_path: Path):
    dir_path = tmp_path / "uncased"
    dir_path.mkdir()
    (dir_path / "foo.ckpt.index").touch()
    (dir_path / "foo.ckpt.data-00000-of-00001").touch()

    prefix = conv._derive_tf_prefix_from_dir(dir_path)
    assert prefix == dir_path / "foo.ckpt"


def test_derive_tf_prefix_raises_when_no_ckpt_index(tmp_path: Path):
    dir_path = tmp_path / "uncased"
    dir_path.mkdir()

    with pytest.raises(FileNotFoundError):
        conv._derive_tf_prefix_from_dir(dir_path)


# ---------------------------------------------------------------------------
# setup_bert_base
# ---------------------------------------------------------------------------

def test_setup_bert_base_from_checkpoints(monkeypatch, tmp_path: Path):
    """
    OPTION 1: starting from TF checkpoint directory.
    """
    root = tmp_path
    cp_dir = root / "uncased_L-12_H-768_A-12"
    cp_dir.mkdir()
    # minimal files needed for _derive_tf_prefix_from_dir
    (cp_dir / "bert_model.ckpt.index").touch()
    (cp_dir / "bert_model.ckpt.data-00000-of-00001").touch()

    vocab_src = cp_dir / "vocab.txt"
    vocab_src.write_text("foo\nbar\n", encoding="utf-8")

    cfg_src = cp_dir / "bert_config.json"
    cfg_src.write_text(json.dumps({"hidden_size": 32, "num_hidden_layers": 1,
                                   "num_attention_heads": 4, "intermediate_size": 64}), encoding="utf-8")

    assets_dir = root / "bert-base-local"

    # fake convert_tf_bert_to_torch to avoid TF dependency in this test
    def fake_convert(prefix, cfg_json, out_dir):
        # ensure inputs look right
        assert Path(prefix).parent == cp_dir
        assert Path(cfg_json) == cfg_src
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # write dummy model + config
        torch.save({"dummy": torch.tensor(1)}, out_dir / "pytorch_model.bin")
        (out_dir / "config.json").write_text(json.dumps({"from_converter": True}), encoding="utf-8")
        return out_dir

    monkeypatch.setattr(conv, "convert_tf_bert_to_torch", fake_convert)

    result_dir = conv.setup_bert_base(
        checkpoints=cp_dir,
        model_params=None,
        vocab=vocab_src,
        config=cfg_src,
        output_dir=assets_dir,
        overwrite=False,
    )

    assert result_dir == assets_dir
    assert (assets_dir / "pytorch_model.bin").is_file()
    assert (assets_dir / "vocab.txt").is_file()
    assert (assets_dir / "config.json").is_file()

    # config.json should match cfg_src contents (setup_bert_base overwrites converter's dummy config)
    with open(assets_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_loaded = json.load(f)
    assert "hidden_size" in cfg_loaded
    assert cfg_loaded["hidden_size"] == 32

    assert (assets_dir / "vocab.txt").read_text(encoding="utf-8") == vocab_src.read_text(encoding="utf-8")


def test_setup_bert_base_from_model_params(tmp_path: Path):
    """
    OPTION 2: starting from an existing PyTorch .bin
    """
    root = tmp_path
    src_dir = root / "src_assets"
    src_dir.mkdir()

    model_src = src_dir / "pytorch_model.bin"
    torch.save({"dummy": torch.tensor([1, 2, 3])}, model_src)

    vocab_src = src_dir / "vocab.txt"
    vocab_src.write_text("foo\nbar\n", encoding="utf-8")

    cfg_src = src_dir / "config.json"
    cfg_src.write_text(json.dumps({"hidden_size": 64, "num_hidden_layers": 2,
                                   "num_attention_heads": 4, "intermediate_size": 128}), encoding="utf-8")

    assets_dir = root / "bert-base-local-from-bin"

    result_dir = conv.setup_bert_base(
        checkpoints=None,
        model_params=model_src,
        vocab=vocab_src,
        config=cfg_src,
        output_dir=assets_dir,
        overwrite=False,
    )

    assert result_dir == assets_dir
    target_model = assets_dir / "pytorch_model.bin"
    target_vocab = assets_dir / "vocab.txt"
    target_cfg = assets_dir / "config.json"

    assert target_model.is_file()
    assert target_vocab.is_file()
    assert target_cfg.is_file()

    # content equality for model file (bytes)
    assert target_model.read_bytes() == model_src.read_bytes()
    assert target_vocab.read_text(encoding="utf-8") == vocab_src.read_text(encoding="utf-8")
    assert json.loads(target_cfg.read_text(encoding="utf-8"))["hidden_size"] == 64


def test_setup_bert_base_requires_exactly_one_mode(tmp_path: Path):
    vocab = tmp_path / "vocab.txt"
    cfg = tmp_path / "config.json"
    vocab.write_text("x\ny\n", encoding="utf-8")
    cfg.write_text(json.dumps({"hidden_size": 32, "num_hidden_layers": 1,
                               "num_attention_heads": 4, "intermediate_size": 64}), encoding="utf-8")

    # neither provided
    with pytest.raises(ValueError):
        conv.setup_bert_base(
            checkpoints=None,
            model_params=None,
            vocab=vocab,
            config=cfg,
        )

    # both provided
    with pytest.raises(ValueError):
        conv.setup_bert_base(
            checkpoints=tmp_path / "cp_dir",
            model_params=tmp_path / "model.bin",
            vocab=vocab,
            config=cfg,
        )


def test_setup_bert_base_missing_vocab_or_config(tmp_path: Path):
    vocab = tmp_path / "vocab.txt"
    cfg = tmp_path / "config.json"
    # create only vocab
    vocab.write_text("foo\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        conv.setup_bert_base(
            checkpoints=tmp_path / "cp",
            model_params=None,
            vocab=vocab,
            config=cfg,
        )

    # create config only
    cfg.write_text("{}", encoding="utf-8")
    os.remove(vocab)

    with pytest.raises(FileNotFoundError):
        conv.setup_bert_base(
            checkpoints=tmp_path / "cp",
            model_params=None,
            vocab=vocab,
            config=cfg,
        )


def test_setup_bert_base_existing_assets_without_overwrite(tmp_path: Path):
    """
    If config.json / vocab.txt already exist and point to different files, setup_bert_base
    should raise FileExistsError when overwrite=False.
    """
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    # existing different config and vocab
    existing_cfg = assets_dir / "config.json"
    existing_cfg.write_text(json.dumps({"existing": True}), encoding="utf-8")
    existing_vocab = assets_dir / "vocab.txt"
    existing_vocab.write_text("old\n", encoding="utf-8")

    # new src files
    vocab_src = tmp_path / "vocab_src.txt"
    cfg_src = tmp_path / "cfg_src.json"
    vocab_src.write_text("new\n", encoding="utf-8")
    cfg_src.write_text(json.dumps({"hidden_size": 32, "num_hidden_layers": 1,
                                   "num_attention_heads": 4, "intermediate_size": 64}), encoding="utf-8")

    # model_params path (exists)
    model_src = tmp_path / "model.bin"
    torch.save({"dummy": torch.tensor([1])}, model_src)

    with pytest.raises(FileExistsError):
        conv.setup_bert_base(
            checkpoints=None,
            model_params=model_src,
            vocab=vocab_src,
            config=cfg_src,
            output_dir=assets_dir,
            overwrite=False,
        )


def test_setup_bert_base_reuse_same_files_without_copy(tmp_path: Path):
    """
    If config/vocab already live in output_dir and we point config/vocab to those same paths,
    there should be no error and no extra copy.
    """
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    cfg_path = assets_dir / "config.json"
    cfg_path.write_text(json.dumps({"hidden_size": 32, "num_hidden_layers": 1,
                                    "num_attention_heads": 4, "intermediate_size": 64}), encoding="utf-8")

    vocab_path = assets_dir / "vocab.txt"
    vocab_path.write_text("foo\nbar\n", encoding="utf-8")

    model_src = tmp_path / "model.bin"
    torch.save({"dummy": torch.tensor([1])}, model_src)

    result_dir = conv.setup_bert_base(
        checkpoints=None,
        model_params=model_src,
        vocab=vocab_path,
        config=cfg_path,
        output_dir=assets_dir,
        overwrite=False,
    )

    assert result_dir == assets_dir
    assert (assets_dir / "pytorch_model.bin").is_file()
    # config.json and vocab.txt still present and unchanged
    assert json.loads(cfg_path.read_text(encoding="utf-8"))["hidden_size"] == 32
    assert vocab_path.read_text(encoding="utf-8").startswith("foo")


# ---------------------------------------------------------------------------
# interactive_setup_bert_base
# ---------------------------------------------------------------------------

def test_interactive_setup_bert_base_delegates_to_setup(monkeypatch, tmp_path: Path):
    """
    Verify that interactive_setup_bert_base:
      - prompts for inputs
      - forwards checkpoints/config/vocab/output_dir to setup_bert_base
      - returns whatever setup_bert_base returns
    """
    # fake user input
    cp_dir = tmp_path / "cp_dir"
    cp_dir.mkdir()
    cfg_path = tmp_path / "bert_config.json"
    vocab_path = tmp_path / "vocab.txt"
    cfg_path.write_text("{}", encoding="utf-8")
    vocab_path.write_text("foo\n", encoding="utf-8")

    # preset answers for input()
    answers = [
        str(cp_dir),
        str(cfg_path),
        str(vocab_path),
    ]

    def fake_input(prompt: str) -> str:
        return answers.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)

    # capture arguments passed to setup_bert_base
    called = {}

    def fake_setup_bert_base(**kwargs):
        called.update(kwargs)
        return Path(kwargs["output_dir"])

    monkeypatch.setattr(conv, "setup_bert_base", fake_setup_bert_base)

    out_dir = tmp_path / "assets"
    result = conv.interactive_setup_bert_base(output_dir=out_dir)

    assert result == out_dir
    assert called["checkpoints"] == str(cp_dir)
    assert called["vocab"] == str(vocab_path)
    assert called["config"] == str(cfg_path)
    assert called["output_dir"] == out_dir
    assert called["overwrite"] is False
