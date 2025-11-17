# tests/test_bert_wordpiece.py
"""
Tests for local_llm.tokenization.bert_wordpiece

Covers:
- SPECIAL_TOKENS
- load_vocab
- BasicTokenizer
- WordPieceTokenizer
- EncodeOutput
- BertInputEncoder
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_llm.tokenization import bert_wordpiece as bw


# ---------------------------------------------------------------------------
# SPECIAL_TOKENS
# ---------------------------------------------------------------------------

def test_special_tokens_contains_required_entries():
    required = {"pad", "unk", "cls", "sep", "mask"}
    assert isinstance(bw.SPECIAL_TOKENS, dict)
    assert required.issubset(bw.SPECIAL_TOKENS.keys())
    # Check that they look like BERT-style brackets
    for key in required:
        tok = bw.SPECIAL_TOKENS[key]
        assert tok.startswith("[") and tok.endswith("]")


# ---------------------------------------------------------------------------
# load_vocab
# ---------------------------------------------------------------------------

def test_load_vocab_happy_path(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    # lines: index = line number (0-based)
    lines = [
        "[PAD]\n",
        "[UNK]\n",
        "[CLS]\n",
        "[SEP]\n",
        "[MASK]\n",
        "hello\n",
        "world\n",
    ]
    vocab_path.write_text("".join(lines), encoding="utf-8")

    vocab = bw.load_vocab(vocab_path)
    assert isinstance(vocab, dict)
    # indices should follow line numbers
    for i, line in enumerate(lines):
        tok = line.rstrip("\n")
        assert vocab[tok] == i

    # required tokens must be present
    for key in ("pad", "unk", "cls", "sep", "mask"):
        assert bw.SPECIAL_TOKENS[key] in vocab


def test_load_vocab_ignores_blank_lines(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "[PAD]\n\n[UNK]\n  \n[CLS]\n[SEP]\n[MASK]\nfoo\n",
        encoding="utf-8",
    )

    vocab = bw.load_vocab(vocab_path)
    # blank / whitespace lines do not become tokens
    assert "" not in vocab
    assert " " not in vocab

    # but indices still correspond to non-empty line order
    assert vocab["[PAD]"] == 0
    assert vocab["foo"] == 6 - 1  # last non-empty line


def test_load_vocab_missing_special_tokens_raises(tmp_path: Path):
    vocab_path = tmp_path / "vocab.txt"
    # Deliberately omit [MASK]
    vocab_path.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\nfoo\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        bw.load_vocab(vocab_path)

    msg = str(exc.value)
    assert "vocab.txt missing required tokens" in msg
    assert "[MASK]" in msg


# ---------------------------------------------------------------------------
# BasicTokenizer
# ---------------------------------------------------------------------------

def test_basic_tokenizer_lowercases_and_strips_accents():
    bt = bw.BasicTokenizer(lower=True)
    text = "Café À LA CARTE"
    toks = bt.tokenize(text)
    # "cafe", "a", "la", "carte" in some segmentation
    joined = " ".join(toks)
    assert "cafe" in joined
    assert "café" not in joined
    assert joined == joined.lower()


def test_basic_tokenizer_no_lower_when_disabled():
    bt = bw.BasicTokenizer(lower=False)
    text = "Hello World"
    toks = bt.tokenize(text)
    assert "Hello" in toks
    assert "hello" not in toks


def test_basic_tokenizer_punctuation_separation():
    bt = bw.BasicTokenizer(lower=True)
    text = "Hello,world! (Test)"
    toks = bt.tokenize(text)
    # Expect punctuation split out as separate tokens
    assert "hello" in toks
    assert "world" in toks
    assert "," in toks
    assert "!" in toks
    assert "(" in toks
    assert ")" in toks


def test_basic_tokenizer_handles_multiple_spaces_and_tabs():
    bt = bw.BasicTokenizer(lower=True)
    text = "Hello   \t world"
    toks = bt.tokenize(text)
    # No empty tokens
    assert toks == ["hello", "world"]


def test_basic_tokenizer_protected_tokens_preserved_case_and_position():
    bt = bw.BasicTokenizer(lower=True)
    text = "Hello [SEP] WORLD [CLS] test"
    toks = bt.tokenize(text)
    # protected tokens should appear verbatim, not lowercased
    assert "[SEP]" in toks
    assert "[CLS]" in toks
    # but non-protected words are lowercased
    assert "hello" in toks
    assert "WORLD" not in toks
    assert "world" in toks
    # order preserved
    assert toks == ["hello", "[SEP]", "world", "[CLS]", "test"]


def test_basic_tokenizer_empty_and_none():
    bt = bw.BasicTokenizer(lower=True)
    assert bt.tokenize("") == []
    # type-wise, it expects a str; avoid passing None, but we can enforce it crashes clearly
    with pytest.raises(AttributeError):
        bt.tokenize(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# WordPieceTokenizer
# ---------------------------------------------------------------------------

def _make_simple_vocab():
    """
    Small vocab for WordPiece tests.
    """
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "hello": 5,
        "world": 6,
        "##s": 7,
        "##ly": 8,
        "great": 9,
    }
    return vocab


def test_wordpiece_tokenizer_exact_match():
    vocab = _make_simple_vocab()
    wpt = bw.WordPieceTokenizer(vocab)
    pieces = wpt.tokenize(["hello", "world"])
    assert pieces == ["hello", "world"]


def test_wordpiece_tokenizer_greedy_longest_match():
    vocab = _make_simple_vocab()
    wpt = bw.WordPieceTokenizer(vocab)
    pieces = wpt.tokenize(["hellos"])
    # "hello" + "##s"
    assert pieces == ["hello", "##s"]


def test_wordpiece_tokenizer_unknown_token_when_no_split_possible():
    vocab = _make_simple_vocab()
    wpt = bw.WordPieceTokenizer(vocab)
    pieces = wpt.tokenize(["xyz"])
    assert pieces == [bw.SPECIAL_TOKENS["unk"]]


def test_wordpiece_tokenizer_mixed_known_and_unknown():
    vocab = _make_simple_vocab()
    wpt = bw.WordPieceTokenizer(vocab)
    pieces = wpt.tokenize(["hello", "xyz", "world"])
    assert pieces == [
        "hello",
        bw.SPECIAL_TOKENS["unk"],
        "world",
    ]


def test_wordpiece_tokenizer_respects_max_input_chars_per_word():
    vocab = _make_simple_vocab()
    wpt = bw.WordPieceTokenizer(vocab, max_input_chars_per_word=5)
    # 6 chars > 5 → should emit [UNK] directly
    pieces = wpt.tokenize(["abcdef"])
    assert pieces == [bw.SPECIAL_TOKENS["unk"]]


def test_wordpiece_convert_tokens_to_ids_uses_unk_for_missing():
    vocab = _make_simple_vocab()
    wpt = bw.WordPieceTokenizer(vocab)
    pieces = ["hello", "##s", "nope"]
    ids = wpt.convert_tokens_to_ids(pieces)
    assert ids[0] == vocab["hello"]
    assert ids[1] == vocab["##s"]
    assert ids[2] == vocab["[UNK]"]


# ---------------------------------------------------------------------------
# EncodeOutput + BertInputEncoder
# ---------------------------------------------------------------------------

def _build_vocab_for_encoder():
    """
    Build a tiny vocab consistent with SPECIAL_TOKENS + a few tokens/subwords.
    """
    # maintain deterministic indices; load_vocab isn't used here, so we match the expected layout
    return {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "this": 5,
        "is": 6,
        "a": 7,
        "test": 8,
        ".": 9,
        "another": 10,
        "example": 11,
        "##s": 12,
    }


def test_encode_output_dataclass_structure():
    eo = bw.EncodeOutput(input_ids=[1, 2], token_type_ids=[0, 0], attention_mask=[1, 1])
    assert eo.input_ids == [1, 2]
    assert eo.token_type_ids == [0, 0]
    assert eo.attention_mask == [1, 1]


def test_bert_input_encoder_basic_sequence_padding_and_mask():
    vocab = _build_vocab_for_encoder()
    encoder = bw.BertInputEncoder(vocab=vocab, max_len=8)

    text = "This is a test."
    out = encoder.encode(text)

    # length should equal max_len
    assert len(out.input_ids) == 8
    assert len(out.token_type_ids) == 8
    assert len(out.attention_mask) == 8

    # CLS at position 0 and SEP present somewhere
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    pad_id = vocab["[PAD]"]

    assert out.input_ids[0] == cls_id
    assert sep_id in out.input_ids

    # attention_mask: 1s for non-pad, 0s for pad
    for tid, mask in zip(out.input_ids, out.attention_mask):
        if tid == pad_id:
            assert mask == 0
        else:
            assert mask == 1

    # token_type_ids should be all zeros for single-sentence classification
    assert set(out.token_type_ids) <= {0}


def test_bert_input_encoder_truncation_preserves_final_sep():
    vocab = _build_vocab_for_encoder()
    max_len = 6
    encoder = bw.BertInputEncoder(vocab=vocab, max_len=max_len)

    # Construct text that will produce > max_len tokens before truncation
    text = "This is a test example."
    out = encoder.encode(text)

    assert len(out.input_ids) == max_len
    sep_id = vocab["[SEP]"]
    # last token must be [SEP] after truncation
    assert out.input_ids[-1] == sep_id


def test_bert_input_encoder_empty_text_still_has_cls_sep():
    vocab = _build_vocab_for_encoder()
    encoder = bw.BertInputEncoder(vocab=vocab, max_len=4)

    out = encoder.encode("")

    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]
    pad_id = vocab["[PAD]"]

    assert out.input_ids[0] == cls_id
    # second token should be [SEP] (no content tokens)
    assert out.input_ids[1] == sep_id
    # remaining positions are PAD
    assert out.input_ids[2:] == [pad_id, pad_id]
    # mask: 1,1,0,0
    assert out.attention_mask == [1, 1, 0, 0]


def test_bert_input_encoder_uses_internal_basic_and_wordpiece_tokenizers():
    vocab = _build_vocab_for_encoder()
    encoder = bw.BertInputEncoder(vocab=vocab, max_len=10)

    # Word "tests" is not in vocab but "test" and "##s" are.
    # BasicTokenizer lowercases + strips punctuation; WordPiece should split into "test", "##s".
    text = "Tests."
    out = encoder.encode(text)

    # We don't know exact positions, but ensure we see test + subword in input_ids
    test_id = vocab["test"]
    sub_id = vocab["##s"]

    assert test_id in out.input_ids
    assert sub_id in out.input_ids


def test_bert_input_encoder_respects_custom_basic_tokenizer(tmp_path: Path):
    """
    Provide a custom BasicTokenizer that doesn't lowercase, to verify it's actually used.
    """
    vocab = _build_vocab_for_encoder()

    class NoLowerBasic(bw.BasicTokenizer):
        def __init__(self):
            super().__init__(lower=False)

    encoder = bw.BertInputEncoder(vocab=vocab, max_len=8, basic_tokenizer=NoLowerBasic())

    text = "This"
    out = encoder.encode(text)
    # With lower=False, BasicTokenizer will preserve case "This".
    # Since "This" (capital T) is not in vocab, WordPiece will produce [UNK]
    # and thus the payload between CLS/SEP should be [UNK].
    unk_id = vocab["[UNK]"]
    # Find CLS and SEP positions
    cls_id = vocab["[CLS]"]
    sep_id = vocab["[SEP]"]

    assert out.input_ids[0] == cls_id
    # payload: 1 token before SEP
    payload_ids = out.input_ids[1:]
    assert unk_id in payload_ids
    assert sep_id in payload_ids


def test_bert_input_encoder_raises_if_max_len_too_small_for_cls_sep():
    vocab = _build_vocab_for_encoder()
    # max_len must allow at least CLS + SEP => 2 tokens
    with pytest.raises(ValueError):
        # You may need to add this guard in your encoder if not present;
        # if you don't, you can relax this test accordingly.
        bw.BertInputEncoder(vocab=vocab, max_len=1)
