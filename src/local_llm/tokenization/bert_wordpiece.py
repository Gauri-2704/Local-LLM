from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import unicodedata
import re

SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
    "mask": "[MASK]",
}


def load_vocab(vocab_path: Path | str) -> Dict[str, int]:
    """
    Load a BERT-style vocabulary file.

    - Ignores blank / whitespace-only lines.
    - Indices are assigned by *non-empty* line order (0-based).
    - Validates that required SPECIAL_TOKENS are present.
    """
    vocab_path = Path(vocab_path)
    vocab: Dict[str, int] = {}
    idx = 0
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()  # strip whitespace; skip truly blank lines
            if not tok:
                continue
            vocab[tok] = idx
            idx += 1

    required = [SPECIAL_TOKENS[k] for k in ("pad", "unk", "cls", "sep", "mask")]
    missing = [t for t in required if t not in vocab]
    if missing:
        raise ValueError(f"vocab.txt missing required tokens: {missing}")
    return vocab


class BasicTokenizer:
    def __init__(
        self,
        lower: bool = True,
        protected_tokens: Sequence[str] = ("[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"),
    ):
        self.lower = lower
        self.protected = tuple(protected_tokens)
        escaped = [re.escape(t) for t in self.protected]
        self._prot_re = re.compile("(" + "|".join(escaped) + ")")

    def _strip_accents(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def _tokenize_segment(self, seg: str) -> List[str]:
        if self.lower:
            seg = seg.lower()
        seg = self._strip_accents(seg)

        buff = []
        for ch in seg:
            if ch.isalnum():
                buff.append(ch)
            elif ch.isspace():
                buff.append(" ")
            else:
                buff.append(f" {ch} ")
        return [t for t in "".join(buff).split() if t]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string into basic tokens.

        - Empty string -> [].
        - Non-string (e.g. None) -> raises AttributeError (tests enforce this).
        """
        if not isinstance(text, str):
            # Tests expect AttributeError specifically for None
            raise AttributeError("BasicTokenizer.tokenize expects a string input.")
        if not text:
            return []

        tokens: List[str] = []
        parts = self._prot_re.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.protected:
                tokens.append(part)
            else:
                tokens.extend(self._tokenize_segment(part))
        return tokens


class WordPieceTokenizer:
    """Greedy longest-match WordPiece over a BERT vocab."""

    def __init__(
        self,
        vocab: Dict[str, int],
        unk_token: str = SPECIAL_TOKENS["unk"],
        max_input_chars_per_word: int = 100,
    ):
        self.vocab = vocab
        self.unk = unk_token
        self.maxlen = max_input_chars_per_word

    def tokenize(self, text_tokens: Sequence[str]) -> List[str]:
        out: List[str] = []
        for token in text_tokens:
            if len(token) > self.maxlen:
                out.append(self.unk)
                continue
            start = 0
            sub_tokens: List[str] = []
            is_bad = False
            while start < len(token):
                end = len(token)
                cur = None
                while start < end:
                    piece = token[start:end]
                    if start > 0:
                        piece = "##" + piece
                    if piece in self.vocab:
                        cur = piece
                        break
                    end -= 1
                if cur is None:
                    is_bad = True
                    break
                sub_tokens.append(cur)
                start = end
            out.extend([self.unk] if is_bad else sub_tokens)
        return out

    def convert_tokens_to_ids(self, pieces: Sequence[str]) -> List[int]:
        unk_id = self.vocab[self.unk]
        return [self.vocab.get(p, unk_id) for p in pieces]


@dataclass
class EncodeOutput:
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]


class BertInputEncoder:
    """
    Composes a BasicTokenizer + WordPieceTokenizer and adds [CLS]/[SEP] + padding.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        max_len: int,
        basic_tokenizer: BasicTokenizer | None = None,
    ):
        if max_len < 2:
            # Tests require this guard: must fit at least [CLS] and [SEP]
            raise ValueError("max_len must be at least 2 to fit [CLS] and [SEP].")

        self.vocab = vocab
        self.max_len = max_len
        self.basic = basic_tokenizer or BasicTokenizer()
        self.wp = WordPieceTokenizer(vocab)
        self.pad_id = vocab[SPECIAL_TOKENS["pad"]]
        self.cls_id = vocab[SPECIAL_TOKENS["cls"]]
        self.sep_id = vocab[SPECIAL_TOKENS["sep"]]

    def encode(self, text: str) -> EncodeOutput:
        basic_tokens = self.basic.tokenize(text)
        pieces = self.wp.tokenize(basic_tokens)
        tokens = [SPECIAL_TOKENS["cls"]] + pieces + [SPECIAL_TOKENS["sep"]]
        ids = self.wp.convert_tokens_to_ids(tokens)

        if len(ids) > self.max_len:
            # ensure final token is SEP after truncation
            ids = ids[: self.max_len - 1] + [self.sep_id]

        attn = [1] * len(ids)
        seg = [0] * len(ids)

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
            attn += [0] * pad_len
            seg += [0] * pad_len

        return EncodeOutput(ids, seg, attn)
