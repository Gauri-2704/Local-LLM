# local_llm/__init__.py
from __future__ import annotations

from .convert import convert_tf_bert_to_torch, interactive_setup_bert, setup_bert
from .models.bert import BertConfig, BertModel, masked_mean_pool
from .tokenization.bert_wordpiece import (
    load_vocab,
    BasicTokenizer,
    WordPieceTokenizer,
    BertInputEncoder,
    EncodeOutput,
    SPECIAL_TOKENS,
)
from .pipelines.text_classification import (
    BertClassifierHead,
    BertTextClassifier,
    build_bert_input_encoder,
)

__all__ = [
    # conversion / setup
    "convert_tf_bert_to_torch",
    "interactive_setup_bert",
    "setup_bert",
    # core model
    "BertConfig",
    "BertModel",
    "masked_mean_pool",
    # tokenization
    "load_vocab",
    "BasicTokenizer",
    "WordPieceTokenizer",
    "BertInputEncoder",
    "EncodeOutput",
    "SPECIAL_TOKENS",
    # pipelines
    "BertClassifierHead",
    "BertTextClassifier",
    "build_bert_input_encoder",
]
