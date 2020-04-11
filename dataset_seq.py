import seqtools as sq

import os

import pandas as pd
import seqtools as sq
from yummycurry import curry
from dpu_utils.codeutils import split_identifier_into_parts

from parsing.sitter_lang import get_parser
from tree_sitter import Parser, Tree, TreeCursor, Node

def indentifier_split(token_list):
    token_series = pd.Series(token_list)
    token_identifier_parts = token_series.apply(split_identifier_into_parts)
    return token_identifier_parts

def randomize_mask(seq):
    #ref: https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py#L179
    raise NotImplementedError

def seq_all(_input_path):
    _sample_df = pd.read_pickle(_input_path)
    codes = _sample_df["code"]
    docs = _sample_df["docstring"]
    code_bytes = sq.smap(curry(str.encode)(encoding="utf-8"), codes)
    languages = _sample_df["language"]
    parsers = sq.smap(get_parser, languages)
    asts = sq.smap(Parser.parse, parsers, code_bytes)
    code_tokens = _sample_df["code_tokens"]
    doc_tokens = _sample_df["docstring_tokens"]
    code_split_identifiers = sq.smap(indentifier_split, code_tokens)
    code_tokens_with_identifier_split = sq.smap(pd.Series.explode, code_split_identifiers)
    doc_split_identifiers = sq.smap(indentifier_split, doc_tokens)
    doc_tokens_with_identifier_split = sq.smap(pd.Series.explode, doc_split_identifiers)
    _dict_all = locals()
    _dict_return =  {k:v for k,v in _dict_all.items() if not k.startswith("_")}
    # print(_dict_return.keys())
    return _dict_return