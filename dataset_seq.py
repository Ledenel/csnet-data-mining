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

def seq_all(input_path):
    samples_df = pd.DataFrame.read_pickle(input_path)
    codes = samples_df["code"]
    docs = samples_df["docstring"]
    code_bytes = sq.smap(curry(str.encode)(encoding="utf-8"), codes)
    languages = samples_df["language"]
    parsers = sq.smap(get_parser, languages)
    asts = sq.smap(Parser.parse, parsers, code_bytes)
    code_tokens = samples_df["code_tokens"]
    doc_tokens = samples_df["docstring_tokens"]
    code_split_identifiers = sq.smap(indentifier_split, code_tokens)
    code_tokens_with_identifier_split = sq.smap(pd.Series.explode, code_split_identifiers)
    doc_split_identifiers = sq.smap(indentifier_split, doc_tokens)
    doc_tokens_with_identifier_split = sq.smap(pd.Series.explode, doc_split_identifiers)
    return locals()
