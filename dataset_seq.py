import seqtools as sq

import os

import pandas as pd
import seqtools as sq
from yummycurry import curry

from parsing.sitter_lang import get_parser
from tree_sitter import Parser, Tree, TreeCursor, Node

def seq_all(input_path):
    samples_df = pd.DataFrame.read_pickle(input_path)
    codes = samples_df["code"]
    docs = samples_df["docstring"]
    code_bytes = sq.smap(curry(str.encode)(encoding="utf-8"), codes)
    languages = samples_df["language"]
    parsers = sq.smap(get_parser, languages)
    asts = sq.smap(Parser.parse, parsers, code_bytes)
    
    return locals()
