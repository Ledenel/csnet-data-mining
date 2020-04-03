languages = ('python', 'javascript', 'java', 'ruby', 'php', 'go')

import dataset
import pandas as pd
import os

dataset.init()

import glob

all_jsonls = glob.glob("**/*.jsonl.gz", recursive=True)

jsonl_series = pd.Series(all_jsonls, name="path")
jsonl_path_table = jsonl_series.apply(os.path.basename)\
.str.extract("^([a-zA-Z]+)_([a-zA-Z]+)_([0-9]+)\\.jsonl\\.gz$").merge(jsonl_series, left_index=True, right_index=True)

rule all:
    input:
        "stats/field_context_l2_stats.csv"

rule language_stat:
    input:
        jsonl_path_table["path"]
    output:
        "stats/field_context_l2_stats.csv"
    script:
        "stat.py"
