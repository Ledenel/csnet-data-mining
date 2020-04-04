import dataset
import pandas as pd
import os

dataset.init()

import glob

all_jsonls = glob.glob("**/*.jsonl.gz", recursive=True)

language_reg="(python|javascript|java|ruby|php|go)"
split_reg="(train|test|valid)"
chunk_reg="([0-9]+)"
dataset_reg=f"{language_reg}_{split_reg}_{chunk_reg}"

jsonl_series = pd.Series(all_jsonls, name="path")
jsonl_path_table = jsonl_series.apply(os.path.basename).str.extract(
    f"^({dataset_reg})\\.jsonl\\.gz$"
)
jsonl_path_table.columns = ["dataset", "language", "split", "chunk"]
jsonl_path_table["chunk"] = jsonl_path_table["chunk"].astype(int)
jsonl_path_table["path"] = jsonl_series

wildcard_constraints:
    language=language_reg,
    split=split_reg,
    chunk=chunk_reg,
    dataset=dataset_reg,

rule all:
    input:
        # "stats/field_context_l2_stats.csv",
        "data_cache/go_train_0.pkl",

rule language_stat:
    input:
        jsonl_path_table["path"]
    output:
        "stats/field_context_l2_stats.csv"
    script:
        "stat.py"

rule build_corpus:
    input:
        "data_cache/{dataset}.pkl"
    output:
        "corpus/{dataset}.txt"
    run:
        import dataset
        raise NotImplementedError

rule cache_dataset_to_pickle:
    input:
        "data/{language}/final/jsonl/{split}/{language}_{split}_{chunk}.jsonl.gz"
    output:
        "data_cache/{language}_{split}_{chunk}.pkl"
    run:
        import pandas as pd
        df = pd.read_json(input[0], compression="gzip", lines=True)
        df.to_pickle(output[0])
