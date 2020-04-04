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

rule extract_language_stat:
    # input:
    #     jsonl_path_table["path"]
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
        out = "data_cache/{language}_{split}_{chunk}.pkl",
        meta = "data_cache/{language}_{split}_{chunk}.meta.log",
        len_stat_csv = "data_cache/{language}_{split}_{chunk}.len_stat.csv",
        value_counts_csv = "data_cache/{language}_{split}_{chunk}.value_count_stat.csv",
    run:
        import pandas as pd
        import contextlib as ctx
        df = pd.read_json(input[0], compression="gzip", lines=True)
        df.to_pickle(output.out)

        def nan_when_exception(func):
            def _nan_wrapper(x):
                try:
                    return func(x)
                except Exception:
                    return None
            return _nan_wrapper

        with open(output.meta, "w") as fmeta:
            with ctx.redirect_stdout(fmeta):
                print(f"len: {len(df)}")
                print(f"columns: {df.columns}")
        len_df = df.applymap(nan_when_exception(len)).dropna(axis="columns")
        len_df.describe(percentiles=[0.05,0.2,0.5,0.8,0.95]).T.to_csv(output.len_stat_csv)

        # value_count_df = df.apply(pd.Series.value_counts, axis="index")
        # FIXME: don't value count unhashable type (like list) or value count on list values(via explode?)(str is iterable, no recursive count).
        # value_count_df.to_csv(output.value_counts_csv)
