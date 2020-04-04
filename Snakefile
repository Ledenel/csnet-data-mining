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
jsonl_path_table.columns = ["dataset_chunk", "language", "split", "chunk"]
jsonl_path_table["chunk"] = jsonl_path_table["chunk"].astype(int)
jsonl_path_table["path"] = jsonl_series

wildcard_constraints:
    language=language_reg,
    split=split_reg,
    chunk=chunk_reg,
    dataset_chunk=dataset_reg,

rule all:
    input:
        # "stats/field_context_l2_stats.csv",
        # directory("stats/go_train_0")
        "stats/go_train_0.value_count_stat.csv",
        "stats/ruby_all.value_count_stat.csv",

rule extract_language_stat:
    # input:
    #     jsonl_path_table["path"]
    output:
        "stats/field_context_l2_stats.csv"
    script:
        "stat.py"

rule build_corpus:
    input:
        "data_cache/{dataset_chunk}.pkl"
    output:
        "corpus/{dataset_chunk}.txt"
    run:
        import dataset
        raise NotImplementedError

rule cache_dataset_chunk_to_pickle:
    input:
        "data/{language}/final/jsonl/{split}/{language}_{split}_{chunk}.jsonl.gz"
    output:
        out = "data_cache/{language}_{split}_{chunk}.pkl",
    run:
        import pandas as pd
        
        df = pd.read_json(input[0], compression="gzip", lines=True)
        df.to_pickle(output.out)

rule stat_of_dataset:
    input:
        "data_cache/{dataset}.pkl"
    output:
        # directory("stats/{dataset}"),
        meta = "stats/{dataset}.meta.log",
        len_stat_csv = "stats/{dataset}.len_stat.csv",
        value_counts_csv = "stats/{dataset}.value_count_stat.csv", 
    run:
        import contextlib as ctx
        import pandas as pd
        df = pd.read_pickle(input[0])

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

        import utils
        utils.value_counts_describe(df).to_csv(output.value_counts_csv)

        # value_count_df = df.apply(pd.Series.value_counts, axis="index")
        # don't value count unhashable type (like list) or value count on list values(via explode?)(str is iterable, no recursive count).
        # value_count_df.to_csv(output.value_counts_csv)

from yummycurry import curry

@curry
def combine_format(keys, format, series):
    output_dict = dict(zip(keys, series))
    return format.format(**output_dict)

@curry
def dataset_format_paths(output_args, format, wildcards):
    """
    function constructor to feed path table columns into specific path format.
    @param: output_args column names in path_table.
    @param: format path format string (you can use name in output_args).
    @wildcards: remain for snakemake call.

    @returns: a list of path filtered by 'conditions' in wildcards.
    """
    filter = (jsonl_path_table.iloc[:,0] == 0) | True
    for key, value in wildcards.items():
        filter &= jsonl_path_table[key] == value
    
    args_values = jsonl_path_table[filter][output_args]
    format_paths = args_values.apply(combine_format(output_args, format), axis="columns")
    return format_paths

#equally func
def dataset_format_paths_language(wildcards):
    return dataset_format_paths(["dataset_chunk"], "data_cache/{dataset_chunk}.pkl", wildcards)

rule merge_language_dataset:
    input:
        dataset_format_paths_language,
        # dataset_format_paths(["dataset_chunk"], "data_cache/{dataset_chunk}.pkl"),
    output:
        "data_cache/{language}_all.pkl"
    run:
        import pandas as pd
        merged = pd.concat([pd.read_pickle(path) for path in input])
        merged.to_pickle(output[0])
