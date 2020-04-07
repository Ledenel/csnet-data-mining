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
# jsonl_path_table["chunk"] = jsonl_path_table["chunk"].astype(int)
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
        "profill/go_train_0.seq_benchmark.csv",

rule extract_language_stat:
    # input:
    #     jsonl_path_table["path"]
    output:
        "stats/field_context_l2_stats.csv"
    script:
        "stat.py"

checkpoint build_params_of_seq_benchmark:
    input:
        "data_cache/{dataset}.pkl"
    output:
        "data_cache/{dataset}.seq_benchmark.params.pkl"
    run:
        import pandas as pd
        import numpy as np
        dataset_df = pd.read_pickle(input[0])
        cpu_cores = 2 ** np.arange(1, np.log2(os.cpu_count()) + 1).astype(int)
        np.append(cpu_cores, os.cpu_count())
        cpu_cores.sort()
        seq_cpu_cores = np.unique(cpu_cores)

        seq_method = ["thread", "process", "sharedmem"]
        seq_nbuffer = [16,256]

        from dataset_seq import seq_all
        seq_name = seq_all(input[0]).keys()

        from itertools import product, chain

        param_combinations = \
        product(
            seq_name,
            chain(
                [(0, thread, 0)],
                product(
                    cpu_cores,
                    seq_method,
                    seq_nbuffer,
                )
            )
        )

        param_combination_dicts = list({
            'seq_name': seq_name,
            'seq_cores': seq_cores,
            'seq_method': seq_method,
            'seq_nbuffer': seq_nbuffer,
        } for seq_name,
            (seq_cores,
            seq_method,
            seq_nbuffer) in param_combinations
        )
        import pickle
        with open(output[0], "wb") as f:
            pickle.dump(param_combination_dicts, f)



#FIXME: bulid list wrapper (is it necessary?)
#FIXME:
# ERROR:snakemake.logging:InputFunctionException in line 127 of /mnt/d/workspace/playground/csnet-data-mining/Snakefile:
# RecursionError: maximum recursion depth exceeded
# Wildcards:
# dataset=go_train_0
def _seq_benchmark_files_from_params(wildcards):
    import pickle
    #
    with open(checkpoints.build_params_of_seq_benchmark.get(dataset=wildcards.dataset).output[0], mode="rb") as f:
        for item in pickle.load(f):
            yield ("profill/temp/{{dataset}}/"
            "{seq_name}--{seq_method}--{seq_cores}--{seq_nbuffer}.float.pkl").format(**item).format(**wildcards)

def seq_benchmark_files_from_params(wildcards):
    return list(_seq_benchmark_files_from_params(wildcards))

#TODO: add resources limit to benchmark: always use all resources to block anything else.

rule benchmark_one:
    input:
        dataset = "data_cache/{dataset}.pkl",
    output:
        "profill/temp/{dataset}/{seq_name}--{seq_method}--{seq_cores}--{seq_nbuffer}.float.pkl"
    script:
        "seq_benchmark_one.py"

def pickle_load(file_path):
    import pickle
    with open(file_path, "rb") as f:
        return pickle.load(f)

# FIXME: ERROR:snakemake.logging:InputFunctionException in line 127 of /mnt/d/workspace/playground/csnet-data-mining/Snakefile:
# RecursionError: maximum recursion depth exceeded
# Wildcards:
# dataset=go_train_0
rule benchmark_seq_merge:
    input:
        seq_benchmark_files_from_params,
        config="data_cache/{dataset}.seq_benchmark.params.pkl",
    output:
        "profill/{dataset}.seq_benchmark.csv",
        best="profill/{dataset}.seq_best.csv",
    run:
        import pickle
        import pandas as pd
        from collections import defaultdict
        merged_dict = defaultdict(dict)
        for benchmark_path, config_param_dict in zip(input[:-1], pickle_load(input.config)):
            param_cols = [f"{param}-{value}"
                 for param, value in config_param_dict.items()]
            for col in param_cols:
                merged_dict[file_path][col] = pickle_load(file_path)
        
        result = pd.DataFrame(merged_dict)
        result.to_csv(output[0])
        result_best_option = result.T.apply(np.argmin).apply(lambda x: result.columns[x])
        result_best_value = result.T.min()
        pd.DataFrame({
            "option": result_best_option,
            "value": result_best_value,
        }).to_csv(output.best)

        
rule build_corpus_raw:
    input:
        "data_cache/{dataset_chunk}.pkl"
    output:
        code="corpus/{dataset_chunk}_code.txt",
        doc="corpus/{dataset_chunk}_doc.txt",
    run:
        from dataset_seq import seq_all
        seq_dicts = seq_all(input[0])

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

rule copy_test_column_from_ruby_test_1:
    input:
        "stats/ruby_test_1.columns.pkl"
    output:
        "stats/column.txt"
    run:
        import pandas as pd
        with open(output[0], "w") as f:
            for item in pd.read_pickle(input[0]):
                f.write(f"{item}\n")
        


rule stat_of_dataset:
    input:
        "data_cache/{dataset}.pkl"
    output:
        # directory("stats/{dataset}"),
        meta = "stats/{dataset}.meta.log",
        columns = "stats/{dataset}.columns.pkl",
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
        pd.to_pickle(df.columns, output.columns)
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
