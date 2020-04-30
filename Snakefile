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
        # "stats/go_train_0.value_count_stat.csv",
        # "stats/ruby_all.value_count_stat.csv",
        # "profill/go_train_0.seq_best.csv",
        # "corpus/go_train_0.doc.txt",
        # "corpus/go_train_0.code.txt",
        # "corpus/tokenizer/go_train_0.code-size=20000/vocab.txt",
        expand("roberta_{lang}_all.done", lang="python|javascript|java|ruby|php|go".split("|")),
        # "stats/ruby-valid_0-combined_label-counts.csv"
        expand("stats/{lang}-all-{lb}-counts.csv", lang="python|javascript|java|ruby|php|go".split("|"), lb=["type_label", "combined_label"])
        # directory("model_param/pretrain/go_train_0-tokenizer:size=20000"),

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
        "profill/{dataset}.seq_benchmark.params.pkl"
    run:
        import pandas as pd
        import numpy as np
        dataset_df = pd.read_pickle(input[0])
        cpu_cores = 2 ** np.arange(1, np.log2(os.cpu_count()) + 1).astype(int)
        np.append(cpu_cores, os.cpu_count())
        cpu_cores.sort()
        seq_cpu_cores = np.unique(cpu_cores)

        seq_method = ["thread", "process", "sharedmem"]
        seq_nbuffer = [16,64]

        from dataset_seq import seq_all
        seq_name = seq_all(input[0]).keys()

        from itertools import product, chain

        param_combinations = \
        product(
            seq_name,
            chain(
                [(0, "thread", 0)],
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

rule extract_benchmark_seq:
    input:
        seq_benchmark_files_from_params,
        config="profill/{dataset}.seq_benchmark.params.pkl",
    output:
        "profill/{dataset}.seq_benchmark.csv",
        
    run:
        import pickle
        import pandas as pd
        import numpy as np
        from collections import defaultdict
        merged_record = list()
        for benchmark_path, config_param_dict in zip(input[:-1], pickle_load(input.config)):
            param_cols = [f"{param}-{value}"
                 for param, value in config_param_dict.items()]
            sub_record = {col:value for col, value in config_param_dict.items()}
            sub_record["value"] = pickle_load(benchmark_path)
            merged_record.append(sub_record)

        result = pd.DataFrame(merged_record)
        result.to_csv(output[0])

rule pick_benchmark_seq_best:
    input:
        "profill/{dataset}.seq_benchmark.csv"
    output:
        best="profill/{dataset}.seq_best.csv"
    run:
        import pandas as pd
        import numpy as np
        result = pd.read_csv(input[0])
        print(result.columns)
        result["params"] = result["seq_cores"].astype(str) + "-" + result["seq_method"].astype(str) + "-" + result["seq_nbuffer"].astype(str)
        result = result.pivot(index="seq_name", columns="params", values=["value"])
        result_best_option = result.T.apply(np.argmin).apply(lambda x: result.columns[x])
        result_best_value = result.T.min()
        pd.DataFrame({
            "option": result_best_option,
            "value": result_best_value,
        }).to_csv(output.best)
        # result.to_csv(output[0])

        
rule build_corpus_raw:
    input:
        "data_cache/{dataset}.pkl"
    output:
        "corpus/{dataset}.{corpus_type,(code|doc)}.txt",
    run:
        from dataset_seq import seq_all
        seq_dicts = seq_all(input[0])
        corpus_split = seq_dicts[f"{wildcards.corpus_type}_tokens_with_identifier_split"]
        with open(output[0], "w") as f:
            for sample in corpus_split:
                f.write(" ".join(sample))
                f.write("\n")

rule build_counter_on_dataset:
    input:
        "data_cache/{dataset}.pkl"
    output:
        "corpus/{dataset}.{corpus_type,(code|doc)}.counter.json",
    run:
        from dataset_seq import seq_all
        from collections import Counter
        seq_dicts = seq_all(input[0])
        corpus_split = seq_dicts[f"{wildcards.corpus_type}_tokens_with_identifier_split"]
        raise NotImplementedError
        # for item in 
        # with open(output[0], "w") as f:
        #     for sample in corpus_split:
        #         f.write(" ".join(sample))
        #         f.write("\n")

rule train_tokenizer:
    input:
        "corpus/{dataset}.{corpus_type}.txt",
    output:
        "corpus/tokenizer/{dataset}.{corpus_type}-size={vocab}/vocab.txt",
    run:
        from tokenizers import BertWordPieceTokenizer
        tokenizer = BertWordPieceTokenizer(
            strip_accents=True, lowercase=True,
        )

        trainer = tokenizer.train(
            input,
            vocab_size=int(wildcards.vocab),
            min_frequency=2,
            show_progress=True,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            limit_alphabet=1000,
            wordpieces_prefix="##",
        )

        import os
        tokenizer.save(os.path.dirname(output[0]))

rule pretrain_bert:
    input:
        train = "data_cache/{lang}_train_{extra}.pkl",
        doc_tokenizer = "corpus/tokenizer/{lang}_train_{extra}.doc-{config}/vocab.txt",
        code_tokenizer = "corpus/tokenizer/{lang}_train_{extra}.code-{config}/vocab.txt",
    output:
        model_ckpt = directory("model_param/pretrain/{lang}_train_{extra}-tokenizer:{config}"),
    run:
        from tokenizers import BertWordPieceTokenizer
        from transformers import BertModel
        
        raise NotImplementedError

rule train_eval_bert_scratch_dev:
    input:
        train = "data_cache/{lang}_train_{extra}.pkl",
        valid = "data_cache/{lang}_valid_{extra}.pkl",
        test = "data_cache/{lang}_test_{extra}.pkl",
    output:
        done = touch("bert_scratch_{lang}_{extra}_fast.done")
    params:
        seed = 127,
        fast = True,
    resources:
        gpus = 1,
    script:
        "roberta_eval.py"

rule roberta_train:
    input:
        train = "data_cache/{lang}_train_{extra}.pkl",
        valid = "data_cache/{lang}_valid_{extra}.pkl",
        test = "data_cache/{lang}_test_{extra}.pkl",
        fast_dev = "bert_scratch_{lang}_{extra}_fast.done",
    output:
        done = touch("roberta_{lang}_{extra}.done")
    params:
        seed = 127,
        fast = False,
    resources:
        gpus = 1,
    script:
        "roberta_eval.py"

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

from pymonad.Reader import curry

@curry
def combine_format(keys, format, series):
    output_dict = dict(zip(keys, series))
    return format.format(**output_dict)

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
        merged.reset_index(inplace=True)
        merged.to_pickle(output[0])

rule merge_language_dataset_split_part:
    input:
        dataset_format_paths_language,
        # dataset_format_paths(["dataset_chunk"], "data_cache/{dataset_chunk}.pkl"),
    output:
        "data_cache/{language}_{split}_all.pkl"
    run:
        import pandas as pd
        merged = pd.concat([pd.read_pickle(path) for path in input])
        merged.reset_index(inplace=True)
        merged.to_pickle(output[0])

rule count_ast_label:
    input:
        "data_cache/{lang}_{spec}.pkl"
    output:
        "stats/{lang}-{spec}-{ast_label}-counts.csv"
    script:
        "ast_label.py"
# import pandas as pd
# from dataset_seq import seq_all
# import ast_label_pretrain as pt
# from collections import Counter
# seq_dict = seq_all(input[0])


# labels = pt.seq_from_code_ast(sq_all)
# cnt = Counter()
# for sample in labels[wildcard.ast_label]:
#     for i, label in enumerate(sample):
#         cnt[(i, label)] += 1
# summary_df = pd.DataFrame(cnt.values())
# summary_df.index = pd.MultiIndex.from_tuples(cnt.keys())
# summary_df.unstack(0)
# summary_df.to_csv(output[0])
