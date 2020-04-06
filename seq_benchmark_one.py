import pandas as pd
import seqtools as sq
import numpy as np
import os
from time import perf_counter
from itertools import product
from dataset_seq import seq_all

# snakemake options:
# input:
#     "data_cache/{dataset}.pkl"
# output:
#     "profill/temp/{dataset}/{seq_name}--{seq_method}--{seq_cores}--{seq_nbuffer}.float.pkl"


input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

def try_prefetch(seq, cores, method, buffered):
    try:
        # print(f"building {cores}-{method},fetch{buffered}")
        return sq.prefetch(seq, cores, method, buffered)
    except ValueError:
        return None

#FIXME: add "raw" option in checkpoint build_params_of_seq_benchmark.
benchmark_seqs = {"raw": seq}
benchmark_seqs.update(
try:
    start = perf_counter()
    for _ in seq:
        pass
    end = perf_counter()
    benchmark_timing[name] = end - start
except Exception:
    benchmark_timing[name] = None
