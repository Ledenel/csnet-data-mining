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

#add "raw" option in checkpoint build_params_of_seq_benchmark.
#TODO clean up, take create_prefetch into perf account.
#TODO using snakemake-provided benchmark option.
print(f"extracting {output[0]}")
seq_origin = seq_all(input[0])[wildcards.seq_name]
try:
    start = perf_counter()
    seq = try_prefetch(seq_origin, int(wildcards.seq_cores), wildcards.seq_method, int(wildcards.seq_nbuffer))
    for _ in seq:
        pass
    end = perf_counter()
    value = end - start
except Exception:
    value = None 
import pickle
with open(output[0], "wb") as f_pickle:
    pickle.dump(value, f_pickle)
