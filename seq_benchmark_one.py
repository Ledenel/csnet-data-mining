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
        print(f"building {cores}-{method},fetch{buffered}")
        return sq.prefetch(seq, cores, method, buffered)
    except ValueError:
        return None


benchmarks = []
for seq_name, seq in seq_all(input[0]).items():
    print(f"building {seq_name}")
    benchmark_seqs = {"raw": seq}
    benchmark_seqs.update(
        {f"{cores}-{method}:fetch{nbuffer}": try_prefetch(seq, cores, method, nbuffer)
            for method, cores, nbuffer in product(
                 ["thread", "process", "sharedmem"],
                 np.unique(cpu_cores),
                 [16,256],
            )}
    )
    benchmark_timing = {}
    for name, seq in benchmark_seqs.items():
        print(f"testing {seq_name},{name}")
        
        if seq is not None:
            try:
                start = perf_counter()
                for _ in seq:
                    pass
                end = perf_counter()
                benchmark_timing[name] = end - start
            except Exception:
                benchmark_timing[name] = None
        else:
            benchmark_timing[name] = None
    benchmarks.append(pd.Series(benchmark_timing, name=seq_name))

result = pd.DataFrame(benchmarks)
# result.apply(lambda x: pd.to_timedelta(x, unit="sec"))
result.to_csv(output[0])

result_best_option = result.T.apply(np.argmin).apply(lambda x: result.columns[x])
result_best_value = result.T.min()
pd.DataFrame({
    "option": result_best_option,
    "value": result_best_value,
}).to_csv(output.best)

