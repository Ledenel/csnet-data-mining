import pandas as pd
from dataset_seq import seq_all
import ast_label_pretrain as pt
from collections import Counter

input = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards




sq_all = seq_all(input[0])
labels = pt.seq_from_code_ast(sq_all)

# sanity check
for k, v in labels.items():
    print(f"{k}:")
    for _ in range(5):
        try:
            print(f"{len(v)}:{v}")
            v = v[0]
        except (TypeError, IndexError):
            break
    print("\n\n\n")

cnt = Counter()
for sample in labels[wildcards.ast_label]:
    for node_labels in sample:
        for i, label in enumerate(node_labels):
            cnt[(i, label)] += 1
summary_df = pd.DataFrame(cnt.values())
summary_df.index = pd.MultiIndex.from_tuples(cnt.keys())
summary_df.unstack(0).to_csv(output[0])
