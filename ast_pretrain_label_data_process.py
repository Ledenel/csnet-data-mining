
import pandas as pd
import dataset_seq as ds
import ast_label_pretrain as ap
import seqtools as sq
from itertools import chain

from utils import fetch_snakemake_from_latest_run

try:
    snakemake
except NameError:
    snakemake = fetch_snakemake_from_latest_run(__file__)

seqs_all = ds.seq_all(snakemake.input[0])
seqs_labels = ap.seq_from_code_ast(seqs_all)
sub_code_indexes = seqs_labels["sub_code_indexes"]
type_label = seqs_labels[snakemake.params.label_type]
sample_ids = sq.smap(
    lambda samp_list, samp_index:
    [samp_index] * len(samp_list), 
    sub_code_indexes, 
    range(0, len(sub_code_indexes))
)
df = pd.DataFrame({
    "index": chain.from_iterable(sub_code_indexes),
    "sample_id": chain.from_iterable(sample_ids),
    "label": chain.from_iterable(type_label),
})
df = df[df["index"].apply(lambda x:x[0] != x[1])].reset_index()
df.to_pickle(snakemake.output[0])
