import seqtools as sq

import os

import gzip

def load_chunk(input_path):
    with gzip.open(input_path, "r") as f:
        return f.read().splitlines()

