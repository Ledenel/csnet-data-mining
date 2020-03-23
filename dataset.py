from subprocess import call

import os

import torch.utils.data as dt

from functools import lru_cache
from collections import defaultdict

import gzip
import json
import glob
import re
import psutil
import sys
import math

import pkg_resources

DATA_DIR = pkg_resources.resource_filename(__name__, "data")


class CodeSearchDataPath:
    def __init__(self, path):
        super().__init__()
        self.full_path = os.path.abspath(path)
        _, name = os.path.split(path)
        reg = re.compile("^([a-zA-Z]+)_([a-zA-Z]+)_([0-9]+).*$")
        self.lang, self.split, self.chunk_num = reg.match(name).groups()
        self.chunk_num = int(self.chunk_num)
        self.precomputed_len = None


class CodeSearchChunk(dt.Dataset):
    def __init__(self, path):
        super().__init__()
        if not isinstance(path, CodeSearchDataPath):
            path = CodeSearchDataPath(path)
        self.path = path

    @property
    @lru_cache(1)
    def data(self):
        with gzip.open(self.path.full_path, "r") as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return self.path.precomputed_len if self.path.precomputed_len is not None else len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def esetimate_max_cache_count_by_system_memory(datapath: CodeSearchDataPath, using_percent=0.6):
    _, available, *_ = psutil.virtual_memory()
    test_item = CodeSearchChunk(datapath)
    return math.floor(available * using_percent / sys.getsizeof(test_item.data))


class CodeSearchChunkPool():
    def __init__(self, root_path=DATA_DIR, chunk_full_len=30000):
        self.root_path = os.path.abspath(root_path)
        init(root_path)
        dataset_paths = (CodeSearchDataPath(path)
                         for path in glob.glob("**/*.jsonl.gz", recursive=True))
        self.path_map = {(datapath.lang, datapath.split, datapath.chunk)                         : datapath for datapath in dataset_paths}

        path_collect = defaultdict(list)
        for datapath in self.path_map.values():
            path_collect[(datapath.lang, datapath.split)].append(datapath)

        self.chunk_full_len = chunk_full_len
        for _, path_list in path_collect.items():
            path_list.sort(key=lambda x: x.chunk_num)
            for path in path_list[:-1]:
                path.precomputed_len = self.chunk_full_len

        self.max_cache = esetimate_max_cache_count_by_system_memory(self.path_map.values()[0])
        print(f"max cache num: {self.max_cache}")

        @lru_cache(self.max_cache)
        def get(lang, split, chunk):
            return CodeSearchChunk(self.path_map[(lang, split, chunk)])

        self.get_func = get

    def __getitem__(self, lang, split, chunk):
        get_func = self.get_func
        return get_func(lang, split, chunk)


class CodeSearchDatasetLoader():
    def __init__(self, root_path=DATA_DIR):
        self.root_path = os.path.abspath(root_path)
        init(root_path)
        self.pool = CodeSearchChunkPool(root_path)

    @property
    def max_chunks(self):
        return len(self.gz_dataset_chunk)

    @property
    @lru_cache(1)
    def language(self):
        return list(set(x.lang for x in self.gz_dataset_chunks))

    @property
    @lru_cache(1)
    def split(self):
        return list(set(x.split for x in self.gz_dataset_chunks))

    def get(self, language=None, split=None, chunk_slice=None):
        def is_needed(chunk: CodeSearchChunk):
            cond = True
            if language is not None:
                cond = cond and chunk.path.lang == language
            if split is not None:
                cond = cond and chunk.path.split == split
            if chunk_slice is not None:
                if isinstance(chunk_slice, int):
                    cond = cond and chunk.path.chunk_num == chunk_slice
                elif isinstance(chunk_slice, slice):
                    cond = cond and chunk.path.chunk_num in range(
                        *chunk_slice.indices(self.max_chunks))
                else:
                    raise ValueError(f"invalid chunk index {chunk_slice}")
            return cond

        return dt.ConcatDataset([x for x in self.gz_dataset_chunks if is_needed(x)])


def init(destination_dir=DATA_DIR):
    old_dir = os.getcwd()
    os.makedirs(destination_dir, exist_ok=True)
    os.chdir(destination_dir)

    for language in ('python', 'javascript', 'java', 'ruby', 'php', 'go'):
        if os.path.exists(f'{language}.unzipped'):
            print(f"already unzipped {language} dataset.")
        else:
            if os.path.exists(f'{language}.zip'):
                print(f"already downloaded {language} dataset.")
            else:
                call(
                    ['wget', f'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip', '-O', f'{language}.zip'])

            call(['unzip', '-o', f'{language}.zip'])
            call(['touch', f'{language}.unzipped'])

    os.chdir(old_dir)


if __name__ == "__main__":
    init()
