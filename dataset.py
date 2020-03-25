from subprocess import call

import os

import torch.utils.data as dt

from functools import lru_cache
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

import zipfile
import gzip
import json
import glob
import re
import pkg_resources
import sys
import math

import requests
import psutil

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
    
    ORDER = {
        "train" : 1,
        "valid" : 2,
        "test" : 3,
    }

    def __str__(self):
        return f"{self.lang}-{self.split}-{self.chunk_num}"
    
    def _tuple_key(self):
        return self.lang, CodeSearchDataPath.ORDER[self.split], self.chunk_num

    def __lt__(self, other):
        return self._tuple_key() < other._tuple_key()



class CodeSearchChunk(dt.Dataset):
    def __init__(self, path):
        super().__init__()
        if not isinstance(path, CodeSearchDataPath):
            path = CodeSearchDataPath(path)
        self.path = path
        print(f"loading {path}")
        with gzip.open(self.path.full_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return self.path.precomputed_len if self.path.precomputed_len is not None else len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def esetimate_max_cache_count_by_system_memory(datapath: CodeSearchDataPath, using_percent=0.3):
    _, available, *_ = psutil.virtual_memory()
    test_item = CodeSearchChunk(datapath)
    data_size = sys.getsizeof(test_item.data)
    return math.floor(available * using_percent / data_size)


class CodeSearchChunkPool():
    def __init__(self, root_path=DATA_DIR, chunk_full_len=30000, max_chunks_in_memory=None):
        self.root_path = os.path.abspath(root_path)
        init(root_path)
        dataset_paths = [CodeSearchDataPath(path)
                         for path in glob.glob("**/*.jsonl.gz", recursive=True)]
        self.dataset_paths = dataset_paths
        self.path_map = {(datapath.lang, datapath.split, datapath.chunk_num)
                          : datapath for datapath in dataset_paths}

        path_collect = defaultdict(list)
        for datapath in self.path_map.values():
            path_collect[(datapath.lang, datapath.split)].append(datapath)

        self.chunk_full_len = chunk_full_len
        for _, path_list in path_collect.items():
            path_list.sort(key=lambda x: x.chunk_num)
            for path in path_list[:-1]:
                path.precomputed_len = self.chunk_full_len
                precomputed_path = path

        self.max_cache = max_chunks_in_memory or esetimate_max_cache_count_by_system_memory(precomputed_path)
        print(f"max cache num: {self.max_cache}")

        @lru_cache(self.max_cache)
        def get(datapath: CodeSearchDataPath):
            return CodeSearchChunk(datapath)

        self.get_func = get

    def get_by_path(self, datapath: CodeSearchDataPath):
        get_func = self.get_func
        return get_func(datapath)

    def __getitem__(self, lang, split, chunk):
        return self.get_by_path(self.path_map[(lang, split, chunk)])

class LazyLoadCodeSearchChunk(dt.Dataset):
    def __init__(self, path:CodeSearchDataPath, pool:CodeSearchChunkPool):
        super().__init__()
        self.path = path
        self.pool = pool

    def __getitem__(self, index):
        return self._gen()[index]

    def _gen(self):
        return self.pool.get_by_path(self.path)

    def __len__(self):
        return self.path.precomputed_len or len(self._gen())

class CodeSearchDatasetLoader():
    def __init__(self, root_path=DATA_DIR, max_chunks_in_memory=None):
        self.root_path = os.path.abspath(root_path)
        init(self.root_path)
        self.pool = CodeSearchChunkPool(root_path, max_chunks_in_memory=max_chunks_in_memory)

    @property
    @lru_cache(1)
    def language(self):
        return list(set(x.lang for x in self.pool.dataset_paths))

    @property
    @lru_cache(1)
    def split(self):
        return list(set(x.split for x in self.pool.dataset_paths))

    def get(self, language=None, split=None, chunk_slice=None):
        def is_needed(path: CodeSearchDataPath):
            cond = True
            if language is not None:
                cond = cond and path.lang == language
            if split is not None:
                cond = cond and path.split == split
            if chunk_slice is not None:
                if isinstance(chunk_slice, int):
                    cond = cond and path.chunk_num == chunk_slice
                elif isinstance(chunk_slice, slice):
                    cond = cond and path.chunk_num in range(
                        *chunk_slice.indices(self.max_chunks))
                else:
                    raise ValueError(f"invalid chunk index {chunk_slice}")
            return cond
        needed_paths = [
            path for path in self.pool.dataset_paths if is_needed(path)
        ]
        needed_paths.sort()
        return dt.ConcatDataset(
            [
                LazyLoadCodeSearchChunk(path, self.pool)
                for path in needed_paths
            ]
        )

def load_try():
    for language in ('python', 'javascript', 'java', 'ruby', 'php', 'go'):
        if os.path.exists(f'{language}.unzipped'):
            print(f"already unzipped {language} dataset.")
        else:
            if os.path.exists(f'{language}.zip'):
                print(f"already downloaded {language} dataset.")
            else:
                g = requests.get(
                    f'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip', stream=True)
                with open(f'{language}.zip', 'wb') as sav:
                    for chunk in tqdm(
                        g.iter_content(chunk_size=1024),
                        desc=f"downloading {language} dataset",
                        total=math.ceil(int(g.headers['Content-length'])/1024),
                        unit="KB"
                    ):
                        sav.write(chunk)

            try:
                with zipfile.ZipFile(f'{language}.zip', 'r') as zip_ref:
                    members = zip_ref.namelist()
                    for zipinfo in tqdm(members, desc=f'unzipping {language}.zip'):
                        zip_ref.extract(zipinfo)
            except zipfile.BadZipFile:
                print(f"{language}.zip is broken: removed and retry.")
                os.remove(f"{language}.zip")
                load_try()


            Path(f'{language}.unzipped').touch()


def init(destination_dir=DATA_DIR):
    old_dir = os.getcwd()
    os.makedirs(destination_dir, exist_ok=True)
    os.chdir(destination_dir)
    load_try()
    os.chdir(old_dir)


if __name__ == "__main__":
    init()
