from subprocess import call

import os

import torch.utils.data as dt

from itertools import count, chain
from collections import defaultdict, namedtuple, Counter
from pathlib import Path

from tqdm import tqdm
from pympler import asizeof
from attributedict.collections import AttributeDict

import zipfile
import gzip
import json
import glob
import re
import pkg_resources
import sys
import math
import multiprocessing

import requests
import psutil
import ring
import logging

from parsing.sitter_lang import get_parser
# import methodtools

from typing import Iterable
from tree_sitter import TreeCursor, Node


def node_cursor_iter(cursor) -> Iterable[TreeCursor]:
    yield cursor
    if cursor.goto_first_child():
        yield from node_cursor_iter(cursor)
        while cursor.goto_next_sibling():
            yield from node_cursor_iter(cursor)
        cursor.goto_parent()


DATA_DIR = pkg_resources.resource_filename(__name__, "data")

# _code_sample_fields = [
#     'repo',
#     'path',
#     'func_name',
#     'original_string',
#     'language',
#     'code',
#     'code_tokens',
#     'docstring',
#     'docstring_tokens',
#     'sha',
#     'url',
#     'partition',
#     'ast_root',
# ]

# CodeSample = namedtuple(
#     "CodeSample",
#     _code_sample_fields
# )


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
        "train": 1,
        "valid": 2,
        "test": 3,
    }

    def __str__(self):
        return f"{self.lang}-{self.split}-{self.chunk_num}"

    def _tuple_key(self):
        return self.lang, CodeSearchDataPath.ORDER[self.split], self.chunk_num

    def __lt__(self, other):
        return self._tuple_key() < other._tuple_key()


class CodeSearchChunk(dt.Dataset):
    def __init__(self, path, max_picked=None):
        super().__init__()
        if not isinstance(path, CodeSearchDataPath):
            path = CodeSearchDataPath(path)
        self.path = path
        if max_picked is not None:
            pick_indexes = range(max_picked)
        else:
            pick_indexes = count(0)
        logging.debug(f"loading {path}")
        with gzip.open(self.path.full_path, "r") as f:
            self.data = [self.preprocess(line)
                         for _, line in zip(pick_indexes, f)]
        logging.debug(f"loading vocab for {path}")
        self.code_vocab = Counter(
            word
            for sample in self.data
            for word in sample.code_tokens
        )
        

    def preprocess(self, line):
        item_template = json.loads(line)
        item = AttributeDict(item_template)

        # convert code to bytes
        item.code = item.code.encode("utf-8")

        # get ast
        parser = get_parser(item.language)
        ast = parser.parse(item.code)

        text_ctx = []
        for cur in node_cursor_iter(ast.walk()):
            cur: TreeCursor
            if not cur.node.children:
                node = cur.node
                node: Node
                text_ctx.append(
                    (cur.current_field_name(),
                     node.type,
                     item.code[node.start_byte:node.end_byte].decode('utf-8'),
                     )
                )

        item.text_ctx = text_ctx

        return item

    def __len__(self):
        return self.path.precomputed_len if self.path.precomputed_len is not None else len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def code_search_chunk_get_func(datapath: CodeSearchDataPath):
    return CodeSearchChunk(datapath)


class CodeSearchChunkPool():
    def __init__(self, root_path=DATA_DIR, chunk_full_len=30000, max_chunks_in_memory=None, using_percent=0.6):
        self.root_path = os.path.abspath(root_path)
        init(root_path)
        dataset_paths = [
            CodeSearchDataPath(path)
            for path in glob.glob("**/*.jsonl.gz", recursive=True)
        ]
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

        self.using_percent = using_percent
        self.max_cache = max_chunks_in_memory or self.estimate_max_cache_count_by_system_memory()
        logging.debug(f"max cache num: {self.max_cache}")

        self.get_func = ring.lru(maxsize=self.max_cache)(
            code_search_chunk_get_func)

    def get_by_path(self, datapath: CodeSearchDataPath):
        get_func = self.get_func
        return get_func(datapath)

    def estimate_max_cache_count_by_system_memory(self, estimate_count=50):
        datapath = self.dataset_paths[0]
        using_percent = self.using_percent
        _, available, *_ = psutil.virtual_memory()
        test_item = CodeSearchChunk(datapath, max_picked=estimate_count)
        # TODO using pympler to estimate full size
        data_size = asizeof.asizeof(test_item)
        logging.debug(
            f"size:{data_size}, len:{len(test_item)}, available:{available}")
        actual_chunk_estimated = data_size / \
            len(test_item) * self.chunk_full_len
        return math.floor(available * using_percent / actual_chunk_estimated)

    def __getitem__(self, lang_split_chunk_triple):
        lang, split, chunk = lang_split_chunk_triple
        return self.get_by_path(self.path_map[(lang, split, chunk)])


def load_from_pool(pool_path_tuple):
    pool, path = pool_path_tuple
    return pool.get_by_path(path)


class LazyLoadCodeSearchChunk(dt.Dataset):
    def __init__(self, path: CodeSearchDataPath, pool: CodeSearchChunkPool):
        super().__init__()
        self.path = path
        self.pool = pool

    def __getitem__(self, index):
        return self._load()[index]

    def _load(self):
        return load_from_pool((self.pool, self.path))

    def __len__(self):
        return self.path.precomputed_len or len(self._load())


class CodeSearchDataset(dt.ConcatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @ring.lru()
    @property
    def code_vocab(self):
        vocab = Counter()
        for chunk in self.datasets:
            chunk: CodeSearchChunk
            vocab.update(chunk.code_vocab)
        return vocab


class CodeSearchDatasetLoader():
    def __init__(self, root_path=DATA_DIR, max_chunks_in_memory=None):
        self.root_path = os.path.abspath(root_path)
        init(self.root_path)
        self.pool = CodeSearchChunkPool(
            root_path, max_chunks_in_memory=max_chunks_in_memory)

    @ring.lru()
    @property
    def language(self):
        return list(set(x.lang for x in self.pool.dataset_paths))

    @ring.lru()
    @property
    def split(self):
        return list(set(x.split for x in self.pool.dataset_paths))

    def get(self, language=None, split=None, chunk_slice=None, force_lazy=False):
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
                        *chunk_slice.indices(len(self.pool.dataset_paths)))
                else:
                    raise ValueError(f"invalid chunk index {chunk_slice}")
            return cond
        needed_paths = [
            path for path in self.pool.dataset_paths if is_needed(path)
        ]
        needed_paths.sort()
        lazy_dataset_list = [
            LazyLoadCodeSearchChunk(path, self.pool)
            for path in needed_paths
        ]
        if len(needed_paths) <= self.pool.max_cache and not force_lazy:
            pool_path_tuples_no_cache = [
                path for path in needed_paths if self.pool.get_func.has(path)]
            with multiprocessing.Pool() as pool:
                for chunk in tqdm(
                    pool.imap_unordered(
                        CodeSearchChunk, pool_path_tuples_no_cache),
                    desc=f"loading chunks ({pool._processes}-proc)",
                    total=len(needed_paths)
                ):
                    chunk: CodeSearchChunk
                    self.pool.get_func.set(chunk, chunk.path)
        dataset = dt.ConcatDataset(lazy_dataset_list)
        return dataset


def load_try():
    for language in ('python', 'javascript', 'java', 'ruby', 'php', 'go'):
        if os.path.exists(f'{language}.unzipped'):
            logging.info(f"already unzipped {language} dataset.")
        else:
            if os.path.exists(f'{language}.zip'):
                logging.info(f"already downloaded {language} dataset.")
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
                logging.warn(f"{language}.zip is broken: removed and retry.")
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
