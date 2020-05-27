from tree_sitter import Language, Parser

from functools import lru_cache

import subprocess
import pkg_resources
import re

import logging

import os
so_choice = "build/my-languages.so"
if os.name == "nt":
    so_choice = "build/my-languages-win.so"

so_path = pkg_resources.resource_filename(__name__, so_choice)
# logging.info("language_so_path:{}", so_path)

@lru_cache(10)
def get_parser(name):
    parser = Parser()
    parser.set_language(get_language(name))
    return parser

def get_symbols(path, symbol_type="T"):
    cmd = ["nm", "-p", "-D", path]
    nm_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    symbol_text, error = nm_process.communicate()
    if nm_process.returncode != 0:
        raise ValueError(f"command {cmd} failed: {error}")
    for line in symbol_text.splitlines(keepends=False):
        splited = line.split()
        typ = splited[-2]
        name = splited[-1]
        if typ == symbol_type:
            yield name

def get_available_languages_iter():
    reg = re.compile("tree_sitter_(.*)")
    for symbol in get_symbols(so_path):
        if symbol.startswith("tree_sitter_") and "external_scanner" not in symbol:
            yield reg.match(symbol).group(1)

@lru_cache(1)
def get_available_languages():
    return list(get_available_languages_iter())

@lru_cache(10)
def get_language(name):
    return Language(so_path,name)

# logging.info(f"available {get_available_languages()}")
