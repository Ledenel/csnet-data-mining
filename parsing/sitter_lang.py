from tree_sitter import Language, Parser

from functools import lru_cache

import pkg_resources
so_path = pkg_resources.resource_filename(__name__, "build/my-languages.so")
print("language_so_path:", so_path)

@lru_cache(10)
def get_parser(name):
    parser = Parser()
    parser.set_language(get_language(name))
    return parser

@lru_cache(10)
def get_language(name):
    return Language(so_path,name)
    
