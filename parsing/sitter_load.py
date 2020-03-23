import multiprocessing
multiprocessing.set_start_method("spawn")
from tree_sitter import Language, Parser
import pkg_resources
import os
from glob import glob

dir_root = pkg_resources.resource_filename(__name__, "sitter-compo")
dir_asterick = os.path.join(os.path.abspath(dir_root), "*")
print(dir_asterick)
dirs = glob(dir_asterick,recursive=False)

print(dirs)
Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  dirs
)

