CodeSearchNet data mining.

**for linux only.**

## Reproduce

Required python >= 3.7.

install requirements via `pip install -r requirements.txt`

then snakemake is available. 

run `snakemake -j 4` to reproduce data mining.

you'd like to change 4 to your prefered concurrent job count (e.g. cpu counts).


## Using GPUs

By default it will only use cpu version of torch. to use GPUs, run `snakemake -j 4 --config gpu_ids=[0]` to use target gpu id (you can run `nvidia-smi` to check your gpu id mappings.)
