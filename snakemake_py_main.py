from snakemake import snakemake as sk

if __name__ == "__main__":
    sk("Snakefile", resources={"gpus":1}, config={"gpu_ids":0}, debug=True)