import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoConfig
from dataset_seq import seq_all
import torch
import torch.nn.functional as F
from torch import nn
import seqtools as sq
from tqdm import tqdm
from pymonad.Reader import curry
import numpy as np
from torch.utils.data._utils.collate import default_collate, default_convert
from pytorch_lightning.loggers import WandbLogger
import ast
import finetuning

class RobertaCodeQuerySoftmax(finetuning.CodeQuerySoftmax):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        self.model = AutoModel.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True, config=hparams["roberta_config"])

    def forward(self, *args, **kwargs):
        _, embedding = self.model(*args, **kwargs)
        return embedding

def get_hparams(snakemake):
    seed = int(snakemake.params.seed)

    fast = snakemake.params.fast
#     pct = 0.05 if fast else 1.0
    test_batch_size = 1000#int(1000 * pct)

    fast_str = ("_fast" if fast else "")
    run_name = f"roberta_base_on_{snakemake.wildcards.lang}_{snakemake.wildcards.extra}{fast_str}"
    wandb_logger = pl.loggers.WandbLogger( 
        name=run_name,
        project="csnet-roberta",
    )
    config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
    ckpt = pl.callbacks.ModelCheckpoint(filepath='saved_module/'+run_name+'/{epoch}-mrr{val_loss:.4f}', mode="max")
    # print(f"{snakemake.config}")
    if "gpu_ids" in snakemake.config:
        gpu_ids = ast.literal_eval(str(snakemake.config["gpu_ids"]).strip())
    else:
        gpu_ids = 0

    hparams = {
        "dev_mode": fast,
        "seed": seed,
        "roberta_config": config.to_dict(),
        "datapath": snakemake.input,
        "test_batch": 1000,
        "train_batch": 64,
        "train_max_len": 200,
    }
    return hparams

def main(snakemake, model=None, hparams_override=None):
    seed = int(snakemake.params.seed)

    fast = "dev_fast" in snakemake.config and int(snakemake.config["dev_fast"])
#     pct = 0.05 if fast else 1.0
    test_batch_size = 1000#int(1000 * pct)

    fast_str = ("_fast" if fast else "")
    run_name = f"roberta_base_on_{snakemake.wildcards.lang}_{snakemake.wildcards.extra}{fast_str}"
    wandb_logger = pl.loggers.WandbLogger( 
        name=run_name,
        project="csnet-roberta",
    )
    config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
    ckpt = pl.callbacks.ModelCheckpoint(filepath='saved_module/'+run_name+'/{epoch}-mrr{val_loss:.4f}', mode="max")
    # print(f"{snakemake.config}")
    if "gpu_ids" in snakemake.config:
        gpu_ids = ast.literal_eval(str(snakemake.config["gpu_ids"]).strip())
    else:
        gpu_ids = 0

    hparams = {
        "dev_mode": fast,
        "seed": seed,
        "roberta_config": config.to_dict(),
        "datapath": snakemake.input,
        "test_batch": 1000,
        "train_batch": 64,
        "train_max_len": 200,
        "method": "roberta-pretrain",
    }

    if hparams_override is not None:
        hparams.update(hparams_override)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = pl.Trainer(
        gpus=gpu_ids,
        fast_dev_run=fast,
        logger=wandb_logger,
        checkpoint_callbacks=[ckpt],
        max_epochs=5
        # amp_level='O1',
    )
    if model is None:
        model = RobertaCodeQuerySoftmax(hparams)
    trainer.fit(model)
    # TODO: verify that lighting will pick best model evaluated in validation.
    trainer.test(model)

if __name__ == "__main__":
    main(snakemake)