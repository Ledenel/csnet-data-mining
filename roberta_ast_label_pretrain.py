import ast_label_pretrain
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
from utils import fetch_snakemake_from_latest_run

class RobertaPretrain(ast_label_pretrain.AstLabelPretrain):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        self.model = AutoModel.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True, config=hparams["roberta_config"])

    def forward(self, *args, **kwargs):
        _, embedding = self.model(*args, **kwargs)
        return embedding

#TODO: copied from roberta_eval.py
if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        snakemake = fetch_snakemake_from_latest_run(__file__)

    datapath = snakemake.input
    params = snakemake.params
    seed = int(snakemake.params.seed)
    fast = snakemake.params.fast
    lang = snakemake.wildcards.lang 
    extra = snakemake.wildcards.extra
    if "gpu_ids" in snakemake.config:
        gpu_ids = ast.literal_eval(str(snakemake.config["gpu_ids"]).strip())
    else:
        gpu_ids = 0
#     pct = 0.05 if fast else 1.0
    test_batch_size = 1000#int(1000 * pct)

    fast_str = ("_fast" if fast else "")
    run_name = f"roberta_ast_label_pretrain_on_{lang}_{extra}{fast_str}"
    wandb_logger = pl.loggers.WandbLogger( 
        name=run_name,
        project="csnet-roberta",
    )
    config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
    ckpt = pl.callbacks.ModelCheckpoint(filepath='saved_module/'+run_name+'/{epoch}-mrr{val_loss:.4f}', mode="min")
    # print(f"{snakemake.config}")


    hparams = {
        "dev_mode": fast,
        "seed": seed,
        "roberta_config": config.to_dict(),
        "datapath": datapath,
        "test_batch": 1000,
        "train_batch": 64,
        "train_max_len": 64,
        "snake_params": params,
    }
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = pl.Trainer(
        gpus=gpu_ids,
        fast_dev_run=fast,
        logger=wandb_logger,
        checkpoint_callbacks=[ckpt],
        max_epochs=1
        # amp_level='O1',
    )
    model = RobertaPretrain(hparams)
    trainer.fit(model)

