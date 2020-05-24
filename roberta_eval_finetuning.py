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
from roberta_ast_label_pretrain import FinetuningRoberta, RobertaPretrain


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

import roberta_eval

if __name__ == "__main__":
    load_path = snakemake.input.model
    pretrained_model = RobertaPretrain.load_from_checkpoint(f"{load_path}/{{epoch}}")
    finetuning_model = FinetuningRoberta(roberta_eval.get_hparams(snakemake))
    finetuning_model.model = pretrained_model.model
    roberta_eval.main(snakemake, finetuning_model, hparams_override={
        "method": "roberta-pretrain-with-ast_label",
    })