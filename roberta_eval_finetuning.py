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
from roberta_mask_pretrain import RobertaMaskPretrain, FinetuningMaskRoberta


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
    override_dict = {
        "method": "roberta-pretrain-with-ast_label",
        "max_epochs": 5,
    }
    model_cls_name = snakemake.params.get("model_pretrain_cls", "RobertaPretrain")
    model_finetuning_name = snakemake.params.get("model_finetuning_cls", "FinetuningRoberta")
    override_dict["pretrain_name"] = model_cls_name
    override_dict["finetuning_name"] = model_finetuning_name
    pretrain_cls = globals()[model_cls_name]
    fintuning_cls = globals()[model_finetuning_name]
    pretrained_model = pretrain_cls.load_from_checkpoint(load_path)
    override_dict["pretrain"] = pretrained_model.hparams
    finetuning_model = fintuning_cls(roberta_eval.get_hparams(snakemake, hparams_override=override_dict))
    finetuning_model.model = pretrained_model.model
    roberta_eval.main(snakemake, finetuning_model, hparams_override=override_dict)