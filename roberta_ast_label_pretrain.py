import pandas as pd
from torch.utils.data import DataLoader

import ast_label_pretrain
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoConfig

from ast_label_pretrain import fetch_code_pieces, seq_from_code_ast, label_tokenize
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

from finetuning import tokenize_plus
from utils import fetch_snakemake_from_latest_run
import finetuning
import roberta_eval


class LinearLabelSoftmax(torch.nn.Module):
    def __init__(self, embedding_len, label_len):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.linear = torch.nn.Linear(embedding_len, label_len)

    def forward(self, embedding, label):
        return self.loss(
            self.linear(embedding),
            label
        )

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)


class LinearAllLabelSigmoid(torch.nn.Module):
    def __init__(self, embedding_len, label_len):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.linear = torch.nn.Linear(embedding_len, label_len)

    def forward(self, embedding, label):
        return self.loss(
            self.linear(embedding),
            label
        )

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)


class LabelTokenizer:
    def __init__(self, path, embedding_len, mode="least_parent"):
        df = pd.read_csv(path, header=[0, 1])
        self.mapper = {v: i for i, v in enumerate(df[df.columns[0]])}
        self.mode = mode
        self.embedding_len = embedding_len
        self.single_softmax = LinearLabelSoftmax(embedding_len, len(self.mapper))
        self.all_sigmoid = LinearAllLabelSigmoid(embedding_len, len(self.mapper))

    def loss_module(self):
        if self.mode == "least_parent":
            return self.single_softmax
        elif self.mode == "all_except_one_parent":
            return self.all_sigmoid
        raise ValueError(f"unrecognized mode {self.mode}.")

    def least_parent_tensor(self, labels):
        return torch.tensor(self.mapper[labels[0]])

    def all_except_one_parent_tensor(self, labels):
        with torch.no_grad():
            label_tensor = torch.zeros(len(self.mapper))
            labels = labels[:-1]
            for label in labels:
                label_tensor[self.mapper[label]] = 1
            return label_tensor

    def process(self, labels):
        mode_func = getattr(self, f"{self.mode}_tensor")
        return mode_func(labels)


class AstLabelPretrain(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.datapath = hparams["datapath"]
        self.label_tokenizer = LabelTokenizer(self.datapath.label_summary, hparams["roberta_config"]["hidden_size"],
                                              mode=hparams["snake_params"].label_mode)
        self.loss_module = self.label_tokenizer.loss_module()

    def _preload_data(self, file_path, label_file_path, batch_size=1000, max_len=None):
        seqs = seq_all(file_path)
        codes = seqs["codes"]
        label_index_df = pd.read_pickle(label_file_path)
        piece_labels = label_index_df["label"]
        indexes = label_index_df["index"]
        sample_ids = label_index_df["sample_id"]
        code_pieces = fetch_code_pieces(codes, sample_ids, indexes)

        ast_label_seqs = seq_from_code_ast(seqs)
        sub_codes, labels = ast_label_seqs["sub_code_pieces"], ast_label_seqs[self.hparams["snake_params"].label_type]

        # TODO try different sample strategy for sub_codes.
        # code_pieces, piece_labels = sq.concatenate(sub_codes), sq.concatenate(labels)
        # code_pieces = sq.smap(utf8decode, code_pieces)
        tok_codes = sq.smap(tokenize_plus(self.tokenizer, max_len, True), code_pieces)
        tok_piece_labels = sq.smap(label_tokenize(self.label_tokenizer), piece_labels)
        return sq.collate([tok_codes, tok_piece_labels])

    def _load(self, file_path, label_file_path, batch_size=1000, max_len=None, **kwargs):
        return DataLoader(self._preload_data(file_path, label_file_path, batch_size=batch_size, max_len=max_len),
                          batch_size=batch_size, **kwargs)

    def val_dataloader(self):
        return self._load(self.datapath.valid, self.datapath.valid_label,
                          batch_size=self.hparams["snake_params"].train_batch,
                          max_len=self.hparams["snake_params"].train_max_len)

    def train_dataloader(self):
        return self._load(self.datapath.train, self.datapath.train_label,
                          batch_size=self.hparams["snake_params"].train_batch,
                          max_len=self.hparams["snake_params"].train_max_len, shuffle=True)

    def training_step(self, batch, batch_idx):
        code, label = batch
        code_embeddings = self(**code)
        loss = self.loss_module(code_embeddings, label)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        collate_loss = default_collate(outputs)
        val_loss = collate_loss["loss"].mean()
        return {
            "val_loss": val_loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)  # TODO: Fine-tuning: lr=1e-5


class RobertaPretrain(AstLabelPretrain):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        self.model = AutoModel.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True, config=hparams["roberta_config"])

    def forward(self, *args, **kwargs):
        _, embedding = self.model(*args, **kwargs)
        return embedding


class FinetuningRoberta(finetuning.CodeQuerySoftmax):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)

    def forward(self, *args, **kwargs):
        _, embedding = self.model(*args, **kwargs)
        return embedding


# TODO: copied from roberta_eval.py
if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        snakemake = fetch_snakemake_from_latest_run(__file__)

    datapath = snakemake.input
    params = snakemake.params
    seed = int(snakemake.params.seed)
    fast = "dev_fast" in snakemake.config and int(snakemake.config["dev_fast"])
    lang = snakemake.wildcards.lang
    extra = snakemake.wildcards.extra
    label_type = snakemake.params.label_type
    if "gpu_ids" in snakemake.config:
        gpu_ids = ast.literal_eval(str(snakemake.config["gpu_ids"]).strip())
    else:
        gpu_ids = 0
    test_batch_size = 1000  # int(1000 * pct)

    fast_str = ("_fast" if fast else "")
    run_name = f"roberta_ast_label_pretrain_on_{lang}_{extra}-{label_type}"
    wandb_logger = pl.loggers.WandbLogger(
        name=run_name,
        project="csnet-roberta",
    )
    config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
    saved_path = snakemake.output.model  # 'pretrained_module/' + run_name + '/{epoch}'
    ckpt = pl.callbacks.ModelCheckpoint(filepath=saved_path, mode="min")
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
        # logger=wandb_logger,
        # checkpoint_callback=[ckpt],
        max_epochs=1
        # amp_level='O1',
    )
    model = RobertaPretrain(hparams)
    trainer.fit(model)
    trainer.save_checkpoint(saved_path)
    # load_path = snakemake.input.model
    # pretrained_model = RobertaPretrain.load_from_checkpoint(load_path)
    # model = FinetuningRoberta()
    # finetuning_model = FinetuningRoberta(roberta_eval.get_hparams(snakemake))
    # finetuning_model.model = model.model
    # finetuning_model.tokenizer = model.tokenizer
    # roberta_eval.main(snakemake, finetuning_model, hparams_override={
    #     "method": "roberta-pretrain-with-ast_label",
    # })
