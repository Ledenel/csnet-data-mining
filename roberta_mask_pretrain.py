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


@curry
def tokenize_pair_plus(tokenizer, max_len, pad_to_max_length, text, text_pair):
    if max_len is None:
        max_len = tokenizer.max_len
    encode_dict = tokenizer.encode_plus(  # using encode_plus for single text tokenization (in seqtools.smap).
        text,
        text_pair=text_pair,
        add_special_tokens=True,
        # return_tensors="pt", # pt makes a batched tensor (1,512), flat it to avoid wrong batching
        max_length=max_len,
        pad_to_max_length=pad_to_max_length,
    )
    for key in encode_dict:
        encode_dict[key] = np.array(encode_dict[key])
    return default_convert(encode_dict)
    # move tokenize items into prepare_data / test_dataloader, let batch tensors stay on cuda.


@curry
def random_mask(all_vocab, tok_tensor: torch.Tensor):
    with torch.no_grad():
        # FIXME: add special token mask step at first. do not pick special token as mask.
        vocab = torch.tensor(all_vocab)
        random_toks = vocab[torch.randint(len(vocab), tok_tensor.shape)]
        mask_toks = torch.tensor(tok_tensor).fill_(-100)
        tok_picked = tok_tensor.type(torch.float).fill_(0.15)
        rnd_picked = torch.bernoulli(tok_picked)
        tok_masked = tok_tensor.type(torch.float).fill_(0.8) * rnd_picked
        rnd_masked = torch.bernoulli(tok_masked)
        tok_random_replaced = tok_tensor.type(torch.float).fill_(0.5) * rnd_masked
        rnd_random_replaced = torch.bernoulli(tok_random_replaced)
        return torch.where(
            rnd_picked.type(torch.bool),
            torch.where(
                rnd_random_replaced.type(torch.bool),
                random_toks,
                torch.where(
                    rnd_masked.type(torch.bool),
                    mask_toks,
                    tok_tensor
                )
            ),
            tok_tensor
        )


class MaskPretrain(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.datapath = hparams["datapath"]

    def _preload_data(self, file_path, batch_size=1000, max_len=None):
        seqs = seq_all(file_path)
        codes = seqs["codes"]
        docs = seqs["docs"]
        # TODO try different sample strategy for sub_codes.
        tok_both = sq.smap(
            tokenize_pair_plus(
                self.tokenizer, max_len, True
            ), docs, codes)
        # FIXME: here <PAD> included for random mask.
        tok_only = sq.smap(lambda x: x["input_ids"], tok_both)
        tok_piece_labels = sq.smap(
            random_mask(list(self.tokenizer.get_vocab().values())),
            tok_only
        )
        return sq.collate([tok_both, tok_piece_labels])

    def _load(self, file_path, batch_size=1000, max_len=None, **kwargs):
        return DataLoader(self._preload_data(file_path, batch_size=batch_size, max_len=max_len),
                          batch_size=batch_size, **kwargs)

    def val_dataloader(self):
        return self._load(self.datapath.valid,
                          batch_size=self.hparams["snake_params"].train_batch,
                          max_len=self.hparams["snake_params"].train_max_len)

    def train_dataloader(self):
        return self._load(self.datapath.train,
                          batch_size=self.hparams["snake_params"].train_batch,
                          max_len=self.hparams["snake_params"].train_max_len, shuffle=True)

    def training_step(self, batch, batch_idx):
        code, labels = batch
        masked_lm_loss = self(**code, masked_lm_labels=labels)
        return {"loss": masked_lm_loss}

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


class RobertaMaskPretrain(MaskPretrain):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        # FIXME: using mask lm head in pretraining, but using automodel in finetuning.

        # self.lm_model = AutoModel.from_pretrained(
        #     "huggingface/CodeBERTa-small-v1", resume_download=True, config=hparams["roberta_config"])

        self.model = AutoModelWithLMHead.from_config(
            AutoConfig.from_pretrained(
                "huggingface/CodeBERTa-small-v1", resume_download=True, config=hparams["roberta_config"]
            )
        )

    def forward(self, *args, **kwargs):
        tup = self.model(*args, **kwargs)
        return tup[0]


class FinetuningMaskRoberta(finetuning.CodeQuerySoftmax):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)

    def forward(self, *args, **kwargs):
        _, embedding = self.model.roberta(*args, **kwargs)
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
    label_type = "mask"
    if "gpu_ids" in snakemake.config:
        gpu_ids = ast.literal_eval(str(snakemake.config["gpu_ids"]).strip())
    else:
        gpu_ids = 0
    test_batch_size = 1000  # int(1000 * pct)

    fast_str = ("_fast" if fast else "")
    run_name = f"roberta_mask_pretrain_on_{lang}_{extra}-{label_type}"
    wandb_logger = pl.loggers.WandbLogger(
        name=run_name,
        project="csnet-roberta",
    )
    config = AutoConfig.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
    saved_path = snakemake.output.model  # 'pretrained_module/' + run_name + '/{epoch}'
    ckpt = pl.callbacks.ModelCheckpoint(filepath=saved_path, mode="min")
    # print(f"{snakemake.config}")

    roberta_config = config.to_dict()
    hparams = {
        "dev_mode": fast,
        "seed": seed,
        "roberta_config": roberta_config,
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
        max_epochs=5
        # amp_level='O1',
    )
    model = RobertaMaskPretrain(hparams)
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
