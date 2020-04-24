import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from dataset_seq import seq_all
import torch
import torch.nn.functional as F
from torch import nn
import seqtools as sq
from tqdm import tqdm
from yummycurry import curry
import numpy as np
from torch.utils.data._utils.collate import default_collate, default_convert
from pytorch_lightning.loggers import WandbLogger
import ast

@curry
def tokenize_plus(tokenizer, text, max_len=None, pad_to_max_length=True):
    if max_len is None:
        max_len = tokenizer.max_len
    encode_dict = tokenizer.encode_plus(  # using encode_plus for single text tokenization (in seqtools.smap).
        text,
        add_special_tokens=True,
        # return_tensors="pt", # pt makes a batched tensor (1,512), flat it to avoid wrong batching
        max_length=max_len,
        pad_to_max_length=pad_to_max_length,
    )
    for key in encode_dict:
        encode_dict[key] = np.array(encode_dict[key])
    return default_convert(encode_dict)
    # move tokenize items into prepare_data / test_dataloader, let batch tensors stay on cuda.


def no_collate(data_list):
    return data_list


def shallow_collate(data_list):
    return [default_collate([x])[0] for x in data_list]


class RobertaCodeQuerySoftmax(pl.LightningModule):
    def __init__(self, inputs, test_batch=1000):
        super().__init__()
        self.datapath = inputs
        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        self.model = AutoModel.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        self.test_batch = test_batch

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def _preload_data(self, file_path, batch_size=1000, max_len=None):
        seqs = seq_all(file_path)
        codes, docs = seqs["codes"], seqs["docs"]
        tok_codes = sq.smap(tokenize_plus(self.tokenizer, max_len=max_len), codes)
        tok_docs = sq.smap(tokenize_plus(self.tokenizer, max_len=max_len), docs)
        return sq.collate([tok_codes, tok_docs])

    def _load(self, file_path, batch_size=1000, max_len=None, **kwargs):
        return DataLoader(self._preload_data(file_path, batch_size=batch_size, max_len=max_len), batch_size=batch_size, **kwargs)

    def test_dataloader(self):
        return self._load(self.datapath.test, batch_size=self.test_batch, collate_fn=no_collate)

    def val_dataloader(self):
        return self._load(self.datapath.valid, collate_fn=no_collate)

    def train_dataloader(self):
        return self._load(self.datapath.train, batch_size=20, max_len=200) #TODO batch_size:64 len:200

    def training_step(self, batch, batch_idx):
        code, query = batch

        _, code_embeddings = self(**code)
        _, query_embeddings = self(**query)
        # dot product for each
        each_code_similarity_per_query = query_embeddings @ code_embeddings.T
        log_softmaxes = nn.LogSoftmax(1)(each_code_similarity_per_query)
        losses = F.nll_loss(log_softmaxes, target=torch.arange(
        len(code_embeddings)).to(log_softmaxes.device), reduce="mean")
        return {
            "loss": losses.mean(),
            "log": {
                "cross_entropy_loss": losses.mean(),
            }
        }
    
    def validation_step(self, batch, batch_idx):
        metrics = self.test_step(batch, batch_idx)
        return {
            "val_loss": metrics["test_loss"],
            "log": metrics
        }

    def validation_epoch_end(self, outputs):
        return {
            "val_loss": torch.stack([x["val_loss"] for x in outputs]).mean()
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5) #TODO: Fine-tuning: lr=1e-5

    def _tokenize(self, inputs, pad_to_max_length=True):
        return self.tokenizer.batch_encode_plus(  # DEPRECATED
            inputs,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.tokenizer.max_len,
            pad_to_max_length=pad_to_max_length,
        )

    def test_step(self, batch, batch_idx, tiny_batch=16):
        code_embeddings = []
        query_embeddings = []
        # print(f"{type(batch[0][0])}")
        for tiny_code, tiny_query in DataLoader(batch, batch_size=tiny_batch, shuffle=False):
            _, code_tiny_embeddings = self(**tiny_code)
            _, query_tiny_embeddings = self(**tiny_query)

            code_embeddings.append(code_tiny_embeddings)
            query_embeddings.append(query_tiny_embeddings)

        code_embeddings = torch.cat(code_embeddings)
        query_embeddings = torch.cat(query_embeddings)

        # dot product for each
        each_code_similarity_per_query = query_embeddings @ code_embeddings.T

        # compute mrr
        correct_scores = torch.diag(each_code_similarity_per_query)
        scores_bigger_mask = each_code_similarity_per_query >= correct_scores.unsqueeze(
            -1)
        scores_bigger_sum = scores_bigger_mask.sum(1).type(torch.float)
        reiprocal_rank = 1 / scores_bigger_sum
        mrr = reiprocal_rank.mean()

        return {
            'test_loss': mrr,
            'rank': scores_bigger_sum,
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([d['test_loss'] for d in outputs]).mean()
        self.logger.experiment.summary["mrr"] = loss
        return {
            'test_loss': loss,
        }


if __name__ == "__main__":
    seed = int(snakemake.params.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fast = snakemake.params.fast
#     pct = 0.05 if fast else 1.0
    test_batch_size = 1000#int(1000 * pct)

    fast_str = ("_fast" if fast else "")
    run_name = f"roberta_base_on_{snakemake.wildcards.lang}_{snakemake.wildcards.extra}{fast_str}"
    wandb_logger = pl.loggers.WandbLogger( 
        name=run_name,
        project="csnet-roberta",
    )
    ckpt = pl.callbacks.ModelCheckpoint(filepath='saved_module/'+run_name+'/{epoch}-mrr{val_loss:.4f}', mode="max")
    # print(f"{snakemake.config}")
    if "gpu_ids" in snakemake.config:
        gpu_ids = ast.literal_eval(str(snakemake.config["gpu_ids"]).strip())
    else:
        gpu_ids = 0
    trainer = pl.Trainer(
        gpus=gpu_ids,
        fast_dev_run=fast,
        logger=wandb_logger,
        checkpoint_callbacks=[ckpt]
    )
    model = RobertaCodeQuerySoftmax(snakemake.input, test_batch=test_batch_size)
    trainer.fit(model)
    # TODO: verify that lighting will pick best model evaluated in validation.
    trainer.test(model)
