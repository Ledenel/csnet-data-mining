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


@curry
def tokenize_plus(tokenizer, text, pad_to_max_length=True):
    encode_dict = tokenizer.encode_plus(  # using encode_plus for single text tokenization (in seqtools.smap).
        text,
        add_special_tokens=True,
        # return_tensors="pt", # pt makes a batched tensor (1,512), flat it to avoid wrong batching
        max_length=tokenizer.max_len,
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

    def _load(self, file_path, batch_size=1000, **kwargs):
        seqs = seq_all(file_path)
        codes, docs = seqs["codes"], seqs["docs"]
        tok_codes = sq.smap(tokenize_plus(self.tokenizer), codes)
        tok_docs = sq.smap(tokenize_plus(self.tokenizer), docs)
        return DataLoader(sq.collate([tok_codes, tok_docs]), batch_size=batch_size, **kwargs)

    def test_dataloader(self):
        return self._load(self.datapath.test, batch_size=self.test_batch, collate_fn=no_collate)

    def train_dataloader(self):
        return self._load(self.datapath.train, batch_size=200)

    def training_step(self):
        code, query = batch
        code = self._tokenize(code)
        query = self._tokenize(query)

        code_embeddings = self(**code)
        query_embeddings = self(**query)
        # dot product for each
        each_code_similarity_per_query = query_embeddings @ code_embeddings.T
        log_softmaxes = nn.LogSoftmax(1)(each_code_similarity_per_query)
        losses = F.nll_loss(log_softmaxes, target=torch.arange(
            len(code_embeddings)), reduce="mean")

        return losses.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

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
        for tiny_code, tiny_query in tqdm(DataLoader(batch, batch_size=tiny_batch, shuffle=False)):
            _, code_tiny_embeddings = self(**tiny_code)
            _, query_tiny_embeddings = self(**tiny_query)

            code_embeddings.append(code_tiny_embeddings)
            query_embeddings.append(query_tiny_embeddings)

        code_embeddings = torch.cat(code_embeddings)
        query_embeddings = torch.cat(query_embeddings)

        # for tiny_code, tiny_query in DataLoader()

        # print(f"code:{code}")

        # _, code_embeddings = self(**code) # no more than 12582912000 (in batch 1000, 12GB)
        # _, query_embeddings = self(**query)

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
            "log": {
                "rank": scores_bigger_sum,
                "mrr": mrr,
            }
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([d['test_loss'] for d in outputs]).mean()
        self.logger.experiment.summary["mrr"] = loss
        return {
            'test_loss': loss,
            'log': { #FIXME: only train_step and valid are logged in pytorch lightning
                'test_loss': loss,
            }
        }


if __name__ == "__main__":
    seed = int(snakemake.params.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fast = snakemake.params.fast
#     pct = 0.05 if fast else 1.0
    test_batch_size = 1000#int(1000 * pct)
    model = RobertaCodeQuerySoftmax(snakemake.input, test_batch=test_batch_size)
    fast_str = ("_fast" if fast else "")
    wandb_logger = pl.loggers.WandbLogger( 
        name=f"roberta_base_on_{snakemake.wildcards.lang}_{snakemake.wildcards.extra}{fast_str}",
        project="csnet-roberta",
    )
    trainer = pl.Trainer(
        gpus=[1],
        fast_dev_run=fast,
        logger=wandb_logger,
    )
    trainer.test(model)
