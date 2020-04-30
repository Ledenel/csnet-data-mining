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

@curry
def tokenize_plus(tokenizer, max_len, pad_to_max_length, text):
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
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        
        self.datapath = hparams["datapath"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True)
        self.model = AutoModel.from_pretrained(
            "huggingface/CodeBERTa-small-v1", resume_download=True, config=hparams["roberta_config"])
        

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def _preload_data(self, file_path, batch_size=1000, max_len=None):
        seqs = seq_all(file_path)
        codes, docs = seqs["codes"], seqs["docs"]
        tok_codes = sq.smap(tokenize_plus(self.tokenizer, max_len, True), codes)
        tok_docs = sq.smap(tokenize_plus(self.tokenizer, max_len, True), docs)
        return sq.collate([tok_codes, tok_docs])

    def _load(self, file_path, batch_size=1000, max_len=None, **kwargs):
        return DataLoader(self._preload_data(file_path, batch_size=batch_size, max_len=max_len), batch_size=batch_size, **kwargs)

    def test_dataloader(self):
        return self._load(self.datapath.test, batch_size=self.hparams['test_batch'], collate_fn=no_collate)

    def val_dataloader(self):
        return self._load(self.datapath.valid, collate_fn=no_collate)

    def train_dataloader(self):
        return self._load(self.datapath.train, batch_size=self.hparams['train_batch'], max_len=self.hparams['train_max_len'], shuffle=True) #TODO batch_size:64 len:200

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
            "val_loss": metrics["mrr"],
            "log": metrics
        }

    def validation_epoch_end(self, outputs):
        collate_loss = default_collate(outputs)
        val_loss = collate_loss["val_loss"].mean()
        return {
            "val_loss": val_loss,
            "log": {
                k:v.mean() for k,v in collate_loss["log"].items()
            }
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
        rank = scores_bigger_sum
        
        result = {
            'mrr': mrr,
        }
        for topk in [1,3,5,10,20,100]:
            result[f"top{topk}-acc"] = (rank <= topk).sum() / float(len(rank))
        return result

    def test_epoch_end(self, outputs):
        collate_loss = default_collate(outputs)
        loss = collate_loss['mrr'].mean()
        for k,v in collate_loss.items():
            self.logger.experiment.summary[k] = v.mean()
        return {
            'test_loss': loss,
        }


if __name__ == "__main__":
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
        "train_batch": 32,
        "train_max_len": 200,
    }
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = pl.Trainer(
        gpus=gpu_ids,
        fast_dev_run=fast,
        logger=wandb_logger,
        checkpoint_callbacks=[ckpt],
        # amp_level='O1',
    )
    model = RobertaCodeQuerySoftmax(hparams)
    trainer.fit(model)
    # TODO: verify that lighting will pick best model evaluated in validation.
    trainer.test(model)
