import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from dataset_seq import seq_all
import torch
import torch.nn.functional as F
from torch import nn
import seqtools as sq


class RobertaCodeQuerySoftmax(pl.LightningModule):
    def __init__(self, inputs):
        super().__init__()
        self.datapath = inputs
        self.tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
        self.model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1", resume_download=True)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _load(self, file_path, batch_size=1000):
        seqs = seq_all(file_path)
        codes, docs = seqs["codes"], seqs["docs"]
        return DataLoader(sq.collate([codes, docs]), batch_size=batch_size)

    def test_dataloader(self):
        return self._load(self.datapath.test)

    def train_dataloader(self):
        return self._load(self.datapath.train)
    
    def training_step(self):
        code, query = batch
        code = self._tokenize(code)
        query = self._tokenize(query)
        
        code_embeddings = self(**code)
        query_embeddings = self(**query)


        each_code_similarity_per_query = query_embeddings @ code_embeddings.T # dot product for each
        log_softmaxes = nn.LogSoftmax(1)(each_code_similarity_per_query)
        losses = F.nll_loss(log_softmaxes, target=torch.arange(len(code_embeddings)), reduce="mean")

        return losses.mean()

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def _tokenize(self, inputs, pad_to_max_length=True):
        return self.tokenizer.batch_encode_plus(
            inputs,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.tokenizer.max_len,
            pad_to_max_length=pad_to_max_length,
        )
        

    def test_step(self, batch, batch_idx):
        code, query = batch
        code = self._tokenize(code)
        query = self._tokenize(query)
        
        # print(f"code:{code}")

        _, code_embeddings = self(**code)
        _, query_embeddings = self(**query)

        each_code_similarity_per_query = query_embeddings @ code_embeddings.T # dot product for each
        
        # compute mrr
        correct_scores = torch.diag(each_code_similarity_per_query)
        scores_bigger_mask = each_code_similarity_per_query >= correct_scores.unsqueeze(-1)
        scores_bigger_sum = scores_bigger_mask.sum(1).type(torch.float)
        reiprocal_rank = 1 / scores_bigger_sum
        mrr = reiprocal_rank.mean()
        
        return mrr



if __name__ == "__main__":
    model = RobertaCodeQuerySoftmax(snakemake.input)
    trainer = pl.Trainer()
    trainer.test(model)