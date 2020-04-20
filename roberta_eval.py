import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from dataset_seq import seq_all
import torch
import torch.nn.functional as F
from torch import nn
import seqtools as sq
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

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
        return self._load(self.datapath.train, batch_size=200)
    
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
        
        code_embeddings = []
        query_embeddings = []

        for tiny_code, tiny_query in tqdm(sq.batch(sq.collate([code, query]), k=50, collate_fn=default_collate)):
            
            code = self._tokenize(tiny_code)
            query = self._tokenize(tiny_query)
            
            _, code_tiny_embeddings = self(**code) 
            _, query_tiny_embeddings = self(**query)

            code_embeddings.append(code_tiny_embeddings)
            query_embeddings.append(query_tiny_embeddings)
        
        code_embeddings = torch.cat(code_embeddings)
        query_embeddings = torch.cat(query_embeddings)

        # for tiny_code, tiny_query in DataLoader()

        # print(f"code:{code}")

        # _, code_embeddings = self(**code) # FIXME: no more than 12582912000 (in batch 1000, 12GB)
        # _, query_embeddings = self(**query)

        each_code_similarity_per_query = query_embeddings @ code_embeddings.T # dot product for each
        
        # compute mrr
        correct_scores = torch.diag(each_code_similarity_per_query)
        scores_bigger_mask = each_code_similarity_per_query >= correct_scores.unsqueeze(-1)
        scores_bigger_sum = scores_bigger_mask.sum(1).type(torch.float)
        reiprocal_rank = 1 / scores_bigger_sum
        mrr = reiprocal_rank.mean()
        
        return mrr
    
    def test_epoch_end(self, outputs):
        return {'test_loss': torch.stack(outputs).mean()}



if __name__ == "__main__":
    model = RobertaCodeQuerySoftmax(snakemake.input)
    trainer = pl.Trainer(gpus=0, fast_dev_run=False)
    trainer.test(model)
    