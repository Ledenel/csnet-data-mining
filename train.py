import transformers

from torch import nn
import torch

from dataset_seq import seq_all

from pymonad.Reader import curry

# input:
#     train_data = "data_cache/{lang}_train_{extra}.pkl",
# output:
#     done = touch("bert_scratch.done")

input = snakemake.input
output = snakemake.output


def count_all(tokens):
    token_count = Counter()
    token_count["[UNK]"] = 0
    token_count["[PAD]"] = 0
    token_count["[SEP]"] = 0
    token_count["[CLS]"] = 0
    token_count["[MASK]"] = 0
    for sample in tokens:
        token_count.update(sample)
    
    return token_count

def get_counter_map(token_counter):
    return {word: i for i, word in enumerate(token_counter.keys())}

@curry
def tokenize(token_counter, token_list):
    counter_map = get_counter_map(token_counter)
    return torch.tensor([counter_map.get(word, default=counter_map["[UNK]"]) for word in token_list])

@curry
def pad_to_1d(token_counter, size, tensor):
    counter_map = get_counter_map(token_counter)
    if size <= len(tensor):
        return tensor[:size]
    else:
        return torch.F.pad(tensor, size - len(tensor), mode="constant", value=counter_map["[PAD]"])

code_token_counter = count_all(code_tokens)
doc_token_counter = count_all(doc_tokens)

class CodeQuerySoftmaxBertModel(nn.Module):
    def __init__(self, code_token_counter, query_token_counter):
        self.code_token_counter = code_token_counter
        get_counter_map(code_token_counter)
        self.code_config = transformers.BertConfig(
            vocab_size=len(code_token_counter),
            pad_token_id=get_counter_map(code_token_counter)["[PAD]"]
        )
        self.code_model = transformers.BertModel(self.code_config)

        self.query_token_counter = query_token_counter
        self.query_config = transformers.BertConfig(
            vocab_size=len(query_token_counter),
            pad_token_id=get_counter_map(query_token_counter)["[PAD]"]
        )
        self.query_model = transformers.BertModel(self.query_config)
    
    def forward(query_batch, code_batch):
        query_embeddings = self.query_model(query_batch)
        code_embeddings = self.code_model(code_batch)
        each_code_similarity_per_query = query_embeddings @ code_embeddings.T # dot product for each
        log_softmaxes = nn.LogSoftmax(1)(each_code_similarity_per_query)
        losses = torch.F.nll_loss(log_softmaxes, target=torch.range(len(code_embeddings)), reduce="mean")

        # compute mrr
        correct_scores = torch.diag(each_code_similarity_per_query)
        scores_bigger_mask = each_code_similarity_per_query >= correct_scores.unsqueeze(-1)
        scores_bigger_sum = scores_bigger_mask.sum(1).type(torch.float)
        reiprocal_rank = 1 / scores_bigger_sum
        mrr = reiprocal_rank.mean()
        return query_embeddings, code_embeddings, losses, reiprocal_rank, mrr


#train, test and evaluation.

ds_train = seq_all(input.train_data)

code_tokens = ds_train["code_tokens_with_identifier_split"]
doc_tokens = ds_train["doc_tokens_with_identifier_split"]

code_token_counter = count_all(code_tokens)
doc_token_counter = count_all(doc_token_counter)

import seqtools as sq

code_tokenized = sq.smap(tokenize(code_token_counter), code_tokens)
doc_tokenized = sq.smap(tokenize(doc_token_counter), doc_tokens)

code_pad = sq.smap(pad_to_1d(code_token_counter, 200), code_tokenized)
doc_pad = sq.smap(pad_to_1d(doc_token_counter, 200), doc_tokenized)

#batch and pad dataset.

model = CodeQuerySoftmaxBertModel(code_token_counter, doc_token_counter)

opt = torch.optim.Adam(model.parameters())

from tqdm import trange, tqdm

train_data = sq.collate(doc_pad, code_pad)

for epoch in trange(50):
    for query_batch, code_batch in tqdm(torch.utils.data.DataLoader(train_data)):
        opt.zero_grad()
        query_embeddings, code_embeddings, losses, reiprocal_rank, mrr = model(query_batch, code_batch)
        losses.mean().backward()
        opt.step()

#TODO: add validation and early stopping.



#TODO: add test via test-dataset.
        
test_code_pad, test_doc_pad = ... # previous 

for query_batch, code_batch in tqdm(torch.utils.data.DataLoader(test_data)):
    query_embeddings, code_embeddings, losses, reiprocal_rank, mrr = model(query_batch, code_batch)
    # use mrr for testing (validation).







        


        



