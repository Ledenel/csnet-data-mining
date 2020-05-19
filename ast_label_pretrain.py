from parsing.sitter_lang import get_parser
from tree_sitter import Node, TreeCursor
import seqtools as sq
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from dataset_seq import seq_all

def node_cursor_iter(cursor):
    yield cursor.copy()
    if cursor.goto_first_child():
        yield from node_cursor_iter(cursor)
        while cursor.goto_next_sibling():
            yield from node_cursor_iter(cursor)
        cursor.goto_parent()


def self_chain(cur):
    field_name = cur.current_field_name()
    yield field_name, cur.copy()
    while cur.goto_parent():
        yield field_name, cur.copy()
        field_name = cur.current_field_name()


def node_asts(ast):
    return [
        cursor.node
        for cursor in node_cursor_iter(ast.walk())
    ]


def _sub_code_pieces(ast: Node, code):
    return [
        code[node.start_byte:node.end_byte]
        for node in node_asts(ast)
    ]


def _sub_labels(ast):
    return [
        [
            (name, parent.node.type)
            for name, parent in self_chain(cursor)
        ]
        for cursor in node_cursor_iter(ast.walk())
    ]


def seq_from_code_ast(_seq_dict):
    _code_bytes = _seq_dict["code_bytes"]
    _asts = _seq_dict["asts"]
    sub_code_pieces = sq.smap(_sub_code_pieces, _asts, _code_bytes)
    # sub_asts = sq.smap(_sub_labels, _asts)
    sub_labels = sq.smap(_sub_labels, _asts)
    type_label = sq.smap(
        lambda lbs: [[x[1] for x in labels] for labels in lbs],
        sub_labels
    )
    combined_label = sq.smap(
        lambda lbs: [[f"{x[0]}-{x[1]}" for x in labels] for labels in lbs],
        sub_labels
    )

    _dict_all = locals()
    _dict_return = {k: v for k, v in _dict_all.items()
                    if not k.startswith("_")}
    # print(_dict_return.keys())
    return _dict_return

import torch

class LabelTokenizer:
    def __init__(self, path, mode="least_parent"):
        df = pd.read_csv(path, header=[0,1])
        self.mapper = {v:i for i,v in enumerate(df[df.columns[0]])}
        self.mode = mode

    def loss_module(self):
        if mode == "least_parent":
            return torch.nn.CrossEntropyLoss()
        raise ValueError(f"unrecognized mode {self.mode}.")

    def least_parent_tensor(self, labels):
        return torch.tensor(self.mapper[labels[0]])
            
    def process(self, labels):
        mode_func = getattr(self, f"{self.mode}_tensor")
        return mode_func(labels)

from pymonad.Reader import curry
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

@curry
def label_tokenize(label_tokenizer, tensor):
    return label_tokenizer.process(tensor)

# TODO copied from finetuning.py
class AstLabelPretrain(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.datapath = hparams["datapath"]
        self.label_tokenizer = LabelTokenizer(self.datapath.label_summary, mode=hparams["snake_params"].label_mode)
        
    def _preload_data(self, file_path, batch_size=1000, max_len=None):
        seqs = seq_all(file_path)
        ast_label_seqs = seq_from_code_ast(seqs)
        sub_codes, labels = ast_label_seqs["sub_code_pieces"], ast_label_seqs[self.hparams["snake_params"].label_type]
        #TODO try different sample strategy for sub_codes.
        code_pieces, piece_labels = sq.concatenate(sub_codes), sq.concatenate(labels)
        tok_codes = sq.smap(tokenize_plus(self.tokenizer, max_len, True), code_pieces)
        tok_piece_labels = sq.smap(label_tokenize(self.label_tokenizer), piece_labels)
        return sq.collate([tok_codes, tok_piece_labels])

    def _load(self, file_path, batch_size=1000, max_len=None, **kwargs):
        return DataLoader(self._preload_data(file_path, batch_size=batch_size, max_len=max_len), batch_size=batch_size, **kwargs)

    def val_dataloader(self):
        return self._load(self.datapath.valid)

    def train_dataloader(self):
        return self._load(self.datapath.train, batch_size=self.hparams["snake_params"].train_batch, shuffle=True)

    def training_step(self, batch, batch_idx):
        code, label = batch
        code_embeddings = self(**code)
        loss = self.label_tokenizer.loss_module()(code_embeddings, label)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        collate_loss = default_collate(outputs)
        val_loss = collate_loss.mean()
        return {
            "val_loss": val_loss,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5) #TODO: Fine-tuning: lr=1e-5


