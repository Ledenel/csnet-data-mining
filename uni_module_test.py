from collections import OrderedDict

import torch
from torch import nn, tensor


def named_sequential(**kwargs):
    return nn.Sequential(OrderedDict(kwargs))


def dict_rename(**rename_table):
    def _rename_func(d: dict):
        return OrderedDict({
            rename_table.get(key, key): value for key, value in d.items()
        })

    return _rename_func


class PureFunc(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Residual(nn.Module):
    def __init__(self, non_linear):
        super().__init__()
        self.non_linear = non_linear

    def forward(self, x):
        return x + self.non_linear(x)


class BroadcastDict(nn.ModuleDict):
    def __init__(self, **kwmodules):
        super().__init__(kwmodules)

    def forward(self, x):
        return OrderedDict(
            [(key, func(x)) for key, func in self.items()]
        )


class BroadcastList(nn.ModuleList):
    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, x):
        return [func(x) for func in self]


class ScatterDict(nn.ModuleDict):
    def __init__(self, **kwmodules):
        super().__init__(kwmodules)

    def forward(self, scatter_dict):
        return OrderedDict(
            [(key, func(scatter_dict[key])) for key, func in self.items()]
        )


class ScatterList(nn.ModuleList):
    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, scatter_list):
        return [func(x) for func, x in zip(self, scatter_list)]


class FillDefaultDict(nn.ModuleDict):
    def __init__(self, **kwmodules):
        super().__init__(kwmodules)

    def forward(self, src_dict_or_x):
        if isinstance(src_dict_or_x, dict):
            src_dict = OrderedDict(src_dict_or_x)
        else:
            src_dict = OrderedDict()
        result_dict = OrderedDict()
        for key, func in self.items():
            if key in src_dict:
                result_dict[key] = src_dict[key]
            else:
                result_dict[key] = func(src_dict_or_x)
        return result_dict


class AppendInput(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return [self.module(x), x]


class SequentialWithInput(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for module in self:
            x = [module(x), x]
        return x


class MyBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def _embeddings(self, hidden_size, layer_norm_eps, hidden_dropout_prob, vocab_size, pad_token_id,
                    type_vocab_size, max_position_embeddings):
        return named_sequential(
            id_filling=BroadcastDict(
                input_ids=lambda x: x,
                position_ids=lambda x: torch.arange(
                    x.shape[1], dtype=torch.long,
                    device=x.device
                ).unsqueeze(0).expand(x.size()),
                token_type_ids=lambda x: torch.zeros(
                    x.size(), dtype=torch.long,
                    device=x.device
                ),
            ),
            rename=PureFunc(dict_rename(
                input_ids="word_embeddings",
                position_ids="position_embeddings",
                token_type_ids="token_type_embeddings",
            )),
            fusion=ScatterDict(
                # FIXME: expected <some>embeddings, actual fusion.<some>embeddings
                word_embeddings=nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id),
                position_embeddings=nn.Embedding(max_position_embeddings, hidden_size),
                token_type_embeddings=nn.Embedding(type_vocab_size, hidden_size),
            ),
            stack=PureFunc(lambda dict: torch.stack(dict.values())),
            reduce=PureFunc(torch.sum),
            # FIXME: expected LayerNorm, actual layer_norm
            layer_norm=nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            dropout=nn.Dropout(hidden_dropout_prob),
        )

    def _encoder(self, num_layers):
        return named_sequential(
            layer=nn.Sequential(
                *[self._layer() for _ in range(num_layers)]
            )
        )

    def _layer(self):
        return named_sequential(
            attention=self._attention(),
            intermediate=AppendInput(self._intermediate()),
            output=self._layer_output(),
        )

    def _attention(self):
        pass

    def _intermediate(self):
        pass

    def _layer_output(self, intermediate_size, hidden_size, hidden_dropout_prob,
                      layer_norm_eps):  # (intermediate_output, attention_output)
        return named_sequential(
            # FIXME: expected dense dropout, actual dense_transform.dense dense_trasform.dropout
            dense_transform=Residual(named_sequential(
                dense=nn.Linear(intermediate_size, hidden_size),
                # FIXME: expected LayerNorm, actual layer_norm
                dropout=nn.Dropout(hidden_dropout_prob),
            )),
            layer_norm=nn.LayerNorm(hidden_size, eps=layer_norm_eps),
        )

    def _pooler(self, hidden_size):
        return named_sequential(
            reduce=PureFunc(lambda hidden: hidden[:, 0]),
            dense=nn.Linear(hidden_size, hidden_size),
            activation=nn.Tanh(),
        )


class SeqContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Linear(4, 8)
        self.act = nn.ReLU6()
        self.ln2 = nn.Linear(8, 4)
        self.layer = nn.Sequential(
            self.ln,
            self.act,
            self.ln2
        )
        self.multi_layer = nn.Sequential(
            *([self.layer] * 3)
        )

    def forward(self, input):
        return self.multi_layer(input)


if __name__ == '__main__':
    model = SeqContainer()
    print(list(model.parameters()))
    print(model)
    print(model(tensor([1.0, 1.0, 1.0, 2.0])))
    model2 = SeqContainer()
    state_dict = model.state_dict()
    state_dict["ln2.bias"] = tensor([1.0, 2, 3, 4])
    model2.load_state_dict(state_dict, strict=False)
    model2.multi_layer[2] = nn.Identity()
    model.multi_layer[1] = nn.Identity()
    print(model)
    print(model(tensor([1.0, 1.0, 1.0, 2.0])))
    print(model2)
    print(model2(tensor([1.0, 1.0, 1.0, 2.0])))
