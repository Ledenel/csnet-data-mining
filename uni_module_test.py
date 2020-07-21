from collections import OrderedDict

import torch
from torch import nn, tensor


def named_sequential(**kwargs):
    return nn.Sequential(OrderedDict(kwargs))


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


class BroadCastDict(nn.ModuleDict):
    def __init__(self, **kwmodules):
        super().__init__(kwmodules)

    def forward(self, *args, **kwargs):
        return OrderedDict(
            [(key, func(*args, **kwargs)) for key, func in self.items()]
        )


class BroadCastList(nn.ModuleList):
    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        return [func(*args, **kwargs) for func in self]


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def _embeddings(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
        return named_sequential(
            fusion=BroadCastDict(
                # FIXME: expected <some>embeddings, actual fusion.<some>embeddings

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
        pass

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
