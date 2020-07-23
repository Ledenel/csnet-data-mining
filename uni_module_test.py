from collections import OrderedDict, defaultdict
from functools import wraps

import torch
from torch import nn, tensor
import inspect


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
        self.__wrapped__ = func

    def forward(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


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


def satisfy_args(func, args, kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args)
    picked_kwargs = {}
    for param in sig.parameters.values():
        if param.name not in bound_args.arguments \
                and param.kind is not inspect.Parameter.VAR_POSITIONAL \
                and param.kind is not inspect.Parameter.VAR_KEYWORD \
                and param.kind is not inspect.Parameter.POSITIONAL_ONLY:
            if param.name in kwargs:
                picked_kwargs[param.name] = kwargs[param.name]
            elif param.default is inspect.Parameter.empty:
                return bound_args.args, None
    for name in bound_args.kwargs:
        if name in picked_kwargs:
            raise TypeError("multiple values for argument '{}'".format(name))
    picked_kwargs.update(bound_args.kwargs)
    return bound_args.args, picked_kwargs


class FillDefaultDict(nn.ModuleDict):
    def __init__(self, **kw_funcs):
        kw = OrderedDict([(k, PureFunc(v)) for k, v in kw_funcs.items()])
        super().__init__(kw)

    def forward(self, src):
        if isinstance(src, dict):
            args, kwargs = [], dict(src)
        else:
            args, kwargs = [src], {}

        is_dirty = True
        while is_dirty:
            is_dirty = False
            for key, func in self.items():
                if key not in kwargs:
                    bound_args, picked_kwargs = satisfy_args(func, args, kwargs)
                    if picked_kwargs is not None:
                        kwargs[key] = func(*bound_args, **picked_kwargs)
                        is_dirty = True
        result_dict = OrderedDict()
        for key in self:
            if key not in kwargs:
                raise ValueError("Could not satisfy default value for '{}'".format(key))
            else:
                result_dict[key] = kwargs[key]
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


def as_kwargs(func):
    def _kwarg_caller(**kwargs):
        return func(kwargs)

    return _kwarg_caller()


class DictPipe(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dic: dict):
        out = dict(dic)
        for name, module in self._modules.items():
            if name not in out:
                # TODO: when dealing with pure number key,
                # assume inner module is a plain function, input is a plain tensor?
                # but we just output more, which can be done by IntermediateLayerGetter
                out_submodule = module(out)
                # TODO: is out["a.b"] better than out["a"]["b"]?
                if isinstance(out_submodule, dict):
                    out_submodule = {
                        "{}.{}".format(name, key): item
                        for key, item in out_submodule.items()
                    }
                    out.update(out_submodule)
                else:
                    out[name] = out_submodule

        return out


class MyBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()

    def _embeddings(self, hidden_size, layer_norm_eps, hidden_dropout_prob, vocab_size, pad_token_id,
                    type_vocab_size, max_position_embeddings):
        return named_sequential(
            id_filling=FillDefaultDict(
                input_ids=lambda input_ids: input_ids,
                position_ids=lambda input_ids: torch.arange(
                    input_ids.shape[1], dtype=torch.long,
                    device=input_ids.device
                ).unsqueeze(0).expand(input_ids.size()),
                token_type_ids=lambda input_ids: torch.zeros(
                    input_ids.size(), dtype=torch.long,
                    device=input_ids.device
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

    def _layer(self, intermediate_size, hidden_size, hidden_dropout_prob,
               layer_norm_eps):
        """
        10  | model.encoder.layer.0.attention                  | BertAttention     | 2 M
        20  | model.encoder.layer.0.intermediate               | BertIntermediate  | 2 M
        21  | model.encoder.layer.0.intermediate.dense         | Linear            | 2 M
        22  | model.encoder.layer.0.output                     | BertOutput        | 2 M
        23  | model.encoder.layer.0.output.dense               | Linear            | 2 M
        24  | model.encoder.layer.0.output.LayerNorm           | LayerNorm         | 1 K
        25  | model.encoder.layer.0.output.dropout
        """
        return named_sequential(
            attention=self._attention(),
            output=Residual(named_sequential(
                intermediate=self._intermediate(),
                # FIXME: expected dense dropout, actual dense_transform.dense dense_trasform.dropout
                dense=nn.Linear(intermediate_size, hidden_size),
                # FIXME: expected LayerNorm, actual layer_norm
                dropout=nn.Dropout(hidden_dropout_prob),
            )),
            layer_norm=nn.LayerNorm(hidden_size, eps=layer_norm_eps),
        )

    def _attention(self):
        pass

    def _intermediate(self):
        pass

    # def _layer_output(self, intermediate_size, hidden_size, hidden_dropout_prob,
    #                   layer_norm_eps):  # (intermediate_output, attention_output)
    #     return named_sequential(
    #         # FIXME: expected dense dropout, actual dense_transform.dense dense_trasform.dropout
    #         dense_transform=Residual(named_sequential(
    #             dense=nn.Linear(intermediate_size, hidden_size),
    #             # FIXME: expected LayerNorm, actual layer_norm
    #             dropout=nn.Dropout(hidden_dropout_prob),
    #         )),
    #         layer_norm=nn.LayerNorm(hidden_size, eps=layer_norm_eps),
    #     )

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
