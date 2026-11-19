from __future__ import annotations
import torch
import dataclasses


def map(func, *datas, key=None):
    if (datas[0] is None) or (len(datas) == 0):
        return None
    elif isinstance(datas[0], torch.Tensor):
        return func(*datas, key=key)
    elif isinstance(datas[0], dict):
        return {key: map(func, *[data[key] for data in datas], key=key) for key in datas[0].keys()}
    elif isinstance(datas[0], (list, tuple)):
        return type(datas[0])(map(func, *[data[i] for data in datas]) for i in range(len(datas[0])))
    elif dataclasses.is_dataclass(datas[0]):
        return type(datas[0])(**{
            field.name: map(func, *[getattr(data, field.name) for data in datas], key=field.name)
            for field in dataclasses.fields(datas[0])
        })
    else:
        return func(*datas)
