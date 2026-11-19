from __future__ import annotations
import torch
import dataclasses
from scene_builder._pdm.utils import tree


def def_to(cls):
    def recursive_to(data, device: torch.DeviceObjType):
        def apply_to(x, **kwargs):
            if hasattr(x, 'to'):
                return x.to(device=device)
            return x
        return tree.map(apply_to, data)
    cls.to = recursive_to
    return cls


def def_replace(cls):
    def replace(self, **kwargs):
        return self.__class__(**{**self.__dict__, **kwargs})
    cls.replace = replace
    return cls


def def_index_select(cls):
    def index_select(self, dim: int, index: torch.Tensor|list[int]):
        if isinstance(index, torch.Tensor):
            index = index.to(dtype=torch.long)
        else:
            index = torch.as_tensor(index, dtype=torch.long)

        def _index_select(x: torch.Tensor):
            if x is None or x.numel() == 0:
                return x
            return x.index_select(dim=dim, index=index.to(device=x.device))

        return self.__class__(**{
            field.name: _index_select(getattr(self, field.name))
            for field in dataclasses.fields(self)
        })
    cls.index_select = index_select
    return cls


def def_flatten(cls):
    def flatten(self, start_dim: int, end_dim: int):
        return self.__class__(**{
            field.name: getattr(self, field.name).flatten(start_dim=start_dim, end_dim=end_dim)
            for field in dataclasses.fields(self)
        })
    cls.flatten = flatten
    return cls


def def_shape(field: str):
    def decorator(cls):
        @property
        def shape(self):
            return getattr(self, field).shape
        cls.shape = shape
        return cls
    return decorator


def def_device(field: str):
    def decorator(cls):
        @property
        def device(self):
            return getattr(self, field).device
        cls.device = device
        return cls
    return decorator


def def_from_dict(cls):
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    cls.from_dict = from_dict
    return cls
