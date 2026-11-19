from __future__ import annotations
import torch
import dataclasses
from typing import Optional
from scene_builder._pdm.dtypes import deco


@deco.def_to
@deco.def_replace
@deco.def_shape(field='tensor')
@deco.def_device(field='tensor')
@dataclasses.dataclass
class Metric:
    tensor: torch.Tensor = torch.empty((0,))
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)

    def mean(self, dim: Optional[int]=None) -> torch.Tensor:
        tensor_sum = torch.where(self.valid, self.tensor, 0.0).sum(dim=dim)
        valid_sum = self.valid.sum(dim=dim).clamp(min=1)
        mean = tensor_sum / valid_sum
        return mean

    def max(self, dim: Optional[int]=None, inf: float=1e+3) -> torch.Tensor:
        filled = torch.where(
            condition = self.valid,
            input = self.tensor,
            other = -inf,
        )
        x = filled.max(dim=dim)
        if dim is not None:
            x = x[0]
        y = Metric(
            tensor = x,
            valid = x > -inf,
        )
        return y
