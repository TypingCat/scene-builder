from __future__ import annotations
import torch
import dataclasses
from scene_builder._pdm.dtypes import deco


def _align_batch(tensor: torch.Tensor, batch_rank: int, target_batch_rank: int) -> torch.Tensor:
    """Inject size-1 dims between tensor's batch prefix and its trailing
    matrix/vector dims so that batch broadcasts against a target with
    ``target_batch_rank`` batch dims.
    """
    extra = target_batch_rank - batch_rank
    if extra < 0:
        raise ValueError(
            f"target batch rank {target_batch_rank} < matrix batch rank {batch_rank}"
        )
    if extra == 0:
        return tensor
    base = tensor.shape[:batch_rank]
    tail = tensor.shape[batch_rank:]
    return tensor.reshape(*base, *([1] * extra), *tail)


@deco.def_to
@deco.def_replace
@deco.def_flatten
@deco.def_shape(field='inverse_matrix')
@deco.def_device(field='inverse_matrix')
@dataclasses.dataclass
class Transform:
    inverse_matrix: torch.Tensor

    @classmethod
    def from_pose(
        cls,
        pos_xy: torch.Tensor,
        dir_xy: torch.Tensor,
    ) -> torch.Tensor:
        inverse_matrix = torch.stack([
              dir_xy[..., 0], dir_xy[..., 1], - pos_xy[..., 0]*dir_xy[..., 0] - pos_xy[..., 1]*dir_xy[..., 1],
            - dir_xy[..., 1], dir_xy[..., 0],   pos_xy[..., 0]*dir_xy[..., 1] - pos_xy[..., 1]*dir_xy[..., 0],
        ], dim=-1)
        inverse_matrix = inverse_matrix.reshape(*inverse_matrix.shape[:-1], 2, 3)
        return cls(inverse_matrix=inverse_matrix)

    def get_pos_xy(
        self,
        pos_xy: torch.Tensor,
    ) -> torch.Tensor:
        rot = self.inverse_matrix[..., :2, :2]
        trans = self.inverse_matrix[..., :2, 2]
        batch_rank = rot.ndim - 2
        target_batch_rank = pos_xy.ndim - 1
        rot = _align_batch(rot, batch_rank, target_batch_rank)
        trans = _align_batch(trans, batch_rank, target_batch_rank)
        return (pos_xy.unsqueeze(-2) @ rot.transpose(-1, -2)).squeeze(-2) + trans

    def get_pos_xy_world(
        self,
        pos_xy_local: torch.Tensor,
    ) -> torch.Tensor:
        rot = self.inverse_matrix[..., :2, :2]
        trans = self.inverse_matrix[..., :2, 2]
        pos_origin = -(trans.unsqueeze(-2) @ rot).squeeze(-2)
        batch_rank = rot.ndim - 2
        target_batch_rank = pos_xy_local.ndim - 1
        rot = _align_batch(rot, batch_rank, target_batch_rank)
        pos_origin = _align_batch(pos_origin, batch_rank, target_batch_rank)
        return (pos_xy_local.unsqueeze(-2) @ rot).squeeze(-2) + pos_origin

    def get_dir_xy(
        self,
        dir_xy: torch.Tensor,
    ) -> torch.Tensor:
        rot = self.inverse_matrix[..., :2, :2]
        batch_rank = rot.ndim - 2
        target_batch_rank = dir_xy.ndim - 1
        rot = _align_batch(rot, batch_rank, target_batch_rank)
        return (dir_xy.unsqueeze(-2) @ rot.transpose(-1, -2)).squeeze(-2)

    def get_dir_xy_world(
        self,
        dir_xy_local: torch.Tensor,
    ) -> torch.Tensor:
        rot = self.inverse_matrix[..., :2, :2]
        batch_rank = rot.ndim - 2
        target_batch_rank = dir_xy_local.ndim - 1
        rot = _align_batch(rot, batch_rank, target_batch_rank)
        return (dir_xy_local.unsqueeze(-2) @ rot).squeeze(-2)


def pick_ego_route(
    paths,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`scene.paths` 또는 reference['paths'] 에서 ego 의 route polyline 추출.

    `is_route=True` 인 path 중 valid point 개수 최대 row 선택. 다중 path 를
    flatten 하면 path 경계에서 가상 segment 가 생겨 projection 오류를
    유발하므로 단일 연속 polyline 만 쓴다.

    `paths` 는 `feature.Path` dataclass 또는 dict — `pos_xy/dir_xy/valid/is_route/near_link`
    필드에 동일하게 접근.

    Returns (pos [B, W, 2], dir [B, W, 2], valid [B, W], near_link [B, W],
             has_route [B]).
    """
    pos_all       = paths['pos_xy']    if isinstance(paths, dict) else paths.pos_xy
    dir_all       = paths['dir_xy']    if isinstance(paths, dict) else paths.dir_xy
    valid_all     = paths['valid']     if isinstance(paths, dict) else paths.valid
    is_route      = paths['is_route']  if isinstance(paths, dict) else paths.is_route
    near_link_all = paths['near_link'] if isinstance(paths, dict) else paths.near_link

    B = valid_all.shape[0]
    device = pos_all.device

    if is_route.dim() >= valid_all.dim():
        is_route = is_route.any(dim=-1)

    route_count = (valid_all & is_route.unsqueeze(-1)).sum(dim=-1)
    has_route = route_count.sum(dim=-1) > 0

    best_k = route_count.argmax(dim=-1)
    bi = torch.arange(B, device=device)

    pos = pos_all[bi, best_k]
    dir_ = dir_all[bi, best_k]
    valid = valid_all[bi, best_k] & has_route[:, None]
    near_link = near_link_all[bi, best_k].to(torch.long)
    return pos, dir_, valid, near_link, has_route
