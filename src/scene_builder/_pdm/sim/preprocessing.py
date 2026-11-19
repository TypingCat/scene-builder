from __future__ import annotations
import torch
from scene_builder._pdm.dtypes import feature
from scene_builder._pdm.utils.geometry import Transform


def compute_scene_transform(
    scene: feature.Scene,
    now: int,
) -> tuple[Transform, torch.Tensor]:
    """Create local-coordinate Transform from ego pose at F_now.

    Args:
        scene: Global-coordinate Scene
        now: 클립 내 현재 프레임 인덱스 (configs.time.now)

    Returns:
        (Transform [B, 2, 3], scene_valid [B])
    """
    F_now = now
    ego_mask = scene.agents.is_ego & scene.agents.valid[:, :, F_now]
    scene_valid = ego_mask.any(dim=1)
    ego_idx = ego_mask.float().argmax(dim=1)

    B = scene.agents.pos_xy.shape[0]
    bi = torch.arange(B, device=scene.agents.pos_xy.device)
    origin_pos = scene.agents.pos_xy[bi, ego_idx, F_now]
    origin_dir = scene.agents.dir_xy[bi, ego_idx, F_now]

    return Transform.from_pose(pos_xy=origin_pos, dir_xy=origin_dir), scene_valid


def transform_scene(
    scene: feature.Scene,
    transform: Transform,
    valid: torch.Tensor,
) -> feature.Scene:
    """Transform global-coordinate Scene to local coordinates.

    Args:
        scene: Global-coordinate Scene
        transform: global->local Transform (inverse_matrix [B, 2, 3])
        valid: ego validity [B]

    Returns:
        Local-coordinate Scene
    """
    vm = valid
    t = Transform(inverse_matrix=transform.inverse_matrix.unsqueeze(1))

    def _tf_posdir(store, has_vel=False):
        """Transform a store with pos_xy/dir_xy if non-empty."""
        if store.valid.numel() == 0:
            return store
        new_valid = store.valid & vm[:, None, None]
        mask = new_valid.unsqueeze(-1)
        kw = dict(
            pos_xy=t.get_pos_xy(store.pos_xy) * mask,
            dir_xy=t.get_dir_xy(store.dir_xy),
            valid=new_valid,
        )
        if has_vel:
            kw["vel_xy"] = t.get_dir_xy(store.vel_xy)
            kw["acc_xy"] = t.get_dir_xy(store.acc_xy)
        return store.replace(**kw)

    def _tf_traffic(store):
        if store.valid.numel() == 0:
            return store
        new_valid = store.valid & vm[:, None, None]

        slot_valid = new_valid.any(dim=-1, keepdim=True)
        kw = dict(
            pos_xy=t.get_pos_xy(store.pos_xy) * slot_valid,
            valid=new_valid,
        )
        if store.dir_xy.numel() > 0:
            kw['dir_xy'] = t.get_dir_xy(store.dir_xy)
        return store.replace(**kw)

    return scene.replace(
        agents=_tf_posdir(scene.agents, has_vel=True),
        links=_tf_posdir(scene.links),
        lanes=_tf_posdir(scene.lanes),
        markers=_tf_posdir(scene.markers),
        traffics=_tf_traffic(scene.traffics),
        paths=_tf_posdir(scene.paths),
    )
