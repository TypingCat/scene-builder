from __future__ import annotations
import torch
import dataclasses
from typing import Optional
from scene_builder._pdm.dtypes import deco


class AgentType:
    unknown = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3


AGENT_TYPE = AgentType()


class TrafficState:
    unknown = 0
    go = 1
    caution = 2
    stop = 3

    @staticmethod
    def from_waymo(state: torch.Tensor) -> torch.Tensor:
        """Waymo 9-state → 4-state TrafficState."""
        out = torch.zeros_like(state)
        out = torch.where((state == 3) | (state == 6), out.new_full([], 1), out)
        out = torch.where((state == 2) | (state == 5) | (state == 8), out.new_full([], 2), out)
        out = torch.where((state == 1) | (state == 4) | (state == 7), out.new_full([], 3), out)
        return out


class LinkType:
    unknown = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3

    @staticmethod
    def from_road_type(t: torch.Tensor) -> torch.Tensor:
        """RoadType (1-3) → LinkType. Non-lane-center → 0."""
        out = torch.zeros_like(t)
        out = torch.where((t == 1) | (t == 2), out.new_full([], 1), out)
        out = torch.where(t == 3, out.new_full([], 3), out)
        return out


class LinkSection:
    unknown = 0
    intersection = 1
    school_zone = 2
    tunnel = 3
    hi_pass = 4
    freeway = 5

    @staticmethod
    def from_road_type(t: torch.Tensor) -> torch.Tensor:
        """RoadType → LinkSection. Only freeway(1) maps; rest → unknown."""
        out = torch.zeros_like(t)
        out = torch.where(t == 1, out.new_full([], 5), out)
        return out


@deco.def_to
@deco.def_replace
@deco.def_index_select
@deco.def_from_dict
@deco.def_shape(field='valid')
@deco.def_device(field='valid')
@dataclasses.dataclass
class Agent:
    id: torch.Tensor = torch.empty((0,), dtype=torch.long)
    type: torch.Tensor = torch.empty((0,), dtype=torch.long)
    time_s: torch.Tensor = torch.empty((0,), dtype=torch.float)
    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    vel_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    acc_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    size_xyz: torch.Tensor = torch.empty((0,), dtype=torch.float)
    is_ego: torch.Tensor = torch.empty((0,), dtype=torch.bool)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)


@deco.def_to
@deco.def_replace
@deco.def_index_select
@deco.def_from_dict
@deco.def_shape(field='valid')
@deco.def_device(field='valid')
@dataclasses.dataclass
class Traffic:

    id: torch.Tensor = torch.empty((0,), dtype=torch.long)
    type: torch.Tensor = torch.empty((0,), dtype=torch.long)
    target_link: torch.Tensor = torch.empty((0,), dtype=torch.long)


    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)


    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)

    state: torch.Tensor = torch.empty((0,), dtype=torch.long)
    time_s: torch.Tensor = torch.empty((0,), dtype=torch.float)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)


@deco.def_to
@deco.def_replace
@deco.def_index_select
@deco.def_from_dict
@deco.def_shape(field='valid')
@deco.def_device(field='valid')
@dataclasses.dataclass
class Path:
    near_link: torch.Tensor = torch.empty((0,), dtype=torch.long)
    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    arc_length: torch.Tensor = torch.empty((0,), dtype=torch.float)
    is_route: torch.Tensor = torch.empty((0,), dtype=torch.bool)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)


@deco.def_to
@deco.def_replace
@deco.def_from_dict
@deco.def_device(field='valid')
@dataclasses.dataclass
class Goal:
    """정차 목적지 — 여기 도달하면 서고, 도달한 뒤에도 서 있는다.

    루트와 갈린다: 루트는 갈 길이고 goal 은 설 자리다. 없는 장면이 흔하므로
    (목적지 없이 도로를 따라가는 주행) ``valid`` 가 있느냐 없느냐를 말한다 —
    거리 조건이 아니다. 시간축이 없어 ``index_select`` 는 붙지 않는다.
    """
    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)


@deco.def_to
@deco.def_replace
@deco.def_index_select
@deco.def_from_dict
@deco.def_shape(field='valid')
@deco.def_device(field='valid')
@dataclasses.dataclass
class Link:
    id: torch.Tensor = torch.empty((0,), dtype=torch.long)
    type: torch.Tensor = torch.empty((0,), dtype=torch.long)
    section: torch.Tensor = torch.empty((0,), dtype=torch.long)
    speed_limit: torch.Tensor = torch.empty((0,), dtype=torch.float)
    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)
    entry_links: torch.Tensor = torch.empty((0,), dtype=torch.long)
    exit_links: torch.Tensor = torch.empty((0,), dtype=torch.long)
    left_links: torch.Tensor = torch.empty((0,), dtype=torch.long)
    right_links: torch.Tensor = torch.empty((0,), dtype=torch.long)
    left_lanes: torch.Tensor = torch.empty((0,), dtype=torch.long)
    right_lanes: torch.Tensor = torch.empty((0,), dtype=torch.long)


@deco.def_to
@deco.def_replace
@deco.def_index_select
@deco.def_from_dict
@deco.def_shape(field='valid')
@deco.def_device(field='valid')
@dataclasses.dataclass
class Lane:
    id: torch.Tensor = torch.empty((0,), dtype=torch.long)
    type: torch.Tensor = torch.empty((0,), dtype=torch.long)
    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)


@deco.def_to
@deco.def_replace
@deco.def_index_select
@deco.def_from_dict
@deco.def_shape(field='valid')
@deco.def_device(field='valid')
@dataclasses.dataclass
class Marker:
    id: torch.Tensor = torch.empty((0,), dtype=torch.long)
    type: torch.Tensor = torch.empty((0,), dtype=torch.long)
    pos_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    dir_xy: torch.Tensor = torch.empty((0,), dtype=torch.float)
    valid: torch.Tensor = torch.empty((0,), dtype=torch.bool)


@dataclasses.dataclass
class Meta:
    """Per-scene metadata attached to a Scene.

    All fields are ``[B]``-indexed to mirror the batch dimension of Scene's
    tensor fields. Stored on Scene as a non-tensor sidecar: pure Python lists,
    not moved to device by ``.to()`` (tree.map descends dataclasses but leaves
    non-tensor leaves untouched).
    """
    scene_id:   list[str]       | None = None
    scene_tags: list[list[str]] | None = None
    file_path:  list[str]       | None = None
    tags:       list[dict]      | None = None


@deco.def_to
@deco.def_replace
@deco.def_from_dict
@deco.def_device(field='agents')
@dataclasses.dataclass
class Scene:
    agents: Agent = dataclasses.field(default_factory=Agent)
    links: Link = dataclasses.field(default_factory=Link)
    lanes: Lane = dataclasses.field(default_factory=Lane)
    markers: Marker = dataclasses.field(default_factory=Marker)
    traffics: Traffic = dataclasses.field(default_factory=Traffic)
    paths: Path = dataclasses.field(default_factory=Path)
    goal: Goal = dataclasses.field(default_factory=Goal)
    meta: Meta = dataclasses.field(default_factory=Meta)

    def index_select(self, dim: int, index: torch.Tensor|list[int]) -> Scene:
        if isinstance(index, torch.Tensor):
            index = index.to(dtype=torch.long)
        else:
            index = torch.as_tensor(index, dtype=torch.long)
        return Scene(
            agents = self.agents.index_select(dim=dim, index=index),
            links = self.links.index_select(dim=dim, index=index),
            lanes = self.lanes.index_select(dim=dim, index=index),
            markers = self.markers.index_select(dim=dim, index=index),
            traffics = self.traffics.index_select(dim=dim, index=index),
            paths = self.paths.index_select(dim=dim, index=index),
            goal = self.goal,
        )


@deco.def_to
@deco.def_replace
@dataclasses.dataclass
class Output:
    """Unified model output (inference predictions + model-specific extra data).

    Common trajectory fields (filled by to_output / rollout):
        pos_xy:   [B, A, S, 2]
        dir_xy:   [B, A, S, 2]
        size_xyz: [B, A, S, 3]
        valid:    [B, A, S]
        is_ego:   [B, A, S]
        type:     [B, A, S]

    Model-specific extra data in ``extra`` dict:
        token-based: a2a_tokens_logits, a2a_tokens, r2r_tokens_logits, etc.
        regression: trajectory, probability, prediction, etc.
    """
    pos_xy: torch.Tensor
    dir_xy: torch.Tensor
    size_xyz: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    is_ego: Optional[torch.Tensor] = None
    type: Optional[torch.Tensor] = None
    batch_valid: Optional[torch.Tensor] = None
    extra: Optional[dict] = None

    def extract_ego_trajectory(self, num_past_frames: int = 0) -> dict[str, torch.Tensor]:
        """Ego 에이전트의 미래 궤적 추출. B=0, A=0 고정.

        Args:
            num_past_frames: 과거 프레임 수. 0이면 전체, >0이면 미래만 반환.

        Returns:
            dict with pos_xy [num_future_frames, 2], dir_xy [num_future_frames, 2]
        """
        return {
            'pos_xy': self.pos_xy[0, 0, num_past_frames:],
            'dir_xy': self.dir_xy[0, 0, num_past_frames:],
        }

    def __post_init__(self):
        if self.batch_valid is not None:
            return
        for field in dataclasses.fields(self):
            if field.name == 'extra':
                continue
            val = getattr(self, field.name)
            if val is not None and isinstance(val, torch.Tensor) and val.ndim >= 1:
                B = val.shape[0]
                self.batch_valid = torch.ones(B, dtype=torch.bool, device=val.device)
                return

    @classmethod
    def from_graph(cls, graph, configs) -> Output:
        """Convert rollout Graph → Output with common trajectory fields."""
        agent = graph["agent_tokens"]
        B, A, S = agent.pos_xy.shape[:3]
        device = agent.pos_xy.device

        agent_type = (
            agent.type
            if hasattr(agent, "type")
            else torch.ones(B, A, S, dtype=torch.long, device=device)
        )
        return cls(
            pos_xy=agent.pos_xy,
            dir_xy=agent.dir_xy,
            size_xyz=agent.size_xyz,
            valid=agent.valid,
            is_ego=agent.is_ego,
            type=agent_type,
        )
