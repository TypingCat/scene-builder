from __future__ import annotations
import dataclasses
from scene_builder._pdm.dtypes import deco


@deco.def_from_dict
@dataclasses.dataclass
class TimeConfig:
    """모델 시간 설정.

    num_past_frames/num_future_frames는 모델이 사용하는 프레임 수.
    num_frames = past + future (= 심볼 F).
    데이터 해상도(frame_unit)는 DataloaderConfig 소유.
    """
    num_past_frames: int | None = None
    num_future_frames: int | None = None

    @property
    def now(self):
        return self.num_past_frames - 1

    @property
    def num_frames(self):
        return self.num_past_frames + self.num_future_frames

    def past_idx(self) -> list[int]:
        """과거 프레임 인덱스. [0, 1, ..., now]."""
        return list(range(self.num_past_frames))

    def future_idx(self) -> list[int]:
        """미래 프레임 인덱스. [num_past_frames, ..., num_frames-1]."""
        return list(range(self.num_past_frames, self.num_frames))

    def idx(self) -> list[int]:
        """전체 프레임 인덱스. [0, ..., num_frames-1]."""
        return list(range(self.num_frames))


@deco.def_from_dict
@dataclasses.dataclass
class DrivableAreaConfig:
    """Shared reconstruction settings for dataspec-missing road polygons.

    ``method`` remains only so older resolved run configs can be loaded.  DAC
    now always uses the shared polygon representation.
    """
    method: str = 'polygon'
    max_links: int = 64
    max_lanes: int = 64
    default_half_width_m: float = 1.5
    min_half_width_m: float = 0.5
    max_half_width_m: float = 8.0
    link_chunk: int = 4
    polygon_chunk: int = 8

    def __post_init__(self):
        if self.method not in ('polygon', 'edge_knn', 'corridor'):
            raise ValueError(
                "drivable_area.method must be polygon (legacy edge_knn/corridor "
                f"values are accepted for config migration), got {self.method!r}"
            )


@deco.def_from_dict
@dataclasses.dataclass(frozen=True)
class ControllerConfig:
    """Steering and speed controller settings for closed-loop simulation."""
    look_ahead_min_m: float = 2.0
    look_ahead_max_m: float = 25.0
    look_ahead_k_s: float = 1.5
    kp_speed: float = 1.5
    ki_speed: float = 3.0
    end_brake_frac: float = 0.3


@deco.def_from_dict
@dataclasses.dataclass
class DynamicsConfig:
    """Vehicle, cyclist and point-mass dynamics settings by agent type."""
    vehicle: dict = dataclasses.field(default_factory=lambda: {
        'max_vel': 30.0, 'max_acc': 9.0, 'anchor_length_m': (4.635, 11.0),
        'wheelbase_m': (3.0, 4.4), 'delay_s': (0.1, 0.6),
        'accel': (5.0, 3.0), 'brake': (8.0, 6.0),
        'max_steer': 0.52, 'drag_coef': 0.05,
        'wheelbase_l_ratio': 0.8, 'wheelbase_min_m': 0.5, 'steer_delay_max_s': 0.6})
    cyclist: dict = dataclasses.field(default_factory=lambda: {
        'max_vel': 12.0, 'max_acc': 6.0, 'wheelbase_m': 1.05, 'delay_s': 0.1,
        'accel': 3.5, 'brake': 5.0, 'max_steer': 0.52, 'drag_coef': 0.05})
    pedestrian: dict = dataclasses.field(default_factory=lambda: {'max_vel': 6.0, 'max_acc': 5.0})
    unknown: dict = dataclasses.field(default_factory=lambda: {'max_vel': 30.0, 'max_acc': 9.0})

    def max_vel_by_type(self) -> tuple:
        """feasibility 속도 천장을 AgentType 순(unknown, vehicle, pedestrian, cyclist)으로."""
        return (self.unknown['max_vel'], self.vehicle['max_vel'],
                self.pedestrian['max_vel'], self.cyclist['max_vel'])

    def max_acc_by_type(self) -> tuple:
        """feasibility 가속 천장을 AgentType 순으로."""
        return (self.unknown['max_acc'], self.vehicle['max_acc'],
                self.pedestrian['max_acc'], self.cyclist['max_acc'])
