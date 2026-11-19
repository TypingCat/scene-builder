from __future__ import annotations
import dataclasses
import torch
from scene_builder._pdm.dtypes import feature


@dataclasses.dataclass(frozen=True)
class DrivableAreaPolygons:
    """Batched padded polygon rings reconstructed from vehicle links.

    Attributes:
        vertices: ``[B, K, V, 2]`` polygon vertices. Rings are implicit; the
            last valid vertex connects back to the first.
        valid: ``[B, K, V]`` valid vertex mask, packed as a prefix.
        link_id: ``[B, K]`` source link id (zero for invalid polygons).
        link_section: ``[B, K]`` source link section.
    """

    vertices: torch.Tensor
    valid: torch.Tensor
    link_id: torch.Tensor
    link_section: torch.Tensor

    @property
    def B(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_polygons(self) -> int:
        return self.vertices.shape[1]

    def contains(self, points: torch.Tensor, *, polygon_chunk: int = 8) -> torch.Tensor:
        """Return whether each point lies in the polygon union.

        ``points`` has shape ``[B, ..., 2]`` and the result ``[B, ...]``.
        As in NAVSIM's Shapely ``contains`` query, points exactly on a polygon
        boundary are not considered contained.
        """
        return self.contains_each(points, polygon_chunk=polygon_chunk).any(dim=-1)

    def contains_each(
        self,
        points: torch.Tensor,
        *,
        polygon_chunk: int = 8,
    ) -> torch.Tensor:
        """Return point membership for every polygon as ``[B, ..., K]``."""
        return points_in_polygons(
            points, self.vertices, self.valid, polygon_chunk=polygon_chunk,
        )

    def matches_link_ids(
        self,
        link_ids: torch.Tensor,
        link_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return polygons whose source link occurs in a valid id sequence.

        ``link_ids`` and ``link_valid`` have shape ``[B, ...]``.  This lets
        route-aware metrics select on-route polygons while retaining the same
        geometry used by map-wide metrics such as DAC.
        """
        if link_ids.shape != link_valid.shape or link_ids.shape[0] != self.B:
            raise ValueError("link_ids/link_valid shape or polygon batch mismatch")
        flat_ids = link_ids.reshape(self.B, -1)
        flat_valid = link_valid.reshape(self.B, -1)
        return (
            (self.link_id[..., None] == flat_ids[:, None])
            & flat_valid[:, None]
            & (flat_ids[:, None] > 0)
            & self.valid.any(dim=-1)[..., None]
        ).any(dim=-1)

    def intersects_boxes(
        self,
        box_corners: torch.Tensor,
        *,
        polygon_chunk: int = 8,
    ) -> torch.Tensor:
        """Return box/polygon intersections as ``[B, ..., K]``.

        ``box_corners`` has shape ``[B, ..., 4, 2]`` with vertices ordered
        around each oriented box.  Intersection is closed: boundary contact
        counts, matching Shapely's ``intersects`` predicate used by NAVSIM.
        """
        return polygons_intersect_boxes(
            box_corners,
            self.vertices,
            self.valid,
            polygon_chunk=polygon_chunk,
        )


def _gather_entities(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather an entity axis (dim=1), preserving all trailing dimensions."""
    shape = [*index.shape, *value.shape[2:]]
    gather_index = index.reshape(*index.shape, *([1] * (value.ndim - 2)))
    gather_index = gather_index.expand(shape)
    return value.gather(1, gather_index)


def _nearest_boundary_half_widths(
    link_pos: torch.Tensor,
    link_valid: torch.Tensor,
    left_ids: torch.Tensor,
    right_ids: torch.Tensor,
    lanes: feature.Lane,
    center_xy: torch.Tensor,
    *,
    default_half_width_m: float,
    min_half_width_m: float,
    max_half_width_m: float,
    max_lanes: int,
    link_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure station-wise link-to-referenced-boundary distances."""
    B, K, L = link_valid.shape
    default = torch.full_like(link_pos[..., 0], default_half_width_m)
    if lanes is None or lanes.valid.numel() == 0 or not lanes.valid.any():
        return default, default.clone()

    ref_ids = torch.cat([left_ids, right_ids], dim=-1).reshape(B, -1)
    referenced = (
        (lanes.id[:, :, None] == ref_ids[:, None, :])
        & (ref_ids[:, None, :] > 0)
    ).any(dim=-1)
    boundary_valid = lanes.valid & referenced[..., None]
    if not boundary_valid.any():
        return default, default.clone()

    num_lanes = min(max_lanes, lanes.valid.shape[1])
    lane_dist = (lanes.pos_xy - center_xy[:, None, None]).norm(dim=-1)
    lane_dist = lane_dist.masked_fill(~boundary_valid, float("inf")).amin(dim=-1)
    _, lane_index = lane_dist.topk(num_lanes, largest=False)
    lane_pos = _gather_entities(lanes.pos_xy, lane_index)
    lane_valid = _gather_entities(boundary_valid, lane_index)
    lane_id = lanes.id.gather(1, lane_index)

    left_of = (
        (lane_id[:, None, :, None] == left_ids[:, :, None, :])
        & (left_ids[:, :, None, :] > 0)
    ).any(dim=-1)
    right_of = (
        (lane_id[:, None, :, None] == right_ids[:, :, None, :])
        & (right_ids[:, :, None, :] > 0)
    ).any(dim=-1)

    seg_start = lane_pos[:, :, :-1]
    seg_vec = lane_pos[:, :, 1:] - seg_start
    seg_sq = (seg_vec * seg_vec).sum(dim=-1).clamp(min=1e-8)
    seg_valid = lane_valid[:, :, :-1] & lane_valid[:, :, 1:]

    left_width = torch.full_like(default, float("inf"))
    right_width = torch.full_like(default, float("inf"))
    for start in range(0, K, link_chunk):
        end = min(start + link_chunk, K)
        station = link_pos[:, start:end, :, None, None, :]
        relative = station - seg_start[:, None, None]
        fraction = (
            (relative * seg_vec[:, None, None]).sum(dim=-1)
            / seg_sq[:, None, None]
        ).clamp(0.0, 1.0)
        offset = relative - fraction.unsqueeze(-1) * seg_vec[:, None, None]
        distance = offset.norm(dim=-1)
        distance = distance.masked_fill(
            ~seg_valid[:, None, None], float("inf"),
        )

        for belongs_to_side, destination in (
            (left_of, left_width),
            (right_of, right_width),
        ):
            side_mask = belongs_to_side[:, start:end, None, :, None]
            measured = distance.masked_fill(~side_mask, float("inf"))
            destination[:, start:end] = measured.flatten(start_dim=3).amin(dim=-1)

    def _usable(width: torch.Tensor) -> torch.Tensor:
        return (
            width.isfinite()
            & (width >= min_half_width_m)
            & (width <= max_half_width_m)
            & link_valid
        )

    def _link_median(width: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
        unmeasured = torch.full_like(width, float("nan"))
        measured = torch.where(usable, width, unmeasured)
        return measured.nanmedian(dim=-1, keepdim=True).values


    usable_left, usable_right = _usable(left_width), _usable(right_width)
    median_left = _link_median(left_width, usable_left)
    median_right = _link_median(right_width, usable_right)
    fill_left = torch.where(median_left.isnan(), median_right, median_left)
    fill_right = torch.where(median_right.isnan(), median_left, median_right)
    fill_left = torch.where(fill_left.isnan(), default[..., :1], fill_left)
    fill_right = torch.where(fill_right.isnan(), default[..., :1], fill_right)
    return (
        torch.where(usable_left, left_width, fill_left.expand_as(left_width)),
        torch.where(usable_right, right_width, fill_right.expand_as(right_width)),
    )


def _pack_ribbon(
    left: torch.Tensor,
    right: torch.Tensor,
    station_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack left-forward/right-reverse vertices into prefix-valid rings."""
    B, K, L, _ = left.shape
    V = 2 * L
    vertices = torch.zeros(B, K, V, 2, dtype=left.dtype, device=left.device)

    rank = station_valid.long().cumsum(dim=-1) - 1
    count = station_valid.sum(dim=-1)
    unused_slot = torch.full_like(rank, V - 1)

    left_index = torch.where(station_valid, rank, unused_slot)
    vertices.scatter_(
        2,
        left_index[..., None].expand(B, K, L, 2),
        torch.where(station_valid[..., None], left, torch.zeros_like(left)),
    )

    right_index = 2 * count[..., None] - 1 - rank
    right_index = torch.where(station_valid, right_index, unused_slot)
    vertices.scatter_(
        2,
        right_index[..., None].expand(B, K, L, 2),
        torch.where(station_valid[..., None], right, torch.zeros_like(right)),
    )

    valid = torch.arange(V, device=left.device)[None, None] < (2 * count)[..., None]
    valid &= count[..., None] >= 2
    return vertices, valid


def build_drivable_area_polygons(
    links: feature.Link,
    lanes: feature.Lane,
    center_xy: torch.Tensor,
    *,
    max_links: int = 64,
    max_lanes: int = 64,
    default_half_width_m: float = 1.5,
    min_half_width_m: float = 0.5,
    max_half_width_m: float = 8.0,
    link_chunk: int = 4,
) -> DrivableAreaPolygons | None:
    """Reconstruct local drivable polygons from link and boundary polylines.

    Vehicle links nearest ``center_xy`` are converted to ribbon polygons.  When
    referenced boundary geometry is available, its distance supplies the left
    and right width at each station; missing or implausible measurements use
    the lane-width fallback.
    """
    if links is None or links.valid.numel() == 0 or links.valid.shape[-1] < 2:
        return None
    if center_xy.ndim != 2 or center_xy.shape != (links.valid.shape[0], 2):
        raise ValueError(
            f"center_xy must be [B, 2], got {tuple(center_xy.shape)}",
        )

    drivable = (
        (links.type == feature.LinkType.vehicle)
        & (links.valid.sum(dim=-1) >= 2)
    )
    if not drivable.any():
        return None

    B, num_links, _ = links.valid.shape
    K = min(max_links, num_links)
    distance = (links.pos_xy - center_xy[:, None, None]).norm(dim=-1)
    distance = distance.masked_fill(~links.valid, float("inf")).amin(dim=-1)
    distance = distance.masked_fill(~drivable, float("inf"))
    _, selected = distance.topk(K, largest=False)

    link_pos = _gather_entities(links.pos_xy, selected)
    link_dir = _gather_entities(links.dir_xy, selected)
    link_valid = _gather_entities(links.valid, selected)
    selected_drivable = drivable.gather(1, selected)
    link_valid &= selected_drivable[..., None]
    link_id = links.id.gather(1, selected)
    link_id = torch.where(selected_drivable, link_id, torch.zeros_like(link_id))
    link_section = (
        links.section.gather(1, selected)
        if links.section.shape == links.id.shape
        else torch.zeros_like(link_id)
    )
    link_section = torch.where(
        selected_drivable, link_section, torch.zeros_like(link_section),
    )
    left_ids = _gather_entities(links.left_lanes, selected)
    right_ids = _gather_entities(links.right_lanes, selected)

    left_width, right_width = _nearest_boundary_half_widths(
        link_pos,
        link_valid,
        left_ids,
        right_ids,
        lanes,
        center_xy,
        default_half_width_m=default_half_width_m,
        min_half_width_m=min_half_width_m,
        max_half_width_m=max_half_width_m,
        max_lanes=max_lanes,
        link_chunk=link_chunk,
    )

    tangent = link_dir / link_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    geometric = torch.zeros_like(link_pos)
    geometric[:, :, :-1] = link_pos[:, :, 1:] - link_pos[:, :, :-1]
    geometric[:, :, -1] = link_pos[:, :, -1] - link_pos[:, :, -2]
    geometric = geometric / geometric.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tangent = torch.where(
        (link_dir.norm(dim=-1, keepdim=True) > 1e-8),
        tangent,
        geometric,
    )


    step = (link_pos[:, :, 1:] - link_pos[:, :, :-1]).norm(dim=-1)
    step_valid = link_valid[:, :, 1:] & link_valid[:, :, :-1]
    spacing = (step * step_valid).sum(dim=-1) / step_valid.sum(dim=-1).clamp(min=1)
    rank = link_valid.long().cumsum(dim=-1) - 1
    count = link_valid.sum(dim=-1, keepdim=True)
    is_first = link_valid & (rank == 0)
    is_last = link_valid & (rank == count - 1)
    overhang = 0.5 * spacing[..., None, None] * tangent
    link_pos = link_pos + overhang * (is_last[..., None].to(overhang.dtype)
                                      - is_first[..., None].to(overhang.dtype))

    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    left = link_pos + normal * left_width[..., None]
    right = link_pos - normal * right_width[..., None]
    vertices, polygon_valid = _pack_ribbon(left, right, link_valid)
    return DrivableAreaPolygons(
        vertices=vertices,
        valid=polygon_valid,
        link_id=link_id,
        link_section=link_section,
    )


def points_in_polygons(
    points: torch.Tensor,
    vertices: torch.Tensor,
    valid: torch.Tensor,
    *,
    polygon_chunk: int = 8,
    boundary_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Strict point-in-polygon query using the even-odd rule.

    Args:
        points: ``[B, ..., 2]``.
        vertices: ``[B, K, V, 2]``.
        valid: prefix-valid mask ``[B, K, V]``.

    Returns:
        ``[B, ..., K]``; boundary points are False.
    """
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError(f"points must be [B, ..., 2], got {tuple(points.shape)}")
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError(
            f"vertices must be [B, K, V, 2], got {tuple(vertices.shape)}",
        )
    if valid.shape != vertices.shape[:-1] or points.shape[0] != vertices.shape[0]:
        raise ValueError("points/polygon batch or valid shape mismatch")
    if polygon_chunk <= 0:
        raise ValueError("polygon_chunk must be positive")

    B, K, V, _ = vertices.shape
    point_shape = points.shape[1:-1]
    flat_points = points.reshape(B, -1, 2)
    Q = flat_points.shape[1]
    result = torch.zeros(B, Q, K, dtype=torch.bool, device=points.device)
    vertex_index = torch.arange(V, device=points.device)[None, None]

    for start in range(0, K, polygon_chunk):
        end = min(start + polygon_chunk, K)
        polygon = vertices[:, start:end]
        polygon_valid = valid[:, start:end]
        count = polygon_valid.sum(dim=-1)
        next_index = torch.where(
            vertex_index + 1 < count[..., None],
            vertex_index + 1,
            torch.zeros_like(vertex_index),
        )
        next_vertex = polygon.gather(
            2, next_index[..., None].expand(B, end - start, V, 2),
        )
        edge_valid = vertex_index < count[..., None]

        a = polygon[:, None]
        b = next_vertex[:, None]
        p = flat_points[:, :, None, None]
        px, py = p[..., 0], p[..., 1]
        ax, ay = a[..., 0], a[..., 1]
        bx, by = b[..., 0], b[..., 1]

        crosses_y = (ay > py) != (by > py)
        delta_y = by - ay
        safe_delta_y = torch.where(
            delta_y.abs() > 1e-12, delta_y, torch.ones_like(delta_y),
        )
        x_intersection = ax + (py - ay) * (bx - ax) / safe_delta_y
        crossings = crosses_y & (px < x_intersection) & edge_valid[:, None]
        inside = (crossings.sum(dim=-1) % 2) == 1

        edge = b - a
        relative = p - a
        cross = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
        dot = (relative * edge).sum(dim=-1)
        edge_sq = (edge * edge).sum(dim=-1)
        on_boundary = (
            (cross.abs() <= boundary_tolerance)
            & (dot >= -boundary_tolerance)
            & (dot <= edge_sq + boundary_tolerance)
            & edge_valid[:, None]
        ).any(dim=-1)
        result[:, :, start:end] = inside & ~on_boundary & (count[:, None] >= 3)

    return result.reshape(B, *point_shape, K)


def polygons_intersect_boxes(
    box_corners: torch.Tensor,
    vertices: torch.Tensor,
    valid: torch.Tensor,
    *,
    polygon_chunk: int = 8,
) -> torch.Tensor:
    """Closed intersection test between oriented boxes and simple polygons.

    A hit occurs when a box corner is inside a polygon or any polygon edge
    intersects the box.  The latter also covers polygons contained by a box
    and all boundary-touching cases.
    """
    if box_corners.ndim < 3 or box_corners.shape[-2:] != (4, 2):
        raise ValueError(
            "box_corners must be [B, ..., 4, 2], "
            f"got {tuple(box_corners.shape)}",
        )
    if vertices.ndim != 4 or vertices.shape[-1] != 2:
        raise ValueError(
            f"vertices must be [B, K, V, 2], got {tuple(vertices.shape)}",
        )
    if valid.shape != vertices.shape[:-1] or box_corners.shape[0] != vertices.shape[0]:
        raise ValueError("box/polygon batch or valid shape mismatch")
    if polygon_chunk <= 0:
        raise ValueError("polygon_chunk must be positive")

    B, K, V, _ = vertices.shape
    box_shape = box_corners.shape[1:-2]
    boxes = box_corners.reshape(B, -1, 4, 2)


    result = points_in_polygons(
        boxes,
        vertices,
        valid,
        polygon_chunk=polygon_chunk,
    ).any(dim=-2)

    center = boxes.mean(dim=-2)
    longitudinal = boxes[..., 0, :] - boxes[..., 3, :]
    lateral = boxes[..., 0, :] - boxes[..., 1, :]
    half_length = longitudinal.norm(dim=-1) * 0.5
    half_width = lateral.norm(dim=-1) * 0.5
    longitudinal = longitudinal / longitudinal.norm(
        dim=-1, keepdim=True,
    ).clamp(min=1e-8)
    lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    half = torch.stack([half_length, half_width], dim=-1)

    vertex_index = torch.arange(V, device=vertices.device)[None, None]
    for start in range(0, K, polygon_chunk):
        end = min(start + polygon_chunk, K)
        polygon = vertices[:, start:end]
        polygon_valid = valid[:, start:end]
        count = polygon_valid.sum(dim=-1)
        next_index = torch.where(
            vertex_index + 1 < count[..., None],
            vertex_index + 1,
            torch.zeros_like(vertex_index),
        )
        next_vertex = polygon.gather(
            2, next_index[..., None].expand(B, end - start, V, 2),
        )
        edge_valid = (vertex_index < count[..., None]) & (count[..., None] >= 3)

        relative_a = polygon[:, None] - center[:, :, None, None]
        relative_b = next_vertex[:, None] - center[:, :, None, None]
        axis_long = longitudinal[:, :, None, None]
        axis_lat = lateral[:, :, None, None]
        local_a = torch.stack(
            [
                (relative_a * axis_long).sum(dim=-1),
                (relative_a * axis_lat).sum(dim=-1),
            ],
            dim=-1,
        )
        local_b = torch.stack(
            [
                (relative_b * axis_long).sum(dim=-1),
                (relative_b * axis_lat).sum(dim=-1),
            ],
            dim=-1,
        )

        delta = local_b - local_a
        box_half = half[:, :, None, None]
        parallel = delta.abs() <= 1e-12
        safe_delta = torch.where(parallel, torch.ones_like(delta), delta)
        t1 = (-box_half - local_a) / safe_delta
        t2 = (box_half - local_a) / safe_delta
        near = torch.minimum(t1, t2)
        far = torch.maximum(t1, t2)
        outside_parallel = parallel & (local_a.abs() > box_half)
        near = torch.where(
            parallel,
            torch.full_like(near, float("-inf")),
            near,
        )
        far = torch.where(
            parallel,
            torch.where(
                outside_parallel,
                torch.full_like(far, float("-inf")),
                torch.full_like(far, float("inf")),
            ),
            far,
        )
        t_min = near.amax(dim=-1)
        t_max = far.amin(dim=-1)
        edge_hits = (
            (t_min <= t_max)
            & (t_max >= 0.0)
            & (t_min <= 1.0)
            & edge_valid[:, None]
        ).any(dim=-1)
        result[:, :, start:end] |= edge_hits

    return result.reshape(B, *box_shape, K)
