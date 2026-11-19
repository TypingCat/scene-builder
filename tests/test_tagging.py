import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch

from scene_builder._pdm.scoring import PDMScorer
from scene_builder.cli import main
from scene_builder.dataspec import decode
from scene_builder.dataspec.schema import empty_scene
from scene_builder.dataspec.shard import SceneDataset, make_stamp, write_scene_index, write_scene_shard
from scene_builder.dataspec.spec import DATASPEC, AgentType, LinkSection, LinkType
from scene_builder.tagging import write_tags


@pytest.fixture
def dataset(tmp_path, request):
    ego_frames = getattr(request, 'param', 32)
    scene = empty_scene()
    agents = scene['agents']
    agents['id'][:3] = torch.tensor([1, 2, 3])
    agents['type'][:3] = torch.tensor([AgentType.vehicle, AgentType.vehicle, AgentType.pedestrian])
    agents['is_ego'][0] = True
    agents['valid'][0, :ego_frames] = True
    agents['valid'][1, 20:25] = True
    agents['valid'][2, :25] = True
    agents['size_xyz'][:3] = torch.tensor([4.8, 1.8, 1.5])
    agents['time_s'][:, :32, 0] = torch.arange(32) * 0.1
    agents['pos_xy'][0, :32, 0] = torch.arange(32) * 0.5
    agents['pos_xy'][1:3, :, 1] = 50
    agents['dir_xy'][:, :, 0] = 1
    agents['vel_xy'][0, :32, 0] = 5
    links, paths = scene['links'], scene['paths']
    links['id'][0] = 1
    links['type'][0] = LinkType.vehicle
    links['section'][0] = LinkSection.freeway
    for geometry in (links, paths):
        geometry['valid'][0, :101] = True
        geometry['pos_xy'][0, :101, 0] = torch.arange(101) * 0.5
        geometry['dir_xy'][0, :101, 0] = 1
    paths['is_route'][0] = True
    paths['near_link'][0, :101] = 1
    paths['arc_length'][0, :101, 0] = links['pos_xy'][0, :101, 0]
    scene['meta']['scene_id'] = 'straight-road'
    write_scene_shard([scene], str(tmp_path), stamp=make_stamp(dict(DATASPEC)), filename='mini.scenes')
    write_scene_index(str(tmp_path))
    return SceneDataset(str(tmp_path), loads=decode.loads)


def test_tag_command_classifies_existing_shards(dataset, tmp_path):
    assert main(['tag', '-i', dataset.path]) == 0
    row = json.loads((tmp_path / 'tags.jsonl').read_text(encoding='utf-8'))
    assert row['scene_id'] == 'straight-road'
    assert row['scene'] == {
        'Vehicles': 1, 'Pedestrians': 1, 'Cyclists': 0, 'Traffics': 0, 'Intersections': 0,
        'Freeways': 1, 'Links': 1.0, 'Lanes': 0.0,
    }
    assert row['route']['Mapless'] == row['route']['NoRoute'] == 0
    assert row['ego']['vel'] == pytest.approx(5.0, abs=1e-5)
    assert row['ego']['Δpos'] == pytest.approx(5.5, abs=1e-5)
    assert 'Score' not in row


class StraightPlanner:
    manifest = {'frame_unit': 0.1, 'num_input_frames': 20, 'num_output_frames': 40,
                'controller': {}, 'dynamics': {}}

    def __init__(self):
        self.calls = 0

    def plan(self, scene):
        self.calls += 1
        past, future = self.manifest['num_input_frames'], self.manifest['num_output_frames']
        frames = past + future
        agents = scene.agents
        pos = agents.pos_xy[:, :1, :frames].clone()
        dirs = agents.dir_xy[:, :1, :frames].clone()
        direction = dirs[:, :, past - 1:past]
        distances = torch.arange(1, future + 1)[None, None, :, None] * 0.5
        pos[:, :, past:] = pos[:, :, past - 1:past] + direction * distances
        dirs[:, :, past:] = direction
        valid = torch.ones_like(agents.valid[:, :1, :frames])
        return {'pos_xy': pos, 'dir_xy': dirs, 'valid': valid, 'is_ego': valid,
                'size_xyz': agents.size_xyz[:, :1, None].expand(-1, -1, frames, -1),
                'type': agents.type[:, :1, None].expand(-1, -1, frames),
                'batch_valid': valid[:, 0, 0]}


def test_score_tags_use_valid_rollout_window(dataset, tmp_path):
    planner = StraightPlanner()
    scorer = PDMScorer(planner, frame_unit=0.1, device=torch.device('cpu'))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    output = tmp_path / 'score.jsonl'
    try:
        write_tags(dataset, output, scorer)
    finally:
        torch.set_num_threads(threads)

    assert planner.calls == 12
    score = json.loads(output.read_text(encoding='utf-8'))['Score']
    assert 0 <= score['EPDMS'] <= 1
    assert score['EC'] == 1.0
    assert score['SLC'] is None

    scene = dataset[0]
    scene['paths']['is_route'] = np.zeros_like(scene['paths']['is_route'])
    assert all(value is None for value in scorer.score(scene).values())
    assert planner.calls == 12


@pytest.mark.skipif(
    not shutil.which('ninja')
    or not shutil.which('cl' if sys.platform == 'win32' else os.environ.get('CXX', 'c++')),
    reason='embedded PDM requires Ninja and a configured C++ compiler',
)
@pytest.mark.parametrize('dataset', [21], indirect=True)
def test_score_command_runs_embedded_pdm(dataset, tmp_path):
    output = tmp_path / 'embedded.jsonl'
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert main(['tag', '-i', dataset.path, '--score', '-o', str(output)]) == 0
    finally:
        torch.set_num_threads(threads)
    score = json.loads(output.read_text(encoding='utf-8'))['Score']
    assert 0 <= score['EPDMS'] <= 1
    assert score['SLC'] is None
