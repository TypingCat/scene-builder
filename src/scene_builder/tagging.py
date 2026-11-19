# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from tqdm import tqdm

from .dataspec.shard import INDEX_NAME, SceneDataset
from .scenario_tags import classify


def _dataset_state(dataset: SceneDataset) -> tuple:
    paths = (Path(dataset.path) / INDEX_NAME, *(Path(p) for p in dataset.shard_paths))
    return tuple((s.st_ino, s.st_size, s.st_mtime_ns) for s in (p.stat() for p in paths))


def write_tags(dataset: SceneDataset, output: Path, scorer=None) -> None:
    """Stream one tag row per scene and publish the sidecar only after completion."""
    initial_state = _dataset_state(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                prefix=f'.{output.name}.', suffix='.part',
                                dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for index in tqdm(range(len(dataset)), desc='장면 태깅', unit='scene'):
                scene = dataset[index]
                tags = classify(scene)
                if scorer is not None:
                    tags['Score'] = scorer.score(scene)
                row = {'index': index, 'scene_id': scene['meta']['scene_id'], **tags}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if _dataset_state(dataset) != initial_state:
            raise ValueError('태깅 중 입력 데이터셋이 바뀌었다. 장면 구축을 끝낸 뒤 다시 실행할 것.')
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
