# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

"""Load an external PDM deployment."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

import torch


def _require_file(folder: Path, name: str) -> Path:
    path = folder / name
    if not path.is_file():
        raise ValueError(f'PDM 배포 파일이 없다: {path}')
    return path


def _compile_bridge(source: Path, output: Path) -> None:
    from torch.utils.cpp_extension import include_paths, library_paths
    import pybind11

    compiler = shutil.which('c++')
    if compiler is None:
        raise ValueError('PDM 브리지 빌드에 C++17 컴파일러(c++)가 필요하다.')
    command = [
        compiler, '-O3', '-shared', '-std=c++17', '-fPIC',
        f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
        str(source), '-o', str(output),
        f'-I{source.parent}', f'-I{pybind11.get_include()}', f'-I{sysconfig.get_path("include")}',
        *(f'-I{path}' for path in include_paths()),
        *(f'-L{path}' for path in library_paths()),
        '-ltorch_python', '-ltorch_cpu', '-ltorch', '-lc10', '-ldl',
    ]
    subprocess.run(command, check=True)


class Deployment:
    def __init__(self, folder: str, *, dataspec: dict, device: torch.device):
        if sys.platform != 'linux':
            raise ValueError('PDM 배포물(run.so)을 실행하려면 Linux 환경이 필요하다.')
        try:
            import pybind11  # noqa: F401
            import yaml
        except ImportError as exc:
            raise ValueError('PDM 채점에 score 추가 의존성(PyYAML, pybind11)이 필요하다.') from exc
        root = Path(folder).expanduser().resolve()
        manifest_path = _require_file(root, 'manifest.yaml')
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
        except yaml.YAMLError as exc:
            raise ValueError(f'PDM manifest를 해석할 수 없다: {manifest_path}') from exc
        if not isinstance(manifest, dict):
            raise ValueError(f'PDM manifest 형식이 잘못됐다: {manifest_path}')
        self.manifest = manifest
        if manifest.get('model_name') != 'pdm_planner' or manifest.get('backend') != 'none':
            raise ValueError('model_name=pdm_planner, backend=none인 배포물이 필요하다.')
        if manifest.get('torch_version') != str(torch.__version__):
            raise ValueError(f'PDM 배포 Torch {manifest.get("torch_version")} / 실행 Torch {torch.__version__}')
        deployed_spec = manifest.get('dataspec')
        if not isinstance(deployed_spec, dict):
            raise ValueError('PDM manifest에 dataspec이 없다.')
        mismatch = {k: (v, deployed_spec.get(k)) for k, v in dataspec.items()
                    if k not in ('frame_unit', 'num_frames') and deployed_spec.get(k) != v}
        if mismatch:
            raise ValueError(f'PDM 배포 형상 불일치 (장면, 배포): {mismatch}')

        runtime = _require_file(root, 'run.so')
        source = _require_file(root, 'bridge.cpp')
        types = _require_file(root, 'types.h')
        if manifest.get('types_sha256') != hashlib.sha256(types.read_bytes()).hexdigest():
            raise ValueError('PDM manifest와 types.h 해시가 다르다.')

        ort_version = manifest.get('onnxruntime_version')
        if ort_version:
            sdk = os.environ.get('ONNXRUNTIME_SDK_ROOT')
            if not sdk:
                raise ValueError('배포물의 ONNX Runtime SDK 경로를 ONNXRUNTIME_SDK_ROOT로 지정해야 한다.')
            lib = Path(sdk).expanduser() / ort_version / 'lib' / 'libonnxruntime.so'
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


        with tempfile.TemporaryDirectory(prefix='scene-builder-pdm-') as cache:
            bridge = Path(cache) / '_runtime_bridge.so'
            _compile_bridge(source, bridge)
            spec = importlib.util.spec_from_file_location('_runtime_bridge', bridge)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        device_index = device.index if device.type == 'cuda' else -1
        self._backend = module.RuntimeBridge(str(runtime), device_index)
        if (self._backend.num_input_frames != int(manifest['num_input_frames'])
                or self._backend.num_output_frames != int(manifest['num_output_frames'])
                or abs(self._backend.frame_unit - float(manifest['frame_unit'])) > 1e-6):
            raise ValueError('PDM 배포 manifest와 실행부의 시간창이 다르다.')

    def plan(self, scene):
        return self._backend.plan(scene)
