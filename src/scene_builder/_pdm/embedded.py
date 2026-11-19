# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

"""Build and run the bundled rule-based PDM planner using the installed PyTorch."""

from __future__ import annotations

from functools import cached_property
import os
from pathlib import Path
import shutil
import sys


class EmbeddedPlanner:

    manifest = {
        'frame_unit': 0.1,
        'num_input_frames': 20,
        'num_output_frames': 40,
        'controller': {'kp_speed': 1.0, 'ki_speed': 0.5},
        'dynamics': {},
    }

    @cached_property
    def _native(self):
        from torch.utils.cpp_extension import load

        if shutil.which('ninja') is None:
            raise ValueError('내장 PDM 빌드에 Ninja가 필요하다. uv sync --extra score를 실행할 것.')
        if sys.platform == 'win32' and shutil.which('cl') is None:
            raise ValueError('내장 PDM은 Visual Studio 2022의 x64 Native Tools 명령 프롬프트에서 실행할 것.')
        files = Path(__file__).with_name('native')
        windows = sys.platform == 'win32'
        abi_check = 'TORCH_DONT_CHECK_COMPILER_ABI'


        if windows:
            previous = os.environ.get(abi_check)
            os.environ[abi_check] = '1'
        try:
            return load(
                name='scene_builder_pdm',
                sources=[str(path) for path in sorted(files.glob('*.cpp'))],
                extra_cflags=['/O2', '/bigobj', '/utf-8'] if windows else ['-O2'],
                with_cuda=False,
            )
        finally:
            if windows:
                if previous is None:
                    os.environ.pop(abi_check, None)
                else:
                    os.environ[abi_check] = previous

    def plan(self, scene):
        manifest = self.manifest
        return self._native.plan(scene, manifest['num_input_frames'],
                                 manifest['num_output_frames'], manifest['frame_unit'])
