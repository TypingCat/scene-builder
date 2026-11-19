# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
from collections import Counter

import numpy as np

from . import pipeline
from .dataspec import decode
from .dataspec.check import ITEMS, check_scene, check_stamp_item
from .dataspec.shard import INDEX_NAME, SceneDataset, read_stamp
from .dataspec.spec import FIELDS


def _need_input(path: str, dataset: bool) -> str | None:
    if not os.path.exists(path):
        return f'입력 경로가 없다: {path}'
    if dataset and not os.path.isfile(os.path.join(path, INDEX_NAME)):
        return f'장면 데이터셋이 아니다 ({INDEX_NAME} 없음): {path}'
    return None


def cmd_build(args) -> int:
    return pipeline.build(args.input, args.output, workers=args.workers,
                          limit=args.limit, overwrite=args.overwrite)


def cmd_count(args) -> int:
    return pipeline.count(args.input, verbose=args.verbose)


def cmd_tag(args) -> int:
    from .tagging import write_tags

    ds = _open(args.input)
    output = Path(args.output) if args.output else Path(ds.path) / 'tags.jsonl'
    protected = (Path(ds.path) / INDEX_NAME, *(Path(p) for p in ds.shard_paths))
    if output.resolve() in {p.resolve() for p in protected}:
        print('scene-builder tag: 장면 데이터 파일을 태그 출력으로 쓸 수 없다.', file=sys.stderr)
        return 1
    scorer = None
    try:
        if args.score or args.pdm_deploy:
            from ._pdm.scoring import PDMScorer

            dataspec = (ds.stamp or {}).get('dataspec')
            if not dataspec:
                raise ValueError('PDM 채점에는 장면 데이터 규격이 필요하다.')
            if args.pdm_deploy:
                scorer = PDMScorer.from_deployment(args.pdm_deploy, dataspec, args.device)
            else:
                scorer = PDMScorer.embedded(dataspec, args.device)
        write_tags(ds, output, scorer)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'scene-builder tag: {exc}', file=sys.stderr)
        return 1
    print(output)
    return 0


def _open(path: str) -> SceneDataset:
    return SceneDataset(path, loads=decode.loads)


def cmd_info(args) -> int:
    ds = _open(args.input)
    size = sum(os.path.getsize(p) for p in ds.shard_paths)
    print(f'dataset   {ds.path}')
    print(f'shards    {len(ds.shard_paths)}')
    print(f'scenes    {len(ds)}')
    print(f'size      {size / 2**30:.2f} GiB ({size / max(len(ds), 1) / 2**20:.2f} MiB/scene)')
    st = ds.stamp or {}
    print(f"stamp     stage={st.get('stage')} format=v{st.get('format_version')} "
          f"torch={st.get('torch_version')} pickle={st.get('pickle_protocol')}")
    print(f"dataspec  {st.get('dataspec')}")
    return 0


def cmd_dump(args) -> int:
    ds = _open(args.input)
    scene = ds[args.index]
    print(f"meta  {scene.get('meta')}")
    for g, fields in FIELDS.items():
        v = scene[g].get('valid')
        if v is not None and v.ndim >= 1:
            used = int(v.reshape(v.shape[0], -1).any(axis=1).sum())
            print(f'{g}  (사용 슬롯 {used}/{v.shape[0]})')
        else:
            print(f'{g}  (valid={bool(v)})')
        for f in fields:
            a = scene[g][f]
            print(f'    {f:<13} {str(tuple(a.shape)):<16} {a.dtype.name:<8} '
                  f'min={_fmt(np.min(a)) if a.size else "-"} max={_fmt(np.max(a)) if a.size else "-"}')
    return 0


def _fmt(x) -> str:
    return f'{x:.3f}' if isinstance(x, (float, np.floating)) else str(x)


def cmd_check(args) -> int:
    ds = _open(args.input)
    problems_total = Counter()
    failed = 0

    for p in ds.shard_paths:
        for prob in check_stamp_item(read_stamp(p)):
            print(f'{os.path.basename(p)} {prob}')
            problems_total[prob.item] += 1

    indices = list(range(len(ds)))
    if args.sample and args.sample < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.sample))
    for i in indices:
        probs = check_scene(ds[i])
        if probs:
            failed += 1
            for prob in probs:
                problems_total[prob.item] += 1
                if args.verbose or failed <= 20:
                    print(f'scene {i} {prob}')

    print(f'\n점검 장면 {len(indices)} / {len(ds)}, 문제 있는 장면 {failed}')
    for item, name in ITEMS.items():
        mark = 'OK ' if problems_total[item] == 0 else 'NG '
        print(f'  {mark} {item} {name}' + (f'  ({problems_total[item]}건)' if problems_total[item] else ''))
    return 0 if not problems_total else 1


def cmd_view(args) -> int:
    from .view import write_grid, write_html, write_png

    ds = _open(args.input)
    os.makedirs(args.output, exist_ok=True)
    scenes = [ds[i] for i in args.index]
    for i, scene in zip(args.index, scenes):
        stem = f"{i:06d}_{str(scene['meta'].get('scene_id', ''))[:16]}"
        if not args.no_html:
            path = os.path.join(args.output, stem + '.html')
            write_html(scene, path, stride=args.stride, radius=args.radius)
            print(path)
        if args.png:
            path = os.path.join(args.output, stem + '.png')
            write_png(scene, path, radius=args.radius)
            print(path)
    if args.grid:
        path = os.path.join(args.output, 'grid.png')
        write_grid(scenes, path, cols=args.grid, radius=args.radius)
        print(path)
    return 0


EXAMPLES = {
    'build': [
        'scene-builder build -i tfrecords -o scenes',
        'scene-builder build -i tfrecords -o scenes --limit 1',
    ],
    'count': ['scene-builder count -i tfrecords -v'],
    'info': ['scene-builder info -i scenes'],
    'dump': ['scene-builder dump -i scenes --index 0'],
    'tag': ['scene-builder tag -i scenes',
            'scene-builder tag -i scenes --score -o scores.jsonl'],
    'check': ['scene-builder check -i scenes --sample 1000'],
    'view': ['scene-builder view -i scenes -o figures --index 0 1 2 --png --grid 3'],
}

REQUIRED = {
    'build': ('input', 'output'),
    'count': ('input',),
    'info': ('input',),
    'dump': ('input', 'index'),
    'tag': ('input',),
    'check': ('input',),
    'view': ('input', 'output', 'index'),
}

INPUT_HELP = {
    'build': 'TFExample 주행기록(TFRecord) 폴더 또는 파일',
    'count': 'TFExample 주행기록(TFRecord) 폴더 또는 파일',
}
OUTPUT_HELP = {
    'build': '장면 데이터셋을 쓸 폴더 (없으면 만든다)',
    'view': '그림을 쓸 폴더 (없으면 만든다)',
}


def _epilog(cmd: str | None) -> str:
    cmds = [cmd] if cmd else list(EXAMPLES)
    lines = ['예:'] + [f'  {e}' for c in cmds for e in EXAMPLES[c]]
    return '\n'.join(lines)


def _guide(parser: argparse.ArgumentParser, missing: list[str]) -> int:
    names = ', '.join(f'--{m}' for m in missing)
    print(f'{parser.prog}: 필요한 인자가 없다 — {names}\n', file=sys.stderr)
    parser.print_help(sys.stderr)
    return 2


def _korean_titles(p: argparse.ArgumentParser) -> None:
    p._optionals.title = '옵션'
    for a in p._actions:
        if isinstance(a, argparse._HelpAction):
            a.help = '이 안내를 보이고 끝낸다'


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    fmt = argparse.RawDescriptionHelpFormatter
    ap = argparse.ArgumentParser(
        prog='scene-builder', formatter_class=fmt,
        description='TFExample 주행기록 변환, 시나리오 분류, 장면 난이도 분류',
        epilog=_epilog(None))
    _korean_titles(ap)
    sub = ap.add_subparsers(dest='cmd', metavar='<명령>', title='명령')
    subs = {}

    def add(cmd: str, help_: str, fn) -> argparse.ArgumentParser:
        p = sub.add_parser(cmd, help=help_, description=help_, epilog=_epilog(cmd),
                           formatter_class=fmt)
        _korean_titles(p)
        p.add_argument('-i', '--input', metavar='DIR',
                       help=INPUT_HELP.get(cmd, '장면 데이터셋 폴더 (build 의 출력)'))
        if cmd in OUTPUT_HELP:
            p.add_argument('-o', '--output', metavar='DIR', help=OUTPUT_HELP[cmd])
        p.set_defaults(fn=fn)
        subs[cmd] = p
        return p

    p = add('build', '장면 데이터셋 구축', cmd_build)
    p.add_argument('--workers', type=int, default=0, metavar='N', help='병렬 수 (0 = 물리 코어와 메모리÷3 중 작은 값)')
    p.add_argument('--limit', type=int, default=0, metavar='N', help='앞에서 입력 샤드 N개만')
    p.add_argument('--overwrite', action='store_true', help='이미 있는 출력도 다시 쓴다 (기본: 건너뛰기)')

    p = add('count', '입력 레코드 수 세기', cmd_count)
    p.add_argument('-v', '--verbose', action='store_true', help='샤드별로 출력')

    add('info', '데이터셋의 샤드·장면 수, 크기, 스탬프', cmd_info)

    p = add('dump', '장면 한 장의 필드 형상·자료형·범위', cmd_dump)
    p.add_argument('--index', type=int, metavar='I', help='장면 전역 번호')

    p = add('tag', '시나리오 태그와 선택적 Score 태그 계산', cmd_tag)
    p.add_argument('-o', '--output', metavar='FILE', help='태그 JSONL (기본: 입력 폴더의 tags.jsonl)')
    score = p.add_mutually_exclusive_group()
    score.add_argument('--score', action='store_true', help='내장 PDM으로 Score 계산')
    score.add_argument('--pdm-deploy', metavar='DIR', help='외부 PDM 배포물로 Score 계산')
    p.add_argument('--device', default='cpu', metavar='DEVICE', help='Score 계산 장치 (cpu 또는 cuda:N)')

    p = add('check', '유효성 점검 9항목', cmd_check)
    p.add_argument('--sample', type=int, default=0, metavar='N', help='무작위 N장만 (0 = 전부)')
    p.add_argument('--seed', type=int, default=0, metavar='S', help='표본 추출 씨앗')
    p.add_argument('-v', '--verbose', action='store_true', help='문제를 전부 출력')

    p = add('view', '장면 그림 (HTML 재생, PNG, 격자)', cmd_view)
    p.add_argument('--index', type=int, nargs='+', metavar='I', help='장면 전역 번호 (여러 개)')
    p.add_argument('--png', action='store_true', help='장면마다 현재 프레임 PNG')
    p.add_argument('--grid', type=int, default=0, metavar='COLS', help='격자 PNG 한 장 (열 수)')
    p.add_argument('--no-html', action='store_true', help='HTML 을 쓰지 않는다')
    p.add_argument('--stride', type=int, default=5, metavar='K', help='슬라이더 프레임 간격 (5 = 2 Hz)')
    p.add_argument('--radius', type=int, default=50, metavar='M', help='자차 중심 표시 반경 [m]')

    return ap, subs


def main(argv: list[str] | None = None) -> int:
    ap, subs = build_parser()
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help(sys.stderr)
        return 2
    missing = [k for k in REQUIRED[args.cmd] if getattr(args, k, None) in (None, [])]
    if missing:
        return _guide(subs[args.cmd], missing)
    err = _need_input(args.input, dataset=args.cmd not in ('build', 'count'))
    if err:
        print(f'scene-builder {args.cmd}: {err}', file=sys.stderr)
        return 1
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
