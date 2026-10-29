# Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-ETRI-Commercial

from __future__ import annotations

import glob
import multiprocessing as mp
import os
import struct
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

from .dataspec.shard import INDEX_NAME, PART_EXT, SCENES_EXT, write_scene_index

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

GIB = 2 ** 30


def _count_tfrecord(path: str) -> int:
    count = 0
    with open(path, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack('<Q', header)[0]
            f.seek(4 + length + 4, 1)
            count += 1
    return count


def _is_input(name: str) -> bool:
    return ('.tfrecord' in name and not name.endswith(SCENES_EXT)
            and PART_EXT not in name and name != INDEX_NAME)


def input_shards(src: str) -> list[str]:
    if os.path.isfile(src):
        return [src]
    paths = sorted(p for p in glob.glob(os.path.join(src, '*.tfrecord*'))
                   if _is_input(os.path.basename(p)))
    if not paths:
        raise FileNotFoundError(f"TFRecord 파일이 없다: {src}")
    return paths


def output_name(input_path: str) -> str:
    return os.path.basename(input_path) + SCENES_EXT


_progress = None


def _init_worker(counter=None) -> None:
    global _progress
    _progress = counter
    import torch
    torch.set_num_threads(1)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def convert_shard(input_path: str, out_dir: str) -> dict:
    from .convert import convert_raw, stamp
    from .dataspec.shard import write_scene_shard
    from .tfexample import iter_scenes

    t0 = time.perf_counter()
    records = 0
    scenes = []
    for raw in iter_scenes(input_path):
        records += 1
        scenes.extend(convert_raw(raw))
        if _progress is not None:
            with _progress.get_lock():
                _progress.value += 1
    name = output_name(input_path)
    if scenes:
        write_scene_shard(scenes, out_dir, stamp=stamp(), filename=name)
    size = os.path.getsize(os.path.join(out_dir, name)) if scenes else 0
    return dict(name=name, records=records, scenes=len(scenes),
                dropped=records - len(scenes), bytes=size,
                seconds=time.perf_counter() - t0)


def _run(input_path: str, out_dir: str) -> dict:
    try:
        return convert_shard(input_path, out_dir)
    except Exception:
        return dict(name=output_name(input_path), error=traceback.format_exc())


def default_workers() -> int:
    import psutil
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    mem_gib = psutil.virtual_memory().total / GIB
    return max(1, min(physical, int(mem_gib // 3)))


def build(tfexample: str, out: str, *, workers: int = 0, limit: int = 0,
          overwrite: bool = False) -> int:
    from tqdm import tqdm

    shards = input_shards(tfexample)
    if limit:
        shards = shards[:limit]
    os.makedirs(out, exist_ok=True)
    todo = [p for p in shards
            if overwrite or not os.path.exists(os.path.join(out, output_name(p)))]
    skipped = len(shards) - len(todo)
    workers = max(1, min(workers or default_workers(), len(todo) or 1))
    print(f'입력 샤드 {len(shards)} (할 일 {len(todo)}, 건너뜀 {skipped}) · 워커 {workers} · 출력 {out}')

    t0 = time.perf_counter()
    done, failed = [], []
    interrupted = False
    if todo:
        total_records = sum(_count_tfrecord(p) for p in todo)
        print(f'레코드 {total_records} · tensorflow·torch 불러오는 중 …', flush=True)
        _init_worker()
        ctx = mp.get_context('spawn')
        counter = ctx.Value('q', 0)
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                   initializer=_init_worker, initargs=(counter,))
        bar = tqdm(total=total_records, unit='rec', dynamic_ncols=True, smoothing=0.05)
        try:
            pending = {pool.submit(_run, p, out) for p in todo}
            bar.set_postfix_str(f'샤드 0/{len(todo)} · 워커 준비 중')
            while pending:
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    r = fut.result()
                    if 'error' in r:
                        failed.append(r)
                        tqdm.write(f"[실패] {r['name']}\n{r['error']}")
                    else:
                        done.append(r)
                bar.n = counter.value
                bar.set_postfix_str(f'샤드 {len(done) + len(failed)}/{len(todo)}')
                bar.refresh()
        except KeyboardInterrupt:
            interrupted = True
            pool.shutdown(wait=False, cancel_futures=True)
        except BrokenProcessPool as e:
            pool.shutdown(wait=False, cancel_futures=True)
            bar.close()
            print(f'\n[중단] 워커 프로세스가 비정상 종료했다: {e}\n'
                  f'워커 수를 줄여(--workers 4) 다시 실행해 보라. 끝난 샤드는 건너뛴다.')
            write_scene_index(out)
            return 1
        else:
            pool.shutdown(wait=True)
        finally:
            bar.close()

    write_scene_index(out)
    wall = time.perf_counter() - t0
    records = sum(r['records'] for r in done)
    scenes = sum(r['scenes'] for r in done)
    size = sum(r['bytes'] for r in done)
    print(f'\n완료 샤드 {len(done)} · 실패 {len(failed)} · 건너뜀 {skipped}')
    print(f'레코드 {records} → 장면 {scenes} (경로 없음 제외 {records - scenes})')
    if done:
        per = sum(r['seconds'] for r in done) / len(done)
        print(f'시간 {wall:.0f}s (샤드당 평균 {per:.0f}s, 워커 {workers}) · 크기 {size / GIB:.2f} GiB'
              f' ({size / max(scenes, 1) / 2**20:.2f} MiB/장면)')
    if interrupted:
        print('중단됨 — 끝난 샤드는 남아 있다. 같은 명령으로 이어서 돈다.')
        return 130
    return 1 if failed else 0


def count(tfexample: str, *, verbose: bool = False) -> int:
    shards = input_shards(tfexample)
    total = 0
    for p in shards:
        n = _count_tfrecord(p)
        total += n
        if verbose:
            print(f'{os.path.basename(p)}  {n}')
    size = sum(os.path.getsize(p) for p in shards)
    print(f'샤드 {len(shards)} · 레코드 {total} · {size / GIB:.1f} GiB')
    return 0
