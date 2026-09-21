"""Bounded real-provider pilot; preserves the selected scientific feature bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from sciel.cli.main import _run_engine
from sciel.config import load_config, validate
from sciel.engine.factory import build_engine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('profile')
    parser.add_argument('run_dir')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--islands',type=int,default=4)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    directory=Path(args.run_dir).resolve()
    if directory.exists():
        raise SystemExit('Pilot requires a fresh destination')
    cfg=load_config(args.profile)
    cfg.update(budget=args.iterations,max_iterations=args.iterations,num_islands=args.islands,
               max_islands=args.islands,dynamic_islands=False)
    cfg['parallel_search']['workers']=args.workers
    cfg=validate(cfg)
    if not cfg['parallel_search']['enabled'] or cfg['llm'].get('mock'):
        raise SystemExit('Pilot requires enabled parallel search and a real provider')
    if cfg.get('warm_start') is not None:
        raise SystemExit('Pilot must start from the canonical task seed')
    engine=build_engine(cfg,str(root/'tasks'),str(directory),python=sys.executable)
    started=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_SELF)
    try:
        result=_run_engine(engine)
    finally:
        engine.close()
    after=resource.getrusage(resource.RUSAGE_SELF)
    summary={'schema':'sciel-real-parallel-pilot/1','profile':str(Path(args.profile).resolve()),
        'run_dir':str(directory),'workers':args.workers,'initial_islands':args.islands,
        'iteration_limit':args.iterations,'elapsed_seconds':time.monotonic()-started,
        'cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'peak_rss_bytes':after.ru_maxrss if sys.platform=='darwin' else after.ru_maxrss*1024,
        'checkpoint_bytes':(directory/'checkpoint.json').stat().st_size,'result':result}
    (directory/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
