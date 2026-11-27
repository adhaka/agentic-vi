"""Reproduce Task 001 from the repository root in a new output directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agentic_vi.targets import make_gaussian_target
from run_baseline import run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sources = ['protocol.yaml', 'agentic_vi/algorithms/mfvi.py', 'agentic_vi/evaluation.py',
               'agentic_vi/targets.py', 'run_baseline.py', 'research/task001/run.py']
    manifest = {
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sha256': {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
        'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'platform': platform.platform(), 'processor': platform.processor(),
        'device': 'cpu', 'intraop_threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(), 'failures': [],
    }
    start = time.perf_counter()
    cpu_start = time.process_time()
    for kind in ('independent', 'correlated', 'ill_conditioned'):
        for dimension in (10, 50, 100):
            args = argparse.Namespace(target=kind, dimension=dimension, steps=3000,
                                      learning_rate=.03, mc_samples=128, log_every=100,
                                      dtype='float64')
            cov = make_gaussian_target(dimension, kind).covariance
            np.save(output / f'{kind}_d{dimension}_cov.npy', cov.numpy(), allow_pickle=False)
            for seed in range(5):
                label = f'{kind}_d{dimension}_seed{seed}'
                try:
                    data = run_one(args, seed)
                    (output / f'{label}.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
                except Exception:
                    manifest['failures'].append({'run': label, 'traceback': traceback.format_exc()})
            print(kind, dimension, flush=True)
    manifest.update(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return int(bool(manifest['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
