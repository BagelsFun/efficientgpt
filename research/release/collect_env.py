"""Record the machine a reproduction ran on, so results from different GPUs can be compared fairly.

    python -m research.release.collect_env --out runs/repro_<gpu>_<date>/env.json

Writes OS, CPU, Python and package versions, the GPU (name, compute capability, memory, driver, clocks, power limit,
PCIe link), current utilisation and the Ollama version if installed. Every field is best effort: a missing tool gives
null, never an error. Nothing is sent anywhere; read the file before attaching it."""

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SMI_FIELDS = ['name', 'driver_version', 'memory.total', 'clocks.max.sm', 'clocks.max.memory', 'power.limit',
              'pcie.link.gen.current', 'pcie.link.width.current', 'utilization.gpu', 'temperature.gpu']


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def version(module):
    try:
        return getattr(importlib.import_module(module), '__version__', 'installed')
    except Exception:
        return None


def cpu_name():
    if sys.platform.startswith('linux'):
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def collect():
    env = {'os': platform.platform(), 'kernel': platform.release(), 'cpu': cpu_name(), 'python': sys.version.split()[0],
           'packages': {m: version(m) for m in ('torch', 'triton', 'transformers', 'torchao', 'numpy', 'tiktoken')}}
    try:
        import torch
        env['torch_cuda'] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env['gpu'] = {'name': p.name, 'capability': f'{p.major}.{p.minor}', 'memory_gib': round(p.total_memory / 2**30, 2),
                          'sm_count': p.multi_processor_count, 'device_count': torch.cuda.device_count()}
    except Exception as e:  # torch missing or broken: still record the rest
        env['torch_error'] = repr(e)
    smi = run(['nvidia-smi', f'--query-gpu={",".join(SMI_FIELDS)}', '--format=csv,noheader,nounits'])
    if smi:
        env['nvidia_smi'] = dict(zip(SMI_FIELDS, [x.strip() for x in smi.splitlines()[0].split(',')]))
    if shutil.which('ollama'):
        env['ollama'] = run(['ollama', '--version'])
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    env = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(env, indent=1), encoding='utf-8')
    gpu = env.get('gpu', {})
    print(f"{gpu.get('name', 'no CUDA GPU')} (sm_{gpu.get('capability', '?').replace('.', '')}), "
          f"torch {env['packages']['torch']}, transformers {env['packages']['transformers']} -> {args.out}")


if __name__ == '__main__':
    main()
