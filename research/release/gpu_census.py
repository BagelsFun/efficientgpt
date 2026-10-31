"""Which of the release test suites actually exercise the GPU?

    python -m research.release.gpu_census --out runs/<stamp>/test_gpu_census.json

Runs every suite of research.release.clean_checkout twice in the current checkout (each in its own process): once normally and once with
CUDA hidden (CUDA_VISIBLE_DEVICES=-1). A test that passes in the normal run but skips or fails without CUDA needs the GPU;
one that passes both times runs on the CPU (it may still use the GPU opportunistically, which the CUDA-hidden run rules
out as a requirement). Per-test outcomes are collected with a unittest result object, not parsed from logs.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from .clean_checkout import SUITES


class _Result(unittest.TextTestResult):
    """Records one outcome per test id (robust against tests that print to stderr)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.outcomes = {}

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.setdefault(test.id(), 'ok')

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes[test.id()] = 'skipped'

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes[test.id()] = 'FAIL'

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes[test.id()] = 'ERROR'

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.outcomes[test.id()] = 'FAIL'


def collect(modules, out_path):
    """Child-process entry: run the modules and write {test id: outcome} plus the run count as JSON."""
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(resultclass=_Result, verbosity=0, stream=io.StringIO()).run(suite)
    Path(out_path).write_text(json.dumps({'ran': result.testsRun, 'outcomes': result.outcomes,
                                          'ok': result.wasSuccessful()}), encoding='utf-8')


def run(modules, hide_cuda):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    if hide_cuda:
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    with tempfile.TemporaryDirectory() as d:
        out_path = Path(d) / 'outcomes.json'
        p = subprocess.run([sys.executable, '-m', 'research.release.gpu_census', '--collect', str(out_path), *modules],
                           capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        if not out_path.exists():
            raise SystemExit(f'collector failed for {modules}:' + chr(10) + p.stderr[-2000:])
        r = json.loads(out_path.read_text(encoding='utf-8'))
    return r['outcomes'], 0 if r['ok'] else 1, r['ran'], ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--collect', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('modules', nargs='*', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.collect:
        return collect(args.modules, args.collect)
    if not args.out:
        ap.error('--out is required')
    res = {'suites': {}}
    for name, modules in SUITES.items():
        gpu, rc_gpu, ran_gpu, _ = run(modules, False)
        cpu, rc_cpu, ran_cpu, _ = run(modules, True)
        needs = sorted(t for t, s in gpu.items() if s == 'ok' and cpu.get(t) != 'ok')
        res['suites'][name] = {
            'ran_with_gpu': ran_gpu, 'returncode_with_gpu': rc_gpu, 'recorded_with_gpu': len(gpu),
            'skipped_with_gpu': sorted(t for t, s in gpu.items() if s == 'skipped'),
            'ran_cuda_hidden': ran_cpu, 'returncode_cuda_hidden': rc_cpu,
            'needs_gpu': needs, 'cpu_only': sorted(t for t, s in gpu.items() if s == 'ok' and cpu.get(t) == 'ok'),
            'cuda_hidden_outcomes_of_gpu_tests': {t: cpu.get(t, 'missing') for t in needs},
            'not_ok_with_gpu': {t: s for t, s in gpu.items() if s not in ('ok', 'skipped')}}
        s = res['suites'][name]
        print(f"{name:22s} ran={ran_gpu} recorded={len(gpu)} needs_gpu={len(needs)} cpu_only={len(s['cpu_only'])}", flush=True)
    res['total_ran'] = sum(s['ran_with_gpu'] or 0 for s in res['suites'].values())
    res['total_needs_gpu'] = sum(len(s['needs_gpu']) for s in res['suites'].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f"total ran {res['total_ran']}, need GPU {res['total_needs_gpu']}")


if __name__ == '__main__':
    main()
