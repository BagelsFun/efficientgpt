"""Clean-checkout reproduction check: run every test suite from a fresh `git worktree` of a commit.

    python -m research.release.clean_checkout --ref HEAD --out runs/<stamp>/clean_checkout.json [--keep]

Proves the release does not depend on untracked files in the working copy. Steps:
1. `git worktree add --detach <tmp> <ref>` (a pristine tree of exactly the committed files).
2. Copy in only the gitignored inputs the tests need: the TinyStories checkpoint (its SHA-256 must match
   docs/CHECKPOINT_MANIFEST.sha256, so a corrupted or different checkpoint fails the check) and the tokenizer (hash
   recorded). Originals are only read.
3. Run the three suites (Phase 5, KernelAgent-Ada + real model, assistant) with the current interpreter, cwd = worktree;
   record pass / fail / skip counts.
4. Remove the worktree (unless --keep).
Not covered: a fresh virtual environment (needs ~3 GB of downloads; see REPRODUCE.md and requirements-lock.txt) and the
HF / Ollama model caches, which are global per user.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SUITES = {
    'phase5': ['research.phase4.test_decoder', 'research.phase4.test_compact', 'research.phase5.test_pool',
               'research.phase5.test_prefix', 'research.phase5.test_graph_gpu', 'research.phase5.test_analyze_ci',
               'research.phase5.test_near_tie', 'research.phase5.test_compiled_step', 'research.phase5.test_inprocess',
               'research.phase5.test_eos', 'research.phase5.test_sampling', 'research.phase5.test_ragged',
               'research.phase5.test_gpu_combinations', 'research.v1.test_demo', 'research.v1.test_report'],
    'kernelagent_realmodel': ['research.kernelagent.test_evaluate', 'research.kernelagent.test_search',
                              'research.realmodel.test_int8_linear', 'research.realmodel.test_fast_decode'],
    'assistant': ['assistant.test_assistant', 'assistant.bench.test_quality_suite', 'assistant.train.test_train'],
    'release': ['research.release.test_release'],
}
INPUTS = {'out-tinystories-v2/baseline-2500.pt': 'manifest', 'data/tinystories_v2/tokenizer.json': 'record',
          # found by the first clean-checkout run: test_gpu_combinations reads the validation split (V3)
          'data/tinystories_v2/val.bin': 'record'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    out = {}
    for line in Path('docs/CHECKPOINT_MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            digest, name = line.split(maxsplit=1)
            out[name.lstrip('*')] = digest
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', default='HEAD')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--no-inputs', action='store_true',
                    help='do not copy the private checkpoint / data: simulates an external reproducer (tests needing them must skip)')
    args = ap.parse_args()
    commit = subprocess.run(['git', 'rev-parse', args.ref], capture_output=True, text=True, check=True).stdout.strip()
    tree = Path(tempfile.mkdtemp(prefix='efficientgpt_clean_'))
    shutil.rmtree(tree)
    subprocess.run(['git', 'worktree', 'add', '--detach', str(tree), commit], check=True, capture_output=True)
    res = {'commit': commit, 'worktree': str(tree), 'python': sys.executable, 'inputs': {}, 'suites': {},
           'mode': 'external (no private inputs)' if args.no_inputs else 'full'}
    try:
        expected = manifest()
        for rel, policy in ({} if args.no_inputs else INPUTS).items():
            digest = sha256(rel)
            ok = policy != 'manifest' or expected.get(rel) == digest
            res['inputs'][rel] = {'sha256': digest, 'policy': policy, 'ok': ok}
            if not ok:
                raise SystemExit(f'{rel}: SHA-256 does not match docs/CHECKPOINT_MANIFEST.sha256')
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rel, tree / rel)
        for name, modules in SUITES.items():
            t = time.perf_counter()
            p = subprocess.run([sys.executable, '-m', 'unittest', *modules], cwd=tree, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', env=dict(__import__('os').environ, PYTHONIOENCODING='utf-8'))
            tail = p.stderr[-3000:]
            ran = re.search(r'Ran (\d+) tests', tail)
            skipped = re.search(r'skipped=(\d+)', tail)
            failures = re.search(r'failures=(\d+)', tail)
            errors = re.search(r'errors=(\d+)', tail)
            res['suites'][name] = {'returncode': p.returncode, 'ran': int(ran.group(1)) if ran else None,
                                   'failures': int(failures.group(1)) if failures else 0,
                                   'errors': int(errors.group(1)) if errors else 0,
                                   'skipped': int(skipped.group(1)) if skipped else 0,
                                   'seconds': round(time.perf_counter() - t, 1), 'tail': tail[-600:]}
            print(f"{name:22s} rc={p.returncode} ran={res['suites'][name]['ran']} skipped={res['suites'][name]['skipped']}",
                  flush=True)
    finally:
        if not args.keep:
            subprocess.run(['git', 'worktree', 'remove', '--force', str(tree)], capture_output=True)
    res['all_passed'] = all(s['returncode'] == 0 for s in res['suites'].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('ALL PASSED' if res['all_passed'] else 'FAILURES', flush=True)


if __name__ == '__main__':
    main()
