"""Coding-agent evaluation (assistant-coding-v1): fix a bug so that visible AND hidden tests pass.

    python -m assistant.bench.coding_agent --backend ollama:qwen3.5:4b --out runs/<stamp>/coding_<name>.json

Configurations (same 10 tasks from assistant.bench.coding_tasks):
  given  the buggy file and the failing visible-test output are in the prompt; no tools (model-only coding)
  agent  only the instruction; the model must use read_file / list_dir / search_code / run_unittests in a throwaway
         workspace (run_unittests is auto-approved only for that task's visible test module inside the workspace)
The reply must contain the complete corrected file in a ```python block. The grader writes it into a fresh copy and runs
visible + hidden tests in a subprocess (timeout; dangerous-import screen). Reported per task: pass, failure reason,
tool calls (valid / rejected), model calls. Hidden tests never enter the model's workspace.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .coding_tasks import TASKS
from .grading import CODE_BLOCK_RE, DANGEROUS_RE, extract_code

PROTOCOL = 'assistant-coding-v1'
INSTRUCTION = ('The unit tests in research/tasklib/test_{name}.py fail. Find the bug in research/tasklib/{name}.py and fix '
               'it without changing the documented behaviour. Reply with the complete corrected research/tasklib/{name}.py '
               'in one ```python code block.')


def write_workspace(root, t, module_source, hidden=False):
    pkg = Path(root) / 'research' / 'tasklib'
    pkg.mkdir(parents=True, exist_ok=True)
    (Path(root) / 'research' / '__init__.py').write_text('', encoding='utf-8')
    (pkg / '__init__.py').write_text('', encoding='utf-8')
    (pkg / f"{t['name']}.py").write_text(module_source, encoding='utf-8')
    tests = f"import unittest\n\nfrom research.tasklib.{t['name']} import *  # noqa: F401,F403\n\n\nclass TestVisible(unittest.TestCase):\n{t['visible']}"
    if hidden:
        tests += f"\n\nclass TestHidden(unittest.TestCase):\n{t['hidden']}"
    (pkg / f"test_{t['name']}.py").write_text(tests + '\n', encoding='utf-8')


def run_tests(root, name, timeout=60):
    p = subprocess.run([sys.executable, '-m', 'unittest', f'research.tasklib.test_{name}'], cwd=root, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode == 0, (p.stdout + p.stderr)[-1500:]


def grade(t, module_source):
    """(passed, reason) for a candidate file, on visible + hidden tests in a fresh workspace."""
    if DANGEROUS_RE.search(module_source):
        return False, 'dangerous import or call'
    with tempfile.TemporaryDirectory() as d:
        write_workspace(d, t, module_source, hidden=True)
        try:
            ok, out = run_tests(d, t['name'])
        except subprocess.TimeoutExpired:
            return False, 'timeout'
    return ok, 'ok' if ok else out.strip().splitlines()[-1][:200]


def target_block(answer, t):
    """The code block that defines the task's function/class (a model may also echo the test file, which is longer;
    review V1 #4). Falls back to the longest block."""
    import re
    names = re.findall(r'^(?:def|class) (\w+)', t['fixed'], re.M)
    blocks = CODE_BLOCK_RE.findall(answer)
    defining = [b for b in blocks if any(re.search(rf'^(?:def|class) {n}\b', b, re.M) for n in names)
                and 'unittest.TestCase' not in b]
    return max(defining, key=len) if defining else extract_code(answer)


def run_task(backend, t, config, system, max_steps, tool_rule=True, force_final=True):
    from ..agent import Agent
    from ..context import ContextManager
    from ..tools import Toolbox
    with tempfile.TemporaryDirectory() as d:
        write_workspace(d, t, t['buggy'])
        test_module = f"research.tasklib.test_{t['name']}"
        instruction = INSTRUCTION.format(name=t['name'])
        if config == 'given':
            _, failing = run_tests(d, t['name'])
            visible_tests = (Path(d) / 'research' / 'tasklib' / f"test_{t['name']}.py").read_text(encoding='utf-8')
            prompt = (f"{instruction}\n\nresearch/tasklib/{t['name']}.py:\n```python\n{t['buggy']}```\n\n"
                      f"research/tasklib/test_{t['name']}.py:\n```python\n{visible_tests}```\n\n"
                      f"Test output:\n```\n{failing}\n```")
            toolbox = None
        else:
            prompt = instruction + f' The test module is {test_module}.'
            toolbox = Toolbox(d, approve=lambda n, a: n == 'run_unittests' and isinstance(a, dict) and a.get('module') == test_module,
                              test_timeout=60)
        agent = Agent(backend, ContextManager(system, 7000), toolbox, None, max_steps=max_steps, use_rag=False,
                      use_tools=config == 'agent', max_tokens=1024, tool_rule=tool_rule,
                      force_final=force_final)
        t0 = time.perf_counter()
        try:
            turn = agent.ask(prompt)
            answer, err = turn.answer, None
        except Exception as e:  # a crashed turn is a failure, reason kept
            turn, answer, err = None, '', f'{type(e).__name__}: {e}'
        calls = [c for s in (turn.steps if turn else []) for c in s['tool_calls']]
        results = [r for s in (turn.steps if turn else []) for r in s.get('tool_results', [])]
    code = target_block(answer, t) if answer else None
    passed, reason = grade(t, code + '\n') if code else (False, 'no code block' if not err else err)
    return {'task': t['name'], 'config': config, 'passed': passed, 'reason': reason, 'error': err,
            'tool_calls': [c['name'] for c in calls], 'tool_errors': sum(str(r).startswith(('ERROR', 'DENIED')) for r in results),
            'model_calls': len(turn.steps) if turn else 0, 'seconds': round(time.perf_counter() - t0, 1),
            'answer': answer[:3000]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--quant', default='bf16')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--configs', nargs='+', default=['given', 'agent'])
    ap.add_argument('--max-steps', type=int, default=8)
    ap.add_argument('--no-tool-rule', action='store_true', help='reproduce runs before ASSIST-06')
    ap.add_argument('--no-force-final', action='store_true', help='reproduce runs before ASSIST-06')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from ..agent import INTERACTIVE_SYSTEM
    if args.backend.startswith('ollama:'):
        from ..backends import OllamaBackend
        backend = OllamaBackend(args.backend.split(':', 1)[1], options={'presence_penalty': 0.0})
    else:
        from ..cli import make_backend
        backend = make_backend(args.backend, args.quant, args.revision)
    out = {'protocol': PROTOCOL, 'backend': args.backend, 'quant': args.quant, 'system': 'INTERACTIVE_SYSTEM',
           'decoding': 'greedy', 'tool_rule': not args.no_tool_rule,
           'force_final': not args.no_force_final, 'configs': {}}
    for cfg in args.configs:
        rows = [run_task(backend, t, cfg, INTERACTIVE_SYSTEM, args.max_steps, not args.no_tool_rule, not args.no_force_final)
                for t in TASKS]
        out['configs'][cfg] = {'passed': f"{sum(r['passed'] for r in rows)}/{len(rows)}", 'rows': rows}
        print(cfg, out['configs'][cfg]['passed'], [r['task'] for r in rows if not r['passed']], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
