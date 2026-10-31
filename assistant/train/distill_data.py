"""Execution-verified distillation data for the Python / PyTorch / GPU-performance specialisation.

    python -m assistant.train.distill_data --n 120 --out results/assistant/data/distill_v1.jsonl

Teacher: the locally installed Ollama `qwen3.5:4b` (Apache-2.0; outputs usable for training). For each request the
teacher invents ONE new function task in a given topic and returns a NAME line plus a solution block and a tests block
(v2; v1 asked for JSON, which failed 37% of the time and once put the body into the prompt). The student prompt is the
def line + docstring extracted from the solution's AST, so it can never contain the body. Kept only
if: it parses; the solution passes its own tests in the sandboxed grader; it passes the dangerous-code screen; its name
and text are not near-duplicates of an earlier example; and it does not collide with ANY assistant-quality-v1 item
(both splits, by entry name or ≥ 0.5 token-Jaccard on the signature), so evaluation stays untouched. Every rejection
is logged with its reason.
"""

import argparse
import ast
import json
import random
import re
import time
from pathlib import Path

from ..bench import quality_v1
from ..bench.grading import grade_code

TOPICS = [
    'string processing', 'parsing and validation', 'lists and dictionaries', 'sorting and searching', 'math and number theory',
    'numerics with floats (stability, rounding)', 'dates and durations (pure arithmetic, no datetime import needed)',
    'matrix and vector math on nested lists', 'GPU performance arithmetic (bandwidth, FLOPs, occupancy, memory sizes)',
    'transformer shape and memory arithmetic (parameters, KV cache, activations)', 'bit manipulation',
    'simple algorithms on graphs given as adjacency dicts', 'statistics (mean, median, percentiles, variance)',
    'PyTorch tensor operations (the function receives and returns torch tensors; import torch inside the function)',
]

TEACHER_PROMPT = '''Invent ONE new, self-contained Python programming task about: {topic}.
Requirements:
- A single top-level function with type hints and a one-line docstring that fully specifies the behaviour, including edge cases.
- Difficulty: {difficulty}. Use only the Python standard library{torch_note}. No file, network or OS access.
- Provide 4 to 6 assert-based tests that exercise normal and edge cases. Tests must be deterministic.
- Do NOT reuse any of these function names: {avoid}.
Reply in exactly this format and nothing else:
NAME: <function name>
```python
<the complete solution: def line with type hints, the docstring, and the body>
```
```python
<4 to 6 assert statements, each on its own line>
```'''

TOKEN = re.compile(r'[a-z0-9_]+')


def tokens(s):
    return set(TOKEN.findall(s.lower()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


BLOCK_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


def signature_of(solution, name):
    """Prompt text derived from the solution itself: the def line(s) plus the docstring, never the body.

    Returns None unless `name` is a top-level function with a docstring and at least one real statement after it.
    """
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    doc = ast.get_docstring(fn) if fn else None
    if fn is None or not doc or len(fn.body) < 2:
        return None
    lines = solution.splitlines()
    header = '\n'.join(lines[fn.lineno - 1:fn.body[0].lineno - 1])  # def line(s) up to the docstring
    indent = ' ' * (fn.body[0].col_offset or 4)
    doc_lines = doc.splitlines()
    quoted = f'{indent}"""{doc_lines[0]}' + ''.join(f'\n{indent}{l}' if l else '\n' for l in doc_lines[1:]) + '"""'
    return f'{header}\n{quoted}'


def parse_teacher(text):
    """NAME line + solution block + tests block -> {name, signature, solution, tests} or None."""
    m = re.search(r'NAME:\s*`?([A-Za-z_][A-Za-z0-9_]*)', text)
    blocks = BLOCK_RE.findall(text)
    if not m or len(blocks) < 2:
        return None
    name, solution, test_src = m.group(1), blocks[0].strip(), blocks[1].strip()
    signature = signature_of(solution, name)
    try:
        tests = [ast.get_source_segment(test_src, s) for s in ast.parse(test_src).body if isinstance(s, ast.Assert)]
    except SyntaxError:
        return None
    if signature is None or not tests:
        return None
    return {'name': name, 'signature': signature, 'solution': solution, 'tests': tests}


def to_example(obj):
    prompt = f"Write this Python function. Reply with only the complete function in one ```python code block.\n\n{obj['signature'].strip()}"
    return {'messages': [{'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': f"```python\n{obj['solution'].strip()}\n```"}],
            'name': obj['name'], 'tests': obj['tests']}


def load_kept(path):
    """Kept rows of an interrupted run. A torn last line (killed mid-write) is skipped and terminated with a newline so
    the next appended row starts on its own line instead of being glued to the fragment."""
    text = path.read_text(encoding='utf-8')
    kept = []
    for n, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            kept.append(json.loads(line))
        except json.JSONDecodeError:
            print(f'skipping malformed line {n + 1} of {path}', flush=True)
    if text and not text.endswith('\n'):
        with path.open('a', encoding='utf-8') as f:
            f.write('\n')
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=120, help='teacher requests (kept examples will be fewer)')
    ap.add_argument('--teacher', default='qwen3.5:4b')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted run: reload kept rows (names, dedup) from --out, start at request --start')
    ap.add_argument('--start', type=int, default=0, help='first request index (topic rotation continues from here)')
    args = ap.parse_args()
    if args.out.exists() and not args.resume:
        raise SystemExit(f'{args.out} exists; pass --resume --start <next request index> to continue it')
    from ..backends import OllamaBackend
    teacher = OllamaBackend(args.teacher, options={'presence_penalty': 0.0})
    # Fresh runs keep the original stream (reproducible); a resumed run draws fresh difficulties.
    rng = random.Random(args.seed if args.start == 0 else args.seed * 1_000_003 + args.start)
    held = quality_v1.items()
    banned_names = {i['entry'] for i in held if i['category'] == 'code'}
    banned_sigs = [tokens(i['prompt']) for i in held]
    kept, log, names = [], [], set(banned_names)
    if args.resume and args.out.exists():
        kept = load_kept(args.out)
        names |= {e['name'] for e in kept}
        last = max((e['k'] for e in kept if 'k' in e), default=None)
        if last is not None and args.start <= last:
            print(f'warning: --start {args.start} <= last kept request index {last}', flush=True)
        print(f'resuming at request {args.start} with {len(kept)} kept examples', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k in range(args.start, args.n):
        topic = TOPICS[k % len(TOPICS)]
        difficulty = rng.choice(['easy', 'medium', 'medium', 'hard'])
        prompt = TEACHER_PROMPT.format(topic=topic, difficulty=difficulty, avoid=', '.join(sorted(names))[:1500],
                                       torch_note=' and torch' if 'PyTorch' in topic else '')
        try:
            reply = teacher.chat([{'role': 'user', 'content': prompt}], max_tokens=900, temperature=0.8)
        except Exception as e:
            log.append({'k': k, 'topic': topic, 'status': f'teacher error {type(e).__name__}'})
            continue
        obj = parse_teacher(reply.text)
        status = 'kept'
        if obj is None:
            status = 'unparseable (format, or no docstring/body)'
        elif obj['name'] in names:
            status = 'duplicate or held-out name'
        elif max(jaccard(tokens(obj['signature']), s) for s in banned_sigs) >= 0.5:
            status = 'too similar to an evaluation item'
        elif any(jaccard(tokens(obj['signature']), tokens(e['messages'][0]['content'])) >= 0.7 for e in kept):
            status = 'near-duplicate of a kept example'
        else:
            ok, why = grade_code(f"```python\n{obj['solution']}\n```", obj['name'], [str(t) for t in obj['tests']], timeout=20)
            if not ok:
                status = f'solution failed its tests: {why[:80]}'
        log.append({'k': k, 'topic': topic, 'difficulty': difficulty, 'status': status, 'name': obj['name'] if obj else None,
                    'teacher_tokens': reply.stats.completion_tokens})
        if status == 'kept':
            names.add(obj['name'])
            kept.append(to_example(obj) | {'topic': topic, 'difficulty': difficulty, 'k': k})  # k: resume point
            with args.out.open('a', encoding='utf-8') as f:
                f.write(json.dumps(kept[-1]) + '\n')
        if (k + 1) % 10 == 0:
            print(f'{k + 1}/{args.n} requests, kept {len(kept)}, {time.perf_counter() - t0:.0f}s', flush=True)
    summary = {'requests': args.n - args.start, 'start': args.start, 'kept_total': len(kept),
               'kept': sum(1 for x in log if x['status'] == 'kept'), 'teacher': args.teacher, 'seed': args.seed,
               'outcomes': {s: sum(1 for x in log if x['status'].split(':')[0] == s) for s in sorted({x['status'].split(':')[0] for x in log})},
               'seconds': round(time.perf_counter() - t0, 1)}
    log_path = args.out.with_suffix(f'.log{"_from%d" % args.start if args.start else ""}.json')
    log_path.write_text(json.dumps({'summary': summary, 'log': log}, indent=2), encoding='utf-8')
    teacher.unload()
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
