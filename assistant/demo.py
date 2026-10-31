"""EfficientGPT assistant V1 demo: two documented tasks, run end to end and graded (no human judgement needed).

    python -m assistant.demo [--backend ollama:qwen3.5:4b]

Task 1 (repo question with tools): "What default num_ctx does OllamaBackend use in assistant/backends.py?" — the agent may
retrieve and read files (read-only tools); graded by the assistant-agent-v1.2 rule (answer 8192).
Task 2 (bug fix with tools): the `median` task of assistant-coding-v1 in a throwaway workspace — the agent reads the code,
may run the visible tests (auto-approved for that module only), and must return a corrected file that passes the visible
AND hidden tests.
Prints each tool call, the answer, the verdict, and the wall time; exits 0 only if both tasks pass. Measured pass rates
over the full suites are in docs/EXPERIMENT_LOG.md (ASSIST-05/06); one demo run is an example, not a benchmark.
"""

import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', default='ollama:qwen3.5:4b')
    args = ap.parse_args()
    from .agent import INTERACTIVE_SYSTEM, Agent
    from .bench.agent_e2e import EVAL_EXCLUDE, QUESTIONS, classify
    from .bench.coding_agent import run_task
    from .bench.coding_tasks import TASKS
    from .cli import make_backend
    from .context import ContextManager
    from .retrieval import Retriever
    from .tools import Toolbox
    if args.backend.startswith('ollama:'):
        from .backends import OllamaBackend
        backend = OllamaBackend(args.backend.split(':', 1)[1], options={'presence_penalty': 0.0})
    else:
        backend = make_backend(args.backend, 'nf4')
    ok = []

    question, accept = next(q for q in QUESTIONS if 'num_ctx' in q[0])
    print(f'== Task 1: {question}')
    retriever = Retriever('.', exclude=EVAL_EXCLUDE)
    agent = Agent(backend, ContextManager(INTERACTIVE_SYSTEM, 7000),
                  Toolbox('.', approve=lambda n, a: False, retriever=retriever, exclude=EVAL_EXCLUDE), retriever,
                  max_steps=6, max_tokens=512)
    t0 = time.perf_counter()
    turn = agent.ask(question)
    for step in turn.steps:
        for call in step['tool_calls']:
            print(f"   tool: {call['name']}({call['arguments']})")
    verdict = classify(turn.answer, accept)
    print(f'   answer: {turn.answer.strip()[:300]}\n   verdict: {verdict}  ({time.perf_counter() - t0:.1f} s)')
    ok.append(verdict == 'correct')

    task = next(t for t in TASKS if t['name'] == 'median')
    print('\n== Task 2: fix the bug in research/tasklib/median.py (visible + hidden tests)')
    t0 = time.perf_counter()
    row = run_task(backend, task, 'agent', INTERACTIVE_SYSTEM, max_steps=8)
    print(f"   tools: {row['tool_calls']}\n   verdict: {'PASS' if row['passed'] else 'FAIL'} ({row['reason']})  "
          f"({time.perf_counter() - t0:.1f} s)")
    ok.append(row['passed'])
    print(f"\nDemo: {sum(ok)}/2 tasks passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == '__main__':
    main()
