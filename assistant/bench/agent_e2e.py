"""Agent end-to-end evaluation on repository questions (assistant-agent-v1.2).

Answers are facts stated in this repository's files, graded by a regex on the
normalized answer. Three configurations separate model ability from assistance:
  model_only   no retrieval, no tools
  rag          retrieved excerpts in the prompt, no tools
  rag_tools    retrieval + read-only tools (read_file, list_dir, search_code, retrieve_docs)
Execute-level tools are never approved here.

v1.2 adds 12 questions (20 total; v1.1 results cover the first 8 only) and classifies every answer as
correct / abstained (says it does not know or needs the file) / unsupported (wrong value, false claim about the file, fabricated content, or deflection). A model-only configuration
cannot know repo facts, so its unsupported-answer rate measures calibration. `--system calibrated` tests a prompt that
forbids guessing unseen values.

    python -m assistant.bench.agent_e2e --backend ollama:qwen3.5:4b --out results/assistant/bench/<name>_agent.json
"""

import argparse
import json
import re
import time
from pathlib import Path

PROTOCOL = 'assistant-agent-v1.2'
# v1.1: benchmark/test files contain the questions and expected answers, so retrieval and tools must not see them
# (v1 runs leaked them; see EXPERIMENT_LOG ASSIST-02).
EVAL_EXCLUDE = ('assistant/bench/*', 'assistant/test_assistant.py')
QUESTIONS = [  # (question, regex the normalized answer must match)
    ('What is the value of NEAR_TIE_LOGIT_MARGIN in research/phase5/benchmark_review.py?', r'1e-2|0\.01\b'),
    ('Which permission level does the run_unittests tool have in assistant/tools.py?', r'\bexecute\b'),
    ('What is the default eos_id of EOSDecoder in research/phase5/eos.py?', r'(?<![\w.])0(?!\.?\d|\w)'),
    ('What string is PROTOCOL set to in research/kernelagent/tasks.py?', r'kernelagent-ada/tasks-v1'),
    ('Which function in research/phase5/sampling.py implements inverse-CDF sampling from probabilities and uniforms?', r'sample_from'),
    ('Which triton-windows version is recorded in docs/ENVIRONMENT.md?', r'3\.7\.1'),
    ('What achievable DRAM bandwidth in GB/s does the project use as its roofline (see docs/EXPERIMENT_LOG.md)?', r'(?<![\w.])211(?!\.?\d|\w)'),
    ('In research/phase5/decode_attention.py, what (SPLIT, BLOCK_N, num_warps) does default_config return when batch*heads >= 48?',
     r'64\D{1,25}32\D{1,25}4\b'),
    # v1.2 additions (answers verified in non-excluded files, 2026-10-31)
    # numbers are anchored (not part of a longer number or identifier: "4.5", "line 34" must not match 4; review KA-06)
    ('What is MEM_FRACTION in research/kernelagent/evaluate.py?', r'(?<![\w.])0?\.45(?!\d)'),
    ('What is TOL_FACTOR in research/kernelagent/evaluate.py?', r'(?<![\w.])4(?:\.0)?(?!\.?\d|\w)'),
    ('How many back-to-back calls (INNER) does research/kernelagent/evaluate.py capture per CUDA graph?', r'(?<![\w.])20(?!\.?\d|\w)'),
    ('What is the default --budget of research/kernelagent/search.py?', r'(?<![\w.])12(?!\.?\d|\w)'),
    ('What is SINGLE_PASS_MIN_PROGRAMS in research/phase5/decode_attention.py?', r'(?<![\w.])48(?!\.?\d|\w)'),
    ('Which Qwen3-1.7B revision hash does assistant/train/train_qlora.py pin?', r'70d244cc'),
    ('What is the default LoRA --rank in assistant/train/train_qlora.py?', r'(?<![\w.])16(?!\.?\d|\w)'),
    ('What default num_ctx does OllamaBackend use in assistant/backends.py?', r'(?<![\w.])8192(?!\.?\d|\w)|8,192'),
    ('What is VOCAB in research/kernelagent/tasks.py?', r'(?<![\w.])4096(?!\.?\d|\w)|4,096'),
    ('What is SAMPLE_TOL in research/kernelagent/evaluate.py?', r'1e-0?5(?!\d)|0\.00001(?!\d)'),
    ('What is DEFAULT_TIMEOUT (seconds) in research/kernelagent/evaluate.py?', r'(?<![\w.])300(?!\.?\d|\w)'),
    ('According to reports/KERNELAGENT_ADA_REPORT.md, which template family won the decode-attention task?', r'single[_ ]pass'),
]
# Abstention: says it lacks the information / access, or asks for the file (up to two words between negation and verb:
# "I cannot directly read ..."). Checked after FALSE_ABSENCE_RE, because "X is not defined in the file" is a factual
# claim about the file (false here: every answer exists), not an admission of uncertainty.
ABSTAIN_RE = re.compile(
    r"((?:cannot|can't|can not|unable to|do not|don't|does not have|doesn't have)(?: \w+){0,2} (?:access|read|see|know|"
    r"determine|verify|tell|open|view|look at)|do(?:n't| not) have (?:access|enough|the (?:file|information|content))|"
    r"no access|not (?:sure|certain|able to)|without (?:access|seeing|reading|the file)|need to (?:see|read|look at|check)|"
    r"please (?:provide|share|upload|paste|read|run|check)|(?:is|was) not (?:provided|shown|available|included) (?:in|to)|"
    r"i do not know|i don't know|unknown to me|"
    # honest phrasings found in the V1 audit (graded unsupported before)
    r"(?:could not|couldn't|did not|didn't) find|have no information|so i cannot say|cannot say|"
    r"not in the (?:provided|given) (?:context|excerpts))")
FALSE_ABSENCE_RE = re.compile(
    r"((?:is|are) not (?:explicitly |directly )?(?:defined|specified|stated|mentioned|named|set|listed|given|recorded)"
    r"|does not (?:explicitly )?(?:specify|contain|define|mention|state|name|list|include|record)|no such (?:variable|constant|function))")
from ..agent import CALIBRATION_RULE as CALIBRATED_SYSTEM_SUFFIX  # noqa: E402  (single source of the prompt text)


# A specific value offered as a guess ("typically 16", "8 or 16", "would be 0.9"). Combined with an abstention or a
# matching value it makes the verdict ambiguous, so such answers go to manual review (independent review, KA-06).
GUESS_RE = re.compile(r"(\b(?:guess|probably|likely|typically|usually|commonly|standard|often|might be|could be|would be)\b"
                      r"[^.]{0,60}\d|\d+(?:\.\d+)?\s*or\s*\d)")
VERDICTS = ('correct', 'abstained', 'unsupported', 'review', 'error')


def classify(answer, accept):
    """correct / abstained / unsupported, or 'review' when signals conflict: a matching value inside a hedge or next to
    an abstention, an abstention plus a claim that the fact is absent, or an abstention plus a guessed value.
    'unsupported' = wrong value, false claim that the fact is absent, fabricated content, or a deflection that neither
    answers nor admits uncertainty. Crashed turns are 'error' (set by the caller), never 'unsupported'."""
    a = norm(answer)
    correct, abstain = re.search(accept, a), ABSTAIN_RE.search(a)
    absent, guess = FALSE_ABSENCE_RE.search(a), GUESS_RE.search(a)
    if correct:
        return 'review' if (abstain or guess) else 'correct'
    if absent:
        return 'review' if abstain else 'unsupported'
    if abstain:
        return 'review' if guess else 'abstained'
    return 'unsupported'


def norm(s):
    return re.sub(r'\s+', ' ', s.lower().replace('`', '').replace('*', '')).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--quant', default='bf16')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--configs', nargs='+', default=['model_only', 'rag', 'rag_tools'])
    ap.add_argument('--system', choices=['default', 'calibrated'], default='default')
    ap.add_argument('--first', type=int, default=0, help='>0: only the first N questions (8 = the v1.1 set)')
    ap.add_argument('--no-tool-rule', action='store_true', help='reproduce runs before ASSIST-06')
    ap.add_argument('--no-force-final', action='store_true', help='reproduce runs before ASSIST-06')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from ..agent import DEFAULT_SYSTEM, Agent
    from ..cli import make_backend
    from ..context import ContextManager
    from ..retrieval import Retriever
    from ..tools import Toolbox
    if args.backend.startswith('ollama:'):
        from ..backends import OllamaBackend
        backend = OllamaBackend(args.backend.split(':', 1)[1], options={'presence_penalty': 0.0})
    else:
        backend = make_backend(args.backend, args.quant, args.revision)
    retriever = Retriever('.', exclude=EVAL_EXCLUDE)
    system = DEFAULT_SYSTEM + (CALIBRATED_SYSTEM_SUFFIX if args.system == 'calibrated' else '')
    questions = QUESTIONS[:args.first] if args.first else QUESTIONS
    results = {'protocol': PROTOCOL, 'backend': args.backend, 'quant': args.quant, 'system': args.system,
               'tool_rule': not args.no_tool_rule, 'force_final': not args.no_force_final, 'configs': {}}
    for cfg in args.configs:
        rows = []
        for q, accept in questions:
            agent = Agent(backend, ContextManager(system, 7000), Toolbox('.', approve=lambda n, a: False, retriever=retriever, exclude=EVAL_EXCLUDE),
                          retriever, max_steps=6, use_rag=cfg != 'model_only', use_tools=cfg == 'rag_tools', max_tokens=512,
                          tool_rule=not args.no_tool_rule, force_final=not args.no_force_final)
            t = time.perf_counter()
            try:
                turn = agent.ask(q)
                answer, err = turn.answer, None
            except Exception as e:  # a crashed turn counts as a failure, with the reason kept
                turn, answer, err = None, '', f'{type(e).__name__}: {e}'
            verdict = 'error' if err else classify(answer, accept)
            ok = verdict == 'correct'
            rows.append({'question': q, 'passed': ok, 'verdict': verdict, 'answer': answer[:6000], 'error': err,
                         'tool_calls': [c['name'] for s in (turn.steps if turn else []) for c in s['tool_calls']],
                         'model_calls': len(turn.steps) if turn else 0, 'seconds': round(time.perf_counter() - t, 2)})
        counts = {v: sum(r['verdict'] == v for r in rows) for v in VERDICTS}
        results['configs'][cfg] = {'score': f"{counts['correct']}/{len(rows)}", **counts, 'rows': rows}
        print(cfg, json.dumps(counts), flush=True)
    if hasattr(backend, 'unload'):
        backend.unload()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
