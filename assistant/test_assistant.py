"""CPU tests for the assistant core (no model needed): parsing, context, retrieval, tool permissions, agent loop."""
import tempfile
import unittest
from pathlib import Path

from .agent import Agent
from .backends import GenStats, Reply, split_thinking_and_tools
from .context import ContextManager
from .retrieval import BM25, Retriever
from .tools import Toolbox


class FakeBackend:
    """Replays scripted replies; records the messages it was given."""
    name = 'fake'

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, **kw):
        self.seen.append((messages, tools))
        return self.replies.pop(0)


class TestParsing(unittest.TestCase):
    def test_think_and_tool_calls(self):
        text, think, calls = split_thinking_and_tools(
            '<think>plan it</think>Sure.<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>')
        self.assertEqual((text, think), ('Sure.', 'plan it'))
        self.assertEqual(calls, [{'name': 'read_file', 'arguments': {'path': 'a.py'}}])

    def test_malformed_and_unterminated(self):
        _, _, calls = split_thinking_and_tools('<tool_call>{not json}</tool_call>')
        self.assertEqual(calls[0]['name'], '__malformed__')
        text, think, _ = split_thinking_and_tools('<think>ran out of budget')
        self.assertEqual((text, think), ('', 'ran out of budget'))


class TestContext(unittest.TestCase):
    def test_budget_drops_oldest_and_keeps_system(self):
        ctx = ContextManager('SYS', budget_tokens=60, count=lambda t: len(t.split()))
        for i in range(10):
            ctx.add({'role': 'user', 'content': f'question {i} ' + 'word ' * 8})
        msgs = ctx.build()
        self.assertEqual(msgs[0], {'role': 'system', 'content': 'SYS'})
        self.assertIn('question 9', msgs[-1]['content'])
        self.assertNotIn('question 0', ' '.join(m['content'] for m in msgs))
        self.assertGreater(ctx.dropped(), 0)

    def test_never_starts_with_orphan_tool_result(self):
        ctx = ContextManager('S', budget_tokens=12, count=lambda t: len(t.split()))
        ctx.add({'role': 'user', 'content': 'a b c d e f g h'})
        ctx.add({'role': 'tool', 'content': 'x'})
        ctx.add({'role': 'assistant', 'content': 'done'})
        self.assertNotEqual(ctx.build()[1]['role'], 'tool')


class TestRetrieval(unittest.TestCase):
    def test_bm25_ranks_matching_chunk_first(self):
        s = BM25(['cuda graph replay decode step', 'tokenizer vocabulary merges', 'graph capture of cuda kernels'])
        scores = s.scores('cuda graph')
        self.assertEqual(max(range(3), key=lambda i: scores[i]) in (0, 2), True)
        self.assertEqual(scores[1], 0.0)

    def test_repo_index_finds_known_doc(self):
        r = Retriever(Path(__file__).resolve().parents[1])
        hits = r.search('near-tie token rule margin', k=5, mode='bm25')
        self.assertTrue(any('PHASE5_CONTRACT' in h.path or 'benchmark_review' in h.path for h in hits), [h.path for h in hits])


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / 'docs').mkdir()
        (root / 'docs' / 'a.md').write_text('line one\nline two\nalpha beta\n')
        (root / 'out-model').mkdir()
        (root / 'out-model' / 'ckpt.pt').write_bytes(b'\0')
        (root / 'secret.env').write_text('TOKEN=x')
        self.approvals = []
        self.box = Toolbox(root, approve=lambda n, a: (self.approvals.append(n), False)[1])

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_and_search(self):
        self.assertIn('2: line two', self.box.call('read_file', {'path': 'docs/a.md'}))
        self.assertIn('docs/a.md:3: alpha beta', self.box.call('search_code', {'pattern': 'alpha'}))

    def test_confinement(self):
        for path in ('../outside.txt', 'out-model/ckpt.pt', 'secret.env', '.git/config'):
            self.assertTrue(self.box.call('read_file', {'path': path}).startswith('ERROR'), path)
        self.assertNotIn('out-model', self.box.call('list_dir', {'path': '.'}))

    def test_execute_requires_approval(self):
        result = self.box.call('run_unittests', {'module': 'research.phase5.test_pool'})
        self.assertTrue(result.startswith('DENIED'))
        self.assertEqual(self.approvals, ['run_unittests'])

    def test_rejects_arbitrary_commands(self):
        box = Toolbox(self.tmp.name, approve=lambda n, a: True)
        self.assertTrue(box.call('run_unittests', {'module': 'os; rm -rf /'}).startswith('ERROR'))
        from .tools import ToolError
        with self.assertRaises(ToolError):
            box.call('shell', {'cmd': 'dir'})


class TestAgentLoop(unittest.TestCase):
    def test_tool_call_then_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'notes.md').write_text('the answer is 42\n')
            fake = FakeBackend([Reply('', tool_calls=[{'name': 'read_file', 'arguments': {'path': 'notes.md'}}]),
                                Reply('It is 42.')])
            agent = Agent(fake, ContextManager('S'), Toolbox(tmp), None, use_rag=False)
            turn = agent.ask('what is the answer?')
            self.assertEqual(turn.answer, 'It is 42.')
            self.assertTrue(turn.used_tools)
            second_call_messages = fake.seen[1][0]
            self.assertTrue(any(m['role'] == 'tool' and '42' in m['content'] for m in second_call_messages))

    def test_unknown_tool_is_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeBackend([Reply('', tool_calls=[{'name': 'shell', 'arguments': {'cmd': 'dir'}}]), Reply('ok')])
            turn = Agent(fake, ContextManager('S'), Toolbox(tmp), None, use_rag=False).ask('x')
            self.assertEqual(turn.answer, 'ok')
            self.assertTrue(any(m['role'] == 'tool' and 'unknown tool' in m['content'] for m in fake.seen[1][0]))

    def test_step_limit(self):
        loop = [Reply('', tool_calls=[{'name': 'list_dir', 'arguments': {'path': '.'}}]) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(FakeBackend(loop), ContextManager('S'), Toolbox(tmp), None, max_steps=3, use_rag=False,
                          force_final=False)  # forced-final path: TestAgentFixes
            self.assertIn('limit', agent.ask('loop forever').answer)


class TestAgentFixes(unittest.TestCase):
    """ASSIST-06: tool rule only on tool turns; one forced no-tool answer when the step limit is hit."""

    def _box(self, d):
        (Path(d) / 'a.py').write_text('X = 1\n', encoding='utf-8')
        return Toolbox(d)

    def test_tool_rule_and_forced_final(self):
        from .agent import TOOL_RULE
        loop = Reply('', tool_calls=[{'name': 'read_file', 'arguments': {'path': 'a.py'}}], stats=GenStats())
        with tempfile.TemporaryDirectory() as d:
            fb = FakeBackend([loop, loop, Reply('X is 1', stats=GenStats())])
            turn = Agent(fb, ContextManager('sys', 4000), self._box(d), None, max_steps=2, use_rag=False).ask('X?')
            self.assertEqual(turn.answer, 'X is 1')
            self.assertTrue(turn.steps[-1].get('forced_final'))
            self.assertIsNone(fb.seen[-1][1])  # final call offers no tools
            self.assertIn(TOOL_RULE.strip(), fb.seen[0][0][0]['content'])
            fb2 = FakeBackend([loop, loop])
            old = Agent(fb2, ContextManager('sys', 4000), self._box(d), None, max_steps=2, use_rag=False,
                        tool_rule=False, force_final=False).ask('X?')
            self.assertEqual(old.answer, '(stopped: tool-step limit reached)')
            self.assertNotIn(TOOL_RULE.strip(), fb2.seen[0][0][0]['content'])
        # V1 audit #2: the forced call carries no TOOL_RULE, and tool calls inside it are recorded apart, never counted
        with tempfile.TemporaryDirectory() as d:
            sneaky = Reply('final', tool_calls=[{'name': 'read_file', 'arguments': {'path': 'a.py'}}], stats=GenStats())
            fb4 = FakeBackend([loop, sneaky])
            t4 = Agent(fb4, ContextManager('sys', 4000), self._box(d), None, max_steps=1, use_rag=False).ask('X?')
            self.assertNotIn(TOOL_RULE.strip(), fb4.seen[-1][0][0]['content'])
            self.assertEqual((t4.answer, t4.steps[-1]['tool_calls'], len(t4.steps[-1]['ignored_tool_calls'])), ('final', [], 1))
        fb3 = FakeBackend([Reply('no tools here', stats=GenStats())])
        Agent(fb3, ContextManager('sys', 4000), None, None, use_rag=False, use_tools=False).ask('hi')
        self.assertNotIn(TOOL_RULE.strip(), fb3.seen[0][0][0]['content'])


class TestV1AuditFixes(unittest.TestCase):
    def test_toolbox_never_crashes_on_bad_arguments(self):  # V1 audit #5
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'a.py').write_text('x = 1\n', encoding='utf-8')
            box = Toolbox(d)
            self.assertTrue(box.call('read_file', ['a.py']).startswith('ERROR'))
            self.assertTrue(box.call('read_file', {'path': 'a.py', 'start_line': 'abc'}).startswith('ERROR'))

    def test_grader_picks_the_defining_block(self):  # V1 audit #4: an echoed (longer) test file must not be graded
        from .bench.coding_agent import target_block
        from .bench.coding_tasks import TASKS
        t = next(x for x in TASKS if x['name'] == 'median')
        echo = 'import unittest\n\n\nclass TestVisible(unittest.TestCase):\n' + t['visible'] + t['hidden'] * 3
        answer = f"Fix:\n```python\n{t['fixed']}```\nTests:\n```python\n{echo}```"
        self.assertEqual(target_block(answer, t).strip(), t['fixed'].strip())

    def test_more_abstention_phrasings(self):  # V1 audit #3
        from .bench.agent_e2e import QUESTIONS, classify
        pattern = QUESTIONS[8][1]
        for a in ('I could not find MEM_FRACTION in the provided excerpts.', 'I have no information about MEM_FRACTION.',
                  'MEM_FRACTION is not in the provided context.', 'The excerpts do not show it, so I cannot say.'):
            self.assertEqual(classify(a, pattern), 'abstained', a)


class TestGraphBuckets(unittest.TestCase):
    def test_pick_bucket(self):  # V3: request-sized static caches (a fixed 4096 cache made decode about 2x slower)
        from .backends import HFBackend, pick_bucket
        b = HFBackend.GRAPH_BUCKETS
        self.assertEqual(pick_bucket(317, b, 4096), 384)
        self.assertEqual(pick_bucket(384, b, 4096), 384)
        self.assertEqual(pick_bucket(385, b, 4096), 512)
        self.assertEqual(pick_bucket(1, b, 4096), 256)
        self.assertEqual(pick_bucket(4096, b, 4096), 4096)
        self.assertIsNone(pick_bucket(4097, b, 4096))
        self.assertEqual(pick_bucket(600, b, 1000), 1000)  # limit below the next bucket is still usable


class TestGraphChatPath(unittest.TestCase):
    """V3 audit: HFBackend._chat_graph must stop on every generation_config EOS id and stream complete characters."""

    def test_eos_set_and_incremental_streaming(self):
        import types
        try:
            import torch
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
                                                local_files_only=True)
        except Exception as e:
            self.skipTest(f'tokenizer unavailable: {e}')
        from .backends import HFBackend
        text = 'GPU 🧬 龘 done'
        ids = tok(text).input_ids + [151643, 11, 12]  # <|endoftext|> must stop generation (generate() stops there)

        class FakeDecoder:
            def generate(self, input_ids, n, eos_ids=(), on_token=None):
                out = []
                for t in ids[:n]:
                    out.append(t)
                    on_token(t)
                    if t in eos_ids:
                        break
                return out, {'ttft_s': 0.01, 'total_s': 0.1, 'tokens': len(out), 'capture_s': 0.0}
        b = HFBackend.__new__(HFBackend)
        b.tokenizer, b.torch, b.fast_cache_len = tok, torch, 4096
        b.model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[151645, 151643]))
        b._decoder = {256: FakeDecoder()}
        pieces = []
        reply = b._chat_graph(torch.zeros((1, 5), dtype=torch.long), 64, pieces.append)
        self.assertEqual(reply.text, text)
        self.assertEqual(''.join(pieces), text)
        self.assertFalse(any('�' in p for p in pieces), pieces)
        self.assertEqual(reply.stats.completion_tokens, len(tok(text).input_ids) + 1)  # stopped at <|endoftext|>


class TestEvalLeakage(unittest.TestCase):
    """Benchmark files hold the expected answers: the eval's retriever and tools must never surface them."""

    def test_excluded_files_are_invisible(self):
        from .bench.agent_e2e import EVAL_EXCLUDE
        root = Path(__file__).resolve().parents[1]
        r = Retriever(root, exclude=EVAL_EXCLUDE)
        self.assertFalse(any(c.path.startswith('assistant/bench/') or c.path == 'assistant/test_assistant.py' for c in r.chunks))
        box = Toolbox(root, retriever=r, exclude=EVAL_EXCLUDE)
        self.assertTrue(box.call('read_file', {'path': 'assistant/bench/agent_e2e.py'}).startswith('ERROR'))
        self.assertTrue(box.call('read_file', {'path': 'assistant/test_assistant.py'}).startswith('ERROR'))
        hits = box.call('search_code', {'pattern': 'kernelagent-ada/tasks-v1', 'max_results': 50})
        self.assertNotIn('assistant/bench', hits)
        self.assertNotIn('test_assistant', hits)
        self.assertIn('research/kernelagent/tasks.py', hits)  # the real source stays findable


class TestAgentBenchGrading(unittest.TestCase):
    def test_answer_patterns(self):
        import re
        from .bench.agent_e2e import QUESTIONS, norm
        good = ['It is 1e-2.', 'The level is **execute**.', 'eos_id defaults to 0', '`kernelagent-ada/tasks-v1`',
                'sample_from(probs, u)', 'triton-windows==3.7.1.post27', 'about 211 GB/s', 'SPLIT=64, BLOCK_N=32, num_warps=4']
        bad = ['0.1', 'read', '200', 'tasks-v2', 'sample', '3.6.0', '2110', '32, 64, 4']
        good += ['MEM_FRACTION = 0.45', 'TOL_FACTOR is 4.0', 'INNER = 20 calls', 'default budget 12', '48 programs',
                 'revision 70d244cc86cc', 'rank 16', 'num_ctx=8192', 'VOCAB = 4096', 'SAMPLE_TOL = 1e-5', '300 seconds',
                 'the single_pass family']
        bad += ['0.5', '8.0', '200', '120', '64', 'main', '8', '4096', '50257', '1e-6', '30 s', 'split_k']
        self.assertEqual(len(QUESTIONS), 20)
        for (q, pattern), g, b in zip(QUESTIONS, good, bad):
            self.assertRegex(norm(g), pattern, q)
            self.assertIsNone(re.search(pattern, norm(b)), q)

    def test_abstention_classes(self):
        from .bench.agent_e2e import QUESTIONS, classify
        pattern = QUESTIONS[8][1]
        self.assertEqual(classify('MEM_FRACTION = 0.45', pattern), 'correct')
        # real answers from the 2026-10-31 run (runs/20261031_v4/agent_v12) that the first regex mislabelled
        for a in ("I don't have access to research/kernelagent/evaluate.py.", 'That value is not shown in the context.',
                  'I cannot determine it without reading the file.', 'I would need to read evaluate.py to check.',
                  'I cannot directly read files from your local filesystem. To find the value, please run the command',
                  'I do not know the value of MEM_FRACTION. Please read the file to find it.'):
            self.assertEqual(classify(a, pattern), 'abstained', a)
        for a in ('It is 0.5.',
                  'The variable MEM_FRACTION is not defined in the file research/kernelagent/evaluate.py.',
                  'The report does not specify which template family won the decode-attention task.',
                  'MEM_FRACTION is a parameter commonly used in GPU code to control memory use.'):
            self.assertEqual(classify(a, pattern), 'unsupported', a)
        # independent review (KA-06): conflicting signals go to manual review, never silently correct/abstained
        rank, tol = QUESTIONS[14][1], QUESTIONS[9][1]
        cases = [('The rank is commonly 8 or 16.', rank, 'review'),
                 ('I cannot read it; see line 4.', tol, 'review'),
                 ('The provided context does not mention MEM_FRACTION; I would need to read evaluate.py.', pattern, 'review'),
                 ('Without reading the file I would guess it is 0.9.', pattern, 'review'),
                 ('MEM_FRACTION is typically 0.9.', pattern, 'unsupported')]
        for a, pat, want in cases:
            self.assertEqual(classify(a, pat), want, a)
        for a in ('TOL_FACTOR is typically 4.5', 'see line 34', '4096 tokens', 'x4'):
            self.assertNotEqual(classify(a, tol), 'correct', a)  # anchored numbers


class TestCodingTasks(unittest.TestCase):
    """assistant-coding-v1 is only meaningful if every bug is real and every reference fix passes the hidden tests."""

    def test_tasks_are_valid(self):
        from .bench.coding_agent import grade, run_tests, write_workspace
        from .bench.coding_tasks import TASKS
        self.assertEqual(len({t['name'] for t in TASKS}), len(TASKS))
        for t in TASKS:
            with tempfile.TemporaryDirectory() as d:
                write_workspace(d, t, t['buggy'])
                self.assertFalse(run_tests(d, t['name'])[0], f"{t['name']}: visible tests must fail on the buggy code")
                self.assertFalse((Path(d) / 'research' / 'tasklib' / 'TestHidden').exists())
                self.assertNotIn('TestHidden', (Path(d) / 'research' / 'tasklib' / f"test_{t['name']}.py").read_text())
            self.assertEqual(grade(t, t['fixed']), (True, 'ok'), t['name'])
            self.assertFalse(grade(t, t['buggy'])[0], t['name'])
            hidden_only = dict(t, visible='    def test_placeholder(self):\n        pass\n')
            self.assertFalse(grade(hidden_only, t['buggy'])[0], f"{t['name']}: hidden tests alone must catch the bug")
            self.assertFalse(grade(t, 'import os\n' + t['fixed'])[0], t['name'])


if __name__ == '__main__':
    unittest.main()
