"""CPU tests for the Phase C pipeline: teacher-output parsing, prompt construction, dedup helpers, loss masking."""
import unittest

from .distill_data import jaccard, parse_teacher, signature_of, to_example, tokens
from ..bench.grading import grade_code

SOLUTION = '''def clamp(x: int, lo: int, hi: int) -> int:
    """Clamp x into [lo, hi].

    Assumes lo <= hi.
    """
    return max(lo, min(hi, x))'''
REPLY = f'''Here is a task.
NAME: clamp
```python
{SOLUTION}
```
```python
assert clamp(5, 0, 3) == 3
assert clamp(-1, 0, 3) == 0
assert clamp(
    2, 0, 3) == 2
```'''


class TestDistill(unittest.TestCase):
    def test_parse_teacher(self):
        obj = parse_teacher(REPLY)
        self.assertEqual(obj['name'], 'clamp')
        self.assertEqual(len(obj['tests']), 3)  # multi-line assert kept whole
        self.assertIsNone(parse_teacher('NAME: clamp\nno code blocks here'))
        self.assertIsNone(parse_teacher(REPLY.replace('NAME: clamp', 'NAME: other')))  # name not defined in solution

    def test_prompt_never_contains_the_body(self):
        sig = signature_of(SOLUTION, 'clamp')
        self.assertIn('def clamp(x: int, lo: int, hi: int) -> int:', sig)
        self.assertIn('Clamp x into [lo, hi].', sig)
        self.assertIn('Assumes lo <= hi.', sig)
        self.assertNotIn('return', sig)
        compile(sig + '\n    pass\n', 'sig', 'exec')  # the prompt is a valid stub
        self.assertIsNone(signature_of('def f(x):\n    return x', 'f'))                 # no docstring
        self.assertIsNone(signature_of('def f(x):\n    """Doc only."""', 'f'))          # no body
        self.assertIsNone(signature_of('def f(:\n', 'f'))                              # syntax error

    def test_example_and_verification(self):
        obj = parse_teacher(REPLY)
        ex = to_example(obj)
        self.assertEqual([m['role'] for m in ex['messages']], ['user', 'assistant'])
        self.assertNotIn('return max', ex['messages'][0]['content'])
        self.assertTrue(grade_code(ex['messages'][1]['content'], 'clamp', obj['tests'])[0])
        self.assertFalse(grade_code(ex['messages'][1]['content'], 'clamp', ['assert clamp(5, 0, 3) == 5'])[0])

    def test_similarity(self):
        self.assertGreater(jaccard(tokens('def clamp(x, lo, hi)'), tokens('def clamp(x, lo, hi) docstring')), 0.5)
        self.assertLess(jaccard(tokens('def clamp(x, lo, hi)'), tokens('def roman_to_int(s)')), 0.2)


class TestResume(unittest.TestCase):
    def test_torn_last_line(self):  # KA-06 review: resume must survive a write killed mid-line
        import json
        import tempfile
        from pathlib import Path
        from .distill_data import load_kept
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'out.jsonl'
            p.write_text(json.dumps({'name': 'a', 'k': 3}) + '\n' + '{"name": "b", "mess', encoding='utf-8')
            self.assertEqual([r['name'] for r in load_kept(p)], ['a'])
            with p.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'name': 'c', 'k': 9}) + '\n')
            self.assertEqual([r['name'] for r in load_kept(p)], ['a', 'c'])  # 'c' not glued to the fragment


class TestDistillEval(unittest.TestCase):
    def test_load_split_and_mcnemar(self):
        import json
        import tempfile
        from pathlib import Path
        from ..bench.distill_eval import load_split, mcnemar
        with tempfile.TemporaryDirectory() as d:
            data, names = Path(d) / 'data.jsonl', Path(d) / 'names.json'
            data.write_text('\n'.join(json.dumps({'name': n, 'messages': [], 'tests': []}) for n in 'abc') + '\n', encoding='utf-8')
            names.write_text(json.dumps(['a', 'c']), encoding='utf-8')
            self.assertEqual([r['name'] for r in load_split(data, names)], ['a', 'c'])
            names.write_text(json.dumps(['a', 'zz']), encoding='utf-8')
            with self.assertRaises(SystemExit):
                load_split(data, names)
        self.assertEqual(mcnemar({'a': 0, 'b': 1}, {'a': 0, 'b': 1}), (0, 0, 1.0))
        w, l, p = mcnemar({str(i): False for i in range(10)}, {str(i): True for i in range(10)})
        self.assertEqual((w, l), (10, 0))
        self.assertAlmostEqual(p, 2 / 1024)


class TestLossMask(unittest.TestCase):
    def test_only_assistant_tokens_are_labelled(self):
        try:
            from transformers import AutoTokenizer
            from .train_qlora import MODEL, REVISION, encode
            tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        except Exception as e:  # tokenizer not downloaded on this machine
            self.skipTest(f'tokenizer unavailable: {e}')
        ids, labels = encode(tok, to_example(parse_teacher(REPLY))['messages'], 1024)
        self.assertEqual(len(ids), len(labels))
        target = tok.decode([t for t, l in zip(ids, labels) if l != -100])
        self.assertIn('return max(lo, min(hi, x))', target)
        self.assertNotIn('Here is a task', target)
        self.assertTrue(labels[0] == -100 and labels[-1] != -100)


if __name__ == '__main__':
    unittest.main()
