"""GPU tests: the evaluator must reject mutated, unsafe, crashing and hanging candidates (mutation testing)."""
import json
import unittest
from pathlib import Path

import torch

from . import templates
from .evaluate import candidate_id, evaluate, seen_ids


def mutate(task, fam, old, new):
    src = templates.render(task, fam, templates.FAMILIES[task][fam]['default'])
    assert old in src, (task, fam, old)
    return src.replace(old, new, 1)


@unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
class TestEvaluator(unittest.TestCase):
    def status(self, task, src, **kw):
        return evaluate(task, src, timing=False, **kw)['status']

    def test_default_is_correct(self):
        src = templates.render('argmax', 'row_whole', {'WARPS': 4})
        self.assertEqual(self.status('argmax', src), 'correct')

    def test_index_mutation_rejected(self):  # argmin instead of argmax
        self.assertEqual(self.status('argmax', mutate('argmax', 'row_whole', 'tl.argmax(x, 0)', 'tl.argmin(x, 0)')),
                         'incorrect')

    def test_tie_break_mutation_rejected(self):  # >= keeps the LAST chunk's index on ties (dupmax edge case)
        src = mutate('argmax', 'row_chunked', 'take = m > best', 'take = m >= best')
        self.assertEqual(self.status('argmax', src), 'incorrect')

    def test_small_float_error_rejected(self):  # 0.1% scale error on softmax is far outside 4x eager fp32 error
        src = templates.render('temperature_softmax', 'row_whole', templates.FAMILIES['temperature_softmax']['row_whole']['default'])
        src = src.replace('def run(inp):', 'def run(inp):\n    return _run(inp) * 1.001\n\n\ndef _run(inp):', 1)
        self.assertEqual(self.status('temperature_softmax', src), 'incorrect')

    def test_sample_offbyone_rejected(self):
        src = templates.render('inverse_cdf_sample', 'scan_whole_row', templates.FAMILIES['inverse_cdf_sample']['scan_whole_row']['default'])
        src = src.replace('def run(inp):', 'def run(inp):\n    return (_run(inp) + 1).clamp(max=4095)\n\n\ndef _run(inp):', 1)
        self.assertEqual(self.status('inverse_cdf_sample', src), 'incorrect')

    def test_attention_mask_mutation_rejected(self):  # reads one KV slot past the position
        src = templates.render('decode_attention', 'single_pass', templates.FAMILIES['decode_attention']['single_pass']['default'])
        src = src.replace('def run(inp):', "def run(inp):\n    inp = dict(inp, position=(inp['position'] + 1).clamp(max=inp['k'].size(2) - 1))\n    return _run(inp)\n\n\ndef _run(inp):", 1)
        self.assertEqual(self.status('decode_attention', src), 'incorrect')

    def test_screen_rejects_host_access(self):
        src = 'import os\n' + templates.render('argmax', 'row_whole', {'WARPS': 4})
        res = evaluate('argmax', src, timing=False)
        self.assertEqual(res['status'], 'rejected_screen')
        self.assertIn('import os', res['problems'])

    def test_reference_smuggling_rejected(self):
        src = templates.render('argmax', 'row_whole', {'WARPS': 4}).replace(
            'def run(inp):', "def run(inp):\n    return inp['logits'].argmax(-1, keepdim=True)\n\n\ndef _run(inp):", 1)
        res = evaluate('argmax', src, timing=False)
        self.assertEqual(res['status'], 'rejected_screen')  # regression: tensor-method form used to pass the screen
        self.assertIn('tensor attribute .argmax', res['problems'])

    def test_screen_bypasses_from_review(self):  # regression: each of these passed the screen before KA-01 review
        from .screen import screen
        base = templates.render('temperature_softmax', 'row_whole', {'WARPS': 4})
        smuggle = 'def run(inp):\n    return {}\n\n\ndef _run(inp):'
        for header, body, expect in [
                ('import torch as tl\n', "tl.softmax(inp['logits'] / inp['temperature'], -1)", 'import alias torch as tl'),
                ('import torch as tt\n', "tt.softmax(inp['logits'], -1)", 'import alias torch as tt'),
                ('', "(inp['logits'] / inp['temperature']).exp2()", 'tensor attribute .exp2'),
                ('', "inp['logits'].aminmax(dim=-1)[1]", 'tensor attribute .aminmax'),
                ('', "inp['logits'].nansum()", 'tensor attribute .nansum')]:
            src = header + base.replace('def run(inp):', smuggle.format(body), 1)
            self.assertIn(expect, screen(src), body)
        self.assertIn('rebinds tl', screen(base + '\ntl = torch\n'))
        self.assertIn('rebinds torch', screen(base + '\ndef f(torch):\n    return 0\n'))

    def test_input_mutation_rejected(self):  # regression: kernel overwrites its input, reference then saw forged inputs
        base = templates.render('temperature_softmax', 'row_whole', {'WARPS': 4})
        launch = '    _k[(b,)](out, x, n, x.stride(0), 1.0, BLOCK=triton.next_power_of_2(n), num_warps=4)\n'
        src = base.replace('    return out', launch + '    return out', 1)
        res = evaluate('temperature_softmax', src, timing=False)
        self.assertEqual(res['status'], 'incorrect')
        self.assertTrue(all('inputs modified' in r.get('error', '') for r in res['correctness']))

    def test_int8_gemv_mutations_rejected(self):  # R1 real-model task: the contract must catch dequantisation bugs
        good = templates.render('int8_gemv', 'dot16', templates.FAMILIES['int8_gemv']['dot16']['default'])
        self.assertEqual(self.status('int8_gemv', good), 'correct')
        no_scale = good.replace('out = acc * s[None, :]', 'out = acc')
        skip_last_k = good.replace('for k0 in range(0, K, BLOCK_K):', 'for k0 in range(0, K - BLOCK_K, BLOCK_K):')
        unsigned = good.replace('acc += tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16))',
                                'acc += tl.dot(x.to(tl.bfloat16), w.to(tl.uint8).to(tl.bfloat16))')
        for name, src in (('no_scale', no_scale), ('skip_last_k', skip_last_k), ('unsigned', unsigned)):
            self.assertNotEqual(src, good, name)
            self.assertEqual(self.status('int8_gemv', src), 'incorrect', name)

    def test_import_error_and_timeout(self):
        base = templates.render('argmax', 'row_whole', {'WARPS': 4})
        self.assertEqual(self.status('argmax', base + '\nraise ValueError("boom")\n'), 'import_error')
        self.assertEqual(self.status('argmax', base + '\nwhile True:\n    pass\n', timeout=20), 'timeout')

    def test_history_dedup(self):
        p = Path('results/kernelagent/_test_history.jsonl')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'id': candidate_id('argmax', 'x')}) + '\n', encoding='utf-8')
        try:
            self.assertEqual(seen_ids(p), {candidate_id('argmax', 'x')})
            self.assertNotEqual(candidate_id('argmax', 'x'), candidate_id('softmax', 'x'))
        finally:
            p.unlink()


if __name__ == '__main__':
    unittest.main()
