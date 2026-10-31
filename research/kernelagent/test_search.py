"""CPU tests for the search controller with a synthetic evaluator (no GPU, deterministic)."""
import math
import unittest

from . import templates
from .search import Search, key, meta_order, neighbours, points

TASK = 'temperature_softmax'


def fake(bad_family=None, calls=None):
    """ms = 1 + distance (in grid steps) from a hidden optimum; one family can be made always-incorrect."""
    def fn(task, fam, params):
        if calls is not None:
            calls.append(key(fam, params))
        if fam == bad_family:
            return {'status': 'incorrect'}
        space = templates.FAMILIES[task][fam]['space']
        d = sum(abs(space[k].index(v) - (len(space[k]) - 1)) for k, v in params.items())
        off = sorted(templates.FAMILIES[task]).index(fam) * 0.5
        ms = 1.0 + d + off
        return {'status': 'correct', 'performance': {'1x4096': {'median_ms': {'candidate': ms}}},
                'launch_metadata': [{'n_spills': 0}]}
    return fn


class TestSearch(unittest.TestCase):
    def test_budget_and_no_duplicates(self):
        for strat in ('single_shot', 'random', 'local', 'local_meta'):
            s = Search(TASK, fake(), budget=7, seed=1)
            out = s.run(strat)
            self.assertLessEqual(out['evaluated'], 7, strat)
            keys = [t['key'] for t in out['trace']]
            self.assertEqual(len(keys), len(set(keys)), strat)

    def test_single_shot_is_family_defaults(self):
        out = Search(TASK, fake(), budget=100).run('single_shot')
        self.assertEqual(out['evaluated'], len(templates.FAMILIES[TASK]))

    def test_local_beats_single_shot_on_smooth_landscape(self):
        base = Search(TASK, fake(), budget=100).run('single_shot')['best']['score_ms']
        local = Search(TASK, fake(), budget=20, seed=0).run('local')['best']['score_ms']
        self.assertLess(local, base)
        self.assertEqual(Search(TASK, fake(), budget=len(points(TASK))).run('local')['best']['score_ms'], 1.0)

    def test_incorrect_never_incumbent(self):
        bad = sorted(templates.FAMILIES[TASK])[0]  # the family with the best synthetic times
        out = Search(TASK, fake(bad_family=bad), budget=30, seed=0).run('random')
        self.assertNotEqual(out['best']['family'], bad)
        self.assertGreater(out['statuses'].get('incorrect', 0), 0)

    def test_history_reuse_costs_no_evaluation(self):
        fam, spec = next(iter(templates.FAMILIES[TASK].items()))
        cache = {key(fam, spec['default']): fake()(TASK, fam, spec['default'])}
        calls = []
        out = Search(TASK, fake(calls=calls), budget=100, cache=cache).run('single_shot')
        self.assertNotIn(key(fam, spec['default']), calls)
        self.assertTrue(out['trace'][0]['from_history'])

    def test_neighbours_one_step(self):
        fam, spec = next(iter(templates.FAMILIES[TASK].items()))
        for k, direction, q in neighbours(TASK, fam, spec['default']):
            diff = [p for p in q if q[p] != spec['default'][p]]
            self.assertEqual(diff, [k])
            self.assertEqual(abs(direction), 1)

    def test_meta_order(self):
        spill = {'launch_metadata': [{'n_spills': 3}]}
        self.assertLess(meta_order(spill, ('WARPS', -1, {})), meta_order(spill, ('WARPS', 1, {})))
        low = {'launch_metadata': [{}], 'performance': {'s': {'roofline_fraction': 0.1}}}
        self.assertLess(meta_order(low, ('WARPS', 1, {})), meta_order(low, ('WARPS', -1, {})))

    def test_local_spends_full_budget(self):  # regression: local stopped early when no default was correct
        out = Search(TASK, lambda *a: {'status': 'incorrect'}, budget=9, seed=0).run('local')
        self.assertEqual((out['evaluated'], out['unused_budget']), (9, 0))

    def test_load_cache_rules(self):  # regression: held-out, transient, stale and contended rows were reused
        import json, tempfile
        from pathlib import Path
        from .evaluate import PROTOCOL, candidate_id
        from .search import load_cache
        fam, spec = next(iter(templates.FAMILIES[TASK].items()))
        p = spec['default']
        good = {'task': TASK, 'family': fam, 'params': p, 'split': 'dev', 'protocol': PROTOCOL, 'status': 'correct',
                'id': candidate_id(TASK, templates.render(TASK, fam, p)), 'trials': 8, 'contended': False,
                'performance': {'s': {'median_ms': {'candidate': 1.0}}}}
        bad = [dict(good, split='heldout'), dict(good, status='crash'), dict(good, status='timeout'),
               dict(good, id='stale'), dict(good, contended=True), dict(good, trials=2), dict(good, protocol='old')]
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / 'h.jsonl'
            h.write_text(''.join(json.dumps(r) + '\n' for r in bad), encoding='utf-8')
            self.assertEqual(load_cache(h, TASK, 8), {})
            h.write_text(json.dumps(good) + '\n', encoding='utf-8')
            self.assertEqual(list(load_cache(h, TASK, 8)), [key(fam, p)])

    def test_score_inf_for_failures(self):
        from .search import score
        self.assertEqual(score({'status': 'timeout'}), math.inf)
        self.assertEqual(score({'status': 'correct'}), math.inf)  # no timings (e.g. --no-timing)


if __name__ == '__main__':
    unittest.main()
