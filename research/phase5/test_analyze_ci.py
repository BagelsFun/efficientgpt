"""CPU tests for the bootstrap ratio analysis on synthetic timings."""
import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from .analyze_ci import analyze, classify, load_raw, ratio_ci


def fake(rng, scale, rounds=5, n=15, drift=0.0):
    """{round: [ms]} with per-round multiplicative drift and request noise."""
    return {r: list(scale * (1 + drift * rng.standard_normal()) * (1 + 0.03 * rng.standard_normal(n)))
            for r in range(1, rounds + 1)}


class TestRatioCI(unittest.TestCase):
    def test_clear_speedup_is_win(self):
        rng = np.random.default_rng(1)
        r = ratio_ci(fake(rng, 13.0), fake(rng, 10.0), boots=2000)
        self.assertAlmostEqual(r['ratio'], 1.3, delta=0.05)
        self.assertGreater(r['ci_low'], 1.0)
        self.assertEqual(classify(r), 'WIN')

    def test_identical_distributions_inconclusive(self):
        rng = np.random.default_rng(2)
        r = ratio_ci(fake(rng, 10.0), fake(rng, 10.0), boots=2000)
        self.assertLessEqual(r['ci_low'], 1.0)
        self.assertGreaterEqual(r['ci_high'], 1.0)
        self.assertEqual(classify(r), 'INCONCLUSIVE')

    def test_round_drift_widens_interval(self):
        quiet = ratio_ci(fake(np.random.default_rng(3), 11.0), fake(np.random.default_rng(4), 10.0), boots=2000)
        noisy = ratio_ci(fake(np.random.default_rng(3), 11.0, drift=0.08),
                         fake(np.random.default_rng(4), 10.0, drift=0.08), boots=2000)
        self.assertGreater(noisy['ci_high'] - noisy['ci_low'], quiet['ci_high'] - quiet['ci_low'])

    def test_slower_and_small_gain(self):
        self.assertEqual(classify({'ratio': 0.9, 'ci_low': 0.85, 'ci_high': 0.95}), 'SLOWER')
        self.assertEqual(classify({'ratio': 1.03, 'ci_low': 1.01, 'ci_high': 1.05}), 'INCONCLUSIVE')

    def test_deterministic_for_seed(self):
        rng = np.random.default_rng(5)
        a, b = fake(rng, 12.0), fake(rng, 10.0)
        self.assertEqual(ratio_ci(a, b, boots=500, seed=7), ratio_ci(a, b, boots=500, seed=7))

    def test_unequal_round_sizes_rejected(self):
        with self.assertRaises(ValueError):
            ratio_ci({1: [1.0, 2.0], 2: [1.0]}, {1: [1.0, 2.0], 2: [1.0]})

    def test_raw_csv_round_trip_and_aa_label(self):
        rng = np.random.default_rng(6)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['round', 'mode', 'batch', 'prompt_tokens',
                                                  'generated_tokens', 'trial', 'variant', 'ms'])
                w.writeheader()
                for p, scales in [(8, (30.0, 12.0, 10.0)), (192, (60.0, 20.0, 20.0))]:
                    for mode, scale in zip(('prefix_sdpa_eager', 'full_graph', 'adaptive_graph'), scales):
                        for rnd, times in fake(rng, scale).items():
                            for i, ms in enumerate(times):
                                w.writerow(dict(round=rnd, mode=mode, batch=1, prompt_tokens=p,
                                                generated_tokens=16, trial=i // 3, variant=i % 3, ms=ms))
            results = analyze(load_raw(path), boots=1000)
        self.assertEqual(len(results), 6)
        aa = [r for r in results if r['aa_control']]
        self.assertEqual(len(aa), 1)
        self.assertEqual((aa[0]['prompt_tokens'], aa[0]['verdict']), (192, 'AA_PASS'))
        win = next(r for r in results if r['prompt_tokens'] == 8 and r['baseline'] == 'full_graph')
        self.assertEqual(win['verdict'], 'WIN')


if __name__ == '__main__':
    unittest.main()
