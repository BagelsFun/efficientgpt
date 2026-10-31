"""CPU tests for the in-process benchmark's ordering and comparison selection."""
import unittest
from collections import Counter
from .benchmark_inprocess import comparisons_for, rotation


class TestInProcessHelpers(unittest.TestCase):
    def test_rotation_balances_positions(self):
        modes = ['a', 'b', 'c']
        counts = Counter((m, pos) for step in range(6) for pos, m in enumerate(rotation(modes, step)))
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len(counts), 9)

    def test_rotation_two_modes_alternates(self):
        self.assertEqual([rotation(['a', 'b'], s) for s in range(3)], [['a', 'b'], ['b', 'a'], ['a', 'b']])

    def test_comparisons_use_contract_direction_and_cover_first_mode(self):
        pairs = comparisons_for(['full_graph', 'adaptive_graph', 'compiled_adaptive_graph'])
        self.assertIn(('full_graph', 'adaptive_graph'), pairs)
        self.assertIn(('adaptive_graph', 'compiled_adaptive_graph'), pairs)
        self.assertIn(('full_graph', 'compiled_adaptive_graph'), pairs)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertNotIn(('adaptive_graph', 'full_graph'), pairs)

    def test_unknown_pair_falls_back_to_first_mode_baseline(self):
        self.assertEqual(comparisons_for(['compiled_static', 'compiled_full_graph']),
                         [('compiled_static', 'compiled_full_graph')])


if __name__ == '__main__':
    unittest.main()
