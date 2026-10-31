"""Mutation check: each ragged bug must make test_ragged fail."""
import sys, unittest, inspect, textwrap
sys.path.insert(0, '.')
import research.phase5.ragged as R
MUTANTS = {
    'prefill_reads_last_padded_slot': ('ragged_prefill', 'x.gather(1, (lengths - 1).view(b, 1, 1)', 'x.gather(1, (lengths * 0 + p - 1).view(b, 1, 1)'),
    'finish_ignores_lengths': ('RaggedEngine.finish', 'idx = self.lengths.view(-1, 1)', 'idx = self.lengths.max().view(-1, 1)'),
    'shared_position_embedding': ('RaggedEngine.step', 'tr.wpe(self.position).unsqueeze(1)', 'tr.wpe(self.position[:1]).unsqueeze(1)'),
    'mask_uses_max_position': ('RaggedEngine.step', 'allowed = self.locations <= self.position.view(b, 1, 1, 1)', 'allowed = self.locations <= self.position.max().view(1, 1, 1, 1)'),
}
src0 = open(R.__file__).read()
for name, (where, old, new) in MUTANTS.items():
    assert old in src0, name
    exec(compile(src0.replace(old, new), R.__file__, 'exec'), R.__dict__)
    import research.phase5.test_ragged as T
    T.RaggedEngine, T.ragged_prefill = R.RaggedEngine, R.ragged_prefill
    res = unittest.TextTestRunner(stream=open('nul' if sys.platform == 'win32' else '/dev/null', 'w')).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(T.TestRaggedCPU))
    failed = [t.id().split('.')[-1] for t, _ in res.failures + res.errors]
    print(f'{name}: {"KILLED" if failed else "SURVIVED"} by {failed}')
    exec(compile(src0, R.__file__, 'exec'), R.__dict__)
