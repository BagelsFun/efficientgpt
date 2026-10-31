"""The quality suite's own correctness: reference solutions must pass every grader; wrong ones must fail."""
import unittest

from . import quality_v1
from .grading import grade, grade_code, grade_math, grade_mcq

REFERENCE = {
    'is_palindrome': '''def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]''',
    'merge_intervals': '''def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out''',
    'rle_encode': '''def rle_encode(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)''',
    'top_k_frequent': '''def top_k_frequent(nums, k):
    from collections import Counter
    c = Counter(nums)
    return sorted(c, key=lambda v: (-c[v], v))[:k]''',
    'matmul': '''def matmul(a, b):
    if not a or len(a[0]) != len(b):
        raise ValueError("shape mismatch")
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]''',
    'softmax': '''def softmax(xs):
    import math
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]''',
    'chunked': '''def chunked(seq, n):
    if n <= 0:
        raise ValueError("n must be positive")
    return [seq[i:i + n] for i in range(0, len(seq), n)]''',
    'flatten': '''def flatten(nested):
    out = []
    for x in nested:
        out.extend(flatten(x) if isinstance(x, list) else [x])
    return out''',
    'roman_to_int': '''def roman_to_int(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        total += -v[ch] if i + 1 < len(s) and v[s[i + 1]] > v[ch] else v[ch]
    return total''',
    'binary_search': '''def binary_search(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(arr) and arr[lo] == target else -1''',
    'count_params': '''def count_params(shapes):
    total = 0
    for s in shapes:
        n = 1
        for d in s:
            n *= d
        total += n
    return total''',
    'conv_out_size': '''def conv_out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1''',
    'moving_average': '''def moving_average(xs, w):
    if w > len(xs):
        return []
    return [sum(xs[i:i + w]) / w for i in range(len(xs) - w + 1)]''',
    'kv_cache_bytes': '''def kv_cache_bytes(layers, heads, head_dim, seq_len, batch, bytes_per_elem=2):
    return 2 * layers * heads * head_dim * seq_len * batch * bytes_per_elem''',
    'lru_misses': '''def lru_misses(capacity, accesses):
    from collections import OrderedDict
    cache, misses = OrderedDict(), 0
    for a in accesses:
        if a in cache:
            cache.move_to_end(a)
        else:
            misses += 1
            cache[a] = True
            if len(cache) > capacity:
                cache.popitem(last=False)
    return misses''',
    'parse_duration': '''def parse_duration(s):
    import re
    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", s)
    if not s or not m:
        raise ValueError("bad duration")
    h, mi, se = (int(g) if g else 0 for g in m.groups())
    return 3600 * h + 60 * mi + se''',
}


class TestSuite(unittest.TestCase):
    def test_counts_and_splits(self):
        items = quality_v1.items()
        self.assertEqual(len(items), 52)
        self.assertEqual(len({i['id'] for i in items}), 52)
        for cat in ('code', 'math', 'mcq'):
            splits = [i['split'] for i in items if i['category'] == cat]
            self.assertEqual(splits.count('dev'), splits.count('test'), cat)

    def test_reference_solutions_pass(self):
        for item in quality_v1.items():
            if item['category'] == 'code':
                ok, why = grade(item, f"Here you go:\n```python\n{REFERENCE[item['entry']]}\n```")
            elif item['category'] == 'math':
                ok, why = grade(item, f"Some working with 3 and 7.\nAnswer: {item['answer']:,}")
            else:
                ok, why = grade(item, f"{item['answer']}")
            self.assertTrue(ok, (item['id'], why))

    def test_wrong_answers_fail(self):
        item = next(i for i in quality_v1.items() if i['id'] == 'code/binary_search')
        wrong = 'def binary_search(arr, target):\n    return arr.index(target) if target in arr else 0'
        self.assertFalse(grade_code(f'```python\n{wrong}\n```', item['entry'], item['tests'])[0])
        self.assertFalse(grade_code('```python\nimport os\ndef binary_search(a, t):\n    return -1\n```', 'binary_search', ['pass'])[0])
        self.assertFalse(grade_code('```python\ndef binary_search(a, t):\n    while True: pass\n```', 'binary_search',
                                    ['binary_search([1], 1)'], timeout=3)[0])
        self.assertFalse(grade_math('Answer: 41', 42)[0])
        self.assertTrue(grade_math('the result is 3 then 42', 42)[0])
        self.assertFalse(grade_mcq('C', 'B')[0])
        self.assertTrue(grade_mcq('<think>maybe A</think>The answer is B.', 'B')[0])


if __name__ == '__main__':
    unittest.main()
