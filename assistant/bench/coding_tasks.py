"""Bug-fix tasks for the coding-agent evaluation (assistant-coding-v1). Self-written, pure Python, no GPU.

Each task: a buggy module, its reference fix (used only to validate the task), visible tests (shown to / runnable by the
model; they fail on the buggy code) and hidden tests (edge cases, run only by the grader, so hard-coding the visible
cases fails). The module lives at research/tasklib/<name>.py inside a throwaway workspace.
"""

TASKS = []


def task(name, buggy, fixed, visible, hidden):
    TASKS.append({'name': name, 'buggy': buggy.strip() + '\n', 'fixed': fixed.strip() + '\n',
                  'visible': visible.strip('\n') + '\n', 'hidden': hidden.strip('\n') + '\n'})  # keep method indent


task('moving_average', '''
def moving_average(xs, window):
    """Means of every full window of `window` consecutive values (len(xs) - window + 1 results)."""
    if window <= 0:
        raise ValueError('window must be positive')
    return [sum(xs[i:i + window]) / window for i in range(len(xs) - window)]
''', '''
def moving_average(xs, window):
    """Means of every full window of `window` consecutive values (len(xs) - window + 1 results)."""
    if window <= 0:
        raise ValueError('window must be positive')
    return [sum(xs[i:i + window]) / window for i in range(len(xs) - window + 1)]
''', '''
    def test_basic(self):
        self.assertEqual(moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])
''', '''
    def test_edges(self):
        self.assertEqual(moving_average([5], 1), [5.0])
        self.assertEqual(moving_average([1, 2], 3), [])
        self.assertEqual(moving_average([2, 4, 6], 3), [4.0])
        with self.assertRaises(ValueError):
            moving_average([1], 0)
''')

task('median', '''
def median(xs):
    """Median of a non-empty list of numbers (mean of the two middle values for even length)."""
    if not xs:
        raise ValueError('empty')
    s = sorted(xs)
    return s[len(s) // 2]
''', '''
def median(xs):
    """Median of a non-empty list of numbers (mean of the two middle values for even length)."""
    if not xs:
        raise ValueError('empty')
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
''', '''
    def test_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)
''', '''
    def test_more(self):
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([7]), 7)
        self.assertEqual(median([1, 1, 2, 10]), 1.5)
        with self.assertRaises(ValueError):
            median([])
''')

task('capacity_bucket', '''
def pick_capacity(needed, buckets=(80, 128, 192, 256)):
    """Smallest bucket that can hold `needed` tokens; ValueError if none can."""
    for b in sorted(buckets):
        if b > needed:
            return b
    raise ValueError(f'{needed} tokens exceed the largest bucket')
''', '''
def pick_capacity(needed, buckets=(80, 128, 192, 256)):
    """Smallest bucket that can hold `needed` tokens; ValueError if none can."""
    for b in sorted(buckets):
        if b >= needed:
            return b
    raise ValueError(f'{needed} tokens exceed the largest bucket')
''', '''
    def test_exact_fit(self):
        self.assertEqual(pick_capacity(128), 128)
''', '''
    def test_more(self):
        self.assertEqual(pick_capacity(1), 80)
        self.assertEqual(pick_capacity(129), 192)
        self.assertEqual(pick_capacity(256), 256)
        self.assertEqual(pick_capacity(5, buckets=(16, 4, 8)), 8)
        with self.assertRaises(ValueError):
            pick_capacity(257)
''')

task('top_k_indices', '''
def top_k_indices(values, k):
    """Indices of the k largest values, largest first; ties go to the lower index."""
    order = sorted(range(len(values)), key=lambda i: values[i])[::-1]
    return order[:k]
''', '''
def top_k_indices(values, k):
    """Indices of the k largest values, largest first; ties go to the lower index."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]
''', '''
    def test_ties(self):
        self.assertEqual(top_k_indices([1, 3, 3, 2], 2), [1, 2])
        self.assertEqual(top_k_indices([5, 5, 5], 2), [0, 1])
''', '''
    def test_more(self):
        self.assertEqual(top_k_indices([0.1, 0.9, 0.5], 1), [1])
        self.assertEqual(top_k_indices([2, 1], 5), [0, 1])
        self.assertEqual(top_k_indices([], 3), [])
        self.assertEqual(top_k_indices([-1, -1, 0, -1], 3), [2, 0, 1])
''')

task('ring_buffer', '''
class RingBuffer:
    """Fixed-capacity buffer keeping the most recent items; items() returns them oldest first."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity, self.data, self.start = capacity, [], 0

    def push(self, x):
        if len(self.data) < self.capacity:
            self.data.append(x)
        else:
            self.data[self.start] = x
            self.start = self.start + 1

    def items(self):
        return self.data[self.start:] + self.data[:self.start]
''', '''
class RingBuffer:
    """Fixed-capacity buffer keeping the most recent items; items() returns them oldest first."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity, self.data, self.start = capacity, [], 0

    def push(self, x):
        if len(self.data) < self.capacity:
            self.data.append(x)
        else:
            self.data[self.start] = x
            self.start = (self.start + 1) % self.capacity

    def items(self):
        return self.data[self.start:] + self.data[:self.start]
''', '''
    def test_wraps_twice(self):
        r = RingBuffer(2)
        for x in range(5):
            r.push(x)
        self.assertEqual(r.items(), [3, 4])
''', '''
    def test_more(self):
        r = RingBuffer(3)
        for x in range(10):
            r.push(x)
            self.assertEqual(r.items(), list(range(max(0, x - 2), x + 1)))
        one = RingBuffer(1)
        for x in 'abc':
            one.push(x)
        self.assertEqual(one.items(), ['c'])
        with self.assertRaises(ValueError):
            RingBuffer(0)
''')

task('lru_cache', '''
from collections import OrderedDict


class LRUCache:
    """get/put cache that evicts the least recently used key when full; get() counts as a use."""

    def __init__(self, capacity):
        self.capacity, self.d = capacity, OrderedDict()

    def get(self, key, default=None):
        if key not in self.d:
            return default
        return self.d[key]

    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)
''', '''
from collections import OrderedDict


class LRUCache:
    """get/put cache that evicts the least recently used key when full; get() counts as a use."""

    def __init__(self, capacity):
        self.capacity, self.d = capacity, OrderedDict()

    def get(self, key, default=None):
        if key not in self.d:
            return default
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)
''', '''
    def test_get_refreshes(self):
        c = LRUCache(2)
        c.put('a', 1)
        c.put('b', 2)
        c.get('a')
        c.put('c', 3)
        self.assertEqual(c.get('a'), 1)
        self.assertIsNone(c.get('b'))
''', '''
    def test_more(self):
        c = LRUCache(2)
        c.put('a', 1)
        c.put('b', 2)
        c.put('a', 10)  # update counts as a use
        c.put('c', 3)
        self.assertEqual((c.get('a'), c.get('b'), c.get('c')), (10, None, 3))
        self.assertEqual(c.get('zz', 'dflt'), 'dflt')
        c.get('missing')  # a miss must not break ordering
        c.put('d', 4)
        self.assertIsNone(c.get('a'))

    def test_get_refreshes_other_keys(self):  # hidden: get() counts as a use (fails on the buggy code; review V1)
        c = LRUCache(3)
        for k in 'xyz':
            c.put(k, k.upper())
        self.assertEqual(c.get('x'), 'X')
        c.put('w', 'W')  # evicts y (least recently used), not x
        self.assertEqual((c.get('x'), c.get('y')), ('X', None))
''')

task('parse_config', '''
def parse_config(text):
    """Parse `key = value` lines into a dict of strings. Blank lines and lines starting with # are ignored;
    whitespace around keys and values is stripped; a later key overrides an earlier one."""
    out = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        out[key.strip()] = value.strip()
    return out
''', '''
def parse_config(text):
    """Parse `key = value` lines into a dict of strings. Blank lines and lines starting with # are ignored;
    whitespace around keys and values is stripped; a later key overrides an earlier one."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        key, value = s.split('=', 1)
        out[key.strip()] = value.strip()
    return out
''', '''
    def test_blank_and_indented_comment(self):
        self.assertEqual(parse_config('a = 1\\n\\n   # note\\nb=2'), {'a': '1', 'b': '2'})
''', '''
    def test_more(self):
        self.assertEqual(parse_config(''), {})
        self.assertEqual(parse_config('x = a=b\\nx = c'), {'x': 'c'})
        self.assertEqual(parse_config('  k  =  v  \\n\\t\\n#c'), {'k': 'v'})
        self.assertEqual(parse_config('url = http://h/p?q=1'), {'url': 'http://h/p?q=1'})
''')

task('human_bytes', '''
def human_bytes(n):
    """Format a byte count with binary units: 1023 -> '1023 B', 1024 -> '1.0 KiB', 1536 -> '1.5 KiB', up to TiB."""
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    value, i = float(n), 0
    while value > 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f'{int(value)} B' if i == 0 else f'{value:.1f} {units[i]}'
''', '''
def human_bytes(n):
    """Format a byte count with binary units: 1023 -> '1023 B', 1024 -> '1.0 KiB', 1536 -> '1.5 KiB', up to TiB."""
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    value, i = float(n), 0
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f'{int(value)} B' if i == 0 else f'{value:.1f} {units[i]}'
''', '''
    def test_boundary(self):
        self.assertEqual(human_bytes(1024), '1.0 KiB')
''', '''
    def test_more(self):
        self.assertEqual(human_bytes(0), '0 B')
        self.assertEqual(human_bytes(1023), '1023 B')
        self.assertEqual(human_bytes(1536), '1.5 KiB')
        self.assertEqual(human_bytes(1024 ** 2), '1.0 MiB')
        self.assertEqual(human_bytes(8 * 1024 ** 3), '8.0 GiB')
        self.assertEqual(human_bytes(1024 ** 5), '1024.0 TiB')
''')

task('chunk_windows', '''
def chunk_windows(n_tokens, size, overlap):
    """(start, end) token windows of length <= size covering 0..n_tokens, consecutive windows sharing `overlap` tokens;
    the last window ends at n_tokens. Requires 0 <= overlap < size."""
    if not 0 <= overlap < size:
        raise ValueError('need 0 <= overlap < size')
    out, start = [], 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        out.append((start, end))
        start += size
    return out
''', '''
def chunk_windows(n_tokens, size, overlap):
    """(start, end) token windows of length <= size covering 0..n_tokens, consecutive windows sharing `overlap` tokens;
    the last window ends at n_tokens. Requires 0 <= overlap < size."""
    if not 0 <= overlap < size:
        raise ValueError('need 0 <= overlap < size')
    out, start = [], 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            break
        start += size - overlap
    return out
''', '''
    def test_overlap(self):
        self.assertEqual(chunk_windows(10, 4, 2), [(0, 4), (2, 6), (4, 8), (6, 10)])
''', '''
    def test_more(self):
        self.assertEqual(chunk_windows(0, 4, 1), [])
        self.assertEqual(chunk_windows(3, 4, 1), [(0, 3)])
        self.assertEqual(chunk_windows(8, 4, 0), [(0, 4), (4, 8)])
        self.assertEqual(chunk_windows(9, 4, 1), [(0, 4), (3, 7), (6, 9)])
        with self.assertRaises(ValueError):
            chunk_windows(5, 3, 3)
''')

task('percentile', '''
def percentile(xs, p):
    """Linear-interpolation percentile (numpy's default 'linear' method) of a non-empty list, p in [0, 100]."""
    if not xs or not 0 <= p <= 100:
        raise ValueError('need non-empty xs and 0 <= p <= 100')
    s = sorted(xs)
    pos = p / 100 * len(s)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
''', '''
def percentile(xs, p):
    """Linear-interpolation percentile (numpy's default 'linear' method) of a non-empty list, p in [0, 100]."""
    if not xs or not 0 <= p <= 100:
        raise ValueError('need non-empty xs and 0 <= p <= 100')
    s = sorted(xs)
    pos = p / 100 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
''', '''
    def test_median_and_max(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4], 100), 4)
''', '''
    def test_more(self):
        self.assertEqual(percentile([7], 95), 7)
        self.assertEqual(percentile([3, 1, 2], 0), 1)
        self.assertAlmostEqual(percentile([10, 20, 30, 40, 50], 95), 48.0)
        self.assertAlmostEqual(percentile(list(range(101)), 37), 37.0)
        with self.assertRaises(ValueError):
            percentile([1], 101)
''')
