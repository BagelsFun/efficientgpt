"""assistant-quality-v1: 52 original, objectively graded items (written for this project; no external data).

Categories: 'code' (Python function, hidden assert tests), 'math' (integer answer),
'mcq' (PyTorch / CUDA / GPU-performance, one letter). Each item has split
'dev' or 'test'. The 'test' items are held out: never use them for prompt
tuning or as training data (fine-tuning comparisons are reported on them).
"""

PROTOCOL = 'assistant-quality-v1'

CODE = [
    ('is_palindrome', 'def is_palindrome(s: str) -> bool:\n    """True if s reads the same backwards, ignoring case and non-alphanumeric characters."""',
     ['assert is_palindrome("A man, a plan, a canal: Panama")', 'assert not is_palindrome("race a car")',
      'assert is_palindrome("")', 'assert is_palindrome("No lemon, no melon")', 'assert not is_palindrome("ab")']),
    ('merge_intervals', 'def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    """Merge overlapping or touching closed intervals; return them sorted by start."""',
     ['assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]', 'assert merge_intervals([[1,4],[4,5]]) == [[1,5]]',
      'assert merge_intervals([]) == []', 'assert merge_intervals([[5,7],[1,2]]) == [[1,2],[5,7]]', 'assert merge_intervals([[1,10],[2,3]]) == [[1,10]]']),
    ('rle_encode', 'def rle_encode(s: str) -> str:\n    """Run-length encode: "aaabcc" -> "a3b1c2"; empty string -> ""."""',
     ['assert rle_encode("aaabcc") == "a3b1c2"', 'assert rle_encode("") == ""', 'assert rle_encode("z") == "z1"',
      'assert rle_encode("aabbaa") == "a2b2a2"']),
    ('top_k_frequent', 'def top_k_frequent(nums: list[int], k: int) -> list[int]:\n    """k most frequent values, ordered by frequency descending, ties by smaller value first."""',
     ['assert top_k_frequent([1,1,1,2,2,3], 2) == [1, 2]', 'assert top_k_frequent([4,4,5,5,6], 2) == [4, 5]',
      'assert top_k_frequent([7], 1) == [7]', 'assert top_k_frequent([3,3,2,2,1,1,1], 3) == [1, 2, 3]']),
    ('matmul', 'def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n    """Matrix product of nested lists; raise ValueError if inner dimensions differ."""',
     ['assert matmul([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]', 'assert matmul([[1,2,3]], [[1],[2],[3]]) == [[14]]',
      'try:\n    matmul([[1,2]], [[1,2]])\n    assert False\nexcept ValueError:\n    pass']),
    ('softmax', 'def softmax(xs: list[float]) -> list[float]:\n    """Numerically stable softmax of a non-empty list."""',
     ['r = softmax([1.0, 2.0, 3.0]); assert abs(sum(r) - 1) < 1e-9 and r[2] > r[1] > r[0]',
      'r = softmax([1000.0, 1000.0]); assert all(abs(v - 0.5) < 1e-9 for v in r)', 'r = softmax([-1e9, 0.0]); assert r[1] > 0.999']),
    ('chunked', 'def chunked(seq: list, n: int) -> list[list]:\n    """Split seq into consecutive lists of length n (last may be shorter); raise ValueError if n <= 0."""',
     ['assert chunked([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]', 'assert chunked([], 3) == []', 'assert chunked([1,2], 5) == [[1,2]]',
      'try:\n    chunked([1], 0)\n    assert False\nexcept ValueError:\n    pass']),
    ('flatten', 'def flatten(nested: list) -> list:\n    """Flatten arbitrarily nested lists into one list, preserving order."""',
     ['assert flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]', 'assert flatten([]) == []', 'assert flatten([[[]],[1]]) == [1]']),
    ('roman_to_int', 'def roman_to_int(s: str) -> int:\n    """Convert a valid Roman numeral (I..MMMCMXCIX) to an integer."""',
     ['assert roman_to_int("III") == 3', 'assert roman_to_int("MCMXCIV") == 1994', 'assert roman_to_int("XLIX") == 49',
      'assert roman_to_int("MMMCMXCIX") == 3999']),
    ('binary_search', 'def binary_search(arr: list[int], target: int) -> int:\n    """Index of the leftmost occurrence of target in sorted arr, or -1."""',
     ['assert binary_search([1,2,2,2,3], 2) == 1', 'assert binary_search([1,3,5], 4) == -1', 'assert binary_search([], 1) == -1',
      'assert binary_search([5,5,5], 5) == 0', 'assert binary_search([1,2,3,4,5,6,7], 7) == 6']),
    ('count_params', 'def count_params(shapes: list[tuple[int, ...]]) -> int:\n    """Total number of elements across tensors with the given shapes (a 0-dim shape () has 1 element)."""',
     ['assert count_params([(384, 1152), (1152,)]) == 443520', 'assert count_params([]) == 0', 'assert count_params([()]) == 1',
      'assert count_params([(2, 3, 4), (5,)]) == 29']),
    ('conv_out_size', 'def conv_out_size(n: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:\n    """Output length of a 1-D convolution with input n, kernel k, stride s, padding p, dilation d (PyTorch formula)."""',
     ['assert conv_out_size(32, 3) == 30', 'assert conv_out_size(32, 3, 1, 1) == 32', 'assert conv_out_size(32, 3, 2, 1) == 16',
      'assert conv_out_size(10, 3, 1, 0, 2) == 6', 'assert conv_out_size(7, 7) == 1']),
    ('moving_average', 'def moving_average(xs: list[float], w: int) -> list[float]:\n    """Averages of each length-w window (len(xs) - w + 1 values); [] if w > len(xs)."""',
     ['assert moving_average([1,2,3,4], 2) == [1.5, 2.5, 3.5]', 'assert moving_average([1,2], 3) == []', 'assert moving_average([5], 1) == [5.0]']),
    ('kv_cache_bytes', 'def kv_cache_bytes(layers: int, heads: int, head_dim: int, seq_len: int, batch: int, bytes_per_elem: int = 2) -> int:\n    """Bytes needed to store keys and values for every layer of a transformer KV cache."""',
     ['assert kv_cache_bytes(6, 6, 64, 256, 16, 4) == 75497472', 'assert kv_cache_bytes(1, 1, 1, 1, 1) == 4',
      'assert kv_cache_bytes(28, 8, 128, 4096, 1) == 469762048']),
    ('lru_misses', 'def lru_misses(capacity: int, accesses: list[int]) -> int:\n    """Number of cache misses for an LRU cache of the given capacity serving the accesses in order."""',
     ['assert lru_misses(2, [1,2,1,3,2]) == 4', 'assert lru_misses(1, [1,1,1]) == 1', 'assert lru_misses(3, []) == 0',
      'assert lru_misses(2, [1,2,3,1,2,3]) == 6']),
    ('parse_duration', 'def parse_duration(s: str) -> int:\n    """Seconds in a duration like "1h30m15s" (each of h, m, s optional, in that order); raise ValueError if invalid or empty."""',
     ['assert parse_duration("1h30m15s") == 5415', 'assert parse_duration("45s") == 45', 'assert parse_duration("2h") == 7200',
      'assert parse_duration("10m5s") == 605', 'try:\n    parse_duration("abc")\n    assert False\nexcept ValueError:\n    pass',
      'try:\n    parse_duration("")\n    assert False\nexcept ValueError:\n    pass']),
]

MATH = [
    ('A GPU has 24 streaming multiprocessors. Each SM can hold at most 48 resident warps, and a warp has 32 threads. What is the maximum number of resident threads on the GPU?', 36864),
    ('Mia buys 3 notebooks at $4 each and 2 pens at $1.50 each, and pays with a $20 bill. How much change does she get, in cents?', 500),
    ('A tank fills at 12 litres per minute and drains at 5 litres per minute at the same time. Starting empty, after how many minutes does it hold 84 litres?', 12),
    ('What is the sum of all integers from 1 to 200 inclusive that are divisible by 3 or by 5?', 9368),
    ('A model has 6 layers. Each layer has 4 attention weight matrices of size 384x384 and an MLP with one 384x1536 and one 1536x384 matrix. Ignoring biases and embeddings, how many weights are there in total?', 10616832),
    ('A KV cache stores keys and values for batch 16, sequence length 256, 6 layers, 6 heads, head dimension 64, in float32. How many MiB (2^20 bytes) does it need?', 72),
    ('3 workers paint a fence in 12 hours. How many hours would 4 workers take, working at the same rate?', 9),
    ('A sequence has a1 = 2 and a(n+1) = 3*a(n) - 1. What is a5?', 122),
    ('In how many ways can you choose 3 items from 10 distinct items, ignoring order?', 120),
    ('What is 19^2 - 11^2?', 240),
    ('A kernel reads 64 MB and writes 32 MB in 0.5 ms. What is its effective bandwidth in GB/s, using 1 GB = 1000 MB?', 192),
    ('A clock shows 7:45. What time is it 135 minutes later, expressed as minutes past midnight on a 24-hour clock?', 600),
    ('A rectangle has perimeter 50 and length 17. What is its area?', 136),
    ('What is the largest prime number less than 100?', 97),
    ('How many trailing zeros does 100! (100 factorial) have?', 24),
    ('An 8-billion-parameter model is quantized to 4 bits per parameter. How many gigabytes (10^9 bytes) do its weights take?', 4),
]

MCQ = [
    ('Which call makes the CPU wait until all queued work on the current CUDA device has finished?',
     ['torch.cuda.empty_cache()', 'torch.cuda.synchronize()', 'torch.no_grad()', 'torch.backends.cudnn.benchmark = True'], 'B'),
    ('Which PyTorch expression modifies tensor x in place?', ['x.add(1)', 'x + 1', 'x.add_(1)', 'torch.add(x, 1)'], 'C'),
    ('How many threads are in a warp on NVIDIA GPUs?', ['16', '32', '64', '128'], 'B'),
    ('What is the main purpose of CUDA Graphs?',
     ['Reduce CPU launch overhead by replaying a captured sequence of GPU work', 'Increase numerical precision',
      'Compress model weights', 'Shard a model across GPUs'], 'A'),
    ('What does torch.inference_mode() do compared with ordinary execution?',
     ['Enables gradient tracking', 'Disables autograd recording and version-counter tracking, reducing overhead',
      'Moves the model to the GPU', 'Turns on mixed precision'], 'B'),
    ('Which floating-point format has 8 exponent bits and 7 explicit mantissa bits?', ['float16', 'bfloat16', 'float32', 'float64'], 'B'),
    ('A memory-bound kernel is limited mainly by:', ['Arithmetic throughput', 'Memory bandwidth', 'Disk speed', 'The number of CPU cores'], 'B'),
    ('What does a transformer KV cache store during autoregressive decoding?',
     ['Gradients of the loss', 'Keys and values of previous tokens so they are not recomputed', 'Optimizer states', 'Only token embeddings'], 'B'),
    ('What does model.eval() change in PyTorch?',
     ['It disables gradient computation', 'It switches layers such as dropout and batch norm to inference behaviour',
      'It frees GPU memory', 'It quantizes the model'], 'B'),
    ('Which approach most reduces GPU memory when fine-tuning a 7B-parameter model on an 8 GB GPU?',
     ['Full fine-tuning in float32', 'LoRA on a float16 base model', 'QLoRA: a 4-bit quantized base model with LoRA adapters', 'Doubling the batch size'], 'C'),
    ('What is the default backend of torch.compile?', ['TensorRT', 'Inductor', 'XLA', 'ONNX Runtime'], 'B'),
    ('In CUDA, coalesced global memory access means:',
     ['Threads of a warp access contiguous addresses, so few memory transactions are needed', 'Each thread uses shared memory',
      'All accesses are atomic', 'The L2 cache is disabled'], 'A'),
    ('CUDA shared memory is:', ['Off-chip DRAM visible to all blocks', 'Fast on-chip memory shared by the threads of one block',
                                'Pinned host memory', 'A cache for constant data only'], 'B'),
    ('Why time GPU code with torch.cuda.Event(enable_timing=True) instead of time.time() alone?',
     ['Events are recorded on the GPU stream, so asynchronous kernel launches do not distort the measurement',
      'Events make kernels run faster', 'Events disable the caching allocator', 'Autograd requires events'], 'A'),
    ('GPU occupancy refers to:', ['The ratio of active warps to the maximum warps an SM supports', 'The fraction of device memory in use',
                                  'Power consumption', 'The current clock frequency'], 'A'),
    ('Which statement about float16 and bfloat16 is true?',
     ['float16 has a larger exponent range than bfloat16', 'bfloat16 has the same exponent range as float32 but less precision',
      'bfloat16 has more mantissa bits than float16', 'They are bitwise identical formats'], 'B'),
    ('In a decoder-only transformer, the causal mask ensures that:',
     ['Each token attends only to itself and earlier positions', 'Tokens attend in both directions',
      'Padding tokens are ignored', 'Batch elements are independent'], 'A'),
    ('What does tensor.contiguous() return?',
     ['Always the same tensor object', 'A tensor whose memory is laid out in row-major order, copying only if needed',
      'A copy on the GPU', 'A tensor detached from autograd'], 'B'),
    ('Gradient (activation) checkpointing trades:',
     ['Extra compute for lower memory, by recomputing activations during the backward pass', 'Extra memory for less compute',
      'Precision for speed', 'Disk space for RAM'], 'A'),
    ('What is the best first step before optimizing a slow PyTorch model?',
     ['Rewrite everything in CUDA', 'Profile it to find the actual bottleneck', 'Increase the batch size arbitrarily', 'Switch to float64'], 'B'),
]


def items():
    """All items with ids and splits (alternating dev/test within each category, so both halves are balanced)."""
    out = []
    for i, (name, signature, tests) in enumerate(CODE):
        out.append({'id': f'code/{name}', 'category': 'code', 'split': 'dev' if i % 2 == 0 else 'test',
                    'prompt': f'Write this Python function. Reply with only the complete function in one ```python code block.\n\n{signature}',
                    'entry': name, 'tests': tests})
    for i, (q, a) in enumerate(MATH):
        out.append({'id': f'math/{i:02d}', 'category': 'math', 'split': 'dev' if i % 2 == 0 else 'test',
                    'prompt': f'{q}\nThink briefly, then give the final integer on the last line as "Answer: <number>".', 'answer': a})
    for i, (q, opts, a) in enumerate(MCQ):
        body = '\n'.join(f'{l}. {o}' for l, o in zip('ABCD', opts))
        out.append({'id': f'mcq/{i:02d}', 'category': 'mcq', 'split': 'dev' if i % 2 == 0 else 'test',
                    'prompt': f'{q}\n{body}\nReply with only the letter of the correct option.', 'answer': a})
    return out
