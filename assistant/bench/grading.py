"""Objective graders for assistant-quality-v1.

Code is executed in a fresh `python -I` subprocess in an empty temporary directory
with a timeout, after a screen that refuses code touching processes, files or
the network (a filter, not a sandbox). Math: the "Answer:" line, else the last
integer. MCQ: the first standalone option letter.
"""

import re
import subprocess
import sys
import tempfile
import textwrap

CODE_BLOCK_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)
DANGEROUS_RE = re.compile(r'\b(import\s+(os|sys|subprocess|shutil|socket|ctypes|pathlib|signal|multiprocessing|threading)\b|'
                          r'from\s+(os|sys|subprocess|shutil|socket|ctypes|pathlib)\b|__import__|\bopen\s*\(|\beval\s*\(|\bexec\s*\()')
INT_RE = re.compile(r'-?\d[\d,]*')
GRADING_VERSION = 'v1.1'
# v1.1 (2026-10-30): the prompt asks for "only the complete function", so answers that rely on common standard-library
# modules without importing them are not failed for that alone. Applied identically to every model; v1 = no preamble.
PREAMBLE = ('import math\nimport re\nimport collections\nfrom collections import Counter, OrderedDict, defaultdict, deque\n'
            'import itertools\nimport functools\nimport heapq\nfrom typing import *\n')


def extract_code(text):
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return max(blocks, key=len)
    return text if 'def ' in text else ''


def grade_code(response, entry, tests, timeout=10, preamble=True):
    """-> (passed: bool, reason). preamble=False reproduces grading v1."""
    code = extract_code(response)
    if f'def {entry}' not in code:
        return False, 'no function definition found'
    if DANGEROUS_RE.search(code):
        return False, 'refused: code uses a disallowed module or builtin'
    program = (PREAMBLE if preamble else '') + code + '\n\n' + '\n'.join(tests) + '\nprint("ALL_TESTS_PASSED")\n'
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, '-I', '-c', program], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, 'timeout'
    if proc.returncode == 0 and 'ALL_TESTS_PASSED' in proc.stdout:
        return True, 'passed'
    err = (proc.stderr.strip().splitlines() or ['failed'])[-1]
    return False, err[:200]


def grade_math(response, answer):
    m = re.findall(r'Answer:\s*\**\s*(-?\d[\d,]*)', response)
    nums = m or INT_RE.findall(response)
    if not nums:
        return False, 'no integer found'
    got = int(nums[-1].replace(',', ''))
    return got == answer, f'got {got}'


def grade_mcq(response, answer):
    text = re.sub(r'<think>.*?</think>', '', response, flags=re.S).strip()
    m = re.search(r'\b([ABCD])\b', text)
    if not m:
        return False, 'no option letter'
    return m.group(1) == answer, f'got {m.group(1)}'


def grade(item, response, preamble=True):
    if item['category'] == 'code':
        return grade_code(response, item['entry'], item['tests'], preamble=preamble)
    if item['category'] == 'math':
        return grade_math(response, item['answer'])
    return grade_mcq(response, item['answer'])


def dedent(s):
    return textwrap.dedent(s).strip('\n')
