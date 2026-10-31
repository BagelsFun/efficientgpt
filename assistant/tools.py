"""Permission-controlled tools for the assistant.

Permission levels: 'read' (allowed automatically, confined to the workspace root,
never into checkpoints, data, .git or secrets) and 'execute' (requires the
approval callback to return True for every single call). There are no write or
delete tools; code the model proposes is returned as text for a human to apply.
"""

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

DENY_PARTS = {'.git', '.venv', 'data', 'results'}
DENY_SUFFIX = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors', '.pkl', '.zip', '.env', '.key', '.pem'}
TEST_MODULE_RE = re.compile(r'^(research|assistant)(\.[A-Za-z_][A-Za-z0-9_]*)+$')


class ToolError(Exception):
    pass


class Toolbox:
    def __init__(self, root, approve=None, retriever=None, python=None, test_timeout=900, exclude=()):
        """`exclude`: extra glob patterns (POSIX relative paths) the tools must treat as inaccessible."""
        self.root = Path(root).resolve()
        self.exclude = tuple(exclude)
        self.approve = approve or (lambda name, args: False)
        self.retriever = retriever
        self.python = python or sys.executable
        self.test_timeout = test_timeout
        self.log = []
        self.tools = {
            'read_file': ('read', self.read_file, 'Read lines of a text file in the workspace.',
                          {'path': 'string', 'start_line': 'integer', 'end_line': 'integer'}, ['path']),
            'list_dir': ('read', self.list_dir, 'List a workspace directory.', {'path': 'string'}, ['path']),
            'search_code': ('read', self.search_code, 'Regex search over workspace text files; returns path:line: text.',
                            {'pattern': 'string', 'max_results': 'integer'}, ['pattern']),
            'retrieve_docs': ('read', self.retrieve_docs, 'Retrieve the most relevant documentation/code chunks for a question.',
                              {'query': 'string', 'k': 'integer'}, ['query']),
            'run_unittests': ('execute', self.run_unittests, 'Run a Python unittest module of this repo (needs user approval).',
                              {'module': 'string'}, ['module']),
        }

    def schemas(self, allowed=('read', 'execute')):
        out = []
        for name, (level, _, desc, props, required) in self.tools.items():
            if level in allowed:
                out.append({'type': 'function', 'function': {
                    'name': name, 'description': desc,
                    'parameters': {'type': 'object', 'properties': {k: {'type': v} for k, v in props.items()},
                                   'required': required}}})
        return out

    def _path(self, rel):
        p = (self.root / str(rel)).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError('path is outside the workspace')
        rel_parts = p.relative_to(self.root).parts
        if any(part in DENY_PARTS or part.startswith('out-') for part in rel_parts) or p.suffix.lower() in DENY_SUFFIX:
            raise ToolError('access to this path is not permitted')
        if any(fnmatch.fnmatch(p.relative_to(self.root).as_posix(), pat) for pat in self.exclude):
            raise ToolError('access to this path is not permitted')
        return p

    def call(self, name, arguments):
        if name not in self.tools:
            raise ToolError(f'unknown tool {name}')
        level, fn = self.tools[name][:2]
        if not isinstance(arguments, dict):  # malformed model output must not crash the agent (review V1 #5)
            self.log.append((name, arguments, 'ERROR'))
            return 'ERROR: arguments must be a JSON object'
        if level == 'execute' and not self.approve(name, arguments):
            self.log.append((name, arguments, 'DENIED'))
            return 'DENIED: the user did not approve this action.'
        try:
            result = fn(**arguments)
        except TypeError as e:
            result = f'ERROR: bad arguments: {e}'
        except ToolError as e:
            result = f'ERROR: {e}'
        except (ValueError, OSError, subprocess.TimeoutExpired) as e:  # e.g. int('abc') line numbers, test timeouts
            result = f'ERROR: {type(e).__name__}: {e}'
        self.log.append((name, arguments, 'ok' if not str(result).startswith(('ERROR', 'DENIED')) else str(result)[:80]))
        return result

    def read_file(self, path, start_line=1, end_line=None):
        p = self._path(path)
        if not p.is_file():
            raise ToolError('not a file')
        lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
        s = max(1, int(start_line))
        e = min(len(lines), int(end_line) if end_line else s + 199)
        return '\n'.join(f'{i}: {lines[i - 1]}' for i in range(s, e + 1)) + (f'\n... ({len(lines)} lines total)' if e < len(lines) else '')

    def list_dir(self, path='.'):
        p = self._path(path)
        if not p.is_dir():
            raise ToolError('not a directory')
        items = []
        for c in sorted(p.iterdir()):
            try:
                self._path(c.relative_to(self.root))
            except ToolError:
                continue
            items.append(c.name + ('/' if c.is_dir() else ''))
        return '\n'.join(items[:300])

    def search_code(self, pattern, max_results=30):
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f'bad regex: {e}')
        hits = []
        for f in sorted(self.root.rglob('*')):
            if f.suffix not in ('.py', '.md', '.txt') or not f.is_file():
                continue
            try:
                self._path(f.relative_to(self.root))
            except ToolError:
                continue
            for i, line in enumerate(f.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if rx.search(line):
                    hits.append(f'{f.relative_to(self.root).as_posix()}:{i}: {line.strip()[:200]}')
                    if len(hits) >= int(max_results):
                        return '\n'.join(hits)
        return '\n'.join(hits) or 'no matches'

    def retrieve_docs(self, query, k=4):
        if self.retriever is None:
            raise ToolError('retrieval is not configured')
        return self.retriever.format(self.retriever.search(query, int(k)))

    def run_unittests(self, module):
        if not TEST_MODULE_RE.match(module):
            raise ToolError('module must look like research.x.test_y or assistant.x')
        proc = subprocess.run([self.python, '-m', 'unittest', module], cwd=self.root, capture_output=True, text=True,
                              timeout=self.test_timeout)
        tail = (proc.stdout + proc.stderr)[-3000:]
        return f'exit code {proc.returncode}\n{tail}'
