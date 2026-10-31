"""Static policy screen for candidate source (a filter, not a proof of safety).

Allowed imports: triton, triton.language, torch, math. Allowed torch attribute
chains: allocation helpers and dtype/device names only, so a candidate cannot
hide the reference computation (torch.softmax, F.scaled_dot_product_attention,
torch.topk, ...) or touch the host (os, subprocess, files, network, eval/exec).
Requires at least one @triton.jit kernel and a module-level `run(inp)`.
"""

import ast

ALLOWED_IMPORTS = {'triton', 'triton.language', 'torch', 'math'}
ALLOWED_TORCH = {'empty', 'empty_like', 'zeros', 'zeros_like', 'full', 'float32', 'float16', 'bfloat16',
                 'int32', 'int64', 'long', 'bool', 'device'}
BANNED_NAMES = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'vars', 'getattr', 'setattr',
                'delattr', 'input', 'breakpoint', 'memoryview', 'os', 'sys', 'subprocess', 'socket', 'shutil',
                'pathlib', 'importlib', 'pickle', 'ctypes', 'F', 'nn'}
# Attributes allowed on anything that is not tl./triton./torch.: shape/launch plumbing only (allowlist, not blacklist:
# a blacklist missed x.aminmax, x.exp2, x.nansum, ... in review). Compute must happen inside @triton.jit kernels.
TENSOR_ATTRS = {'append', 'device', 'dtype', 'numel', 'shape', 'size', 'stride', 'to', 'clamp', 'contiguous', 'view',
                'reshape', 'unsqueeze', 'squeeze', 'ndim', 'element_size', 'is_contiguous'}
ALIASES = {('triton.language', 'tl')}  # the only permitted `import x as y`
RESERVED = {'tl', 'triton', 'torch', 'math'}  # may not be rebound (assignment, argument, loop target, def name)


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def screen(source):
    """Return a list of violations (empty = passes)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f'syntax error: {e}']
    problems, has_jit, has_run = [], False, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    problems.append(f'import {a.name}')
                elif a.asname and (a.name, a.asname) not in ALIASES:
                    problems.append(f'import alias {a.name} as {a.asname}')
        elif isinstance(node, ast.ImportFrom):
            problems.append(f'from-import {node.module}')
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f'banned name {node.id}')
        elif isinstance(node, ast.Name) and node.id in RESERVED and isinstance(node.ctx, (ast.Store, ast.Del)):
            problems.append(f'rebinds {node.id}')
        elif isinstance(node, ast.arg) and node.arg in RESERVED:
            problems.append(f'rebinds {node.arg}')
        elif isinstance(node, ast.Attribute):
            chain = _chain(node)
            if chain and chain[0] == 'torch' and len(chain) >= 2 and chain[1] not in ALLOWED_TORCH:
                problems.append('torch.' + '.'.join(chain[1:]))
            if not (chain and chain[0] in ('tl', 'triton', 'torch', 'math')) and node.attr not in TENSOR_ATTRS:
                problems.append(f'tensor attribute .{node.attr}')
            if node.attr.startswith('__') and node.attr != '__init__':
                problems.append(f'dunder attribute {node.attr}')
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in RESERVED:
                problems.append(f'rebinds {node.name}')
            if isinstance(node, ast.ClassDef):
                continue
            if node.name == 'run' and node in tree.body:
                has_run = True
            for dec in node.decorator_list:
                if _chain(dec) == ['triton', 'jit']:
                    has_jit = True
    if not has_jit:
        problems.append('no @triton.jit kernel')
    if not has_run:
        problems.append('no module-level run(inp)')
    return sorted(set(problems))
