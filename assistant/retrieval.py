"""Local retrieval over the repository: BM25 (always) fused with optional dense embeddings (Ollama nomic-embed-text).

Chunks .md/.py/.txt files under the given roots. Dense vectors are cached on disk
keyed by chunk hash, so re-indexing only embeds changed text. Fusion is
reciprocal-rank fusion (k=60).
"""

import fnmatch
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?')
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'results', 'data', 'out-tinystories-v2', 'out-efficientgpt-small',
             'out-shakespeare-char', 'out-tinystories-head3', 'assets'}


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def key(self):
        return hashlib.sha1(f'{self.path}:{self.start_line}:{self.text}'.encode()).hexdigest()


def tokenize(text):
    out = []
    for t in TOKEN_RE.findall(text):
        t = t.lower()
        out.append(t)
        if '_' in t:
            out.extend(p for p in t.split('_') if p)
    return out


def chunk_files(root, include=('.md', '.py', '.txt'), max_lines=40, overlap=8, max_bytes=400_000, exclude=()):
    """`exclude`: glob patterns on the POSIX relative path (e.g. benchmark files that contain answers)."""
    root = Path(root)
    chunks = []
    for path in sorted(root.rglob('*')):
        if path.suffix not in include or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if any(fnmatch.fnmatch(path.relative_to(root).as_posix(), pat) for pat in exclude):
            continue
        if path.stat().st_size > max_bytes:
            continue
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        step = max_lines - overlap
        for s in range(0, max(1, len(lines)), step):
            body = '\n'.join(lines[s:s + max_lines]).strip()
            if body:
                chunks.append(Chunk(str(path.relative_to(root)).replace('\\', '/'), s + 1, min(len(lines), s + max_lines), body))
            if s + max_lines >= len(lines):
                break
    return chunks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokenize(query)
        out = []
        for d, L in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


class Retriever:
    def __init__(self, root, embedder=None, cache_path=None, exclude=()):
        self.root = Path(root)
        self.chunks = chunk_files(self.root, exclude=exclude)
        self.bm25 = BM25([c.path + ' ' + c.text for c in self.chunks])
        self.embedder, self.vectors = embedder, None
        if embedder is not None:
            self.vectors = self._embed_all(Path(cache_path) if cache_path else None)

    def _embed_all(self, cache_path):
        cache = {}
        if cache_path and cache_path.exists():
            cache = json.loads(cache_path.read_text())
        missing = [c for c in self.chunks if c.key not in cache]
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            vecs = self.embedder(['search_document: ' + c.path + '\n' + c.text[:2000] for c in batch])
            for c, v in zip(batch, vecs):
                cache[c.key] = v
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({c.key: cache[c.key] for c in self.chunks}))
        return [cache[c.key] for c in self.chunks]

    def search(self, query, k=5, mode='hybrid'):
        """mode: 'bm25' | 'dense' | 'hybrid' (reciprocal-rank fusion)."""
        ranks = []
        if mode in ('bm25', 'hybrid'):
            s = self.bm25.scores(query)
            ranks.append(sorted(range(len(s)), key=lambda i: -s[i]))
        if mode in ('dense', 'hybrid') and self.vectors is not None:
            qv = self.embedder(['search_query: ' + query])[0]
            qn = math.sqrt(sum(x * x for x in qv)) or 1.0
            s = [sum(a * b for a, b in zip(qv, v)) / (qn * (math.sqrt(sum(x * x for x in v)) or 1.0)) for v in self.vectors]
            ranks.append(sorted(range(len(s)), key=lambda i: -s[i]))
        fused = Counter()
        for r in ranks:
            for pos, i in enumerate(r[:200]):
                fused[i] += 1.0 / (60 + pos)
        return [self.chunks[i] for i, _ in fused.most_common(k)]

    @staticmethod
    def format(chunks, max_chars=6000):
        out, used = [], 0
        for c in chunks:
            block = f'[{c.path}:{c.start_line}-{c.end_line}]\n{c.text}'
            if used + len(block) > max_chars:
                break
            out.append(block)
            used += len(block)
        return '\n\n'.join(out)
