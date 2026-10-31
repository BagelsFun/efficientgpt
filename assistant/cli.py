"""Terminal chat interface.

    .venv\\Scripts\\python.exe -m assistant.cli --backend ollama:qwen3.5:4b
    .venv\\Scripts\\python.exe -m assistant.cli --backend hf:Qwen/Qwen3-1.7B --quant bf16

Commands: /reset  /rag on|off  /tools on|off  /think on|off  /stats  /quit
Read-only tools run automatically; test execution asks for confirmation each time.
"""

import argparse
import sys
from pathlib import Path

from .agent import INTERACTIVE_SYSTEM, Agent
from .context import ContextManager


def make_backend(spec, quant='bf16', revision=None, static_cache=False, adapter=None):
    kind, _, name = spec.partition(':')
    if kind == 'ollama':
        from .backends import OllamaBackend
        return OllamaBackend(name)
    if kind == 'hf':
        from .backends import HFBackend
        return HFBackend(name, revision=revision, quant=quant, static_cache=static_cache, adapter=adapter)
    raise SystemExit(f'unknown backend {spec!r} (use ollama:<model> or hf:<repo>)')


def ask_approval(name, args):
    answer = input(f'\n[approval] allow {name}({args})? [y/N] ').strip().lower()
    return answer == 'y'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', default='ollama:qwen3.5:4b')
    ap.add_argument('--quant', default='bf16', choices=('bf16', 'fp16', 'nf4', 'int8', 'bnb8'),
                    help='HF only. int8 = EfficientGPT weight-only int8 (approximate; 1.55x decode vs bf16 static, R1)')
    ap.add_argument('--revision', default=None)
    ap.add_argument('--no-static-cache', action='store_true',
                    help='HF only: use the dynamic KV cache (the static cache decodes 2.15x faster on an idle RTX 4060, '
                         'ASSIST-04, but compiles on the first call and is not bit-identical to the dynamic path)')
    ap.add_argument('--root', default='.')
    ap.add_argument('--embed', action='store_true', help='hybrid retrieval with Ollama nomic-embed-text')
    args = ap.parse_args()
    from .retrieval import Retriever
    from .tools import Toolbox
    backend = make_backend(args.backend, args.quant, args.revision, static_cache=not args.no_static_cache)
    embedder = None
    if args.embed:
        from .backends import OllamaBackend
        embedder = OllamaBackend('nomic-embed-text').embed
    retriever = Retriever(args.root, embedder, Path('results/assistant/embed_cache.json') if embedder else None)
    count = getattr(backend, 'count_tokens', None)
    context = ContextManager(INTERACTIVE_SYSTEM, 6000, count) if count else ContextManager(INTERACTIVE_SYSTEM, 6000)
    agent = Agent(backend, context, Toolbox(args.root, ask_approval, retriever), retriever)
    print(f'{backend.name} ready ({len(retriever.chunks)} chunks indexed). /quit to exit.')
    last = None
    while True:
        try:
            text = input('\nyou> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == '/quit':
            break
        if text == '/reset':
            context.reset()
            print('(context cleared)')
            continue
        if text.startswith(('/rag', '/tools', '/think')):
            key, _, val = text[1:].partition(' ')
            setattr(agent, {'rag': 'use_rag', 'tools': 'use_tools', 'think': 'think'}[key], val.strip() == 'on')
            print(f'({key} {"on" if val.strip() == "on" else "off"})')
            continue
        if text == '/stats':
            print(last.steps if last else '(no turns yet)')
            continue
        print('assistant> ', end='', flush=True)
        last = agent.ask(text, on_token=lambda t: print(t, end='', flush=True))
        if last.used_tools:
            print(f'\n(tools used: {[c["name"] for s in last.steps for c in s["tool_calls"]]})')
        s = last.steps[-1]['stats']
        print(f"\n[{s['completion_tokens']} tok, ttft {s['ttft_s']:.2f}s, {s['decode_tok_s']:.1f} tok/s]")
    return 0


if __name__ == '__main__':
    sys.exit(main())
