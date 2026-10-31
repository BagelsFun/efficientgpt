"""Model backends with measured (not estimated) timing.

Every backend implements `chat(messages, tools=None, max_tokens, temperature,
think, on_token) -> Reply`. Stats: client-side time to first token (first
streamed content or thinking token), completion tokens, decode tokens/s,
prompt tokens/s where the backend reports it, and the timing source.
"""

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class GenStats:
    ttft_s: float = float('nan')
    total_s: float = float('nan')
    prompt_tokens: int = 0
    completion_tokens: int = 0
    decode_tok_s: float = float('nan')   # generated tokens after the first / time after the first token
    prompt_tok_s: float = float('nan')   # prefill throughput when the backend reports it
    source: str = ''
    token_times: list = field(default_factory=list)  # perf_counter() when each generated token became available


@dataclass
class Reply:
    text: str
    thinking: str = ''
    tool_calls: list = field(default_factory=list)  # [{'name': str, 'arguments': dict}]
    stats: GenStats = field(default_factory=GenStats)


THINK_RE = re.compile(r'<think>(.*?)</think>', re.S)
TOOL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def split_thinking_and_tools(text):
    """Qwen-style raw text -> (answer, thinking, tool_calls)."""
    thinking = '\n'.join(m.strip() for m in THINK_RE.findall(text))
    text = THINK_RE.sub('', text)
    if '<think>' in text:  # unterminated thinking (budget ran out)
        thinking, text = (thinking + text.split('<think>', 1)[1]).strip(), ''
    calls = []
    for raw in TOOL_RE.findall(text):
        try:
            obj = json.loads(raw)
            calls.append({'name': obj.get('name', ''), 'arguments': obj.get('arguments', {}) or {}})
        except json.JSONDecodeError:
            calls.append({'name': '__malformed__', 'arguments': {'raw': raw}})
    return TOOL_RE.sub('', text).strip(), thinking, calls


class OllamaBackend:
    """Ollama /api/chat over HTTP (stdlib only). Timing: client TTFT plus server-reported token counts and durations."""

    def __init__(self, model, host='http://127.0.0.1:11434', num_ctx=8192, keep_alive='10m', options=None):
        self.model, self.host, self.num_ctx, self.keep_alive = model, host.rstrip('/'), num_ctx, keep_alive
        self.options = dict(options or {})
        self.name = f'ollama:{model}'

    def _post(self, path, payload, stream):
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=600)

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0, think=False, on_token: Optional[Callable] = None):
        payload = {'model': self.model, 'messages': messages, 'stream': True, 'think': bool(think),
                   'keep_alive': self.keep_alive,
                   'options': {'num_predict': max_tokens, 'temperature': temperature, 'num_ctx': self.num_ctx, **self.options}}
        if tools:
            payload['tools'] = tools
        t0 = time.perf_counter()
        first, text, thinking, calls, final, times = None, [], [], [], {}, []
        with self._post('/api/chat', payload, True) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get('message', {})
                piece, tpiece = msg.get('content', ''), msg.get('thinking', '')
                if piece or tpiece or msg.get('tool_calls'):
                    now = time.perf_counter()
                    times.append(now)  # per streamed chunk: usually one token, occasionally two
                    if first is None:
                        first = now
                if piece:
                    text.append(piece)
                    if on_token:
                        on_token(piece)
                if tpiece:
                    thinking.append(tpiece)
                for c in msg.get('tool_calls') or []:
                    fn = c.get('function', {})
                    args = fn.get('arguments', {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {'raw': args}
                    calls.append({'name': fn.get('name', ''), 'arguments': args})
                if chunk.get('done'):
                    final = chunk
        t1 = time.perf_counter()
        ev, evd = final.get('eval_count', 0), final.get('eval_duration', 0) / 1e9
        pe, ped = final.get('prompt_eval_count', 0), final.get('prompt_eval_duration', 0) / 1e9
        stats = GenStats(ttft_s=(first - t0) if first else float('nan'), total_s=t1 - t0, prompt_tokens=pe,
                         completion_tokens=ev, decode_tok_s=ev / evd if evd else float('nan'),
                         prompt_tok_s=pe / ped if ped else float('nan'), source='client ttft + ollama server counts', token_times=times)
        return Reply(''.join(text).strip(), ''.join(thinking).strip(), calls, stats)

    def embed(self, texts, model='nomic-embed-text'):
        with self._post('/api/embed', {'model': model, 'input': list(texts), 'keep_alive': self.keep_alive}, False) as r:
            return json.loads(r.read())['embeddings']

    def loaded_models(self):
        with urllib.request.urlopen(self.host + '/api/ps', timeout=10) as r:
            return json.loads(r.read()).get('models', [])

    def unload(self):
        try:
            self._post('/api/generate', {'model': self.model, 'keep_alive': 0}, False).close()
        except Exception:
            pass


def _sync_streamer_class():
    from transformers import TextStreamer

    class SyncStreamer(TextStreamer):
        """Collects decoded text and the time of the first generated token (put() syncs via tolist())."""

        def __init__(self, tokenizer, on_token=None):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=False)
            self.on_token, self.first, self.pieces, self.times = on_token, None, [], []

        def put(self, value):
            was_prompt = self.skip_prompt and self.next_tokens_are_prompt
            super().put(value)
            if not was_prompt:
                now = time.perf_counter()
                self.times.append(now)  # one put() per generated token (after the prompt)
                if self.first is None:
                    self.first = now

        def on_finalized_text(self, text, stream_end=False):
            if text:
                self.pieces.append(text)
                if self.on_token:
                    self.on_token(text)

    return SyncStreamer


def _SyncStreamer(tokenizer, on_token=None):
    return _sync_streamer_class()(tokenizer, on_token)


def pick_bucket(need, buckets, limit):
    """Smallest cache capacity >= need among the buckets (capped by limit); None if the request does not fit."""
    for c in sorted(b for b in buckets if b <= limit) + [limit]:
        if c >= need:
            return c
    return None


class HFBackend:
    """Hugging Face transformers on the local GPU. quant: 'bf16' | 'fp16' | 'nf4' (bitsandbytes 4-bit).

    static_cache=True uses transformers' static KV cache, which lets generate() compile the decode step
    (first calls pay compilation; see EXPERIMENT_LOG ASSIST-02 for measured effect).
    """

    def __init__(self, model_id, revision=None, quant='bf16', device='cuda', static_cache=False, adapter=None,
                 fast_decode=False, fast_cache_len=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.quant = torch, device, quant
        self.static_cache = static_cache
        self.name = f'hf:{model_id}:{quant}' + (':static' if static_cache else '') + (f'+lora:{adapter}' if adapter else '')
        kw = {'revision': revision}
        if quant == 'nf4':
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            kw['device_map'] = {'': 0}
        elif quant == 'bnb8':  # bitsandbytes LLM.int8 (baseline for the int8 path)
            from transformers import BitsAndBytesConfig
            kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            kw['device_map'] = {'': 0}
        else:  # bf16 / fp16; 'int8' loads bf16 and quantises below
            kw['dtype'] = torch.float16 if quant == 'fp16' else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        if quant not in ('nf4', 'bnb8'):
            self.model.to(device)
        if quant == 'int8':  # weight-only int8, EfficientGPT Triton kernel (research/realmodel, approximate)
            from research.realmodel.int8_linear import quantize_model
            quantize_model(self.model)
        elif quant == 'torchao_int8':  # baseline: torchao int8 weight-only (per-channel), same layers as 'int8'
            from torch import nn
            from torchao.quantization import Int8WeightOnlyConfig, quantize_
            lm = self.model.get_output_embeddings()
            if lm is not None and lm.weight is self.model.get_input_embeddings().weight:  # untie, as quantize_model does
                lm.weight = nn.Parameter(lm.weight.detach().clone(), requires_grad=False)
                self.model.config.tie_word_embeddings = False
            quantize_(self.model, Int8WeightOnlyConfig())
            torch.cuda.empty_cache()
        if adapter:  # LoRA adapter trained by assistant.train.train_qlora (base weights untouched)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        # fast_decode: greedy requests use research.realmodel.fast_decode.GraphGreedyDecoder (one CUDA graph per token, no
        # per-token host sync; V3). Sampling, or prompts longer than fast_cache_len, fall back to generate().
        self.fast_decode, self.fast_cache_len, self._decoder = fast_decode, fast_cache_len, None
        if fast_decode:
            self.name += ':graph'

    GRAPH_BUCKETS = (256, 384, 512, 1024, 2048, 4096)
    MAX_GRAPH_DECODERS = 3  # each bucket holds a KV cache + a graph pool; least recently used are dropped (V3 audit)

    def _eos_ids(self):
        tok = self.tokenizer
        gen = getattr(self.model, 'generation_config', None)
        ids = getattr(gen, 'eos_token_id', None) if gen is not None else None
        ids = set(ids if isinstance(ids, (list, tuple)) else ([ids] if ids is not None else []))
        # generate() stops on every id in generation_config (Qwen3: <|im_end|> and <|endoftext|>); so must we (V3 audit)
        return (ids | {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}) - {None}

    def _chat_graph(self, input_ids, max_tokens, on_token):
        # Adaptive capacity (EfficientGPT Phase 5): static-cache attention scans the whole cache every step, so one fixed
        # 4096-slot cache made decode about 2x slower than a request-sized one (V3 bench). One captured graph per bucket,
        # each request uses the smallest bucket that fits prompt + new tokens.
        from research.realmodel.fast_decode import GraphGreedyDecoder
        cap = pick_bucket(input_ids.shape[1] + max_tokens, self.GRAPH_BUCKETS, self.fast_cache_len)
        if self._decoder is None:
            self._decoder = {}
        capture_s = 0.0
        if cap not in self._decoder:
            while len(self._decoder) >= self.MAX_GRAPH_DECODERS:
                del self._decoder[next(iter(self._decoder))]  # oldest-used first (dict order is refreshed below)
                self.torch.cuda.empty_cache()
            t = time.perf_counter()
            self._decoder[cap] = GraphGreedyDecoder(self.model, max_cache_len=cap)
            capture_s = time.perf_counter() - t
        decoder = self._decoder.pop(cap)
        self._decoder[cap] = decoder  # mark most recently used
        tok = self.tokenizer
        eos = self._eos_ids()
        kept = []
        shown = ['']

        times = []

        def emit(t):
            # incremental decode: a character split across byte-level tokens is only emitted once complete
            times.append(time.perf_counter())
            if t in eos:
                return
            kept.append(t)
            text = tok.decode(kept, skip_special_tokens=False)
            if text.endswith('�'):
                return
            if on_token and len(text) > len(shown[0]):
                on_token(text[len(shown[0]):])
            shown[0] = text
        out, st = decoder.generate(input_ids, max_tokens, eos_ids=eos, on_token=emit)
        st['capture_s'] = capture_s + st.get('capture_s', 0.0)
        raw = tok.decode([t for t in out if t not in eos], skip_special_tokens=False)
        for special in ('<|im_end|>', '<|endoftext|>'):
            raw = raw.replace(special, '')
        text, thinking, calls = split_thinking_and_tools(raw)
        n = len(out)
        decode = (n - 1) / (st['total_s'] - st['ttft_s']) if n > 1 and st['total_s'] > st['ttft_s'] else float('nan')
        stats = GenStats(ttft_s=st['ttft_s'], total_s=st['total_s'], prompt_tokens=input_ids.shape[1], completion_tokens=n,
                         decode_tok_s=decode,
                         source=f"client wall clock (graph decoder, pinned-memory token stream; capture {st['capture_s']:.1f} s excluded)", token_times=times)
        return Reply(text, thinking, calls, stats)

    def count_tokens(self, text):
        return len(self.tokenizer(text).input_ids)

    def chat(self, messages, tools=None, max_tokens=512, temperature=0.0, think=False, on_token: Optional[Callable] = None):
        torch = self.torch
        ids = self.tokenizer.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                                 enable_thinking=bool(think), return_tensors='pt', return_dict=True)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        if self.fast_decode and temperature == 0 and ids['input_ids'].shape[1] + max_tokens <= self.fast_cache_len:
            return self._chat_graph(ids['input_ids'], max_tokens, on_token)
        # Synchronous streamer on the calling thread: a worker thread would hang the caller if generate() raised, and
        # Inductor's CUDA-graph trees (used by the static cache) keep thread-local state (EXPERIMENT_LOG ASSIST-03).
        streamer = _SyncStreamer(self.tokenizer, on_token)
        gen = {'max_new_tokens': max_tokens, 'streamer': streamer, 'do_sample': temperature > 0,
               'pad_token_id': self.tokenizer.eos_token_id}
        if self.static_cache:
            gen['cache_implementation'] = 'static'
        if temperature > 0:
            gen.update(temperature=temperature, top_p=0.95, top_k=20)
        else:
            gen.update(temperature=None, top_p=None, top_k=None)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            seq = self.model.generate(**ids, **gen)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        first = streamer.first
        n_prompt = ids['input_ids'].shape[1]
        n_new = seq.shape[1] - n_prompt
        raw = ''.join(streamer.pieces)
        for tok in ('<|im_end|>', '<|endoftext|>'):
            raw = raw.replace(tok, '')
        text, thinking, calls = split_thinking_and_tools(raw)
        decode = (n_new - 1) / (t1 - first) if first and n_new > 1 and t1 > first else float('nan')
        stats = GenStats(ttft_s=(first - t0) if first else float('nan'), total_s=t1 - t0, prompt_tokens=n_prompt,
                         completion_tokens=n_new, decode_tok_s=decode, source='client wall clock (streamer)', token_times=streamer.times)
        return Reply(text, thinking, calls, stats)
