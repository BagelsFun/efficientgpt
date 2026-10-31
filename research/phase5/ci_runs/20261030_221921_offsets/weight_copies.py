"""In-process test: do independent copies of the model weights run at different speeds? (6 copies, own graphs, interleaved)."""
import sys, copy, statistics, torch, json
sys.path.insert(0, '.')
from pathlib import Path
from tokenizers import Tokenizer
from research.phase4.benchmark_decode import load_model, make_prompts
from research.phase5.adaptive_pool import DecoderPool
torch.backends.cuda.matmul.fp32_precision = 'tf32'
base = load_model(Path('out-tinystories-v2/baseline-2500.pt')); tok = Tokenizer.from_file('data/tinystories_v2/tokenizer.json')
copies = [base] + [copy.deepcopy(base) for _ in range(5)]
out = {}
with torch.inference_mode():
    for b in (1, 16):
        x = make_prompts(tok, 4096, b, 192, 1)[0]
        decs = [DecoderPool(m, 'full').get(x, 32)[0] for m in copies]
        for d in decs:
            for _ in range(3): d.generate(x, 32)
        times = [[] for _ in decs]
        for i in range(60):
            order = list(range(len(decs))); k = i % len(decs); order = order[k:] + order[:k]
            for j in order:
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); decs[j].generate(x, 32); e.record(); torch.cuda.synchronize(); times[j].append(s.elapsed_time(e))
        med = [statistics.median(t) for t in times]
        out[f'B{b}'] = {'copy_medians_ms': [round(m, 3) for m in med], 'cv': statistics.pstdev(med) / statistics.mean(med),
                        'min_max_ratio': max(med) / min(med),
                        'weight_ptr_mod_2MiB': [m.transformer.h[0].mlp.c_fc.weight.data_ptr() % (2 << 20) for m in copies]}
        print(f'B{b}', json.dumps(out[f"B{b}"]), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=2)
