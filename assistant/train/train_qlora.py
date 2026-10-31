"""QLoRA adapter for Qwen3-1.7B on chat-format JSONL (loss on assistant tokens only).

    python -m assistant.train.train_qlora --data results/assistant/data/distill_v1.jsonl --out out-assistant/qwen3-1.7b-lora-<stamp>

Base: 4-bit NF4 (bitsandbytes, double quant, bf16 compute), gradient checkpointing; LoRA on all attention and MLP
projections. A fixed fraction of examples is held out for validation loss. Writes the adapter, a metrics JSON with
train/val loss per step and the peak GPU memory, and never touches existing checkpoints (refuses an existing --out).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'


def encode(tok, messages, max_len):
    """input_ids + labels with -100 on everything except the final assistant turn."""
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    full = tok.apply_chat_template(messages, enable_thinking=False, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError('chat template prefix mismatch')
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return f_ids, labels[:len(f_ids)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--max-len', type=int, default=1024)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=0, help='>0: stop early (smoke test)')
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [json.loads(l) for l in args.data.open(encoding='utf-8')]
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    train = [encode(tok, r['messages'], args.max_len) for r in train_rows]
    val = [encode(tok, r['messages'], args.max_len) for r in val_rows]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, quantization_config=bnb, device_map={'': 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(train) * args.epochs / args.accum))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 10)) *
                                              max(0.0, 1 - s / total_steps) if total_steps > 1 else 1.0)

    def loss_on(ex):
        ids = torch.tensor([ex[0]], device='cuda')
        labels = torch.tensor([ex[1]], device='cuda')
        return model(input_ids=ids, labels=labels).loss

    def val_loss():
        model.eval()
        with torch.no_grad():
            vals = [float(loss_on(ex)) for ex in val]
        model.train()
        return sum(vals) / len(vals)

    torch.cuda.reset_peak_memory_stats()
    metrics = {'model': MODEL, 'revision': REVISION, 'data': str(args.data), 'train_examples': len(train), 'val_examples': len(val),
               'trainable_params': trainable, 'args': {k: str(v) for k, v in vars(args).items()}, 'steps': []}
    metrics['val_loss_before'] = val_loss()
    print(f"train {len(train)} val {len(val)} trainable {trainable:,} steps {total_steps} val_loss_before {metrics['val_loss_before']:.4f}", flush=True)
    model.train()
    t0, step, order = time.perf_counter(), 0, []
    while step < total_steps:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        acc = 0.0
        for _ in range(args.accum):
            if not order:
                order = list(range(len(train)))
                random.shuffle(order)
            loss = loss_on(train[order.pop()]) / args.accum
            loss.backward()
            acc += float(loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        row = {'step': step, 'train_loss': round(acc, 4), 'lr': sched.get_last_lr()[0], 'elapsed_s': round(time.perf_counter() - t0, 1)}
        if step % 10 == 0 or step == total_steps:
            row['val_loss'] = round(val_loss(), 4)
        metrics['steps'].append(row)
        print(json.dumps(row), flush=True)
    metrics['val_loss_after'] = val_loss()
    metrics['peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    metrics['peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    metrics['train_seconds'] = round(time.perf_counter() - t0, 1)
    args.out.mkdir(parents=True)
    model.save_pretrained(args.out)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (args.out / 'val_names.json').write_text(json.dumps([r['name'] for r in val_rows]), encoding='utf-8')
    print(json.dumps({k: v for k, v in metrics.items() if k != 'steps'}, indent=1))


if __name__ == '__main__':
    main()
