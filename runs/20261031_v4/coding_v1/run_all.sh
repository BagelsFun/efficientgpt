#!/bin/sh
# assistant-coding-v1 (accuracy only; GPU shared with distillation, timings not valid)
export PYTHONIOENCODING=utf-8
D=runs/20261031_v4/coding_v1
.venv/Scripts/python.exe -m assistant.bench.coding_agent --backend ollama:qwen3.5:4b --out $D/qwen35_4b.json > $D/qwen35_4b.log 2>&1
.venv/Scripts/python.exe -m assistant.bench.coding_agent --backend hf:Qwen/Qwen3-1.7B --quant nf4 --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --out $D/qwen3_1.7b_nf4.json > $D/qwen3_1.7b_nf4.log 2>&1
echo done
