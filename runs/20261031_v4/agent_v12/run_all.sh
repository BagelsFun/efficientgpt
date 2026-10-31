#!/bin/sh
# Agent v1.2 calibration study (accuracy only; GPU shared with distillation, timings not valid).
export PYTHONIOENCODING=utf-8
D=runs/20261031_v4/agent_v12
for sys in default calibrated; do
  .venv/Scripts/python.exe -m assistant.bench.agent_e2e --backend hf:Qwen/Qwen3-1.7B --quant nf4 --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --configs model_only rag_tools --system $sys --out $D/qwen3_1.7b_nf4_$sys.json > $D/qwen3_1.7b_nf4_$sys.log 2>&1
  .venv/Scripts/python.exe -m assistant.bench.agent_e2e --backend ollama:qwen3.5:4b --configs model_only rag_tools --system $sys --out $D/qwen35_4b_$sys.json > $D/qwen35_4b_$sys.log 2>&1
done
echo done
