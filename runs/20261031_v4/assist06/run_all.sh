#!/bin/sh
# ASSIST-06: re-measure after the tool rule + forced final answer (accuracy only; GPU shared with distillation)
export PYTHONIOENCODING=utf-8
D=runs/20261031_v4/assist06
HF="--backend hf:Qwen/Qwen3-1.7B --quant nf4 --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
.venv/Scripts/python.exe -m assistant.bench.coding_agent --backend ollama:qwen3.5:4b --configs agent --out $D/coding_qwen35_4b.json > $D/coding_qwen35_4b.log 2>&1
.venv/Scripts/python.exe -m assistant.bench.coding_agent $HF --configs agent --out $D/coding_qwen3_1.7b_nf4.json > $D/coding_qwen3_1.7b_nf4.log 2>&1
.venv/Scripts/python.exe -m assistant.bench.agent_e2e --backend ollama:qwen3.5:4b --configs model_only rag_tools --system calibrated --out $D/agent_qwen35_4b_calibrated.json > $D/agent_qwen35_4b.log 2>&1
.venv/Scripts/python.exe -m assistant.bench.agent_e2e $HF --configs model_only rag_tools --system calibrated --out $D/agent_qwen3_1.7b_nf4_calibrated.json > $D/agent_qwen3_1.7b_nf4.log 2>&1
echo done
