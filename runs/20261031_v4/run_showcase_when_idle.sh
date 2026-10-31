#!/bin/sh
# Wait until the GPU has been idle (<= 5% utilisation) for 6 consecutive samples 20 s apart (2 min), max 4 h, then run
# the showcase benchmark. The previous attempt ran under an external GPU load (19-30%) and is kept as CONTAMINATED.
export PYTHONIOENCODING=utf-8
ok=0; waited=0
while [ $ok -lt 6 ]; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$u" -le 5 ]; then ok=$((ok+1)); else ok=0; fi
  sleep 20; waited=$((waited+20))
  if [ $waited -ge 14400 ]; then echo "GPU never idle for 2 min within 4 h; not run"; exit 2; fi
done
echo "GPU idle; starting benchmark after waiting ${waited}s"
.venv/Scripts/python.exe -m research.realmodel.bench_int8 --out runs/20261031_v4/showcase_bench --modes int8_graph llamacpp_q8 bf16_static torchao_int8_static_at --rounds 4 --reps 5 > runs/20261031_v4/showcase_bench.console.log 2>&1
echo "benchmark exit=$?"
