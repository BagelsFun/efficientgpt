#!/bin/sh
# KA-07 review follow-ups: 2 more fresh-process rounds for the headline pair, and a bf16-vs-bf16 noise floor for quality.
export PYTHONIOENCODING=utf-8
D=runs/20261031_v4/int8_e2e
.venv/Scripts/python.exe -m research.realmodel.bench_int8 --out $D --modes bf16_static int8_static --rounds 4 --reps 5 > $D/extra_rounds.log 2>&1
.venv/Scripts/python.exe -m research.realmodel.bench_int8 --out $D --child bf16_dynamic --child-out $D/bf16_dynamic_quality.json --reps 1 --child-quality --quality-parts nll,greedy > $D/noise_floor.log 2>&1
echo done
