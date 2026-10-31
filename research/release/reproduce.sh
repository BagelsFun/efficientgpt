#!/usr/bin/env bash
# Independent reproduction of the Qwen3-1.7B benchmark on any NVIDIA GPU (sm_80+, >= 8 GB), Linux or WSL2.
#
#   bash research/release/reproduce.sh            # env + tests + demo + benchmark (all baselines it can run), ~1 h
#   bash research/release/reproduce.sh --quick    # env + tests + demo + int8 vs bf16 only (shorter; not timed)
#   bash research/release/reproduce.sh --dry-run  # print the commands, run nothing
#   options: --skip-tests  --no-llamacpp  --deferred-stop (also run the transformers DeferredStopCheck measurement)
#
# Run from the repository root inside the venv from docs/REPRODUCE_ON_ANOTHER_MACHINE.md §2. Everything is written to
# runs/repro_<gpu>_<date>/ and packed into runs/repro_<gpu>_<date>.tar.gz; nothing is uploaded. Each step's log is kept,
# and a failing step is recorded and the script continues (exit status 1 at the end), so a partial result is still
# worth sending.
set -u

QUICK=0; DRY=0; TESTS=1; LLAMA=auto; DSTOP=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;; --dry-run) DRY=1 ;; --skip-tests) TESTS=0 ;; --no-llamacpp) LLAMA=0 ;; --deferred-stop) DSTOP=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

[ -f research/realmodel/bench_int8.py ] || { echo "run this from the repository root" >&2; exit 2; }
PY=${PYTHON:-python}
# one GPU, numbered as nvidia-smi numbers it (multi-GPU machines: set CUDA_VISIBLE_DEVICES to choose)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GPU_ID=${CUDA_VISIBLE_DEVICES%%,*}
GPU=$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -cs 'A-Za-z0-9' '_' | sed 's/_$//')
OUT=runs/repro_${GPU:-nogpu}_$(date -u +%Y%m%d_%H%M%S)
STATUS=()

step() {  # step <name> <command...>: run, log to $OUT/logs/<name>.log, record the exit code
  local name=$1; shift
  echo "== $name: $*"
  [ "$DRY" = 1 ] && return 0
  "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  STATUS+=("$name rc=$rc")
  [ $rc -eq 0 ] && echo "   ok" || echo "   FAILED (rc=$rc), see $OUT/logs/$name.log"
}

[ "$DRY" = 1 ] || mkdir -p "$OUT/logs"
echo "output: $OUT"

step env "$PY" -m research.release.collect_env --out "$OUT/env.json"
if [ "$DRY" = 0 ]; then  # without serial numbers, UUIDs or local file paths
  nvidia-smi -q -i "$GPU_ID" 2>&1 | grep -viE 'serial number|uuid|board part number|board id' > "$OUT/nvidia_smi_q.txt"
  "$PY" -m pip freeze --exclude-editable 2>&1 | sed -E 's#@ file://.*#@ <local file>#' > "$OUT/pip_freeze.txt"
fi

UTIL=$(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "$UTIL" ] && [ "$UTIL" -gt 15 ] 2>/dev/null; then
  echo "WARNING: GPU utilisation is ${UTIL}%. Close other GPU programs; timings from a busy GPU are flagged 'contended'."
fi

if [ "$TESTS" = 1 ]; then
  step tests_realmodel "$PY" -m unittest -v research.realmodel.test_int8_linear research.realmodel.test_fast_decode
  step tests_kernelagent "$PY" -m unittest -v research.kernelagent.test_evaluate research.kernelagent.test_search
fi
step demo "$PY" -m research.realmodel.demo --tokens 128

MODES="int8_graph bf16_static"
if [ "$QUICK" = 0 ]; then
  MODES="$MODES torchao_int8_static_at"
  if [ "$LLAMA" = auto ]; then command -v ollama > /dev/null 2>&1 && LLAMA=1 || LLAMA=0; fi
  if [ "$LLAMA" = 1 ]; then
    step ollama_pull ollama pull qwen3:1.7b-q8_0
    MODES="$MODES llamacpp_q8"
  else
    echo "   (llama.cpp baseline skipped: Ollama not found)"
  fi
fi
# shellcheck disable=SC2086
step bench "$PY" -m research.realmodel.bench_int8 --out "$OUT/bench" --modes $MODES --rounds 4 --reps 5
step analyze "$PY" -m research.realmodel.analyze_int8 --dir "$OUT/bench"
if [ "$DSTOP" = 1 ]; then
  step deferred_stop_qwen "$PY" -m research.realmodel.deferred_stop_cuda --out "$OUT/deferred_stop_qwen3_1.7b.json" --reps 12
  step deferred_stop_smollm "$PY" -m research.realmodel.deferred_stop_cuda --out "$OUT/deferred_stop_smollm2_135m.json" \
    --reps 12 --model HuggingFaceTB/SmolLM2-135M-Instruct
fi

[ "$DRY" = 1 ] && exit 0
printf '%s\n' "${STATUS[@]}" > "$OUT/STATUS.txt"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
cat "$OUT/STATUS.txt"
[ -f "$OUT/bench/REPORT.md" ] && { echo; cat "$OUT/bench/REPORT.md"; }
echo
echo "Done. Attach $OUT.tar.gz to a reproduction report (Discussions > Hardware results, or the issue template)."
echo "Look through every file first, logs included: they hold your hardware, package versions and cache paths, which may"
echo "contain your user name. Nothing was uploaded."
FAILED=$(grep -vc 'rc=0$' "$OUT/STATUS.txt")
[ "$FAILED" = 0 ] || { echo "$FAILED step(s) failed; the archive still contains everything that ran." >&2; exit 1; }
