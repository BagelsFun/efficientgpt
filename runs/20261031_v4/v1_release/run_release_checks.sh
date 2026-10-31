#!/bin/sh
# V1 pre-release: full regression suite, then a fresh reproduction of the headline Phase 5 benchmark (idle GPU).
export PYTHONIOENCODING=utf-8
D=runs/20261031_v4/v1_release
P=.venv/Scripts/python.exe
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $D/gpu_before_tests.txt
$P -m unittest research.phase4.test_decoder research.phase4.test_compact research.phase5.test_pool research.phase5.test_prefix research.phase5.test_graph_gpu research.phase5.test_analyze_ci research.phase5.test_near_tie research.phase5.test_compiled_step research.phase5.test_inprocess research.phase5.test_eos research.phase5.test_sampling research.phase5.test_ragged research.phase5.test_gpu_combinations research.v1.test_demo research.v1.test_report > $D/tests_phase5.txt 2>&1; echo "phase5 rc=$?" >> $D/test_rc.txt
$P -m unittest research.kernelagent.test_evaluate research.kernelagent.test_search research.realmodel.test_int8_linear > $D/tests_kernelagent.txt 2>&1; echo "kernelagent rc=$?" >> $D/test_rc.txt
$P -m unittest assistant.test_assistant assistant.bench.test_quality_suite assistant.train.test_train > $D/tests_assistant.txt 2>&1; echo "assistant rc=$?" >> $D/test_rc.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $D/gpu_before_bench.txt
mkdir -p research/phase5/ci_runs/20261031_v1_repro
$P -m research.phase5.benchmark_inprocess --modes adaptive_graph compiled_triton_full_graph prefix_sdpa_eager fp16_compiled_triton_full_graph --output research/phase5/ci_runs/20261031_v1_repro/benchmark_raw.csv > research/phase5/ci_runs/20261031_v1_repro/console.log 2>&1; echo "bench rc=$?" >> $D/test_rc.txt
echo done
