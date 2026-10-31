"""KernelAgent-Ada: evidence-driven Triton kernel search for EfficientGPT hot operators.

Pipeline: task -> candidate source (template generator) -> AST policy screen ->
isolated subprocess (timeout, GPU memory cap) compile + correctness + CUDA-graph
timing against eager / Inductor / best existing -> statistics -> feedback to the
search strategy. See reports/KERNELAGENT_ADA_REPORT.md.
"""
