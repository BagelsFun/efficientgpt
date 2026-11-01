# Raw files: NVIDIA L4 / Linux (Google Colab) check, 2026-11-01

Evidence for [`reports/L4_LINUX_VALIDATION_2026-11-01.md`](../../reports/L4_LINUX_VALIDATION_2026-11-01.md). Run by the
author on the author's Colab account: **not an independent reproduction**, and separate from the RTX 4060 / Windows
results in `runs/20261031_v4/`.

- Source: the run directory `repro_NVIDIA_L4_20261101_043259` written by `research/release/reproduce.sh --quick` at
  commit bb1a5bf (04:33–04:47 UTC), archived after the script had finished and before the runtime was deleted
  (`snap_044833.tar.gz`, 15,280 bytes, sha256 `4eb24ba3fc820c5297f52c16abc55480d726617de902a6b2ba558c6175b2251f`).
  `console.log` is the script's console output from the same archive (renamed from `repro.log`); everything else keeps
  its original path.
- Changed before publication: `nvidia_smi_q.txt` lost four lines, the `GPU PDI` (a per-device identifier the bb1a5bf
  filter missed) and three ID fields that were `N/A` (`Host ID`, `GPU Fabric GUID`, `CliqueId`), exactly as the
  filter fixed in commit 718e3b3 removes them. No other file differs from the archive.
- `bench/summary.json` is the benchmark script's quick summary, which pools the 5 timed requests of every round; the
  reported figures come from `bench/analysis.json` / `bench/REPORT.md` (ratio of medians of round medians, bootstrap
  over rounds), which is the method used for the published RTX 4060 results.
