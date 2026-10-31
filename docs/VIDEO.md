# Showcase video: what it shows and where each number comes from

A 74-second walkthrough (1080p, narrated, English subtitles burned in): https://youtu.be/SKWrki4NHn4

## Structure
| Time | Scene | What is on screen | Kind of footage |
|---|---|---|---|
| 0:00 | Hook | Qwen3-1.7B, one RTX 4060, 111.9 tok/s | photo (credited below) + data |
| 0:12 | Problem | batch-1 decode is weight-bandwidth bound; with `generate()` + int8 the GPU idles ≈ 2 ms per token | visualisation of documented numbers |
| 0:17 | Fix | one CUDA graph per token (`research/realmodel/fast_decode.py`), steps 91.3 → 96.8 → 105.1 → 111.9 tok/s | real code (screenshot of GitHub) + data |
| 0:25 | Result | `python -m research.realmodel.demo` on the RTX 4060 | recorded output, replayed at recorded speed; cuts marked |
| 0:32 | Comparison | the README table: parity with llama.cpp Q8_0 (1.02×, below the 5% bar), 1.33× torchao, 1.89× HF bf16; llama.cpp has the faster first token | data |
| 0:44 | KernelAgent-Ada | a kernel that drops the int8 scale is rejected; a correct one is validated (16/16, held-out 20/20, 1.21× faster in isolation); that microbenchmark winner is 8.5% slower inside the model, so it is not shipped | recorded output (excerpts) + data |
| 0:59 | Evidence | 156/156 tests from a clean checkout; README table | real screenshots |
| 1:06 | End | "Don't take the numbers on trust. Read the code." | — |

The terminal panels replay output captured with arrival timestamps from real runs on 2026-10-31; a direct screen recording
was not used. Every scene carries an on-screen label saying which kind of footage it is.

## Figure audit
| Figure | Source in this repository |
|---|---|
| 111.9 / 109.9 / 84.4 / 59.3 tok/s; 1.02× [1.015, 1.051]; 1.33×; 1.89×; TTFT 54 vs 33 ms | `BENCHMARK_REPORT.md` §1a |
| ≈ 3.4 GB read per token in bf16 → ≈ 62 tok/s ceiling; int8 ceiling ≈ 120 tok/s | `reports/EFFICIENTGPT_TECHNICAL_REPORT.md` §2 |
| 91.3 tok/s with `generate()` + int8; GPU idle ≈ 2 ms/token (10.95 ms wall vs ≈ 9.0 ms of kernels) | `reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md` §3; technical report §5 |
| Step ladder 91.3 → 96.8 → 105.1 → 111.9 | `reports/EFFICIENTGPT_V3_RESEARCH_REPORT.md` §3; `docs/ARCHITECTURE.md` |
| Recorded run: 108.5 tok/s, TTFT 56 ms, 185 tokens (a single run, not the benchmark) | the recording itself (`python -m research.realmodel.demo`) |
| Rejected candidate: error 5.77e+05 > bound 179 | recorded `python -m research.showcase.demo_kernelagent` output |
| Validated candidate: 16/16, 20/20 held out, 14.23 vs 17.24 µs (1.21×) | same recording |
| Split-K end to end: 10.32 vs 9.51 ms per step (+8.5%), ≈ 197 extra reduce kernels per token | `research/realmodel/int8_linear.py` (comment); V3 report §3 |
| 156/156 tests | `runs/20261031_v4/recognition_clean_checkout.json` |

## Credits
- **Photographs:** two frames by 极客湾Geekerwan from Wikimedia Commons, **CC BY 3.0**
  (https://creativecommons.org/licenses/by/3.0), cropped and colour-graded:
  [frame 07](https://commons.wikimedia.org/wiki/File:Video_%C3%BCber_die_GeForce_RTX_4060_(%E6%9E%81%E5%AE%A2%E6%B9%BEGeekerwan)_07.png)
  and [frame 16](https://commons.wikimedia.org/wiki/File:Video_%C3%BCber_die_GeForce_RTX_4060_(%E6%9E%81%E5%AE%A2%E6%B9%BEGeekerwan)_16.png).
  Attribution is also shown on screen.
- **Narration:** BagelsFun's own voice (no synthetic voice).
- **Music and sound:** original, synthesised for this video; no samples or third-party music.
- **Everything else** (screenshots of this repository, code, recorded output, charts): this project, MIT.
- The poster image `docs/media/showcase_poster.jpg` is a frame of the comparison scene and contains no third-party material.
