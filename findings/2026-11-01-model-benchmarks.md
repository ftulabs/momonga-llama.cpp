# llama-bench comparisons: Spark 4B Q8_0 vs Cyber-Tiel 35B-A3B

**Date:** 2026-11-01
**Build:** mainline `dde879b`, `-ngl 99 -fa 1 -t 4`, 3 repetitions
**Method:** `llama-bench -p 512 -n 128`, no speculation (llama-bench does not exercise MTP)

## Spark-X2.6-4B-Q8_0

`/home/triph/models/Spark-X2.6-4B-Q8_0.gguf` (4.07 GiB, 4.11 B params, arch `spark2_5`).

| Property | Value |
|---|---|
| blocks | 36 |
| embedding | 2560 |
| heads / KV heads | 16 / 4 |
| context | 1,048,576 |
| MTP head | **none** (no `nextn` key in metadata) |
| quantization | Q8_0 |

| test | t/s |
|---|---:|
| pp512 | **470.19 +/- 5.01** |
| tg128 | **16.46 +/- 0.03** |

Server measurements with 400-token generations: 14.83-15.14 tok/s, identical across
`-ub 32 / 256 / 1024` and `-t 2 / 4`. Dense batch-1 decode is purely bandwidth-bound, so there is
nothing to tune. With no MTP head, speculation is unavailable, so **16.5 tok/s is the unassisted
ceiling for this file**.

## Cyber-Tiel 35B-A3B, same benchmark

| Model | Size | Params | pp512 | tg128 |
|---|---|---:|---:|---:|
| Spark X2.6 4B Q8_0 | 4.07 GiB | 4.11 B | **470.19** | 16.46 |
| Cyber-Tiel 35B-A3B IQ4_XS | 16.87 GiB | 35.51 B | 224.53 | **17.04** |

## Interpretation

The 4B dense model is **slightly slower at decode** than the 35B MoE, despite having 8.6x fewer
parameters. The arithmetic:

- **Spark Q8_0:** 4.37 GB read per token at 16.46 tok/s = **~72 GB/s**, about 52% of the
  ~137 GB/s peak.
- **Cyber-Tiel MoE:** only ~3B params active at 4.25 bpw, about 1.6 GB read per token at
  17.04 tok/s = **~27 GB/s**, about 20% of peak.

The MoE is inefficient per byte but moves far fewer bytes. The int8 4B model is more bandwidth-
efficient but pays for all 4.1 B parameters on every token.

Where Spark clearly wins is **prefill**: 470 vs 224 tok/s (2.1x), because dense attention has no
expert routing.

## Implications

- A 4B dense model is not a shortcut to higher decode speed on Xavier. Both are constrained by
  the same ~137 GB/s memory system, and the MoE already exploits sparsity.
- With MTP speculation the 35B reaches 31.5 tok/s in steady state, nearly double the dense 4B's
  unassisted 16.5. Speculation is the decisive advantage, not model size.
- Spark is bandwidth-bound at only ~52% of peak with Q8_0. Requantizing to Q4_0 or Q4_K_M would
  read ~2.3 GB/token and could roughly double decode to ~30 tok/s. Untested; needs ~2.4 GB disk.
  Tracked in `TODO.md`.
