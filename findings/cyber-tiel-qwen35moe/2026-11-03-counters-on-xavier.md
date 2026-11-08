# Hardware counters work on this Xavier - and they overturn "ALU-bound"

Date: 2026-11-03
Status: capability established, one published claim corrected, one experiment opened

## Why this file exists

Earlier audits stated that no profilers existed on this board (`nsys`, `ncu` and `nvprof` were
absent) and that the two things worth having - per-kernel timings under CUDA graphs, and
hardware counters - were both unreachable: CUPTI cannot profile graph-launched kernels, and PC
sampling is unsupported on Tegra. Those statements were inherited rather than measured.

The packages were installed from the `momonga-extra` RPM feed today
(`cuda-cupti-dev` 11.4.298+1, `nsight-systems-cli` 2022.5.2.171, `nsight-compute-cli`
2022.3.0). Both tools run, and both of the "unreachable" capabilities work.

**Prediction, recorded after the fact and wrong:** I expected the counter API to initialise but
report no usable metrics for `gv11b`, and expected graph-launched kernels to be invisible. Both
were false. This file is a capability discovery, not a hypothesis test, so there was no
pre-registered prediction.

## What was installed, and the one trap

| Package | Purpose | Works |
|---|---|---|
| `cuda-cupti-dev` 11.4.298+1 | `cupti.h`, `cupti_activity.h`, `cupti_profiler_target.h` in `/usr/local/cuda-11.4/include` | yes |
| `nsight-systems-cli` 2022.5.2.171 | `nsys`, timeline and kernel summary | yes as of `-r1`, see below |
| `nsight-compute-cli` 2022.3.0 | `ncu`, per-kernel counters | yes |
| `sysstat` 12.7.5 | `iostat`, `mpstat`, `pidstat`, `sar` | yes |
| `fio` 3.36 | storage benchmark | yes as of feed `2026.11.03-2`, see below |

Three traps, all cost time:

1. **`dnf` root and user keep separate metadata caches.** `sudo dnf makecache` refreshed the
   root cache; a following unprivileged `dnf list` still reported metadata from the previous day
   and "No matching Packages to list". The packages were visible all along under `sudo`. Always
   query and install as the same user, i.e. both with `sudo`.
2. **`fio` was uninstallable for about two hours.** `fio-3.36+git0+b2403d413e-r0` required
   `libaio.so.1()(64bit)` and `libaio1 >= 0.3.113`, neither of which was in `momonga-extra`, and
   `libaio.so*` was not on the filesystem. **Fixed in feed release `2026.11.03-2`**, which adds
   `libaio1` 0.3.113-r0 (providing `LIBAIO_0.1`, `LIBAIO_0.4`, `LIBAIO_0.5`); `fio` now installs
   and links `/usr/lib/libaio.so.1`. Rule of thumb: a package that installs but fails at a
   missing soname is a feed gap worth reporting, not a reason to fall back to `dd`.
3. **nvcc against this glibc needs the workaround include path.** Without
   `-Xcompiler=-isystem,/usr/include/cuda-compat-workarounds`, every build dies in
   `/usr/include/bits/math-vector-64.h` with `__Float32x4_t is undefined`: the system glibc is
   built for gcc 13.5 and g++-10.3.0 has no such types. The path is already in
   `build-volta/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make`; it was missing from the
   recipe written in `AGENTS.md` section 4. Full working command:

   ```
   /usr/local/cuda-11.4/bin/nvcc -ccbin /usr/bin/g++-10.3.0 \
     -Xcompiler=-isystem,/usr/include/c++/10.3.0 \
     -Xcompiler=-isystem,/usr/include/c++/10.3.0/aarch64-poky-linux \
     -Xcompiler=-isystem,/usr/include/cuda-compat-workarounds \
     -I/usr/local/cuda-11.4/include -L/usr/local/cuda-11.4/lib \
     "--generate-code=arch=compute_72,code=[compute_72,sm_72]" ...
   ```

## Capability 1: CUPTI needs root

A minimal probe (`/tmp/cupti_probe.cu`) calls `cuptiActivityRegisterCallbacks`,
`cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)` and `cuptiProfilerInitialize`, launches one
kernel twice - once plainly, once through a captured `cudaGraphExec_t` - and counts records.

| Call | as user `triph` | as root |
|---|---|---|
| `cuptiActivityRegisterCallbacks` | 35 `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` | 0 `CUPTI_SUCCESS` |
| `cuptiActivityEnable(KERNEL)` | 35 | 0 |
| `cuptiActivityFlushAll` | 35 | 0 |
| kernel records, plain launch | 0 | 1 (3021.4 us) |
| **kernel records, graph launch** | 0 | **1 (3016.2 us)** |
| `cuptiProfilerInitialize` | 35 | 0 |
| `cuptiProfilerGetCounterAvailability` | - | 0, image 2240 bytes |

Two facts follow.

**Graph-launched kernels are visible to the activity API.** The 3016.2 us record came from a
kernel that only ever existed inside a `cudaGraphExec_t`. The earlier note that "CUPTI cannot
profile kernels launched by CUDA graphs" is wrong for the activity API, and wrong for `ncu`
too - see below. Record it as retracted.

**Everything under CUPTI requires root on this board.** This is a privilege check, not a
capability gap. Nothing in the workflow is blocked by it; the probe and `ncu` are both run
under `sudo`. No driver parameter or security setting was changed to achieve this, and none
needs to be.

`cuptiDeviceGetChipName` returns `CUPTI_ERROR_NOT_INITIALIZED` unless the profiler interface was
initialised first; that is a probe ordering artefact, not a platform limitation. The chip name
for this device is `gv11b` and `ncu --chip gv11b` enumerates its metrics offline.

## Capability 2: `ncu` collects real counters, including through graphs

```
sudo ncu --kernel-name regex:spin --launch-skip 1 --launch-count 1 \
     --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg /tmp/cupti_probe
```

profiles the graph-launched kernel in one pass, and against `bench/opbench` - which runs under
the ggml CUDA-graph backend - it profiles `mul_mat_vec_q_moe` without complaint. Per-kernel
timing under graphs is therefore available *without* `GGML_CUDA_OP_TIME`, which disables graphs
and perturbs what it measures.

## `nsys` was record-only for about two hours, then fixed in the same day's feed

With `nsight-systems-cli-...-r0`, `sudo nsys profile` succeeded and wrote a `.qdstrm`, but
`nsys stats` could not read it - `Importer error: The importer binary and its dependencies were
not found` - because the RPM shipped only the Tegra target tree with no QDSTRM importer.

**Fixed in `-r1` (feed release `2026.11.03-2`)**, which bundles `QdstrmImporter` and its
headless dependencies under `/opt/nvidia/nsight-systems/2022.5.2/host-linux-armv8/` and adds
`/usr/bin/QdstrmImporter`, with the Qt/WebEngine GUI components stripped. Verified end to end:

```
sudo nsys profile --force-overwrite true -o /tmp/nsys_memtest /tmp/memtest
  -> Generated: /tmp/nsys_memtest.nsys-rep        (import happens automatically)
sudo nsys stats --report gpukernsum /tmp/nsys_memtest.nsys-rep
  -> stream_read   1 instance   2,423,104 ns     grid 256    block 256
```

The report name is `gpukernsum` on this build, not the `cuda_gpu_kern_sum` used by newer
releases; `nsys stats --help-reports` lists what exists here. The 2.42 ms figure matches the
110.9 GB/s stream measurement, so the timeline and the counters agree.

Both tools now work on-device. `ncu` gives counters and per-kernel detail but serialises and
replays launches; `nsys` gives true timing of a whole run without perturbing it.

## Capability 3: which counters are real on `gv11b`

`ncu --query-metrics-mode all --chip gv11b` lists 154,667 metric names, and that number is a
trap: a listed metric can still be dead. The allocation was checked against a kernel with known
traffic - `/tmp/memtest.cu` reads a 256 MiB buffer with coalesced `float4` loads, so every byte
must come from device memory.

| Metric | Reported | Expected | Verdict |
|---|---|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 8,388,608 sectors | 268,435,456 / 32 = 8,388,608 | exact |
| `lts__t_sectors_op_read.sum` | 8,388,624 sectors | ~8,388,608 | good |
| `lts__t_sector_hit_rate.pct` | 0.00 | 0 for a pure stream | good |
| `lts__t_bytes.sum` | 268.44 MB | 256 MiB | exact |
| `lts__d_sectors_fill_device.sum` | **0** | 8,388,608 | **dead** |
| `lts__d_sectors_fill_sysmem.sum` | 8,388,613 | 8,388,608 | good - this is the one |

The kernel ran in 2.42 ms, which is 110.9 GB/s and matches the separately measured 109.7-112
GB/s read ceiling, so the timing and the counters agree.

Two consequences:

- **There are no `dram__` counters at all on `gv11b`** - zero metrics with that prefix. True
  DRAM counters do not exist on this part.
- **On Tegra the device-memory fill counter is `lts__d_sectors_fill_sysmem`, not
  `..._fill_device`.** GPU memory is system memory, so L2 fills from "sysmem" *are* the reads
  that came from LPDDR4x. `lts__d_sectors_fill_sysmem.sum * 32` is real DRAM read traffic.
  Anyone who reads only `fill_device` concludes the chip has no working memory counters. That
  is exactly the mistake this board's earlier audits made.

## Measurement: the MoE expert matmul, with counters instead of inference

`bench/opbench`, `OPBENCH_SINGLE=1 OPBENCH_K=2048 OPBENCH_N=512 OPBENCH_TOK=6`, 256 experts of
which 8 are routed, `mul_mat_vec_q_moe`, one replayed launch. Type 21 is IQ3_S.

| Type | Kernel time | Warp inst | L2 read | L2 hit | DRAM fill | Achieved BW | % of 111 GB/s |
|---|---|---|---|---|---|---|---|
| IQ3_S | 381.66 us | 10,407,936 | 679,911 sect | 31.54% | 676,257 sect = 21.64 MB | 56.7 GB/s | 51% |
| IQ4_XS | 288.58 us | 6,402,048 | 839,045 sect | 27.11% | 835,820 sect = 26.75 MB | 92.7 GB/s | 84% |
| Q6_K | 501.38 us | 8,073,216 | 1,295,438 sect | 19.44% | 1,291,398 sect = 41.32 MB | 82.4 GB/s | 74% |
| Q4_0 | 321.06 us | 4,558,848 | 888,027 sect | 25.97% | 884,578 sect = 28.31 MB | 88.2 GB/s | 79% |

The timing-only version of this comparison is what produced the published "IQ3_S is ALU-bound"
claim: IQ3_S moves 19% fewer bytes than IQ4_XS and still takes 32% longer. The counters say the
same thing about the *outcome* and something different about the *cause*.

## Correction: IQ3_S is not ALU-bound, it is latency-bound at low occupancy

Warp stall accounting, in warps resident per issue-active cycle:

| Stall reason | IQ3_S | IQ4_XS |
|---|---|---|
| `long_scoreboard` (waiting on a load) | **3.05** | **5.30** |
| `not_selected` | 1.48 | 2.11 |
| `math_pipe_throttle` | 1.16 | 1.71 |
| `wait` (fixed latency) | 1.02 | 1.22 |
| `lg_throttle` | 0.50 | 1.72 |
| `short_scoreboard` | 0.19 | 0.36 |

Memory-dependent stalls dominate both types. `math_pipe_throttle` - the signature of an ALU
bound kernel - is the *third* reason on IQ3_S, at a third of the memory stall count. It is not
ALU-bound.

Occupancy explains why the memory stalls are exposed:

| | IQ3_S | IQ4_XS |
|---|---|---|
| block / grid | 32 / 12,288 | 32 / 12,288 |
| registers per thread | 79 | 60 |
| blocks per SM (register limit) | 24 | 32 |
| architectural occupancy cap | 37.50% | 50% |
| achieved occupancy | 34.93% | 46.16% |
| waves per SM | 64 | 48 |

The kernel is issued as one warp per block, so occupancy *is* the resident block count. IQ3_S
pays 79 registers per thread, which caps it at 24 warps per SM = 37.5% of the 64-warp
architectural maximum, and it achieves nearly all of that cap (34.9%). With only ~24 warps
resident and every warp stalling on an `iq3s_grid` lookup, latency cannot be hidden, so the
kernel reaches 51% of the DRAM ceiling where IQ4_XS reaches 84%.

Restated correctly: **IQ3_S executes 1.63x the warp instructions of IQ4_XS while achieving 61%
of its bandwidth, because its register footprint caps occupancy at 37.5%, and the memory-latency
stalls that follow dominate its runtime.** The instruction count and the occupancy are not
independent - the same extra decode work that adds instructions also costs registers.

This does not change any previous end-to-end result. It changes where to look for one.

## What this opens

1. **Per-kernel timing of the real server under graphs.** `nsys` or `ncu` on `llama-server`
   with graphs enabled replaces `GGML_CUDA_OP_TIME`, which disables graphs and therefore
   measures a different program. The 90-of-118 ms MoE figure was obtained the perturbing way and
   should be re-taken.
2. **The occupancy question on IQ3_S.** The counter evidence predicts that raising resident
   warps from 24 toward 32 moves IQ3_S off 51% of the ceiling. The obvious lever is register
   pressure in the `iq3_s` MMVQ path (79 -> 64). The obvious risk is that cutting registers
   spills and lands slower, which is measurable in one `ncu` run of `opbench`. This is a
   kernel-level prediction and says nothing yet about tok/s: the same experiment must end with a
   `bench/one.sh` run, or it does not count.
3. **`sysstat` is in place** but `fio` is not, so storage numbers stay on `iostat`/`dd`.

## eMMC (closing note, not a work item)

`fio` confirms the eMMC is the documented ~88-90 MB/s and saturates at queue depth 1; an 18.12 GB
model therefore streams in 201 s, which reproduces the observed 3 min to `/health`. An NVMe is
being installed, which is the right fix - there is nothing to tune on the eMMC side and decoding
speed is unaffected by it.

## Artifacts

- `/tmp/cupti_probe.cu` - CUPTI capability and graph-visibility probe
- `/tmp/memtest.cu` - known-traffic kernel used to validate the memory counters
- `/tmp/gv11b_metrics.txt` - full 154,667-line metric list for `gv11b`
- `bench/opbench` build against `build-volta`; needs
  `LD_LIBRARY_PATH=<build>/bin:<build>/ggml/src`
