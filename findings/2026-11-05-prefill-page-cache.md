# Prefill streams the whole model through the page cache, and that is where 16 s of TTFT lives (2026-11-05)

Status: prediction recorded before the run.

## 1. The discrepancy this has to explain

The binary that produced the recorded prefill baseline was copied out unchanged and re-measured
tonight at the record's own prompt length. `bench/prefill.csv` `h3-adopted-np400` (2026-11-04):

```
h3-adopted-np400,8192,...,9563,120302.7 ms,79.53 tok/s
h3-adopted-np400,8192,...,9563,120295.7 ms,79.53
h3-adopted-np400,8192,...,9563,120485.4 ms,79.40
```

Same binary tonight (`bench/bins/A`, `NAME=repro-9563`, `LENS=8192`, same flags):

```
repro-9563,8192,...,9563,108742.2 ms,88.00 tok/s   (rep 1)
repro-9563,8192,...,9563,104193.2 ms,91.82         (rep 2)
repro-9563,8192,...,9563,103894.1 ms,92.09         (rep 3)
```

**A 15.7% between-session difference with no code difference**, which is 25x the 0.56% prefill
floor measured with six arms. At `LENS=590` and `LENS=2385` the same comparison is +12% and +15%.
Decode moved the other way (35.88-36.53 tok/s tonight against a 36.93 record), so the board did not
get faster.

## 2. Mechanism

`bench/prefill.sh` and `bench/run-best.sh` both start the server with `--load-mode none`, so the
18.12 GB GGUF is **mmap'd and faulted in lazily**. Prefill at `-ub 32` streams *all* of the model's
weights for every 32 tokens - that is the whole reason its budget is 6.6 s per 590 tokens - so a
prefill request is a sequential read of the entire file as seen by the page cache.

Measured storage bandwidth on the same device, 1.5 GiB direct read:
**`/dev/mmcblk2p1`, 88.1 MB/s**. A completely cold 18.12 GB therefore costs **~206 s** on top of
the ~104 s of GPU work.

So the recorded 120.3 s is not a property of the kernel: it is what this configuration reads when
part of the working set is *not* resident and has to come back over the SD card. Three 79.4-79.5
reps in a row (rather than one slow first rep and two fast ones) point at a cache that is being
re-evicted continuously, which is exactly what the boot `llama-server.service` would do - it holds
24.5 GB, and 24.5 + 18.1 does not fit in 30.9 GB of RAM.

## 3. Prediction, written before the run

`sync` then `echo 3 > /proc/sys/vm/drop_caches`, then `LENS=8192`, `REPS=2`, arm A:

- **rep 1 (cold): TTFT 250-320 s, 30-38 tok/s.** 104 s of GPU work plus 18.12 GB at 88.1 MB/s =
  206 s, i.e. ~310 s total, and the page-fault path may not stream perfectly.
- **rep 2 (warm): 103-105 s, ~92 tok/s**, i.e. tonight's value reproduced to within 1%.
- The delta between the two reps is entirely storage, so it should scale with the file size and not
  with prompt length.

If rep 1 instead lands at 92 tok/s, then `--load-mode none` is not faulting (the mapping is hot
from the previous arm's run and drop_caches did not evict it) and the 15.7% discrepancy has no
explanation yet - which would then have to be reported as an open anomaly rather than a mechanism.

## 3a. Result: falsified. The file is not re-read during prefill.

```
sync; echo 3 > /proc/sys/vm/drop_caches        # buff/cache 9477 -> 1477 MB
cold-A,8192,...,9563,109301.7 ms,87.55 tok/s   (rep 1, cold)
cold-A,8192,...,9563,103979.0 ms,92.02         (rep 2, warm)
```

A cold page cache costs **5.1%**, not the predicted 200 s. So `--load-mode none` still copies every
tensor into its backend buffer at start-up: the 18.12 GB is read once when the server loads and
prefill afterwards reads it from device memory, never from `/dev/mmcblk2p1`. The storage path is
therefore not in TTFT at all, and hypothesis 2 above is dead.

The 5.1% that *is* there is the tail of start-up page faults (the load is still in progress when
`/health` returns), plus the 5.2 s first-request graph capture which rep 1 also pays.

## 3b. What the 15.7% therefore is, and what it is not

Not storage, not the binary, not the flags, not the prompt, not the board clock (decode moved the
other way, 36.9 -> 36.5 tok/s). The remaining candidate is the one quantity prefill is actually
bound by: the **achievable DRAM read rate**, which `bench/tlbstream-occ.cu` measures at 92 GB/s for
this pattern tonight. Prefill moved 79.5 -> 92.0 tok/s, a factor of 1.157, and if the earlier
session's achievable rate for the same pattern was 1/1.157 = 86% of tonight's - 79 GB/s instead of
92 - the whole gap is explained with no other term. That is consistent with the shape of the data
(decode is L1/`lg_throttle`-bound and did not move; prefill is the DRAM-bound workload) but it is
**not verifiable retroactively**, because no pure-load probe was run in the earlier session.

**Consequence for the repo, applied in the same pass:** every absolute prefill/TTFT claim is
conditional on the session's achievable DRAM rate, which has been observed to vary by 15.7%. The
documented TTFT at 9,563 tokens becomes **104 s (warm, service stopped, this session) with 120.3 s
standing as the worse-state figure**, not a single number. Only same-session A/B deltas are
quotable for prefill, and a prefill win below ~1% is not resolvable in one session regardless of
arm count.

## 4. Why it matters beyond bookkeeping

If the prediction holds:

1. Every TTFT number in this repo is conditionally on the weights being resident. The doc-level
   number must be stated as "~104 s with the model's pages in RAM, up to ~310 s cold", not 120.2 s.
2. **A warm-up request at load time moves the whole cold cost out of the first user request.**
   One 32-token prefill touches every expert of every layer, so it is sufficient to pull the file
   into the page cache - the user's first real request then pays 104 s, not 310 s.
3. `--load-mode` other than `none` (`mlock`, `mmap+mlock`, eager) would pin the weights, at the cost
   of a longer start-up and RAM that cannot be shared. Not to be changed without measuring, because
   pinning 18 GB against a 30.9 GB budget is what the boot service already does to this board.

## 3c. The DRAM-rate candidate in 3b is itself falsified (02:21)

Section 3b left one candidate alive: that the earlier session simply had a lower achievable DRAM
read rate. It was tested directly by asking whether the *machine* moved at all between the two
states, with the same probe and the same prompt point.

| quantity | 00:5x state (92.0 tok/s at 9,563) | 01:57-02:21 state (76.5 tok/s at 9,563) |
|---|---|---|
| `bench/tlbstream 18 20`, chunked | 93.1 / 92.7 / 93.1 / 93.1 GB/s | 93.2 / 92.7 / 93.1 / 93.1 GB/s |
| `bench/tlbstream 18 20`, contiguous | 102.2-102.9 GB/s | 102.2-102.9 GB/s |
| GPU `devfreq` cur / min / max | - (not read) | 1,377,000,000 / 1,377,000,000 / 1,377,000,000 |
| `nvpmodel` | MAXN | MAXN |
| load average / top CPU process | - | 0.60 (falling), nothing above 1.8% |

The pure-load rate for the exact 18 GB chunked pattern is unchanged to 0.1%, the GPU is pinned at
its 1377 MHz maximum, the power mode is MAXN and the host is idle - while prefill at the same
token count fell 17% with a binary that was shown equivalent by an interleaved A/new/new/A run
(76.52 / 75.90 / 76.16 / 75.63 tok/s at n_prompt 9,563, so a 1.2% spread against a 17% effect).
Storage is out (section 3a, 5.1%), the binary is out, flags are out, clocks are out.

**So the drift is inside the served process, and the surviving candidate is the physical placement
of the 18 GB model allocation** - which pages of system RAM the weights land on, and therefore how
the GPU's scattered expert reads map to DRAM banks and TLB entries. It is allocated afresh at every
server start, it is not controlled by anything in this repo, and it is not visible with the tools
available here (`tegrastats` reports no EMC clock; there are no `dram__` counters on `gv11b`). Two
server starts twenty minutes apart differ by up to 17% against an in-session A/A floor of 0.56% at
9,563 tokens.

**Consequence, unchanged in direction and now the only defensible framing:** prefill and TTFT are
quoted as a *range* (76-92 tok/s at 9,563 tokens, TTFT 104-125 s) and only same-session,
same-server-start, interleaved A/B deltas are quotable. The 120.2 s in `KNOWN-GOOD.md` and
`README.md` remains the conservative documented figure. Do not compare a prefill arm measured in
one server load with an arm measured in another, and do not spend more time root-causing the
absolute level: three candidate causes (storage, clocks, DRAM rate) have each been measured and
eliminated, and none of the three is a lever this project can pull.
