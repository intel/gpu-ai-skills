---
name: vtune-gpu-hotspots
description: "Run and interpret Intel VTune Profiler gpu-hotspots analysis: inside-the-kernel GPU performance on Intel GPUs, including XVE/EU activity and stalls, occupancy, GPU memory bandwidth and caches, SIMD use, stall-sampling source analysis, and per-kernel (computing-task) hotspots, with a bytes / time check against peak before any sampled metric. Use when the user wants VTune and asks which GPU kernels take the time or why a kernel is slow, usually after vtune-xpu-offload showed the device is the limit. Triggers on: vtune gpu-hotspots, -collect gpu-hotspots, GPU kernel slow in VTune, XVE stalls, EU occupancy, VTune GPU bandwidth. For torch.profiler traces use torch-xpu-profile; for unitrace use xpu-profile-unitrace; for a vLLM server use vtune-vllm."
---

# VTune GPU Compute / Media Hotspots

Looks *inside* GPU kernels: are the execution units busy, and if not, what are they
waiting for. Run this after `vtune-xpu-offload` has established that the device — not
the host and not the transfers — is the limiter.

- Knobs, modes, reports: `references/knobs-and-reports.md`
- Checked against VTune 2026.6 on Arc Pro B70 in an A/B test with and without this skill. Run plain
  `vtune` commands; check knob and report names with `vtune -help collect gpu-hotspots`
  and `vtune -help report <name>`.

---

## 1. When to use

- `xpu-offload` showed **GPU utilization > 80%** and the wall clock is still too
  long.
- One or a few compute tasks dominate GPU time and you need to know why.
- The user asks about EU occupancy, stalls, SIMD efficiency, or GPU cache behaviour.
- You need per-source-line or per-instruction attribution inside a kernel.
- Media / rendering pipelines where fixed-function and EU work must be separated.

**Do not use it when:**

- The GPU is idle or under-used → `vtune-xpu-offload` first. Optimizing a kernel that
  runs 12% of the time is wasted effort.
- Transfers dominate → `vtune-xpu-offload`.
- You do not yet know which kernel matters → get that from `xpu-offload`, then come
  back with `computing-tasks-of-interest`.

## 2. Preconditions

- Intel GPU with the driver stack in place, and **GPU performance metrics access**.
  On Linux this commonly needs `dev.i915.perf_stream_paranoid=0` (i915) or
  `dev.xe.observation_paranoid=0` (xe) and membership in the render/video group. Run
  `vtune-self-checker.sh` (in the VTune `bin64` folder) first and report the exact gate
  if metrics are unavailable — do not present empty columns as "no stalls".
- **Leave start-up out.** Model loading, compilation and warm-up dominate a whole-run
  profile and make the GPU look idle. Start with `-start-paused` and resume around the
  region you care about (ITT `__itt_resume`, or `vtune -command resume -r <dir>`).
  Summary percentages are averaged over the whole collection, paused time included.
- **Time it once without VTune.** The profiled run is slower (tracing adds host time per
  kernel launch). Compare speeds and compute "GPU busy" against the un-profiled time.
- **Two-phase workflow**: characterization mode first (cheap, whole-application),
  then source-analysis mode on one named kernel (expensive, targeted).
- Device debug info in the kernel binary for source-level attribution. JIT library
  kernels (oneDNN, Triton) have none.

## 3. Collect

Phase 1 — characterization (find the expensive kernel and its stall profile):

```bash
# Overview: EU activity, occupancy, stall reasons, GPU bandwidth
vtune -collect gpu-hotspots -r <dir> -- ./sycl_app

# With host memory bandwidth context
vtune -collect gpu-hotspots -knob collect-memory-bandwidth=true -r <dir> -- ./app

# Instruction-count characterization instead of the overview metric set
vtune -collect gpu-hotspots -knob characterization-mode=instruction-count -r <dir> -- ./app
```

Phase 2 — source analysis on the kernel identified in phase 1. Start with
`stall-sampling`: on VTune 2026.6, `bb-latency` and `characterization-mode=instruction-count`
crashed on oneDNN JIT kernels (exit 4, "GTPin ERROR", "Data file is corrupted"), while
`stall-sampling` worked.

```bash
# Stall sampling: stall counts per kernel (and per line when source is available)
vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis \
      -knob source-analysis=stall-sampling \
      -knob computing-tasks-of-interest=<kernel_name> -r <dir> -- ./app

# Basic-block latency: which regions of the kernel cost the most (own SYCL kernels)
vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis \
      -knob source-analysis=bb-latency \
      -knob computing-tasks-of-interest=<kernel_name> -r <dir> -- ./app

# Memory latency: which accesses stall
vtune -collect gpu-hotspots -knob gpu-profiling-mode=source-analysis \
      -knob source-analysis=mem-latency \
      -knob computing-tasks-of-interest=<kernel_name> -r <dir> -- ./app
```

Additional variants:

```bash
# Finer stall sampling for short kernels
vtune -collect gpu-hotspots -knob gpu-sampling-interval=0.5 \
      -knob stall-reasons-sampling-interval=0.0002 -r <dir> -- ./app

# Steady state of a long run (resume from the program or with vtune -command resume)
vtune -collect gpu-hotspots -start-paused -r <dir> -- ./train.py
```

Power and frequency: on VTune 2026.6, `-knob analyze-power-usage=true` is accepted but
gpu-hotspots records no power data ("plugin [power] was disabled because it doesn't
support the given collection type"). Read the GPU's own counters instead: sysfs
`.../gt0/freq0/act_freq`, `.../throttle/reasons`, hwmon `energy1_input`, or `xpu-smi stats`.

## 4. Knobs (VTune 2026.6 — re-verify with `vtune -help collect gpu-hotspots`)

| Knob | Default | Values | When to change it |
|---|---|---|---|
| `gpu-profiling-mode` | `characterization` | `characterization`, `source-analysis` | `source-analysis` only after you know which kernel to dissect |
| `characterization-mode` | `overview` | `overview`, `global-memory-accesses`, `compute-extended`, `instruction-count`, `video-decode`, `video-enhancement` | `instruction-count` to see SIMD width usage and instruction mix (may crash on JIT library kernels) |
| `source-analysis` | `bb-latency` | `bb-latency`, `mem-latency`, `stall-sampling` | Start with `stall-sampling`; `mem-latency` when the stall profile points at memory |
| `computing-tasks-of-interest` | (none) | `name` or `name#start#step#stop` | Required in practice for source-analysis — restricts the expensive instrumentation |
| `gpu-sampling-interval` | `1` (ms) | 0.005–10 | Lower for short kernels; raise on long runs to cut overhead |
| `target-gpu` | all GPUs | `0:<bus>:<dev>.<fn>` from `vtune -help collect gpu-hotspots` | Set to the GPU the program uses on multi-GPU hosts |
| `stall-reasons-sampling-interval` | `0.0005` (ms) | 0.0001–2 | Lower for more stall detail on brief kernels; raising it reduces overhead |
| `dump-compute-task-binaries` | `true` | `true`, `false` | Keep on for assembly-level views; `false` to shrink results |
| `collect-programming-api` | `true` | `true`, `false` | Leave on to correlate kernels with host API calls |
| `collect-memory-bandwidth` | `false` | `true`, `false` | `true` for host DRAM context (uncore access needed) |
| `analyze-power-usage` | `false` | `true`, `false` | No power data with gpu-hotspots on 2026.6 (see §3) |

Source-analysis mode is **significantly** more intrusive than characterization. Always
pair it with `computing-tasks-of-interest`.

## 5. Reports

```bash
# GPU metric summary: EU activity, occupancy, stalls, bandwidth
vtune -report summary -r <dir> -format csv

# Per-kernel table. -limit alone keeps rows in no particular order and can drop the
# top kernel; always sort by time when you limit.
vtune -report hotspots -r <dir> -group-by computing-task \
      -sort-desc "Computing Task:Total Time" -format csv -limit 25

# Source lines inside a kernel (source-analysis result)
vtune -report hotspots -r <dir> -source-object function=<kernel_name> \
      -source-search-dir <src> -format csv

# Assembly-level view
vtune -report hotspots -r <dir> -source-object function=<kernel_name> \
      -group-by address -format csv

# Raw GPU counters per kernel (without -group-by it returns a header only)
vtune -report hw-events -r <dir> -group-by computing-task -format csv

# Steady-state window only (-report timeline -format csv fails on 2026.6:
# "column-by is missing"; use -time-filter for a time window instead)
vtune -report hotspots -r <dir> -time-filter 30.0:60.0 \
      -group-by computing-task -format csv

# A/B a kernel change
vtune -report summary -r <baseline> -r <candidate> -format csv
```

## 6. Interpretation

**Step 1 — check the kernel against peak before reading any sampled metric.** VTune's
per-kernel EU, occupancy and bandwidth values are sampled (1 ms by default). For kernels
shorter than the interval they are diluted by the gaps between kernels and read low.
Compute the achieved rate yourself:

- bandwidth = bytes the kernel must move (e.g. weight bytes for a GEMV) ÷ its total time
  from the per-kernel table, against the device's DRAM peak;
- FLOP rate = 2 × M × N × K ÷ time, against the device's compute peak for the dtype.

Example (Qwen3-8B decode GEMV on Arc Pro B70): VTune showed Idle 61%, occupancy 38% and
207 GB/s, while bytes ÷ time was 559 GB/s, 92% of the 608 GB/s peak. The kernel was
memory-bound and near its limit; the Idle/occupancy rows below did not apply.

If the kernel is above ~80% of the relevant peak, say that kernel tuning can gain little
and point at the work itself: fewer bytes (lower precision weights), more work per byte
(batching several requests), or fewer and larger kernels (fusing the many small ones).

**Step 2 —** read EU activity and occupancy only for kernels that are not near a peak —
they split the problem in two.

| Signal | Threshold | Verdict | Fix direction |
|---|---|---|---|
| EU Array Active | > 80% | EUs genuinely busy | Reduce work: better algorithm, fewer instructions, wider SIMD |
| EU Array Stalled | > 30% | Threads resident but waiting | Read the stall reasons below |
| EU Array Idle | > 30% | Not enough threads resident (only if Step 1 shows the kernel is not near peak) | Increase occupancy: more work-items, smaller per-thread resource use |
| Occupancy | < 50% | Under-populated EUs (only if Step 1 shows the kernel is not near peak) | Larger global range, smaller work-group resource footprint (registers, SLM) |
| SIMD width utilization low | — | Divergence or narrow SIMD compilation | Reduce branch divergence; check the chosen SIMD width |
| GPU memory bandwidth near peak (bytes ÷ time, Step 1) | > 80% | Device-memory-bound | Library kernels at this level are done: reduce bytes (precision) or batch. Own kernels: improve locality, SLM, coalescing |
| L3 / GPU cache hit rate low | — | Poor access locality | Tile the computation; coalesce neighbouring work-items onto neighbouring addresses |
| GPU frequency below nominal | — | Throttling | Check power/thermal; results are not comparable to datasheet peaks |
| Fixed-function units saturated (media) | — | Not an EU problem | Rebalance between EU and fixed-function paths |

Stall reasons, and what each usually means:

| Stall reason | Meaning | Fix direction |
|---|---|---|
| Memory / send pipeline | Waiting on loads or stores | Coalesce accesses, use SLM, prefetch, increase arithmetic intensity |
| SBID / dependency | Waiting on a previous instruction's result — **including memory loads**. A GEMV that only streams memory at 590 GB/s (near peak) had SBID as its top stall (72%) | Check bandwidth first (Step 1). If near peak, SBID is memory wait: reduce bytes. Otherwise: more independent work per thread (unroll), reorder for ILP |
| Barrier / sync | Work-group synchronization | Fewer barriers, smaller work-groups, restructure the algorithm |
| Instruction fetch | Kernel code footprint | Smaller kernels, less aggressive inlining/unrolling |
| Control flow | Divergent branches | Restructure so a sub-group takes one path; use select over branch |
| Shared local memory | SLM bank conflicts or capacity | Pad to avoid conflicts; reduce SLM per work-group |

**Occupancy is not the goal.** High occupancy with high memory stalls just means more
threads waiting. Fix the access pattern first; raise occupancy when the stall is
latency that more threads can hide.

## 7. Next analysis

| Finding | Next |
|---|---|
| Kernel is efficient; GPU still not the whole story | Back to `vtune-xpu-offload` for host/transfer balance |
| Device memory bandwidth-bound, own kernel | Stay here: tiling, SLM, access coalescing; re-measure |
| Kernel already near roofline (library GEMM/GEMV) | Algorithmic change, lower precision, batching; no kernel tuning |
| Many small kernels between the big ones | Fuse them (e.g. `torch.compile`), fewer launches |
| Throttling suspected | GPU sysfs / hwmon counters or `xpu-smi stats` (see §3) |
| The workload is an LLM server (vLLM) and the trace is full of load/tokenizer noise | `vtune-vllm` — ROI profiling, prefill/decode isolation |

## 8. Common failures

| Symptom | Cause | Response |
|---|---|---|
| All GPU metric columns empty | GPU metrics access not granted | Report the exact gate (`dev.i915.perf_stream_paranoid` / `dev.xe.observation_paranoid`, render group); do not read empty as zero |
| Stall reasons missing | Metrics unsupported on this GPU generation, or interval too coarse | Run `vtune-self-checker.sh`; lower `stall-reasons-sampling-interval` |
| Source view shows no lines | No device debug info, or wrong `-source-search-dir` | Rebuild with device debug info; point at the source tree. JIT library kernels (oneDNN) have no source or assembly view ("Assembly code ... is not available") |
| Collection takes far longer than the app | `source-analysis` without a kernel filter | Add `computing-tasks-of-interest=<kernel>` |
| `bb-latency` / `instruction-count` run exits 4, "GTPin ERROR" | Instrumentation fails on JIT library kernels | Use `source-analysis=stall-sampling` |
| Top kernel missing from the kernel table | `-limit` without a sort | Add `-sort-desc "Computing Task:Total Time"` |
| Kernel name not matching `computing-tasks-of-interest` | Mangled or templated name | Copy the exact name from the `computing-task` report |
| Numbers differ run to run | Frequency variation, other GPU tenants | Repeat; report ranges; read `act_freq` and throttle reasons in sysfs |
| Result huge | Binaries dumped for many kernels | `-knob dump-compute-task-binaries=false` once you no longer need assembly |
| Metrics missing in a container | Render device not exposed | Pass `/dev/dri` through; verify group membership inside the container |

## Limitations and boundaries

- Looks inside GPU kernels only. Whether the time goes to the host, to transfers or to
  the device is `vtune-xpu-offload`; vLLM servers are `vtune-vllm`.
- VTune's per-kernel EU, occupancy and bandwidth values are 1 ms samples. On sub-ms
  kernels they read low; do not tune on them without the Step 1 check in §6.
- JIT library kernels (oneDNN, Triton) have no source or assembly view; source analysis
  stops at the per-kernel stall mix (`stall-sampling`).
- Power and frequency are not collected by gpu-hotspots on VTune 2026.6 (§3).
- GPU metrics need access settings (`dev.xe.observation_paranoid`, render group). Ask the
  user before changing any system setting, and confirm before running a collection that
  restarts their workload. Do not install packages or change kernel settings without
  user approval.
