---
name: vtune-xpu-offload
description: "Run and interpret Intel VTune Profiler xpu-offload analysis (formerly gpu-offload): whether time goes to the host, to CPU-GPU transfers, or to the device, with GPU busy computed against un-profiled timing, per phase. Covers compute-task counts and duration, transfer size and time, kernel launch and sync overhead, for SYCL, OpenCL, Level Zero, OpenMP offload and PyTorch XPU. Use when the user wants VTune to find whether the GPU, the CPU or data copies are the bottleneck, as the first VTune GPU analysis. Triggers on: vtune xpu-offload, -collect xpu-offload, gpu-offload, VTune GPU utilization, host to device transfer in VTune, kernel launch overhead. For torch.profiler use torch-xpu-profile; inside a kernel use vtune-gpu-hotspots; vLLM servers use vtune-vllm."
---

# VTune XPU Offload

The first GPU-side analysis. It answers a single structural question: **is the time
going to the host, to data transfer, or to the device?** Only after that is it worth
looking inside kernels.

- Knobs, reports, transfer/queue recipes: `references/knobs-and-reports.md`
- Checked against VTune 2026.6 on Arc Pro B70 in an A/B test with and without this skill. Run plain
  `vtune` commands; check knob and report names with `vtune -help collect xpu-offload`
  and `vtune -help report <name>`.

---

## 1. When to use

- Any application that offloads to an Intel GPU or accelerator and is slower than
  expected.
- SYCL / DPC++ / OpenCL / Level Zero / OpenMP target / oneAPI codes.
- PyTorch, IPEX, or vLLM on XPU where GPU utilization is unknown or low.
- The user asks whether the GPU is being kept busy, or whether transfers dominate.
- `performance-snapshot` or `hpc-performance` showed GPU activity worth pursuing.

**Do not use it when:**

- You already know the device is busy and the kernels themselves are slow →
  `vtune-gpu-hotspots`.
- There is no accelerator in the picture.
- The host side is the obvious limit and no device work exists → a CPU profile
  (`vtune -collect hotspots`).

## 2. Version and tooling notes — read before building a command

Traps, verified on VTune 2026.4 and 2026.6:

1. **`-collect gpu-offload` no longer exists.** It was renamed. Running it fails:
   use `xpu-offload`.
2. **`-collect npu` no longer exists.** The error message suggests
   `xpu-offload -knob profile-npu=true`, but **that knob does not exist** on
   2026.4 or 2026.6 (`Cannot find knob profile-npu`). Do not paste it into a command.
   Check `vtune -help collect xpu-offload` on the installed version before promising
   NPU profiling.
3. **The profiled run is slower than the real one, and only on the host side.** Tracing
   adds host time to every kernel launch. On a PyTorch decode step with ~245 small
   kernels the step went from 4.26 ms to 7.83 ms under xpu-offload, so the GPU looked
   44% busy while it is 81% busy without VTune. **Always time the same work once
   without VTune** (wall clock or `torch.xpu.synchronize()` around the region) and use
   that time in every busy % and speed-up estimate (§7).

Use an absolute, descriptive result directory, and never reuse an existing one.

## 3. Preconditions

- Intel GPU with a working driver stack; GPU metrics need the i915/Xe driver and, on
  Linux, appropriate access to the device. Verify with `vtune-self-checker.sh` (in the
  VTune `bin64` folder).
- **GPU metrics may require permission setup** (membership in the `render`/`video`
  group, `dev.i915.perf_stream_paranoid=0` on i915 or `dev.xe.observation_paranoid=0` on
  xe — check the self-checker output before assuming a driver bug).
- Level Zero or OpenCL runtime present for API-level tracing.
- Build with device symbols / debug info if you want kernel source attribution later.
- For frameworks (PyTorch XPU, vLLM), know which device the work should land on and
  confirm it is not silently falling back to CPU — that is a frequent "GPU is slow"
  root cause that needs no profiler.

## 4. Collect

```bash
# Standard offload pass
vtune -collect xpu-offload -r <dir> -- ./sycl_app

# Add host-side CPU sampling to see what the host is doing while the GPU waits
vtune -collect xpu-offload -knob collect-cpu-sampling=true -r <dir> -- ./sycl_app

# Host stacks for kernel-submission attribution
vtune -collect xpu-offload -knob collect-cpu-sampling=true \
      -knob enable-stack-collection=true -r <dir> -- ./sycl_app

# Host call stack of each GPU task and copy (2026.6)
vtune -collect xpu-offload -knob enable-tasks-stack-collection=true -r <dir> -- ./app

# Add host memory bandwidth (needs VTune's sampling driver; without it the collection
# fails: "Failed to connect to PMU reservation service (PAX)" — load it with insmod-sep)
vtune -collect xpu-offload -knob collect-memory-bandwidth=true -r <dir> -- ./app

# ITT-instrumented phases only (framework or user tasks)
vtune -collect xpu-offload -knob enable-user-tasks=true -r <dir> -- ./app

# Narrow to the steady-state region of a long inference/training run
vtune -collect xpu-offload -start-paused -r <dir> -- ./app
#   then: vtune -command resume -r <dir> ... vtune -command stop -r <dir>
```

## 5. Knobs (VTune 2026.6 — re-verify with `vtune -help collect xpu-offload`)

| Knob | Default | Values | When to change it |
|---|---|---|---|
| `collect-cpu-sampling` | `false` | `true`, `false` | `true` when the GPU is idle and you need to know what the host is doing |
| `enable-stack-collection` | `false` | `true`, `false` | `true` to attribute kernel submissions and transfers to host call sites |
| `enable-tasks-stack-collection` | `false` | `true`, `false` | `true` to see the host call stack behind each GPU task and copy |
| `collect-memory-bandwidth` | `false` | `true`, `false` | `true` when host↔device transfers are large; needs VTune's sampling driver (PAX) |
| `collect-cpu-gpu-bandwidth` | `false` | `true`, `false` | Host-device bandwidth |
| `collect-programming-api` | — | `true`, `false` | Trace DPC++ / OpenCL / Level Zero API calls |
| `enable-characterization-insights` | `false` | `true`, `false` | Add GPU hardware metrics |
| `analyze-power-usage` | `false` | `true`, `false` | Needs the SoC Watch driver; without it: "Unable to start analysis. Socwatch driver cannot be found" |
| `enable-user-tasks` | `true` | `true`, `false` | Leave on — ITT tasks give phase names (framework ops, user regions) |
| `itt-domains` | `all` | domain list | Restrict to specific ITT domains when framework instrumentation is noisy |
| `target-gpu` | all GPUs | from `vtune -help collect xpu-offload` | Multi-GPU hosts only: set to the GPU the program uses |

There is no `profile-npu` knob on 2026.6 despite the deprecation message for
`-collect npu`.

Power and frequency without SoC Watch: the GPU driver exposes them. On xe: sysfs
`/sys/class/drm/card*/device/tile0/gt0/freq0/act_freq` and `.../throttle/reasons`, hwmon
`energy1_input`; or `xpu-smi stats`.

## 6. Reports

```bash
# Elapsed time, totals. On 2026.6 there is no GPU-busy line, and "% of Elapsed Time"
# divides by the whole collection, paused time and model load included.
vtune -report summary -r <dir> -format csv

# GPU compute time vs transfer time: row "Compute" = total GPU compute time
vtune -report hotspots -r <dir> -group-by computing-task-purpose -format csv

# Phase times from the program's ITT tasks (e.g. prefill, decode_step) and
# Level Zero API time per call type
vtune -report hotspots -r <dir> -group-by task-domain,task -format csv

# Per-kernel / per-computing-task view — the primary table for this analysis
vtune -report hotspots -r <dir> -group-by computing-task \
      -sort-desc "Computing Task:Total Time" -format csv -limit 25

# Host functions (requires collect-cpu-sampling=true)
vtune -report hotspots -r <dir> -group-by function -format csv -limit 25

# Narrow to the steady-state window
vtune -report hotspots -r <dir> -time-filter 30.0:60.0 -group-by computing-task -format csv

# Which host thread submits what
vtune -report hotspots -r <dir> -group-by thread -format csv

# Submission call paths (stacks required)
vtune -report callstacks -r <dir> -format csv -limit 25

# Before / after a batching or USM change
vtune -report summary -r <baseline> -r <candidate> -format csv
```

Note: there is **no** `tasks` or `top-tasks` report type on 2026.x. Use
`-report hotspots -group-by computing-task`. Confirm available groupings with
`vtune -report hotspots -r <dir> -group-by=?` (works); `vtune -report ?` and
`-report-knob=?` do not (use `vtune -help report hotspots`). `-report timeline -format csv`
fails on 2026.6 ("column-by is missing"); use `-time-filter` for a window.

Python frameworks: host stacks stop in the interpreter (`[Skipped stack frame(s)]`,
`_PyEval_EvalFrameDefault`) and do not show Python source lines; the libtorch frames
(e.g. `THPVariable_tensor` → `_copy_xpu`) still name the op.

## 7. Interpretation

**First compute GPU busy correctly, per phase:**

- GPU busy = GPU compute time in the region (row "Compute" of
  `-group-by computing-task-purpose`, or the sum of kernel times) ÷ the **un-profiled**
  wall time of the same region (§2 trap 3). Dividing by the profiled time understates it,
  most for many short kernels.
- Split by phase before giving one figure: ITT tasks (`-group-by task-domain,task`) or
  separate runs. Prefill and decode differ (Qwen3-8B on B70: ~100% vs 81% busy).
- Device-to-device copies (KV-cache `torch.cat`: `CatArrayBatchedCopy*`, dtype
  `CopyScalarFunc`) are GPU **compute** tasks, not CPU ↔ GPU transfers. They are missing
  from the transfer columns; count them as kernels.

Then decide which of four situations you are in, and act:

| GPU busy (un-profiled) | Transfer share | Host share | Situation | Action |
|---|---|---|---|---|
| Low (< 50%) | Low | High | **Host-bound** — the GPU is starved | Profile the host (`collect-cpu-sampling=true`); fewer launches (graphs, `torch.compile`, fusion), overlap submission with compute |
| Low (< 50%) | High (> 30%) | Low | **Transfer-bound** | Fewer/larger transfers, USM shared or device memory, keep data resident, overlap copy with compute |
| Low (< 50%) | Low | Low | **Latency/serialization-bound** | Gaps between kernels: use out-of-order queues, multiple queues, remove per-kernel syncs |
| High (> 80%) | Low | Low | **Device-bound** | The GPU is genuinely the limit → `vtune-gpu-hotspots` |
| 50-80% | Low | Some | **Mixed** | Both apply; give the ceiling of each (Amdahl below) |

If the busy % from the profiled run and from the un-profiled time fall in different rows,
use the un-profiled one and tell the user that VTune's tracing made the GPU look idle.

Additional signals:

| Signal | Threshold | Verdict | Fix direction |
|---|---|---|---|
| Many very short compute tasks | avg duration < ~10 µs | Launch overhead dominates | Fuse kernels, batch work, increase per-kernel work |
| Kernel count enormous vs total GPU time | — | Submission overhead | Kernel fusion; reduce graph granularity |
| Idle gaps between consecutive kernels | — | Serialized submission / implicit sync | Out-of-order queue, avoid `wait()` per kernel, use events |
| Transfer time ≈ compute time | — | No overlap | Double-buffer; async copies on a separate queue |
| Same buffer copied repeatedly | — | Data not kept resident | Allocate device-resident memory, transfer once |
| Host DRAM bandwidth saturated during transfers | > 80% | Host memory is the transfer limit | Pinned/USM host memory, fewer staging copies |
| Device frequency below nominal | — | Thermal/power throttling | Read sysfs `act_freq` / `throttle/reasons` or `xpu-smi stats`; cooling, power cap |
| GPU utilization high but throughput low | — | Busy doing inefficient work | `vtune-gpu-hotspots` (occupancy, stalls, memory) |

**Amdahl on the device.** If 40% of elapsed time is host work, making the GPU
infinitely fast yields at most 1.67×. State the ceiling before recommending kernel
optimization — computed from un-profiled times.

## 8. Next analysis

| Finding | Next |
|---|---|
| Device-bound, kernels are the cost | `vtune-gpu-hotspots` |
| Host-bound | `-collect hotspots` on the host, or `collect-cpu-sampling=true` here |
| Device throttling suspected | GPU sysfs / hwmon counters, `xpu-smi stats` |
| Everything balanced, still too slow | Algorithmic: larger batches, better kernels, mixed precision |
| The workload is a vLLM / OpenAI-compatible LLM server | `vtune-vllm` for the ROI setup, then `vtune-gpu-hotspots` for the kernels |

## 9. Common failures

| Symptom | Cause | Response |
|---|---|---|
| `-collect gpu-offload` fails | Renamed on 2026.x | Use `xpu-offload` |
| `-collect npu` fails | Removed | No working replacement knob on 2026.x; check `vtune -help collect xpu-offload` |
| `Cannot find knob profile-npu` | The suggested knob does not exist | Do not use it; report the limitation |
| `analyze-power-usage`: "Socwatch driver cannot be found" | SoC Watch driver not installed | Use sysfs / hwmon / `xpu-smi` for power and frequency |
| `collect-memory-bandwidth`: "Failed to connect to PMU reservation service (PAX)" | VTune sampling driver not loaded | `vtune-self-checker.sh`; load it with `insmod-sep`, or drop the knob |
| No GPU data at all | Driver/permission issue, or work never reached the device | `vtune-self-checker.sh`; verify the app actually selected the GPU device |
| GPU looks half idle, but un-profiled it is fast | Tracing overhead on many short kernels | Use the un-profiled time (§7) |
| No host functions in the report | `collect-cpu-sampling=false` | Re-collect with it `true` |
| No `computing-task` grouping | Runtime not traced (no Level Zero/OpenCL interception) | Confirm the runtime and that the app links it dynamically |
| Kernel names unreadable / mangled | Missing device debug info | Rebuild with device debug info for source-level work |
| Result dominated by framework startup | Whole run collected | `-start-paused` + ITT resume around steady state |
| Metrics missing inside a container | Device not exposed, or perf paranoid setting | Pass the render device into the container; check the paranoid sysctl. Capabilities are fixed at `docker run` and cannot be added later |

## Limitations and boundaries

- Answers host vs transfer vs device. It does not explain why a kernel is slow: that is
  `vtune-gpu-hotspots`. vLLM servers: `vtune-vllm` for the ROI setup.
- Tracing slows the host side of every launch; busy % and speed-ups need un-profiled
  timing of the same work (§2 trap 3, §7).
- No NPU analysis on VTune 2026.x (`-collect npu` and `profile-npu` do not exist).
- `analyze-power-usage` needs the SoC Watch driver and `collect-memory-bandwidth` the
  VTune sampling driver; without them the collection does not start.
- Python host stacks stop in the interpreter; Python source lines are not shown.
- Ask the user before changing system settings (paranoid sysctls, drivers) or installing
  packages, and confirm before re-running their workload under the profiler.
