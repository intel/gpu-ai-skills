# gpu-hotspots — Knobs, Modes and Reports

From `vtune -help collect gpu-hotspots` (checked on VTune 2026.6). Re-verify with
`vtune -help collect gpu-hotspots` on the installed version.

---

## 1. The two-phase workflow

This analysis has two distinct modes, and using them in the wrong order wastes a lot
of time.

| | Phase 1: `characterization` | Phase 2: `source-analysis` |
|---|---|---|
| Scope | Whole application, all kernels | One or a few named kernels |
| Cost | Moderate | High — heavy instrumentation |
| Answers | Which kernel is expensive, and is it stalled / idle / active | Which lines or instructions inside that kernel cost the most |
| Prerequisite | None beyond GPU metrics access | You must already know the kernel name |
| Knob | `gpu-profiling-mode=characterization` (default) | `gpu-profiling-mode=source-analysis` + `computing-tasks-of-interest` |

Never start with source-analysis. Get the kernel name from
`-report hotspots -group-by computing-task` on a characterization result.

---

## 2. Knobs in detail

### `gpu-profiling-mode` (default `characterization`)

`characterization` samples GPU hardware metrics across the run. `source-analysis`
instruments the kernel to attribute cost to basic blocks or memory instructions,
which requires re-running the workload with much heavier overhead.

### `characterization-mode` (default `overview`)

| Value | Gives you |
|---|---|
| `overview` | EU array active / stalled / idle, occupancy, GPU memory bandwidth, cache metrics, stall reasons |
| `instruction-count` | Instruction mix by SIMD width and type — shows whether the kernel compiled to the SIMD width you expected and how much of the work is control flow versus math. Crashed on oneDNN JIT kernels on 2026.6 (exit 4, "GTPin ERROR") |
| `global-memory-accesses`, `compute-extended` | Memory-access and extended compute metric sets (2026.6) |

Use `instruction-count` when you suspect divergence, an unexpected SIMD width, or
want to compare instruction efficiency across kernel variants.

### `source-analysis` (default `bb-latency`)

| Value | Gives you | Use when |
|---|---|---|
| `bb-latency` | Latency attributed to basic blocks — a cost map over the kernel's control-flow graph | You don't yet know which part of the kernel is expensive |
| `mem-latency` | Latency attributed to individual memory instructions | The stall profile already points at memory |

### `computing-tasks-of-interest` (default: none)

Restricts collection to named kernels. Effectively mandatory for source-analysis:
without it, every kernel is instrumented and the run can take orders of magnitude
longer.

Use the exact name from the `computing-task` report — SYCL and templated kernel names
are long and mangled, so copy rather than retype.

### `gpu-sampling-interval` (default `1` ms, range 0.1–10)

Interval for GPU hardware metric sampling. Kernels shorter than the interval may be
missed entirely: for sub-millisecond kernels use 0.1–0.5 ms. Raise it on long runs to
control overhead and result size.

### `stall-reasons-sampling-interval` (default `0.0005` ms, range 0.0001–2)

Much finer than the metric interval because stall attribution needs many samples per
kernel invocation. Lower it (toward 0.0001) for very short kernels; raise it if
overhead is distorting the measurement.

### `dump-compute-task-binaries` (default `true`)

Stores the compiled GPU binaries in the result so assembly-level views work later.
Leave on while investigating; set `false` for long runs with many kernels when
result size matters and you only need aggregate metrics.

### `collect-programming-api` (default `true`)

Traces the host-side runtime API (Level Zero / OpenCL) so kernels can be correlated
with their submissions. Leave on — it costs little and makes the timeline readable.

### `collect-memory-bandwidth` (default `false`) / `analyze-power-usage` (default `false`)

Host DRAM bandwidth (needs uncore access and VTune's sampling driver) and device
power/frequency respectively. On 2026.6, gpu-hotspots accepts `analyze-power-usage=true`
but records no power data ("plugin [power] was disabled because it doesn't support the
given collection type"). When a long run's performance drifts, read the GPU's sysfs
`act_freq` and `throttle/reasons`, hwmon `energy1_input`, or `xpu-smi stats` instead.

---

## 3. Collection combinations

| Goal | Command shape |
|---|---|
| Find the expensive kernel | `vtune -collect gpu-hotspots -r <dir> -- ./app` |
| SIMD / instruction mix | `-knob characterization-mode=instruction-count` |
| Stall counts inside a kernel (start here) | `-knob gpu-profiling-mode=source-analysis -knob source-analysis=stall-sampling -knob computing-tasks-of-interest=<k>` |
| Cost map inside a kernel (own SYCL kernels) | `-knob gpu-profiling-mode=source-analysis -knob source-analysis=bb-latency -knob computing-tasks-of-interest=<k>` |
| Which loads stall | `-knob gpu-profiling-mode=source-analysis -knob source-analysis=mem-latency -knob computing-tasks-of-interest=<k>` |
| Sub-millisecond kernels | `-knob gpu-sampling-interval=0.1 -knob stall-reasons-sampling-interval=0.0001` (per-kernel values still read low; check bytes ÷ time) |
| Long run, keep result small | `-knob gpu-sampling-interval=5 -knob dump-compute-task-binaries=false` |
| Suspected throttling | GPU sysfs / hwmon counters or `xpu-smi stats` (no power data from gpu-hotspots on 2026.6) |
| Steady state of training | `-start-paused` + `vtune -command resume -r <dir>` |

---

## 4. Report cookbook

```bash
# GPU metric summary
vtune -report summary -r <dir> -format csv

# Kernels ranked by GPU time (-limit without -sort-desc keeps unsorted rows and
# can drop the top kernel)
vtune -report hotspots -r <dir> -group-by computing-task \
      -sort-desc "Computing Task:Total Time" -format csv -limit 25

# Source lines inside the analysed kernel
vtune -report hotspots -r <dir> -source-object function=<kernel> \
      -source-search-dir <src> -format csv

# Assembly / instruction level
vtune -report hotspots -r <dir> -source-object function=<kernel> \
      -group-by address -format csv

# Raw GPU counters per kernel (without -group-by: header only)
vtune -report hw-events -r <dir> -group-by computing-task -format csv

# -report timeline -format csv fails on 2026.6 ("column-by is missing");
# use -time-filter on the hotspots report for a time window

# Restrict to a window
vtune -report hotspots -r <dir> -time-filter 30.0:60.0 \
      -group-by computing-task -format csv

# Compare two kernel implementations
vtune -report summary -r <variant_a> -r <variant_b> -format csv
```

Confirm what a result supports:

```bash
vtune -report hotspots -r <dir> -group-by=?
vtune -report hotspots -r <dir> -report-knob=?
```

---

## 5. Column aliases to match on

| Concept | Aliases |
|---|---|
| EU activity | `EU Array Active`, `EU Array Stalled`, `EU Array Idle`, `XVE Array Active`, `XVE Threads Occupancy`, `EU Threads Occupancy` |
| Occupancy | `Occupancy`, `Thread Occupancy`, `Active Threads` |
| SIMD | `SIMD Width`, `SIMD Utilization`, `SIMD1`, `SIMD8`, `SIMD16`, `SIMD32` |
| GPU memory | `GPU Memory Bandwidth`, `GPU L3 Bandwidth`, `Memory Read Bandwidth`, `Memory Write Bandwidth`, `L3 Cache Hit Rate`, `Sampler Busy` |
| Stalls | `Stalled`, `Send Pipeline`, `SBID`, `Barrier`, `Instruction Fetch`, `Control Flow`, `Shared Local Memory` |
| Kernel time | `GPU Time`, `Average Time`, `Instance Count`, `Total Time` |
| Power | `GPU Frequency`, `Average GPU Frequency`, `GPU Power` |
| Fixed function | `Render/GPGPU Busy`, `Media Engine`, `Sampler`, `VDBOX`, `Blitter` |

Note the naming shift across GPU generations: older reports say **EU**, newer ones say
**XVE** for the same execution units. Match on both. An absent column means the metric
was not collected or is not exposed on this GPU — say so rather than reporting zero.

---

## 6. Diagnosis flow inside a kernel

```
First: bytes ÷ kernel time (or FLOPs ÷ time) vs the device peak.
Above ~80% of peak → the kernel is at its limit; stop here. Reduce the work
(lower precision, batching, fusing small kernels), not the kernel.
Sampled per-kernel EU / occupancy / bandwidth values of sub-ms kernels read low.

EU Array Active > 80%?
├── Yes → EUs are busy. Reduce instruction count.
│         instruction-count mode: is SIMD width as expected?
│         ├── Narrow SIMD / high SIMD1 → divergence; restructure branches
│         └── Expected width → algorithmic reduction, cheaper math, lower precision
└── No → why are they not active?
    ├── EU Array Idle high → not enough resident threads
    │    ├── Occupancy low → global range too small; work-group uses too many
    │    │                   registers or too much SLM
    │    └── Occupancy fine, still idle → submission gaps, not a kernel problem
    │                                     → vtune-xpu-offload
    └── EU Array Stalled high → threads resident but waiting; read stall reasons
         ├── Memory / send → coalesce, SLM tiling, prefetch, raise arithmetic intensity
         ├── SBID / dependency → also counts waits on memory loads: if bytes ÷ time
         │                       is near peak, it is memory; else unroll, reorder for ILP
         ├── Barrier → fewer barriers, smaller work-groups
         ├── Control flow → reduce divergence within the sub-group
         ├── SLM → pad to break bank conflicts, cut SLM per group
         └── Instruction fetch → smaller kernel code footprint
```

---

## 7. GPU optimization moves, ranked by usual payoff

0. **Check the distance to peak first.** If bytes ÷ time or FLOPs ÷ time is already
   near the device peak (common for oneDNN GEMM/GEMV), the moves below gain little;
   say so, and point at precision, batching or fusing the surrounding small kernels.
1. **Coalesce memory access.** Consecutive work-items should touch consecutive
   addresses. This is the single largest lever on most Intel GPU kernels.
2. **Tile through SLM.** Stage reused data in shared local memory instead of
   re-reading global memory; watch for bank conflicts.
3. **Raise arithmetic intensity.** More FLOPs per byte moved — fuse operations so
   intermediates stay in registers.
4. **Fix divergence.** Arrange work so a sub-group follows one path; prefer select
   over branch for cheap alternatives.
5. **Right-size the work-group.** Balance registers and SLM per group against
   occupancy; a smaller footprint lets more threads become resident.
6. **Unroll for ILP.** Give each thread independent instructions so dependency stalls
   overlap.
7. **Reduce precision** where accuracy allows (fp32 → bf16/fp16, int8 for inference).
8. **Reduce barriers.** Each one costs the whole group's slack.

Re-measure after each change with the same command and compare with
`vtune -report summary -r <before> -r <after>`. Report the delta on the kernel *and*
on end-to-end elapsed time — a 2× kernel win on a kernel that is 15% of runtime is
7% overall, and saying so is part of the job.
