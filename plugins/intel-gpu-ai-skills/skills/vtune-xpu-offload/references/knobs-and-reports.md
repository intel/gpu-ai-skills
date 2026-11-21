# xpu-offload — Knobs, Reports, Transfers and Queues

From `vtune -help collect xpu-offload` (checked on VTune 2026.6). Re-verify with
`vtune -help collect xpu-offload` on the installed version.

---

## 1. Version reality check

| Command | Status on 2026.4 / 2026.6 |
|---|---|
| `-collect gpu-offload` | **Removed** — renamed to `xpu-offload` |
| `-collect npu` | **Removed** |
| `-collect xpu-offload -knob profile-npu=true` | Suggested by the deprecation message but the knob **does not exist** |
| `-report tasks` / `-report top-tasks` | **Not a report type** — use `-report hotspots -group-by computing-task` |
| `-report timeline -format csv` | Fails on 2026.6: "the required argument column-by is missing" — use `-time-filter` |
| `vtune -report ?`, `-report-knob=?` | Fail — use `vtune -help report`, `vtune -help report hotspots` |
| `-knob analyze-power-usage=true` | Needs the SoC Watch driver ("Socwatch driver cannot be found") |
| `-knob collect-memory-bandwidth=true` | Needs VTune's sampling driver ("Failed to connect to PMU reservation service (PAX)") |

Always confirm with `vtune -help collect xpu-offload` on the installed version before
quoting a knob to the user. 2026.6 also has `enable-tasks-stack-collection`,
`collect-cpu-gpu-bandwidth`, `collect-programming-api`, `enable-characterization-insights`,
`media-metrics`, and `target-gpu` on multi-GPU hosts.

---

## 2. Knobs in detail

### `collect-cpu-sampling` (default `false`)

Off by default, which surprises people: without it the report shows GPU activity but
no host functions. Turn it on whenever GPU utilization is low — the answer to "why is
the GPU idle" is almost always on the host.

Cost: adds normal CPU sampling overhead to the collection.

### `enable-stack-collection` (default `false`)

Host call stacks for kernel submissions and memory transfers. This is how you get
from "3.2 s spent in `zeMemoryCopy`" to "3.2 s spent copying in `Model::forward` at
model.cpp:88". Requires `collect-cpu-sampling=true` to be useful. For a stack per GPU
task use `enable-tasks-stack-collection=true` (2026.6).

Python programs: the stack stops in the interpreter (`[Skipped stack frame(s)]`,
`_PyEval_EvalFrameDefault`); Python source lines are not shown. The libtorch frames
(`THPVariable_tensor` → `to_device` → `_copy_xpu`) still name the op.

### `collect-memory-bandwidth` (default `false`)

Host DRAM bandwidth. Enable when transfer volumes are large: host↔device copies read
and write host memory, so a transfer-bound app may actually be host-memory bound.
Needs uncore access (bare metal) and VTune's sampling driver; without the driver the
collection fails ("Failed to connect to PMU reservation service (PAX)"). Check with
`vtune-self-checker.sh`, load it with `insmod-sep`.

### `analyze-power-usage` (default `false`)

Device power and frequency behaviour. Needs the SoC Watch driver; without it the
collection does not start ("Unable to start analysis. Socwatch driver cannot be
found"). The GPU driver exposes the same data: on xe, sysfs `.../gt0/freq0/act_freq`,
`.../throttle/reasons` and hwmon `energy1_input`; or `xpu-smi stats`. A device sustained
below nominal frequency invalidates any per-kernel comparison against a datasheet number.

### `enable-user-tasks` (default `true`) and `itt-domains` (default `all`)

ITT task instrumentation. Frameworks (oneDNN, IPEX, and others) emit ITT domains, so
this often gives free phase labels — `conv2d`, `attention`, `all_reduce` — without
any code change. Restrict `itt-domains` when a chatty framework floods the timeline;
list the domains you want.

To add your own regions:

```c
#include <ittnotify.h>
__itt_domain* d = __itt_domain_create("MyApp");
__itt_string_handle* h = __itt_string_handle_create("decode_step");
__itt_task_begin(d, __itt_null, __itt_null, h);
/* work */
__itt_task_end(d);
```

---

## 3. Collection combinations

| Goal | Command shape |
|---|---|
| Structural first pass | `vtune -collect xpu-offload -r <dir> -- ./app` |
| GPU is idle, find out why | `-knob collect-cpu-sampling=true -knob enable-stack-collection=true` |
| Transfer-heavy app | `-knob collect-memory-bandwidth=true` (sampling driver needed) |
| Suspected throttling | GPU sysfs / hwmon, `xpu-smi stats` (or `analyze-power-usage=true` with SoC Watch) |
| Long training/inference run | `-start-paused` + ITT resume, or `-resume-after=<sec> -d <sec>` |
| Noisy framework ITT output | `-knob itt-domains=<domain1,domain2>` |
| Running process | `-target-pid <pid> -d 60` |
| Large result risk | `-data-limit=4000` |

Pausing a long run from outside:

```bash
vtune -collect xpu-offload -start-paused -r <dir> -- ./train.py &
# after warmup completes
vtune -command resume -r <dir>
sleep 60
vtune -command stop -r <dir>
```

`vtune -command mark -r <dir>` inserts a marker to separate phases in the timeline.

---

## 4. Report cookbook

```bash
# Summary: elapsed time and totals (no GPU-busy line on 2026.6; "% of Elapsed Time"
# includes paused time and model load)
vtune -report summary -r <dir> -format csv

# GPU compute vs transfers: row "Compute" = total GPU compute time
vtune -report hotspots -r <dir> -group-by computing-task-purpose -format csv

# Phase times from ITT tasks, and Level Zero API time per call type
vtune -report hotspots -r <dir> -group-by task-domain,task -format csv

# Per-kernel table (the main one)
vtune -report hotspots -r <dir> -group-by computing-task \
      -sort-desc "Computing Task:Total Time" -format csv -limit 25

# Host side (needs collect-cpu-sampling=true)
vtune -report hotspots -r <dir> -group-by function -format csv -limit 25

# Submission call paths (needs stacks)
vtune -report callstacks -r <dir> -format csv -limit 25
vtune -report top-down   -r <dir> -format csv -limit 40

# Steady state only (-report timeline -format csv fails on 2026.6)
vtune -report hotspots -r <dir> -time-filter 30.0:60.0 \
      -group-by computing-task -format csv

# Per-thread submission behaviour
vtune -report hotspots -r <dir> -group-by thread -format csv

# A/B
vtune -report summary -r <baseline> -r <candidate> -format csv
```

Discover what a given result supports rather than guessing:

```bash
vtune -help report                                # report types
vtune -report hotspots -r <dir> -group-by=?       # groupings (works)
vtune -help report hotspots                       # report knobs
```

---

## 5. Column aliases to match on

| Concept | Aliases |
|---|---|
| Device time | `GPU Time`, `Total Time`, `Compute Task Time`, `Execution Time` |
| Utilization | `GPU Utilization`, `EU Array Active`, `GPU Busy`, `Idle` |
| Task counts | `Instance Count`, `Task Count`, `Computing Task`, `Average Time` |
| Transfers | `Data Transferred`, `Transfer Size`, `Transfer Time`, `Host to Device`, `Device to Host`, `Memory Copy` |
| Host side | `CPU Time`, `Effective Time`, `Elapsed Time` |
| Overhead | `Submission Time`, `Queue Time`, `Synchronization Time`, `API Time` |
| Power | `GPU Frequency`, `Average Frequency`, `Package Power`, `GPU Power` |

Absent column = not collected. Transfer columns absent usually means the runtime
wasn't traced; GPU columns absent usually means a driver or permission problem.

---

## 6. Transfer optimization ladder

Apply in order — earlier steps are cheaper and usually larger wins.

1. **Don't transfer.** Keep intermediate results resident on the device across
   kernels. The most common finding in this analysis is a buffer copied back to the
   host only to be copied straight back.
2. **Transfer once.** Move read-only inputs (weights, tables, meshes) to device
   memory during setup, not per iteration.
3. **Transfer larger.** A few large copies beat many small ones — per-transfer
   overhead is significant at small sizes.
4. **Choose the right memory kind.** For SYCL/oneAPI:
   - `malloc_device` — fastest device access, explicit copies.
   - `malloc_shared` (USM shared) — convenient, migration cost, good for
     sparse/irregular access.
   - `malloc_host` — pinned host memory; faster DMA than ordinary host allocations.
   Pick device memory for data the kernels hammer; pinned host memory for streaming.
5. **Overlap.** Async copies on a separate queue while compute runs; double-buffer
   the pipeline so copy(N+1) overlaps compute(N).
6. **Compress or reduce precision** in transit when the receiving kernel tolerates it.

---

## 7. Queue and submission optimization

| Symptom in the timeline | Cause | Fix |
|---|---|---|
| Kernel, gap, kernel, gap … regular gaps | Per-kernel `wait()` / in-order queue with host sync | Out-of-order queue; use events for dependencies only where needed |
| Thousands of sub-10 µs kernels | Too-fine granularity | Fuse kernels; batch iterations into one launch |
| GPU idle at the start of every iteration | Host prepares data serially | Pipeline: prepare batch N+1 on the host while the GPU runs N |
| One host thread submits for many streams | Submission serialized | One queue per stream; more submitting threads |
| Long tail after the last kernel | Synchronous readback of results | Overlap readback; only sync what is needed |
| GPU busy but wall clock unchanged | Kernels are inefficient, not scarce | `vtune-gpu-hotspots` |

---

## 8. Framework specifics

### PyTorch / IPEX on XPU

- Verify the tensors actually live on `xpu` — silent CPU fallback for unsupported ops
  is the most common cause of "the GPU isn't being used". Check with the framework's
  own device reporting before profiling further.
- Framework ITT domains give per-op names via `enable-user-tasks=true`.
- Profile a steady-state window: skip model load, warmup, and the first few
  iterations with `-start-paused` + resume.
- Time the same window once without VTune (`torch.xpu.synchronize()` before reading the
  clock). Eager PyTorch launches hundreds of small kernels per step and VTune's tracing
  slows each launch: a 4.26 ms decode step took 7.83 ms under xpu-offload. Busy % and
  speed-ups must use the un-profiled time.
- HF `DynamicCache` appends the KV cache with `torch.cat`: `CatArrayBatchedCopy*` GPU
  kernels (~10% of decode GPU time in Qwen3-8B), not transfers. A static KV cache
  removes them.
- Batch size is usually the biggest lever on offload efficiency — small batches
  produce many short kernels and a starved device.

### SYCL / DPC++

- Use out-of-order queues unless ordering is genuinely required.
- `-knob enable-stack-collection=true` maps submissions back to the source line.
- Kernel names come from the device binary; build with device debug info for
  readable names and source attribution.

### OpenMP target offload

- Check `map` clauses: `map(to:)` / `map(from:)` / `map(alloc:)` control transfer
  volume, and a missing `map(alloc:)` on a scratch array causes needless copies.
- `nowait` plus `depend` enables overlap of transfer and compute.
