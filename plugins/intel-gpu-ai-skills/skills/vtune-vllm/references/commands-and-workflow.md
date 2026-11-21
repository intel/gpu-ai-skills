# VTune commands and workflows for vLLM

Reference for `SKILL.md`. Its sections were moved out of `SKILL.md` to keep it short and
then edited; most come from the vendored vllm-xpu-profiling-toolkit `SKILL.md`
(Apache-2.0, see `PROVENANCE.md` in this folder).
Read `SKILL.md` §0 first: on vLLM's default mode the kernel table misses the decode
kernels, and every speed comparison must use un-profiled runs.

## 1. Control-collection cheat sheet


From the official Intel VTune Profiler Cheat Sheet
(https://www.intel.com/content/dam/develop/external/us/en/documents/vtune-profiler-cheat-sheet.pdf):

```
vtune <-action> [-action-option] [-global-option] [[--] <target> [target-options]]

ACTION 1 — Run Analysis:
  vtune -collect gpu-hotspots [options] -- <target>
  Key flags:
    -r, -result-dir=<str>       Result directory path
    -start-paused               Start collection paused, resume later  <- KEY FLAG
    -resume-after=<seconds>     Delay collection N seconds after start
    -k, -knob=<str>             Analysis-specific modifier (repeatable)
    -target-pid=<uint>          Attach to running process by PID
    -target-process=<str>       Attach to running process by name

ACTION 3 — Control Running Analysis:
  vtune -command pause   -r <result-dir>   <- Pause collection
  vtune -command resume  -r <result-dir>   <- Resume collection
  vtune -command stop    -r <result-dir>   <- Stop and finalize
  vtune -command status  -r <result-dir>   <- Print current status
  vtune -command mark    -r <result-dir>   <- Place reference timestamp

Example from cheat sheet:
  vtune -command resume -r r000hs
```

## 2. Command examples

#### 3.1 Basic GPU-Hotspots, Start Paused

```bash
vtune \
  -collect gpu-hotspots \
  -start-paused \
  -result-dir ./vtune_results/vllm_roi \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --port 8000 \
       --dtype bfloat16
```

`vtune ... -- vllm serve ...` fails: "File `vllm` is not an executable binary" (the
`vllm` command is a Python script). Use the module form above, or
`-- python3 /opt/venv/bin/vllm serve ...` (path from `command -v vllm`), or
`-- python3 -m vllm.entrypoints.cli.main serve ...`.

#### 3.2 With GPU-Hotspots Knobs for Xe Architecture

> Knob names vary across VTune releases. Verify against your install with
> `vtune -help collect gpu-hotspots` before pasting. The CPU-collector knobs
> `sampling-interval` and `enable-stack-collection` are NOT valid here and
> will fail with `Cannot find knob ...`.

```bash
vtune \
  -collect gpu-hotspots \
  -start-paused \
  -knob characterization-mode=overview \
  -knob collect-programming-api=true \
  -knob gpu-sampling-interval=1 \
  -result-dir ./vtune_results/vllm_roi \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --port 8000
```

Common valid knobs (gpu-hotspots, checked on VTune 2026.6):

| Knob | Values | Purpose |
|------|--------|---------|
| `gpu-profiling-mode` | `characterization` *(default)*, `source-analysis` | Top-level mode. `source-analysis` enables EU-stall / instruction-level attribution (one kernel at a time). |
| `characterization-mode` | `overview` *(default)*, `global-memory-accesses`, `compute-extended`, `instruction-count`, `video-decode`, `video-enhancement` | Sub-mode under `characterization`. |
| `source-analysis` | `bb-latency`, `mem-latency`, `stall-sampling` | Sub-mode under `source-analysis`; start with `stall-sampling` on library kernels. |
| `computing-tasks-of-interest` | `<kernel-name>` or `name#start#step#stop` | Restrict source-analysis to one kernel. |
| `gpu-sampling-interval` | float ms (default 1.0, range 0.005–10) | EU sampling rate. |
| `collect-programming-api` | `true` / `false` | Capture Level Zero / SYCL / OpenCL API timeline (recommended for vLLM). |

Removed or renamed on 2026.6 ("Cannot find knob" / "Cannot set knob"):
`computing-task-of-interest` (now `computing-tasks-of-interest`),
`collect-host-gpu-data-transfers`, `enable-gpu-runtimes`, and the `characterization-mode`
values `global-local-accesses`, `full-compute`, `dynamic-instruction-count`.

#### 3.3 Source-Analysis Mode (EU Stalls, Xe-Specific)

```bash
vtune \
  -collect gpu-hotspots \
  -start-paused \
  -knob gpu-profiling-mode=source-analysis \
  -knob source-analysis=stall-sampling \
  -knob computing-tasks-of-interest=<kernel-name> \
  -result-dir ./vtune_results/vllm_xe_source \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --port 8000
```

`gpu-vendor=intel` was removed in newer releases — VTune auto-detects.
Source-analysis is most useful when you've already identified a hotspot
in characterization mode and want EU stalls for that one kernel.

#### 3.4 External Collection Control (from Benchmark Script)

```bash
# Terminal 1: Launch vLLM under VTune (paused)
export VTUNE_RESULT=$(realpath ./vtune_results/run_$(date +%Y%m%d_%H%M%S))
vtune -collect gpu-hotspots -start-paused -result-dir $VTUNE_RESULT \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B --port 8000 &

# Terminal 2: Benchmark script controls collection timing
sleep 30                                      # Wait for server ready + warmup
vtune -command resume -r $VTUNE_RESULT        # START profiling
python run_benchmark.py --requests 50         # Run profiled workload
vtune -command pause  -r $VTUNE_RESULT        # STOP profiling
vtune -command stop   -r $VTUNE_RESULT        # Finalize result
```

#### 3.5 Attach to Running vLLM Process

```bash
python -m vllm.entrypoints.openai.api_server --model ... --port 8000 &
VLLM_PID=$!

vtune -collect gpu-hotspots \
      -start-paused \
      -target-pid $VLLM_PID \
      -result-dir ./vtune_results/attach_run

vtune -command resume -r ./vtune_results/attach_run
```

## 3. ITT API: inline collection control


The ITT (Instrumentation and Tracing Technology) API allows controlling
collection from within the profiled process — tightest possible ROI.

#### 4.1 Python ittapi Package

```bash
pip install ittapi   # Intel official Python binding
```

```python
import ittapi

ittapi.pause()    # Ensure paused at startup (belt-and-suspenders)

# Enter ROI
ittapi.resume()
# ... GPU inference work ...
import torch
torch.xpu.synchronize()   # CRITICAL: flush GPU queue before pause
ittapi.pause()
```

#### 4.2 ctypes Wrapper (Zero Dependencies)

For production use without pip. Full implementation in `references/itt-api.md`.

> **VTune 2026.6 ships no `libittnotify.so`** — only `sdk/lib64/libittnotify.a`. The
> ctypes fallback then has nothing to load and ITT calls are silently no-ops. Install
> `ittapi` (`pip install ittapi`, passing the proxy into the container if needed), or
> build a small shared shim from the `.a`. For window ROIs, prefer external
> `vtune -command resume/pause` (§2, "External Collection Control"): it needs no ITT library at all.

```python
from vtune_itt import itt_resume, itt_pause, vtune_roi

# As context manager:
with vtune_roi():            # auto-resume, auto-sync, auto-pause
    model.forward(input_ids)

# As decorator:
@roi()
def execute_model(self, ...):
    ...
```

## 4. Complete production workflow


```bash
#!/bin/bash
source /opt/intel/oneapi/setvars.sh
unset ZE_ENABLE_TRACING_LAYER   # Let VTune manage this

MODEL="meta-llama/Llama-3-8B"
PORT=8000
RESULT_DIR=$(realpath ./vtune_results/$(date +%Y%m%d_%H%M%S))

# Step 1: Launch vLLM under VTune (paused)
vtune -collect gpu-hotspots \
      -start-paused \
      -knob collect-programming-api=true \
      -result-dir $RESULT_DIR \
      -- python -m vllm.entrypoints.openai.api_server \
           --model $MODEL --port $PORT &
#   NOTE: vLLM >= 0.17 removed --disable-log-requests. Request logging is off
#   by default now; pass --enable-log-requests / --no-enable-log-requests only
#   if you need to override it explicitly.

# Step 2: Wait for server ready
until curl -sf http://localhost:$PORT/health > /dev/null; do sleep 2; done

# Step 3: Warmup (VTune stays PAUSED)
python benchmark_requests.py --num-requests 10 --quiet
sleep 3   # Drain async GPU work

# Step 4: Resume -> Benchmark -> Pause
vtune -command resume -r $RESULT_DIR
python benchmark_requests.py --num-requests 50 --max-tokens 256
vtune -command pause -r $RESULT_DIR

# Step 5: Finalize
vtune -command stop -r $RESULT_DIR
echo "Open: vtune-gui $RESULT_DIR"
```

## 5. VTune GUI validation


```
Open result: vtune-gui <result-dir>

Validation checklist:
[x] Summary tab -> GPU Time > 0, matches ROI duration
[x] Bottom-up tab -> Top kernels are GEMM/attention (sort by GPU Time desc)
[x] Timeline tab -> GPU activity shows bursts only during ROI window
[x] Platform tab -> Level Zero queue shows kernel submissions
[x] Top Hotspots -> Kernel names: xetla_gemm, esimd_attention, sycl_native_*

Red flags:
[!] GPU Time = 0        -> Missing torch.xpu.synchronize() before pause
[!] Empty GPU timeline  -> ZE tracing env conflict (references/troubleshooting.md)
[!] Huge CPU fraction   -> ROI too wide; move to Worker.execute_model()
[!] Truncated kernels   -> pause() fired before GPU drain
```

For numeric health bounds (EU active %, GPU BW %, expected GPU Time per
prefill/decode step on BMG/PVC) and the explicit pre-flight + post-run check
sequence, see `references/validation-and-flow.md` (sections C-F).

## 6. Baseline vs. candidate diff


```bash
BASE=./vtune_results/baseline
CAND=./vtune_results/candidate

# Side-by-side hotspots (sorted by GPU Time, top 30 each)
for D in $BASE $CAND; do
  vtune -report hotspots -r $D -group-by computing-task \
        -sort-desc "Computing Task:Total Time" -format csv -csv-delimiter comma -limit 30 \
    | awk -F, 'NR==1 || $0!~/^,*$/' \
    > $D/hotspots.csv
done

# Quick regression diff: kernel name + GPU Time delta
python3 - <<'PY'
import csv, os, sys
def load(p):
    with open(p) as f:
        return {r['Computing Task']: float(r.get('Computing Task:Total Time', 0) or 0)
                for r in csv.DictReader(f) if r.get('Computing Task')}
b, c = load(f"{os.environ['BASE']}/hotspots.csv"), load(f"{os.environ['CAND']}/hotspots.csv")
keys = sorted(set(b)|set(c), key=lambda k: -(c.get(k,0) - b.get(k,0)))
print(f"{'kernel':<60} {'base(s)':>10} {'cand(s)':>10} {'delta(s)':>10}")
for k in keys[:25]:
    print(f"{k[:60]:<60} {b.get(k,0):>10.3f} {c.get(k,0):>10.3f} "
          f"{c.get(k,0)-b.get(k,0):>+10.3f}")
PY
```

Use `vtune -command mark -r $RESULT_DIR -message "request_42_start"` from a
benchmark script to drop named timeline markers — these appear in the report
output and let you correlate hotspot deltas with specific requests without
ITT plumbing.

## 7. Per-step vs window ROI

(Moved from `SKILL.md` §5.1.)

The default patch in `references/vllm-integration.md` opens an ROI on every
`Worker.execute_model` call: 2N transitions for N steps and many fragmented
sub-ROIs in the trace. For most analyses, a **single window-level ROI** that
spans the whole profiled batch is cleaner and cheaper.

| Scheme | Resume/pause count | Trace fragmentation | Phase tagging | When to use |
|--------|-------------------|---------------------|---------------|-------------|
| Per-step | 2 × steps | High (many short ROIs) | Hardest signal | Phase-isolated runs (decode-only, prefill-only) and per-step kernel attribution |
| Batched window (recommended) | 2 (open once, close once) | Low (one continuous ROI) | Use ITT *tasks* (not resume/pause) | Steady-state hotspot ranking, BW/EU summaries, regression baselines |
| Hybrid | 2 + ITT tasks per step | Low + per-step task lanes | Best | Most production runs — open ROI once, tag each step with an ITT task |

Hybrid pattern (recommended default):

```python
# Worker patch — single resume/pause, but every step is an ITT task
import ittapi, torch
from vtune_itt import itt_resume, itt_pause

_dom    = ittapi.domain_create("vllm.window")
_open   = False

def _vtune_execute_model(self, scheduler_output, *args, **kwargs):
    global _open
    if not _open:
        itt_resume()       # opened once for the whole profiled batch
        _open = True

    label = classify_step(scheduler_output)   # see §3 of vllm-integration.md
    h     = ittapi.string_handle_create(label)
    ittapi.task_begin(_dom, h)
    try:
        return _orig_execute(self, scheduler_output, *args, **kwargs)
    finally:
        torch.xpu.synchronize()  # per-step sync keeps task end accurate
        ittapi.task_end(_dom)

# Close the window once at the end of the benchmark, e.g. via SIGTERM handler:
import atexit
def _close_roi():
    if _open:
        torch.xpu.synchronize()
        itt_pause()
atexit.register(_close_roi)
```

The benchmark script controls when the window opens with `vtune -command resume`
(workers re-resume idempotently); the per-step ITT *tasks* give phase lanes in
the Tasks tab without paying the per-step resume/pause cost.

## 8. Report-script design rules and hand commands

(Moved from `SKILL.md` §11.1.)

`run_vtune_vllm.sh` follows these rules so the result dir stays small:

1. `stderr` goes to a temporary file outside the result dir (mixed into the
   CSV it corrupts the header). No `.err` sidecar files. If a report fails, the
   first 5 error lines are printed inline and the output file is removed.
2. Empty CSVs are deleted (no zero-byte placeholders).
3. After finalization the raw collector byte-stream is dropped with
   `vtune -finalize -discard-raw-data`. The `.vtune` project still opens in
   the GUI but re-finalization / in-kernel source-view will not work. Set
   `VTUNE_KEEP_RAW=1` on the launch to preserve the raw data. Typical
   saving: 50-70% (a ~160 MB result drops to ~60 MB).

The equivalent hand commands, for reference:

```bash
vtune -report summary       -r $RESULT_DIR -format csv > $RESULT_DIR/summary.csv
# There is no `gpu-hotspots`, `tasks` or `top-tasks` report on VTune 2026.6.
vtune -report hotspots      -r $RESULT_DIR -group-by computing-task \
                            -sort-desc "Computing Task:Total Time" \
                            -format csv -csv-delimiter comma -limit 25 > $RESULT_DIR/hotspots.csv
vtune -report hotspots      -r $RESULT_DIR -group-by task-domain,task \
                            -format csv -csv-delimiter comma > $RESULT_DIR/tasks.csv

# Shrink the result dir (optional; run after all reports are generated).
vtune -finalize -r $RESULT_DIR -discard-raw-data
```
