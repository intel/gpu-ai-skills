---
name: vtune-vllm
description: "Intel VTune Profiler region-of-interest (ROI) profiling for vLLM serving on Intel GPUs (gpu-hotspots). Use when the user wants VTune on a vLLM or OpenAI-compatible LLM server: leaving model load, compile and warm-up out of the trace (vtune -start-paused, vtune -command resume/pause, ITT), decode or prefill windows, patching Worker.execute_model in the spawned EngineCore, vLLM XPU graphs hiding decode kernels, profiling overhead, VTune in a container, and headless kernel reports. Triggers on: profile vLLM with VTune, VTune ROI or decode-only profiling of a vLLM server, ittapi, ZE_ENABLE_TRACING_LAYER. For torch.profiler traces of a vLLM server use vllm-xpu-profile; not an LLM server: vtune-gpu-hotspots; device vs host split: vtune-xpu-offload."
---

# VTune Conditional Profiling for vLLM on Intel Xe GPU

> **Scope.** This is the framework-specific skill for **vLLM**. It assumes the analysis
> type is already `gpu-hotspots` and answers *how to aim it at an LLM server*. For the
> analysis type itself (knobs, reports, metric thresholds) read `../vtune-gpu-hotspots/SKILL.md`;
> for deciding whether the bottleneck is on the device or the host, `../vtune-xpu-offload/SKILL.md`.
> Verify every knob and report flag against the installed VTune with
> `vtune -help collect gpu-hotspots` and `vtune -help report <name>` before building a
> command — the versions noted below are what upstream validated, not what is installed here.

> **Prerequisite: VTune Profiler is a separate install.** The oneAPI Base
> Toolkit alone (`/opt/intel/oneapi/` with `compiler`, `mkl`, `pti`, etc.) does
> NOT include VTune. If `command -v vtune` returns nothing after
> `source /opt/intel/oneapi/setvars.sh`, install the VTune Profiler package
> separately (e.g. `apt install intel-oneapi-vtune` or the standalone
> installer). `run_vtune_vllm.sh` fails preflight with the same hint.
>
> Validated on `intel/vllm:0.17.0-xpu` (vLLM 0.1.dev14456+gde3f7fe65, torch
> 2.10) with oneAPI 2025.3 base kit + VTune installed alongside. Verified
> against `intel/vllm:0.14.1-xpu` at repo genesis; both v1 Worker paths hold.
> Re-checked on `vllm/vllm-openai-xpu` (vLLM 0.33.0, torch 2.14) with VTune 2026.6 on
> Arc Pro B70: section 0 and the notes marked "0.33" come from that test
> (an A/B test with and without this skill).

## 0. Read first: what decides whether the answer is right

These three points caused the wrong answers in our tests. Apply them before reading any
number from a vLLM profile.

1. **vLLM's default mode hides the decode kernels.** By default vLLM compiles the model
   and replays each decode step as an XPU graph. VTune does not list the kernels inside a
   graph replay. On Qwen3-8B the default-mode kernel table held 3.4 s of GPU time for
   ~1,000 decode steps; with graphs off it held 31 s. What remains visible (the lm_head
   GEMM, the sampler, prefill GEMMs) is not the decode workload.
   - For the kernel ranking, profile with graphs off: `--enforce-eager`, or
     `--compilation-config.cudagraph_mode=NONE` (keeps torch.compile). Un-profiled, these
     run at the same speed as the default (256 vs 259 tok/s on Qwen3-8B, 8 at a time).
   - Never divide the default-mode kernel time by the window and call it "GPU busy".
     The GPU engine time in `-report summary` ("GPU Time") still counts graph replays.
2. **The profiled server is slower than the real one.** gpu-hotspots added +39% per
   output token with graphs off and +5% in default mode. Time every speed you report or
   compare with **un-profiled** `vllm bench serve` runs. A profiled eager run vs an
   un-profiled default run measures VTune, not eager mode.
3. **GPU busy = GPU kernel time per decode step ÷ the un-profiled time per output token
   (TPOT).** On Qwen3-8B: 29.7 ms of kernels per step vs 29.9 ms TPOT — the GPU is
   busy all the time and there are no host gaps to remove. Gaps seen in a profiled
   window are VTune's overhead unless they also show in un-profiled timing.

Decode on one GPU is normally memory-bound: bytes of weights read per step ÷ GEMM time per
step is close to the DRAM peak (580 of 608 GB/s on B70). VTune's 1 ms sampled values
for the same GEMMs (XVE Active ~5%, ~394 GB/s) are diluted — do not act on them without
this bytes ÷ time check. Levers that move tokens/s then are fewer bytes per token (fp8 /
int4 weights: +58% on B70) and more requests per step (batching: 3.2× from 8 to 32 at a
time); kernel tuning has little headroom.

## How to Use This Skill

- **This file** — architecture, cheat sheet reference, quick-start commands, workflow
- **`references/validation-and-flow.md`** — annotated capture timeline, pre-flight + post-run verification checks (read this if you want to know *why* the ROI looks the way it does)
- **`references/itt-api.md`** — ITT API deep dive, C bindings, Python wrappers
- **`references/vllm-integration.md`** — All vLLM patch patterns (prefill, decode, per-request, middleware)
- **`references/troubleshooting.md`** — GPU timeline failures, ZE conflicts, sync issues, **healthy-trace cheat sheet** (§11)
- **`scripts/run_vtune_vllm.sh`** — the end-to-end launcher (preflight → paused collect → resume/pause window → finalize → CSV reports)
- **`scripts/serve_with_vtune.py`** — patched-server entry point; **`scripts/vtune_itt.py`** — the zero-dependency ctypes ITT wrapper

**Layered skills.** This skill owns *ROI mechanics for a vLLM server* only. Pair it with the
analysis-type skill (`vtune-xpu-offload`, then `vtune-gpu-hotspots`) for knobs and metric
bands. If the server runs in a container — the common deployment — see §3.6:
capabilities are fixed at `docker run` and cannot be added afterwards, and GPU metrics
need `/dev/dri` plus render-group membership *inside* the container.

---

## 1. Control-Collection Commands

`vtune -collect gpu-hotspots -start-paused -r <dir> -- <server>` starts collection paused;
`vtune -command resume | pause | stop | status | mark -r <dir>` controls it from another
shell. Full cheat sheet: `references/commands-and-workflow.md` §1.

---

## 2. Architecture: Global vs Conditional Profiling

### Global Profiling (avoid for vLLM)

VTune collects from process start — captures everything:
- Python interpreter startup (~2-5s noise)
- vLLM engine initialization, model load, KV cache alloc
- Tokenizer overhead (CPU-bound, pollutes GPU timeline)
- HTTP server / asyncio event loop overhead
- Warmup requests (skews kernel timing distributions)
- Network I/O between requests

**Result**: 10-50GB trace files, GPU timeline dominated by non-inference events.

### Conditional / ROI-Based Profiling (target approach)

```
[Process Start]
     |
     v
[VTune: PAUSED] <- Zero collection overhead outside ROI
     |
     v
[vLLM init, model load, KV alloc, warmup] <- EXCLUDED
     |
     v
[ROI Start] -> itt_resume() or vtune -command resume
     |
     v
[Inference: prefill + decode GPU kernels] <- COLLECTED
     |
     v
[GPU Sync: torch.xpu.synchronize()]       <- MANDATORY
     |
     v
[ROI End] -> itt_pause() or vtune -command pause
     |
     v
[VTune: PAUSED again]
```

**Quantitative benefits**:
| Metric              | Global     | Conditional    |
|---------------------|------------|----------------|
| Result size         | 10-50 GB   | 100 MB - 2 GB  |
| GPU timeline noise  | Very high  | Minimal        |
| Warmup contamination| Always     | None           |
| Analysis time in GUI| Minutes    | Seconds        |
| Kernel attribution  | Low        | High           |

---

## 3. Complete VTune Command Examples

### 3.1 Basic GPU-Hotspots, Start Paused

```bash
vtune -collect gpu-hotspots -start-paused -result-dir ./vtune_results/vllm_roi \
  -- python -m vllm.entrypoints.openai.api_server \
       --model <model> --port 8000 --enforce-eager
```

`vtune ... -- vllm serve ...` fails: "File `vllm` is not an executable binary" (the
`vllm` command is a Python script). Use the module form above, or
`-- python3 /opt/venv/bin/vllm serve ...` (path from `command -v vllm`), or
`-- python3 -m vllm.entrypoints.cli.main serve ...`.

Knobs, source analysis, external control from a benchmark script, and attach mode:
`references/commands-and-workflow.md` §2. Knobs checked on VTune 2026.6:
`gpu-profiling-mode`, `characterization-mode` (`overview`, `global-memory-accesses`,
`compute-extended`, `instruction-count`), `source-analysis` (`stall-sampling` first),
`computing-tasks-of-interest`, `gpu-sampling-interval` (0.005-10 ms),
`collect-programming-api`. Not valid on 2026.6: `computing-task-of-interest`,
`collect-host-gpu-data-transfers`, `enable-gpu-runtimes`, and the `characterization-mode`
values `global-local-accesses`, `full-compute`, `dynamic-instruction-count`.

### 3.6 VTune on the Host, vLLM in a Container

Checked with `vllm/vllm-openai-xpu` (0.33) and VTune 2026.6 installed on the host:

```bash
docker run -d --name vllm-vt --device /dev/dri --ipc=host --net=host --privileged \
  -v /opt/intel/oneapi/vtune:/opt/intel/oneapi/vtune:ro \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $PWD/results:/results \
  --entrypoint bash vllm/vllm-openai-xpu:latest -c \
  'source /opt/intel/oneapi/vtune/latest/env/vars.sh &&
   vtune -collect gpu-hotspots -start-paused -r /results/roi -- \
     python3 /opt/venv/bin/vllm serve Qwen/Qwen3-8B --max-model-len 4096 --enforce-eager'
# wait for /health, send warm-up requests, then:
docker exec vllm-vt bash -c 'source /opt/intel/oneapi/vtune/latest/env/vars.sh; vtune -command resume -r /results/roi'
# ... the load ...
docker exec vllm-vt bash -c 'source /opt/intel/oneapi/vtune/latest/env/vars.sh; vtune -command pause -r /results/roi; vtune -command stop -r /results/roi'
sudo chown -R $(id -u):$(id -g) results   # results written in the container are root-owned
```

- Run VTune inside the container. Attaching from the host to the container's root-owned
  server process failed ("no permissions") with every capability set we tried.
- `-start-paused` also hides the ~5-7 minutes of start-up under VTune (torch.compile and
  graph capture are much slower when traced); wait for `/health` before the warm-up.
- `vtune -command stop` ends the server; VTune then exits 4, but the result is finalized.
- Read reports from the host only after `chown` (otherwise "Insufficient permissions"),
  or in a second container that mounts the result folder.

---

## 4. ITT API: Inline Collection Control

Python `ittapi` (`ittapi.resume()` / `ittapi.pause()`, after `torch.xpu.synchronize()`)
or the ctypes wrapper in `scripts/vtune_itt.py`: `references/commands-and-workflow.md` §3
and `references/itt-api.md`.

> **VTune 2026.6 ships no `libittnotify.so`** — only `sdk/lib64/libittnotify.a`. The
> ctypes fallback then has nothing to load and ITT calls are silently no-ops. Install
> `ittapi` (`pip install ittapi`, passing the proxy into the container if needed), or
> build a small shared shim from the `.a`. For window ROIs, prefer external
> `vtune -command resume/pause` (§1): it needs no ITT library at all.

---

## 5. ROI Boundaries: Best Choices for LLM

This skill targets **vLLM v1 exclusively** (default ≥0.10, including
`intel/vllm:0.14.1-xpu` and every 0.17-xpu build validated since). The v0
worker path (`vllm.worker.worker.Worker`, `ExecuteModelRequest`,
`seq_group_metadata_list`) is not supported — the wrapper raises `RuntimeError`
at import time if v1 isn't available.

Key v1 primitives the wrapper relies on:

| Concern | v1 path |
|---------|---------|
| Worker class | `vllm.v1.worker.gpu_worker.Worker` |
| `execute_model` arg | `SchedulerOutput` |
| Phase detection | `bool(scheduler_output.scheduled_new_reqs)` and `num_scheduled_tokens` |
| API server entry | `runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)` |

| Target | Integration Point (v1) | Granularity | Noise Level |
|--------|------------------------|-------------|-------------|
| Full inference (recommended) | `vllm.v1.worker.gpu_worker.Worker.execute_model` | Per-step | Low |
| Pure GPU compute (tightest) | `model.forward` (post-load patch) | Per-forward | Minimal |
| Prefill phase only | Conditional in patched `execute_model` (check `scheduled_new_reqs`) | Per-prefill | Low |
| Decode phase only | Conditional in patched `execute_model` (check `scheduled_cached_reqs`) | Per-decode | Low |
| Per-request | `AsyncLLM.generate` (parent process) | Per-request | Medium |
| Full pipeline | `EngineCore.step` (in subprocess) | Per-step | Medium |

**Never ROI above `EngineCore.step` / `LLMEngine.step`** — includes HTTP,
tokenizer, asyncio noise.

### 5.1 Per-step vs window ROI

Per-step resume/pause gives 2 × steps transitions and a fragmented trace. For steady-state
hotspot ranking prefer one window ROI (opened by `vtune -command resume`) with an ITT *task*
per step for phase lanes. Comparison table and the hybrid worker patch:
`references/commands-and-workflow.md` §7.

---

## 6. GPU Synchronization Requirement

Always call `torch.xpu.synchronize()` before pausing. This is the #1 source
of empty GPU timelines when omitted.

```
Without sync:                    With sync:
[resume]                         [resume]
  GPU work submitted             GPU work submitted
[pause] <- too early!           torch.xpu.synchronize() <- wait for GPU
  GPU kernels still running!    [pause] <- safe, all kernels captured
  VTune misses them entirely!
```

Overhead of `synchronize()`: 1-50 us depending on queue depth.
For LLM forward passes (10-500 ms), this is negligible.

---

## 7. Complete Production Workflow

`scripts/run_vtune_vllm.sh` runs it end to end (preflight, paused launch, warm-up,
resume → benchmark → pause, finalize, CSV reports, post-run checks). The step-by-step
shell version is in `references/commands-and-workflow.md` §4.

---

## 8. Multi-Process Considerations

vLLM v1 always runs the engine core in a separate process from the API server,
even with TP=1. Tensor parallelism adds one GPU worker process per rank.

```
api_server (parent)  --ZMQ-->  EngineCore (subprocess)  --spawn-->  Worker rank 0
                                                                    Worker rank 1
                                                                    ...
```

| Control Method | Scope |
|----------------|-------|
| `vtune -command resume` | All processes under VTune umbrella (parent + all descendants) |
| `ittapi.resume()` / `__itt_resume()` | Only the calling process |

**Implications for v1:**
- ITT calls in the API server parent are useless for GPU profiling — the GPU
  work happens in the worker subprocess. Patch `Worker.execute_model` instead.
- **vLLM-XPU starts the EngineCore with `spawn`, so a patch applied in the parent
  does not reach it** (checked on 0.17 and 0.33: `serve_with_vtune.py` logs
  "Patched ... execute_model" once, in the API server; no "ROI window OPENED", no
  `vllm.*` ITT domain in the result). Load the patch in every process: put it in a
  `sitecustomize.py` on `PYTHONPATH` (Python imports it at start-up in each spawned
  process), or register a vLLM `general_plugins` entry point. Do not load it in the
  benchmark client. `run_vtune_vllm.sh` now does this with `scripts/sitecustomize.py`.
- The launcher's outer `vtune -command resume` records **every** step of the
  benchmark window. `VTUNE_PHASE` only chooses which steps get ITT task labels (and
  gpu-hotspots on 2026.6 records no ITT tasks). For a decode-dominated window use long
  answers and short prompts; for a decode-only kernel table, split steps by kernel shape
  or run xpu-offload, which keeps ITT tasks.
- For multi-rank ITT control, each rank must call ITT independently.
- External `vtune -command resume/pause` is the simplest control surface for v1
  because it covers all descendants regardless of spawn semantics.

See `references/vllm-integration.md` for per-rank patterns.

---

## 9. Xe GPU Architecture Caveats

1. **Level Zero tracing**: VTune manages `ZE_ENABLE_TRACING_LAYER` automatically.
   Never set it manually — causes empty GPU timelines (on 0.33 a run with it set had an
   empty per-kernel table).

1b. **XPU graphs**: the default mode replays decode steps as graphs and VTune lists no
   kernels inside them (§0). Profile with `--enforce-eager` or
   `--compilation-config.cudagraph_mode=NONE` for kernel tables.

2. **Tile affinity** (multi-tile Xe, e.g., PVC):
   ```bash
   export ZE_AFFINITY_MASK=0   # Profile tile 0 only
   ```

3. **Decode is BW-bound**: Expect 10-40% EU utilization for decode on large
   models — correct, not a profiling failure. Confirm with bytes of weights ÷ GEMM
   time per step against the DRAM peak (§0); the sampled bandwidth reads low.

4. **Prefill is compute-bound**: Expect 50-80% EU utilization for long sequences.

---

## 10. VTune GUI Validation

Checklist and red flags: `references/commands-and-workflow.md` §5; numeric bounds:
`references/validation-and-flow.md` (sections C-F).

---

## 11. Headless Reporting & A/B Diff

The GUI is not required to extract hotspots. Use this when:

- running profiling in CI / on a headless node
- you need a numeric baseline vs. a candidate (regression gate)
- you want grep-able artifacts archived alongside the result dir

### 11.1 Single-result reports

`run_vtune_vllm.sh` emits three CSVs (report names fixed for VTune 2026.6: older
versions of the script asked for `gpu-hotspots` and `top-tasks`, which do not exist):

| File | Contents |
|---|---|
| `summary.csv` | GPU Time, EU active %, BW %, elapsed |
| `hotspots.csv` | Top 25 computing tasks by GPU Time |
| `tasks.csv`   | ITT task domains/labels (only when ITT tasks were emitted) |

The script keeps stderr out of the CSVs, deletes empty CSVs and drops raw data after
finalization (`VTUNE_KEEP_RAW=1` keeps it). The equivalent hand commands:
`references/commands-and-workflow.md` §8.

### 11.2 Baseline vs. candidate diff

Side-by-side kernel tables and a GPU-time delta script: `references/commands-and-workflow.md` §6.

---

## Reference Files

| File | Contents |
|------|----------|
| `references/commands-and-workflow.md` | Control-collection cheat sheet, command examples, ITT snippets, production workflow, GUI checklist, A/B diff (original to this pack) |
| `references/validation-and-flow.md` | End-to-end captured-flow timeline (T0..T10), pre-flight + post-run verification, healthy numeric bounds |
| `references/itt-api.md` | ITT API bindings, ctypes wrapper, C extension, task/domain/frame API, overhead table |
| `references/vllm-integration.md` | Monkey-patch patterns, batched/per-step ROI, sharper phase classifier, prefill/decode/mixed/cache-hit, per-request ROI, async, benchmark script |
| `references/troubleshooting.md` | Empty timelines, ZE conflicts, sync issues, environment checklist, GUI validation, **healthy-trace cheat sheet** (§11) |
