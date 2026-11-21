# vLLM Integration Patterns for VTune Conditional Profiling

## vLLM Version Compatibility (READ FIRST)

This skill targets **vLLM v1 exclusively** (default since ~0.10, including
`intel/vllm:0.14.1-xpu` and every 0.17-xpu build validated since). The wrapper
raises `RuntimeError` if `vllm.v1.worker.gpu_worker.Worker` isn't importable —
we intentionally do not fall back to the legacy v0 path.

Key v1 primitives every pattern below depends on:

| Concern | v1 path |
|---------|---------|
| Worker class | `vllm.v1.worker.gpu_worker.Worker` |
| Engine core | `EngineCore` subprocess (multi-process; separate from the API server) |
| `execute_model` arg | `SchedulerOutput` with `scheduled_new_reqs` / `scheduled_cached_reqs` and `num_scheduled_tokens` |
| API server `main()` | NOT exported — invoke via `python -m` or `runpy` |
| ITT in engine subprocess | must be initialized inside the subprocess, not the API parent |

**Multi-process gotcha:** `vllm.entrypoints.openai.api_server` runs in a
parent process; the GPU worker runs in an `EngineCore` subprocess fork/spawn'd
later. ITT calls made in the parent BEFORE the fork won't reach the child if
the child spawns fresh. Patch `Worker.execute_model` *as a class method on the
imported class* before the engine starts — the subprocess re-imports the
patched class. For external `vtune -command resume/pause`, this is moot: vtune
controls all descendants under its umbrella.

## vLLM Call Stack (v1, Profiling Perspective)

```
HTTP POST /v1/completions
    |
    v  [API process: parent]
AsyncLLM.generate()                  [asyncio coroutine]
    |
    v  [IPC: ZMQ to EngineCore subprocess]
EngineCore.step()                    [scheduler + KV cache management]
    |
    +-- Scheduler.schedule()         -> SchedulerOutput
    |
    +-- model_executor.execute_model(scheduler_output)
            |
            v  [GPU worker process(es): one per TP rank]
        Worker.execute_model(scheduler_output)   [BEST ROI BOUNDARY]
            |
            +-- gpu_model_runner.execute_model()
                    |
                    v
                model.forward()      [PURE GPU COMPUTE — tightest ROI]
                    |
                    +-- Attention (paged, flash)
                    +-- GEMM (QKV, FFN, LM head)
                    +-- Sampling kernels
```

**Rule**: The lower in the stack, the tighter and cleaner the ROI.

**Phase detection in v1 (sharper than the boolean):** a `SchedulerOutput` can be
prefill-only, decode-only, *mixed* (chunked prefill landing alongside decode), or
a *prefix-cache hit* (new request, but every prompt token already cached → zero
new compute). Bucketing all four as "prefill" pollutes prefill hotspots with
zero-compute steps and miscategorizes mixed steps. Use the explicit classifier
below.

```python
# classify_step: returns one of "prefill", "decode", "mixed", "cache_hit", "empty"
def classify_step(so) -> str:
    new    = getattr(so, "scheduled_new_reqs", None) or []
    cached = getattr(so, "scheduled_cached_reqs", None) or []

    # tokens that will actually run through the model this step
    sched  = getattr(so, "num_scheduled_tokens", {}) or {}
    new_tok = sum(sched.get(getattr(r, "req_id", None), 0) for r in new)
    cac_tok = sum(sched.get(getattr(r, "req_id", None), 0) for r in cached)

    if new and new_tok == 0 and not cached:
        return "cache_hit"               # prefix-cache hit: skip in prefill stats
    if new and cached:
        return "mixed"                   # chunked prefill + decode in same step
    if new:
        return "prefill"
    if cached:
        return "decode"
    return "empty"
```

When tagging ITT tasks, embed the *shape* in the task name so the Tasks tab
ranks shape-correlated outliers directly:

```python
def step_label(so, phase: str) -> str:
    sched = getattr(so, "num_scheduled_tokens", {}) or {}
    bs    = len(sched)
    tok   = sum(sched.values())
    return f"{phase} bs={bs} tok={tok}"
```

`string_handle_create` should be cached per label (cheap dict lookup) so
high-frequency decode steps don't allocate a new handle every step.

---

## 1. Method A: Monkey-Patch at Startup (No vLLM Source Modification)

The cleanest production approach — instrument vLLM without touching its source.

```python
#!/usr/bin/env python3
"""
serve_with_vtune.py

Drop-in replacement launch script for vLLM with VTune ROI instrumentation.

Environment variables:
  VTUNE_ROI_MODE        = model_forward | execute_model (default) | engine_step
  VTUNE_PROFILE_WARMUP  = 0 (default, skip warmup) | 1 (profile warmup too)
  VTUNE_RESULT_DIR      = path to vtune result dir (for external command control)
  VTUNE_WARMUP_DELAY    = seconds to wait before enabling ROI (default: 30)

Launch:
  vtune -collect gpu-hotspots -start-paused -result-dir ./out \
    -- python serve_with_vtune.py --model meta-llama/Llama-3-8B --port 8000
"""

import os
import sys
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vtune.vllm")

# ── Import ITT before vLLM (critical ordering) ─────────────────────────────
from vtune_itt import itt_pause, itt_resume, vtune_roi, roi, itt_available, init_paused

init_paused()   # Belt-and-suspenders: ensure paused before model load
if itt_available():
    logger.info("VTune ITT loaded — collection paused at startup")
else:
    logger.warning("VTune ITT NOT available — profiling control via external commands only")

# ── Configuration ───────────────────────────────────────────────────────────
ROI_MODE      = os.environ.get("VTUNE_ROI_MODE", "execute_model")
PROFILE_WARMUP = os.environ.get("VTUNE_PROFILE_WARMUP", "0") == "1"
WARMUP_DELAY  = float(os.environ.get("VTUNE_WARMUP_DELAY", "30"))

_warmup_done = False
_warmup_lock = threading.Lock()

def _schedule_warmup_done():
    """Mark warmup complete after delay. Replace with signal-based approach for precision."""
    def _worker():
        global _warmup_done
        time.sleep(WARMUP_DELAY)
        with _warmup_lock:
            _warmup_done = True
        logger.info(f"Warmup complete after {WARMUP_DELAY}s — VTune ROI now active")
    threading.Thread(target=_worker, daemon=True).start()

# ── Patch: execute_model (recommended) ─────────────────────────────────────
if ROI_MODE == "execute_model":
    # v1-only. If this import fails, upgrade vLLM — we don't fall back.
    from vllm.v1.worker.gpu_worker import Worker
    import torch

    _original_execute = Worker.execute_model

    def _vtune_execute_model(self, scheduler_output, *args, **kwargs):
        if _warmup_done or PROFILE_WARMUP:
            with vtune_roi(sync_device="xpu"):
                return _original_execute(self, scheduler_output, *args, **kwargs)
        return _original_execute(self, scheduler_output, *args, **kwargs)

    Worker.execute_model = _vtune_execute_model
    logger.info("Patched vllm.v1 Worker.execute_model for VTune ROI")

# ── Patch: model.forward (tightest) ────────────────────────────────────────
elif ROI_MODE == "model_forward":
    # Patch happens post-model-load via get_model hook
    _original_load = None

    def _patch_model_forward(model):
        import torch
        _orig_fwd = model.forward

        def _vtune_forward(*args, **kwargs):
            if _warmup_done or PROFILE_WARMUP:
                with vtune_roi(sync_device="xpu"):
                    return _orig_fwd(*args, **kwargs)
            return _orig_fwd(*args, **kwargs)

        model.forward = _vtune_forward
        logger.info("Patched model.forward for VTune ROI")
        return model

    try:
        import vllm.model_executor.model_loader as _ml
        _original_load = _ml.get_model

        def _patched_get_model(*args, **kwargs):
            model = _original_load(*args, **kwargs)
            return _patch_model_forward(model)

        _ml.get_model = _patched_get_model
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch model loader: {e}. Falling back to execute_model mode.")
        ROI_MODE = "execute_model"

# ── Patch: engine.step (broadest useful; v1 EngineCore subprocess) ─────────
elif ROI_MODE == "engine_step":
    # v1 splits scheduling into EngineCore.step (in a subprocess). Patch there,
    # not the legacy LLMEngine — LLMEngine is a thin front-end shim now.
    from vllm.v1.engine.core import EngineCore
    _original_step = EngineCore.step

    def _vtune_step(self, *args, **kwargs):
        if _warmup_done or PROFILE_WARMUP:
            with vtune_roi(sync_device="xpu"):
                return _original_step(self, *args, **kwargs)
        return _original_step(self, *args, **kwargs)

    EngineCore.step = _vtune_step
    logger.info("Patched vllm.v1 EngineCore.step for VTune ROI")

# ── Schedule warmup completion ──────────────────────────────────────────────
_schedule_warmup_done()

# ── Launch vLLM normally ────────────────────────────────────────────────────
# vLLM v1 does not export `main` from api_server. Use runpy to invoke the
# module's __main__ block exactly like `python -m vllm.entrypoints.openai.api_server`.
if __name__ == "__main__":
    import runpy
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module(
        "vllm.entrypoints.openai.api_server",
        run_name="__main__",
        alter_sys=True,
    )
```

> **Common pitfall:** `from vllm.entrypoints.openai.api_server import main` raises
> `ImportError: cannot import name 'main'` on vLLM v1. The `runpy` form above is
> the supported replacement. Older skill snippets that import `main` are stale.

---

## 2. Prefill-Only Profiling (vLLM v1)

Use `classify_step` from the section above to skip mixed steps and prefix-cache
hits — both pollute "prefill" stats with non-prefill compute.

```python
from vllm.v1.worker.gpu_worker import Worker
from vtune_itt import itt_resume, itt_pause
import ittapi, torch

_prefill_dom = ittapi.domain_create("vllm.prefill")
_handles     = {}   # label -> string_handle (one allocation per shape)

def _h(label: str):
    h = _handles.get(label)
    if h is None:
        h = ittapi.string_handle_create(label)
        _handles[label] = h
    return h

_orig_execute = Worker.execute_model

def _prefill_only_execute(self, so, *args, **kwargs):
    phase = classify_step(so)              # defined once at module top
    profile = (phase == "prefill")          # skip mixed / cache_hit / decode

    if profile:
        ittapi.task_begin(_prefill_dom, _h(step_label(so, "prefill")))
        itt_resume()

    result = _orig_execute(self, so, *args, **kwargs)

    if profile:
        torch.xpu.synchronize()             # MUST sync before pause
        itt_pause()
        ittapi.task_end(_prefill_dom)

    return result

Worker.execute_model = _prefill_only_execute
```

---

## 3. Decode-Only Profiling (vLLM v1)

```python
from vllm.v1.worker.gpu_worker import Worker
from vtune_itt import itt_resume, itt_pause
import ittapi, torch

_decode_dom = ittapi.domain_create("vllm.decode")
_handles    = {}

def _h(label: str):
    h = _handles.get(label)
    if h is None:
        h = ittapi.string_handle_create(label)
        _handles[label] = h
    return h

_orig_execute = Worker.execute_model

def _decode_only_execute(self, so, *args, **kwargs):
    phase = classify_step(so)
    profile = (phase == "decode")           # mixed / prefill / cache_hit excluded

    if profile:
        ittapi.task_begin(_decode_dom, _h(step_label(so, "decode")))
        itt_resume()

    result = _orig_execute(self, so, *args, **kwargs)

    if profile:
        torch.xpu.synchronize()
        itt_pause()
        ittapi.task_end(_decode_dom)

    return result

Worker.execute_model = _decode_only_execute
```

> **Mixed steps with chunked prefill:** `classify_step` returns `"mixed"` for
> these — the patches above skip them, which is the correct default. If you
> want them included in prefill stats, broaden the predicate to
> `phase in ("prefill", "mixed")`. For the cleanest decode isolation also
> launch with `--enable-chunked-prefill=false`.

> **Cache-hit steps** (`classify_step` → `"cache_hit"`) submit zero new tokens
> through the model. Including them in prefill skews the per-step time
> distribution downward; the classifier excludes them by default.

---

## 4. Per-Request Profiling (Selected Requests Only)

Profile only specific request IDs — useful for isolating one request from
steady-state batch overlap noise.

```python
import threading
import subprocess
from typing import Set
from vllm.engine.async_llm_engine import AsyncLLMEngine
import os

# Thread-safe set of request IDs currently being profiled
_profile_request_ids: Set[str] = set()
_lock = threading.Lock()

VTUNE_RESULT_DIR = os.environ.get("VTUNE_RESULT_DIR", "")


def register_profile_request(request_id: str):
    """Call from benchmark script to mark a request for profiling."""
    with _lock:
        _profile_request_ids.add(request_id)


def _vtune_external(command: str):
    """Issue vtune command to control collection."""
    if not VTUNE_RESULT_DIR:
        return
    subprocess.run(
        ["vtune", "-command", command, "-r", VTUNE_RESULT_DIR],
        check=False, capture_output=True
    )


_orig_generate = AsyncLLMEngine.generate


async def _per_request_generate(self, prompt, sampling_params, request_id, **kwargs):
    with _lock:
        should_profile = request_id in _profile_request_ids

    if should_profile:
        _vtune_external("resume")

    async for output in _orig_generate(self, prompt, sampling_params, request_id, **kwargs):
        yield output

    if should_profile:
        # Note: GPU sync must occur inside Worker.execute_model patches
        # External vtune command has no access to torch context here
        _vtune_external("pause")
        with _lock:
            _profile_request_ids.discard(request_id)


AsyncLLMEngine.generate = _per_request_generate
```

---

## 5. Middleware-Based ROI (HTTP-Layer, Wide ROI)

Use when you want to correlate HTTP latency with GPU activity.
Note: This is wider than Worker-level patching — includes serialization.

```python
import time
import os
import subprocess
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

VTUNE_RESULT_DIR = os.environ.get("VTUNE_RESULT_DIR", "")
PROFILE_PATHS    = {"/v1/completions", "/v1/chat/completions"}
PROFILE_EVERY_N  = int(os.environ.get("VTUNE_PROFILE_EVERY_N", "1"))

_request_counter = 0


def _vtune(command: str):
    if VTUNE_RESULT_DIR:
        subprocess.run(
            ["vtune", "-command", command, "-r", VTUNE_RESULT_DIR],
            check=False, capture_output=True
        )


class VTuneROIMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware wrapping inference endpoints with VTune ROI.
    Pair with Worker.execute_model patching for GPU sync correctness.
    """
    async def dispatch(self, request: Request, call_next):
        global _request_counter

        is_inference = request.url.path in PROFILE_PATHS
        _request_counter += is_inference

        should_profile = (
            is_inference and
            VTUNE_RESULT_DIR and
            (_request_counter % PROFILE_EVERY_N == 0)
        )

        if should_profile:
            _vtune("resume")

        response = await call_next(request)

        if should_profile:
            _vtune("pause")   # GPU sync happens in Worker patch

        return response


# Register:
# from vllm.entrypoints.openai.api_server import app
# app.add_middleware(VTuneROIMiddleware)
```

---

## 6. Multi-Rank: Tensor Parallel Profiling

```python
# In Worker.__init__ or process startup for each TP rank:
import ittapi
from vtune_itt import init_paused

def setup_rank_profiling(rank: int, world_size: int):
    """Call once per worker process at startup."""
    ittapi.thread_set_name(f"vllm_gpu_rank{rank}_of_{world_size}")
    init_paused()   # Each rank independently starts paused

# ITT-based ROI in each rank's execute_model (v1-only):
from vllm.v1.worker.gpu_worker import Worker
from vtune_itt import vtune_roi
import torch

_orig_execute = Worker.execute_model

def _multi_rank_execute(self, scheduler_output, *args, **kwargs):
    # Each rank controls its own ITT state independently.
    # vtune -command resume affects all ranks simultaneously (umbrella scope).
    # itt_resume() / itt_pause() affect only THIS rank's process.
    with vtune_roi(sync_device="xpu"):
        return _orig_execute(self, scheduler_output, *args, **kwargs)

Worker.execute_model = _multi_rank_execute
```

**Key rule**: `vtune -command resume` resumes ALL ranks under the VTune umbrella.
`ittapi.resume()` (ITT API) resumes only the calling process (rank).
Choose based on whether you want synchronized or per-rank collection.

---

## 7. Complete Benchmark Script with VTune Control

```python
#!/usr/bin/env python3
"""
benchmark_with_vtune.py

Drive inference load against vLLM while controlling VTune collection.

Usage:
  # In terminal 1: start vLLM under VTune
  export VTUNE_RESULT=$(realpath ./vtune_results/run_001)
  vtune -collect gpu-hotspots -start-paused -result-dir $VTUNE_RESULT \
    -- python serve_with_vtune.py --model meta-llama/Llama-3-8B --port 8000

  # In terminal 2: run this script
  python benchmark_with_vtune.py \
    --result-dir $VTUNE_RESULT \
    --endpoint http://localhost:8000 \
    --warmup 10 --profile 50
"""

import os
import time
import subprocess
import statistics
import argparse
import requests
import sys


def vtune_cmd(command: str, result_dir: str):
    result = subprocess.run(
        ["vtune", "-command", command, "-r", result_dir],
        capture_output=True, text=True
    )
    status = "OK" if result.returncode == 0 else f"ERROR: {result.stderr.strip()}"
    print(f"[VTune] {command.upper():<8} {status} @ {time.strftime('%H:%M:%S')}")


def wait_for_server(endpoint: str, timeout: int = 180):
    print(f"Waiting for vLLM at {endpoint}...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{endpoint}/health", timeout=2)
            if r.status_code == 200:
                print(" ready.")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print()
    raise TimeoutError(f"Server not ready after {timeout}s")


def send_request(endpoint: str, prompt: str, model: str, max_tokens: int) -> dict:
    return requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=300,
    ).json()


def run_batch(endpoint: str, prompt: str, model: str,
              max_tokens: int, num_requests: int, label: str) -> list:
    latencies = []
    for i in range(num_requests):
        t0 = time.perf_counter()
        resp = send_request(endpoint, prompt, model, max_tokens)
        latency = time.perf_counter() - t0
        latencies.append(latency)
        tokens = resp.get("usage", {}).get("completion_tokens", "?")
        print(f"  [{label}] {i+1:3d}/{num_requests} | {latency:.3f}s | {tokens} tokens")
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "meta-llama/Llama-3-8B"))
    parser.add_argument("--prompt", default="Explain the theory of quantum entanglement in detail:")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile", type=int, default=50)
    args = parser.parse_args()

    result_dir = os.path.realpath(args.result_dir)

    # 1. Wait for server
    wait_for_server(args.endpoint)

    # 2. Warmup (VTune stays paused)
    print(f"\n=== WARMUP: {args.warmup} requests (VTune PAUSED) ===")
    run_batch(args.endpoint, args.prompt, args.model,
              args.max_tokens, args.warmup, "warmup")

    # 3. Brief pause to drain async GPU work from warmup
    print("\nDraining GPU queue after warmup...")
    time.sleep(3)

    # 4. Resume VTune and run profiled benchmark
    print(f"\n=== PROFILED: {args.profile} requests ===")
    vtune_cmd("resume", result_dir)

    latencies = run_batch(args.endpoint, args.prompt, args.model,
                          args.max_tokens, args.profile, "profiled")

    vtune_cmd("pause", result_dir)

    # 5. Finalize
    vtune_cmd("stop", result_dir)

    # 6. Print statistics
    print(f"\n=== Results ({args.profile} profiled requests) ===")
    print(f"  Mean latency : {statistics.mean(latencies):.3f}s")
    print(f"  Median       : {statistics.median(latencies):.3f}s")
    print(f"  P90          : {sorted(latencies)[int(len(latencies)*0.90)]:.3f}s")
    print(f"  P99          : {sorted(latencies)[int(len(latencies)*0.99)]:.3f}s")
    print(f"  Min / Max    : {min(latencies):.3f}s / {max(latencies):.3f}s")
    print(f"\nOpen result: vtune-gui {result_dir}")


if __name__ == "__main__":
    main()
```

---

## 8. Async Considerations

vLLM's engine is asyncio-based. Key points:

1. **ITT calls are thread-safe** — safe to call from asyncio coroutines
2. **External vtune commands** — blocking subprocess; use
   `asyncio.create_subprocess_exec` for non-blocking variant
3. **`torch.xpu.synchronize()` from async context** — blocking but safe
   (acquires GIL momentarily, doesn't deadlock asyncio)
4. **Request overlap in steady-state** — multiple requests overlap in the
   engine. For single-request isolation:

```bash
# Disable request batching for clean single-request profiling:
vtune -collect gpu-hotspots -start-paused -result-dir ./out \
  -- python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3-8B \
       --max-num-seqs 1 \
       --port 8000
# NOTE: --disable-log-requests was removed in vLLM >= 0.17 (log-requests
# defaults to off). Use --no-enable-log-requests only if you need to
# force-disable it on an older build.
```
