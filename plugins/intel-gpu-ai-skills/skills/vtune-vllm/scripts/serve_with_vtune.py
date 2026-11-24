#!/usr/bin/env python3
"""
serve_with_vtune.py - Launch vLLM OpenAI server with phase-aware VTune ROI.

Patches Worker.execute_model so that, while the gate is open, only the
requested phase's batches are profiled. Targets vLLM v1 exclusively
(vllm.v1.worker.gpu_worker.Worker — default in intel/vllm:0.14.1-xpu and
every supported release since). If the v1 Worker can't be imported, the
wrapper refuses to start rather than silently downgrading.

Phase classification:
  prefill    - new request(s) entering the model with new tokens to compute
  decode     - only cached requests, advancing past their prompts
  mixed      - chunked prefill landing alongside decode in the same step
  cache_hit  - new request(s) but every prompt token already cached
               (zero new compute; would skew prefill stats - excluded by default)
  empty      - no scheduled work (pad / dummy step)

Environment variables:
  VTUNE_PHASE          prefill | decode | mixed | both     (default: both)
                       both => prefill + decode + mixed; cache_hit always excluded.
  VTUNE_ROI_MODE       window (default) | per_step | per_step_isolate
                       window           - one resume on first profiled step,
                                          one pause at process exit; ITT tasks
                                          tag every step (sharp Tasks tab,
                                          minimal trace fragmentation).
                       per_step         - resume/pause every profiled step
                                          (legacy; high transition cost).
                       per_step_isolate - per_step semantics but rejects mixed
                                          steps too (cleanest decode isolation
                                          when chunked prefill is on).
  VTUNE_ROI_GATE       path to a sentinel file; ROI activates while it exists
                       (set by run_vtune_vllm.sh, created right before
                       benchmark starts, removed when benchmark ends).
  VTUNE_PROFILE_WARMUP 1 to ignore the gate and always profile (default: 0)

Why a sentinel file?
  vLLM v1 spawns the EngineCore in a subprocess. Env-var values are snapshotted
  at fork time and cannot be updated in the running child. A sentinel path is
  shared across processes, the launcher creates/removes it on exact benchmark
  boundaries, and the worker reads it lock-free per call.

Launch (via run_vtune_vllm.sh, but for reference):
  vtune -collect gpu-hotspots -start-paused -result-dir ./out \
    -- python serve_with_vtune.py --model <id> --port 8000
"""

import atexit
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [vtune.vllm] %(message)s",
)
log = logging.getLogger("vtune.vllm")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vtune_itt import (  # noqa: E402
    init_paused, itt_resume, itt_pause, itt_available,
    domain_create, task_begin, task_end,
)

init_paused()
log.info("ITT available: %s", itt_available())

PHASE = os.environ.get("VTUNE_PHASE", "both").strip().lower()
if PHASE not in {"prefill", "decode", "mixed", "both"}:
    log.warning("VTUNE_PHASE=%r invalid; falling back to 'both'", PHASE)
    PHASE = "both"

ROI_MODE = os.environ.get("VTUNE_ROI_MODE", "window").strip().lower()
if ROI_MODE not in {"window", "per_step", "per_step_isolate"}:
    log.warning("VTUNE_ROI_MODE=%r invalid; falling back to 'window'", ROI_MODE)
    ROI_MODE = "window"

PROFILE_WARMUP = os.environ.get("VTUNE_PROFILE_WARMUP", "0") == "1"
ROI_GATE_PATH = os.environ.get("VTUNE_ROI_GATE", "")

if not ROI_GATE_PATH and not PROFILE_WARMUP:
    log.warning(
        "VTUNE_ROI_GATE not set - worker will never enable ROI. "
        "Set VTUNE_ROI_GATE=<path> and create that file when benchmark starts, "
        "or set VTUNE_PROFILE_WARMUP=1 to profile unconditionally."
    )

# One ITT domain per phase bucket so the Tasks tab groups them cleanly.
_DOMAINS = {
    "window":  domain_create("vllm.window"),
    "prefill": domain_create("vllm.prefill"),
    "decode":  domain_create("vllm.decode"),
    "mixed":   domain_create("vllm.mixed"),
}


def _gate_open() -> bool:
    if PROFILE_WARMUP:
        return True
    if not ROI_GATE_PATH:
        return False
    return os.path.exists(ROI_GATE_PATH)


# ---------- Phase classification (vLLM v1 only) ----------


def _classify_v1(scheduler_output) -> tuple:
    """v1 SchedulerOutput -> (phase, batch_size, scheduled_tokens).

    phase is one of: prefill | decode | mixed | cache_hit | empty.
    See module docstring for definitions.
    """
    new = getattr(scheduler_output, "scheduled_new_reqs", None) or []
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    sched = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}

    # vLLM >= 0.10 passes a CachedRequestData object (a `req_ids` list) instead of a
    # list of request objects; iterating it raised TypeError, so every step was
    # classified "empty" and never profiled.
    if hasattr(cached, "req_ids"):
        cached_ids = list(cached.req_ids or [])
    else:
        cached_ids = [getattr(r, "req_id", None) for r in (cached or [])]
    cached_ids = [r for r in cached_ids if r is not None]

    new_tok = sum(sched.get(getattr(r, "req_id", None), 0) for r in new)
    cac_tok = sum(sched.get(r, 0) for r in cached_ids)
    cached = cached_ids
    bs = len(new) + len(cached)
    tok = new_tok + cac_tok

    if not new and not cached:
        return ("empty", bs, tok)
    if new and new_tok == 0 and not cached:
        return ("cache_hit", bs, tok)
    if new and cached:
        return ("mixed", bs, tok)
    if new:
        return ("prefill", bs, tok)
    return ("decode", bs, tok)


def _phase_selected(phase: str) -> bool:
    """Returns True if `phase` should be profiled given VTUNE_PHASE."""
    if phase in ("cache_hit", "empty"):
        return False
    if PHASE == "both":
        # Include mixed in 'both' so chunked-prefill steps aren't dropped silently.
        return phase in ("prefill", "decode", "mixed")
    if PHASE == "mixed":
        return phase == "mixed"
    if PHASE == "prefill":
        # In per_step_isolate mode, drop mixed steps from prefill stats too.
        if ROI_MODE == "per_step_isolate":
            return phase == "prefill"
        return phase in ("prefill", "mixed")
    if PHASE == "decode":
        if ROI_MODE == "per_step_isolate":
            return phase == "decode"
        # In window/per_step modes, mixed is treated as prefill-bearing
        # (decode kernels are still present but compute-bound prefill kernels
        # dominate), so excluded from decode stats by default.
        return phase == "decode"
    return False


def _step_label(phase: str, bs: int, tok: int) -> str:
    """Shape-tagged ITT task label. The label space is bounded (a few dozen
    distinct (bs, tok) tuples per run) so string_handle_get caches them all."""
    return f"{phase} bs={bs} tok={tok}"


# ---------- Window-mode ROI bookkeeping ----------

_window_open = False


def _open_window():
    """Open the ROI window the first time a profiled step is seen."""
    global _window_open
    if _window_open:
        return
    itt_resume()
    _window_open = True
    log.info("VTune ROI window OPENED (mode=%s, phase=%s)", ROI_MODE, PHASE)


def _close_window():
    """Close the ROI window at process exit. Safe to call multiple times."""
    global _window_open
    if not _window_open:
        return
    try:
        import torch
        if torch.xpu.is_available():
            torch.xpu.synchronize()
    except Exception as exc:
        log.warning("close_window: GPU sync failed: %s", exc)
    itt_pause()
    _window_open = False
    log.info("VTune ROI window CLOSED")


atexit.register(_close_window)


# ---------- Patching ----------

def _patch_worker(worker_cls, classify):
    import torch
    if getattr(worker_cls.execute_model, "_vtune_patched", False):
        return  # already patched in this process (sitecustomize + __main__)
    orig = worker_cls.execute_model

    def wrapped(self, *args, **kwargs):
        req = args[0] if args else kwargs.get("scheduler_output")
        try:
            phase, bs, tok = classify(req) if req is not None else ("empty", 0, 0)
        except Exception as e:
            log.debug("phase classify failed: %s", e)
            phase, bs, tok = ("empty", 0, 0)

        gate = _gate_open()
        active = gate and _phase_selected(phase)

        if not active:
            return orig(self, *args, **kwargs)

        domain = _DOMAINS.get(phase) or _DOMAINS["window"]
        label = _step_label(phase, bs, tok)

        if ROI_MODE == "window":
            _open_window()
            task_begin(domain, label)
            try:
                return orig(self, *args, **kwargs)
            finally:
                # Per-step sync keeps task end timestamps accurate even though
                # the outer resume/pause pair only fires once per process.
                if torch.xpu.is_available():
                    torch.xpu.synchronize()
                task_end(domain)
        else:
            # per_step or per_step_isolate
            itt_resume()
            task_begin(domain, label)
            try:
                return orig(self, *args, **kwargs)
            finally:
                if torch.xpu.is_available():
                    torch.xpu.synchronize()
                task_end(domain)
                itt_pause()

    wrapped._vtune_patched = True
    worker_cls.execute_model = wrapped
    log.info("Patched %s.execute_model (phase=%s mode=%s pid=%d)",
             worker_cls.__module__ + "." + worker_cls.__name__, PHASE, ROI_MODE, os.getpid())


def _try_patch():
    """Patch the vLLM v1 GPU Worker class. This wrapper is v1-only; if the
    import fails we refuse to start rather than silently profiling everything."""
    try:
        from vllm.v1.worker.gpu_worker import Worker as WorkerV1  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "vllm.v1.worker.gpu_worker.Worker not importable — this wrapper "
            "targets vLLM v1 only. Upgrade to a build that exposes the v1 "
            "engine (intel/vllm >= 0.14.1-xpu, upstream vLLM >= 0.10)."
        ) from e
    _patch_worker(WorkerV1, _classify_v1)


_try_patch()
log.info(
    "ROI gate: %s (gate=%s, profile_warmup=%s, mode=%s, phase=%s)",
    "OPEN" if _gate_open() else "CLOSED",
    ROI_GATE_PATH or "<unset>",
    PROFILE_WARMUP, ROI_MODE, PHASE,
)


# ---------- Hand off to vLLM's normal entrypoint ----------

if __name__ == "__main__":
    # vLLM v1 does not export `main` from api_server. Use runpy to invoke the
    # module's __main__ block exactly like
    # `python -m vllm.entrypoints.openai.api_server`.
    import runpy
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module(
        "vllm.entrypoints.openai.api_server",
        run_name="__main__",
        alter_sys=True,
    )
