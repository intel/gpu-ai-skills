# ITT API Reference for VTune Collection Control

## Overview

The Intel ITT (Instrumentation and Tracing Technology) API is the lowest-latency
mechanism for controlling VTune collection from within the profiled process.
It operates via a shared library (`libittnotify`) that VTune injects or pre-loads.

When VTune is NOT running, all ITT calls are **no-ops** — safe to leave in
production code (minor overhead from a null-pointer guard).

---

## 1. ITT API Hierarchy

```
Control API:
  __itt_resume()            <- Resume ALL collection (global to process)
  __itt_pause()             <- Pause ALL collection (global to process)
  __itt_detach()            <- Permanently detach (irreversible in session)

Task/Domain API (Timeline annotations):
  __itt_domain_create()     <- Named domain for task grouping
  __itt_task_begin()        <- Mark start of named task
  __itt_task_end()          <- Mark end of named task

Thread API:
  __itt_thread_set_name()   <- Name threads (shown in Platform view)

Frame API:
  __itt_frame_begin_v3()    <- Frame start marker (vertical lines in GUI)
  __itt_frame_end_v3()      <- Frame end marker
```

---

## 2. Python ittapi Package (Recommended)

```bash
pip install ittapi       # Intel official, actively maintained
# Requires: VTune >= 2023.0, Python >= 3.7
```

### Basic resume/pause

```python
import ittapi
import torch

# At process start — ensure paused (belt-and-suspenders with -start-paused)
ittapi.pause()

def run_inference_roi(model, input_ids):
    """Wrap any GPU computation as a VTune ROI."""
    ittapi.resume()
    
    output = model.forward(input_ids)
    
    # CRITICAL: synchronize before pause
    torch.xpu.synchronize()
    ittapi.pause()
    
    return output
```

### Task and Domain API (Timeline annotations)

```python
import ittapi
import torch

# Create domains once at module level (cheap handle objects)
inference_domain = ittapi.domain_create("vllm.inference")
prefill_domain   = ittapi.domain_create("vllm.prefill")
decode_domain    = ittapi.domain_create("vllm.decode")

# Intern string handles for zero-allocation task naming
prefill_str  = ittapi.string_handle_create("prefill_forward")
decode_str   = ittapi.string_handle_create("decode_step")
attention_str = ittapi.string_handle_create("attention_kernel")

def profile_prefill(model, input_ids):
    ittapi.resume()
    ittapi.task_begin(prefill_domain, prefill_str)
    
    output = model.forward(input_ids)
    
    torch.xpu.synchronize()
    ittapi.task_end(prefill_domain)
    ittapi.pause()
    return output

def profile_decode_step(model, input_ids, kv_cache):
    ittapi.resume()
    ittapi.task_begin(decode_domain, decode_str)
    
    output = model.forward(input_ids, past_key_values=kv_cache)
    
    torch.xpu.synchronize()
    ittapi.task_end(decode_domain)
    ittapi.pause()
    return output
```

### Thread Naming

```python
import ittapi

def setup_vtune_thread_names(rank: int):
    """Call at worker process startup for readable Timeline view."""
    ittapi.thread_set_name(f"vllm_worker_rank{rank}")
```

---

## 3. Production ctypes Wrapper (Zero Dependencies)

Drop `vtune_itt.py` into your project. All functions are safe to call
even when VTune is not running (no-ops via None guard).

```python
"""
vtune_itt.py — Production ITT wrapper with graceful fallback.
No pip dependencies required.
"""

import ctypes
import ctypes.util
import os
import logging
import functools
import contextlib
from typing import Optional, Callable, Tuple

logger = logging.getLogger(__name__)

# ── Library Discovery ──────────────────────────────────────────────────────

_SEARCH_PATHS = [
    # oneAPI standard install
    "/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so",
    # Standalone VTune Profiler
    "/opt/intel/vtune_profiler/lib64/libittnotify.so",
    # Custom install path from env
    os.path.join(
        os.environ.get("VTUNE_PROFILER_DIR", ""),
        "lib64/libittnotify.so"
    ),
]


def _find_libittnotify() -> Optional[ctypes.CDLL]:
    """Search for libittnotify.so in standard and custom paths."""
    # 1. Try ldconfig-managed paths
    name = ctypes.util.find_library("ittnotify")
    if name:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    # 2. Try known Intel install paths
    for path in _SEARCH_PATHS:
        if path and os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue

    logger.debug("libittnotify not found — ITT calls will be no-ops")
    return None


def _resolve_fn(lib: Optional[ctypes.CDLL], name: str) -> Optional[Callable]:
    """Safely resolve a function from the ITT library."""
    if lib is None:
        return None
    try:
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = []
        return fn
    except AttributeError:
        logger.debug(f"ITT symbol {name} not found in library")
        return None


_lib = _find_libittnotify()
_resume_fn = _resolve_fn(_lib, "__itt_resume")
_pause_fn  = _resolve_fn(_lib, "__itt_pause")

# ── Public Control API ─────────────────────────────────────────────────────

def itt_available() -> bool:
    """Return True if ITT library was found and resume/pause symbols resolved."""
    return _resume_fn is not None and _pause_fn is not None


def itt_resume():
    """Resume VTune collection. No-op if VTune is not attached."""
    if _resume_fn:
        _resume_fn()


def itt_pause():
    """Pause VTune collection. No-op if VTune is not attached."""
    if _pause_fn:
        _pause_fn()


# ── Context Manager ────────────────────────────────────────────────────────

@contextlib.contextmanager
def vtune_roi(sync_device: str = "xpu"):
    """
    Context manager for a VTune Region of Interest.

    Automatically resumes collection on entry, synchronizes the GPU,
    and pauses collection on exit (even if an exception occurs).

    Args:
        sync_device: "xpu" for Intel GPU, "cuda" for CUDA-compat layer,
                     "none" to skip synchronization (not recommended).

    Usage:
        with vtune_roi():
            model.forward(input_ids)

        with vtune_roi(sync_device="cuda"):
            model.forward(input_ids)
    """
    import torch

    itt_resume()
    try:
        yield
    finally:
        # GPU sync before pause — critical for kernel capture completeness
        if sync_device == "xpu":
            if torch.xpu.is_available():
                torch.xpu.synchronize()
        elif sync_device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elif sync_device != "none":
            logger.warning(f"Unknown sync_device: {sync_device}")
        
        itt_pause()


# ── Function Decorator ─────────────────────────────────────────────────────

def roi(sync_device: str = "xpu"):
    """
    Decorator that marks a function as a VTune Region of Interest.

    Usage:
        @roi()
        def execute_model(self, req):
            ...

        # One-off monkey-patching:
        Worker.execute_model = roi()(Worker.execute_model)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with vtune_roi(sync_device=sync_device):
                return fn(*args, **kwargs)
        return wrapper
    return decorator


# ── Startup Helper ─────────────────────────────────────────────────────────

def init_paused():
    """
    Call at process startup to ensure collection starts paused.
    Belt-and-suspenders when using vtune -start-paused.
    Also useful when attaching VTune to a running process.
    """
    if itt_available():
        itt_pause()
        logger.info("VTune ITT: collection initialized to paused state")
    else:
        logger.debug("VTune ITT: library not available, running without profiling control")
```

---

## 4. C Extension for Minimal Overhead (Advanced)

When Python ctypes overhead is unacceptable for sub-microsecond ROIs:

```c
/* vtune_ext.c — compile as Python C extension for direct symbol calls */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ittnotify.h>   /* from VTune SDK include path */

static PyObject* py_resume(PyObject* self, PyObject* args) {
    __itt_resume();
    Py_RETURN_NONE;
}

static PyObject* py_pause(PyObject* self, PyObject* args) {
    __itt_pause();
    Py_RETURN_NONE;
}

static PyObject* py_available(PyObject* self, PyObject* args) {
    /* __itt_resume is a macro that may resolve to a NULL stub */
    return PyBool_FromLong(1);
}

static PyMethodDef VTuneMethods[] = {
    {"resume",    py_resume,    METH_NOARGS, "Resume VTune collection"},
    {"pause",     py_pause,     METH_NOARGS, "Pause VTune collection"},
    {"available", py_available, METH_NOARGS, "Check if ITT is linked"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vtune_module = {
    PyModuleDef_HEAD_INIT, "vtune_ext", NULL, -1, VTuneMethods
};

PyMODINIT_FUNC PyInit_vtune_ext(void) {
    return PyModule_Create(&vtune_module);
}
```

```bash
# Build (adjust VTune SDK path as needed):
VTUNE_SDK=/opt/intel/oneapi/vtune/latest/sdk
gcc -shared -fPIC -O2 -o vtune_ext.so vtune_ext.c \
    $(python3-config --includes --ldflags) \
    -I${VTUNE_SDK}/include \
    -L${VTUNE_SDK}/lib64 \
    -littnotify \
    -Wl,-rpath,${VTUNE_SDK}/lib64
```

---

## 5. Frame API for Request Boundary Markers

Frame markers appear as vertical lines in VTune's Timeline view —
ideal for marking individual inference request boundaries.

```python
import ittapi

_frame_domain = ittapi.domain_create("vllm.requests")

class VTuneRequestFrame:
    """Context manager that marks each inference request as a VTune frame."""
    
    def __init__(self, enable: bool = True):
        self.enable = enable
    
    def __enter__(self):
        if self.enable:
            ittapi.frame_begin(_frame_domain)
        return self
    
    def __exit__(self, *args):
        if self.enable:
            import torch
            torch.xpu.synchronize()
            ittapi.frame_end(_frame_domain)

# Usage in request handler:
with VTuneRequestFrame():
    output = model.generate(input_ids)
```

---

## 6. ITT Overhead Reference Table

| API Call | With VTune Attached | Without VTune (no-op) |
|----------|--------------------|-----------------------|
| `__itt_resume` / `__itt_pause` (C) | 50-200 ns | 2-5 ns (null check) |
| `ittapi.resume()` (Python pip) | 300-800 ns | 10-20 ns |
| ctypes wrapper | 500 ns - 2 µs | 20-50 ns |
| `torch.xpu.synchronize()` | 1-50 µs (queue-depth dependent) | N/A |

For LLM inference (forward pass = 10-500 ms), ITT overhead is always negligible.
The `torch.xpu.synchronize()` call dominates but is required for correctness.

---

## 7. Diagnosing ITT Not Working

```python
# Quick diagnostic — run before vLLM server starts
from vtune_itt import itt_available
import os

print(f"ITT available: {itt_available()}")
print(f"VTUNE_PROFILER_DIR: {os.environ.get('VTUNE_PROFILER_DIR', 'not set')}")

# Check library search
import ctypes.util
lib = ctypes.util.find_library("ittnotify")
print(f"ldconfig finds libittnotify: {lib}")

# Check known paths
for path in [
    "/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so",
    "/opt/intel/vtune_profiler/lib64/libittnotify.so",
]:
    import os.path
    print(f"  {path}: {'EXISTS' if os.path.exists(path) else 'not found'}")
```
