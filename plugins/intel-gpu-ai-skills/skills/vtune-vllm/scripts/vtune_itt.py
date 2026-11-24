"""
vtune_itt.py - ITT API wrapper for VTune ROI control.

Two backends, picked at import:
  1. Intel's official `ittapi` Python package (preferred when installed —
     it ships its own .so linked against libittnotify.a, so it works on
     installs that only provide the static archive).
  2. ctypes fallback that dlopens libittnotify.so from oneAPI / VTune
     install paths (zero-dependency, but only works if a shared .so exists).

If neither backend is available, all calls become safe no-ops so the
launcher still works under a normal (non-VTune) python invocation.

Public API (stable across backends):
  itt_resume(), itt_pause()    - VTune collection control
  itt_available()              - True if a backend was loaded
  init_paused()                - belt-and-suspenders pause at startup
  vtune_roi(sync_device)       - context manager (auto-sync + pause)
  roi(sync_device)             - decorator form

  Domain / task / string-handle primitives for ITT timeline tagging:
    domain_create(name)        -> opaque handle (cached)
    string_handle_get(name)    -> opaque handle (cached, allocation-free hot path)
    task_begin(domain, label)  -> mark start of a named task in domain
    task_end(domain)           -> mark end of the current task in domain
"""

import contextlib
import ctypes
import ctypes.util
import functools
import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Backend selector: prefer Intel's official ittapi package. It avoids the
# .so-vs-.a footgun on VTune installs that only ship libittnotify.a (e.g.,
# oneAPI 2026.0 - the file layout has /opt/intel/oneapi/vtune/2026.0/lib64/
# libittnotify.a but no libittnotify.so).
_BACKEND = None  # one of: "ittapi", "ctypes", None
_ittapi = None
try:
    import ittapi as _ittapi  # type: ignore
    _BACKEND = "ittapi"
    logger.info("vtune_itt: using ittapi package backend")
except ImportError:
    logger.debug("vtune_itt: ittapi package not installed; trying ctypes")

_SEARCH_PATHS = [
    "/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so",
    "/opt/intel/vtune_profiler/lib64/libittnotify.so",
    os.path.join(os.environ.get("VTUNE_PROFILER_DIR", ""), "lib64/libittnotify.so"),
]


def _find_libittnotify() -> Optional[ctypes.CDLL]:
    name = ctypes.util.find_library("ittnotify")
    if name:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    for path in _SEARCH_PATHS:
        if path and os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    return None


def _resolve(lib: Optional[ctypes.CDLL], name: str,
             restype=None, argtypes=None) -> Optional[Callable]:
    if lib is None:
        return None
    try:
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes or []
        return fn
    except AttributeError:
        return None


# ctypes backend - only used when ittapi package wasn't importable.
class _IttId(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint64),
                ("d2", ctypes.c_uint64),
                ("d3", ctypes.c_uint64)]


_ITT_NULL = _IttId(0, 0, 0)
_lib = None
_resume_fn = None
_pause_fn = None
_domain_create_fn = None
_string_handle_create_fn = None
_task_begin_fn = None
_task_end_fn = None

if _BACKEND is None:
    _lib = _find_libittnotify()
    if _lib is not None:
        # Signatures from .../sdk/include/ittnotify.h:
        #   __itt_domain* __itt_domain_create(const char*)
        #   __itt_string_handle* __itt_string_handle_create(const char*)
        #   void __itt_task_begin(__itt_domain*, __itt_id, __itt_id, __itt_string_handle*)
        #   void __itt_task_end(__itt_domain*)
        _resume_fn = _resolve(_lib, "__itt_resume")
        _pause_fn = _resolve(_lib, "__itt_pause")
        _domain_create_fn = _resolve(
            _lib, "__itt_domain_create",
            restype=ctypes.c_void_p, argtypes=[ctypes.c_char_p],
        )
        _string_handle_create_fn = _resolve(
            _lib, "__itt_string_handle_create",
            restype=ctypes.c_void_p, argtypes=[ctypes.c_char_p],
        )
        _task_begin_fn = _resolve(
            _lib, "__itt_task_begin",
            restype=None,
            argtypes=[ctypes.c_void_p, _IttId, _IttId, ctypes.c_void_p],
        )
        _task_end_fn = _resolve(
            _lib, "__itt_task_end",
            restype=None, argtypes=[ctypes.c_void_p],
        )
        if _resume_fn is not None and _pause_fn is not None:
            _BACKEND = "ctypes"
            logger.info("vtune_itt: using ctypes backend (libittnotify.so)")


# ---------- Public control API (backend-dispatched) ----------

def itt_available() -> bool:
    return _BACKEND is not None


def itt_resume():
    if _BACKEND == "ittapi":
        _ittapi.resume()
    elif _resume_fn is not None:
        _resume_fn()


def itt_pause():
    if _BACKEND == "ittapi":
        _ittapi.pause()
    elif _pause_fn is not None:
        _pause_fn()


def init_paused():
    if itt_available():
        itt_pause()
        logger.info("vtune_itt: collection paused at startup (backend=%s)", _BACKEND)
    else:
        logger.info("vtune_itt: no ITT backend available - calls are no-ops")


# ---------- Domain / task / string-handle (cached, allocation-free hot path) ----------
#
# Both backends cache by name; `ittapi` returns a Python object handle and
# the ctypes path returns a c_void_p. Either is fine to use as an opaque
# token in this module - callers don't introspect the value.

_domains: Dict[str, Any] = {}
_string_handles: Dict[str, Any] = {}


def domain_create(name: str):
    """Return cached opaque domain handle for `name`. None if ITT unavailable."""
    h = _domains.get(name)
    if h is not None:
        return h
    if _BACKEND == "ittapi":
        try:
            # ittapi >= 1.1: top-level domain factory.
            h = _ittapi.domain_create(name)
        except AttributeError:
            # Older ittapi: class form.
            h = _ittapi.Domain(name)
    elif _domain_create_fn is not None:
        h = _domain_create_fn(name.encode("utf-8"))
    _domains[name] = h
    return h


def string_handle_get(name: str):
    """Return cached opaque string-handle for `name`. None if ITT unavailable.
    Memoizes so high-frequency decode steps don't allocate a fresh handle
    every step (the per-shape label space is bounded)."""
    h = _string_handles.get(name)
    if h is not None:
        return h
    if _BACKEND == "ittapi":
        try:
            h = _ittapi.string_handle_create(name)
        except AttributeError:
            # Some ittapi builds expose StringHandle as a class only.
            h = _ittapi.StringHandle(name)
    elif _string_handle_create_fn is not None:
        h = _string_handle_create_fn(name.encode("utf-8"))
    _string_handles[name] = h
    return h


def task_begin(domain, label: str):
    """Begin a named task in `domain`. No-op if ITT unavailable."""
    if domain is None:
        return
    if _BACKEND == "ittapi":
        sh = string_handle_get(label)
        if sh is None:
            return
        try:
            _ittapi.task_begin(domain, sh)
        except AttributeError:
            # Older ittapi exposes Task as a context-manager factory; we
            # need the imperative form for symmetric begin/end calls.
            domain.task_begin(sh)
    elif _task_begin_fn is not None:
        sh = string_handle_get(label)
        if sh is None:
            return
        _task_begin_fn(domain, _ITT_NULL, _ITT_NULL, sh)


def task_end(domain):
    if domain is None:
        return
    if _BACKEND == "ittapi":
        try:
            _ittapi.task_end(domain)
        except AttributeError:
            domain.task_end()
    elif _task_end_fn is not None:
        _task_end_fn(domain)


# ---------- High-level helpers ----------

@contextlib.contextmanager
def vtune_roi(sync_device: str = "xpu"):
    """
    Resume collection on entry; synchronize the GPU and pause on exit.

    sync_device: "xpu" | "cuda" | "none" - "none" if the caller has already
    synchronized (skipping sync without a guarantee causes empty GPU
    timelines and zero GPU time in VTune results).
    """
    import torch
    itt_resume()
    try:
        yield
    finally:
        try:
            if sync_device == "xpu" and torch.xpu.is_available():
                torch.xpu.synchronize()
            elif sync_device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif sync_device not in ("xpu", "cuda", "none"):
                logger.warning("vtune_roi: unknown sync_device=%r", sync_device)
        except Exception as exc:
            logger.warning("vtune_roi: GPU sync failed before pause: %s", exc)
        itt_pause()


def roi(sync_device: str = "xpu"):
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with vtune_roi(sync_device=sync_device):
                return fn(*args, **kwargs)
        return wrapper
    return deco
