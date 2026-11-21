"""Apply serve_with_vtune's Worker.execute_model patch in every server process.

vLLM-XPU starts the EngineCore (which runs the GPU work) with `spawn`, so a patch
applied only in the API-server parent never reaches it. run_vtune_vllm.sh puts this
folder on PYTHONPATH and sets VTUNE_VLLM_PATCH=1; Python then imports this file at
start-up in every interpreter, including the spawned EngineCore.

Skipped in `vllm bench` clients (applying the patch there cut the capture to ~20 steps
in an end-to-end test on vLLM 0.17), in other helper processes (e.g. torch.compile workers),
and in any process where vLLM cannot be imported.
"""
import os
import sys

_argv = " ".join(getattr(sys, "orig_argv", sys.argv))
# Server processes only: the API server (runs serve_with_vtune.py) and the processes
# it spawns through multiprocessing (EngineCore, workers). Not compile workers or clients.
_server = "serve_with_vtune" in _argv or "multiprocessing" in _argv
if os.environ.get("VTUNE_VLLM_PATCH") == "1" and _server and "bench" not in _argv:
    try:
        import serve_with_vtune  # noqa: F401  (patches Worker.execute_model on import)
    except Exception as exc:  # never break an unrelated Python process
        print(f"[vtune.vllm] sitecustomize: patch not applied in pid {os.getpid()}: {exc}",
              file=sys.stderr)
