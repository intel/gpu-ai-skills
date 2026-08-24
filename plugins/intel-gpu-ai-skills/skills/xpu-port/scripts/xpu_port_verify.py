#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify a CUDA-to-XPU port with a CPU FP64 reference.

The user supplies a Python builder string that ends by binding `out =
(model, inputs)` where:
  - model is an nn.Module
  - inputs is a dict[str, Tensor] (or an arbitrary positional tuple)

The script:
  1. exec()s the builder in a fresh namespace, copies the model to CPU
     FP64, runs forward → reference output (always tensor-or-tuple).
  2. Re-exec()s the builder in a fresh namespace, moves the model to
     the target device and dtype, runs forward → actual output.
  3. Reports max abs / max rel / mean abs diff per output, and pass /
     fail against per-dtype tolerance.

`PYTORCH_ENABLE_XPU_FALLBACK=0` is set unconditionally at import time
so silent CPU fallback always raises instead of hiding.

When `--no-xpu` is passed, the actual run is CPU at the target dtype.
Useful on a laptop without an Intel GPU; not a substitute for the
target box. Without `--no-xpu`, XPU must be available or the script
exits non-zero rather than silently running on CPU.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_XPU_FALLBACK"] = "0"

import torch
import torch.nn as nn


@contextlib.contextmanager
def _capture_stdout():
    """Redirect builder stdout to stderr so JSON on stdout stays clean."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old
        leaked = buf.getvalue()
        if leaked:
            sys.stderr.write(leaked)


DEFAULT_TOLERANCE = {
    "float32": (1e-5, 1e-6),
    "bfloat16": (1e-2, 1e-3),
    "float16": (5e-3, 1e-3),
}


def _build(code: str) -> tuple[nn.Module, Any]:
    ns: dict[str, Any] = {}
    with _capture_stdout():
        # Bandit B102 suppression justification: `code` is the builder snippet the user passes via
        # --builder/--builder-file. Executing it is this tool's documented
        # purpose (see module docstring); it is the user's own code running in
        # the user's own process at the user's own privilege.
        exec(compile(code, "<builder>", "exec"), ns)  # nosec B102
    if "out" not in ns:
        raise RuntimeError(
            "builder did not bind `out = (model, inputs)`"
        )
    model, inputs = ns["out"]
    if not isinstance(model, nn.Module):
        raise RuntimeError(
            f"builder model is {type(model)}, expected nn.Module"
        )
    return model, inputs


def _move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if isinstance(inputs, dict):
        return {
            k: _move_one(v, device, dtype) for k, v in inputs.items()
        }
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_move_one(v, device, dtype) for v in inputs)
    return _move_one(inputs, device, dtype)


def _move_one(t: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_floating_point():
        return t.to(device=device, dtype=dtype)
    return t.to(device=device)


def _run(model: nn.Module, inputs: Any) -> Any:
    model.eval()
    with torch.no_grad(), _capture_stdout():
        if isinstance(inputs, dict):
            return model(**inputs)
        if isinstance(inputs, (list, tuple)):
            return model(*inputs)
        return model(inputs)


def _flatten_to_tensors(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        return [v for v in out.values() if isinstance(v, torch.Tensor)]
    if isinstance(out, (list, tuple)):
        result: list[torch.Tensor] = []
        for v in out:
            result.extend(_flatten_to_tensors(v))
        return result
    return []


def _diff(ref: torch.Tensor, act: torch.Tensor) -> dict[str, float]:
    ref64 = ref.to(dtype=torch.float64, device="cpu")
    act64 = act.to(dtype=torch.float64, device="cpu")
    if ref64.shape != act64.shape:
        return {
            "shape_mismatch": True,
            "ref_shape": list(ref64.shape),
            "act_shape": list(act64.shape),
        }
    delta = (act64 - ref64).abs()
    denom = ref64.abs().clamp(min=1e-12)
    rel = (delta / denom).max().item()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "max_rel": rel,
        "shape": list(ref64.shape),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify a CUDA-to-XPU port: CPU FP64 reference vs target "
            "device + dtype."
        )
    )
    ap.add_argument(
        "--builder",
        required=False,
        help=(
            "Python source that ends with `out = (model, inputs)`. "
            "inputs is a dict, list, tuple, or single tensor."
        ),
    )
    ap.add_argument(
        "--builder-file",
        type=str,
        default=None,
        help=(
            "Path to a Python file whose contents are used instead of "
            "--builder. One of --builder or --builder-file is required. "
            "The file must still bind `out = ...`."
        ),
    )
    ap.add_argument(
        "--target-dtype",
        default="bfloat16",
        choices=sorted(DEFAULT_TOLERANCE.keys()),
        help="dtype to compare against the FP64 reference",
    )
    ap.add_argument(
        "--no-xpu",
        action="store_true",
        help=(
            "Compare CPU at target dtype vs CPU FP64 (laptop / CI use)."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol", type=float, default=None)
    args = ap.parse_args(argv)

    if not args.builder and not args.builder_file:
        ap.error("one of --builder or --builder-file is required")

    code = args.builder
    if args.builder_file:
        # read_text() closes the handle; a bare open(...).read() leaks it until
        # GC, which CPython happens to make prompt but is not guaranteed.
        code = Path(args.builder_file).read_text(encoding="utf-8")

    target_dtype = getattr(torch, args.target_dtype)
    rtol, atol = DEFAULT_TOLERANCE[args.target_dtype]
    if args.rtol is not None:
        rtol = args.rtol
    if args.atol is not None:
        atol = args.atol

    xpu_present = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not args.no_xpu and not xpu_present:
        print(
            "verify: XPU not available and --no-xpu not set; "
            "refusing to report a CPU-only verification as a pass",
            file=sys.stderr,
        )
        return 2
    have_xpu = (not args.no_xpu) and xpu_present

    # Reference: CPU FP64
    torch.manual_seed(args.seed)
    ref_model, ref_inputs = _build(code)
    ref_model = ref_model.to(device="cpu", dtype=torch.float64)
    ref_inputs_moved = _move_inputs(ref_inputs, torch.device("cpu"), torch.float64)
    ref_out = _run(ref_model, ref_inputs_moved)

    # Actual: target device + dtype
    target_device = torch.device("xpu") if have_xpu else torch.device("cpu")

    torch.manual_seed(args.seed)
    act_model, act_inputs = _build(code)
    act_model = act_model.to(device=target_device, dtype=target_dtype)
    act_inputs_moved = _move_inputs(act_inputs, target_device, target_dtype)
    try:
        act_out = _run(act_model, act_inputs_moved)
    except Exception as exc:
        json.dump(
            {
                "pass": False,
                "reason": "run_error",
                "error": str(exc),
                "device_under_test": str(target_device),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 1

    ref_tensors = _flatten_to_tensors(ref_out)
    act_tensors = _flatten_to_tensors(act_out)

    if len(ref_tensors) != len(act_tensors):
        json.dump(
            {
                "pass": False,
                "reason": "output_count_mismatch",
                "ref_count": len(ref_tensors),
                "act_count": len(act_tensors),
                "device_under_test": str(target_device),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 2

    results: list[dict[str, Any]] = []
    overall_pass = True
    for i, (r, a) in enumerate(zip(ref_tensors, act_tensors)):
        d = _diff(r, a)
        passed = (
            "shape_mismatch" not in d
            and d["max_abs"] <= atol + rtol * float(r.abs().max().clamp(min=1e-12))
        )
        d["pass"] = bool(passed)
        d["index"] = i
        if not passed:
            overall_pass = False
        results.append(d)

    summary = {
        "device_under_test": str(target_device),
        "fp64_reference": "cpu",
        "target_dtype": args.target_dtype,
        "rtol": rtol,
        "atol": atol,
        "seed": args.seed,
        "outputs": results,
        "pass": overall_pass,
        "have_xpu": have_xpu,
    }

    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
