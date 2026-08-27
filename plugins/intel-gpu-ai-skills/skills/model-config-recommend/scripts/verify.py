#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 verification. Take a recommended config, launch it on the actual
target model on the user's hardware + image, run a 30-prompt bench, print a
predicted-vs-measured table.

This uses a Python-stdlib HTTP client (no vllm/openai dep on the host) so the
script works from a fresh checkout without `pip install`. The methodology is
deliberately narrower than the canonical `vllm bench serve` flag set
documented in vllm-xpu-bench: sequential warmup + timed requests, median
TTFT/TPOT only, prefix caching defeated via per-request nonces. For full
percentile metrics or saturation throughput, use vllm-xpu-bench instead.

Usage:
    python3 scripts/verify.py \\
        --image vllm/vllm-openai-xpu:latest \\
        --device arc-pro-b70 \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --quant fp8 --kv-dtype fp8 \\
        --ctx 4096 --concurrency 4

Prints the predicted band from recommend.py alongside the measured medians.
A close match within the band means the recommender's math holds for this
(model, image, hardware) combination.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import docker_container_exists

HERE = Path(__file__).resolve().parent.parent
HW = json.loads((HERE / "data" / "hardware.json").read_text())


def _ensure_docker():
    if not shutil.which("docker"):
        sys.exit("docker is required for verify. Install docker, then retry.")


def _container_exists(name: str) -> bool:
    return docker_container_exists(name)


def predict_for(model_id: str, device_id: str, runtime: str, ctx: int,
                concurrency: int, quant: str, prompt_tokens: int | None = None) -> dict:
    """Run recommend.py logic for a specific quant; return its predicted band.

    prompt_tokens (if given) is used for the TTFT band — match the actual
    bench prompt size so predicted-vs-measured is apples-to-apples.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import kv_bytes_per_token
    from recommend import (fetch_config, parse_model_dims, count_params,
                           evaluate_fit, _kv_bytes)
    from roofline import factors_from_dict, peak_ops_s, phase_roofline

    cfg = fetch_config(model_id, "model-config-recommend/0.1")
    d = parse_model_dims(cfg)
    params = count_params(d)
    device = next(x for x in HW["devices"] if x["id"] == device_id)
    factors = factors_from_dict(HW["framework_factors"][runtime])
    qmeta = HW["quants"][quant]

    kv_dtype = "fp8" if quant != "bf16" else "bfloat16"
    fit = evaluate_fit(d, params, quant, kv_dtype, ctx, concurrency, 1,
                       device["vram_gb"], runtime)

    # Reproduce the recommender's roofline bands for tp=1: prefill latency is
    # the TTFT band, a single decode step is the TPOT band. Match the prompt
    # size the bench actually sends so predicted-vs-measured is apples-to-apples.
    weight_bytes = params * qmeta["bytes_per_param"]
    kv_per_tok = kv_bytes_per_token(d, _kv_bytes(kv_dtype))
    roof_kwargs = {
        "params": params,
        "weight_bytes": weight_bytes,
        "kv_bytes_per_token": kv_per_tok,
        "context_tokens": ctx,
        "include_kv": True,
        "peak_ops_per_s": peak_ops_s(device, qmeta),
        "bandwidth_bytes_per_s": device["memory_bandwidth_gbs"] * 1e9,
        "factors": factors,
    }
    prefill = phase_roofline(name="prefill", tokens=prompt_tokens or ctx,
                             batch=1, **roof_kwargs)
    # Tier 3's bench is single-stream (sequential loop in main()), so predict a
    # single decode step at batch=1 over the bench's working set — NOT the
    # deployment concurrency/ctx that rank_candidates (Tier 1) uses. This makes
    # the TPOT prediction apples-to-apples with the single-stream measurement.
    bench_ctx = prompt_tokens or ctx
    decode = phase_roofline(name="decode", tokens=1, batch=1,
                            **{**roof_kwargs, "context_tokens": bench_ctx})

    return {"fit": fit, "ttft_ms": prefill.latency_ms,
            "tpot_ms": decode.latency_ms,
            "kv_dtype": kv_dtype, "params": params}


def launch(image: str, model_id: str, quant: str, kv_dtype: str, ctx: int,
           concurrency: int, port: int, container: str) -> None:
    if _container_exists(container):
        sys.exit(
            f"Docker container '{container}' already exists. Pick another "
            f"--container name or stop it yourself if it is yours."
        )
    env = [
        ("ZE_AFFINITY_MASK", "0"),
        ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
    ]
    if ctx > 32768:
        env.append(("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1"))
    if os.environ.get("HF_TOKEN"):
        env.append(("HF_TOKEN", os.environ["HF_TOKEN"]))

    cmd = ["docker", "run", "--rm", "-d", "--name", container,
           "--label", "intel-gpu-ai-skills.owner=verify",
           "--device", "/dev/dri", "--ipc=host"]
    for k, v in env:
        cmd += ["-e", f"{k}={v}"]
    cmd += ["-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
            "-p", f"{port}:8000", image,
            model_id,
            "--enforce-eager",
            f"--max-model-len={ctx}",
            "--gpu-memory-utilization=0.85",
            # Verify must measure the un-cached path: prefix cache hits
            # would show TTFT as scheduler-only (~25 ms) instead of real
            # prefill compute, hiding the metric we're trying to verify.
            "--no-enable-prefix-caching"]
    needs_triton_attn = False
    if quant in ("awq", "gptq", "fp8", "inc", "mxfp4"):
        cmd += [f"--quantization={quant}", f"--kv-cache-dtype={kv_dtype}"]
        needs_triton_attn = True
    elif quant == "auto-round":
        cmd += [f"--kv-cache-dtype={kv_dtype}"]
        needs_triton_attn = True
    elif kv_dtype != "bfloat16":
        cmd += [f"--kv-cache-dtype={kv_dtype}"]
        if kv_dtype == "fp8":
            needs_triton_attn = True
    if needs_triton_attn:
        # FlashAttention-XPU rejects fp8 KV (NotImplementedError on Battlemage)
        # and W4A8 quantized matmul kernels route through Triton attention.
        cmd += ["--attention-backend=TRITON_ATTN"]
    subprocess.run(cmd, check=True, capture_output=True)


def wait_ready(port: int, timeout: int = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # Bandit B310 suppression justification: loopback readiness poll against the vLLM server this
            # script just launched; host and scheme are literals.
            with urllib.request.urlopen(  # nosec B310
                    f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"Server did not become ready within {timeout}s")


def streamed_chat(port: int, model: str, prompt_tokens: int,
                  max_tokens: int, seed: int | None = None) -> tuple[float, float, str]:
    # Each call gets a unique nonce so the prompts do not share a prefix
    # (defeats prefix-caching even if --no-enable-prefix-caching is not set).
    # Prompt length is bounded by approx_tokens, not by ctx, so the prompt
    # always fits regardless of model max-model-len.
    import random
    rng = random.Random(seed)
    nonce = " ".join(f"{rng.randint(0, 999999):06d}" for _ in range(8))
    # Aim for ~prompt_tokens tokens. ~3 chars per token is a rough heuristic
    # for English-text models; small overshoot is harmless.
    approx_chars = max(prompt_tokens, 64) * 3
    body = ("Briefly answer: count to ten in English. Context: "
            + ("the quick brown fox jumps over the lazy dog. " *
               (approx_chars // 47 + 1)))[:approx_chars]
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"{nonce} {body}"}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n = 0
    text = []
    # Bandit B310 suppression justification: loopback request to the locally launched vLLM server
    # (http://127.0.0.1 literal above).
    with urllib.request.urlopen(req, timeout=180) as r:  # nosec B310
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload_line = line[5:].strip()
            if payload_line == b"[DONE]":
                break
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            n += 1
            try:
                obj = json.loads(payload_line)
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    text.append(delta)
            except json.JSONDecodeError:
                # Deliberately ignored: a chunk that is not valid JSON still
                # counts as a token for the latency math above (n/first/last
                # are already updated), we just cannot recover its text. The
                # timing numbers are the point of this loop, so a malformed
                # or partial SSE frame must not abort the run.
                pass
    if first is None or last is None or n <= 1:
        return 0.0, 0.0, ""
    return ((first - t0) * 1000,
            ((last - first) / max(n - 1, 1)) * 1000,
            "".join(text)[:80])


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs)//2] if xs else 0.0


def in_band(measured: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= measured <= hi


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--device", required=True, choices=[d["id"] for d in HW["devices"]])
    p.add_argument("--model", required=True)
    p.add_argument("--quant", default="bf16",
                   choices=list(HW["quants"]))
    p.add_argument("--kv-dtype", default=None)
    p.add_argument("--runtime", default="vllm-xpu", choices=("vllm-xpu",),
                   help="Verification currently launches vLLM-XPU only.")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--num-prompts", type=int, default=30)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--container", default=None,
                   help="Container name to create. Defaults to a unique name.")
    args = p.parse_args(argv)

    _ensure_docker()
    bench_prompt_tokens = min(512, args.ctx // 4)
    pred = predict_for(args.model, args.device, args.runtime,
                       args.ctx, args.concurrency, args.quant,
                       prompt_tokens=bench_prompt_tokens)
    kv_dtype = args.kv_dtype or pred["kv_dtype"]

    container = args.container or f"vllm-config-verify-{os.getpid()}"

    print(f"Verifying {args.model}  quant={args.quant}  kv-dtype={kv_dtype}  "
          f"ctx={args.ctx}  c={args.concurrency}")
    print(f"Predicted (for ~{bench_prompt_tokens}-tok prompts):")
    print(f"  TTFT band: {pred['ttft_ms'][0]:.0f}–{pred['ttft_ms'][1]:.0f} ms")
    print(f"  TPOT band: {pred['tpot_ms'][0]:.1f}–{pred['tpot_ms'][1]:.1f} ms")
    print()
    print(f"Launching server (image {args.image})...")
    launch(args.image, args.model, args.quant, kv_dtype, args.ctx,
           args.concurrency, args.port, container)
    try:
        wait_ready(args.port)
        # Bound the prompt size to a fraction of ctx so it always fits.
        prompt_tokens = min(512, args.ctx // 4)
        print(f"Server ready. Warmup (5 requests at ~{prompt_tokens} tok prompt)")
        for i in range(5):
            streamed_chat(args.port, args.model, prompt_tokens, 16, seed=i)
        print(f"Timed bench ({args.num_prompts} prompts, ~{prompt_tokens} tok each)")
        ttfts, tpots = [], []
        sample = ""
        for i in range(args.num_prompts):
            t, p_, s = streamed_chat(args.port, args.model, prompt_tokens, 64,
                                     seed=1000 + i)
            if t and p_:
                ttfts.append(t); tpots.append(p_)
                if i == 0:
                    sample = s
        if not ttfts:
            sys.exit("No successful requests.")
        m_ttft = median(ttfts)
        m_tpot = median(tpots)
        print()
        print("Result:")
        print(f"  Measured TTFT median: {m_ttft:.0f} ms     "
              f"(predicted {pred['ttft_ms'][0]:.0f}–{pred['ttft_ms'][1]:.0f} ms"
              f"  → {'IN BAND' if in_band(m_ttft, pred['ttft_ms']) else 'OUT OF BAND'})")
        print(f"  Measured TPOT median: {m_tpot:.1f} ms     "
              f"(predicted {pred['tpot_ms'][0]:.1f}–{pred['tpot_ms'][1]:.1f} ms"
              f"  → {'IN BAND' if in_band(m_tpot, pred['tpot_ms']) else 'OUT OF BAND'})")
        print(f"  Sample completion: {sample!r}")
        if not (in_band(m_ttft, pred["ttft_ms"]) or in_band(m_tpot, pred["tpot_ms"])):
            print()
            print("Both predictions out of band on this stack. Either:")
            print("  - kernel maturity for this (quant, image) is below the assumed range, OR")
            print("  - the framework MFU/BWE table needs an update — re-run with calibrate.py first.")
        return 0
    finally:
        subprocess.run(["docker", "stop", container],
                       capture_output=True, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
