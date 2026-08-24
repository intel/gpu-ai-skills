#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM-XPU calibration.

Run a short reference bench on the user's host and image, measure the
actual MFU/BWE vLLM achieves, cache the result, then call recommend.py
with the calibrated factors.

The cached calibration is keyed by (image digest, device id). It auto-
invalidates when the image is re-pulled or the device changes.

Usage:
    python3 scripts/calibrate.py \\
        --image intel/vllm:0.17.0-xpu \\
        --device arc-pro-b70 \\
        --reference-model Qwen/Qwen2.5-1.5B-Instruct
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

from common import docker_image_id

HERE = Path(__file__).resolve().parent.parent
HW = json.loads((HERE / "data" / "hardware.json").read_text())
CACHE = Path.home() / ".cache" / "intel-gpu-ai-skills" / "calibration"


def image_digest(image: str) -> str:
    """Return docker image SHA digest (or the image string if inspect fails)."""
    return docker_image_id(image)


def cache_path(image: str, device_id: str) -> Path:
    digest = image_digest(image).replace(":", "_").replace("/", "_")[:64]
    return CACHE / f"{device_id}__{digest}.json"


def fetch_param_count(model_id: str) -> int:
    """Re-use the param counter from recommend.py via import."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recommend import fetch_config, parse_model_dims, count_params
    cfg = fetch_config(model_id, "model-config-recommend/0.1")
    d = parse_model_dims(cfg)
    return count_params(d)


def run_reference_bench(image: str, model_id: str, port: int = 8765,
                        warmup: int = 5, num_prompts: int = 30,
                        prompt_tokens: int = 512,
                        gen_tokens: int = 64) -> dict:
    """Launch a vLLM-XPU server, send warmup + timed bench, return measurements.

    Tries the model in BF16 (no quant) so we measure the BF16 compute tier
    cleanly. Returns medians of TTFT, TPOT, and computed implied MFU/BWE.
    """
    container = f"vllm-config-calibrate-{os.getpid()}"
    print(f"  starting {container} (image {image}, model {model_id})")
    cmd = [
        "docker", "run", "--rm", "-d", "--name", container,
        "--label", "intel-gpu-ai-skills.owner=calibrate",
        "--device", "/dev/dri",
        "--ipc=host",
        "-e", "ZE_AFFINITY_MASK=0",
        "-e", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        # MLA_DISABLE / ALLOW_LONG_MAX_MODEL_LEN / USE_V1 dropped: not needed
        # for the BF16 reference run (no MLA model, ctx=2048 fits, V1 default).
        "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
        "-p", f"{port}:8000",
        image,
        "vllm", "serve", model_id,
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.85",
    ]
    if os.environ.get("HF_TOKEN"):
        cmd[5:5] = ["-e", f"HF_TOKEN={os.environ['HF_TOKEN']}"]
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        # Wait for ready
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                # Bandit B310 suppression justification: loopback readiness poll against the vLLM server
                # this script just launched; host and scheme are literals.
                with urllib.request.urlopen(  # nosec B310
                        f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionResetError):
                time.sleep(5)
        else:
            raise RuntimeError("Server did not become ready within 10 minutes")

        # Warmup
        print(f"  warmup ({warmup} requests)")
        for _ in range(warmup):
            _curl_chat(port, model_id, prompt_tokens=128, max_tokens=16)

        # Timed bench
        print(f"  timed bench ({num_prompts} requests, "
              f"{prompt_tokens}-tok prompt, {gen_tokens}-tok gen)")
        ttfts, tpots = [], []
        for _ in range(num_prompts):
            ttft, tpot = _curl_chat(port, model_id,
                                    prompt_tokens=prompt_tokens,
                                    max_tokens=gen_tokens)
            ttfts.append(ttft)
            tpots.append(tpot)

        ttfts.sort(); tpots.sort()
        median = lambda xs: xs[len(xs)//2]
        return {
            "model": model_id,
            "params": fetch_param_count(model_id),
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "ttft_median_ms": median(ttfts),
            "tpot_median_ms": median(tpots),
        }
    finally:
        subprocess.run(["docker", "stop", container],
                       capture_output=True, check=False)


def _curl_chat(port: int, model: str, prompt_tokens: int,
               max_tokens: int) -> tuple[float, float]:
    """Send one streaming chat completion. Return (ttft_ms, tpot_ms)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "x " * prompt_tokens}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    last = None
    n = 0
    # Bandit B310 suppression justification: loopback request to the locally launched vLLM server
    # (http://127.0.0.1 literal above).
    with urllib.request.urlopen(req, timeout=120) as r:  # nosec B310
        for line in r:
            if not line.startswith(b"data:"):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            if b'[DONE]' in line:
                break
            n += 1
    if first is None or last is None or n <= 1:
        return 0.0, 0.0
    ttft_ms = (first - t0) * 1000
    tpot_ms = ((last - first) / max(n - 1, 1)) * 1000
    return ttft_ms, tpot_ms


def derive_factors(measurement: dict, device: dict) -> dict:
    """Back out actual MFU and BWE from the measured TTFT/TPOT."""
    params = measurement["params"]
    prompt = measurement["prompt_tokens"]
    ttft_s = measurement["ttft_median_ms"] / 1000
    tpot_s = measurement["tpot_median_ms"] / 1000

    # MFU: TTFT = prompt / (peak * MFU / (2 * params)) → MFU = (prompt * 2 * params) / (TTFT * peak)
    peak_flops = device["fp16_bf16_tflops"] * 1e12
    mfu = (prompt * 2 * params) / (ttft_s * peak_flops)
    mfu = max(0.05, min(mfu, 0.95))

    # BWE: TPOT = (weights / BW) / BWE → BWE = weights / (BW * TPOT)
    weights_bytes = params * 2  # bf16 = 2 bytes/param
    bw = device["memory_bandwidth_gbs"] * 1e9
    bwe = weights_bytes / (bw * tpot_s)
    bwe = max(0.05, min(bwe, 0.99))

    return {"mfu_measured": mfu, "bwe_measured": bwe,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "raw": measurement}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="vLLM-XPU image to calibrate against")
    p.add_argument("--device", required=True, choices=[d["id"] for d in HW["devices"]])
    p.add_argument("--reference-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Small model used as the calibration target")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--num-prompts", type=int, default=30)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--show", action="store_true",
                   help="Print the cached calibration if any and exit")
    args = p.parse_args(argv)

    if not shutil.which("docker"):
        sys.exit("docker is required for calibration. Install docker, then retry.")

    cf = cache_path(args.image, args.device)
    if args.show:
        if cf.exists():
            print(cf.read_text())
            return 0
        print(f"No calibration cached at {cf}")
        return 1

    device = next(d for d in HW["devices"] if d["id"] == args.device)
    print(f"Calibrating {device['name']} against {args.image}")
    print(f"Reference model: {args.reference_model}")
    print(f"Cache target:    {cf}")
    print()

    measurement = run_reference_bench(args.image, args.reference_model,
                                      port=args.port,
                                      num_prompts=args.num_prompts,
                                      prompt_tokens=args.prompt_tokens)
    factors = derive_factors(measurement, device)
    # Record the image string so recommend.py --use-calibration can find this
    # file by its content. The filename is keyed by the image *digest*, but the
    # consumer searches for the image *string* (or greps file content), so the
    # digest-named file is otherwise unreachable once docker resolves a digest.
    factors["image"] = args.image
    factors["device"] = args.device
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(factors, indent=2))
    print()
    print(f"Calibration written to {cf}")
    default = HW["framework_factors"]["vllm-xpu"]
    print(f"  measured MFU = {factors['mfu_measured']:.2f}  "
          f"(table default range was {default['mfu_min']:.2f}-{default['mfu_max']:.2f})")
    print(f"  measured BWE = {factors['bwe_measured']:.2f}  "
          f"(table default range was {default['bwe_min']:.2f}-{default['bwe_max']:.2f})")
    print()
    print("Run recommend.py with --use-calibration to anchor predictions to this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
