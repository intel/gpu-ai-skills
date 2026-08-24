#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Show the config-derived LLM roofline for one model and target shape.

This is analytical only: it fetches/parses config.json, estimates dense
decoder params and KV bytes from the model shape, then applies the Arc
B-series hardware table. It does not run Docker, torch, vLLM, or a GPU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from common import count_params, fetch_config, kv_bytes_per_token, parse_model_dims
from roofline import factors_from_dict, peak_ops_s, phase_roofline


HERE = Path(__file__).resolve().parent.parent
HW = json.loads((HERE / "data" / "hardware.json").read_text())


def kv_dtype_bytes(name: str) -> float:
    return {
        "auto": 2.0,
        "bfloat16": 2.0,
        "fp16": 2.0,
        "fp8": 1.0,
        "fp8_e4m3": 1.0,
        "fp8_e5m2": 1.0,
    }.get(name, 2.0)


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1000
    raise AssertionError("unreachable")


def fmt_band(values: tuple[float, float], suffix: str) -> str:
    return f"{values[0]:.1f}-{values[1]:.1f} {suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HF model id or local config.json path.")
    parser.add_argument("--device", default="arc-pro-b70",
                        choices=[d["id"] for d in HW["devices"]])
    parser.add_argument("--runtime", default="vllm-xpu",
                        choices=sorted(HW["framework_factors"]))
    parser.add_argument("--quant", default="bf16", choices=sorted(HW["quants"]))
    parser.add_argument("--kv-dtype", default="bfloat16",
                        choices=("auto", "bfloat16", "fp16", "fp8", "fp8_e4m3", "fp8_e5m2"))
    parser.add_argument("--ctx", type=int, default=4096,
                        help="Prompt/context tokens for prefill and decode KV traffic.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Active decode sequences per replica.")
    parser.add_argument("--decode-tokens", type=int, default=1,
                        help="Decode steps to model.")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.tensor_parallel < 1:
        raise SystemExit("--tensor-parallel must be >= 1")

    cfg = fetch_config(args.model, "llm-roofline/0.1")
    dims = parse_model_dims(cfg)
    params = count_params(dims)
    params_per_rank = params / args.tensor_parallel

    device = next(d for d in HW["devices"] if d["id"] == args.device)
    quant = HW["quants"][args.quant]
    factors = factors_from_dict(HW["framework_factors"][args.runtime])
    weight_bytes = params_per_rank * quant["bytes_per_param"]
    kv_per_token = kv_bytes_per_token(dims, kv_dtype_bytes(args.kv_dtype)) / args.tensor_parallel
    roof_kwargs = {
        "params": params_per_rank,
        "weight_bytes": weight_bytes,
        "kv_bytes_per_token": kv_per_token,
        "context_tokens": args.ctx,
        "include_kv": True,
        "peak_ops_per_s": peak_ops_s(device, quant),
        "bandwidth_bytes_per_s": device["memory_bandwidth_gbs"] * 1e9,
        "factors": factors,
    }
    prefill = phase_roofline(
        name="prefill",
        tokens=args.ctx,
        batch=1,
        **roof_kwargs,
    )
    decode = phase_roofline(
        name="decode",
        tokens=args.decode_tokens,
        batch=args.concurrency,
        **roof_kwargs,
    )

    if args.json:
        print(json.dumps({
            "model": args.model,
            "dims": asdict(dims),
            "params": params,
            "device": device,
            "runtime": args.runtime,
            "quant": quant,
            "kv_dtype": args.kv_dtype,
            "tensor_parallel": args.tensor_parallel,
            "weights_per_rank_bytes": weight_bytes,
            "kv_bytes_per_token_per_rank": kv_per_token,
            "prefill": asdict(prefill),
            "decode": asdict(decode),
        }, indent=2))
        return 0

    print(f"Model:     {args.model} ({params / 1e9:.2f}B params, {dims.family})")
    print(f"Shape:     hidden={dims.hidden}, layers={dims.num_layers}, "
          f"heads={dims.num_attn_heads}, kv_heads={dims.num_kv_heads}")
    if dims.is_moe:
        print("Note:      MoE config detected; dense param estimate is a coarse bound.")
    print(f"Device:    {device['name']} ({device['memory_bandwidth_gbs']} GB/s)")
    print(f"Runtime:   {args.runtime}")
    print(f"Quant:     {args.quant} ({quant['bytes_per_param']} B/param, "
          f"{quant['compute_tier']})")
    print(f"KV dtype:  {args.kv_dtype} ({kv_per_token / 1000:.1f} KB/token/rank)")
    print(f"TP:        {args.tensor_parallel}")
    print(f"Weights:   {fmt_bytes(weight_bytes)} per rank")
    print()
    print("Prefill")
    print(f"  bottleneck:    {prefill.bottleneck}")
    print(f"  moved bytes:   {fmt_bytes(prefill.bytes_moved)}")
    print(f"  intensity:     {prefill.arithmetic_intensity:.1f} ops/byte")
    print(f"  throughput:    {fmt_band(prefill.predicted_tok_s, 'tok/s')}")
    print(f"  phase latency: {fmt_band(prefill.latency_ms, 'ms')}")
    print()
    print(f"Decode ({args.concurrency} active sequence(s), {args.decode_tokens} step(s))")
    print(f"  bottleneck:    {decode.bottleneck}")
    print(f"  moved bytes:   {fmt_bytes(decode.bytes_moved)}")
    print(f"  intensity:     {decode.arithmetic_intensity:.1f} ops/byte")
    print(f"  aggregate:     {fmt_band(decode.predicted_tok_s, 'tok/s')}")
    print(f"  step latency:  {fmt_band(decode.latency_ms, 'ms')}")
    print()
    print("Interpretation: this is a config-derived roofline bound, not a benchmark.")
    print("Use calibration/bench skills to replace broad efficiency bands with measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
