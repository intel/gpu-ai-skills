# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small analytical roofline helpers for LLM serving.

These helpers are intentionally runtime-free: no torch, no device probing,
no benchmark data. They express the shared model used by the recommender:
prefill is usually compute-roof work, decode is usually memory-roof work,
and batching changes aggregate throughput differently from per-request
decode-step latency.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Factors:
    mfu_min: float
    mfu_max: float
    bwe_min: float
    bwe_max: float


@dataclass(frozen=True)
class PhaseResult:
    name: str
    tokens: int
    batch: int
    ops: float
    bytes_moved: float
    arithmetic_intensity: float
    compute_limited_tok_s: tuple[float, float]
    memory_limited_tok_s: tuple[float, float]
    predicted_tok_s: tuple[float, float]
    latency_ms: tuple[float, float]
    average_ms_per_token: tuple[float, float]
    bottleneck: str


def factors_from_dict(row: dict) -> Factors:
    return Factors(
        mfu_min=float(row["mfu_min"]),
        mfu_max=float(row["mfu_max"]),
        bwe_min=float(row["bwe_min"]),
        bwe_max=float(row["bwe_max"]),
    )


def peak_ops_s(device: dict, quant: dict) -> float:
    tier = quant["compute_tier"]
    if tier not in device:
        raise ValueError(f"device has no compute tier {tier!r}")
    return float(device[tier]) * 1e12


def phase_roofline(
    *,
    name: str,
    tokens: int,
    batch: int,
    params: float,
    weight_bytes: float,
    kv_bytes_per_token: float,
    context_tokens: int,
    include_kv: bool,
    peak_ops_per_s: float,
    bandwidth_bytes_per_s: float,
    factors: Factors,
) -> PhaseResult:
    if name not in ("prefill", "decode"):
        raise ValueError("name must be 'prefill' or 'decode'")
    if tokens < 1:
        raise ValueError(f"{name} tokens must be >= 1")
    if batch < 1:
        raise ValueError(f"{name} batch must be >= 1")
    if params <= 0:
        raise ValueError("params must be positive")
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive")

    token_work = tokens * batch
    ops = 2 * params * token_work

    if name == "prefill":
        kv_bytes = kv_bytes_per_token * token_work if include_kv else 0.0
        bytes_moved = weight_bytes + kv_bytes
    else:
        # One decode step streams the weights once for the active batch.
        # KV-cache traffic scales with active context and batch.
        kv_bytes = kv_bytes_per_token * context_tokens * batch if include_kv else 0.0
        bytes_moved = (weight_bytes + kv_bytes) * tokens

    arithmetic_intensity = ops / bytes_moved if bytes_moved else float("inf")
    compute_low = peak_ops_per_s * factors.mfu_min / (2 * params)
    compute_high = peak_ops_per_s * factors.mfu_max / (2 * params)
    memory_low = bandwidth_bytes_per_s * factors.bwe_min * arithmetic_intensity / (2 * params)
    memory_high = bandwidth_bytes_per_s * factors.bwe_max * arithmetic_intensity / (2 * params)

    pred_low = min(compute_low, memory_low)
    pred_high = min(compute_high, memory_high)
    latency_low = token_work / pred_high * 1000
    latency_high = token_work / pred_low * 1000
    average_low = 1000 / pred_high
    average_high = 1000 / pred_low

    mid_compute = (compute_low + compute_high) / 2
    mid_memory = (memory_low + memory_high) / 2
    bottleneck = "compute" if mid_compute < mid_memory else "memory"

    return PhaseResult(
        name=name,
        tokens=tokens,
        batch=batch,
        ops=ops,
        bytes_moved=bytes_moved,
        arithmetic_intensity=arithmetic_intensity,
        compute_limited_tok_s=(compute_low, compute_high),
        memory_limited_tok_s=(memory_low, memory_high),
        predicted_tok_s=(pred_low, pred_high),
        latency_ms=(latency_low, latency_high),
        average_ms_per_token=(average_low, average_high),
        bottleneck=bottleneck,
    )
