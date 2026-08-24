#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Recommend a vLLM-XPU deployment config for an HF model on Intel Arc B-series GPUs.

Tier 1 only: analytical, no GPU, ~2 s. Tier 2 and Tier 3 live in
calibrate.py and verify.py because they launch Docker workloads.

Tier 1 is what most users want and ships with no stored performance data —
predictions come from the spec-sheet roofline applied to the user's target.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from common import (
    ModelDims,
    count_params,
    fetch_config,
    is_local_model,
    kv_bytes_per_token,
    parse_model_dims,
)
from roofline import factors_from_dict, peak_ops_s, phase_roofline

GB = 1024 ** 3
HERE = Path(__file__).resolve().parent.parent
HW = json.loads((HERE / "data" / "hardware.json").read_text())


def _looks_like_device(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    keys = {re.sub(r"[^a-z0-9]", "", str(k).lower()) for k in record}
    return bool(keys & {"deviceid", "devicename", "pcibdfaddress",
                        "bdfaddress", "drmdevice"})


def _count_json_devices(obj: object) -> int:
    if isinstance(obj, list):
        if obj and all(_looks_like_device(item) for item in obj):
            return len(obj)
        return sum(_count_json_devices(item) for item in obj)
    if isinstance(obj, dict):
        if _looks_like_device(obj):
            return 1
        return sum(_count_json_devices(value) for value in obj.values())
    return 0


def _count_text_devices(text: str) -> int:
    ids = set()
    for line in text.splitlines():
        m = re.search(r"\bDevice\s+ID\b\s*[:=]\s*(\d+)", line, re.IGNORECASE)
        if m:
            ids.add(int(m.group(1)))
    return len(ids)


def discover_visible_xpus() -> int:
    """Return the number of host-visible Intel XPUs from xpu-smi discovery."""
    override = os.environ.get("INTEL_SKILLPACK_FAKE_XPU_COUNT")
    if override:
        try:
            count = int(override)
        except ValueError as exc:
            raise RuntimeError("INTEL_SKILLPACK_FAKE_XPU_COUNT must be an integer") from exc
        if count < 0:
            raise RuntimeError("INTEL_SKILLPACK_FAKE_XPU_COUNT must be non-negative")
        return count

    exe = shutil.which("xpu-smi")
    if exe is None:
        raise RuntimeError("xpu-smi not found; run xpu-discover first or omit --discover-host")

    for cmd in ([exe, "discovery", "-j"], [exe, "discovery"]):
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True,
                                  text=True, timeout=15)
        except (subprocess.CalledProcessError, TimeoutError):
            continue
        if "-j" in cmd:
            try:
                count = _count_json_devices(json.loads(proc.stdout))
            except json.JSONDecodeError:
                count = 0
        else:
            count = _count_text_devices(proc.stdout)
        if count:
            return count
    raise RuntimeError("xpu-smi discovery returned no visible Intel XPUs")


def validate_requested_devices(requested: int) -> int:
    if requested < 1:
        raise RuntimeError("--num-devices must be >= 1")
    visible = discover_visible_xpus()
    if requested > visible:
        raise RuntimeError(
            f"--num-devices {requested} exceeds host-visible XPUs ({visible}). "
            "Run xpu-discover, lower --num-devices, or omit --discover-host "
            "only when planning for a different host."
        )
    return visible


def quant_from_config(cfg: dict) -> str | None:
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None
    method = str(qcfg.get("quant_method") or "").lower().replace("_", "-")
    fmt = str(qcfg.get("format") or "").lower().replace("_", "-")
    if method == "auto-round":
        return "auto-round"
    if method in HW["quants"]:
        return method
    if fmt.startswith("auto-round:"):
        return "auto-round"
    return None


def _hf_search_quant_variants(model_id: str) -> dict[str, list[str]]:
    """Search HF for quantized siblings of the target model. Returns {quant: [ids]}.
    Best-effort: HF Hub doesn't expose a clean "is quantized as X" tag, so we
    search by name pattern. Misses are harmless — recommender just won't list.
    """
    org_model = model_id.rsplit("/", 1)[-1]
    found: dict[str, list[str]] = {}
    for tag, query in (
        ("awq", f"{org_model} awq"),
        ("gptq", f"{org_model} gptq"),
        ("auto-round", f"{org_model} autoround"),
        ("auto-round", f"{org_model} auto-round"),
    ):
        url = f"https://huggingface.co/api/models?search={urllib.parse.quote(query)}&limit=5"
        try:
            # Bandit B310 suppression justification: https://huggingface.co literal; only the search query
            # is interpolated, and it is url-quoted above.
            with urllib.request.urlopen(url, timeout=15) as r:  # nosec B310
                for hit in json.loads(r.read()):
                    mid = hit.get("modelId") or hit.get("id")
                    if mid and tag.replace("-", "") in mid.lower().replace("-", ""):
                        found.setdefault(tag, []).append(mid)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            continue
    # Dedupe and trim
    for k in list(found):
        seen = []
        for v in found[k]:
            if v not in seen:
                seen.append(v)
        found[k] = seen[:3]
    return found


def available_quants(model_id: str, cfg: dict, hub_variants: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return quant kinds that are legitimate candidates.

    bf16 and fp8 are runtime paths on the base checkpoint. Pre-quantized
    weight-only formats need either the current checkpoint config to declare
    the method or a discovered sibling checkpoint.
    """
    available: dict[str, list[str]] = {
        "bf16": [model_id],
        "fp8": [model_id],
    }
    detected = quant_from_config(cfg)
    if detected:
        available.setdefault(detected, [model_id])
    for quant, models in hub_variants.items():
        if models:
            available.setdefault(quant, models)
    return available


# ---- Fit and capacity ----

@dataclass
class FitResult:
    fits: bool
    weights_gb: float
    kv_at_target_gb: float
    activations_gb: float
    framework_gb: float
    total_gb: float
    headroom_gb: float
    max_concurrency: int    # at the target context
    max_context: int        # at the target concurrency


@dataclass
class Candidate:
    quant: str
    kv_dtype: str
    tp: int
    dp: int
    fit: FitResult
    ttft_ms: tuple[float, float]
    decode_step_ms: tuple[float, float]
    aggregate_decode_tps: tuple[float, float]
    decode_bottleneck: str
    checkpoints: list[str]
    caveats: list[str]


FRAMEWORK_OVERHEAD_GB = {"vllm-xpu": 2.0}


def evaluate_fit(d: ModelDims, params: int, quant: str, kv_dtype: str,
                 ctx: int, concurrency: int, tp: int,
                 device_vram_gb: float, framework: str,
                 gpu_memory_utilization: float = 1.0) -> FitResult:
    bpp = HW["quants"][quant]["bytes_per_param"]
    weights = (params * bpp) / tp
    kv_per_tok = kv_bytes_per_token(d, _kv_bytes(kv_dtype)) / tp
    kv = kv_per_tok * ctx * concurrency
    # Activation buffer: bounded estimate (residual + scratch).
    act = 2 * concurrency * ctx * d.hidden * 2 + 512 * 1024 ** 2
    fw = FRAMEWORK_OVERHEAD_GB[framework] * GB
    total = weights + kv + act + fw
    device_b = device_vram_gb * GB * gpu_memory_utilization
    headroom = device_b - total

    # Capacity: how many requests fit at this context, OR how much context at this concurrency.
    free = device_b - weights - act - fw
    max_conc = max(0, int(free / max(kv_per_tok * ctx, 1)))
    max_ctx = max(0, int(free / max(kv_per_tok * max(concurrency, 1), 1)))
    return FitResult(
        fits=total <= device_b,
        weights_gb=weights / GB,
        kv_at_target_gb=kv / GB,
        activations_gb=act / GB,
        framework_gb=fw / GB,
        total_gb=total / GB,
        headroom_gb=headroom / GB,
        max_concurrency=max_conc,
        max_context=max_ctx,
    )


def _kv_bytes(kv_dtype: str) -> float:
    return {"auto": 2.0, "bfloat16": 2.0, "fp8": 1.0,
            "fp8_e4m3": 1.0, "fp8_e5m2": 1.0}.get(kv_dtype, 2.0)


# ---- Known-bad / known-good combo flags ----

def known_caveats(quant: str, runtime: str) -> list[str]:
    notes = []
    if quant in ("awq", "auto-round", "inc", "gptq"):
        notes.append(
            "Set --attention-backend TRITON_ATTN — W4A8 kernels "
            "(int4_gemm_w4a8) route through Triton attention. Verify the live "
            "kernel via 'docker logs <name> | grep int4_gemm'; fall-through to "
            "W4A16 is the failure mode (vLLM #38064)."
        )
    if quant == "auto-round":
        notes.append(
            "AutoRound is detected from checkpoint metadata; omit --quantization "
            "and let vLLM read quantization_config."
        )
    if quant == "fp8":
        notes.append(
            "Pair with --kv-cache-dtype fp8 and --attention-backend TRITON_ATTN. "
            "FlashAttention-XPU rejects fp8 KV (NotImplementedError)."
        )
    if quant == "gptq" and runtime == "vllm-xpu":
        notes.append(
            "GPTQ regressed on vLLM 0.19.0 (vLLM #39474). Use vLLM ≤ 0.18.x or "
            "an AutoRound-AWQ-format checkpoint instead."
        )
    return notes


# ---- Candidate ranking ----

def _tp_options(num_devices: int) -> list[int]:
    opts = []
    tp = 1
    while tp <= max(num_devices, 1):
        if num_devices % tp == 0:
            opts.append(tp)
        tp *= 2
    return opts or [1]


def rank_candidates(d: ModelDims, params: int, ctx: int, concurrency: int,
                    device: dict, framework: str, available_variants: dict,
                    num_devices: int,
                    gpu_memory_utilization: float) -> list[Candidate]:
    factors = factors_from_dict(HW["framework_factors"][framework])
    candidates = []

    # Order to prefer: safe fallback first, then memory-saving and quantized paths.
    quant_order = ["bf16", "fp8", "auto-round", "awq", "gptq", "inc", "mxfp4"]

    for quant in quant_order:
        if quant not in HW["quants"]:
            continue
        # Pre-quantized formats are legitimate only when the current
        # checkpoint declares them or a sibling checkpoint was found.
        if quant not in ("bf16", "fp8") and quant not in available_variants:
            continue
        kv_dtype = "fp8" if quant != "bf16" else "bfloat16"

        for tp in _tp_options(num_devices):
            dp = max(1, num_devices // tp)
            fit = evaluate_fit(d, params, quant, kv_dtype, ctx, concurrency,
                               tp, device["vram_gb"], framework,
                               gpu_memory_utilization)
            if not fit.fits:
                continue

            qmeta = HW["quants"][quant]
            params_per_rank = params / tp
            weights_per_rank = params_per_rank * qmeta["bytes_per_param"]
            kv_per_tok = kv_bytes_per_token(d, _kv_bytes(kv_dtype)) / tp
            roof_kwargs = {
                "params": params_per_rank,
                "weight_bytes": weights_per_rank,
                "kv_bytes_per_token": kv_per_tok,
                "context_tokens": ctx,
                "include_kv": True,
                "peak_ops_per_s": peak_ops_s(device, qmeta),
                "bandwidth_bytes_per_s": device["memory_bandwidth_gbs"] * 1e9,
                "factors": factors,
            }
            prefill = phase_roofline(
                name="prefill",
                tokens=ctx,
                batch=1,
                **roof_kwargs,
            )
            decode = phase_roofline(
                name="decode",
                tokens=1,
                batch=max(concurrency, 1),
                **roof_kwargs,
            )

            candidates.append(Candidate(
                quant=quant,
                kv_dtype=kv_dtype,
                tp=tp,
                dp=dp,
                fit=fit,
                ttft_ms=prefill.latency_ms,
                decode_step_ms=decode.latency_ms,
                aggregate_decode_tps=(
                    decode.predicted_tok_s[0] * dp,
                    decode.predicted_tok_s[1] * dp,
                ),
                decode_bottleneck=decode.bottleneck,
                checkpoints=available_variants.get(quant, []),
                caveats=known_caveats(quant, framework),
            ))

    # Prefer layouts that use DP replicas when the model fits per GPU,
    # because replicas scale aggregate throughput/concurrency without
    # adding TP communication. Then keep high-headroom candidates.
    candidates.sort(key=lambda c: (-c.dp, c.tp, -c.fit.headroom_gb))
    return candidates[:4]


def _model_needs_trust_remote_code(cfg: dict) -> bool:
    """True if the HF config declares custom modeling code (auto_map)."""
    if not isinstance(cfg, dict):
        return False
    if cfg.get("auto_map"):
        return True
    text = cfg.get("text_config")
    return isinstance(text, dict) and bool(text.get("auto_map"))


def emit_launch_line(model_id: str, quant: str, kv_dtype: str, tp: int,
                     ctx: int, available_variants: dict, cfg: dict,
                     dp: int = 1, gpu_memory_utilization: float = 0.85,
                     trust_remote_code: bool = False,
                     revision: str | None = None) -> str:
    """Emit a copy-pasteable docker + vllm serve line.

    ``trust_remote_code`` is the caller's explicit opt-in. It is deliberately
    NOT inferred from the model's own ``config.json``: ``auto_map`` is a field
    the untrusted artifact controls, so letting it switch on arbitrary code
    execution would invert the rule that less-secure features require a
    deliberate customer action. When the model declares repo-local code and the
    caller has not opted in, the emitted output says so and stops short of the
    flag.

    ``revision`` pins what is served. Unpinned, vLLM resolves the repo's default
    branch at launch, so the artifact that runs is whatever the publisher last
    pushed -- and "I reviewed this repo" stops being a statement about the code
    that will execute. That matters for the weights alone; it is the whole
    mitigation once ``--trust-remote-code`` is in play, which is why main()
    requires a revision for Hub ids in that case. When both apply the emitted
    line carries ``--revision`` and ``--code-revision``: vLLM resolves
    repo-local modeling code separately from the weights, so pinning one does
    not pin the other.
    """
    # Use the AutoRound checkpoint if quant is auto-round, else the base model.
    served = available_variants.get(quant, [model_id])[0] if quant in available_variants else model_id

    base_env = ["VLLM_WORKER_MULTIPROC_METHOD=spawn"]
    if ctx > 32768:
        base_env.append("VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")

    quant_flag = ""
    needs_triton_attn = False
    if quant == "awq":
        quant_flag = "--quantization awq"
        needs_triton_attn = True
    elif quant == "gptq":
        quant_flag = "--quantization gptq"
        needs_triton_attn = True
    elif quant == "auto-round":
        quant_flag = ""
        needs_triton_attn = True
    elif quant == "inc":
        quant_flag = "--quantization inc"
        needs_triton_attn = True
    elif quant == "mxfp4":
        quant_flag = "--quantization mxfp4"
        needs_triton_attn = True
    elif quant == "fp8":
        quant_flag = "--quantization fp8"
        needs_triton_attn = True

    attn_flag = "--attention-backend TRITON_ATTN" if needs_triton_attn else ""
    tp_flag = f"--tensor-parallel-size {tp}" if tp > 1 else ""
    kv_flag = f"--kv-cache-dtype {kv_dtype}" if kv_dtype != "bfloat16" else ""

    # Detection and opt-in are separate concerns: the config only tells us the
    # model *would* need the flag, never that it may be granted.
    model_declares_remote_code = _model_needs_trust_remote_code(cfg)
    trc_flag = "--trust-remote-code" if (model_declares_remote_code
                                         and trust_remote_code) else ""

    # A local checkout is already a fixed artifact -- pinning a git revision of
    # it means nothing, so the flags are Hub-only.
    pin = revision if (revision and not is_local_model(model_id)) else ""
    rev_flag = f"--revision {pin}" if pin else ""
    # --code-revision only exists to pin remote code, so it is emitted only
    # where remote code is actually being granted.
    code_rev_flag = f"--code-revision {pin}" if (pin and trc_flag) else ""

    if trc_flag:
        if pin:
            trust_clause = f"trust the publisher at revision {pin}"
        elif is_local_model(model_id):
            trust_clause = "trust the source of this local checkout"
        else:
            trust_clause = "trust the publisher at that revision"
        trc_note = (
            "# SECURITY: --trust-remote-code is present because you passed\n"
            "#   --trust-remote-code and this model's config.json declares\n"
            "#   auto_map (repo-local modeling code). It permits arbitrary\n"
            "#   Python from the model repo to execute in the engine. Only run\n"
            f"#   this if you {trust_clause}.\n"
        )
        if code_rev_flag:
            trc_note += (
                "#   --revision and --code-revision pin both the weights and\n"
                "#   the modeling code to that revision, so the code you\n"
                "#   reviewed is the code that runs. Re-review before moving\n"
                "#   the pin.\n"
            )
    elif model_declares_remote_code:
        trc_note = (
            "# SECURITY: this model's config.json declares auto_map, so vLLM\n"
            "#   needs --trust-remote-code to load it -- which permits arbitrary\n"
            "#   Python from the model repo to run in the engine. The flag is\n"
            "#   omitted deliberately: re-run with --trust-remote-code to add it,\n"
            "#   only after you have reviewed the repo and trust the publisher.\n"
        )
    else:
        trc_note = ""

    # Serving a Hub id with no pin is not a defect, but it is a choice the user
    # should make knowingly rather than by omission.
    if not pin and not is_local_model(model_id):
        trc_note += (
            "# NOTE: no --revision, so vLLM resolves this repo's default branch\n"
            "#   at launch and a later push by the publisher changes what you\n"
            "#   serve. Pass --revision <commit-sha> to pin it.\n"
        )

    flags = " \\\n        ".join(filter(None, [
        "--dtype bfloat16" if quant in ("bf16", "fp16") else "",
        "--enforce-eager",
        "--block-size=64",
        trc_flag,
        rev_flag,
        code_rev_flag,
        f"--max-model-len {ctx}",
        f"--gpu-memory-utilization {gpu_memory_utilization:.2f}",
        quant_flag,
        kv_flag,
        attn_flag,
        tp_flag,
    ]))

    multi_xpu = tp > 1
    extra_env = ["CCL_ZE_IPC_EXCHANGE=pidfd"] if multi_xpu else []
    tp_extra_flag = " --disable-custom-all-reduce" if multi_xpu else ""

    def command(replica: int) -> str:
        start = replica * tp
        mask = ",".join(str(i) for i in range(start, start + tp))
        port = 8000 + replica
        name = "vllm-xpu" if dp == 1 else f"vllm-xpu-{replica}"
        env = [f"ZE_AFFINITY_MASK={mask}"] + base_env + extra_env
        env_lines = " \\\n        ".join(f"-e {e}" for e in env)
        return (
            f"docker run --rm -d --name {name} \\\n"
            f"    --device /dev/dri \\\n"
            f"    -v /dev/dri/by-path:/dev/dri/by-path:ro \\\n"
            f"    --group-add \"$(getent group render | cut -d: -f3)\" \\\n"
            f"    --ipc=host \\\n"
            f"        {env_lines} \\\n"
            f"    -e HF_TOKEN=\"$HF_TOKEN\" \\\n"
            f"    -v \"$HOME/.cache/huggingface:/root/.cache/huggingface\" \\\n"
            f"    -p {port}:8000 \\\n"
            f"    intel/vllm:0.17.0-xpu \\\n"
            f"    vllm serve {served} \\\n"
            f"        {flags}{tp_extra_flag}"
        )

    header = (
        "# Pick a current image tag from https://hub.docker.com/r/intel/vllm/tags\n"
        "# (0.17.0-xpu is a known-stable choice as of writing).\n"
        "# First-time debugging: drop '--rm' below so logs survive a crash.\n"
        "# After launch, verify no stale env vars:\n"
        "#   docker logs <name> 2>&1 | grep -i 'Unknown vLLM environment'\n"
    )
    if dp > 1:
        header += (
            f"# DP={dp}, TP={tp}: launch {dp} replicas, then route traffic "
            f"across ports 8000..{8000 + dp - 1}.\n"
        )
    return header + trc_note + "\n\n".join(command(replica) for replica in range(dp))


# ---- Output format ----

def fmt_band(lo: float, hi: float, unit: str) -> str:
    return f"{lo:.0f}–{hi:.0f} {unit}"


def quant_label(quant: str) -> str:
    if quant == "auto-round":
        return "AutoRound checkpoint (vLLM auto-detect; omit --quantization)"
    if quant == "bf16":
        return "bf16 (omit --quantization)"
    return f"--quantization {quant}"


def render(model_id: str, d: ModelDims, params: int, device: dict, framework: str,
           ctx: int, concurrency: int, candidates: list[Candidate],
           available_variants: dict, cfg: dict,
           gpu_memory_utilization: float,
           trust_remote_code: bool = False,
           revision: str | None = None) -> str:
    out = []
    out.append(f"Model:     {model_id}  ({params/1e9:.2f} B params, {d.family})")
    out.append(f"Device:    {device['name']}  ({device['vram_gb']} GB GDDR6, "
               f"{device['memory_bandwidth_gbs']} GB/s)")
    out.append(f"Memory:    {device['vram_gb'] * gpu_memory_utilization:.1f} GB usable "
               f"per GPU (gpu_memory_utilization={gpu_memory_utilization:.2f})")
    out.append(f"Runtime:   {framework} (vLLM-XPU launch/verification)")
    out.append(f"Devices:   {device.get('num_devices', 1)} x {device['name']}")
    out.append(f"Targets:   concurrency={concurrency}, context={ctx} tok")
    out.append("")
    quant_candidates = {k: v for k, v in available_variants.items()
                        if k not in ("bf16", "fp8")}
    if quant_candidates:
        out.append("Pre-quantized checkpoints available (config-detected or Hub search):")
        for q, lst in quant_candidates.items():
            if lst:
                out.append(f"  ✓ {q:12s} {lst[0]}")
        out.append("")
    if not candidates:
        out.append("No fitting candidates at these targets. Reduce concurrency or context.")
        return "\n".join(out)

    out.append("Candidate configs (predictions are physics-bounded ranges, "
               "not measurements):")
    out.append("")
    for i, c in enumerate(candidates, 1):
        f = c.fit
        layout = f"dp={c.dp}, tp={c.tp}"
        out.append(f"--- Candidate {i}: {quant_label(c.quant)}, --kv-cache-dtype "
                   f"{c.kv_dtype}, {layout} ---")
        out.append(f"  Fit:       {f.weights_gb:.1f} GB weights + "
                   f"{f.kv_at_target_gb:.1f} GB KV + "
                   f"{f.activations_gb:.1f} GB act + "
                   f"{f.framework_gb:.1f} GB framework "
                   f"= {f.total_gb:.1f} GB / "
                   f"{device['vram_gb'] * gpu_memory_utilization:.1f} GB usable per GPU "
                   f"(headroom {f.headroom_gb:.1f} GB)")
        out.append(f"  Capacity:  up to {f.max_concurrency} concurrent at "
                   f"{ctx} tok per replica; {f.max_concurrency * c.dp} "
                   f"aggregate across DP={c.dp}. OR {f.max_context:>5} tok at "
                   f"concurrency {concurrency} per replica  (memory-only ceiling; "
                   f"vLLM's --max-num-seqs and routing are separate knobs)")
        out.append(f"  Predicted: TTFT {fmt_band(*c.ttft_ms, 'ms')} "
                   f"(prefill of {ctx} tokens); "
                   f"decode step {fmt_band(*c.decode_step_ms, 'ms')} "
                   f"at concurrency {concurrency} per replica "
                   f"({c.decode_bottleneck}-limited); "
                   f"aggregate decode roughly {c.aggregate_decode_tps[0]:.0f}"
                   f"–{c.aggregate_decode_tps[1]:.0f} tok/s across DP={c.dp}")
        if c.caveats:
            for note in c.caveats:
                out.append(f"  Caveat:    {note}")
        if c.checkpoints:
            out.append(f"  Checkpoint: {c.checkpoints[0]}")
        out.append("")

    # Top-1 launch line
    top = candidates[0]
    out.append("=== Launch line for top candidate ===")
    out.append(emit_launch_line(model_id, top.quant, top.kv_dtype, top.tp,
                                ctx, available_variants, cfg, top.dp,
                                gpu_memory_utilization,
                                trust_remote_code, revision))
    out.append("")
    out.append("=== Verify before trusting ===")
    out.append("These are roofline-bounded predictions. Run vllm-xpu-bench against the launched")
    out.append("server with --num-prompts 30 --max-concurrency N to measure real TTFT and TPOT.")
    out.append("")
    out.append("Out of scope for this skill (verify yourself with bench skills):")
    out.append("  - Expert Parallelism (EP) for MoE")
    out.append("  - Speculative decoding speedup (n-gram, EAGLE, EAGLE3, MTP)")
    out.append("  - Tuning knobs: --block-size, --max-num-batched-tokens, --max-num-seqs")
    out.append("  - Load-balancer policy for DP replicas")
    return "\n".join(out)


# ---- CLI entry ----

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or local config.json path")
    p.add_argument("--device", default="arc-pro-b70",
                   choices=[d["id"] for d in HW["devices"]])
    p.add_argument("--num-devices", type=int, default=1)
    p.add_argument("--discover-host", action="store_true",
                   help="Check xpu-smi discovery before recommending; fail if "
                        "--num-devices exceeds host-visible Intel XPUs. Use "
                        "for deployment on the current host. Omit only for "
                        "planning a different host.")
    p.add_argument("--runtime", default="vllm-xpu", choices=("vllm-xpu",),
                   help="This recommender emits vLLM-XPU configs only.")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="Usable fraction of each GPU's VRAM for vLLM. "
                        "Passed through to the emitted --gpu-memory-utilization flag.")
    p.add_argument("--mode", default="recommend",
                   choices=("recommend",))
    p.add_argument("--no-hub-search", action="store_true",
                   help="Skip live HF Hub query for quant variants (offline mode).")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Security opt-in. Add --trust-remote-code to the emitted "
                        "launch line for models whose config.json declares "
                        "auto_map. This permits arbitrary Python from the model "
                        "repo to execute in the engine, so it is never inferred "
                        "from the model's own metadata -- you must pass it "
                        "deliberately, after reviewing the repo and deciding you "
                        "trust the publisher at that revision. Requires "
                        "--revision for Hub models, so the revision you "
                        "reviewed is the one that runs.")
    p.add_argument("--revision", metavar="REF",
                   help="Hub git revision -- commit SHA, tag, or branch -- to "
                        "read the config from and to pin the emitted launch "
                        "line to. Unset, vLLM resolves the repo's default "
                        "branch at launch, so a later push by the publisher "
                        "changes what you serve. A commit SHA is the only "
                        "value that cannot be moved under you. Ignored for a "
                        "local --model path.")
    p.add_argument("--use-calibration", metavar="IMAGE",
                   help="Anchor MFU/BWE to a previously-cached calibration "
                        "for this (image, device). Run scripts/calibrate.py "
                        "first to produce one.")
    args = p.parse_args(argv)

    if not (0 < args.gpu_memory_utilization <= 1.0):
        p.error("--gpu-memory-utilization must be > 0 and <= 1")

    # Granting arbitrary code execution against a moving branch is not a
    # decision anyone can make meaningfully: whatever the reviewer read, the
    # engine runs the publisher's latest push. Naming the revision is what turns
    # the opt-in into a statement about specific code, so it is required rather
    # than merely recommended. A local path has no branch to move, hence the
    # is_local_model() carve-out.
    if args.trust_remote_code and not args.revision and not is_local_model(args.model):
        p.error(
            "--trust-remote-code requires --revision for a Hub model. Without a "
            "pin, vLLM resolves the repo's default branch at launch and runs "
            "whatever the publisher last pushed -- which is not the code you "
            "reviewed. Pass the commit SHA you reviewed, e.g. "
            "--revision 1a2b3c4d. For a local checkout, pass a path to "
            "--model instead."
        )

    if args.discover_host:
        try:
            visible = validate_requested_devices(args.num_devices)
        except RuntimeError as exc:
            sys.exit(str(exc))
        print(f"# Host discovery: {visible} Intel XPU(s) visible via xpu-smi; "
              f"planning for {args.num_devices}.")
        print()
    elif args.num_devices < 1:
        sys.exit("--num-devices must be >= 1")

    # Calibration narrows the factor band to a single measured point per
    # framework. We replace the runtime's mfu_min/max and bwe_min/max in HW
    # with the measured values so render() shows tightened bands.
    if args.use_calibration:
        from pathlib import Path as _P
        cache = _P.home() / ".cache" / "intel-gpu-ai-skills" / "calibration"
        digest_part = args.use_calibration.replace(":", "_").replace("/", "_")[:64]
        # We accept either a literal image string (we'll glob-match) or a path.
        # A cache entry is a file on disk that anything could have touched since
        # calibrate.py wrote it, so treat it as untrusted: an unreadable or
        # non-UTF-8 entry is skipped rather than aborting the whole glob.
        def _cache_text(p: _P) -> str:
            try:
                return p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                return ""
        candidates = [p for p in cache.glob(f"{args.device}__*.json")
                      if digest_part in p.name
                      or args.use_calibration in _cache_text(p)]
        if not candidates:
            sys.exit(f"No calibration cached at {cache} for "
                     f"({args.device}, {args.use_calibration}). "
                     f"Run scripts/calibrate.py first.")
        try:
            cal = json.loads(_cache_text(candidates[0]))
            mfu_measured = float(cal["mfu_measured"])
            bwe_measured = float(cal["bwe_measured"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            sys.exit(f"calibration cache {candidates[0]} is corrupt "
                     f"({type(exc).__name__}: {exc}). Delete it and re-run "
                     f"scripts/calibrate.py.")
        f = HW["framework_factors"][args.runtime]
        f["mfu_min"] = f["mfu_max"] = mfu_measured
        f["bwe_min"] = f["bwe_max"] = bwe_measured
        print(f"# Using calibration from {candidates[0].name}: "
              f"MFU={mfu_measured:.2f} BWE={bwe_measured:.2f}")
        print()

    cfg = fetch_config(args.model, "model-config-recommend/0.1",
                       args.revision or "main")
    d = parse_model_dims(cfg)
    params = count_params(d)
    device = next(x for x in HW["devices"] if x["id"] == args.device)
    device = dict(device)
    device["num_devices"] = args.num_devices

    if args.no_hub_search:
        hub_variants = {}
    else:
        try:
            import urllib.parse  # noqa: F401  (used inside _hf_search)
            hub_variants = _hf_search_quant_variants(args.model)
        except Exception:
            hub_variants = {}
    available = available_quants(args.model, cfg, hub_variants)
    candidates = rank_candidates(d, params, args.ctx, args.concurrency,
                                 device, args.runtime, available,
                                 args.num_devices,
                                 args.gpu_memory_utilization)
    print(render(args.model, d, params, device, args.runtime,
                 args.ctx, args.concurrency, candidates, available, cfg,
                 args.gpu_memory_utilization,
                 args.trust_remote_code, args.revision))
    return 0


if __name__ == "__main__":
    import urllib.parse  # used by the search helper
    raise SystemExit(main())
