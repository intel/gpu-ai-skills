#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Estimate VRAM needed to serve a Hugging Face model on an Intel GPU.

Pulls config.json from the Hub, computes weights + KV cache +
activations + framework overhead, prints a verdict.

Decoder-only LLM only. VLM/diffusion fall through to a weights-only
floor with a clear caveat.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass


GB = 1024 ** 3
MB = 1024 ** 2
 

# Bytes per parameter, including typical scale/zero overhead at
# group_size=128 for grouped quants.
BYTES_PER_PARAM = {
    "bf16":  2.00,
    "fp16":  2.00,
    "fp8":   1.00,
    "int8":  1.00,
    "int4":  0.55,   # AWQ / GPTQ / AutoRound int4
    "int3":  0.42,
    "int2":  0.30,
    "mxfp4": 0.55,
}

BYTES_PER_KV = {
    "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
}

# Empirical floors on Arc Pro B70.
FRAMEWORK_OVERHEAD_GB = {
    "vllm":   2.0,
    "sglang": 1.5,
    "torch":  0.8,
}

TABLE_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
]


def _http_get_json(url: str) -> dict | None:
    """Fetch JSON from a URL. Returns None on 404; exits on other errors."""
    import os
    headers = {"User-Agent": "model-can-it-fit/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        # Bandit B310 suppression justification: url is built from the https://huggingface.co literal at
        # fetch_config(); scheme and host are not reachable from any parameter.
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        if e.code in (401, 403):
            sys.exit(
                f"HTTP {e.code} fetching {url}. Either the model id is "
                f"wrong (typo, case mismatch) or the model is gated / "
                f"private. Verify the page exists in a browser; if it "
                f"does, set HF_TOKEN (huggingface-cli login) and retry, "
                f"or pass a local path to config.json instead."
            )
        raise


def fetch_config(model_id: str, revision: str = "main") -> dict:
    """Fetch config.json or params.json from HF Hub or local path.

    For diffusion repos (model_index.json at root, no config.json),
    raises SystemExit with a useful message — diffusion pipelines have
    multi-component memory profiles this calculator cannot model.
    """
    if model_id.startswith(("/", ".")) or model_id.endswith(".json"):
        try:
            with open(model_id, encoding="utf-8") as f:
                cfg = json.load(f)
        except OSError as e:
            sys.exit(f"cannot read {model_id}: {e}")
        except UnicodeDecodeError as e:
            sys.exit(f"{model_id} is not valid UTF-8 text: {e}")
        except json.JSONDecodeError as e:
            sys.exit(f"{model_id} is not valid JSON: {e}")
        if not isinstance(cfg, dict):
            sys.exit(
                f"{model_id} must contain a JSON object, got "
                f"{type(cfg).__name__} — point --model at a config.json."
            )
        return cfg

    base = f"https://huggingface.co/{model_id}/raw/{revision}"
    cfg = _http_get_json(f"{base}/config.json")
    if cfg is not None:
        return cfg

    # No config.json: try params.json (some Mistral models use this)
    params = _http_get_json(f"{base}/params.json")
    if params is not None:
        return params

    # No config.json or params.json: check for diffusion-pipeline shape.
    midx = _http_get_json(f"{base}/model_index.json")
    if midx is not None:
        components = sorted(k for k, v in midx.items()
                            if isinstance(v, list) and v and v[0])
        raise SystemExit(
            f"{model_id} is a diffusion pipeline (model_index.json present, "
            f"no top-level config.json). This calculator does not estimate "
            f"diffusion VRAM — peak usage is dominated by intermediate "
            f"latents during denoising, which depend on resolution, step "
            f"count, and scheduler in ways a config-only calc cannot model.\n"
            f"\n"
            f"Components present: {', '.join(components) or '(empty)'}\n"
            f"\n"
            f"As a floor estimate of the largest component's weights, point "
            f"this script at one subdir directly, e.g.:\n"
            f"    --model {model_id}/unet            (or transformer for DiT)\n"
            f"For a real fit answer, run **torch-xpu-bench**'s diffusion "
            f"snippet with --runs 1 and read the Peak XPU memory line."
        )

    raise SystemExit(
        f"HTTP 404 fetching config for {model_id}. Neither config.json, "
        f"params.json, nor model_index.json exists at the root of revision "
        f"'{revision}'. Verify the model id and revision in a browser."
    )


@dataclass
class ModelDims:
    arch_family: str
    hidden: int
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied: bool
    is_moe: bool
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    dense_intermediate: int = 0
    is_vlm: bool = False
    vision_params: int = 0


def _vlm_signal(cfg: dict) -> bool:
    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    if "vision_config" in cfg:
        return True
    if any(s in arch for s in ("VL", "VisionLanguage", "VLForConditional")):
        return True
    if any(s in cfg.get("model_type", "") for s in ("_vl", "vlm", "vision")):
        return True
    return False


def _vision_param_count(vc: dict) -> int:
    """Approximate parameter count of a ViT-style vision tower."""
    h = vc.get("hidden_size", 0)
    layers = vc.get("depth") or vc.get("num_hidden_layers") or 0
    intermediate = vc.get("intermediate_size", 4 * h)
    if not (h and layers):
        return 0
    # ViT layer: attn (4 h^2) + MLP (2 h * intermediate) + 2 norms (2 h).
    per_layer = 4 * h * h + 2 * h * intermediate + 2 * h
    # Patch embedding: roughly hidden * patch_area * channels. Bound modestly.
    patch = vc.get("patch_size", 14)
    in_channels = vc.get("num_channels", 3)
    embed = patch * patch * in_channels * h
    return embed + layers * per_layer


def parse_dims(cfg: dict) -> ModelDims:
    arch = (cfg.get("architectures") or ["unknown"])[0]
    family = cfg.get("model_type", "unknown")
    is_vlm = _vlm_signal(cfg)

    # For VLMs the LLM-backbone fields often live under 'text_config' or 'llm_config'
    # rather than at the root.
    # text_config: Qwen2-VL, Llama-3.2-Vision, etc.
    # llm_config: Nemotron-3-Nano-Omni, some other multimodal models
    text_cfg = cfg.get("text_config") or cfg.get("llm_config") or cfg

    # Support both config.json and params.json field names
    # config.json uses: hidden_size, num_hidden_layers, num_attention_heads
    # params.json uses: dim, n_layers, n_heads (Mistral format)
    hidden = text_cfg.get("hidden_size") or text_cfg.get("d_model") or text_cfg.get("dim")
    num_layers = (
        text_cfg.get("num_hidden_layers")
        or text_cfg.get("n_layer")
        or text_cfg.get("num_layers")
        or text_cfg.get("n_layers")
    )
    num_attn = text_cfg.get("num_attention_heads") or text_cfg.get("n_head") or text_cfg.get("n_heads")
    num_kv = text_cfg.get("num_key_value_heads") or text_cfg.get("n_kv_heads") or num_attn

    # Head dimension - use explicit head_dim if present, otherwise calculate
    head_dim = text_cfg.get("head_dim")
    if head_dim is None:
        head_dim = cfg.get("head_dim")
    if head_dim is None:
        head_dim = hidden // num_attn if (hidden and num_attn) else 0

    vocab = text_cfg.get("vocab_size", cfg.get("vocab_size", 32000))
    # params.json uses "tied_embeddings", config.json uses "tie_word_embeddings"
    tied = bool(text_cfg.get("tie_word_embeddings",
                             cfg.get("tie_word_embeddings",
                             cfg.get("tied_embeddings", False))))

    vision_params = 0
    if is_vlm and isinstance(cfg.get("vision_config"), dict):
        vision_params = _vision_param_count(cfg["vision_config"])

    # MoE detection - do this BEFORE reading intermediate_size to avoid picking wrong field
    # Check multiple field names used by different MoE architectures
    # For VLMs, check text_config first (like other backbone fields)
    # params.json (Mistral) nests these under "moe" key
    moe_cfg = cfg.get("moe", text_cfg.get("moe", {}))

    # Use nested get() with defaults to avoid treating 0 as missing
    num_experts = (
        text_cfg.get("num_local_experts",
        text_cfg.get("num_experts",
        text_cfg.get("n_routed_experts",      # DeepSeek-V4
        cfg.get("num_local_experts",
        cfg.get("num_experts",
        cfg.get("n_routed_experts",
        moe_cfg.get("num_experts", 0)))))))   # params.json (Mistral)
    )
    num_experts_per_tok = (
        text_cfg.get("num_experts_per_tok",
        text_cfg.get("num_experts_per_token",
        cfg.get("num_experts_per_tok",
        cfg.get("num_experts_per_token",
        moe_cfg.get("num_experts_per_tok", 0)))))  # params.json (Mistral)
    )
    num_shared_experts = (
        text_cfg.get("n_shared_experts",
        cfg.get("n_shared_experts",
        moe_cfg.get("num_shared_experts", 0)))     # params.json (Mistral)
    )

    # Hybrid MoE: some models (DeepSeek-V2/V3/V4) replace first K layers with dense FFN
    first_k_dense_replace = (
        text_cfg.get("first_k_dense_replace",
        cfg.get("first_k_dense_replace",
        moe_cfg.get("first_k_dense_replace", 0)))
    )

    is_moe = num_experts and num_experts > 1

    # Intermediate size detection - AFTER MoE detection to pick correct field
    # For MoE models: use moe_intermediate_size (expert FFN size)
    # For dense models: use intermediate_size (standard FFN size)
    # Some dense models (BART/OPT/M2M) use ffn_dim, so only check it for non-MoE
    if is_moe:
        # MoE model: prefer moe_intermediate_size / expert_hidden_dim
        intermediate = (
            text_cfg.get("moe_intermediate_size")      # Qwen3 MoE, DeepSeek-V4
            or cfg.get("moe_intermediate_size")        # Root-level fallback
            or moe_cfg.get("expert_hidden_dim")        # params.json (Mistral)
            or text_cfg.get("ffn_dim")                 # Some MoE variants
            or cfg.get("ffn_dim")
            or text_cfg.get("hidden_dim")              # params.json fallback
            or cfg.get("hidden_dim")
            or text_cfg.get("intermediate_size")       # Fallback
        )
        # Dense FFN size for hybrid MoE (first_k_dense_replace layers)
        # params.json (Mistral) uses hidden_dim for dense FFN
        # config.json (DeepSeek) uses intermediate_size for dense FFN
        dense_intermediate = (
            text_cfg.get("intermediate_size")
            or cfg.get("intermediate_size")
            or text_cfg.get("hidden_dim")              # params.json (Mistral)
            or cfg.get("hidden_dim")
            or 0
        )
    else:
        # Dense model: prefer intermediate_size, then ffn_dim, then hidden_dim
        intermediate = (
            text_cfg.get("intermediate_size")          # Standard dense models
            or text_cfg.get("ffn_dim")                 # BART/OPT/M2M
            or cfg.get("ffn_dim")
            or text_cfg.get("hidden_dim")              # params.json
            or cfg.get("hidden_dim")
        )
        dense_intermediate = 0  # Not used for pure dense models

    # Fall back to 4*hidden if nothing found
    if intermediate is None:
        intermediate = 4 * (hidden or 0)

    missing = [k for k, v in {
        "hidden_size": hidden, "num_hidden_layers": num_layers,
        "num_attention_heads": num_attn,
    }.items() if not v]
    if missing:
        sys.exit(
            f"config.json is missing required keys for an LLM: {missing}. "
            f"Architecture reported: {arch} / {family}. "
            f"This calculator only handles decoder-only LLMs reliably."
        )

    return ModelDims(
        arch_family=family,
        hidden=hidden,
        num_layers=num_layers,
        num_attn_heads=num_attn,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        intermediate=intermediate,
        vocab=vocab,
        tied=tied,
        is_moe=bool(is_moe),
        num_experts=int(num_experts or 0),
        num_experts_per_tok=int(num_experts_per_tok or 0),
        num_shared_experts=int(num_shared_experts),
        first_k_dense_replace=int(first_k_dense_replace),
        dense_intermediate=int(dense_intermediate),
        is_vlm=is_vlm,
        vision_params=vision_params,
    )


def count_params(d: ModelDims) -> int:
    h = d.hidden
    # Q projection dimension: for models with explicit head_dim ≠ hidden/num_heads,
    # Q projects to num_attn_heads * head_dim (not h)
    q_proj_dim = d.num_attn_heads * d.head_dim
    kv_proj_dim = d.num_kv_heads * d.head_dim
    attn_block = (
        h * q_proj_dim     # Q: hidden → num_attn_heads * head_dim
        + h * kv_proj_dim  # K: hidden → num_kv_heads * head_dim
        + h * kv_proj_dim  # V: hidden → num_kv_heads * head_dim
        + q_proj_dim * h   # O: num_attn_heads * head_dim → hidden
    )
    norms = 4 * h

    if d.is_moe:
        # Routed experts (standard MoE)
        routed_ff = d.num_experts * 3 * h * d.intermediate
        # Shared experts (DeepSeek-V4, some Mixtral variants)
        shared_ff = d.num_shared_experts * 3 * h * d.intermediate
        moe_ff_block = routed_ff + shared_ff

        # Hybrid MoE: DeepSeek-V2/V3/V4 replace first K layers with dense FFN
        if d.first_k_dense_replace > 0 and d.dense_intermediate > 0:
            dense_ff_block = 3 * h * d.dense_intermediate
            dense_layers = d.first_k_dense_replace
            moe_layers = d.num_layers - dense_layers

            dense_per_layer = attn_block + dense_ff_block + norms
            moe_per_layer = attn_block + moe_ff_block + norms

            emb = d.vocab * h
            head = 0 if d.tied else d.vocab * h
            return emb + head + (dense_layers * dense_per_layer) + (moe_layers * moe_per_layer)
        else:
            # Pure MoE: all layers use expert FFN
            per_layer = attn_block + moe_ff_block + norms
    else:
        # Dense model
        ff_block = 3 * h * d.intermediate
        per_layer = attn_block + ff_block + norms

    emb = d.vocab * h
    head = 0 if d.tied else d.vocab * h
    return emb + head + d.num_layers * per_layer


def kv_bytes(d: ModelDims, ctx: int, concurrency: int, kv_dtype: str) -> int:
    per_token = 2 * d.num_layers * d.num_kv_heads * d.head_dim
    return int(per_token * ctx * concurrency * BYTES_PER_KV[kv_dtype])


def activation_bytes(d: ModelDims, ctx: int, concurrency: int, dtype: str) -> int:
    # ~2 hidden buffers' worth + 512 MiB scratch
    bytes_per = BYTES_PER_PARAM[dtype]
    return int(2 * concurrency * ctx * d.hidden * bytes_per) + 512 * MB


def fmt_gb(b: int) -> str:
    return f"{b / GB:6.2f} GB"


def calculate_mixed_precision_weights(cfg: dict, d: ModelDims, params: int,
                                       quant: str, tp: int) -> tuple[int, dict | None]:
    """Calculate weight bytes accounting for mixed-precision quantization.

    Some models (e.g., openai/gpt-oss-20b) use quantization_config.modules_to_not_convert
    to keep certain components at full precision while quantizing others.

    Returns:
        (weights_bytes, breakdown_dict or None)
    """
    qcfg = cfg.get("quantization_config", {})
    modules_to_not_convert = qcfg.get("modules_to_not_convert", [])

    # If no selective quantization, use uniform quantization
    if not modules_to_not_convert:
        return int(params * BYTES_PER_PARAM[quant] / tp), None

    # Parse which modules to keep at full precision
    keep_embeddings = any("embed" in m or "lm_head" in m for m in modules_to_not_convert)
    keep_attn = any("attn" in m for m in modules_to_not_convert)
    keep_router = any("router" in m for m in modules_to_not_convert)

    # If not selectively quantizing recognizable components, fall back to uniform
    if not (keep_embeddings or keep_attn):
        return int(params * BYTES_PER_PARAM[quant] / tp), None

    # Calculate component sizes
    h = d.hidden
    vocab = d.vocab
    layers = d.num_layers

    # Embeddings + LM head
    embed_params = vocab * h * (1 if d.tied else 2)

    # Attention blocks per layer
    q_proj_dim = d.num_attn_heads * d.head_dim
    kv_proj_dim = d.num_kv_heads * d.head_dim
    attn_per_layer = (
        h * q_proj_dim +      # Q
        h * kv_proj_dim +     # K
        h * kv_proj_dim +     # V
        q_proj_dim * h +      # O
        4 * h                 # norms
    )
    attn_params = layers * attn_per_layer

    # Router params (small, approximate as 2 * hidden per layer for MoE)
    router_params = 0
    if d.is_moe and keep_router:
        router_params = layers * 2 * h

    # FFN/Expert params = remaining
    ffn_params = params - embed_params - attn_params - router_params - d.vision_params

    # Calculate bytes (bf16 = 2.0 for non-quantized, quant for quantized)
    bf16_bpp = 2.0
    quant_bpp = BYTES_PER_PARAM[quant]

    embed_bytes = embed_params * (bf16_bpp if keep_embeddings else quant_bpp)
    attn_bytes = attn_params * (bf16_bpp if keep_attn else quant_bpp)
    router_bytes = router_params * (bf16_bpp if keep_router else quant_bpp)
    ffn_bytes = ffn_params * quant_bpp
    vision_bytes = d.vision_params * (bf16_bpp if keep_embeddings else quant_bpp)

    total_bytes = int((embed_bytes + attn_bytes + router_bytes + ffn_bytes + vision_bytes) / tp)

    breakdown = {
        "embed_params": embed_params,
        "embed_bytes": int(embed_bytes / tp),
        "embed_bpp": bf16_bpp if keep_embeddings else quant_bpp,
        "attn_params": attn_params,
        "attn_bytes": int(attn_bytes / tp),
        "attn_bpp": bf16_bpp if keep_attn else quant_bpp,
        "router_params": router_params,
        "router_bytes": int(router_bytes / tp),
        "router_bpp": bf16_bpp if keep_router else quant_bpp,
        "ffn_params": ffn_params,
        "ffn_bytes": int(ffn_bytes / tp),
        "ffn_bpp": quant_bpp,
    }

    return total_bytes, breakdown


def estimate(cfg: dict, quant: str, kv_dtype: str, ctx: int,
             concurrency: int, tp: int, runtime: str,
             device_vram_gb: float, gpu_memory_utilization: float = 1.0) -> dict:
    d = parse_dims(cfg)
    params = count_params(d) + d.vision_params

    # Calculate weights with mixed-precision support
    weights, mixed_breakdown = calculate_mixed_precision_weights(cfg, d, params, quant, tp)

    # KV cache is sharded by attention heads (divided by TP)
    # Each GPU stores KV cache only for its subset of heads
    kv = kv_bytes(d, ctx, concurrency, kv_dtype) // tp
    act = activation_bytes(d, ctx, concurrency,
                           quant if quant in ("bf16", "fp16") else "bf16")
    framework = int(FRAMEWORK_OVERHEAD_GB[runtime] * GB)
    total = weights + kv + act + framework
    device_b = int(device_vram_gb * GB)
    usable_b = int(device_b * gpu_memory_utilization)
    free_for_kv = usable_b - weights - act - framework
    kv_per_token = max(kv_bytes(d, 1, 1, kv_dtype) // tp, 1)
    # Base footprint (weights + act + framework) can exceed usable VRAM, in
    # which case free_for_kv is negative and there is no room for any KV.
    # Clamp to 0 here so callers (TP sweep, single-result print) see a
    # consistent "no headroom" signal instead of a negative-divided value.
    if free_for_kv <= 0:
        max_concurrency = 0
        max_context = 0
    else:
        max_concurrency = free_for_kv // max(kv_per_token * ctx, 1)
        max_context = free_for_kv // max(kv_per_token * max(concurrency, 1), 1)
    return {
        "dims": d,
        "params": params,
        "weights": weights,
        "kv": kv,
        "act": act,
        "framework": framework,
        "total": total,
        "device_vram": device_b,
        "usable_vram": usable_b,
        "fits": total <= usable_b,
        "headroom": usable_b - total,
        "max_concurrency": int(max_concurrency),
        "max_context": int(max_context),
        "mixed_breakdown": mixed_breakdown,
    }


def verdict_cell(result: dict, usable_vram_gb: float) -> str:
    if result["fits"]:
        headroom_gb = result["headroom"] / GB
        if headroom_gb < max(1.0, 0.10 * usable_vram_gb):
            return "tight"
        return "fits"
    parts = [
        ("weights", result["weights"]),
        ("KV", result["kv"]),
        ("activations", result["act"]),
        ("framework", result["framework"]),
    ]
    binding = max(parts, key=lambda x: x[1])[0]
    return f"OOM ({binding})"


def print_table(models: list[str], runtime: str, device_vram_gb: float,
                gpu_memory_utilization: float, revision: str) -> int:
    scenarios = [
        ("bf16 / 8K / c=1", "bf16", "bf16", 8192, 1, 1),
        ("bf16 / 4K / c=4", "bf16", "bf16", 4096, 4, 1),
        ("int4 / 32K / c=4", "int4", "fp8", 32768, 4, 1),
    ]
    usable_vram_gb = device_vram_gb * gpu_memory_utilization
    print(f"Runtime: {runtime}, device VRAM: {device_vram_gb:.2f} GB "
          f"(usable {usable_vram_gb:.2f} GB at "
          f"gpu_memory_utilization={gpu_memory_utilization:g})")
    print()
    header = ["Model"] + [s[0] for s in scenarios]
    rows = []
    for model in models:
        cfg = fetch_config(model, revision)
        row = [model.rsplit("/", 1)[-1]]
        for _, quant, kv_dtype, ctx, concurrency, tp in scenarios:
            result = estimate(cfg, quant, kv_dtype, ctx, concurrency,
                              tp, runtime, device_vram_gb,
                              gpu_memory_utilization)
            row.append(verdict_cell(result, usable_vram_gb))
        rows.append(row)

    widths = [max(len(str(x)) for x in col)
              for col in zip(header, *rows)]
    print(" | ".join(str(x).ljust(w) for x, w in zip(header, widths)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    return 0


def parse_tp_sweep(value: str | None) -> list[int]:
    # Sort ascending so the verdict line ("smallest TP that fits") cannot
    # contradict the table when the user passes values out of order
    # (e.g. --tp-sweep 4,2,1). Smallest fit is the cheapest deployment.
    if not value:
        return []
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            tp = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--tp-sweep must be comma-separated integers") from exc
        if tp < 1:
            raise argparse.ArgumentTypeError("--tp-sweep values must be >= 1")
        out.add(tp)
    if not out:
        raise argparse.ArgumentTypeError("--tp-sweep must include at least one TP value")
    return sorted(out)


def print_tp_sweep(cfg: dict, args: argparse.Namespace, tp_values: list[int]) -> None:
    rows = []
    first_fit = None
    for tp in tp_values:
        result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                          args.concurrency, tp, args.runtime,
                          args.device_vram_gb, args.gpu_memory_utilization)
        rows.append((
            tp,
            result["weights"],
            result["kv"],
            result["act"],
            result["framework"],
            result["total"],
            result["headroom"],
            result["fits"],
            result["max_concurrency"],
            result["max_context"],
        ))
        if result["fits"] and first_fit is None:
            first_fit = tp

    print()
    print("TP sweep")
    print(f"  {'TP':>2}  {'weights':>9} {'KV':>9} {'act':>9} "
          f"{'fw':>9} {'total':>9} {'headroom':>9} "
          f"{'fit':>3}  {'max_c':>5}  {'max_ctx':>7}")
    print("  " + "-" * 83)
    for tp, weights, kv, act, framework, total, headroom, fits, max_c, max_ctx in rows:
        verdict = "YES" if fits else "NO"
        print(f"  {tp:>2}  {fmt_gb(weights)} {fmt_gb(kv)} {fmt_gb(act)} "
              f"{fmt_gb(framework)} {fmt_gb(total)} {fmt_gb(headroom)} "
              f"{verdict:>3}  {max_c:>5}  {max_ctx:>7}")
    if first_fit is None:
        print("TP sweep verdict: no requested TP fits.")
    else:
        print(f"TP sweep verdict: smallest requested TP that fits = {first_fit}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="HF model id or path to config.json")
    p.add_argument("--table", action="store_true",
                   help="Print a quick verdict table for common public LLMs.")
    p.add_argument("--table-model", action="append", default=[],
                   help="Model id to include with --table. May be repeated.")
    p.add_argument("--quant", default=None, choices=list(BYTES_PER_PARAM) + [None],
                   help="Quantization format. If not specified, auto-detects from model config (quantization_config.quant_method) or defaults to bf16.")
    p.add_argument("--kv-dtype", default=None, choices=list(BYTES_PER_KV) + [None])
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree (divides weights+KV).")
    p.add_argument("--tp-sweep", default=None,
                   help="Comma-separated TP values to evaluate after the main verdict, e.g. 1,2,4,8.")
    p.add_argument("--runtime", default="vllm", choices=list(FRAMEWORK_OVERHEAD_GB))
    # No default: an assumed VRAM figure produces a confident, wrong
    # verdict. Guessing high is the dangerous direction -- it reports FITS
    # for a launch that OOMs at engine init. Make the caller supply a
    # measured number instead.
    p.add_argument("--device-vram-gb", type=float, default=None,
                   help="Required. Per-device VRAM in GB. Confirm the card "
                        "with `xpu-smi discovery -d <id>` rather than recalling "
                        "it from a spec sheet. Arc Pro B70 = 32, B65 = 32, "
                        "B60 = 24, B50 = 16, B580 = 12. For tensor "
                        "parallelism, pass the smallest card in the set.")
    p.add_argument("--gpu-memory-utilization", type=float, default=1.0,
                   help="Usable fraction of device VRAM for the runtime. "
                        "Use the vLLM --gpu-memory-utilization value for launch planning.")
    p.add_argument("--revision", default="main")
    args = p.parse_args(argv)

    if not (0 < args.gpu_memory_utilization <= 1.0):
        p.error("--gpu-memory-utilization must be > 0 and <= 1")
    if args.device_vram_gb is None:
        p.error(
            "--device-vram-gb is required; there is no safe default.\n"
            "  Identify the target card first -- do not use a remembered spec:\n"
            "      xpu-smi discovery -d <id> | grep -i 'Device Name\\|Memory "
            "Physical Size'\n"
            "  then pass that SKU's GB: B70 32, B65 32, B60 24, B50 16, "
            "B580 12.\n"
            "  For tensor parallelism, pass the SMALLEST card in the set.")
    # Bind before the try so the name is defined on every path. p.error()
    # raises SystemExit, but static analysis cannot see that and flags the
    # later `if tp_sweep:` reads as possibly-uninitialized.
    tp_sweep: list[int] = []
    try:
        tp_sweep = parse_tp_sweep(args.tp_sweep)
    except argparse.ArgumentTypeError as exc:
        p.error(str(exc))

    if args.table:
        models = args.table_model or TABLE_MODELS
        return print_table(models, args.runtime, args.device_vram_gb,
                           args.gpu_memory_utilization, args.revision)
    if not args.model:
        p.error("--model is required unless --table is set")

    cfg = fetch_config(args.model, args.revision)

    # Auto-detect pre-quantized models from config if --quant not specified
    auto_quant = False
    if args.quant is None:
        qcfg = cfg.get("quantization_config", {})
        quant_method = qcfg.get("quant_method", "").lower()
        if quant_method in BYTES_PER_PARAM:
            args.quant = quant_method
            auto_quant = True
        else:
            args.quant = "bf16"

    auto_kv = args.kv_dtype is None
    if auto_kv:
        args.kv_dtype = "fp8" if args.quant in ("fp8", "int4", "int3", "int2", "mxfp4") else "bf16"
    result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                      args.concurrency, args.tp, args.runtime,
                      args.device_vram_gb, args.gpu_memory_utilization)
    d = result["dims"]
    params = result["params"]
    bpp = BYTES_PER_PARAM[args.quant]
    weights = result["weights"]
    kv = result["kv"]
    act = result["act"]
    framework = result["framework"]
    total = result["total"]
    usable_b = result["usable_vram"]
    fits = result["fits"]
    headroom = result["headroom"]

    arch_label = "decoder-only LLM"
    if d.is_moe:
        arch_label = "MoE"
    if d.is_vlm:
        arch_label = "VLM (LLM backbone + vision tower)"

    print(f"Model:             {args.model}")
    print(f"Architecture:      {d.arch_family}  ({arch_label})")
    if d.is_moe:
        expert_str = f"{d.num_experts} routed"
        if d.num_shared_experts:
            expert_str += f" + {d.num_shared_experts} shared"

        # Calculate active experts: routed + shared
        # Shared experts are always-on in addition to top-k routed, not instead of them
        if d.num_experts_per_tok:
            # Known: report routed + shared
            active = d.num_experts_per_tok + d.num_shared_experts
            active_str = f"~{active} active per token"
        elif d.num_shared_experts:
            # Unknown routed top-k, but shared experts present
            # Report "? + N shared" to clarify shared are in addition to unknown routed
            active_str = f"? routed + {d.num_shared_experts} shared active per token"
        else:
            # No information about active experts
            active_str = "? active per token"

        print(f"Experts:           {expert_str}, {active_str}")
        if d.first_k_dense_replace > 0:
            print(f"                   (first {d.first_k_dense_replace} layer(s) use dense FFN)")
    if d.is_vlm:
        print(f"Vision tower:      {d.vision_params/1e9:.2f} B params "
              f"(included in weights below)")
    print(f"Total parameters:  {params/1e9:.2f} B")
    print(f"Quantization:      {args.quant}  ({bpp:.2f} bytes/param)", end="")
    if auto_quant:
        print(f"  (auto-detected from config)")
    else:
        print()
    if auto_kv and args.kv_dtype != "bf16":
        print(f"KV dtype:          {args.kv_dtype}  (auto-paired with --quant {args.quant}; "
              f"override with --kv-dtype)")
    if args.tp > 1:
        print(f"Tensor parallel:   {args.tp}  (weights + KV split across devices)")
    if args.gpu_memory_utilization < 1.0:
        print(f"Memory limit:      {args.gpu_memory_utilization:.2f} of physical VRAM "
              f"(runtime allocation target)")

    # Show mixed-precision breakdown if available
    mixed_breakdown = result.get("mixed_breakdown")
    if mixed_breakdown:
        print()
        print("Mixed-precision weight breakdown:")
        if mixed_breakdown["embed_params"] > 0:
            print(f"  Embeddings      {fmt_gb(mixed_breakdown['embed_bytes'])}   "
                  f"({mixed_breakdown['embed_params']/1e9:.2f}B params @ {mixed_breakdown['embed_bpp']:.2f} B/p)")
        if mixed_breakdown["attn_params"] > 0:
            print(f"  Attention       {fmt_gb(mixed_breakdown['attn_bytes'])}   "
                  f"({mixed_breakdown['attn_params']/1e9:.2f}B params @ {mixed_breakdown['attn_bpp']:.2f} B/p)")
        if mixed_breakdown["router_params"] > 0:
            print(f"  Routers         {fmt_gb(mixed_breakdown['router_bytes'])}   "
                  f"({mixed_breakdown['router_params']/1e9:.2f}B params @ {mixed_breakdown['router_bpp']:.2f} B/p)")
        if mixed_breakdown["ffn_params"] > 0:
            print(f"  FFN/Experts     {fmt_gb(mixed_breakdown['ffn_bytes'])}   "
                  f"({mixed_breakdown['ffn_params']/1e9:.2f}B params @ {mixed_breakdown['ffn_bpp']:.2f} B/p)")
        print(f"  -----              -----")
        print(f"  Weights total   {fmt_gb(weights)}")

    print()
    print("VRAM breakdown")
    print(f"  Weights         {fmt_gb(weights)}")
    print(f"  KV cache        {fmt_gb(kv)}   "
          f"({args.ctx} tok x concurrency {args.concurrency}, kv_dtype {args.kv_dtype})")
    print(f"  Activations     {fmt_gb(act)}   (estimate)")
    print(f"  Framework       {fmt_gb(framework)}   ({args.runtime})")
    print(f"  -----              -----")
    print(f"  Total           {fmt_gb(total)}")
    print()
    per_device = f"  (per device, x{args.tp} TP)" if args.tp > 1 else ""
    print(f"Device VRAM:       {args.device_vram_gb:6.2f} GB{per_device}")
    if args.gpu_memory_utilization < 1.0:
        print(f"Usable VRAM:       {usable_b / GB:6.2f} GB "
              f"(gpu_memory_utilization={args.gpu_memory_utilization:.2f})")
    else:
        # Every line qualifying the VRAM figure is gated on gmu < 1.0, so
        # the default run would otherwise show a nameplate number with
        # nothing marking it optimistic. Both gaps beneath the nameplate --
        # vendor rounding and the driver's allocatable ceiling -- run the
        # same direction, so a tight FITS here is not a launchable result.
        print("Usable VRAM:       all of it  (physical-fit only; no "
              "--gpu-memory-utilization given)")
        print("Note: that is nameplate VRAM, and both gaps beneath it are "
              "optimistic. Vendors round up (a \"24 GB\" Arc Pro B60 exposes "
              "23.91 GiB, measured) and the driver's allocatable ceiling sits "
              "~5% below physical. Re-run with the --gpu-memory-utilization "
              "the runtime will actually use before trusting a tight verdict.")

    if fits:
        pct = 100 * headroom / usable_b
        print(f"Verdict:           FITS  (headroom {fmt_gb(headroom)}, {pct:.1f}%)")
        print(f"Capacity:          up to {result['max_concurrency']} concurrent "
              f"request(s) at ctx {args.ctx}, or ctx {result['max_context']} "
              f"at concurrency {args.concurrency}  (memory-only ceiling)")
        if tp_sweep:
            print_tp_sweep(cfg, args, tp_sweep)
        if d.is_vlm:
            print()
            print("Note: VLM runtime memory grows with image-token count, which "
                  "depends on the resolution and number of images per request. "
                  "The number above covers weights + text-side KV; add ~1-3 "
                  "GiB headroom per concurrent image-bearing request for "
                  "vision-encoder activations.")
        return 0

    deficit = -headroom
    print(f"Verdict:           DOES NOT FIT  (deficit {fmt_gb(deficit)})")

    parts = sorted(
        [("weights", weights), ("KV cache", kv), ("activations", act), ("framework", framework)],
        key=lambda kv: -kv[1],
    )
    binding, _ = parts[0]
    print(f"  Binding constraint: {binding} ({fmt_gb(parts[0][1])}) dominates")

    suggestions: list[str] = []
    if binding == "KV cache":
        if args.ctx > 1024:
            new_ctx = max(1024, args.ctx // 2)
            new_kv = kv_bytes(d, new_ctx, args.concurrency, args.kv_dtype) // args.tp
            suggestions.append(
                f"drop ctx {args.ctx} -> {new_ctx} (saves {fmt_gb(kv - new_kv)})"
            )
        if args.kv_dtype in ("bf16", "fp16"):
            new_kv = kv_bytes(d, args.ctx, args.concurrency, "fp8") // args.tp
            suggestions.append(
                f"kv-dtype {args.kv_dtype} -> fp8 (saves {fmt_gb(kv - new_kv)})"
            )
        if args.concurrency > 1:
            new_kv = kv_bytes(d, args.ctx, 1, args.kv_dtype) // args.tp
            suggestions.append(
                f"concurrency {args.concurrency} -> 1 (saves {fmt_gb(kv - new_kv)})"
            )
    if binding == "weights":
        if args.quant in ("bf16", "fp16"):
            for q in ("fp8", "int4", "int3"):
                new_w = int(params * BYTES_PER_PARAM[q] / args.tp)
                suggestions.append(
                    f"quant {args.quant} -> {q} (saves {fmt_gb(weights - new_w)})"
                )
        if args.tp == 1:
            suggestions.append(
                f"--tp 2 splits weights across two XPUs (saves {fmt_gb(weights // 2)})"
            )
    if not suggestions:
        suggestions.append(
            "model is too big for this device at any reasonable setting; "
            "try a smaller model or more XPUs (--tp)"
        )
    # Always show the TP option for weights-bound cases, even past the cap.
    cap = 4 if (binding == "weights" and args.tp == 1) else 3
    print(f"  Try first: {' or '.join(suggestions[:cap])}")
    if tp_sweep:
        print_tp_sweep(cfg, args, tp_sweep)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
