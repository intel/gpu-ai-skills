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
    # MXFP4: 4 bits plus one ue8m0 (e8m0fnu) scale byte per 32 weights.
    # DeepSeek-V4-Flash measures 0.531 B/param effective, so 0.55 is
    # deliberately ~4% conservative.
    "mxfp4": 0.55,
    # `fp4` is MXFP4 under another name -- DeepSeek-V4 spells its MXFP4
    # experts that way, and vLLM's DeepSeek-V4 quant config reads
    # `expert_dtype="fp4"` as "MXFP4 experts with ue8m0 FP8 linear scales".
    # Kept so an explicit `--quant fp4` still parses; config-declared values
    # normalize to `mxfp4` through QUANT_METHOD_ALIASES so they inherit the
    # supported-kernel status instead of being flagged as an unknown path.
    "fp4":   0.55,
    # NVFP4 scales in blocks of 16, not 32: one fp8 (e4m3) scale per 16 fp4
    # values is 0.5 + 1/16 = 0.5625 B/param packed, before the per-tensor
    # fp32 scale and any block padding. It cannot share the `fp4` row -- 0.55
    # is below its packed floor and would understate every NVFP4 checkpoint.
    "nvfp4": 0.58,
    "fp32":  4.00,
}

BYTES_PER_KV = {
    "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
}

# quantization_config.quant_method spellings that mean a dtype we price
# above. Keys are lowercase quant_method values as they appear in configs.
# Only methods whose name fixes the weight width belong here.
QUANT_METHOD_ALIASES = {
    "fp4":     "mxfp4",       # DeepSeek-V4's spelling of MXFP4
    "modelopt_fp4": "nvfp4",   # TensorRT-ModelOpt's spelling of NVFP4
    # torch dtype spellings, which is how an `expert_dtype` field names itself
    "float32":  "fp32",
    "float16":  "fp16",
    "bfloat16": "bf16",
}

# quant_method spellings that name an algorithm rather than a width: AWQ,
# GPTQ and AutoRound all ship 8-, 4-, 3- and 2-bit checkpoints, so the
# width comes from quantization_config.bits (older AWQ configs: w_bit).
# Mapping the method name straight to int4 understates an Int8 GPTQ
# checkpoint by nearly 2x, which is enough to report FITS for a launch
# that OOMs.
WIDTH_FROM_BITS_METHODS = frozenset({
    "awq", "gptq", "autoround", "auto-round", "auto_round", "intel/auto-round",
})

BITS_TO_QUANT = {8: "int8", 4: "int4", 3: "int3", 2: "int2"}

# Weight dtypes narrower than 16-bit. Derived from the table so a new dtype
# cannot be forgotten here the way `fp4` was.
SUB16_QUANTS = frozenset(
    q for q, bpp in BYTES_PER_PARAM.items() if bpp < 2.0
)

# Weight dtypes that auto-pair KV with fp8, matching the pairings
# vllm-xpu-run/references/quantization.md documents on XPU: fp8 weights and
# the AWQ/GPTQ path. MXFP4 is documented with `auto` KV there, so it is absent
# on purpose -- defaulting it to fp8 would halve the KV figure on an assumption
# the runtime guidance does not make, and a launch without
# --kv-cache-dtype fp8 would then need 2x the estimate.
#
# KV dtype is a launch flag, not a property of the checkpoint: nothing about
# 4-bit expert weights requires an 8-bit cache. `--kv-dtype fp8` remains
# available for any of them, and the report points that out.
FP8_KV_PAIRED_QUANTS = frozenset({"fp8", "int8", "int4"})

# The dtype alone cannot decide it, because a dtype does not identify the row.
# quant_from_config() reduces an AutoRound checkpoint to int8/int4 from `bits`,
# and AutoRound has its own row in that table with `auto` KV -- vLLM detects
# `quant_method=auto-round` and routes it through the gptq/awq loader without
# the fp8 cache those two rows pair. Pairing on the reduced dtype would halve
# KV for a launch that allocates a bf16 cache, which is a false FITS. So when
# the config declares a method that still describes the requested dtype, that
# method has to name a paired row too. A config with no method -- or one an
# explicit --quant has overridden -- leaves the dtype to decide: a 4-bit
# re-quantization hypothetical on XPU means AWQ or GPTQ.
FP8_KV_PAIRED_METHODS = frozenset({"fp8", "awq", "gptq"})

# ...and only on the runtime whose guidance makes that pairing. vLLM's table
# pairs fp8/int4 with `--kv-cache-dtype fp8`; SGLang keeps BF16 KV unless
# `--kv-cache-dtype fp8_e4m3` is asked for explicitly (sglang-xpu-run's
# quantization table), and the torch path has no KV-dtype flag at all --
# transformers allocates the cache in the model dtype. Auto-pairing outside
# vLLM would halve the estimated cache against the runtime's own default,
# which is the false-FITS direction.
FP8_KV_AUTOPAIR_RUNTIMES = frozenset({"vllm"})

# The KV-dtype launch flag each runtime exposes, or None where there is none.
# Used to name the right flag in the report instead of quoting vLLM's at
# every runtime.
KV_DTYPE_FLAG = {
    "vllm":   "--kv-cache-dtype fp8",
    "sglang": "--kv-cache-dtype fp8_e4m3",
    "torch":  None,
}

# Weight dtypes with no documented XPU kernel path. vllm-xpu-run's
# quantization table covers bf16/fp16, fp8, awq/gptq, mxfp4 and auto-round;
# NVFP4 (per-16 fp8 block scales) is not on it and has no measured XPU run
# here. Its byte math is still worth reporting -- the checkpoint on disk
# really is that size -- but a load either refuses or upcasts, and an upcast
# invalidates the verdict, so the estimate has to say so.
#
# MXFP4 is deliberately absent. Traced in vllm/vllm-openai-xpu:latest (vLLM
# 0.29.0): `quant_method: mxfp4` and DeepSeek-V4's `expert_dtype: fp4` take
# different configs and MoE methods (GptOssMxfp4MoEMethod vs Mxfp4MoEMethod)
# but both short-circuit on is_xpu() to Mxfp4MoeBackend.XPU and the same
# XPUExpertsMxFp4 kernel, which "consumes the checkpoint layout directly"
# rather than transforming it. Measured on the gpt-oss path: 12.87 GiB
# allocated against a 13.14 GiB estimate, so the weights stayed packed. The
# DeepSeek expert path is verified by dispatch inspection only -- 155 GiB does
# not fit the hardware here.
NO_XPU_KERNEL_PATH = frozenset({"nvfp4"})

# Runtimes that shard a layer across ranks (by attention head and FFN width)
# and reject a TP whose dimensions do not divide. `torch` is absent on
# purpose: accelerate's device_map places whole modules per device instead.
TENSOR_PARALLEL_RUNTIMES = frozenset({"vllm", "sglang"})

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
    """Approximate parameter count of a ViT-style vision tower.

    The tower's width is `embed_dim` where a config declares one. Qwen2-VL
    names the ViT width that way and reuses `vision_config.hidden_size` for the
    *output* of its patch merger (1280 vs 3584), so reading `hidden_size` there
    prices 32 blocks at 2.8x their width: 4.94 B params against a ~0.68 B
    tower. CLIP-style configs (llava, Mistral-Small) carry no `embed_dim` and
    `hidden_size` is the width. `mlp_ratio` is Qwen2-VL's spelling of the MLP
    width; `intermediate_size` wins where both appear.
    """
    h = vc.get("embed_dim") or vc.get("hidden_size") or 0
    layers = vc.get("depth") or vc.get("num_hidden_layers") or 0
    mlp_ratio = vc.get("mlp_ratio")
    intermediate = (vc.get("intermediate_size")
                    or (h * mlp_ratio if mlp_ratio else 4 * h))
    if not (h and layers):
        return 0
    # ViT layer: attn (4 h^2) + MLP + 2 norms (2 h). A gated MLP holds three
    # matrices (gate/up/down), not two: Qwen2.5-VL's tower is SwiGLU and says
    # so with `hidden_act: silu`, where CLIP-style towers are gelu/quick_gelu
    # with two. Counting two on a gated tower is 22% light on the blocks, and
    # light on weights is the direction that reports a false FITS.
    act = str(vc.get("hidden_act", "")).lower()
    mlp_mats = 3 if any(g in act for g in ("silu", "swish", "glu")) else 2
    per_layer = 4 * h * h + mlp_mats * h * intermediate + 2 * h
    # Patch embedding: roughly hidden * patch_area * channels. Bound modestly.
    patch = vc.get("patch_size", 14)
    in_channels = vc.get("num_channels", 3)
    embed = patch * patch * in_channels * h
    # int(): mlp_ratio is a float in some configs, and a param count is not.
    return int(embed + layers * per_layer)


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
            return (emb + head + (dense_layers * dense_per_layer)
                    + (moe_layers * moe_per_layer) + router_params(d))
        else:
            # Pure MoE: all layers use expert FFN
            per_layer = attn_block + moe_ff_block + norms
    else:
        # Dense model
        ff_block = 3 * h * d.intermediate
        per_layer = attn_block + ff_block + norms

    emb = d.vocab * h
    head = 0 if d.tied else d.vocab * h
    return emb + head + d.num_layers * per_layer + router_params(d)


def kv_bytes(d: ModelDims, ctx: int, concurrency: int, kv_dtype: str) -> int:
    per_token = 2 * d.num_layers * d.num_kv_heads * d.head_dim
    return int(per_token * ctx * concurrency * BYTES_PER_KV[kv_dtype])


def ffn_shard_widths(d: ModelDims) -> tuple[int, ...]:
    """FFN widths a runtime partitions, and only the ones a layer really uses.

    `d.intermediate` is the width in force (moe_intermediate_size on an MoE,
    intermediate_size on a dense model). `d.dense_intermediate` only describes
    real layers on a *hybrid* MoE -- parse_dims fills it from intermediate_size
    for every MoE config, so checking it unconditionally would reject a TP over
    a width no layer of a pure MoE has.
    """
    widths = [d.intermediate]
    if d.is_moe and d.first_k_dense_replace > 0 and d.dense_intermediate:
        widths.append(d.dense_intermediate)
    return tuple(w for w in widths if w)


def kv_shards(d: ModelDims, tp: int) -> int:
    """How many ways the KV cache actually splits at this TP.

    Attention shards by KV head, so KV stops shrinking once TP passes the
    KV-head count: the runtime replicates the group instead, leaving every
    rank one head. A 4-KV-head model at TP 8 therefore stores total_KV/4 per
    device, not total_KV/8 -- dividing by TP past that point understates KV
    and can report FITS for a launch that OOMs.
    """
    return max(min(tp, d.num_kv_heads or tp), 1)


def layer_shards(d: ModelDims, devices: int) -> int:
    """Effective divisor when whole transformer layers are placed per device.

    accelerate's device_map assigns modules, not slices of them, so the
    busiest device holds ceil(num_layers / devices) layers and everything on
    it -- KV and weights alike. A 4-layer model spread over 8 devices still
    keeps a whole layer on each nonempty one, so its per-device footprint is
    total/4, not total/8, and the extra devices sit idle; 33 layers over 8
    devices leaves one device holding 5.

    The divisor is floored, so an uneven split errs heavy rather than light,
    and it stops growing once devices outnumber layers. Still optimistic in
    one respect: embeddings and the LM head land on one of these devices on
    top of its layers, and accelerate balances by module size rather than
    module count.
    """
    if d.num_layers < 1:
        return max(devices, 1)
    per_device = -(-d.num_layers // max(devices, 1))   # ceil
    return max(d.num_layers // per_device, 1)


def unsplittable_bytes(d: ModelDims, quant: str,
                       mixed_breakdown: dict | None = None) -> int:
    """Bytes of the largest tensor no placement strategy can divide.

    The embedding matrix -- and the LM head, same shape, when untied -- is a
    single module weight: device_map assigns it to one device whole. Dividing
    the whole parameter total by a layer-derived divisor can land below it,
    which reports a per-device figure smaller than one mandatory tensor: a
    256k-vocab model with 4 layers over 4 devices is the easy example.

    A floor, not a placement model. The device holding the embedding also
    holds layers, and accelerate balances by module size, so the real peak is
    higher than either this or the even layer share.
    """
    bpp = (mixed_breakdown["embed_bpp"]
           if mixed_breakdown and mixed_breakdown.get("embed_params")
           else BYTES_PER_PARAM[quant])
    return int(d.vocab * d.hidden * bpp)


def kv_divisor(d: ModelDims, tp: int, runtime: str) -> int:
    """Per-device KV divisor for a runtime.

    Head-sharded runtimes cap at the KV-head count (kv_shards); `torch` means
    accelerate's device_map, which places whole layers, so it caps at layer
    granularity (layer_shards). One definition so the verdict and the
    mitigations it prints cannot quote different divisors.
    """
    return (kv_shards(d, tp) if runtime in TENSOR_PARALLEL_RUNTIMES
            else layer_shards(d, tp))


def weight_divisor(d: ModelDims, tp: int, runtime: str) -> int:
    """Per-device weight divisor for a runtime.

    Head-sharded runtimes split every matrix, so weights divide by TP exactly.
    Module placement cannot: the same layer-granularity ceiling that bounds KV
    bounds weights, or a 4-layer model on 8 devices would be reported at half
    the bytes its busiest device really holds.
    """
    return (max(tp, 1) if runtime in TENSOR_PARALLEL_RUNTIMES
            else layer_shards(d, tp))


def activation_bytes(d: ModelDims, ctx: int, concurrency: int, dtype: str) -> int:
    # ~2 hidden buffers' worth + 512 MiB scratch
    bytes_per = BYTES_PER_PARAM[dtype]
    return int(2 * concurrency * ctx * d.hidden * bytes_per) + 512 * MB


def fmt_gb(b: int) -> str:
    return f"{b / GB:6.2f} GB"


def declared_quant_method(cfg: dict) -> str | None:
    """The `quantization_config.quant_method` a config declares, priceable or
    not.

    quant_from_config() returns None both for "no quantization declared" and
    for "a method this script cannot price", and those are different facts. An
    absent method means the checkpoint ships at 16-bit, so an explicit
    `--quant bf16` restates it. A declared-but-unpriceable method means the
    shipped dtype is *unknown*: nothing can restate it, and an as-shipped
    figure would be invented.
    """
    method = str((cfg.get("quantization_config") or {}).get("quant_method", ""))
    return method.lower() or None


def quant_from_config(cfg: dict) -> tuple[str | None, bool]:
    """Weight dtype the config declares, plus whether its width was assumed.

    Returns (dtype, width_assumed). dtype is None when the config declares no
    quantization, or a method this script cannot price -- the caller then
    defaults to bf16.

    For AWQ / GPTQ / AutoRound the method name says nothing about the width,
    so `bits` decides. When a width-parameterized method declares no usable
    width, the fallback is the widest int it ships (`int8`) and the flag comes
    back True so the caller can say the width was assumed: guessing narrow is
    the dangerous direction, because it understates weights and turns an OOM
    into a FITS verdict.
    """
    qcfg = cfg.get("quantization_config") or {}
    method = str(qcfg.get("quant_method", "")).lower()
    if not method:
        return None, False

    if method in WIDTH_FROM_BITS_METHODS:
        bits = qcfg.get("bits", qcfg.get("w_bit"))
        try:
            dtype = BITS_TO_QUANT.get(int(bits))
        except (TypeError, ValueError):
            dtype = None
        return (dtype, False) if dtype else ("int8", True)

    dtype = QUANT_METHOD_ALIASES.get(method, method)
    return (dtype, False) if dtype in BYTES_PER_PARAM else (None, False)


def raw_expert_dtype(cfg: dict) -> str | None:
    """The `expert_dtype` string a config declares, priceable or not.

    expert_dtype_of() returns None both for "no expert dtype" and for one this
    script cannot price, and those two must not be treated alike: the experts
    are ~96% of an MoE, so silently pricing them at the base dtype turns an
    unknown into a guess. main() compares the two to tell the cases apart.
    """
    text_cfg = cfg.get("text_config") or cfg.get("llm_config") or {}
    raw = cfg.get("expert_dtype") or text_cfg.get("expert_dtype")
    return str(raw).lower() if raw else None


def expert_dtype_of(cfg: dict) -> str | None:
    """Weight dtype declared for MoE expert tensors, independent of the
    repo-wide quant_method. Returns None if absent or unpriced.

    DeepSeek-V4 ships `quantization_config.quant_method: fp8` alongside a
    top-level `expert_dtype: fp4`, which makes the checkpoint a three-way
    split: expert FFNs at fp4, embeddings and any untied LM head at bf16 (no
    quantizer here touches those), and the remaining non-expert weights at
    quant_method. Reading quant_method alone prices ~96% of the model at
    double its real size -- for DeepSeek-V4-Flash that is 290.89 GB of
    "weights" against 159.6 GB of actual safetensors.
    """
    dtype = raw_expert_dtype(cfg)
    if not dtype:
        return None
    normalized = QUANT_METHOD_ALIASES.get(dtype, dtype)
    return normalized if normalized in BYTES_PER_PARAM else None


def tp_is_shardable(d: ModelDims, tp: int) -> bool:
    """Whether this model's shard dimensions divide by this TP.

    Three dimensions have to agree, because a runtime partitions all of them:

    - query heads: TP must divide num_attention_heads.
    - KV heads: either they divide evenly across ranks, or the whole group is
      replicated when TP exceeds their count (vLLM's rule). 12 query heads
      with 8 KV heads works at TP 4 but not at TP 6 -- 8 is neither a multiple
      nor a divisor of 6 -- even though 6 divides the query heads.
    - FFN width: column-parallel MLP splits the intermediate dimension, so TP
      must divide every width the model actually uses (see
      ffn_shard_widths()). 12 query heads with intermediate_size 28672 passes
      the head checks at TP 6 and is still rejected, because 28672 does not
      divide by 6.

    This is a necessary condition, not a promise: a runtime can impose further
    constraints (quantization group sizes, expert counts, its own padding
    rules). Callers should say "shard dimensions divide" rather than "will
    launch".
    """
    if tp < 1:
        return False
    if d.num_attn_heads % tp:
        return False
    kv_heads = d.num_kv_heads or d.num_attn_heads
    if not (kv_heads % tp == 0 or tp % kv_heads == 0):
        return False
    return all(w % tp == 0 for w in ffn_shard_widths(d))


def next_shardable_tp(d: ModelDims, tp: int, limit: int = 16) -> int | None:
    """Smallest TP above `tp` that this model's heads can actually shard into.

    A TP the heads cannot support is rejected at engine init, so blindly
    doubling would name a launch that cannot start: a 12-head model is valid
    at TP 4 but not at TP 8.

    Returns None when nothing up to `limit` qualifies, so the caller can drop
    the suggestion instead of printing an unusable one. Device availability is
    not visible here; the caller says how many the figure assumes.
    """
    return next((t for t in range(tp + 1, limit + 1)
                 if tp_is_shardable(d, t)), None)


def next_placement_tp(d: ModelDims, tp: int, limit: int = 16) -> int | None:
    """Smallest device count above `tp` that leaves the busiest device fewer
    layers.

    Module placement improves in steps, not smoothly: a 4-layer model on 3
    devices already has 2 layers on its busiest one, and 4 devices is what
    drops that to 1. So does 6, but two of those six would sit idle -- naming
    6 overstates the hardware the improvement needs. The steps for 40 layers
    are 1, 2, 2, 4, 5, 5, 6, 8: doubling from 4 skips 5 entirely.

    Returns None when nothing up to `limit` improves, which is the case once
    devices outnumber layers.
    """
    current = layer_shards(d, tp)
    return next((t for t in range(tp + 1, limit + 1)
                 if layer_shards(d, t) > current), None)


def is_router_module(name: str) -> bool:
    """Whether a modules_to_not_convert entry names the MoE gate/router.

    Two spellings are in the wild: `router` (gpt-oss's `mlp.router`) and
    `gate` (Mixtral's `block_sparse_moe.gate`, DeepSeek and Qwen3-MoE's
    `mlp.gate`). `gate` matches only as a whole path segment, because
    `gate_proj` and `gate_up_proj` are the dense SwiGLU projections -- FFN
    weights, not routing -- and treating those as a router would move most of
    a layer's params into the wrong row.
    """
    lowered = name.lower()
    return "router" in lowered or "gate" in lowered.split(".")


def is_vision_module(name: str) -> bool:
    """Whether a modules_to_not_convert entry names the vision tower.

    `visual` is what Qwen2-VL and Qwen2.5-VL AWQ/GPTQ builds list; llava and
    Mistral-Small style ignore lists spell it `vision_tower` or
    `vision_model`, and compressed-tensors writes regex entries like
    `re:visual.*`. Substring matching covers all of them.

    The tower needs its own answer because it is not covered by the
    embedding rule: a ViT block is Q/K/V/O and MLP Linear layers, which a
    quantizer does convert. Only an exclusion list keeps it 16-bit.
    """
    lowered = name.lower()
    return "visual" in lowered or "vision" in lowered


def moe_layer_count(d: ModelDims) -> int:
    """Layers that carry expert FFNs (all of them unless a hybrid MoE)."""
    if not d.is_moe:
        return 0
    if d.first_k_dense_replace > 0 and d.dense_intermediate > 0:
        return d.num_layers - d.first_k_dense_replace
    return d.num_layers


def router_params(d: ModelDims) -> int:
    """Gate/router projection params: hidden * num_experts per MoE layer.

    One linear layer per MoE layer, scoring every routed expert. Tiny next to
    the experts themselves (0.04% of DeepSeek-V4-Flash), but 16x a
    `2 * hidden` placeholder on a 32-expert model -- and that matters when
    `modules_to_not_convert` holds the routers at bf16 while everything else
    is 4-bit, because the bf16 uplift is priced on this count.

    Shared experts are always active and have no gate, so they are not scored.
    """
    if not d.is_moe or not d.num_experts:
        return 0
    return moe_layer_count(d) * d.hidden * d.num_experts


def expert_params(d: ModelDims) -> int:
    """Routed + shared expert FFN params -- the tensors `expert_dtype` covers.

    Mirrors the MoE branch of count_params(), so the remainder
    (params - expert_params) is exactly the non-expert weights.
    """
    if not d.is_moe:
        return 0
    per_layer = (d.num_experts + d.num_shared_experts) * 3 * d.hidden * d.intermediate
    return moe_layer_count(d) * per_layer


def calculate_mixed_precision_weights(cfg: dict, d: ModelDims, params: int,
                                       quant: str, tp: int,
                                       expert_dtype: str | None = None
                                       ) -> tuple[int, dict | None]:
    """Calculate weight bytes accounting for mixed-precision quantization.

    Two independent mechanisms, and a model may use both:

    - `quantization_config.modules_to_not_convert` (e.g. openai/gpt-oss-20b)
      keeps named components at full precision while quantizing the rest.
    - a separate `expert_dtype` for the MoE FFNs (e.g. DeepSeek-V4), passed
      in by the caller so an explicit --quant can suppress it.

    Returns:
        (weights_bytes, breakdown_dict or None)
    """
    # `or {}`, not a .get default: a config can carry an explicit
    # "quantization_config": null, and a None here crashes the whole verdict.
    qcfg = cfg.get("quantization_config") or {}
    modules_to_not_convert = qcfg.get("modules_to_not_convert") or []

    e_params = expert_params(d) if expert_dtype else 0
    e_bpp = BYTES_PER_PARAM[expert_dtype] if e_params else BYTES_PER_PARAM[quant]

    def uniform_or_expert_split() -> tuple[int, dict | None]:
        """No usable modules_to_not_convert: a flat dtype for everything, or
        embeddings and/or experts split out at their own dtype.

        Quantizers convert Linear layers, not `nn.Embedding`: GPTQ, AWQ,
        AutoRound and fp8 all leave the embedding matrix -- and an untied LM
        head -- at 16-bit whether or not the config lists them in
        `modules_to_not_convert`. Measured on served
        Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 and Qwen/Qwen2.5-7B-Instruct-AWQ
        (group_size 128, untied, 152064 vocab): vLLM-XPU allocated 5.18 and
        5.20 GiB of weights, against 3.90 GiB if those 1.09 B params are priced
        at int4 and 5.37 GiB if they are held at bf16.
        """
        embeds_stay_wide = quant in SUB16_QUANTS
        if not e_params and not embeds_stay_wide:
            return int(params * BYTES_PER_PARAM[quant] / tp), None
        # Experts at their own dtype, everything else at `quant`, embeddings at
        # whichever of the two the quantizer would really have left them in.
        embed_p = d.vocab * d.hidden * (1 if d.tied else 2)
        embed_bpp = (BYTES_PER_PARAM["bf16"] if embeds_stay_wide
                     else BYTES_PER_PARAM[quant])
        rest_p = max(params - e_params - embed_p, 0)
        embed_bytes = embed_p * embed_bpp
        rest_bytes = rest_p * BYTES_PER_PARAM[quant]
        e_bytes = e_params * e_bpp
        breakdown = {
            "embed_params": embed_p,
            "embed_bytes": int(embed_bytes / tp),
            "embed_bpp": embed_bpp,
            "attn_params": rest_p,
            "attn_bytes": int(rest_bytes / tp),
            "attn_bpp": BYTES_PER_PARAM[quant],
            "attn_label": "Non-expert" if e_params else "Quantized weights",
            "router_params": 0,
            "router_bytes": 0,
            "router_bpp": BYTES_PER_PARAM[quant],
            "ffn_params": e_params,
            "ffn_bytes": int(e_bytes / tp),
            "ffn_bpp": e_bpp,
            # No exclusion list, so a vision tower is inside rest_p at `quant`
            # -- a quantizer converts its Linear layers like any others. Zero
            # here keeps the row out of the printed table.
            "vision_params": 0,
            "vision_bytes": 0,
            "vision_bpp": BYTES_PER_PARAM[quant],
        }
        return int((embed_bytes + rest_bytes + e_bytes) / tp), breakdown

    if not modules_to_not_convert:
        return uniform_or_expert_split()

    # Parse which modules to keep at full precision
    keep_embeddings = any("embed" in m or "lm_head" in m for m in modules_to_not_convert)
    keep_attn = any("attn" in m for m in modules_to_not_convert)
    keep_router = any(is_router_module(m) for m in modules_to_not_convert)
    keep_vision = any(is_vision_module(m) for m in modules_to_not_convert)

    # If not selectively quantizing recognizable components, fall back to
    # uniform (still honoring a separate expert dtype if one was passed).
    # A router-only list counts: routers are one of the components recognized
    # above, so dropping through here would price them at the quantized dtype
    # the config just excluded them from. It only counts for an MoE though --
    # a dense model has no router params to hold back, so there is nothing for
    # a breakdown to show. A vision-only list counts the same way, and it is
    # the common VLM case: Qwen2-VL and Qwen2.5-VL AWQ builds list exactly
    # ["visual"], whose tower is ~0.68 B params the uniform path would price
    # at 4-bit when the checkpoint ships it at bf16.
    if not (keep_embeddings or keep_attn or (keep_router and router_params(d))
            or (keep_vision and d.vision_params)):
        return uniform_or_expert_split()

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

    # Router params, counted from the architecture rather than approximated:
    # a `2 * hidden` placeholder is 16x light on a 32-expert model, which
    # underprices the bf16 uplift this branch exists to charge for.
    router_p = router_params(d) if keep_router else 0

    # FFN/Expert params = remaining
    ffn_params = params - embed_params - attn_params - router_p - d.vision_params

    # Calculate bytes (bf16 = 2.0 for non-quantized, quant for quantized)
    bf16_bpp = 2.0
    quant_bpp = BYTES_PER_PARAM[quant]

    # Embeddings are 16-bit whenever the weight dtype is narrower than 16-bit,
    # listed in modules_to_not_convert or not: a quantizer converts Linear
    # layers, not nn.Embedding. Without this, an attention-only or router-only
    # exclusion list would charge a 152064x3584 matrix at 0.55 B/param and
    # understate a wide-vocab checkpoint by over a gigabyte.
    embed_bpp = (bf16_bpp if keep_embeddings or quant in SUB16_QUANTS
                 else quant_bpp)

    embed_bytes = embed_params * embed_bpp
    attn_bytes = attn_params * (bf16_bpp if keep_attn else quant_bpp)
    router_bytes = router_p * (bf16_bpp if keep_router else quant_bpp)
    # Experts carry their own dtype when the config declares one; any dense
    # FFN left over (hybrid MoE) stays at `quant`. Cap the expert share at
    # what the FFN row actually holds: ffn_params is a subtraction that has
    # already given router params their own row, so pricing the full
    # expert_params() on top of that row would bill those params twice and
    # leave ffn_bpp inconsistent with ffn_params.
    expert_ffn_params = min(e_params, ffn_params) if ffn_params > 0 else 0
    dense_ffn_params = max(ffn_params - expert_ffn_params, 0)
    ffn_bytes = expert_ffn_params * e_bpp + dense_ffn_params * quant_bpp
    # The vision tower decides on its own list entry, not on the embedding
    # rule. That rule holds `nn.Embedding` and an untied head at 16-bit because
    # quantizers convert Linear layers -- and a ViT block is all Linear layers,
    # so it follows `quant` unless the config excludes it. Reusing embed_bpp
    # would hold a whole tower at bf16 for every sub-16-bit dtype and turn a
    # launch that fits into DOES NOT FIT.
    vision_bpp = bf16_bpp if keep_vision else quant_bpp
    vision_bytes = d.vision_params * vision_bpp

    total_bytes = int((embed_bytes + attn_bytes + router_bytes + ffn_bytes + vision_bytes) / tp)

    breakdown = {
        "embed_params": embed_params,
        "embed_bytes": int(embed_bytes / tp),
        "embed_bpp": embed_bpp,
        "attn_params": attn_params,
        "attn_bytes": int(attn_bytes / tp),
        "attn_bpp": bf16_bpp if keep_attn else quant_bpp,
        "router_params": router_p,
        "router_bytes": int(router_bytes / tp),
        "router_bpp": bf16_bpp if keep_router else quant_bpp,
        "ffn_params": ffn_params,
        "ffn_bytes": int(ffn_bytes / tp),
        # Blended when experts and leftover dense FFN differ in dtype.
        "ffn_bpp": ffn_bytes / ffn_params if ffn_params else quant_bpp,
        # Its own row: the tower is subtracted out of ffn_params above, so
        # without this the printed rows would not add up to the total.
        "vision_params": d.vision_params,
        "vision_bytes": int(vision_bytes / tp),
        "vision_bpp": vision_bpp,
    }

    return total_bytes, breakdown


def estimate(cfg: dict, quant: str, kv_dtype: str, ctx: int,
             concurrency: int, tp: int, runtime: str,
             device_vram_gb: float, gpu_memory_utilization: float = 1.0,
             expert_dtype: str | None = None) -> dict:
    d = parse_dims(cfg)
    params = count_params(d) + d.vision_params

    # Calculate weights with mixed-precision support. The divisor is not
    # always `tp`: module placement cannot split a layer, so it caps at layer
    # granularity (weight_divisor).
    weights, mixed_breakdown = calculate_mixed_precision_weights(
        cfg, d, params, quant, weight_divisor(d, tp, runtime), expert_dtype)

    # Module placement cannot split the embedding or LM-head matrix, so no
    # device can hold less than one of them however the layers divide. Without
    # this floor a wide-vocab, few-layer model reports a per-device figure
    # below a tensor it must materialise -- a false FITS.
    weights_floor = 0
    if runtime not in TENSOR_PARALLEL_RUNTIMES:
        floor_b = unsplittable_bytes(d, quant, mixed_breakdown)
        if floor_b > weights:
            weights_floor = floor_b
            weights = floor_b

    # KV cache is sharded by KV head, so each device stores only its subset --
    # but no fewer than one head, which is why the divisor is kv_divisor() and
    # not tp.
    kv_split = kv_divisor(d, tp, runtime)
    kv = kv_bytes(d, ctx, concurrency, kv_dtype) // kv_split
    act = activation_bytes(d, ctx, concurrency,
                           quant if quant in ("bf16", "fp16") else "bf16")
    framework = int(FRAMEWORK_OVERHEAD_GB[runtime] * GB)
    total = weights + kv + act + framework
    device_b = int(device_vram_gb * GB)
    usable_b = int(device_b * gpu_memory_utilization)
    free_for_kv = usable_b - weights - act - framework
    kv_per_token = max(kv_bytes(d, 1, 1, kv_dtype) // kv_split, 1)
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
        "weights_floor": weights_floor,
        "expert_dtype": expert_dtype,
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


def print_tp_sweep(cfg: dict, args: argparse.Namespace, tp_values: list[int],
                   expert_dtype: str | None = None) -> None:
    rows = []
    first_fit = None
    for tp in tp_values:
        result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                          args.concurrency, tp, args.runtime,
                          args.device_vram_gb, args.gpu_memory_utilization,
                          expert_dtype)
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

    # A TP whose shard dimensions do not divide is not a memory question: the
    # runtime rejects it at engine init, so any verdict for it -- FITS most of
    # all -- describes a launch that cannot happen. Check --tp before
    # estimating, and name the values that do divide.
    #
    # Only for the runtimes that enforce it. TENSOR_PARALLEL_RUNTIMES shard a
    # layer across ranks by head and by FFN width; `torch` here means
    # accelerate's device_map, which places whole modules on devices and has
    # no such contract, so the same --tp there is a legitimate planning
    # request and gets a warning rather than a refusal.
    dims = parse_dims(cfg)
    shard_dims = (f"{dims.num_attn_heads} attention heads, "
                  f"{dims.num_kv_heads} KV heads, FFN width(s) "
                  f"{'/'.join(map(str, ffn_shard_widths(dims)))}")
    enforce_tp = args.runtime in TENSOR_PARALLEL_RUNTIMES
    if args.tp < 1:
        p.error("--tp must be >= 1")
    if not tp_is_shardable(dims, args.tp):
        usable = [t for t in range(1, 17) if tp_is_shardable(dims, t)]
        detail = (
            f"--tp {args.tp} does not divide this model's shard dimensions: "
            f"{shard_dims}. TP must divide the attention heads and the FFN "
            f"width, and either divide the KV heads or be a multiple of them "
            f"(the runtime replicates the group above that count).\n"
            f"  Dimensions divide at, up to 16: "
            f"{', '.join(map(str, usable)) or 'none'}")
        if enforce_tp:
            p.error(detail)
        print(f"Note: {detail}\n"
              f"  Estimating anyway: --runtime {args.runtime} splits by "
              f"module placement, not by head, so it is not bound by that "
              f"rule. Confirm the split your loader actually produces.")

    # A sweep is exploratory, so an unshardable value is dropped with a note
    # rather than failing the whole run.
    if tp_sweep and enforce_tp:
        unshardable = [t for t in tp_sweep if not tp_is_shardable(dims, t)]
        tp_sweep = [t for t in tp_sweep if tp_is_shardable(dims, t)]
        if unshardable:
            print(f"Note: dropped --tp-sweep value(s) "
                  f"{', '.join(map(str, unshardable))} -- "
                  f"{shard_dims} cannot shard that way.")

    # Auto-detect pre-quantized models from config if --quant not specified
    config_quant, width_assumed = quant_from_config(cfg)
    declared_method = declared_quant_method(cfg)
    # Declared but unpriceable: the shipped weight dtype is unknown, which is
    # not the same as "unquantized". Keep the two apart -- see shipped_base.
    unpriceable_method = bool(declared_method) and config_quant is None
    auto_quant = False
    explicit_quant = args.quant is not None
    if args.quant is None:
        args.quant = config_quant or "bf16"
        auto_quant = config_quant is not None

    # A config-declared expert dtype describes the checkpoint as shipped, so
    # honor it whenever the requested weight dtype still matches what the
    # config says -- whether it was auto-detected or the user typed the same
    # dtype explicitly. Suppress it only when --quant names a *different*
    # dtype, which posits a uniform re-quantization the config no longer
    # describes (e.g. "what would this cost in int4").
    #
    # "What the config says" is shipped_base, not config_quant. With no
    # quantization_config at all the checkpoint ships its non-expert tensors
    # unquantized, so bf16 restates it -- comparing against config_quant would
    # read an explicit --quant bf16 there as a re-quantization request and drop
    # the expert split. But a *declared* method this script cannot price is a
    # different case: the shipped base is unknown, so no explicit dtype can
    # restate it and no as-shipped figure can be quoted. shipped_base is None
    # there, which makes every explicit --quant an override.
    shipped_base = None if unpriceable_method else (config_quant or "bf16")
    config_expert_dtype = expert_dtype_of(cfg)
    # An expert dtype means nothing without experts. A dense config carrying a
    # stray expert_dtype has zero expert params, so the weight figure is right
    # either way -- but reporting an "Expert weights" line (or a suppressed-
    # expert hypothetical) for a model with no experts is not.
    if config_expert_dtype and not dims.is_moe:
        config_expert_dtype = None

    # A declared-but-unpriceable expert dtype is not the same as no expert
    # dtype. Falling through would price ~96% of an MoE at the base dtype --
    # `expert_dtype: fp6` on an fp8 checkpoint would read as fp8 experts, and
    # anything wider than the base is a false FITS. Refuse instead, unless an
    # explicit differing --quant already asked for a uniform re-quantization,
    # in which case the verdict does not depend on the unknown dtype at all.
    declared_expert = raw_expert_dtype(cfg)
    uniform_requested = explicit_quant and args.quant != shipped_base
    if (declared_expert and not config_expert_dtype and dims.is_moe
            and not uniform_requested):
        p.error(
            f"config declares expert_dtype '{declared_expert}', which this "
            f"script cannot price. The experts are most of an MoE, so "
            f"assuming the base dtype for them would be a guess in the "
            f"direction of a false FITS.\n"
            f"  Priced dtypes: {', '.join(sorted(BYTES_PER_PARAM))}\n"
            f"  Pass an explicit --quant <dtype> for a uniform "
            f"re-quantization estimate that does not depend on it, or add "
            f"'{declared_expert}' to BYTES_PER_PARAM with its measured "
            f"bytes/param.")
    expert_dtype = (config_expert_dtype
                    if not explicit_quant or args.quant == shipped_base
                    else None)

    # The quantization method behind the requested dtype, or None when no
    # method describes it: an explicit --quant that differs from the shipped
    # base is a re-quantization the config's method no longer covers, the same
    # reading the expert split is suppressed on above.
    kv_method = None if uniform_requested else declared_method
    kv_method_paired = kv_method is None or kv_method in FP8_KV_PAIRED_METHODS
    auto_kv = args.kv_dtype is None
    if auto_kv:
        # Pair on the base weight dtype: KV lives on the attention path, which
        # is what `--quant` describes. An expert dtype says nothing about it --
        # the experts are not in that path -- so it does not shrink the default
        # cache, which would be the one optimistic assumption in this report.
        args.kv_dtype = ("fp8" if (args.quant in FP8_KV_PAIRED_QUANTS
                                   and kv_method_paired
                                   and args.runtime in FP8_KV_AUTOPAIR_RUNTIMES)
                         else "bf16")
    # A narrow weight dtype that is *not* auto-paired may still have fp8 KV
    # available; say so rather than silently assuming it, and only where the
    # runtime actually has a flag for it.
    kv_flag = KV_DTYPE_FLAG.get(args.runtime)
    kv_narrow = (args.quant in SUB16_QUANTS
                 or (expert_dtype or "") in SUB16_QUANTS)
    kv_lever = auto_kv and args.kv_dtype == "bf16" and kv_narrow
    result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                      args.concurrency, args.tp, args.runtime,
                      args.device_vram_gb, args.gpu_memory_utilization,
                      expert_dtype)
    d = result["dims"]
    params = result["params"]
    def weights_at(quant: str, tp: int | None = None,
                   expert: str | None = None) -> int:
        """Weights through the same path the verdict took.

        Every figure this report quotes -- as-shipped, upcast, quant
        candidates, wider-device -- has to be one the reader can reproduce by
        re-running with those arguments. Recomputing them from a divisor
        instead skips whatever estimate() does beyond dividing: today the
        runtime-specific KV/weight divisors and the unsplittable-module
        placement floor, tomorrow whatever else lands there.
        """
        return estimate(cfg, quant, args.kv_dtype, args.ctx, args.concurrency,
                        args.tp if tp is None else tp, args.runtime,
                        args.device_vram_gb, args.gpu_memory_utilization,
                        expert)["weights"]
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
    if d.is_vlm and d.vision_params:
        print(f"Vision tower:      {d.vision_params/1e9:.2f} B params "
              f"(included in weights below)")
    elif d.is_vlm:
        # A count of 0.00 B reads as "no tower". Say which field is missing
        # instead, because the gap is in the understating direction: the
        # tower's params are real, they are just not in the figure below.
        print(f"Vision tower:      not sized -- vision_config declares no "
              f"depth/num_hidden_layers, so the weights below cover the LLM "
              f"backbone only")
        print(f"                   Qwen2.5-VL-7B-Instruct-AWQ ships such a "
              f"config: served, vLLM-XPU allocated 6.59 GiB against the "
              f"5.37 GiB estimated here")
    print(f"Total parameters:  {params/1e9:.2f} B")
    print(f"Quantization:      {args.quant}  ({bpp:.2f} bytes/param)", end="")
    if auto_quant:
        print(f"  (auto-detected from config)")
    else:
        print()
    if auto_quant and width_assumed:
        print(f"                   (config says {declared_method} without a "
              f"usable `bits` width; assumed the widest it ships. Pass "
              f"--quant int4/int3/int2 if the checkpoint is narrower)")
    elif unpriceable_method:
        print(f"                   (config declares quant_method "
              f"'{declared_method}', which this script cannot price; the "
              f"shipped weight dtype is unknown)")
        if not explicit_quant:
            print(f"                   priced at bf16, the widest it could be "
                  f"-- pass --quant if you know the real width")
    if expert_dtype:
        # "base dtype", not "non-expert tensors are": embeddings are always
        # bf16, and modules_to_not_convert can hold attention or routers there
        # too, so the breakdown below is what says where each dtype landed.
        print(f"Expert weights:    {expert_dtype}  "
              f"({BYTES_PER_PARAM[expert_dtype]:.2f} bytes/param)  "
              f"(config expert_dtype; {args.quant} is the base dtype for the "
              f"rest -- see breakdown)")
    elif config_expert_dtype:
        # Quote the as-shipped figure too. Without it, a re-quantization
        # hypothetical reads as this checkpoint's real size -- and for
        # fp4-expert models the two differ by nearly 2x. The base for that
        # figure has to come from the config, never from the --quant being
        # suppressed: an "as shipped" line built on the caller's hypothetical
        # would describe a checkpoint that does not exist. shipped_base is the
        # same value the suppression decision above was made against.
        print(f"Expert weights:    {args.quant}  (hypothetical: config "
              f"declares expert_dtype {config_expert_dtype}, suppressed "
              f"because --quant {args.quant} "
              f"{'differs' if shipped_base else 'overrides an unknown base'})")
        if shipped_base:
            shipped = weights_at(shipped_base, expert=config_expert_dtype)
            print(f"                   as shipped ({shipped_base} + "
                  f"{config_expert_dtype} experts) weights would be "
                  f"{fmt_gb(shipped)}; drop --quant for that verdict")
        else:
            # No figure to quote: the non-expert dtype is whatever
            # `declared_method` means, which this script cannot price. Inventing
            # a bf16-based "as shipped" number would describe a checkpoint that
            # does not exist.
            print(f"                   as-shipped size unknown: quant_method "
                  f"'{declared_method}' is unpriced here, so only the "
                  f"{config_expert_dtype} experts are known")
    unsupported = sorted(
        {q for q in (args.quant, expert_dtype) if q in NO_XPU_KERNEL_PATH})
    if unsupported:
        # Quote the bf16 figure: if the runtime upcasts, that is the number
        # that decides the fit, and it is 3-4x the one printed above.
        upcast = weights_at("bf16")
        print(f"Kernel path:       {'/'.join(unsupported)} has no documented "
              f"XPU kernel (this pack documents bf16/fp16, fp8, awq/gptq, "
              f"mxfp4, auto-round on XPU)")
        print(f"                   a load will refuse or upcast; upcast to "
              f"bf16 puts weights at {fmt_gb(upcast)}. Confirm live support "
              f"with **{args.runtime}-xpu-run** -- a load on another runtime "
              f"says nothing about this one's kernels")
    if auto_kv and args.kv_dtype != "bf16":
        print(f"KV dtype:          {args.kv_dtype}  (auto-paired with --quant "
              f"{args.quant} on {args.runtime}; override with --kv-dtype). "
              f"Launch with {kv_flag} or the cache costs 2x this")
    elif kv_lever:
        narrow = args.quant if args.quant in SUB16_QUANTS else expert_dtype
        if kv_flag:
            if args.runtime not in FP8_KV_AUTOPAIR_RUNTIMES:
                why = f"{args.runtime} keeps a bf16 cache unless {kv_flag} is set"
            elif args.quant in FP8_KV_PAIRED_QUANTS and not kv_method_paired:
                # Blame the method, not the dtype: this same width pairs on a
                # gptq checkpoint, so naming the dtype here would read as wrong.
                paired = ', '.join(sorted(FP8_KV_PAIRED_METHODS))
                why = (f"quant_method '{kv_method}' is not one of the rows "
                       f"vllm-xpu-run pairs with fp8 KV ({paired}), whatever "
                       f"`bits` reduces it to")
            else:
                why = (f"{narrow} weights do not imply an fp8 cache, and "
                       f"vllm-xpu-run documents `auto` KV for it")
            print(f"KV dtype:          bf16  (not auto-paired to fp8: {why})")
            print(f"                   pass --kv-dtype fp8 to halve the KV "
                  f"figure if you will launch with {kv_flag}")
        else:
            print(f"KV dtype:          bf16  (not auto-paired to fp8: the "
                  f"{args.runtime} path has no KV-dtype flag -- transformers "
                  f"allocates the cache in the model dtype)")
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
        rows = [
            (mixed_breakdown.get("embed_label", "Embeddings"), "embed"),
            (mixed_breakdown.get("attn_label", "Attention"), "attn"),
            (mixed_breakdown.get("router_label", "Routers"), "router"),
            (mixed_breakdown.get("ffn_label", "FFN/Experts"), "ffn"),
            (mixed_breakdown.get("vision_label", "Vision tower"), "vision"),
        ]
        for label, key in rows:
            if mixed_breakdown[f"{key}_params"] <= 0:
                continue
            print(f"  {label:<15} {fmt_gb(mixed_breakdown[f'{key}_bytes'])}   "
                  f"({mixed_breakdown[f'{key}_params']/1e9:.2f}B params @ "
                  f"{mixed_breakdown[f'{key}_bpp']:.2f} B/p)")
        print(f"  {'-----':<15} {'-----':>9}")
        print(f"  {'Weights total':<15} {fmt_gb(weights)}")

    # When the placement floor binds, the rows above (and a plain
    # total/divisor) no longer explain the weights figure, so say what does.
    if result.get("weights_floor"):
        print()
        print(f"Placement floor:   weights raised to {fmt_gb(weights)} -- "
              f"{d.vocab}x{d.hidden} embedding matrix")
        print(f"                   device_map places it whole, so no device "
              f"holds less than that however the {d.num_layers} layer(s) "
              f"divide. Still a floor: the device holding it also holds "
              f"layers, so verify with torch.xpu.max_memory_allocated() via "
              f"**torch-xpu-bench**")

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
            print_tp_sweep(cfg, args, tp_sweep, expert_dtype)
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
        # Same divisor the verdict used, runtime and all: a torch estimate
        # splits KV by device count, a head-sharded one caps at the KV heads.
        kv_split = kv_divisor(d, args.tp, args.runtime)
        if args.ctx > 1024:
            new_ctx = max(1024, args.ctx // 2)
            new_kv = kv_bytes(d, new_ctx, args.concurrency, args.kv_dtype) // kv_split
            suggestions.append(
                f"drop ctx {args.ctx} -> {new_ctx} (saves {fmt_gb(kv - new_kv)})"
            )
        if args.kv_dtype in ("bf16", "fp16"):
            new_kv = kv_bytes(d, args.ctx, args.concurrency, "fp8") // kv_split
            suggestions.append(
                f"kv-dtype {args.kv_dtype} -> fp8 (saves {fmt_gb(kv - new_kv)})"
            )
        if args.concurrency > 1:
            new_kv = kv_bytes(d, args.ctx, 1, args.kv_dtype) // kv_split
            suggestions.append(
                f"concurrency {args.concurrency} -> 1 (saves {fmt_gb(kv - new_kv)})"
            )
    if binding == "weights":
        # Price candidate quants through the same mixed-precision path as the
        # verdict, so "saves" cannot contradict the weights line above.
        for q in ("fp8", "int4", "int3"):
            if BYTES_PER_PARAM[q] >= BYTES_PER_PARAM[args.quant]:
                continue
            # Price the candidate exactly as running it would be priced. A
            # candidate that restates the checkpoint's own base dtype keeps
            # its expert split (that is what `--quant fp8` on an fp8 + fp4
            # config does); only a candidate that differs is a uniform
            # re-quantization. This case is reachable whenever the current
            # dtype is *wider* than shipped_base -- e.g. a failed --quant
            # bf16 run suggesting fp8 -- and getting it wrong quotes a saving
            # the suggested command does not reproduce.
            candidate_expert = (config_expert_dtype if q == shipped_base
                                else None)
            new_w = weights_at(q, expert=candidate_expert)
            if new_w < weights:
                suggestions.append(
                    f"quant {args.quant} -> {q} (saves {fmt_gb(weights - new_w)})"
                )
        # More XPUs helps a weights-bound fit at any starting TP -- an
        # already-sharded run needs "go wider", not "give up". On a
        # head-sharded runtime the next degree has to be one the model's
        # dimensions divide into, so doubling is wrong for a head count that
        # is not a power of two. Module placement has no such rule, but it
        # does have a ceiling: devices past the layer count take no work, so
        # only offer a doubling that actually buys a coarser split.
        next_tp = (next_shardable_tp(d, args.tp) if enforce_tp
                   else next_placement_tp(d, args.tp))
        if next_tp:
            # Re-estimate rather than scale: scaling assumes the whole figure
            # divides, which is false once the placement floor binds (the
            # embedding matrix does not split, however many devices arrive) and
            # false past the layer count under module placement. If the wider
            # run reports no less, there is nothing to suggest.
            at_next_tp = weights_at(args.quant, tp=next_tp,
                                    expert=expert_dtype)
            if at_next_tp < weights:
                suggestions.append(
                    f"--tp {next_tp} splits weights across {next_tp} XPUs "
                    f"(saves {fmt_gb(weights - at_next_tp)}; "
                    f"needs {next_tp} devices)"
                )
    if not suggestions:
        suggestions.append(
            "model is too big for this device at any reasonable setting; "
            "try a smaller model or more XPUs (--tp)"
        )
    # Always show the TP option for weights-bound cases, even past the cap.
    cap = len(suggestions) if binding == "weights" else 3
    print(f"  Try first: {' or '.join(suggestions[:cap])}")
    if tp_sweep:
        print_tp_sweep(cfg, args, tp_sweep, expert_dtype)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
