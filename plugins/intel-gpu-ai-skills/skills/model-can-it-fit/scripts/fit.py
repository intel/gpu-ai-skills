#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Estimate VRAM needed to serve a Hugging Face model on an Intel GPU.

Pulls config.json from the Hub, computes weights + KV cache +
recurrent state + activations + framework overhead, prints a verdict.

Handles decoder-only LLMs, mixture-of-experts, sliding-window attention,
and hybrid state-space (Mamba / Mamba2 / gated-delta-net) models. VLM
towers are counted as weights; diffusion falls through to a weights-only
floor with a clear caveat.
"""
from __future__ import annotations

import argparse
import fnmatch
import functools
import json
import re
import sys
import urllib.request
from dataclasses import dataclass, field, replace


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

# Element size of a recurrent (conv / SSM) state tensor, keyed by the dtype
# name a config or a runtime flag can use. Recurrent state is never
# quantized by either engine, so this table is separate from BYTES_PER_KV.
BYTES_PER_STATE_DTYPE = {
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0, "half": 2.0,
    "float32": 4.0, "fp32": 4.0, "float": 4.0,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0, "fp8": 1.0,
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

# Empirical floors on Arc Pro B70. The vllm entry is only the --runtime
# choice list's key now: vLLM's non-KV memory is modeled from its own
# profiling pass (vllm_reservation), calibrated on the same card.
FRAMEWORK_OVERHEAD_GB = {
    "vllm":   2.0,
    "sglang": 1.5,
    "torch":  0.8,
}

# --- vLLM's startup memory accounting ---------------------------------------
# vLLM gives the KV cache gpu_memory_utilization x the total the driver
# reports, minus everything its startup profiling pass consumed. Measured on
# an Arc Pro B70 (vllm/vllm-openai-xpu:v0.30.0, DEBUG memory-profile lines,
# 13 models): the driver reports 30.29 GiB of the card's 31.89 GiB, a 0.950
# fraction. 0.946 keeps a nameplate `--device-vram-gb 32` at or under it.
XPU_ALLOCATABLE_FRACTION = 0.946
# Memory the engine holds besides weights after profiling: runtime context,
# sampler and RoPE buffers. 0.59-0.97 GiB across seven bf16 text models;
# AWQ and MXFP4 kernels add workspace (1.10-1.12 GiB).
VLLM_RUNTIME_FLOOR_GB = 1.0
VLLM_QUANT_WORKSPACE_GB = 0.15
# The profiling pass runs one batch of max_num_batched_tokens through the
# model. Its transient peak is the MLP's intermediate tensors -- measured at
# 4 x intermediate_size x 2 bytes per token, within 0.05 GiB on every dense
# model, and linear in the batch (2048 -> 8192 tokens: 0.29 -> 1.11 GiB on
# Qwen2.5-7B). An MoE layer holds that per routed expert: 6 x
# moe_intermediate_size per token-expert on gpt-oss-20b.
VLLM_DENSE_PEAK_WIDTHS = 4
VLLM_MOE_PEAK_WIDTHS = 6
# A VLM's pass also runs the vision encoder on a maximum-size dummy input,
# and that dominates: 1.7-2.3 GiB of peak on the Qwen VL towers, and about
# 1 GiB more held afterwards. Per profiled patch: 4.4 x the tower's MLP width
# x 2 bytes at peak, and 20 KiB held. A Qwen-style tower (spatial merge, no
# fixed image size) is profiled with one image of 16384 tokens; a fixed-size
# tower (SigLIP) with as many images as one batch of tokens holds.
VLLM_VISION_PEAK_WIDTHS = 4.4
VLLM_VISION_HELD_BYTES_PER_PATCH = 20 * 1024
VLLM_QWEN_MAX_IMAGE_TOKENS = 16384

# vLLM's default --max-num-batched-tokens for an OpenAI API server, from the
# device-memory tier table in vllm/engine/arg_utils.py. Small cards get the
# 2048 tier; only >=70 GiB datacentre parts get 8192.
VLLM_LARGE_DEVICE_GIB = 70
VLLM_MAX_NUM_BATCHED_TOKENS_LARGE = 8192
VLLM_MAX_NUM_BATCHED_TOKENS_SMALL = 2048

# vLLM pages a hybrid model's recurrent state in the same block pool as its
# attention KV, so the two page sizes are made equal: the attention block is
# grown to a multiple of the kernel alignment until one block is at least one
# state page, and the state page is then padded up to that block. The
# alignment is the larger of the launch's --block-size (vllm-xpu-run passes
# 64) and the kernel minimum; on XPU a gated-delta-net layer also forces a
# multiple of 64 (platforms/xpu.py).
VLLM_HYBRID_BLOCK_ALIGN = 64
# Prefix caching is on by default for hybrid (attention + SSM) generative
# models, which switches the mamba cache to "align" mode: every running
# request holds two state pages per recurrent layer, not one
# (MambaSpec.max_memory_usage_bytes). Attention-free models keep prefix
# caching off and hold one.
VLLM_ALIGN_MODE_STATE_PAGES = 2

# SGLang does not size its sliding-window pool to the window. It provisions
# the window pool as a fraction of the full-attention pool so the radix
# prefix cache still has something to reuse (--swa-full-tokens-ratio).
SGLANG_SWA_FULL_TOKENS_RATIO = 0.8

# ...and only for the architectures it treats as hybrid SWA
# (is_hybrid_swa_model in srt/configs/model_config.py). Every other model --
# Mistral, Phi-3, Qwen2, Cohere2, and Gemma-2/3, which SGLang explicitly opts
# out -- gets a full-length KV pool on every layer, window or not.
SGLANG_HYBRID_SWA_ARCHS = frozenset({
    "Llama4ForConditionalGeneration",
    "DeepseekV4ForCausalLM",
    "GptOssForCausalLM", "GraniteSWAForCausalLM", "GraniteMoeSWAForCausalLM",
    "MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM",
    "Step3p5ForCausalLM", "Step3p7ForConditionalGeneration",
    "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
    "LagunaForCausalLM", "MellumForCausalLM",
})

# SGLang splits the memory left after weights between a fixed recurrent-state
# pool and the KV pool for hybrid models: the state pool gets
# ratio / (1 + ratio) of it (--mamba-full-memory-ratio, default 0.9). Neither
# pool can borrow from the other, so a hybrid fits only when *both* halves do.
SGLANG_MAMBA_FULL_MEMORY_RATIO = 0.9

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
    # Width of a shared expert when the config sizes it apart from the
    # routed ones (Qwen3-Next, Nemotron-3); 0 means `intermediate`.
    shared_intermediate: int = 0
    # Weight matrices per FFN: 3 for gated (SwiGLU) MLPs, 2 for the
    # squared-ReLU MLPs Nemotron-H uses for both its dense and expert FFNs.
    ffn_matrices: int = 3
    is_vlm: bool = False
    # Every encoder tower's weights (vision, and Gemma-4's audio tower), and
    # the vision blocks' attention / MLP share of it for partial exclusions.
    vision_params: int = 0
    vision_attn_params: int = 0
    vision_mlp_params: int = 0
    vision_num_heads: int = 0
    # Gemma-4 (and Gemma-3n): per-layer input embedding tables plus their
    # projection -- nn.Embedding weights, priced with the embeddings.
    per_layer_embed_params: int = 0
    # Transformer blocks in the vision tower, for exclusion lists that name
    # the tower one layer at a time.
    vision_layers: int = 0
    # Which module naming the tower uses (_VISION_NAMINGS key, "" unknown).
    vision_naming: str = ""
    # What vLLM's profiling pass feeds the tower: patches per image and the
    # tokens each image becomes (0 for a Qwen-style tower, which is profiled
    # with one maximum-size image instead), and the tower's MLP width.
    vision_patches_per_image: int = 0
    vision_tokens_per_image: int = 0
    vision_intermediate: int = 0
    # `architectures[0]`: SGLang keys its sliding-window pool on it.
    architecture: str = ""

    # --- Layer composition -------------------------------------------------
    # For a plain dense model every layer is full attention and every layer
    # carries a feed-forward block, so these default to num_layers in
    # parse_dims. Hybrids split the total: an SSM layer holds recurrent state
    # instead of a KV cache, a sliding-window layer holds a bounded KV cache,
    # and some families (Nemotron-H, Bamba) interleave attention-free and
    # feed-forward-free layers.
    num_full_attn_layers: int = 0
    num_sliding_attn_layers: int = 0
    num_recurrent_layers: int = 0
    num_ffn_layers: int = 0
    sliding_window: int = 0
    # Gemma-4: full-attention layers use a wider head (global_head_dim) and
    # may have their own KV head count; 0 means the same as the rest.
    full_head_dim: int = 0
    full_kv_heads: int = 0
    # Gemma-4: the last `num_kv_shared_layers` layers reuse an earlier
    # layer's cache and allocate none of their own, split by kind.
    kv_shared_full: int = 0
    kv_shared_sliding: int = 0
    # Llama-4's `attention_chunk_size` is chunked local attention, not a
    # sliding window: vLLM holds a whole chunk plus the in-flight tokens.
    window_is_chunked: bool = False
    # Zamba2 runs a few *shared* attention + MLP blocks from its hybrid
    # layers: every hybrid layer keeps its own KV cache, but the weights
    # exist once per shared block. 0 everywhere else.
    shared_attn_blocks: int = 0
    # Input width of the Q/K/V projections when it is not `hidden` (Zamba2
    # feeds the shared block the hidden state concatenated with the
    # embeddings). 0 means `hidden`.
    attn_in_dim: int = 0

    # --- Recurrent state geometry -----------------------------------------
    # Normalized across Mamba, Mamba2, gated-delta-net, KDA and short-conv.
    # The per-layer SSM state is (num_state_heads, state_head_dim_k,
    # state_head_dim_v) and the per-layer conv state is
    # (conv_dim, conv_kernel - 1). `state_kind` says which engine formula
    # shards it: "gdn", "kda", "mamba1", "mamba2" or "short_conv".
    state_kind: str = ""
    num_state_heads: int = 0
    state_head_dim_k: int = 0
    state_head_dim_v: int = 0
    conv_dim: int = 0
    conv_kernel: int = 0
    state_groups: int = 1
    state_inner: int = 0
    # Mamba1 only: rank of the low-rank dt projection (x_proj/dt_proj).
    state_dt_rank: int = 0
    conv_dtype: str = "bfloat16"
    ssm_dtype: str = "bfloat16"
    # Falcon-H1 runs attention and an SSM in every layer, and Zamba2's hybrid
    # layers do the same, so those layers are charged a full KV cache *and* a
    # recurrent state.
    attn_on_recurrent_layers: bool = False
    # Gated delta net and KDA split their heads with a plain divide, so vLLM
    # refuses a TP degree that doesn't divide them. Mamba2 has its own rule
    # in tp_problems().
    state_heads_divide_tp: bool = False
    # MLA (DeepSeek-V2/V3, Kimi-K2, Ling-3.0) caches one compressed latent of
    # kv_lora_rank + qk_rope_head_dim per token per layer instead of K and V,
    # and every rank holds the whole latent. 0 for ordinary attention.
    mla_kv_dim: int = 0
    # Which config keys the state geometry was actually read from, for the
    # refusal message and for tests.
    state_source: list[str] = field(default_factory=list)


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


def _first_int(value: object, default: int = 0) -> int:
    """An int from a config field that some families write as a list.

    Gemma-4 and others spell `patch_size` / `image_size` as `[h, w]`; the
    first entry is the one that sizes a square patch grid.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _vision_param_count(vc: dict, text_hidden: int = 0) -> int:
    """Approximate parameter count of a ViT-style vision tower."""
    return sum(_vision_param_split(vc, text_hidden).values())


def _vision_param_split(vc: dict, text_hidden: int = 0) -> dict[str, int]:
    """{"attn", "mlp", "extra"} parameter counts of a ViT-style vision tower.

    Kept apart because exclusion lists name them apart: a list that holds the
    tower's attention wide but quantizes its MLP prices the two at their real
    sizes (4 h^2 against 3 h I per block, about 1:3 on a gated tower), and
    `visual.merger` or `patch_embed` entries reach the extras.

    Besides the blocks: the patch embedding (times `temporal_patch_size` on
    a video-capable tower), the position table, and whatever maps the tower
    into the language model -- Qwen's patch merger MLP, of which Qwen3-VL has
    one per deepstack index plus the final one (109 M params on Qwen3-VL-4B,
    6% of the model), or Gemma-4's embedding projection.

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
        return {"attn": 0, "mlp": 0, "extra": 0}
    # ViT layer: attn (4 h^2) + MLP + 2 norms (2 h). A gated MLP holds three
    # matrices (gate/up/down), not two: Qwen2.5-VL's tower is SwiGLU and says
    # so with `hidden_act: silu`, where CLIP-style towers are gelu/quick_gelu
    # with two. Counting two on a gated tower is 22% light on the blocks, and
    # light on weights is the direction that reports a false FITS.
    act = str(vc.get("hidden_act") or vc.get("hidden_activation") or "").lower()
    vtype = str(vc.get("model_type", "")).lower()
    # Gemma-4's tower is gated too (measured 4h^2 + 3hI per block) although
    # its activation is spelled gelu_pytorch_tanh.
    mlp_mats = 3 if (any(g in act for g in ("silu", "swish", "glu"))
                     or vtype == "gemma4_vision") else 2
    attn = layers * (4 * h * h + 2 * h)          # Q/K/V/O and the two norms
    mlp = layers * mlp_mats * h * intermediate
    # Patch embedding: hidden * patch_area * channels, times the temporal
    # patch on a video-capable tower.
    patch = _first_int(vc.get("patch_size"), 14)
    in_channels = _first_int(vc.get("num_channels") or vc.get("in_chans"), 3)
    temporal = _first_int(vc.get("temporal_patch_size"), 1) or 1
    embed = patch * patch * in_channels * temporal * h
    pos = _first_int(vc.get("num_position_embeddings")
                     or vc.get("position_embedding_size")) * h
    merge = _first_int(vc.get("spatial_merge_size"), 1) or 1
    out_hidden = _first_int(vc.get("out_hidden_size")) or text_hidden
    projector = 0
    if vc.get("spatial_merge_size") and out_hidden:
        # Qwen2-VL / 2.5-VL / 3-VL / 3.5 merger: fc1 (h m^2)^2, fc2 to the
        # LLM width; Qwen3-VL repeats it for every deepstack index.
        m2 = h * merge * merge
        mergers = 1 + len(vc.get("deepstack_visual_indexes") or [])
        projector = mergers * (m2 * m2 + m2 * out_hidden)
    elif vtype == "gemma4_vision" and out_hidden:
        projector = h * out_hidden
    # int(): mlp_ratio is a float in some configs, and a param count is not.
    return {"attn": int(attn), "mlp": int(mlp), "extra": int(embed + pos + projector)}


def _audio_param_count(ac: dict, text_hidden: int = 0) -> int:
    """Approximate parameter count of a Gemma-4 style conformer audio tower.

    Measured on Gemma-4-E4B: 12 blocks at hidden 1024 are 302 M params,
    24 h^2 per block (attention, two 4x feed-forward modules and the conv
    module), plus the output projection and the embedding projection into
    the language model.
    """
    h = _first_int(ac.get("hidden_size"))
    layers = _first_int(ac.get("num_hidden_layers"))
    if not (h and layers):
        return 0
    out = _first_int(ac.get("output_proj_dims")) or h
    return int(layers * 24 * h * h + h * out + out * (text_hidden or 0))


def _vision_profile_shape(cfg: dict, vc: dict) -> tuple[int, int]:
    """(patches per profiled image, tokens per image) for vLLM's encoder pass.

    A tower with a fixed input size (SigLIP's 896 px at patch 14: 4096
    patches pooled to `mm_tokens_per_image`) states both. A Qwen-style tower
    resizes images and only merges patches, so vLLM profiles it with one
    maximum-size image of VLLM_QWEN_MAX_IMAGE_TOKENS tokens; tokens comes back
    0 to say so. A tower that states neither is charged the Qwen-sized image,
    the larger of the two.
    """
    merge = _first_int(vc.get("spatial_merge_size"), 1) or 1
    image = _first_int(vc.get("image_size"))
    patch = _first_int(vc.get("patch_size"))
    if image and patch:
        per_image = (image // patch) ** 2
        tokens = int(cfg.get("mm_tokens_per_image") or cfg.get("image_seq_length")
                     or 0)
        if not tokens:
            ratio = cfg.get("downsample_ratio")
            tokens = (int(per_image * float(ratio) ** 2) if ratio
                      else per_image // (merge * merge))
        return per_image, max(tokens, 1)
    return VLLM_QWEN_MAX_IMAGE_TOKENS * max(merge, 2) ** 2, 0


# Names a config's `layer_types` list can use. Anything unrecognized is
# treated as full attention, which is the conservative direction: a full KV
# cache is the most expensive thing a layer can hold.
_RECURRENT_LAYER_TYPES = {
    "mamba", "mamba2", "ssm", "recurrent", "conv",
    "linear_attention", "linear_attn", "gated_delta_net",
}
_SLIDING_LAYER_TYPES = {
    "sliding_attention", "sliding_window_attention", "local_attention",
    "local_sliding_attention", "chunked_attention", "chunked_local_attention",
}
_FULL_LAYER_TYPES = {"full_attention", "attention", "global_attention", "full"}
_FFN_ONLY_LAYER_TYPES = {"mlp", "moe", "feed_forward", "ffn"}
# Zamba2's `layers_block_type`: a Mamba layer that also calls a shared
# attention block, so it holds a recurrent state *and* a KV cache.
_HYBRID_LAYER_TYPES = {"hybrid"}

# Single characters used by the `hybrid_override_pattern` string
# (Nemotron-H, Nemotron-3): M = Mamba layer, * = attention layer,
# - = MLP layer, E = MoE layer.
_PATTERN_CHARS = {"m": "recurrent", "*": "full", "-": "ffn", "_": "ffn",
                  "e": "ffn"}

# Pure state-space families: every layer is a recurrent mixer with no
# attention and no feed-forward block, and the config carries no attention
# head count at all.
_PURE_SSM_FAMILIES = {"mamba", "mamba2", "falcon_mamba"}

# Families that run attention and an SSM inside the *same* layer rather than
# alternating them, so every layer is charged both a KV cache and a state.
# Zamba2 ships a `layers_block_type` list, which is read first and charges
# only its "hybrid" layers that way.
_PARALLEL_HYBRID_FAMILIES = {"falcon_h1", "zamba", "zamba2"}

# Families whose SSM state vLLM keeps in float32 when the config does not
# name a dtype (NemotronHForCausalLMConfig.DEFAULT_MAMBA_SSM_CACHE_DTYPE).
_FP32_SSM_STATE_FAMILIES = {"nemotron_h"}

# A bare `sliding_window` key does not mean the model uses a sliding window.
# DeepSeek-V4-Flash carries `sliding_window: 128` for its sparse attention
# indexer while every layer still holds a full KV cache -- honouring that key
# blindly under-estimates its cache by 8x, and under-estimating is what
# produces a FITS verdict for a launch that dies at engine init. So a window
# is only applied when the config says which layers use it, or when the
# family is known to apply it everywhere.
_GLOBAL_SWA_FAMILIES = {
    "mistral", "mistral3", "ministral", "mixtral",
    "phi3", "phi3small", "phimoe",
    "starcoder2",
}

# Qwen configs carry a `sliding_window` value whether or not it is used.
# transformers drops it unless `use_sliding_window` is true -- and the field
# defaults to false, so a config that omits it has no window -- and then
# windows only the layers at index >= `max_window_layers`.
_QWEN_GATED_SWA_FAMILIES = {"qwen2", "qwen2_moe", "qwen2_vl", "qwen3",
                            "qwen3_moe"}

# Families whose sliding/global interleave lives in the modeling code rather
# than in any config key, mapped to the period between global layers. Cohere2
# is not here: every real checkpoint ships `sliding_window_pattern`, which the
# generic stride lookup below already reads.
_INTERLEAVED_SWA_PERIOD = {"gemma2": 2}

# Text-config fields some checkpoints leave to the transformers class
# defaults. gemma-3-4b-it's text_config omits the head counts entirely, so
# without these the model cannot be sized at all.
_TEXT_CONFIG_DEFAULTS = {
    "gemma3_text": {
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
        "vocab_size": 262208, "sliding_window_pattern": 6,
        "tie_word_embeddings": True,
    },
}


def _normalize_dtype(name: object, default: str = "bfloat16") -> str:
    """Map a config/flag dtype name onto a BYTES_PER_STATE_DTYPE key."""
    text = str(name or "").lower().strip()
    if text in ("", "auto", "none"):
        return default
    text = text.replace("torch.", "")
    return text if text in BYTES_PER_STATE_DTYPE else default


def _lookup(sources: list[dict], keys: tuple[str, ...]) -> tuple[object, str]:
    """First truthy value among `keys` across `sources`, with the key used."""
    for src in sources:
        for key in keys:
            value = src.get(key)
            if value not in (None, 0, "", [], {}):
                return value, key
    return None, ""


def _scale_counts(counts: dict, listed: int, num_layers: int) -> dict:
    """Rescale a per-layer tally that does not cover exactly num_layers.

    A layout list is occasionally shorter or longer than num_hidden_layers --
    a repeating block, or a config key this script has matched to the wrong
    family. The ratio between layer kinds is the load-bearing part, so it is
    preserved and applied to the real layer count instead of the tally being
    thrown away (which would silently charge every layer a full KV cache).
    """
    if listed == num_layers or listed <= 0:
        return counts
    # Largest-remainder rounding, so the kinds still sum to num_layers:
    # rounding each on its own turns three 1-layer kinds scaled to 4 layers
    # into 1+1+1 and drops a layer's cache. Ties go to the earlier, costlier
    # kind (full before sliding before recurrent).
    exact = {key: value * num_layers / listed for key, value in counts.items()}
    scaled = {key: int(value) for key, value in exact.items()}
    short = num_layers - sum(scaled.values())
    by_remainder = sorted(exact, key=lambda key: scaled[key] - exact[key])
    for key in by_remainder[:max(short, 0)]:
        scaled[key] += 1
    return scaled


def _layer_type_counts(types: list) -> dict:
    """Tally an explicit per-layer list into layer kinds.

    A "hybrid" entry is counted under its own key, because it is both a
    recurrent layer and an attention layer; _classify_layers adds it to both
    after any rescaling, so the kinds being rescaled still sum to the list.
    """
    counts = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": 0, "hybrid": 0}
    for entry in types:
        name = str(entry).lower()
        if name in _RECURRENT_LAYER_TYPES:
            counts["recurrent"] += 1
        elif name in _SLIDING_LAYER_TYPES:
            counts["sliding"] += 1
        elif name in _FFN_ONLY_LAYER_TYPES:
            counts["ffn"] += 1
        elif name in _HYBRID_LAYER_TYPES:
            counts["hybrid"] += 1
        else:
            counts["full"] += 1
    return counts


def _sliding_window(cfg: dict, text_cfg: dict, family: str = "") -> tuple[int, bool]:
    """(attention window in tokens, window is chunked), or (0, False).

    A window equal to or wider than the trained context is not a window --
    several configs carry `sliding_window` alongside
    `use_sliding_window: false`, and Llama-4 spells its window
    `attention_chunk_size`, which is chunked local attention rather than a
    sliding window and is held differently.
    """
    sources = [text_cfg, cfg]
    for src in sources:
        if src.get("use_sliding_window") is False:
            return 0, False
    if _QWEN_GATED_SWA_FAMILIES & {family, text_cfg.get("model_type")} and not any(
            src.get("use_sliding_window") is True for src in sources):
        return 0, False
    window, key = _lookup(sources, ("sliding_window", "attention_chunk_size",
                                    "window_size", "attention_window_size"))
    if isinstance(window, (list, tuple)):
        # Some configs give a (left, right) pair; the left half is the span
        # of past tokens a query can see, which is what the cache must hold.
        window = window[0] if window else 0
    try:
        return max(int(window or 0), 0), key == "attention_chunk_size"
    except (TypeError, ValueError):
        return 0, False


def _is_kda(sources: list[dict]) -> bool:
    """Kimi-delta-attention hybrids (Ling-3.0 / BailingMoeV3) carry kda_* keys."""
    return any(key.startswith("kda_") or "_kda_" in key
               for src in sources for key in src)


def _classify_layers(cfg: dict, text_cfg: dict, family: str,
                     num_layers: int, window: int) -> dict:
    """Split num_layers into full-attention / sliding / recurrent / FFN counts.

    Reads whichever of the mutually exclusive layout keys a family uses:
    `layer_types` (Gemma-3, Qwen3-Next, gpt-oss, Llama-4),
    `hybrid_override_pattern` (Nemotron-H, Bamba), `attn_layer_indices`, or a
    periodic interleave (`attn_layer_period`, `full_attention_interval`,
    `sliding_window_pattern`).
    """
    sources = [text_cfg, cfg]
    out = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": num_layers,
           "attn_on_recurrent": False, "layout_source": ""}

    def from_list(entries: list, key: str) -> dict:
        counts = _scale_counts(_layer_type_counts(entries), len(entries),
                               num_layers)
        hybrid = counts.pop("hybrid", 0)
        counts["full"] += hybrid
        counts["recurrent"] += hybrid
        out.update(counts, layout_source=key,
                   attn_on_recurrent=bool(hybrid))
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out

    layer_types, key = _lookup(sources, ("layer_types", "layer_type_list"))
    if isinstance(layer_types, (list, tuple)) and layer_types:
        return from_list(list(layer_types), key)

    # Kimi-Linear nests its layout under linear_attn_config, as 1-indexed
    # lists of the KDA layers and the full-attention (MLA) layers.
    linear_cfg = text_cfg.get("linear_attn_config") or cfg.get("linear_attn_config")
    if isinstance(linear_cfg, dict) and isinstance(
            linear_cfg.get("kda_layers"), (list, tuple)):
        kda = {int(i) for i in linear_cfg["kda_layers"]}
        recurrent = sum(1 for i in range(1, num_layers + 1) if i in kda)
        out.update(full=num_layers - recurrent, recurrent=recurrent,
                   layout_source="linear_attn_config.kda_layers")
        return out

    no_rope, key = _lookup(sources, ("no_rope_layers",))
    if isinstance(no_rope, (list, tuple)) and no_rope and window:
        # Llama-4: 1 = layer keeps RoPE and the chunked/local window, 0 = the
        # "NoPE" layer that runs full, unbounded attention instead.
        sliding = sum(1 for v in no_rope if v)
        counts = _scale_counts(
            {"full": len(no_rope) - sliding, "sliding": sliding,
             "recurrent": 0, "ffn": 0},
            len(no_rope), num_layers)
        out.update(counts, layout_source=key)
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out

    pattern, key = _lookup(sources, ("hybrid_override_pattern",
                                     "layers_block_type"))
    if isinstance(pattern, (list, tuple)) and pattern:
        return from_list(list(pattern), key)
    if isinstance(pattern, str) and pattern:
        counts = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": 0}
        for char in pattern:
            counts[_PATTERN_CHARS.get(char.lower(), "full")] += 1
        if window and counts["full"]:
            counts["sliding"], counts["full"] = counts["full"], 0
        counts = _scale_counts(counts, len(pattern), num_layers)
        out.update(counts, layout_source=key)
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out

    if family in _PURE_SSM_FAMILIES:
        out.update(recurrent=num_layers, ffn=0, layout_source="model_type")
        return out

    if family in _PARALLEL_HYBRID_FAMILIES:
        # Attention and SSM sit side by side in every layer.
        out.update(full=num_layers, recurrent=num_layers,
                   attn_on_recurrent=True, layout_source="model_type")
        return out

    indices, key = _lookup(sources, ("attn_layer_indices",))
    if isinstance(indices, (list, tuple)):
        attn = len({int(i) for i in indices if 0 <= int(i) < num_layers})
        out.update(full=attn, recurrent=num_layers - attn, layout_source=key)
        return out

    period, key = _lookup(sources, ("attn_layer_period",))
    if period:
        attn = num_layers // int(period)
        out.update(full=attn, recurrent=num_layers - attn, layout_source=key)
        return out

    group, key = _lookup(sources, ("layer_group_size",))
    if group and _is_kda(sources):
        # BailingMoeV3: the last layer of each group, and every layer in the
        # tail past the last whole group, is MLA; the rest are KDA.
        group = int(group)
        tail = num_layers // group * group
        full = sum(1 for i in range(num_layers)
                   if (i + 1) % group == 0 or i >= tail)
        out.update(full=full, recurrent=num_layers - full, layout_source=key)
        return out

    interval, key = _lookup(sources, ("full_attention_interval",))
    if interval:
        full = num_layers // int(interval)
        out.update(full=full, recurrent=num_layers - full, layout_source=key)
        return out

    stride, key = _lookup(sources, ("sliding_window_pattern",
                                    "global_attn_every_n_layers"))
    if stride and window:
        # Gemma-3, Cohere2: every `stride`-th layer is global, rest windowed.
        full = max(num_layers // int(stride), 1)
        out.update(full=full, sliding=num_layers - full, layout_source=key)
        return out

    if window and family in _INTERLEAVED_SWA_PERIOD:
        period = _INTERLEAVED_SWA_PERIOD[family]
        full = max(num_layers // period, 1)
        out.update(full=full, sliding=num_layers - full,
                   layout_source="model_type")
        return out

    if window and _QWEN_GATED_SWA_FAMILIES & {family, text_cfg.get("model_type")}:
        # _sliding_window() only returns a window for these when
        # use_sliding_window is true. Layers below max_window_layers stay
        # full; transformers' default for the field is 28. (vLLM refuses a
        # layout that mixes the two, so in practice it is all or nothing.)
        # Read directly: 0 is a real value here (every layer windowed), and
        # _lookup treats 0 as absent.
        first = next((src["max_window_layers"] for src in sources
                      if src.get("max_window_layers") is not None), 28)
        first = int(first)
        sliding = max(num_layers - min(first, num_layers), 0)
        out.update(full=num_layers - sliding, sliding=sliding,
                   layout_source="max_window_layers")
        return out

    if window and family in _GLOBAL_SWA_FAMILIES:
        out.update(sliding=num_layers, layout_source="sliding_window")
        return out

    # Window present but unattributable: charge full attention everywhere.
    out.update(full=num_layers)
    return out


def _state_geometry(cfg: dict, text_cfg: dict, hidden: int,
                    pure_ssm: bool = False) -> dict | None:
    """Normalize a family's recurrent-state keys into one shape.

    The per-layer SSM state is (num_state_heads, state_head_dim_k,
    state_head_dim_v) and the per-layer conv state is (conv_dim,
    conv_kernel - 1). Mamba1 collapses to a single head whose value
    dimension is the whole inner width. Returns None when no geometry key
    resolves, so the caller can refuse rather than guess.
    """
    linear_cfg = text_cfg.get("linear_attn_config") or cfg.get("linear_attn_config")
    linear_cfg = linear_cfg if isinstance(linear_cfg, dict) else {}
    sources = [text_cfg, cfg]
    seen: list[str] = []

    def take(keys: tuple[str, ...]):
        value, key = _lookup(sources, keys)
        if key:
            seen.append(key)
        return value

    # Gated delta net (Qwen3-Next, Kimi-Linear): key and value heads are
    # sized independently, and the conv state spans q, k and v.
    num_v_heads = take(("linear_num_value_heads", "num_v_heads"))
    if num_v_heads:
        num_k_heads = take(("linear_num_key_heads", "num_k_heads")) or num_v_heads
        dim_k = take(("linear_key_head_dim", "head_k_dim")) or 0
        dim_v = take(("linear_value_head_dim", "head_v_dim")) or dim_k
        conv_kernel = take(("linear_conv_kernel_dim", "conv_kernel_size")) or 4
        if not (dim_k and dim_v):
            return None
        return {
            "kind": "gdn",
            "num_state_heads": int(num_v_heads),
            "state_head_dim_k": int(dim_k),
            "state_head_dim_v": int(dim_v),
            "conv_dim": int(2 * dim_k * num_k_heads + dim_v * num_v_heads),
            "conv_kernel": int(conv_kernel),
            "state_groups": int(num_k_heads),
            "state_inner": int(dim_v * num_v_heads),
            "state_source": seen,
            "divides_tp": True,
        }

    # Kimi delta attention (Ling-3.0, Kimi-Linear): one head count and
    # head_dim shared by q, k and v, a short conv over all three, and a
    # (head_dim, head_dim) recurrent state per head. Kimi-Linear nests the
    # geometry under linear_attn_config, where head_dim is the KDA head and
    # not the top-level MLA one.
    if _is_kda(sources) or "kda_layers" in linear_cfg:
        kda_sources = [linear_cfg] + sources
        heads, key = _lookup(kda_sources, ("num_heads", "num_attention_heads"))
        seen.append(key)
        dim, key = _lookup(kda_sources, ("head_dim",))
        seen.append(key)
        conv_kernel = _lookup(kda_sources, ("short_conv_kernel_size",))[0] or 4
        if not (heads and dim):
            return None
        return {
            "kind": "kda",
            "num_state_heads": int(heads),
            "state_head_dim_k": int(dim),
            "state_head_dim_v": int(dim),
            "conv_dim": 3 * int(heads) * int(dim),
            "conv_kernel": int(conv_kernel),
            "state_groups": int(heads),
            "state_inner": int(heads) * int(dim),
            "state_source": seen,
            "divides_tp": True,
            # vLLM keeps the KDA recurrent state in fp32 unless told otherwise.
            "ssm_dtype": "float32",
        }

    # Short convolution (LFM2): a conv state and nothing else -- no SSM.
    # vLLM sizes it from conv_dim on LFM2 and from hidden_size on LFM2-MoE.
    l_cache = take(("conv_L_cache",))
    if l_cache:
        width = take(("conv_dim",)) or hidden
        return {
            "kind": "short_conv",
            "num_state_heads": 0,
            "conv_dim": int(width),
            "conv_kernel": int(l_cache),
            "state_inner": int(width),
            "state_source": seen,
        }

    # Mamba / Mamba2 (Nemotron-H, Bamba, Falcon-H1, Codestral-Mamba, Jamba,
    # Zamba2).
    d_state = take(("mamba_d_state", "state_size", "ssm_state_size", "d_state"))
    conv_kernel = take(("mamba_d_conv", "conv_kernel", "d_conv"))
    if not (d_state and conv_kernel):
        return None

    n_heads = take(("mamba_n_heads", "mamba_num_heads", "n_mamba_heads"))
    head_dim = take(("mamba_d_head", "mamba_head_dim", "mamba_headdim"))
    if pure_ssm:
        # Mamba2Config has no attention to disambiguate from, so it uses the
        # plain names a hybrid reserves for its attention heads.
        n_heads = n_heads or take(("num_heads",))
        head_dim = head_dim or take(("head_dim",))

    # Inner width, in descending order of how directly a family states it.
    # Falcon-H1 gives mamba_d_ssm outright. When the head split is stated,
    # vLLM's Mamba2 layers are heads x head_dim wide whatever the expansion
    # factor says -- Nemotron-3-Nano ships expand=2 beside a 64 x 64 split,
    # 5376 against the 4096 vLLM allocates. Otherwise most configs give an
    # expansion factor.
    inner = take(("mamba_d_ssm", "mamba_expand_inner_size"))
    if not inner and pure_ssm:
        # MambaConfig, Mamba2Config and FalconMambaConfig all store the inner
        # width as intermediate_size, and the modeling code reads only that:
        # falcon-mamba-7b ships expand=16 beside intermediate_size=8192.
        inner = take(("intermediate_size",))
    if not inner and n_heads and head_dim:
        inner = int(n_heads) * int(head_dim)
    if not inner:
        expand = take(("mamba_expand", "expand"))
        if expand:
            inner = int(expand) * hidden
    if not inner:
        inner = take(("mamba_intermediate_size", "mamba_d_inner", "d_inner"))
    if not inner:
        return None

    if n_heads and not head_dim:
        head_dim = int(inner) // int(n_heads)
    elif head_dim and not n_heads:
        n_heads = int(inner) // int(head_dim)
    dt_rank = 0
    groups = take(("mamba_n_groups", "n_groups", "mamba_ngroups")) or 1
    if not (n_heads and head_dim):
        # Mamba1: one (d_inner, d_state) state per layer, no head split, and
        # a conv over the inner stream alone -- B and C are projected after
        # the conv, not convolved with it.
        dt_rank = take(("time_step_rank", "mamba_dt_rank", "dt_rank"))
        if not isinstance(dt_rank, int):
            # "auto", or absent: the reference implementation's default.
            dt_rank = -(-int(hidden) // 16)
        return {
            "kind": "mamba1",
            "num_state_heads": 1,
            "state_head_dim_k": int(d_state),
            "state_head_dim_v": int(inner),
            "conv_dim": int(inner),
            "conv_kernel": int(conv_kernel),
            "state_groups": 1,
            "state_inner": int(inner),
            "dt_rank": int(dt_rank),
            "state_source": seen,
        }

    return {
        "kind": "mamba2",
        "num_state_heads": int(n_heads),
        "state_head_dim_k": int(d_state),
        "state_head_dim_v": int(head_dim),
        "conv_dim": int(inner) + 2 * int(groups) * int(d_state),
        "conv_kernel": int(conv_kernel),
        "state_groups": int(groups),
        "state_inner": int(inner),
        "dt_rank": 0,
        "state_source": seen,
    }


def parse_dims(cfg: dict) -> ModelDims:
    arch = (cfg.get("architectures") or ["unknown"])[0]
    family = cfg.get("model_type", "unknown")
    is_vlm = _vlm_signal(cfg)

    # For VLMs the LLM-backbone fields often live under 'text_config' or 'llm_config'
    # rather than at the root.
    # text_config: Qwen2-VL, Llama-3.2-Vision, etc.
    # llm_config: Nemotron-3-Nano-Omni, some other multimodal models
    text_cfg = cfg.get("text_config") or cfg.get("llm_config") or cfg
    defaults = _TEXT_CONFIG_DEFAULTS.get(str(text_cfg.get("model_type", "")))
    if defaults:
        text_cfg = {**defaults, **text_cfg}

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
    vision_num_heads = 0
    vision_layers = 0
    vision_patches_per_image = vision_tokens_per_image = vision_intermediate = 0
    vision_attn_params = vision_mlp_params = 0
    if isinstance(cfg.get("audio_config"), dict):
        # Gemma-4 E-series: the audio tower loads with the model whether or
        # not a request sends audio.
        vision_params += _audio_param_count(cfg["audio_config"], int(hidden or 0))
    if is_vlm and isinstance(cfg.get("vision_config"), dict):
        vision_cfg = cfg["vision_config"]
        # += : an audio tower may already be counted here.
        split = _vision_param_split(vision_cfg, int(hidden or 0))
        vision_params += sum(split.values())
        vision_attn_params, vision_mlp_params = split["attn"], split["mlp"]
        vision_num_heads = int(
            vision_cfg.get("num_attention_heads")
            or vision_cfg.get("num_heads")
            or 0
        )
        vision_layers = int(vision_cfg.get("depth")
                            or vision_cfg.get("num_hidden_layers") or 0)
        vision_patches_per_image, vision_tokens_per_image = _vision_profile_shape(
            cfg, vision_cfg)
        width = vision_cfg.get("embed_dim") or vision_cfg.get("hidden_size") or 0
        ratio = vision_cfg.get("mlp_ratio")
        vision_intermediate = int(vision_cfg.get("intermediate_size")
                                  or (width * ratio if ratio else 4 * width))

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
    # Qwen2-MoE and Qwen3-Next declare their one shared expert only by its
    # width, and Nemotron-3 sizes it apart from the routed experts.
    shared_intermediate = _lookup([text_cfg, cfg], (
        "moe_shared_expert_intermediate_size",
        "shared_expert_intermediate_size"))[0] or 0
    if shared_intermediate and not num_shared_experts:
        num_shared_experts = 1
    gated = str(text_cfg.get("mlp_hidden_act") or "").lower() != "relu2"

    # Hybrid MoE: some models (DeepSeek-V2/V3/V4) replace first K layers with
    # dense FFN; LFM2-MoE calls the same thing num_dense_layers.
    first_k_dense_replace = (
        text_cfg.get("first_k_dense_replace",
        cfg.get("first_k_dense_replace",
        moe_cfg.get("first_k_dense_replace",
        text_cfg.get("num_dense_layers", 0))))
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

    pure_ssm = family in _PURE_SSM_FAMILIES
    if pure_ssm:
        # No attention layer exists to read heads for; Mamba2's `head_dim`
        # belongs to the SSM heads and is read by _state_geometry.
        num_attn, num_kv, head_dim = 0, 0, 0
    missing = [k for k, v in {
        "hidden_size": hidden, "num_hidden_layers": num_layers,
        "num_attention_heads": num_attn or pure_ssm,
    }.items() if not v]
    if missing:
        sys.exit(
            f"config.json is missing required keys for an LLM: {missing}. "
            f"Architecture reported: {arch} / {family}. "
            f"This calculator only handles decoder-only LLMs reliably."
        )

    window, window_is_chunked = _sliding_window(cfg, text_cfg, family)
    layout = _classify_layers(cfg, text_cfg, family, int(num_layers), window)
    if not layout["sliding"]:
        # The config named a window but nothing said which layers use it.
        window, window_is_chunked = 0, False

    # Gemma-4: full-attention layers get their own head geometry, and the
    # last `num_kv_shared_layers` layers reuse an earlier layer's KV cache
    # (vllm/model_executor/models/gemma4.py), so they allocate none.
    full_head_dim = int(text_cfg.get("global_head_dim") or 0)
    full_kv_heads = int(text_cfg.get("num_global_key_value_heads") or 0)
    kv_shared_full = kv_shared_sliding = 0
    shared = int(text_cfg.get("num_kv_shared_layers") or 0)
    types = text_cfg.get("layer_types")
    if shared and isinstance(types, list) and len(types) == int(num_layers):
        tail = _layer_type_counts(types[len(types) - shared:])
        kv_shared_full = tail["full"] + tail["hybrid"]
        kv_shared_sliding = tail["sliding"]
    elif shared:
        kv_shared_full = min(shared, layout["full"])
        kv_shared_sliding = min(shared - kv_shared_full, layout["sliding"])
    # Gemma-4 / Gemma-3n per-layer input embeddings: one table per layer,
    # plus the projection that feeds them from the hidden state.
    ple_vocab = int(text_cfg.get("vocab_size_per_layer_input") or 0)
    ple_dim = int(text_cfg.get("hidden_size_per_layer_input") or 0)
    # ...and each layer's input gate and projection (hidden x ple_dim both
    # ways), 1.3 M params a layer on E4B.
    per_layer_embed = (ple_vocab * int(num_layers) * ple_dim
                       + 3 * int(hidden) * int(num_layers) * ple_dim) if ple_dim else 0

    # Zamba2's hybrid layers call `num_mem_blocks` shared transformer blocks,
    # whose attention is wider than the Mamba stream: it reads the hidden
    # state concatenated with the embeddings, with its own head size.
    shared_attn_blocks = 0
    attn_in_dim = 0
    if family == "zamba2" and layout["attn_on_recurrent"]:
        shared_attn_blocks = int(text_cfg.get("num_mem_blocks") or 1)
        attn_in_dim = int(text_cfg.get("attention_hidden_size") or 2 * hidden)
        head_dim = int(text_cfg.get("attention_head_dim") or head_dim)

    state = _state_geometry(cfg, text_cfg, int(hidden), pure_ssm) or {}
    if layout["recurrent"] and not state:
        searched = ", ".join((
            "linear_num_value_heads/linear_key_head_dim/linear_value_head_dim/"
            "linear_conv_kernel_dim (gated delta net)",
            "kda_*/short_conv_kernel_size or linear_attn_config "
            "(Kimi delta attention)",
            "conv_L_cache (short convolution)",
            "mamba_d_state/state_size/ssm_state_size/d_state",
            "mamba_d_conv/conv_kernel/d_conv",
            "mamba_d_ssm/mamba_expand/mamba_intermediate_size/d_inner",
        ))
        sys.exit(
            f"{arch} / {family} reports {layout['recurrent']} recurrent "
            f"(state-space) layer(s) via '{layout['layout_source']}', but no "
            f"recurrent-state geometry could be read from the config. Without "
            f"it the state pool -- which does not shrink with context and "
            f"often dominates KV on hybrids -- cannot be sized, and a verdict "
            f"would be confidently wrong.\n"
            f"Keys searched: {searched}.\n"
            f"If this family names them differently, pass a local config.json "
            f"with the equivalent keys added, or file an issue with the "
            f"architecture name."
        )

    # Newer exports (Qwen3.5) write `dtype` where older ones wrote
    # `torch_dtype`.
    model_dtype = _normalize_dtype(_lookup(
        [cfg, text_cfg], ("torch_dtype", "dtype"))[0])
    # Qwen3.5 declares its state dtype as mamba_ssm_dtype, which vLLM adopts
    # when --mamba-ssm-cache-dtype is left at auto. Nemotron-H falls back to
    # float32 rather than the model dtype when its config is silent
    # (NemotronHForCausalLMConfig in vLLM), which doubles the SSM half.
    family_ssm_default = ("float32" if _FP32_SSM_STATE_FAMILIES
                          & {family, text_cfg.get("model_type")} else None)
    ssm_dtype = _normalize_dtype(
        _lookup([text_cfg, cfg], ("mamba_ssm_cache_dtype", "ssm_cache_dtype",
                                  "mamba_ssm_dtype"))[0],
        default=state.get("ssm_dtype") or family_ssm_default or model_dtype,
    )

    kv_lora_rank = _lookup([text_cfg, cfg], ("kv_lora_rank",))[0]
    mla_kv_dim = 0
    if kv_lora_rank:
        rope = _lookup([text_cfg, cfg], ("qk_rope_head_dim",))[0] or 64
        mla_kv_dim = int(kv_lora_rank) + int(rope)

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
        shared_intermediate=int(shared_intermediate),
        ffn_matrices=3 if gated else 2,
        is_vlm=is_vlm,
        vision_params=vision_params,
        vision_attn_params=vision_attn_params,
        vision_mlp_params=vision_mlp_params,
        vision_num_heads=vision_num_heads,
        per_layer_embed_params=per_layer_embed,
        full_head_dim=full_head_dim,
        full_kv_heads=full_kv_heads,
        kv_shared_full=kv_shared_full,
        kv_shared_sliding=kv_shared_sliding,
        vision_layers=vision_layers,
        vision_naming=_vision_naming(cfg) if vision_params else "",
        vision_patches_per_image=vision_patches_per_image,
        vision_tokens_per_image=vision_tokens_per_image,
        vision_intermediate=vision_intermediate,
        architecture=arch,
        num_full_attn_layers=layout["full"],
        num_sliding_attn_layers=layout["sliding"],
        num_recurrent_layers=layout["recurrent"],
        num_ffn_layers=(shared_attn_blocks if shared_attn_blocks
                        else layout["ffn"]),
        sliding_window=window,
        window_is_chunked=window_is_chunked,
        shared_attn_blocks=shared_attn_blocks,
        attn_in_dim=attn_in_dim,
        state_kind=state.get("kind", ""),
        num_state_heads=state.get("num_state_heads", 0),
        state_head_dim_k=state.get("state_head_dim_k", 0),
        state_head_dim_v=state.get("state_head_dim_v", 0),
        conv_dim=state.get("conv_dim", 0),
        conv_kernel=state.get("conv_kernel", 0),
        state_groups=state.get("state_groups", 1),
        state_inner=state.get("state_inner", 0),
        state_dt_rank=state.get("dt_rank", 0),
        conv_dtype=model_dtype,
        ssm_dtype=ssm_dtype,
        attn_on_recurrent_layers=layout["attn_on_recurrent"],
        state_heads_divide_tp=state.get("divides_tp", False),
        mla_kv_dim=mla_kv_dim,
        state_source=state.get("state_source", []),
    )


def _recurrent_block_params(d: ModelDims) -> int:
    """Weights of one recurrent (Mamba / Mamba2 / gated-delta-net) layer.

    Mamba2 and gated-delta-net differ in how they name their projections but
    land on the same shapes: one input projection wide enough to feed the
    conv state plus the value stream, a depthwise conv, and one output
    projection back to hidden. The per-head scalars (A_log, D, dt_bias) and
    the gated output norm are small but cheap to include.
    """
    h = d.hidden
    if d.state_kind == "short_conv":
        # LFM2: in_proj to B, C and x, a depthwise conv, out_proj.
        return h * 3 * d.conv_dim + d.conv_dim * d.conv_kernel + d.conv_dim * h
    if not d.num_state_heads:
        return 0
    in_proj = h * (d.conv_dim + d.state_inner + 2 * d.num_state_heads)
    conv = d.conv_dim * d.conv_kernel + d.conv_dim
    out_proj = d.state_inner * h
    scalars = 3 * d.num_state_heads
    gate_norm = d.state_head_dim_v * d.num_state_heads
    if d.state_dt_rank:
        # Mamba1 projects dt, B and C from the inner stream (x_proj), expands
        # dt back (dt_proj), and keeps A and D per channel rather than per
        # head -- ~4% of falcon-mamba-7b that the shared shapes above miss.
        inner, rank = d.state_inner, d.state_dt_rank
        x_proj = inner * (rank + 2 * d.state_head_dim_k)
        dt_proj = rank * inner + inner
        scalars = inner * d.state_head_dim_k + inner
        gate_norm = 0
        return in_proj + conv + out_proj + x_proj + dt_proj + scalars
    return in_proj + conv + out_proj + scalars + gate_norm


def _ffn_param_split(d: ModelDims) -> tuple[int, int, int]:
    """(dense-FFN, shared-expert, routed-expert) parameter counts.

    Kept separate because quantization schemes routinely treat them
    differently: a compressed-tensors export commonly quantizes the routed
    experts -- where nearly all the weight is -- while leaving the dense
    replacement layers and the shared expert at full width. Charging one rate
    across all three is wrong by a factor, not a margin.
    """
    h = d.hidden
    mats = d.ffn_matrices
    ffn_layers = _ffn_layers(d)
    if not d.is_moe:
        return ffn_layers * mats * h * d.intermediate, 0, 0

    routed_per_layer = d.num_experts * mats * h * d.intermediate
    shared_per_layer = (d.num_shared_experts * mats * h
                        * (d.shared_intermediate or d.intermediate))

    dense_layers = 0
    if d.first_k_dense_replace > 0 and d.dense_intermediate > 0:
        dense_layers = min(d.first_k_dense_replace, ffn_layers)
    moe_layers = ffn_layers - dense_layers
    dense = dense_layers * mats * h * d.dense_intermediate
    return dense, moe_layers * shared_per_layer, moe_layers * routed_per_layer


def _attn_proj_params(d: ModelDims) -> int:
    """Q/K/V/O projection parameter count for one attention layer.

    For models with explicit head_dim != hidden/num_heads, Q projects to
    num_attn_heads * head_dim (not hidden).
    """
    h = d.hidden
    in_dim = d.attn_in_dim or h
    q_proj_dim = d.num_attn_heads * d.head_dim
    kv_proj_dim = d.num_kv_heads * d.head_dim
    return (
        in_dim * q_proj_dim     # Q: input → num_attn_heads * head_dim
        + in_dim * kv_proj_dim  # K: input → num_kv_heads * head_dim
        + in_dim * kv_proj_dim  # V: input → num_kv_heads * head_dim
        + q_proj_dim * h        # O: num_attn_heads * head_dim → hidden
    )


def _attn_proj_params_total(d: ModelDims) -> int:
    """Q/K/V/O parameters over every attention layer that owns weights.

    Gemma-4's full-attention layers project to the wider global head
    (global_head_dim 512 against 256: Q/O twice as wide) and may carry their
    own KV head count, so the two kinds are priced separately when they
    differ. Everywhere else this is layers x _attn_proj_params.
    """
    layers = _attn_weight_layers(d)
    if not d.full_head_dim or d.shared_attn_blocks:
        return layers * _attn_proj_params(d)
    full = min(d.num_full_attn_layers, layers)
    wide = _attn_proj_params(replace(d, head_dim=d.full_head_dim,
                                     num_kv_heads=d.full_kv_heads or d.num_kv_heads))
    return full * wide + (layers - full) * _attn_proj_params(d)


def _attn_weight_layers(d: ModelDims) -> int:
    """Attention blocks that exist as weights.

    Equal to the attention layer count, except where layers share blocks:
    Zamba2's hybrid layers each keep a KV cache but call one of
    `num_mem_blocks` shared attention + MLP blocks, so pricing a block per
    hybrid layer would charge 13 copies of weights that exist twice.
    """
    return d.shared_attn_blocks or _attention_layers(d)


def _attention_layers(d: ModelDims) -> int:
    """Layer count to charge the attention-block formula over.

    Hybrids replace attention with an SSM on some layers, and families like
    Nemotron-H also have layers that carry neither. For a plain model all
    three counts collapse to num_layers.
    """
    attn = d.num_full_attn_layers + d.num_sliding_attn_layers
    if attn or d.num_recurrent_layers:
        return attn
    # Directly constructed dimensions that never classified their layers.
    return d.num_layers


def _ffn_layers(d: ModelDims) -> int:
    """Layer count carrying a feed-forward block; 0 for a pure SSM."""
    if d.num_ffn_layers or d.num_recurrent_layers:
        return d.num_ffn_layers
    return d.num_layers


def count_params(d: ModelDims) -> int:
    h = d.hidden
    attn_layers = _attn_weight_layers(d)
    ffn_layers = _ffn_layers(d)

    # Q/K/V/O over the attention layers, plus input + post-attention norms.
    total = _attn_proj_params_total(d) + attn_layers * 2 * h
    total += d.num_recurrent_layers * (_recurrent_block_params(d) + 2 * h)
    if d.shared_attn_blocks:
        # Each hybrid layer projects the shared block's output back with its
        # own hidden x hidden linear.
        total += d.num_full_attn_layers * h * h

    # Dense replacement layers (DeepSeek-V2/V3/V4's first_k_dense_replace),
    # shared experts, and routed experts, each counted over the layers that
    # actually carry them.
    total += sum(_ffn_param_split(d))
    total += ffn_layers * 2 * h   # pre-FFN norm + residual norm
    total += router_params(d)

    emb = d.vocab * h + d.per_layer_embed_params
    head = 0 if d.tied else d.vocab * h
    return emb + head + total


def default_max_num_batched_tokens(device_vram_gb: float) -> int:
    """vLLM's default --max-num-batched-tokens for an OpenAI API server.

    vLLM picks this from a device-memory tier table; only parts at or above
    70 GiB get the 8192 tier. It matters here because it sets how many
    uncommitted tokens a sliding-window block pool must be able to hold.
    """
    if device_vram_gb >= VLLM_LARGE_DEVICE_GIB:
        return VLLM_MAX_NUM_BATCHED_TOKENS_LARGE
    return VLLM_MAX_NUM_BATCHED_TOKENS_SMALL


def max_in_flight_tokens(device_vram_gb: float) -> int:
    """Tokens vLLM must keep addressable beyond a sliding window.

    A windowed layer can only drop a block once every token in it has left
    the window, and vLLM sizes that slack at two scheduler batches.
    """
    return 2 * default_max_num_batched_tokens(device_vram_gb)


def sliding_ctx(d: ModelDims, ctx: int, runtime: str,
                in_flight_tokens: int) -> int:
    """Tokens a sliding-window layer must hold per request.

    The three runtimes charge visibly different amounts for the same window:

    * vllm pages the window and cannot free a block until every token in it
      has left the window, so it holds `window - 1` plus the tokens still in
      flight across the current batches. A chunked-local layer (Llama-4)
      holds a whole chunk plus the in-flight tokens.
    * sglang sizes the window pool as a fraction of the full-attention pool
      (--swa-full-tokens-ratio) instead of to the window, so the radix
      prefix cache still has blocks to reuse -- and only for the
      architectures it runs as hybrid SWA. Everything else gets a full
      pool, window or not.
    * torch / transformers allocates exactly the window.
    """
    window = d.sliding_window
    if not window:
        return ctx
    if runtime == "vllm":
        held = window if d.window_is_chunked else window - 1
        return min(ctx, held + in_flight_tokens)
    if runtime == "sglang":
        if d.architecture not in SGLANG_HYBRID_SWA_ARCHS:
            return ctx
        pool = int(SGLANG_SWA_FULL_TOKENS_RATIO * ctx)
        return min(ctx, max(window, pool))
    return min(ctx, window)


def _kv_elements(d: ModelDims, full: bool) -> int:
    """KV elements one token stores in one layer of the given kind."""
    if d.mla_kv_dim:
        return d.mla_kv_dim
    heads, dim = d.num_kv_heads, d.head_dim
    if full and d.full_head_dim:
        heads, dim = (d.full_kv_heads or heads), d.full_head_dim
    return 2 * heads * dim


def _kv_layer_counts(d: ModelDims) -> tuple[int, int]:
    """(full, sliding) layers that allocate a KV cache of their own."""
    full_layers = d.num_full_attn_layers
    sliding_layers = d.num_sliding_attn_layers
    if not (full_layers or sliding_layers):
        # Pre-layout configs and direct ModelDims construction in callers.
        full_layers = d.num_layers if not d.num_recurrent_layers else 0
    return (max(full_layers - d.kv_shared_full, 0),
            max(sliding_layers - d.kv_shared_sliding, 0))


def kv_bytes(d: ModelDims, ctx: int, concurrency: int, kv_dtype: str,
             runtime: str = "vllm", in_flight_tokens: int = 0,
             block_tokens: int = 0) -> int:
    """KV-cache bytes across every attention layer, before the TP divide.

    Charged per layer rather than per model: a sliding-window layer holds a
    bounded number of tokens no matter how long the context is, a recurrent
    layer holds no KV cache at all, a KV-shared layer (Gemma-4) reuses
    another layer's, and a full-attention layer may use a wider head than
    the windowed ones (Gemma-4's global_head_dim).

    `block_tokens` rounds each request up to whole blocks. It only matters
    where the block is large -- vLLM grows a hybrid model's attention block
    to hundreds of tokens (vllm_block_tokens) -- so callers pass 0 otherwise.
    """
    bytes_per = BYTES_PER_KV[kv_dtype]
    full_layers, sliding_layers = _kv_layer_counts(d)
    full_ctx = ctx
    windowed = sliding_ctx(d, ctx, runtime, in_flight_tokens) if sliding_layers else 0
    if block_tokens:
        full_ctx = _ceil_div(ctx, block_tokens) * block_tokens
        if sliding_layers and windowed < ctx:
            # A window rarely starts on a block boundary: one block more.
            windowed = _ceil_div(windowed, block_tokens) * block_tokens + block_tokens
        windowed = min(windowed, full_ctx)
    elements = (full_layers * full_ctx * _kv_elements(d, True)
                + sliding_layers * windowed * _kv_elements(d, False))
    return int(elements * bytes_per * concurrency)


def kv_shards(d: ModelDims, tp: int) -> int:
    """How many ways the KV cache actually splits across `tp` ranks.

    Each rank holds max(1, kv_heads // tp) KV heads: with fewer KV heads than
    ranks vLLM replicates them, so the split stops at kv_heads. MLA's latent
    is not split at all.
    """
    if tp <= 1 or d.mla_kv_dim:
        return 1
    kv = max(int(d.num_kv_heads or 1), 1)
    return max(1, kv // max(1, kv // tp))


def tp_problems(d: ModelDims, tp: int) -> list[str]:
    """Reasons vLLM would refuse to start this model at `tp`, or []."""
    if tp <= 1:
        return []
    problems = []
    if d.num_attn_heads % tp:
        problems.append(f"{d.num_attn_heads} attention heads do not divide "
                        f"evenly across TP={tp}")
    kv = d.num_kv_heads
    if not d.mla_kv_dim and kv and (kv % tp if kv >= tp else tp % kv):
        problems.append(f"{kv} KV heads neither divide TP={tp} nor are "
                        f"divisible by it")
    # Column-parallel MLP splits the intermediate dimension, so TP must
    # divide every width the model actually uses (see ffn_shard_widths()).
    for width in ffn_shard_widths(d):
        if width % tp:
            problems.append(f"FFN width {width} does not divide evenly "
                            f"across TP={tp}")
            break
    if d.num_recurrent_layers and d.state_heads_divide_tp:
        for count, what in ((d.num_state_heads, "linear-attention heads"),
                            (d.state_groups, "linear-attention key heads")):
            if count % tp:
                problems.append(f"{count} {what} do not divide evenly "
                                f"across TP={tp}")
                break
    if d.num_recurrent_layers and d.state_kind == "mamba2":
        # MambaStateShapeCalculator pads an awkward group count, but the
        # layer refuses to build: MambaMixer2.__init__ asserts
        # `num_heads % tp == 0` and `n_groups % tp == 0 or n_groups == 1`
        # (vllm/model_executor/layers/mamba/mamba_mixer2.py). So 8 groups at
        # TP=16 is not a padded launch, it is no launch.
        if d.num_state_heads % tp:
            problems.append(f"{d.num_state_heads} Mamba2 heads do not divide "
                            f"evenly across TP={tp}")
        elif d.state_groups % tp and d.state_groups != 1:
            problems.append(f"{d.state_groups} Mamba2 groups neither divide "
                            f"TP={tp} nor are a single group")
    if (d.num_recurrent_layers and d.state_kind in ("mamba1", "short_conv")
            and d.conv_dim % tp):
        problems.append(f"recurrent width {d.conv_dim} does not divide evenly "
                        f"across TP={tp}")
    return problems


def _ceil_div(a: int, b: int) -> int:
    return -(-a // max(b, 1))


def _state_layer_elements(d: ModelDims, tp: int) -> tuple[int, int]:
    """(conv, ssm) element counts one recurrent layer holds on one rank.

    Each kind follows vLLM's MambaStateShapeCalculator. The conv state keeps
    `conv_kernel - 1` columns: the current token is computed, not stored.
    """
    tp = max(tp, 1)
    columns = max(d.conv_kernel - 1, 1)
    if d.state_kind == "mamba2":
        # Heads and groups shard together. A single group that cannot split
        # is replicated to every rank, so only the B/C part of the conv
        # grows -- the x part still divides.
        groups = d.state_groups or 1
        if groups % tp:
            groups = tp
        conv_dim = d.state_inner + 2 * groups * d.state_head_dim_k
        conv = _ceil_div(conv_dim, tp) * columns
        ssm = (_ceil_div(d.num_state_heads, tp)
               * d.state_head_dim_v * d.state_head_dim_k)
        return conv, ssm
    if d.state_kind == "mamba1":
        # The inner width divides for both halves: (inner/tp, k-1) and
        # (inner/tp, d_state).
        inner = _ceil_div(d.state_inner, tp)
        return inner * columns, inner * d.state_head_dim_k
    if d.state_kind == "short_conv":
        return _ceil_div(d.conv_dim, tp) * columns, 0
    # Gated delta net and KDA (and directly constructed dims): the conv
    # width and the heads divide; vLLM refuses a TP that splits a head,
    # which tp_problems() reports, so the ceiling only matters there.
    conv = _ceil_div(d.conv_dim, tp) * columns
    ssm = (_ceil_div(d.num_state_heads, tp)
           * d.state_head_dim_k * d.state_head_dim_v)
    return conv, ssm


def _state_layer_bytes(d: ModelDims, tp: int, runtime: str) -> int:
    """Unpadded state bytes of one recurrent layer, one request, one rank."""
    conv, ssm = _state_layer_elements(d, tp)
    ssm_dtype = "float32" if runtime == "sglang" else d.ssm_dtype
    return int(conv * BYTES_PER_STATE_DTYPE[d.conv_dtype]
               + ssm * BYTES_PER_STATE_DTYPE[ssm_dtype])


def state_page_bytes(d: ModelDims, tp: int, runtime: str = "vllm") -> int:
    """Recurrent-state bytes of one copy of one request's state, per rank.

    Summed over the recurrent layers. Every engine allocates at least this
    per running request and never shrinks it with context -- an SSM layer's
    state is the same size for 1 token as for 1M. What a runtime actually
    reserves can be more; see state_reserved_bytes().
    """
    if not (d.num_recurrent_layers and (d.num_state_heads or d.conv_dim)):
        return 0
    return _state_layer_bytes(d, tp, runtime) * d.num_recurrent_layers


def _attn_bytes_per_token_per_layer(d: ModelDims, tp: int, kv_dtype: str) -> int:
    """One attention layer's KV bytes per token on one rank."""
    if d.mla_kv_dim:
        return int(d.mla_kv_dim * BYTES_PER_KV[kv_dtype])
    kv_per_rank = max(1, int(d.num_kv_heads or 1) // max(tp, 1))
    return int(2 * kv_per_rank * d.head_dim * BYTES_PER_KV[kv_dtype])


def vllm_block_tokens(d: ModelDims, tp: int, kv_dtype: str) -> int:
    """vLLM's attention block size for a hybrid model, or 0 if not hybrid.

    vLLM grows the block, in steps of the kernel alignment, until one block
    of a layer's KV is at least one recurrent-state page -- 576 tokens on
    Qwen3.5-4B where the launch asked for 64.
    """
    if not (d.num_recurrent_layers and _attention_layers(d)):
        return 0
    page = _state_layer_bytes(d, tp, "vllm")
    per_token = _attn_bytes_per_token_per_layer(d, tp, kv_dtype)
    if not (page and per_token):
        return 0
    align = VLLM_HYBRID_BLOCK_ALIGN
    return align * _ceil_div(page, align * per_token)


def state_reserved_bytes(d: ModelDims, tp: int, runtime: str = "vllm",
                         kv_dtype: str = "bf16") -> int:
    """Recurrent-state bytes a runtime reserves per running request, per rank.

    vLLM keeps the state in the same block pool as the attention KV of a
    hybrid model, so each state page is padded up to one attention block
    (vllm_block_tokens), and its default prefix caching runs the mamba cache
    in "align" mode, which holds two pages per recurrent layer per request.
    Measured on Qwen3.5-4B: 108 MiB per request against a 49 MiB state. An
    attention-free model has prefix caching off and holds one unpadded page.
    SGLang and transformers hold one copy.
    """
    raw = state_page_bytes(d, tp, runtime)
    if not raw or runtime != "vllm":
        return raw
    block = vllm_block_tokens(d, tp, kv_dtype)
    if not block:
        return raw
    padded = block * _attn_bytes_per_token_per_layer(d, tp, kv_dtype)
    return padded * VLLM_ALIGN_MODE_STATE_PAGES * d.num_recurrent_layers


def sglang_pool_split_bytes(d: ModelDims, kv: int, state: int,
                            runtime: str) -> int:
    """Memory SGLang's fixed hybrid pool split strands, beyond kv + state.

    The state pool is ratio / (1 + ratio) of the memory left after weights
    and the KV pool the rest, and neither can borrow from the other. Both
    halves have to hold their share, so the memory needed is whichever half
    binds, scaled up to the whole: the gap over kv + state is real
    allocation, not slack.
    """
    if runtime != "sglang" or not (state and _attention_layers(d)):
        return 0
    r = SGLANG_MAMBA_FULL_MEMORY_RATIO
    need = max(kv * (1 + r), state * (1 + r) / r)
    return max(int(need) - kv - state, 0)


def ffn_shard_widths(d: ModelDims) -> tuple[int, ...]:
    """FFN widths a runtime partitions, and only the ones a layer really uses.

    `d.intermediate` is the width in force (moe_intermediate_size on an MoE,
    intermediate_size on a dense model). `d.dense_intermediate` only describes
    real layers on a *hybrid* MoE -- parse_dims fills it from intermediate_size
    for every MoE config, so checking it unconditionally would reject a TP over
    a width no layer of a pure MoE has.
    """
    if not _ffn_layers(d):
        return ()
    widths = [d.intermediate]
    if d.is_moe and d.first_k_dense_replace > 0 and d.dense_intermediate:
        widths.append(d.dense_intermediate)
    return tuple(w for w in widths if w)


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
    """Activation estimate for the runtimes without a profiling model.

    ~2 hidden buffers' worth per token in flight + 512 MiB scratch. vLLM does
    not use this: it reserves what its startup profiling pass measured,
    which does not depend on ctx or concurrency (vllm_reservation).
    """
    bytes_per = BYTES_PER_PARAM[dtype]
    return int(2 * concurrency * ctx * d.hidden * bytes_per) + 512 * MB


def vision_profile_patches(d: ModelDims, batched_tokens: int,
                           ctx: int = 0) -> int:
    """Patches vLLM's encoder profiling pass runs through the vision tower.

    A fixed-size tower is profiled with as many images as one batch of
    tokens holds -- or, when the family also takes video (InternVL), with one
    video of as many frames as fit the context: 31 frames of 256 tokens at
    `--max-model-len 8192` on InternVL3-2B, 7,936 encoder tokens, measured.
    """
    if not (d.vision_params and d.vision_patches_per_image):
        return 0
    if not d.vision_tokens_per_image:
        return d.vision_patches_per_image
    budget = max(batched_tokens, d.vision_tokens_per_image)
    if d.vision_naming == "intern" and ctx:
        budget = max(budget, ctx - d.vision_tokens_per_image)
    return max(1, budget // d.vision_tokens_per_image) * d.vision_patches_per_image


def vllm_reservation(d: ModelDims, quant: str, device_vram_gb: float,
                     batched_tokens: int | None = None,
                     ctx: int = 0) -> tuple[int, int, int]:
    """(runtime floor, held encoder buffers, profile peak) vLLM reserves.

    vLLM sizes its KV cache once at startup: gpu_memory_utilization x the
    driver's total, minus what a profiling pass consumed. That pass runs one
    batch of max_num_batched_tokens -- and, for a VLM, a maximum-size image --
    so the reservation is fixed by the model and the device, never by the
    context or concurrency a launch later serves. Charging activations per
    ctx x concurrency instead understated it for VLMs by ~3 GiB and
    overstated it at long contexts by several. Calibrated on an Arc Pro B70
    against 13 models; conservative by 0.03-0.74 GiB, never optimistic.
    Peaks are not divided across TP ranks (unmeasured; the safe side).
    """
    batched = batched_tokens or default_max_num_batched_tokens(device_vram_gb)
    floor = VLLM_RUNTIME_FLOOR_GB
    if BYTES_PER_PARAM.get(quant, 2.0) < 2.0:
        floor += VLLM_QUANT_WORKSPACE_GB
    dense_width = (d.dense_intermediate if d.is_moe else d.intermediate) or 0
    dense = batched * VLLM_DENSE_PEAK_WIDTHS * max(dense_width, d.conv_dim) * 2
    moe = (batched * d.num_experts_per_tok * VLLM_MOE_PEAK_WIDTHS
           * d.intermediate * 2 if d.is_moe else 0)
    patches = vision_profile_patches(d, batched, ctx)
    vision = int(patches * VLLM_VISION_PEAK_WIDTHS * d.vision_intermediate * 2)
    held = patches * VLLM_VISION_HELD_BYTES_PER_PATCH
    return int(floor * GB), int(held), int(max(dense, moe, vision))


def usable_bytes(device_vram_gb: float, gpu_memory_utilization: float,
                 runtime: str) -> int:
    """Memory the runtime may allocate. vLLM's share is of the driver total,
    which sits ~5% under physical; the other runtimes keep their floors."""
    device_b = device_vram_gb * GB
    if runtime == "vllm":
        device_b *= XPU_ALLOCATABLE_FRACTION
    return int(device_b * gpu_memory_utilization)


def fmt_gb(b: int) -> str:
    return f"{b / GB:6.2f} GB"


# Every key a quantization_config can use to list modules left at full
# precision. Exporters disagree on the name but agree on the meaning, so the
# union is read rather than picking one.
_EXCLUSION_KEYS = (
    "modules_to_not_convert",   # gpt-oss (mxfp4), bitsandbytes 4-bit, some GPTQ
    "ignore",                   # compressed-tensors
    "ignored_layers",           # TensorRT Model Optimizer
    "exclude_modules",          # AutoRound
    "exclude",                  # some AWQ exports
    "llm_int8_skip_modules",    # bitsandbytes int8
)

# Module names have to be reconstructed to ask "would this component be
# quantized?", because the config never lists the modules a scheme *does*
# convert. A name is a layer root, a layer index and a member tail, and every
# part has several spellings in use: the same attention projection is
# `model.layers.3.self_attn.q_proj` in a text checkpoint,
# `language_model.model.layers.3.self_attn.q_proj` in a Llama-4 export and
# `model.language_model.layers.3...` in newer VLMs. An entry is matched
# against every spelling, so one this misses cannot silently price a
# full-width tensor at the quantized rate.
_TEXT_LAYER_ROOTS = (
    "model.layers", "language_model.model.layers",
    "model.language_model.layers", "language_model.layers",
    "backbone.layers", "layers",
)
# Vision towers name their blocks per family, and each family pairs its own
# root with its own member names -- a Qwen tower has no `self_attn`, a CLIP
# tower no `attn.qkv`. Mixing them would let an attention-only list like
# ["self_attn"] reach a Qwen2-VL tower it does not touch. Each entry is
# (roots, attention tails, MLP tails).
_VISION_NAMINGS = {
    "qwen": (("visual.blocks", "model.visual.blocks"),
             ("attn.qkv", "attn.proj"),
             ("mlp.fc1", "mlp.fc2", "mlp.gate_proj", "mlp.up_proj",
              "mlp.down_proj", "mlp.linear_fc1", "mlp.linear_fc2")),
    "llama4": (("vision_model.model.layers",),
               ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj"),
               ("mlp.fc1", "mlp.fc2")),
    "pixtral": (("vision_tower.transformer.layers",
                 "model.vision_tower.transformer.layers"),
                ("attention.q_proj", "attention.k_proj", "attention.v_proj",
                 "attention.o_proj"),
                ("feed_forward.gate_proj", "feed_forward.up_proj",
                 "feed_forward.down_proj")),
    "clip": (("vision_tower.vision_model.encoder.layers",
              "model.vision_tower.vision_model.encoder.layers",
              "vision_model.encoder.layers", "vision_tower.encoder.layers"),
             ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.out_proj", "self_attn.o_proj"),
             ("mlp.fc1", "mlp.fc2")),
    "intern": (("vision_model.encoder.layers", "vision_tower.encoder.layer",
                "model.vision_tower.encoder.layer"),
               ("attn.qkv", "attn.proj", "attention.q_proj", "attention.k_proj",
                "attention.v_proj", "attention.projection_layer"),
               ("mlp.fc1", "mlp.fc2")),
}


def _vision_naming(cfg: dict) -> str:
    """Which _VISION_NAMINGS family a VLM's tower uses, or "" if unknown."""
    family = str(cfg.get("model_type", ""))
    vision = cfg.get("vision_config") or {}
    vtype = str(vision.get("model_type", "")) if isinstance(vision, dict) else ""
    if family.startswith("qwen") or vtype.startswith("qwen"):
        return "qwen"
    if family.startswith("llama4"):
        return "llama4"
    if vtype == "pixtral":
        return "pixtral"
    if vtype in ("siglip_vision_model", "siglip2_vision_model",
                 "clip_vision_model"):
        return "clip"
    if family.startswith("intern") or vtype.startswith("intern"):
        return "intern"
    return ""


# Tower modules that live outside the blocks: patch and position embeddings,
# the merger / projector into the language model, and Gemma-4's audio tower.
# They carry no layer index, so they are matched as whole names.
_VISION_EXTRA_NAMES = tuple(
    f"{root}.{tail}" for root in ("visual", "model.visual", "vision_tower",
                                  "model.vision_tower", "vision_model")
    for tail in ("merger", "merger.linear_fc1", "merger.linear_fc2",
                 "merger.mlp.0", "merger.mlp.2", "deepstack_merger_list.0",
                 "patch_embed.proj", "patch_embedder.input_proj", "pos_embed",
                 "patch_embedder.position_embedding_table")
) + ("multi_modal_projector.linear_1", "multi_modal_projector.linear_2",
     "model.multi_modal_projector.linear_1", "mlp1.1", "mlp1.3",
     "model.embed_vision.embedding_projection", "model.embed_audio.embedding_projection",
     "model.audio_tower.layers.0.attention.q_proj", "model.audio_tower.output_proj")


def _vision_namings(component: str) -> list[tuple]:
    """The namings a `vision` or `vision@<family>` component spans."""
    key = component.partition("@")[2]
    return ([_VISION_NAMINGS[key]] if key in _VISION_NAMINGS
            else list(_VISION_NAMINGS.values()))


_FFN_SUFFIXES = {"gate": ("gate_proj", "w1"), "up": ("up_proj", "w3"),
                 "down": ("down_proj", "w2")}


def _ffn_tails(*prefixes: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (member, tuple(f"{p}.{s}" for p in prefixes for s in suffixes))
        for member, suffixes in _FFN_SUFFIXES.items())


# Component -> ((member, tails), ...). A member is one weight matrix of a
# layer; its tails are the module paths, under a layer root, that exporters
# use for it. The three FFN kinds are separate components: a checkpoint that
# quantizes its routed experts and exempts the dense and shared ones is the
# common case, not the exception, and nearly all the weight is in the
# experts.
_COMPONENT_MEMBERS = {
    "attn": (
        ("q", ("self_attn.q_proj", "self_attn.qkv_proj", "self_attn.q_a_proj",
               "self_attn.q_b_proj", "self_attn.query_key_value",
               "attention.wq", "attention.wqkv", "attention.q_proj",
               "mixer.q_proj", "attn.q_proj")),
        ("k", ("self_attn.k_proj", "self_attn.qkv_proj",
               "self_attn.kv_a_proj_with_mqa", "self_attn.query_key_value",
               "attention.wk", "attention.wqkv", "attention.k_proj",
               "mixer.k_proj", "attn.k_proj")),
        ("v", ("self_attn.v_proj", "self_attn.qkv_proj", "self_attn.kv_b_proj",
               "self_attn.query_key_value", "attention.wv", "attention.wqkv",
               "attention.v_proj", "mixer.v_proj", "attn.v_proj")),
        ("o", ("self_attn.o_proj", "self_attn.dense", "self_attn.out_proj",
               "attention.wo", "attention.o_proj", "attention.dense",
               "mixer.o_proj", "attn.o_proj")),
    ),
    "state": (
        ("in", ("linear_attn.in_proj_qkvz", "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_ba", "linear_attn.in_proj_z",
                "mamba.in_proj", "mixer.in_proj", "conv.in_proj")),
        ("out", ("linear_attn.out_proj", "mamba.out_proj", "mixer.out_proj",
                 "conv.out_proj")),
    ),
    "router": (
        ("gate", ("mlp.gate", "mlp.router", "block_sparse_moe.gate",
                  "feed_forward.router", "feed_forward.gate", "mlp.gate.wg",
                  "mixer.gate")),
    ),
    "dense_ffn": (
        ("gate", ("mlp.gate_proj", "mlp.gate_up_proj", "feed_forward.gate_proj",
                  "feed_forward.w1", "mlp.w1", "mlp.fc1", "mixer.up_proj")),
        ("up", ("mlp.up_proj", "mlp.gate_up_proj", "feed_forward.up_proj",
                "feed_forward.w3", "mlp.w3", "mlp.fc1", "mixer.up_proj")),
        ("down", ("mlp.down_proj", "feed_forward.down_proj", "feed_forward.w2",
                  "mlp.w2", "mlp.fc2", "mixer.down_proj")),
    ),
    "shared_ffn": _ffn_tails(
        "mlp.shared_experts", "mlp.shared_expert", "mlp.shared_mlp",
        "feed_forward.shared_expert", "mixer.shared_experts",
        "block_sparse_moe.shared_experts"),
    "expert_ffn": tuple(
        (member, tails + fused) for (member, tails), fused in zip(
            _ffn_tails("mlp.experts.0", "block_sparse_moe.experts.0",
                       "feed_forward.experts.0", "mixer.experts.0"),
            (("mlp.experts.gate_up_proj", "feed_forward.experts.gate_up_proj"),
             ("mlp.experts.gate_up_proj", "feed_forward.experts.gate_up_proj"),
             ("mlp.experts.down_proj", "feed_forward.experts.down_proj")))),
}
_LM_HEAD_NAMES = ("lm_head", "language_model.lm_head", "model.lm_head")


def _member_tails(component: str, member: str) -> tuple[str, ...]:
    """Every tail one member of a component goes by."""
    if component.startswith("vision"):
        if member == "extra":
            return _VISION_EXTRA_NAMES
        part = 1 if member == "attn" else 2
        return tuple(t for naming in _vision_namings(component)
                     for t in naming[part])
    return dict(_COMPONENT_MEMBERS[component])[member]


def _component_members(component: str) -> tuple[str, ...]:
    if component == "lm_head":
        return ("lm_head",)
    if component.startswith("vision"):
        return ("attn", "mlp", "extra")
    return tuple(m for m, _ in _COMPONENT_MEMBERS[component])


def _component_names(component: str, member: str | None = None,
                     layer: int = 0) -> list[str]:
    """Every spelling of a component's (or one member's) modules at `layer`."""
    if component == "lm_head":
        return list(_LM_HEAD_NAMES)
    if component.startswith("vision"):
        names = [f"{root}.{layer}.{tail}"
                 for roots, attn, mlp in _vision_namings(component)
                 for name, tails in (("attn", attn), ("mlp", mlp))
                 if member is None or name == member
                 for tail in tails for root in roots]
        if member in (None, "extra"):
            names += list(_VISION_EXTRA_NAMES)
        return names
    return [f"{root}.{layer}.{tail}"
            for name, tails in _COMPONENT_MEMBERS[component]
            if member is None or name == member
            for tail in tails for root in _TEXT_LAYER_ROOTS]


def _segments_match(entry: str, name: str) -> bool:
    """A plain entry against a module name, on dotted-path boundaries.

    vLLM matches a plain entry against the real module path -- exactly for
    compressed-tensors, by substring for AWQ. The names here are
    reconstructions spanning several families, so a raw substring test lets
    one family's entry hit another's naming: Mixtral AWQ lists `gate` for
    its `block_sparse_moe.gate` router, and a substring test finds `gate`
    inside `gate_up_proj` and prices all 45 B expert params at bf16. So the
    entry has to line up with whole path segments, except that its first
    segment may close an underscore-joined one (`attn` covers `self_attn`).
    """
    want, have = entry.split("."), name.split(".")
    for start in range(len(have) - len(want) + 1):
        first = have[start]
        if not (first == want[0] or first.endswith("_" + want[0])):
            continue
        if have[start + 1:start + len(want)] == want[1:]:
            return True
    return False


def _pattern_matches(pattern: str, name: str) -> bool:
    """Match one exclusion entry against a module name.

    Three spellings are in use and all three have to work, because a
    checkpoint that names attention in a form this misses gets charged the
    quantized rate for tensors that are stored at full width:

    * `re:` prefix -- compressed-tensors. The rest is a regular expression
      applied with `re.match`, so it is anchored at the start of the name.
    * shell globs -- `model.layers.*.self_attn`, common in
      modules_to_not_convert. A pattern naming a parent also covers its
      children, so the glob is retried with a `.*` suffix.
    * anything else -- a module path, matched on segment boundaries
      (_segments_match).
    """
    if pattern.startswith("re:"):
        try:
            return re.match(pattern[3:], name) is not None
        except re.error:
            return False
    if "*" in pattern or "?" in pattern or "[" in pattern:
        return (fnmatch.fnmatchcase(name, pattern)
                or fnmatch.fnmatchcase(name, pattern.rstrip(".") + ".*"))
    return _segments_match(pattern, name)


def exclusion_patterns(qcfg: dict) -> list[str]:
    """Every module pattern a quantization_config leaves at full precision.

    Besides the list keys, AutoRound's per-module `extra_config` can pin a
    module at 16 bits and GPTQ's `dynamic` can skip one with a `-:` regex;
    both mean the same thing as an exclusion entry.
    """
    patterns: list[str] = []
    for key in _EXCLUSION_KEYS:
        value = qcfg.get(key)
        if isinstance(value, str):
            patterns.append(value)
        elif isinstance(value, (list, tuple)):
            patterns.extend(str(entry) for entry in value)
    for name, bits in _per_module_bits(qcfg):
        if bits >= 16:
            patterns.append(name)
    return patterns


def _per_module_bits(qcfg: dict) -> list[tuple[str, int]]:
    """(pattern, bits) overrides from AutoRound extra_config / GPTQ dynamic.

    `dynamic` keys are regexes: `-:` skips quantization (16 bits here), `+:`
    or no prefix overrides the width. Entries with no usable bits are left
    out rather than guessed.
    """
    out: list[tuple[str, int]] = []
    extra = qcfg.get("extra_config")
    if isinstance(extra, dict):
        for name, spec in extra.items():
            bits = spec.get("bits") if isinstance(spec, dict) else None
            if isinstance(bits, int):
                out.append((str(name), bits))
    dynamic = qcfg.get("dynamic")
    if isinstance(dynamic, dict):
        for key, spec in dynamic.items():
            key = str(key)
            if key.startswith("-:"):
                out.append(("re:" + key[2:], 16))
                continue
            bits = spec.get("bits") if isinstance(spec, dict) else None
            if isinstance(bits, int):
                out.append(("re:" + key.removeprefix("+:"), bits))
    return out


_INDEXED_NAME = re.compile(r"^(?P<root>.+?\.(?:layers|blocks|h))\.(?P<idx>\d+)"
                           r"\.(?P<tail>.+)$")


def _member_coverage(patterns: tuple[str, ...], component: str, member: str,
                     num_layers: int) -> float:
    """Fraction of a member's layers, 0..1, that `patterns` cover.

    An entry that names one layer covers that layer only: an exclusion list
    naming `model.layers.0.mlp.down_proj` keeps one layer of 28 wide, not
    all of them. Fully enumerated lists (Llama-4, Mistral-Small) cover every
    layer one name at a time, and a regex or glob covers exactly the layers
    it matches -- ``re:model\\.layers\\.[01]\\..*`` is 2 of 16, not all 16.
    """
    if component == "lm_head":
        return float(any(_pattern_matches(p, n) for p in patterns
                         for n in _LM_HEAD_NAMES))
    layers = max(num_layers, 1)
    tails = _member_tails(component, member)
    vision = component.startswith("vision")
    covered: set[int] = set()
    for p in patterns:
        plain = not p.startswith("re:") and not any(c in p for c in "*?[")
        m = _INDEXED_NAME.match(p) if plain else None
        if m:
            on_vision = any(s in m["root"] for s in ("vision", "visual"))
            tail = m["tail"]
            if on_vision == vision and any(
                    t == tail or t.startswith(tail + ".")
                    or tail.startswith(t + ".") for t in tails):
                covered.add(int(m["idx"]))
            continue

        def hits(layer: int) -> bool:
            return any(_pattern_matches(p, n)
                       for n in _component_names(component, member, layer))

        if plain and not any(ch.isdigit() for ch in p):
            # No index anywhere in a plain entry, so it cannot tell layers
            # apart: one layer stands for all of them.
            if hits(0):
                return 1.0
            continue
        # A regex, a glob, or a plain entry with an index the layer-root
        # pattern did not place: test every layer, because the pattern can
        # select any subset of them.
        covered.update(i for i in range(layers) if hits(i))
        if len(covered) >= layers:
            return 1.0
    return min(len(covered) / layers, 1.0)


def _member_weights(d: ModelDims, component: str) -> dict[str, float]:
    """Relative size of each member of a component, for partial coverage."""
    if component == "attn":
        q = d.num_attn_heads * d.head_dim
        kv = d.num_kv_heads * d.head_dim
        return {"q": q, "k": kv, "v": kv, "o": q}
    if component == "state":
        return {"in": d.conv_dim + d.state_inner, "out": d.state_inner}
    if component.startswith("vision"):
        extra = max(d.vision_params - d.vision_attn_params - d.vision_mlp_params, 0)
        if not (d.vision_attn_params or d.vision_mlp_params):
            return {"attn": 1.0, "mlp": 3.0, "extra": 0.0}
        return {"attn": float(d.vision_attn_params),
                "mlp": float(d.vision_mlp_params), "extra": float(extra)}
    return {member: 1.0 for member in _component_members(component)}


@functools.lru_cache(maxsize=256)
def _component_coverage(tiers: tuple[tuple[float, tuple[str, ...]], ...],
                        component: str, weights: tuple[tuple[str, float], ...],
                        num_layers: int) -> tuple[tuple[float, float], ...]:
    """((bpp, fraction), ...) of a component held at each override tier.

    Tiers are taken widest first, and each member-layer is priced once, at
    the widest tier that names it. Whatever no tier covers is left to the
    caller's quantized rate.
    """
    total = sum(w for _, w in weights) or 1.0
    shares: dict[float, float] = {}
    for member, weight in weights:
        remaining = 1.0
        for bpp, patterns in sorted(tiers, key=lambda t: -t[0]):
            take = min(_member_coverage(patterns, component, member,
                                        num_layers), remaining)
            if take > 0:
                shares[bpp] = shares.get(bpp, 0.0) + take * weight / total
                remaining -= take
    return tuple(sorted(shares.items(), reverse=True))


def _lm_head_quantized(qcfg: dict) -> bool:
    """True only if the config asks for the output projection to be quantized.

    The default is the other way round: AWQ and GPTQ carry an explicit
    `lm_head: true` when they want it, and compressed-tensors has to name it
    in a config group's `targets`.
    """
    if qcfg.get("lm_head") is True:
        return True
    groups = qcfg.get("config_groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            targets = group.get("targets") or []
            if isinstance(targets, str):
                targets = [targets]
            for target in targets:
                if any(_pattern_matches(str(target), name)
                       for name in _LM_HEAD_NAMES):
                    return True
    return False


# Every probed component that is an nn.Linear. A config-group target naming
# the class ("Linear") covers all of them; the output head is left to
# _lm_head_quantized because vLLM's ParallelLMHead is not a LinearBase.
_LINEAR_COMPONENTS = ("attn", "state", "router", "dense_ffn", "shared_ffn",
                      "expert_ffn", "vision")


def _target_matches(target: str, component: str) -> bool:
    """True if a config-group `targets` entry selects this component.

    Targets are positive rules: a module class ("Linear"), a `re:` regex, or
    a module name. A class this script does not recognise selects nothing,
    which prices its modules at full width -- the safe direction.
    """
    if not target.startswith("re:") and "." not in target and target[:1].isupper():
        return target.endswith("Linear")
    return any(_pattern_matches(target, name)
               for name in _component_names(component))


def _group_weight_key(qcfg: dict, group: dict) -> str | None:
    """BYTES_PER_PARAM key of one compressed-tensors config group's weights.

    The format name ("nvfp4-pack-quantized") and ModelOpt's quant_algo sit
    beside the group, so they are read with it.
    """
    weights = group.get("weights")
    if not isinstance(weights, dict):
        return None
    text = " ".join(str(value or "") for value in (
        weights.get("type"), group.get("format"),
        qcfg.get("format"), qcfg.get("quant_algo")))
    return _key_from_bits(weights.get("num_bits"), text,
                          weights.get("group_size"))


def _group_targets(qcfg: dict) -> dict[str, str | None] | None:
    """Component -> weight key of the config group that quantizes it.

    None when no config group declares targets, leaving the exclusion lists
    to decide alone. Otherwise a component no group targets is absent from
    the result: nothing selected it, so it is stored at full width. Where
    two groups select one component the wider width wins.
    """
    groups = qcfg.get("config_groups")
    if not isinstance(groups, dict):
        return None
    resolved: dict[str, str | None] = {}
    declared = False
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        targets = group.get("targets") or []
        if isinstance(targets, str):
            targets = [targets]
        if not targets:
            continue
        declared = True
        key = _group_weight_key(qcfg, group)
        for component in _LINEAR_COMPONENTS:
            if not any(_target_matches(str(t), component) for t in targets):
                continue
            prev = resolved.get(component)
            if component not in resolved or (
                    key and (prev is None
                             or BYTES_PER_PARAM[key] > BYTES_PER_PARAM[prev])):
                resolved[component] = key
    return resolved if declared else None


def _key_from_bits(bits: object, dtype: object = "",
                   group_size: object = None) -> str | None:
    """Map a declared weight bit-width onto a BYTES_PER_PARAM key.

    A 4-bit float is NVFP4 or MXFP4, and the width alone cannot tell them
    apart: NVFP4 scales every 16 values, MXFP4 every 32. Folding NVFP4 into
    mxfp4 underprices it and, worse, drops the NO_XPU_KERNEL_PATH warning for
    a checkpoint vLLM-XPU refuses to load.
    """
    try:
        width = int(bits)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    text = str(dtype or "").lower()
    floating = any(token in text for token in
                   ("float", "fp8", "fp4", "nvfp", "mxfp", "e4m3", "e5m2"))
    if width == 16:
        return "bf16"
    if width == 8:
        return "fp8" if floating else "int8"
    if width == 4:
        if not floating:
            return "int4"
        return ("nvfp4" if "nvfp" in text or str(group_size) == "16"
                else "mxfp4")
    if width == 3:
        return "int3"
    if width == 2:
        return "int2"
    return None


def detect_quant(cfg: dict) -> str | None:
    """Weight format a pre-quantized checkpoint declares, or None.

    Reading `quant_method` alone misses most checkpoints, because the common
    values ("compressed-tensors", "awq", "gptq", "bitsandbytes") name the
    container rather than the width. The width itself is declared elsewhere,
    so it is read from wherever the exporter put it.
    """
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        return None

    method = str(qcfg.get("quant_method", "")).lower()
    if method in BYTES_PER_PARAM:
        return method

    if qcfg.get("load_in_4bit"):
        return "int4"
    if qcfg.get("load_in_8bit"):
        return "int8"

    key = _key_from_bits(
        qcfg.get("bits") or qcfg.get("weight_bits") or qcfg.get("w_bit"),
        qcfg.get("weight_dtype") or qcfg.get("data_type") or method,
        qcfg.get("group_size"),
    )
    if key:
        return key

    # compressed-tensors declares the width per config group.
    groups = qcfg.get("config_groups")
    if isinstance(groups, dict):
        found = []
        for group in groups.values():
            if isinstance(group, dict):
                key = _group_weight_key(qcfg, group)
                if key:
                    found.append(key)
        if found:
            # Widest declared width wins, so a mixed export is never charged
            # less than its largest tensors actually cost.
            return max(found, key=lambda name: BYTES_PER_PARAM[name])

    # Model Optimizer and TensorRT exports name the recipe, not the width.
    algo = str(qcfg.get("quant_algo") or qcfg.get("fmt") or "").lower()
    for token, key in (("fp8", "fp8"), ("nvfp4", "nvfp4"), ("mxfp4", "mxfp4"),
                       ("w4a16", "int4"), ("int4", "int4"),
                       ("w8a8", "int8"), ("int8", "int8")):
        if token in algo:
            return key
    return None


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
    if not isinstance(qcfg, dict):
        return None, False
    method = str(qcfg.get("quant_method", "")).lower()

    if method in WIDTH_FROM_BITS_METHODS:
        bits = qcfg.get("bits", qcfg.get("w_bit"))
        # AutoRound also exports float formats under the same method name
        # (`data_type: nv_fp` / `mx_fp`). Reducing those to an int width
        # would misprice NVFP4 and drop its missing-kernel warning.
        data_type = str(qcfg.get("data_type") or "").lower().replace("_", "")
        if "fp" in data_type or "float" in data_type:
            key = _key_from_bits(bits, data_type, qcfg.get("group_size"))
            if key:
                return key, False
        try:
            dtype = BITS_TO_QUANT.get(int(bits))
        except (TypeError, ValueError):
            dtype = None
        return (dtype, False) if dtype else ("int8", True)

    dtype = QUANT_METHOD_ALIASES.get(method, method)
    if dtype in BYTES_PER_PARAM:
        return dtype, False
    # Container formats (compressed-tensors, bitsandbytes, ModelOpt) declare
    # the width elsewhere; detect_quant() reads it from wherever it lives.
    return detect_quant(cfg), False


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

    The reasons live in tp_problems(); this is the yes/no form. It is a
    necessary condition, not a promise: a runtime can impose further
    constraints (quantization group sizes, expert counts, its own padding
    rules). Callers should say "shard dimensions divide" rather than "will
    launch".
    """
    return tp >= 1 and not tp_problems(d, tp)


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
    mats = d.ffn_matrices
    per_layer = mats * d.hidden * (
        d.num_experts * d.intermediate
        + d.num_shared_experts * (d.shared_intermediate or d.intermediate))
    return moe_layer_count(d) * per_layer


def calculate_mixed_precision_weights(cfg: dict, d: ModelDims, params: int,
                                       quant: str, tp: int,
                                       expert_dtype: str | None = None
                                       ) -> tuple[int, dict | None]:
    """Calculate weight bytes accounting for mixed-precision quantization.

    No production quantization scheme converts every tensor. The embedding
    table is always left alone, output projections usually are, and many
    checkpoints also exempt attention or the router
    (quantization_config.ignore, modules_to_not_convert, and four other
    spellings of the same list). Charging every parameter at the quantized
    rate therefore under-estimates weights, which is the dangerous
    direction: it reports FITS for a launch that runs out of memory.

    Two config mechanisms add to that, and a model may use both:

    - `quantization_config.modules_to_not_convert` (e.g. openai/gpt-oss-20b)
      keeps named components at full precision while quantizing the rest.
    - a separate `expert_dtype` for the MoE FFNs (e.g. DeepSeek-V4), passed
      in by the caller so an explicit --quant can suppress it.

    Returns:
        (weights_bytes, breakdown_dict or None)
    """
    quant_bpp = BYTES_PER_PARAM[quant]
    full_bpp = 2.0

    # Both engines tensor-parallel the vision tower by default (vLLM's
    # mm_encoder_tp_mode defaults to "weights"), so it is divided like the
    # rest. The exception is a tower whose head count does not divide TP:
    # there is no valid split, so every rank holds a full copy.
    vision_replicated = bool(
        d.vision_params and tp > 1
        and (not d.vision_num_heads or d.vision_num_heads % tp)
    )
    # An expert dtype means nothing without experts.
    expert_bpp = (BYTES_PER_PARAM[expert_dtype] if expert_dtype and d.is_moe
                  else quant_bpp)

    if quant_bpp >= full_bpp and expert_bpp >= full_bpp:
        # Nothing is quantized, so there is no mix to break down.
        weights = params * quant_bpp / tp
        if vision_replicated:
            weights += d.vision_params * quant_bpp * (1 - 1 / tp)
        return int(weights), None

    qcfg = cfg.get("quantization_config", {})
    if not isinstance(qcfg, dict):
        qcfg = {}
    patterns = exclusion_patterns(qcfg)
    as_shipped = quant == quant_from_config(cfg)[0]
    method = str(qcfg.get("quant_method", "")).lower()

    # Override tiers: the exclusion lists at full width, plus -- when pricing
    # the checkpoint as shipped -- any per-module widths AutoRound or GPTQ
    # declare (Intel's mixed AutoRound exports keep attention, the shared
    # expert and the linear-attention projections at 8 bits).
    tiers = [(full_bpp, tuple(patterns))]
    if as_shipped:
        by_bits: dict[int, list[str]] = {}
        for name, bits in _per_module_bits(qcfg):
            if bits < 16 and bits in BITS_TO_QUANT:
                by_bits.setdefault(bits, []).append(name)
        tiers += [(BYTES_PER_PARAM[BITS_TO_QUANT[b]], tuple(names))
                  for b, names in by_bits.items()]
    tiers_key = tuple(tiers)

    # Layers each component spans, so an entry naming one layer prices one
    # layer's share rather than the whole component.
    dense_ffn_layers = (min(d.first_k_dense_replace, _ffn_layers(d))
                        if d.is_moe and d.first_k_dense_replace > 0
                        and d.dense_intermediate > 0
                        else (0 if d.is_moe else _ffn_layers(d)))
    span = {
        "attn": _attn_weight_layers(d), "state": d.num_recurrent_layers,
        "router": moe_layer_count(d), "dense_ffn": dense_ffn_layers,
        "shared_ffn": moe_layer_count(d), "expert_ffn": moe_layer_count(d),
        "vision": d.vision_layers, "lm_head": 1,
    }

    def overrides(component: str) -> tuple[tuple[float, float], ...]:
        if component == "lm_head":
            weights = (("lm_head", 1.0),)
        else:
            weights = tuple(_member_weights(d, component).items())
        key = component
        if component == "vision" and d.vision_naming:
            key = f"vision@{d.vision_naming}"
        return _component_coverage(tiers_key, key, weights,
                                   max(span[component], 1))

    # Structural full-width rules, which no config states because no scheme
    # could do otherwise:
    # - Embedding tables: compressed-tensors, AWQ, GPTQ and mxfp4 target
    #   Linear modules and nn.Embedding is not one. (compressed-tensors can
    #   quantize an embedding a config group targets explicitly; that is
    #   rare enough not to model, and pricing it wide is the safe side.)
    # - MoE routers: vLLM builds every router it ships (DeepSeek, Mixtral,
    #   Qwen-MoE, gpt-oss, Llama-4) without a quant config, so a `Linear`
    #   target does not reach it.
    # - Everything but the experts of an mxfp4 checkpoint: vLLM's MXFP4
    #   method only converts MoE expert weights.
    full_width = {"embed", "router"}
    if as_shipped and method == "mxfp4":
        full_width |= {"head", "attn", "state", "dense_ffn", "shared_ffn",
                       "vision"}
    if not _lm_head_quantized(qcfg):
        full_width.add("head")

    # compressed-tensors `targets` are positive rules: a group that selects
    # only the experts leaves attention and the dense FFN at full width even
    # though no exclusion names them. When pricing the checkpoint as shipped,
    # each component also takes its own group's width rather than the
    # widest one detect_quant() reports for the whole model.
    targets = _group_targets(qcfg)
    if targets is not None:
        full_width |= {c for c in _LINEAR_COMPONENTS if c not in targets}
    group_keys = (targets if targets is not None and quant == detect_quant(cfg)
                  else {})

    h = d.hidden
    vocab = d.vocab

    embed_params = vocab * h + d.per_layer_embed_params
    head_params = 0 if d.tied else vocab * h

    # Attention blocks, over attention layers only -- a hybrid's recurrent
    # layers have no Q/K/V/O projections to quantize.
    attn_layers = _attn_weight_layers(d)
    attn_params = _attn_proj_params_total(d) + attn_layers * 4 * h  # + norms

    state_params = d.num_recurrent_layers * _recurrent_block_params(d)

    # Router params, counted from the architecture rather than approximated:
    # a `2 * hidden` placeholder is 16x light on a 32-expert model, which
    # underprices the bf16 uplift an exclusion list charges for.
    router_p = router_params(d)

    dense_ffn_params, shared_ffn_params, expert_ffn_params = _ffn_param_split(d)

    # Whatever the analytic counts above do not account for (norms, biases, a
    # family this script models approximately) is charged with the largest FFN
    # bucket so the components still sum to the total parameter count.
    accounted = (embed_params + head_params + attn_params + state_params
                 + router_p + d.vision_params + dense_ffn_params
                 + shared_ffn_params + expert_ffn_params)
    residual = params - accounted
    if expert_ffn_params:
        expert_ffn_params = max(expert_ffn_params + residual, 0)
    else:
        dense_ffn_params = max(dense_ffn_params + residual, 0)

    components = [
        ("embed", embed_params),
        ("head", head_params),
        ("attn", attn_params),
        ("state", state_params),
        ("router", router_p),
        ("dense_ffn", dense_ffn_params),
        ("shared_ffn", shared_ffn_params),
        ("expert_ffn", expert_ffn_params),
        ("vision", d.vision_params),
    ]

    breakdown: dict = {}
    total_bytes = 0.0
    for name, count in components:
        # Experts carry their own dtype when the config declares one.
        is_expert = name in ("shared_ffn", "expert_ffn")
        rate = expert_bpp if is_expert else quant_bpp
        if group_keys.get(name) and not (is_expert and expert_dtype):
            rate = BYTES_PER_PARAM[group_keys[name]]
        if name in full_width:
            bpp = full_bpp
        else:
            # Blend: the share an override tier names at that tier's width,
            # the rest at the quantized rate.
            shares = overrides("lm_head" if name == "head" else name)
            covered = sum(f for _, f in shares)
            bpp = (sum(b * f for b, f in shares)
                   + max(1.0 - covered, 0.0) * rate)
        # Vision weights are replicated whole on every TP rank when
        # vision_replicated; everything else is sharded across tp ranks.
        divisor = 1 if (name == "vision" and vision_replicated) else tp
        per_rank_bytes = count * bpp / divisor
        total_bytes += per_rank_bytes
        breakdown[f"{name}_params"] = count
        breakdown[f"{name}_bytes"] = int(per_rank_bytes)
        breakdown[f"{name}_bpp"] = bpp
    breakdown["vision_replicated"] = vision_replicated
    breakdown["excluded_patterns"] = patterns

    return int(total_bytes), breakdown


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

    # KV cache is sharded by KV head, down to one head per rank (a full copy
    # for MLA), or by whole layers under module placement -- kv_divisor().
    in_flight = max_in_flight_tokens(device_vram_gb)
    shards = kv_divisor(d, tp, runtime)
    block = vllm_block_tokens(d, tp, kv_dtype) if runtime == "vllm" else 0
    kv = kv_bytes(d, ctx, concurrency, kv_dtype, runtime, in_flight,
                  block) // shards

    # Recurrent state is allocated per running request and does not shrink
    # with context, so it is charged alongside the KV cache rather than
    # inside it. It is already per-rank: the state functions apply the shard.
    # state_page is what the runtime reserves (vLLM pads and doubles it on a
    # hybrid); state_raw is one unpadded copy, reported beside it.
    state_raw = state_page_bytes(d, tp, runtime)
    state_page = state_reserved_bytes(d, tp, runtime, kv_dtype)
    state = state_page * concurrency
    pool_split = sglang_pool_split_bytes(d, kv, state, runtime)

    vllm_profile = None
    if runtime == "vllm":
        # vLLM reserves what its startup profiling pass measured, whatever the
        # context and concurrency: a runtime floor plus any encoder buffers it
        # keeps (Framework), and the pass's transient peak (Activations).
        floor, held, act = vllm_reservation(d, quant, device_vram_gb, None, ctx)
        framework = floor + held
        vllm_profile = {
            "floor": floor, "encoder_held": held, "peak": act,
            "batched_tokens": default_max_num_batched_tokens(device_vram_gb),
            "patches": vision_profile_patches(
                d, default_max_num_batched_tokens(device_vram_gb), ctx),
        }
    else:
        act = activation_bytes(d, ctx, concurrency,
                               quant if quant in ("bf16", "fp16") else "bf16")
        framework = int(FRAMEWORK_OVERHEAD_GB[runtime] * GB)
    total = weights + kv + state + pool_split + act + framework
    device_b = int(device_vram_gb * GB)
    usable_b = usable_bytes(device_vram_gb, gpu_memory_utilization, runtime)
    free_for_kv = usable_b - weights - act - framework
    # Base footprint (weights + act + framework) can exceed usable VRAM, in
    # which case free_for_kv is negative and there is no room for any KV.
    # Clamp to 0 here so callers (TP sweep, single-result print) see a
    # consistent "no headroom" signal instead of a negative-divided value.
    if free_for_kv <= 0:
        max_concurrency = 0
        max_context = 0
    else:
        per_request = kv_bytes(d, ctx, 1, kv_dtype, runtime, in_flight,
                               block) // shards
        # Each request costs its KV and its state -- or, under SGLang's split,
        # whichever of the two pools it exhausts first, scaled to the whole.
        per_request_total = (per_request + state_page
                             + sglang_pool_split_bytes(d, per_request,
                                                       state_page, runtime))
        max_concurrency = free_for_kv // max(per_request_total, 1)
        conc = max(concurrency, 1)
        kv_budget = free_for_kv - state_page * conc
        if runtime == "sglang" and state_page and _attention_layers(d):
            # The KV pool is the 1 / (1 + ratio) share, and the state pool
            # must still hold `conc` requests.
            r = SGLANG_MAMBA_FULL_MEMORY_RATIO
            kv_budget = (int(free_for_kv / (1 + r))
                         if state_page * conc <= free_for_kv * r / (1 + r)
                         else 0)
        extra_at = None
        if runtime == "vllm" and d.vision_naming == "intern":
            # The profiled video grows with the context; re-price it.
            base = sum(vllm_reservation(d, quant, device_vram_gb, None, ctx)[1:])
            extra_at = lambda n: (sum(vllm_reservation(d, quant, device_vram_gb,
                                                       None, n)[1:]) - base)
        max_context = _max_context(d, kv_dtype, conc, tp, runtime, in_flight,
                                   kv_budget, block, extra_at)
    return {
        "dims": d,
        "params": params,
        "weights": weights,
        "kv": kv,
        "state": state,
        "state_page": state_page,
        "state_raw": state_raw,
        "state_slots": concurrency if state_page else 0,
        "pool_split": pool_split,
        "block_tokens": block,
        "in_flight_tokens": in_flight,
        "sliding_ctx": sliding_ctx(d, ctx, runtime, in_flight)
                       if d.num_sliding_attn_layers else 0,
        "act": act,
        "framework": framework,
        "vllm_profile": vllm_profile,
        "total": total,
        "device_vram": device_b,
        "usable_vram": usable_b,
        "fits": total <= usable_b,
        "headroom": usable_b - total,
        "max_concurrency": int(max_concurrency),
        "max_context": int(max_context),
        "mixed_breakdown": mixed_breakdown,
        "tp_problems": (tp_problems(d, tp)
                        if runtime in TENSOR_PARALLEL_RUNTIMES else []),
        "weights_floor": weights_floor,
        "expert_dtype": expert_dtype,
    }


def _max_context(d: ModelDims, kv_dtype: str, concurrency: int, tp: int,
                 runtime: str, in_flight: int, budget: int,
                 block_tokens: int = 0, extra_at=None) -> int:
    """Largest context whose KV cache fits in `budget` bytes.

    Solved by bisection rather than dividing by a per-token cost, because on
    a sliding-window model the cost stops growing once the window is full --
    dividing would report a ceiling far below the real one. `extra_at(n)`
    is any non-KV memory that also moves with the context -- vLLM's
    reservation on an InternVL-style tower, profiled with a video whose
    frame count follows the context -- as a delta from the requested one.
    """
    if budget <= 0:
        return 0
    shards = kv_divisor(d, tp, runtime)

    def fits(n: int) -> bool:
        kv = kv_bytes(d, n, concurrency, kv_dtype, runtime, in_flight,
                      block_tokens) // shards
        return kv + (extra_at(n) if extra_at else 0) <= budget

    low, high = 0, 1 << 24
    if fits(high):
        return high
    while low < high - 1:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid
    return low


def verdict_cell(result: dict, usable_vram_gb: float) -> str:
    if result["fits"]:
        headroom_gb = result["headroom"] / GB
        if headroom_gb < max(1.0, 0.10 * usable_vram_gb):
            return "tight"
        return "fits"
    parts = [
        ("weights", result["weights"]),
        ("KV", result["kv"]),
        ("state", result["state"]),
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
    usable_vram_gb = usable_bytes(device_vram_gb, gpu_memory_utilization,
                                  runtime) / GB
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
    unlaunchable = {}
    for tp in tp_values:
        result = estimate(cfg, args.quant, args.kv_dtype, args.ctx,
                          args.concurrency, tp, args.runtime,
                          args.device_vram_gb, args.gpu_memory_utilization,
                          expert_dtype)
        rows.append((
            tp,
            result["weights"],
            result["kv"],
            result["state"],
            result["act"],
            result["framework"],
            result["total"],
            result["headroom"],
            result["fits"],
            result["max_concurrency"],
            result["max_context"],
        ))
        if result["tp_problems"]:
            # Memory is moot when the engine refuses the split.
            unlaunchable[tp] = result["tp_problems"]
        elif result["fits"] and first_fit is None:
            first_fit = tp

    # The state column is dead weight for the overwhelmingly common case of a
    # model with no recurrent layers, so only show it when there is state.
    show_state = any(row[3] for row in rows)

    print()
    print("TP sweep")
    state_head = f"{'state':>9} " if show_state else ""
    print(f"  {'TP':>2}  {'weights':>9} {'KV':>9} {state_head}{'act':>9} "
          f"{'fw':>9} {'total':>9} {'headroom':>9} "
          f"{'fit':>3}  {'max_c':>5}  {'max_ctx':>7}")
    print("  " + "-" * (93 if show_state else 83))
    for (tp, weights, kv, state, act, framework, total, headroom,
         fits, max_c, max_ctx) in rows:
        verdict = "n/a" if tp in unlaunchable else ("YES" if fits else "NO")
        state_cell = f"{fmt_gb(state)} " if show_state else ""
        print(f"  {tp:>2}  {fmt_gb(weights)} {fmt_gb(kv)} {state_cell}{fmt_gb(act)} "
              f"{fmt_gb(framework)} {fmt_gb(total)} {fmt_gb(headroom)} "
              f"{verdict:>3}  {max_c:>5}  {max_ctx:>7}")
    for tp, problems in unlaunchable.items():
        print(f"  n/a: vLLM will not start at TP={tp}: {'; '.join(problems)}")
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
            f"  {'; '.join(tp_problems(dims, args.tp))}\n"
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

    state = result["state"]

    arch_label = "decoder-only LLM"
    if d.is_moe:
        arch_label = "MoE"
    if d.num_recurrent_layers:
        arch_label = "hybrid SSM + attention" if (
            d.num_full_attn_layers or d.num_sliding_attn_layers
        ) else "state-space model"
        if d.is_moe:
            arch_label += ", MoE"
    elif d.num_sliding_attn_layers:
        arch_label += ", sliding-window attention"
    if d.is_vlm:
        arch_label = "VLM (LLM backbone + vision tower)"

    print(f"Model:             {args.model}")
    print(f"Architecture:      {d.arch_family}  ({arch_label})")
    if d.num_recurrent_layers or d.num_sliding_attn_layers:
        pieces = []
        if d.num_full_attn_layers:
            pieces.append(f"{d.num_full_attn_layers} full attention")
        if d.num_sliding_attn_layers:
            pieces.append(f"{d.num_sliding_attn_layers} sliding "
                          f"(window {d.sliding_window})")
        if d.num_recurrent_layers:
            pieces.append(f"{d.num_recurrent_layers} recurrent")
        print(f"Layers:            {d.num_layers} total: {', '.join(pieces)}")
        if d.attn_on_recurrent_layers:
            print("                   (attention and SSM run in the same "
                  "layer, so both are charged)")
        if d.shared_attn_blocks:
            print(f"                   ({d.num_full_attn_layers} hybrid layer(s) "
                  f"share {d.shared_attn_blocks} attention block(s): one KV "
                  f"cache per layer, one set of weights per block)")
        if d.num_recurrent_layers:
            shape = (f"{d.num_state_heads} head(s) x {d.state_head_dim_k} x "
                     f"{d.state_head_dim_v}, " if d.num_state_heads else "")
            print(f"Recurrent state:   {shape}conv {d.conv_dim} x "
                  f"{d.conv_kernel - 1}  "
                  f"({result['state_raw'] / MB:.1f} MiB per request)")
            if result["state_page"] > result["state_raw"]:
                print(f"                   vllm reserves "
                      f"{result['state_page'] / MB:.1f} MiB: prefix caching "
                      f"holds {VLLM_ALIGN_MODE_STATE_PAGES} state pages per "
                      f"layer, each padded to a {result['block_tokens']}-token "
                      f"attention block")
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
        for key, label in (("embed", "Embeddings"), ("head", "Output head"),
                           ("attn", "Attention"), ("state", "SSM / conv"),
                           ("router", "Routers"), ("dense_ffn", "Dense FFN"),
                           ("shared_ffn", "Shared experts"),
                           ("expert_ffn", "Routed experts"),
                           ("vision", "Vision tower")):
            # Below 5M params both columns round to 0.00 and the line is noise.
            if mixed_breakdown[f"{key}_params"] < 5_000_000:
                continue
            print(f"  {label:<15} {fmt_gb(mixed_breakdown[f'{key}_bytes'])}   "
                  f"({mixed_breakdown[f'{key}_params']/1e9:.2f}B params "
                  f"@ {mixed_breakdown[f'{key}_bpp']:.2f} B/p)")
        print(f"  -----              -----")
        print(f"  Weights total   {fmt_gb(weights)}")
        if mixed_breakdown["vision_replicated"]:
            print(f"  (vision tower replicated on every rank: its "
                  f"{d.vision_num_heads or 'unknown'} head(s) do not divide "
                  f"TP {args.tp})")

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
    kv_note = (f"({args.ctx} tok x concurrency {args.concurrency}, "
               f"kv_dtype {args.kv_dtype})")
    print(f"  KV cache        {fmt_gb(kv)}   {kv_note}")
    if d.num_sliding_attn_layers and result["sliding_ctx"] < args.ctx:
        print(f"                  {d.num_sliding_attn_layers} windowed layer(s) "
              f"hold {result['sliding_ctx']} tok, not {args.ctx} "
              f"({args.runtime} accounting)")
    if state:
        print(f"  Recurrent state {fmt_gb(state)}   "
              f"({result['state_slots']} request slot(s) x "
              f"{result['state_page'] / MB:.1f} MiB; independent of context)")
    if result["pool_split"]:
        r = SGLANG_MAMBA_FULL_MEMORY_RATIO
        print(f"  Pool split      {fmt_gb(result['pool_split'])}   (sglang fixes "
              f"the state pool at {r:g}/{1 + r:g} of free memory and the KV "
              f"pool at 1/{1 + r:g}; the larger need sets both)")
    profile = result.get("vllm_profile")
    if profile:
        # vLLM fixes both rows at startup from one profiling pass, so they do
        # not move with --ctx or --concurrency; say what the pass ran.
        image = (f" + a {profile['patches']}-patch image batch"
                 if profile["patches"] else "")
        print(f"  Activations     {fmt_gb(act)}   (vllm profiling pass: "
              f"{profile['batched_tokens']} batched tokens{image}; fixed at "
              f"startup, not per request)")
        held = (f" + {fmt_gb(profile['encoder_held']).strip()} encoder buffers"
                if profile["encoder_held"] else "")
        print(f"  Framework       {fmt_gb(framework)}   (vllm runtime floor"
              f"{held})")
    else:
        print(f"  Activations     {fmt_gb(act)}   (estimate)")
        print(f"  Framework       {fmt_gb(framework)}   ({args.runtime})")
    print(f"  -----              -----")
    print(f"  Total           {fmt_gb(total)}")
    print()
    per_device = f"  (per device, x{args.tp} TP)" if args.tp > 1 else ""
    print(f"Device VRAM:       {args.device_vram_gb:6.2f} GB{per_device}")
    if args.runtime == "vllm":
        # vLLM's utilization is a share of what the driver reports, which sits
        # ~5% under physical -- 30.29 of 31.89 GiB on a B70.
        exposed = args.device_vram_gb * XPU_ALLOCATABLE_FRACTION
        print(f"Usable VRAM:       {usable_b / GB:6.2f} GB "
              f"(gpu_memory_utilization={args.gpu_memory_utilization:.2f} of "
              f"the {exposed:.2f} GB the driver exposes)")
        if args.gpu_memory_utilization >= 1.0:
            print("Note: vLLM defaults to --gpu-memory-utilization 0.9; pass "
                  "the value the launch will use.")
    elif args.gpu_memory_utilization < 1.0:
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

    for problem in result["tp_problems"]:
        print(f"Warning:           vLLM will not start at TP={args.tp}: {problem}")

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
        [("weights", weights), ("KV cache", kv), ("recurrent state", state),
         ("activations", act), ("framework", framework)],
        key=lambda item: -item[1],
    )
    binding, _ = parts[0]
    print(f"  Binding constraint: {binding} ({fmt_gb(parts[0][1])}) dominates")

    in_flight = result["in_flight_tokens"]

    def kv_at(ctx: int, concurrency: int, kv_dtype: str) -> int:
        block = (vllm_block_tokens(d, args.tp, kv_dtype)
                 if args.runtime == "vllm" else 0)
        return kv_bytes(d, ctx, concurrency, kv_dtype, args.runtime,
                        in_flight, block) // kv_divisor(d, args.tp, args.runtime)

    suggestions: list[str] = []
    if binding == "KV cache":
        if args.ctx > 1024:
            new_ctx = max(1024, args.ctx // 2)
            new_kv = kv_at(new_ctx, args.concurrency, args.kv_dtype)
            suggestions.append(
                f"drop ctx {args.ctx} -> {new_ctx} (saves {fmt_gb(kv - new_kv)})"
            )
        if args.kv_dtype in ("bf16", "fp16"):
            new_kv = kv_at(args.ctx, args.concurrency, "fp8")
            suggestions.append(
                f"kv-dtype {args.kv_dtype} -> fp8 (saves {fmt_gb(kv - new_kv)})"
            )
        if args.concurrency > 1:
            new_kv = kv_at(args.ctx, 1, args.kv_dtype)
            suggestions.append(
                f"concurrency {args.concurrency} -> 1 (saves {fmt_gb(kv - new_kv)})"
            )
    if binding == "recurrent state":
        # The state pool scales only with the number of running requests --
        # shortening the context does nothing for it.
        if args.concurrency > 1:
            new_concurrency = max(1, args.concurrency // 2)
            saved = result["state_page"] * (args.concurrency - new_concurrency)
            suggestions.append(
                f"concurrency {args.concurrency} -> {new_concurrency} "
                f"(saves {fmt_gb(saved)}; the state pool scales with running "
                f"requests, not context)"
            )
        if args.tp == 1:
            at_tp2 = state_reserved_bytes(d, 2, args.runtime, args.kv_dtype)
            suggestions.append(
                f"--tp 2 shards the state pool across two XPUs (saves about "
                f"{fmt_gb(state - at_tp2 * args.concurrency)})"
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
