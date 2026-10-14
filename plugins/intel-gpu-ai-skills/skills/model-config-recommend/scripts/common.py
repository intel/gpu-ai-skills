# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for model-config-recommend scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path


def hf_headers(user_agent: str) -> dict[str, str]:
    headers = {"User-Agent": user_agent}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def http_get_json(url: str, user_agent: str) -> dict | None:
    req = urllib.request.Request(url, headers=hf_headers(user_agent))
    try:
        # Bandit B310 suppression justification: callers build url from the https://huggingface.co literal;
        # scheme and host are not reachable from any parameter.
        with urllib.request.urlopen(req, timeout=30) as response:  # nosec B310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        if exc.code in (401, 403):
            sys.exit(
                f"HTTP {exc.code} on {url}. Either typo, gated/private, "
                f"or rate-limited. Set HF_TOKEN and retry."
            )
        raise


def hf_model_info(model_id: str, user_agent: str) -> dict | None:
    return http_get_json(f"https://huggingface.co/api/models/{model_id}", user_agent)


def looks_like_gguf_repo(info: dict | None) -> bool:
    if not info:
        return False
    siblings = info.get("siblings") or []
    return any((s.get("rfilename") or "").lower().endswith(".gguf")
               for s in siblings)


def is_local_model(model_id: str) -> bool:
    """True if --model names a path on disk rather than a Hub repo id.

    Exported because callers need the same answer fetch_config uses: a local
    checkout has no mutable branch to resolve, so revision-pinning rules apply
    to Hub ids only. Two copies of this test could disagree.
    """
    return model_id.endswith(".json") or model_id.startswith(("/", "."))


def fetch_config(model_id: str, user_agent: str, revision: str = "main") -> dict:
    """Return a local or HF-hosted Transformers config.json.

    A local --model is caller-supplied data, so every way it can fail to be
    a JSON object is turned into a one-line message. Letting json/OSError
    escape here would print a traceback carrying absolute paths.

    ``revision`` is a Hub git revision -- branch, tag, or commit SHA. It
    defaults to ``main`` for backward compatibility, but a caller that pins the
    launch line to a revision must read the config from that same revision:
    recommending against ``main``'s config while pinning the engine elsewhere
    would describe a model the user is not about to run.
    """
    if is_local_model(model_id):
        try:
            cfg = json.loads(Path(model_id).read_text(encoding="utf-8"))
        except OSError as exc:
            sys.exit(f"cannot read {model_id}: {exc}")
        except UnicodeDecodeError as exc:
            sys.exit(f"{model_id} is not valid UTF-8 text: {exc}")
        except json.JSONDecodeError as exc:
            sys.exit(f"{model_id} is not valid JSON: {exc}")
        if not isinstance(cfg, dict):
            sys.exit(
                f"{model_id} must contain a JSON object, got "
                f"{type(cfg).__name__} — point --model at a config.json."
            )
        return cfg
    # quote() the revision, not the model id: a branch name may legitimately
    # contain characters that need escaping in a path segment, and the value
    # arrives from the command line.
    ref = urllib.parse.quote(revision, safe="")
    cfg = http_get_json(f"https://huggingface.co/{model_id}/raw/{ref}/config.json",
                        user_agent)
    if cfg is None:
        info = hf_model_info(model_id, user_agent)
        if looks_like_gguf_repo(info):
            sys.exit(
                f"{model_id} looks like a GGUF repository and has no root "
                f"config.json. Use a Transformers/safetensors checkpoint for "
                f"this analytical LLM path."
            )
        # Name the revision: now that it is caller-supplied, a typo'd SHA or a
        # tag that does not exist lands here, and "no config.json at root" sends
        # the reader looking at the wrong thing.
        sys.exit(
            f"No config.json at root for {model_id} at revision {revision}. "
            f"Either that revision does not exist, or this repo has no "
            f"HF Transformers-style config.json -- which this analytical LLM "
            f"path needs."
        )
    return cfg


# Layer-layout and recurrent-state handling below mirrors the model-can-it-fit
# skill's fit.py. The two skills are installed separately, so neither can import
# the other; the rationale for each rule lives in that skill's
# references/coverage-and-formulas.md, and tests/test_fit.py's
# TestRecommenderParity holds the two to the same numbers.

GiB = 1024 ** 3

BYTES_PER_STATE_DTYPE = {
    "bfloat16": 2.0, "bf16": 2.0,
    "float16": 2.0, "fp16": 2.0, "half": 2.0,
    "float32": 4.0, "fp32": 4.0, "float": 4.0,
    "float8_e4m3fn": 1.0, "float8_e5m2": 1.0, "fp8": 1.0,
}

# vLLM picks --max-num-batched-tokens from a device-memory tier table; only
# parts at or above 70 GiB get the larger tier.
VLLM_LARGE_DEVICE_GIB = 70
VLLM_MAX_NUM_BATCHED_TOKENS_LARGE = 8192
VLLM_MAX_NUM_BATCHED_TOKENS_SMALL = 2048

# vLLM pages a hybrid's recurrent state in the attention block pool: the block
# grows in steps of this alignment until it holds a state page, the page is
# padded to the block, and default prefix caching ("align" mode) keeps two
# pages per recurrent layer per request.
VLLM_HYBRID_BLOCK_ALIGN = 64
VLLM_ALIGN_MODE_STATE_PAGES = 2

# SGLang's --swa-full-tokens-ratio default: its window pool is a fraction of
# the full-attention pool rather than a multiple of the window -- and only for
# the architectures it runs as hybrid SWA.
SGLANG_SWA_FULL_TOKENS_RATIO = 0.8
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

# vLLM's startup memory accounting, calibrated on an Arc Pro B70 against
# vLLM's own memory-profile lines (see model-can-it-fit's fit.py for the
# measurements behind each constant).
XPU_ALLOCATABLE_FRACTION = 0.946
VLLM_RUNTIME_FLOOR_GB = 1.0
VLLM_QUANT_WORKSPACE_GB = 0.15
VLLM_DENSE_PEAK_WIDTHS = 4
VLLM_MOE_PEAK_WIDTHS = 6
VLLM_VISION_PEAK_WIDTHS = 4.4
VLLM_VISION_HELD_BYTES_PER_PATCH = 20 * 1024
VLLM_QWEN_MAX_IMAGE_TOKENS = 16384

_RECURRENT_LAYER_TYPES = {
    "mamba", "mamba2", "ssm", "recurrent", "conv",
    "linear_attention", "linear_attn", "gated_delta_net",
}
_SLIDING_LAYER_TYPES = {
    "sliding_attention", "sliding_window_attention", "local_attention",
    "local_sliding_attention", "chunked_attention", "chunked_local_attention",
}
_FFN_ONLY_LAYER_TYPES = {"mlp", "moe", "feed_forward", "ffn"}
# Zamba2: a Mamba layer that also calls a shared attention block.
_HYBRID_LAYER_TYPES = {"hybrid"}

# Characters used by `hybrid_override_pattern` (Nemotron-H, Nemotron-3):
# M = Mamba layer, * = attention layer, - = MLP layer, E = MoE layer.
_PATTERN_CHARS = {"m": "recurrent", "*": "full", "-": "ffn", "_": "ffn",
                  "e": "ffn"}

# Pure state-space families: no attention heads at all.
_PURE_SSM_FAMILIES = {"mamba", "mamba2", "falcon_mamba"}

# Families that run attention and an SSM inside the same layer, so every layer
# is charged both a KV cache and a state.
_PARALLEL_HYBRID_FAMILIES = {"falcon_h1", "zamba", "zamba2"}

# vLLM keeps these families' SSM state in float32 when the config is silent.
_FP32_SSM_STATE_FAMILIES = {"nemotron_h"}

# A bare `sliding_window` key does not prove the model uses a sliding window --
# DeepSeek-V4-Flash carries one for its sparse-attention indexer while every
# layer still holds a full cache. Apply a window only when the config says
# which layers use it, or when the family is known to apply it everywhere.
_GLOBAL_SWA_FAMILIES = {
    "mistral", "mistral3", "ministral", "mixtral",
    "phi3", "phi3small", "phimoe",
    "starcoder2",
}
# Qwen windows only when use_sliding_window is true, and then only from
# max_window_layers on.
_QWEN_GATED_SWA_FAMILIES = {"qwen2", "qwen2_moe", "qwen2_vl", "qwen3",
                            "qwen3_moe"}
# Cohere2 is not here: every real checkpoint ships `sliding_window_pattern`,
# which the generic stride lookup below already reads.
_INTERLEAVED_SWA_PERIOD = {"gemma2": 2}

# Text-config fields some checkpoints leave to the transformers defaults.
_TEXT_CONFIG_DEFAULTS = {
    "gemma3_text": {
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
        "vocab_size": 262208, "sliding_window_pattern": 6,
        "tie_word_embeddings": True,
    },
}


def _normalize_dtype(name: object, default: str = "bfloat16") -> str:
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


def _ceil_div(a: int, b: int) -> int:
    return -(-a // max(b, 1))


def _scale_counts(counts: dict, listed: int, num_layers: int) -> dict:
    """Rescale a per-layer tally that does not cover exactly num_layers.

    The ratio between layer kinds is the load-bearing part, so it is preserved
    rather than the tally being discarded (which would silently charge every
    layer a full KV cache).
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
    """(window in tokens, window is chunked), or (0, False)."""
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
    """Split num_layers into full-attention / sliding / recurrent / FFN counts."""
    sources = [text_cfg, cfg]
    out = {"full": 0, "sliding": 0, "recurrent": 0, "ffn": num_layers,
           "attn_on_recurrent": False, "layout_source": ""}

    def from_list(entries: list, key: str) -> dict:
        counts = _scale_counts(_layer_type_counts(entries), len(entries),
                               num_layers)
        hybrid = counts.pop("hybrid", 0)
        counts["full"] += hybrid
        counts["recurrent"] += hybrid
        out.update(counts, layout_source=key, attn_on_recurrent=bool(hybrid))
        if not counts["ffn"]:
            out["ffn"] = num_layers
        return out

    layer_types, key = _lookup(sources, ("layer_types", "layer_type_list"))
    if isinstance(layer_types, (list, tuple)) and layer_types:
        return from_list(list(layer_types), key)

    # Kimi-Linear: 1-indexed KDA layer list under linear_attn_config.
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

    out.update(full=num_layers)
    return out


def _state_geometry(cfg: dict, text_cfg: dict, hidden: int,
                    pure_ssm: bool = False) -> dict | None:
    """Normalize a family's recurrent-state keys into one shape.

    Returns None when no geometry key resolves, so the caller can refuse
    rather than guess at a pool that often exceeds the KV cache.
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

    # Gated delta net (Qwen3-Next, Qwen3.5).
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

    # Kimi delta attention (Ling-3.0, Kimi-Linear): q, k and v share one head
    # count and head_dim, with a short conv over all three.
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

    # Short convolution (LFM2): a conv state and nothing else.
    l_cache = take(("conv_L_cache",))
    if l_cache:
        width = take(("conv_dim",)) or hidden
        return {"kind": "short_conv", "num_state_heads": 0,
                "conv_dim": int(width), "conv_kernel": int(l_cache),
                "state_inner": int(width), "state_source": seen}

    # Mamba / Mamba2 (Nemotron-H, Bamba, Falcon-H1, Jamba, Codestral-Mamba,
    # Zamba2).
    d_state = take(("mamba_d_state", "state_size", "ssm_state_size", "d_state"))
    conv_kernel = take(("mamba_d_conv", "conv_kernel", "d_conv"))
    if not (d_state and conv_kernel):
        return None

    n_heads = take(("mamba_n_heads", "mamba_num_heads", "n_mamba_heads"))
    head_dim = take(("mamba_d_head", "mamba_head_dim", "mamba_headdim"))
    if pure_ssm:
        n_heads = n_heads or take(("num_heads",))
        head_dim = head_dim or take(("head_dim",))

    # A stated head split is the width vLLM allocates, whatever the expansion
    # factor says (Nemotron-3-Nano).
    inner = take(("mamba_d_ssm", "mamba_expand_inner_size"))
    if not inner and pure_ssm:
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
    groups = take(("mamba_n_groups", "n_groups", "mamba_ngroups")) or 1
    if not (n_heads and head_dim):
        # Mamba1: one (d_inner, d_state) state per layer, no head split, and a
        # conv over the inner stream alone. dt_rank sizes its x_proj/dt_proj.
        dt_rank = take(("time_step_rank", "mamba_dt_rank", "dt_rank"))
        if not isinstance(dt_rank, int):
            dt_rank = -(-int(hidden) // 16)
        return {"kind": "mamba1", "num_state_heads": 1,
                "state_head_dim_k": int(d_state), "state_head_dim_v": int(inner),
                "conv_dim": int(inner), "conv_kernel": int(conv_kernel),
                "state_groups": 1, "state_inner": int(inner),
                "dt_rank": int(dt_rank), "state_source": seen}

    return {
        "kind": "mamba2",
        "num_state_heads": int(n_heads),
        "state_head_dim_k": int(d_state),
        "state_head_dim_v": int(head_dim),
        "conv_dim": int(inner) + 2 * int(groups) * int(d_state),
        "conv_kernel": int(conv_kernel),
        "state_groups": int(groups),
        "state_inner": int(inner),
        "state_source": seen,
    }


@dataclass(frozen=True)
class ModelDims:
    family: str
    hidden: int
    num_layers: int
    num_attn_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied: bool
    is_moe: bool
    # Layer layout. The three attention kinds cost different amounts, so the
    # cache is charged per layer rather than per model.
    num_full_attn_layers: int = 0
    num_sliding_attn_layers: int = 0
    num_recurrent_layers: int = 0
    sliding_window: int = 0
    window_is_chunked: bool = False
    attn_on_recurrent_layers: bool = False
    # `architectures[0]`, which SGLang keys its window pool on.
    architecture: str = ""
    # What vLLM's startup profiling pass runs: the routed experts' top-k
    # (their width is `intermediate`) and, for a VLM, the tower's input.
    num_experts_per_tok: int = 0
    vision_patches_per_image: int = 0
    vision_tokens_per_image: int = 0
    vision_intermediate: int = 0
    # Recurrent (state-space) geometry, zero for attention-only models.
    # `state_kind` picks vLLM's sharding formula: gdn, kda, mamba1, mamba2 or
    # short_conv.
    state_kind: str = ""
    num_state_heads: int = 0
    state_head_dim_k: int = 0
    state_head_dim_v: int = 0
    conv_dim: int = 0
    conv_kernel: int = 0
    state_groups: int = 0
    state_inner: int = 0
    conv_dtype: str = "bfloat16"
    ssm_dtype: str = "bfloat16"
    state_source: list[str] = field(default_factory=list)
    # Gated delta net and KDA split their heads with a plain divide, so vLLM
    # refuses a TP degree that doesn't divide them. Mamba2 has its own rule.
    state_heads_divide_tp: bool = False
    # MLA caches one kv_lora_rank + qk_rope_head_dim latent per token per
    # layer, whole on every rank. 0 for ordinary attention.
    mla_kv_dim: int = 0
    # --- Weight shapes, mirroring model-can-it-fit's ModelDims ---------------
    # On an MoE, `intermediate` is the expert width and `dense_intermediate`
    # the width of the `first_k_dense_replace` dense layers.
    num_experts: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    dense_intermediate: int = 0
    shared_intermediate: int = 0
    # 3 matrices for a gated MLP, 2 for squared-ReLU (Nemotron-H).
    ffn_matrices: int = 3
    num_ffn_layers: int = 0
    # Zamba2: hybrid layers share `num_mem_blocks` attention blocks whose
    # Q/K/V read `attn_in_dim` (hidden + embeddings) rather than hidden.
    shared_attn_blocks: int = 0
    attn_in_dim: int = 0
    # Mamba1's low-rank dt projection; 0 elsewhere.
    state_dt_rank: int = 0
    # Every encoder tower's weights (vision, and Gemma-4's audio tower).
    vision_params: int = 0
    # Tower head count: 0, or a count that does not divide TP, means the
    # tower has no valid split and every rank holds a copy.
    vision_num_heads: int = 0
    # Gemma-4 / Gemma-3n per-layer input embedding tables + projection.
    per_layer_embed_params: int = 0
    # Gemma-4: full-attention layers' head geometry (0 = same as the rest)
    # and the KV-shared layers that allocate no cache, by kind.
    full_head_dim: int = 0
    full_kv_heads: int = 0
    kv_shared_full: int = 0
    kv_shared_sliding: int = 0
    # Which module naming the tower uses; "intern" towers are profiled with
    # a video whose frame count follows the context.
    vision_naming: str = ""


def parse_model_dims(cfg: dict) -> ModelDims:
    """Parse decoder-only LLM dimensions from a Transformers config."""
    family = cfg.get("model_type", "unknown")
    arch = (cfg.get("architectures") or [""])[0]
    text_cfg = cfg.get("text_config") or cfg
    defaults = _TEXT_CONFIG_DEFAULTS.get(str(text_cfg.get("model_type", "")))
    if defaults:
        text_cfg = {**defaults, **text_cfg}
    hidden = text_cfg.get("hidden_size") or text_cfg.get("d_model")
    num_layers = (text_cfg.get("num_hidden_layers") or text_cfg.get("n_layer")
                  or text_cfg.get("num_layers"))
    num_attn = text_cfg.get("num_attention_heads") or text_cfg.get("n_head")
    num_kv = text_cfg.get("num_key_value_heads", num_attn)
    head_dim = text_cfg.get("head_dim") or cfg.get("head_dim")
    if head_dim is None:
        head_dim = hidden // num_attn if (hidden and num_attn) else 0
    vocab = text_cfg.get("vocab_size", cfg.get("vocab_size", 32000))
    tied = bool(text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))

    # MoE shape, read before the FFN width so an MoE takes its expert width
    # (fit.py's convention): num_experts / num_local_experts / n_routed_experts.
    moe_cfg = cfg.get("moe", text_cfg.get("moe", {}))
    num_experts = _lookup([text_cfg, cfg, moe_cfg], (
        "num_local_experts", "num_experts", "n_routed_experts"))[0] or 0
    experts_per_tok = _lookup([text_cfg, cfg, moe_cfg], (
        "num_experts_per_tok", "num_experts_per_token"))[0] or 0
    num_shared_experts = _lookup([text_cfg, cfg], ("n_shared_experts",))[0] \
        or moe_cfg.get("num_shared_experts", 0) or 0
    shared_intermediate = _lookup([text_cfg, cfg], (
        "moe_shared_expert_intermediate_size", "shared_expert_intermediate_size"))[0] or 0
    if shared_intermediate and not num_shared_experts:
        num_shared_experts = 1
    first_k_dense_replace = (text_cfg.get("first_k_dense_replace")
                             or cfg.get("first_k_dense_replace")
                             or moe_cfg.get("first_k_dense_replace")
                             or text_cfg.get("num_dense_layers") or 0)
    is_moe = bool(num_experts and int(num_experts) > 1)
    if is_moe:
        intermediate = (text_cfg.get("moe_intermediate_size")
                        or cfg.get("moe_intermediate_size")
                        or moe_cfg.get("expert_hidden_dim")
                        or text_cfg.get("ffn_dim") or cfg.get("ffn_dim")
                        or text_cfg.get("hidden_dim") or cfg.get("hidden_dim")
                        or text_cfg.get("intermediate_size"))
        dense_intermediate = (text_cfg.get("intermediate_size")
                              or cfg.get("intermediate_size")
                              or text_cfg.get("hidden_dim") or cfg.get("hidden_dim") or 0)
    else:
        intermediate = (text_cfg.get("intermediate_size")
                        or text_cfg.get("ffn_dim") or cfg.get("ffn_dim")
                        or text_cfg.get("hidden_dim") or cfg.get("hidden_dim"))
        dense_intermediate = 0
    if intermediate is None:
        intermediate = 4 * (hidden or 0)
    gated = str(text_cfg.get("mlp_hidden_act") or "").lower() != "relu2"

    pure_ssm = family in _PURE_SSM_FAMILIES
    if pure_ssm:
        # No attention layer to read heads for; Mamba2's head_dim is the SSM's.
        num_attn, num_kv, head_dim = 0, 0, 0
    missing = [k for k, v in {"hidden_size": hidden, "num_hidden_layers": num_layers,
                              "num_attention_heads": num_attn or pure_ssm}.items()
               if not v]
    if missing:
        sys.exit(f"config.json missing {missing}; not a decoder-only LLM.")

    hidden, num_layers = int(hidden), int(num_layers)
    window, window_is_chunked = _sliding_window(cfg, text_cfg, family)
    layers = _classify_layers(cfg, text_cfg, family, num_layers, window)
    if not layers["sliding"]:
        # The window belongs to something other than the attention layers.
        window, window_is_chunked = 0, False
    shared_attn_blocks = attn_in_dim = 0
    if family == "zamba2" and layers["attn_on_recurrent"]:
        # Zamba2's hybrid layers call `num_mem_blocks` shared attention
        # blocks, which read hidden + embeddings with their own head size.
        shared_attn_blocks = int(text_cfg.get("num_mem_blocks") or 1)
        attn_in_dim = int(text_cfg.get("attention_hidden_size") or 2 * hidden)
        head_dim = int(text_cfg.get("attention_head_dim") or head_dim)

    state = {}
    if layers["recurrent"]:
        geometry = _state_geometry(cfg, text_cfg, hidden, pure_ssm)
        if geometry is None:
            sys.exit(
                f"{family} has {layers['recurrent']} recurrent layer(s) but no "
                f"readable state geometry. Looked for mamba_d_state/state_size, "
                f"mamba_d_conv/conv_kernel, an inner width "
                f"(mamba_d_ssm, mamba_expand, mamba_intermediate_size), the "
                f"linear-attention keys (linear_num_value_heads, "
                f"linear_key_head_dim, linear_attn_config) and conv_L_cache. "
                f"Recurrent state often exceeds the KV cache, so this is not "
                f"safe to guess."
            )
        state = geometry

    dtype = _normalize_dtype(_lookup([cfg, text_cfg], ("torch_dtype", "dtype"))[0])
    # Qwen3.5 declares its state dtype as mamba_ssm_dtype, which vLLM adopts
    # when --mamba-ssm-cache-dtype is left at auto; Nemotron-H falls back to
    # float32.
    family_ssm_default = ("float32" if _FP32_SSM_STATE_FAMILIES
                          & {family, text_cfg.get("model_type")} else None)
    ssm_dtype = _normalize_dtype(
        _lookup([text_cfg, cfg], ("mamba_ssm_cache_dtype", "ssm_cache_dtype",
                                  "mamba_ssm_dtype"))[0],
        default=state.get("ssm_dtype") or family_ssm_default or dtype,
    )
    vision = cfg.get("vision_config")
    vision_patches = vision_tokens = vision_intermediate = vision_params = 0
    vision_num_heads = 0
    vision_naming = ""
    if isinstance(cfg.get("audio_config"), dict):
        vision_params += _audio_param_count(cfg["audio_config"], hidden)
    if isinstance(vision, dict) and (vision.get("embed_dim") or vision.get("hidden_size")) \
            and (vision.get("depth") or vision.get("num_hidden_layers")):
        vision_params += _vision_param_count(vision, hidden)
        vision_num_heads = int(vision.get("num_attention_heads")
                               or vision.get("num_heads") or 0)
        vtype = str(vision.get("model_type", ""))
        if family.startswith("intern") or vtype.startswith("intern"):
            vision_naming = "intern"
        elif family.startswith("qwen") or vtype.startswith("qwen"):
            vision_naming = "qwen"
        vision_patches, vision_tokens = _vision_profile_shape(cfg, vision)
        width = vision.get("embed_dim") or vision.get("hidden_size")
        ratio = vision.get("mlp_ratio")
        vision_intermediate = int(vision.get("intermediate_size")
                                  or (width * ratio if ratio else 4 * width))
    # Gemma-4: wider global heads, KV-shared tail layers, per-layer embeddings.
    full_head_dim = int(text_cfg.get("global_head_dim") or 0)
    full_kv_heads = int(text_cfg.get("num_global_key_value_heads") or 0)
    kv_shared_full = kv_shared_sliding = 0
    shared = int(text_cfg.get("num_kv_shared_layers") or 0)
    types = text_cfg.get("layer_types")
    if shared and isinstance(types, list) and len(types) == num_layers:
        tail = _layer_type_counts(types[len(types) - shared:])
        kv_shared_full = tail["full"] + tail["hybrid"]
        kv_shared_sliding = tail["sliding"]
    elif shared:
        kv_shared_full = min(shared, layers["full"])
        kv_shared_sliding = min(shared - kv_shared_full, layers["sliding"])
    ple_vocab = int(text_cfg.get("vocab_size_per_layer_input") or 0)
    ple_dim = int(text_cfg.get("hidden_size_per_layer_input") or 0)
    per_layer_embed = (ple_vocab * num_layers * ple_dim
                       + 3 * hidden * num_layers * ple_dim) if ple_dim else 0
    kv_lora_rank = _lookup([text_cfg, cfg], ("kv_lora_rank",))[0]
    mla_kv_dim = 0
    if kv_lora_rank:
        rope = _lookup([text_cfg, cfg], ("qk_rope_head_dim",))[0] or 64
        mla_kv_dim = int(kv_lora_rank) + int(rope)
    return ModelDims(
        family, hidden, num_layers, int(num_attn or 0),
        int(num_kv or 0), int(head_dim), int(intermediate),
        int(vocab), tied, is_moe,
        num_full_attn_layers=layers["full"],
        num_sliding_attn_layers=layers["sliding"],
        num_recurrent_layers=layers["recurrent"],
        sliding_window=window,
        window_is_chunked=window_is_chunked,
        attn_on_recurrent_layers=layers["attn_on_recurrent"],
        architecture=arch,
        num_experts_per_tok=int(experts_per_tok) if is_moe else 0,
        vision_patches_per_image=vision_patches,
        vision_tokens_per_image=vision_tokens,
        vision_intermediate=vision_intermediate,
        state_kind=state.get("kind", ""),
        num_state_heads=state.get("num_state_heads", 0),
        state_head_dim_k=state.get("state_head_dim_k", 0),
        state_head_dim_v=state.get("state_head_dim_v", 0),
        conv_dim=state.get("conv_dim", 0),
        conv_kernel=state.get("conv_kernel", 0),
        state_groups=state.get("state_groups", 0),
        state_inner=state.get("state_inner", 0),
        conv_dtype=dtype,
        ssm_dtype=ssm_dtype,
        state_source=state.get("state_source", []),
        state_heads_divide_tp=state.get("divides_tp", False),
        mla_kv_dim=mla_kv_dim,
        num_experts=int(num_experts) if is_moe else 0,
        num_shared_experts=int(num_shared_experts) if is_moe else 0,
        first_k_dense_replace=int(first_k_dense_replace) if is_moe else 0,
        dense_intermediate=int(dense_intermediate),
        shared_intermediate=int(shared_intermediate) if is_moe else 0,
        ffn_matrices=3 if gated else 2,
        num_ffn_layers=shared_attn_blocks if shared_attn_blocks else layers["ffn"],
        shared_attn_blocks=shared_attn_blocks,
        attn_in_dim=attn_in_dim,
        state_dt_rank=state.get("dt_rank", 0),
        vision_params=vision_params,
        vision_num_heads=vision_num_heads,
        per_layer_embed_params=per_layer_embed,
        full_head_dim=full_head_dim,
        full_kv_heads=full_kv_heads,
        kv_shared_full=kv_shared_full,
        kv_shared_sliding=kv_shared_sliding,
        vision_naming=vision_naming,
    )


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


def _recurrent_block_params(dims: ModelDims) -> int:
    """Weights of one recurrent (Mamba / Mamba2 / gated-delta-net) layer.

    Mamba2 and gated-delta-net differ in how they name their projections but
    land on the same shapes: one input projection wide enough to feed the
    conv state plus the value stream, a depthwise conv, and one output
    projection back to hidden. The per-head scalars (A_log, D, dt_bias) and
    the gated output norm are small but cheap to include.
    """
    h = dims.hidden
    if dims.state_kind == "short_conv":
        # LFM2: in_proj to B, C and x, a depthwise conv, out_proj.
        return h * 3 * dims.conv_dim + dims.conv_dim * dims.conv_kernel + dims.conv_dim * h
    if not dims.num_state_heads:
        return 0
    in_proj = h * (dims.conv_dim + dims.state_inner + 2 * dims.num_state_heads)
    conv = dims.conv_dim * dims.conv_kernel + dims.conv_dim
    out_proj = dims.state_inner * h
    scalars = 3 * dims.num_state_heads
    gate_norm = dims.state_head_dim_v * dims.num_state_heads
    if dims.state_dt_rank:
        # Mamba1 projects dt, B and C from the inner stream (x_proj), expands
        # dt back (dt_proj), and keeps A and D per channel rather than per
        # head -- ~4% of falcon-mamba-7b that the shared shapes above miss.
        inner, rank = dims.state_inner, dims.state_dt_rank
        x_proj = inner * (rank + 2 * dims.state_head_dim_k)
        dt_proj = rank * inner + inner
        scalars = inner * dims.state_head_dim_k + inner
        gate_norm = 0
        return in_proj + conv + out_proj + x_proj + dt_proj + scalars
    return in_proj + conv + out_proj + scalars + gate_norm


def _ffn_param_split(dims: ModelDims) -> tuple[int, int, int]:
    """(dense-FFN, shared-expert, routed-expert) parameter counts.

    Kept separate because quantization schemes routinely treat them
    differently: a compressed-tensors export commonly quantizes the routed
    experts -- where nearly all the weight is -- while leaving the dense
    replacement layers and the shared expert at full width. Charging one rate
    across all three is wrong by a factor, not a margin.
    """
    h = dims.hidden
    mats = dims.ffn_matrices
    ffn_layers = _ffn_layers(dims)
    if not dims.is_moe:
        return ffn_layers * mats * h * dims.intermediate, 0, 0

    routed_per_layer = dims.num_experts * mats * h * dims.intermediate
    shared_per_layer = (dims.num_shared_experts * mats * h
                        * (dims.shared_intermediate or dims.intermediate))

    dense_layers = 0
    if dims.first_k_dense_replace > 0 and dims.dense_intermediate > 0:
        dense_layers = min(dims.first_k_dense_replace, ffn_layers)
    moe_layers = ffn_layers - dense_layers
    dense = dense_layers * mats * h * dims.dense_intermediate
    return dense, moe_layers * shared_per_layer, moe_layers * routed_per_layer


def _attn_proj_params(dims: ModelDims) -> int:
    """Q/K/V/O projection parameter count for one attention layer.

    For models with explicit head_dim != hidden/num_heads, Q projects to
    num_attn_heads * head_dim (not hidden).
    """
    h = dims.hidden
    in_dim = dims.attn_in_dim or h
    q_proj_dim = dims.num_attn_heads * dims.head_dim
    kv_proj_dim = dims.num_kv_heads * dims.head_dim
    return (
        in_dim * q_proj_dim     # Q: input → num_attn_heads * head_dim
        + in_dim * kv_proj_dim  # K: input → num_kv_heads * head_dim
        + in_dim * kv_proj_dim  # V: input → num_kv_heads * head_dim
        + q_proj_dim * h        # O: num_attn_heads * head_dim → hidden
    )


def _attn_proj_params_total(dims: ModelDims) -> int:
    """Q/K/V/O parameters over every attention layer that owns weights.

    Gemma-4's full-attention layers project to the wider global head
    (global_head_dim 512 against 256: Q/O twice as wide) and may carry their
    own KV head count, so the two kinds are priced separately when they
    differ. Everywhere else this is layers x _attn_proj_params.
    """
    layers = _attn_weight_layers(dims)
    if not dims.full_head_dim or dims.shared_attn_blocks:
        return layers * _attn_proj_params(dims)
    full = min(dims.num_full_attn_layers, layers)
    wide = _attn_proj_params(replace(dims, head_dim=dims.full_head_dim,
                                     num_kv_heads=dims.full_kv_heads or dims.num_kv_heads))
    return full * wide + (layers - full) * _attn_proj_params(dims)


def _attn_weight_layers(dims: ModelDims) -> int:
    """Attention blocks that exist as weights.

    Equal to the attention layer count, except where layers share blocks:
    Zamba2's hybrid layers each keep a KV cache but call one of
    `num_mem_blocks` shared attention + MLP blocks, so pricing a block per
    hybrid layer would charge 13 copies of weights that exist twice.
    """
    return dims.shared_attn_blocks or _attn_layer_count(dims)


def _ffn_layers(dims: ModelDims) -> int:
    """Layer count carrying a feed-forward block; 0 for a pure SSM."""
    if dims.num_ffn_layers or dims.num_recurrent_layers:
        return dims.num_ffn_layers
    return dims.num_layers


def moe_layer_count(dims: ModelDims) -> int:
    """Layers that carry expert FFNs (all of them unless a hybrid MoE)."""
    if not dims.is_moe:
        return 0
    if dims.first_k_dense_replace > 0 and dims.dense_intermediate > 0:
        return dims.num_layers - dims.first_k_dense_replace
    return dims.num_layers


def router_params(dims: ModelDims) -> int:
    """Gate/router projection params: hidden * num_experts per MoE layer.

    One linear layer per MoE layer, scoring every routed expert. Tiny next to
    the experts themselves (0.04% of DeepSeek-V4-Flash), but 16x a
    `2 * hidden` placeholder on a 32-expert model -- and that matters when
    `modules_to_not_convert` holds the routers at bf16 while everything else
    is 4-bit, because the bf16 uplift is priced on this count.

    Shared experts are always active and have no gate, so they are not scored.
    """
    if not dims.is_moe or not dims.num_experts:
        return 0
    return moe_layer_count(dims) * dims.hidden * dims.num_experts


def _attn_layer_count(dims: ModelDims) -> int:
    """Attention layers as a single count (the weight formulas' view)."""
    full, sliding = _attention_layers(dims)
    return full + sliding


def count_params(dims: ModelDims) -> int:
    """Parameter count from config shape, vision tower included.

    The same model as model-can-it-fit's count_params (held to parity by
    tests/test_fit.py::TestRecommenderParity): attention blocks over the
    layers that have them, recurrent projections over the SSM layers, dense /
    shared / routed expert FFNs over the layers that carry each, routers,
    embeddings, an untied head, and the vision tower.
    """
    h = dims.hidden
    total = _attn_proj_params_total(dims) + _attn_weight_layers(dims) * 2 * h
    total += dims.num_recurrent_layers * (_recurrent_block_params(dims) + 2 * h)
    if dims.shared_attn_blocks:
        total += dims.num_full_attn_layers * h * h
    total += sum(_ffn_param_split(dims))
    total += _ffn_layers(dims) * 2 * h
    total += router_params(dims)
    emb = h * dims.vocab + dims.per_layer_embed_params
    head = 0 if dims.tied else dims.vocab * h
    return emb + head + total + dims.vision_params


def vision_replicated(dims: ModelDims, tp: int) -> bool:
    """Whether every rank holds a whole copy of the encoder tower.

    Both engines tensor-parallel the tower by default, but a tower whose head
    count does not divide TP -- or that states none, as Gemma-4 does -- has no
    valid split, so each rank keeps a full copy. Mirrors model-can-it-fit's
    calculate_mixed_precision_weights.
    """
    return bool(dims.vision_params and tp > 1
                and (not dims.vision_num_heads or dims.vision_num_heads % tp))


def per_rank_weight_bytes(dims: ModelDims, params: int, bytes_per_param: float,
                          tp: int) -> float:
    """Weight bytes one rank holds: sharded, plus any replicated tower."""
    weights = params * bytes_per_param / tp
    if vision_replicated(dims, tp):
        weights += dims.vision_params * bytes_per_param * (1 - 1 / tp)
    return weights


def active_params(dims: ModelDims) -> int:
    """Parameters a decode step reads, for the roofline.

    count_params() is what sits in memory; this is what one token touches:
    every attention, recurrent and dense-FFN weight, the shared experts and
    the router, the output head, and only `num_experts_per_tok` of the
    routed experts. Not the embedding table (one row is gathered), not the
    per-layer embedding tables, and not the vision or audio towers, which a
    text token never runs. gpt-oss-20b: 20.9 B in memory, 3.6 B per token.
    """
    total = count_params(dims) - dims.vision_params - dims.per_layer_embed_params
    if not dims.tied:
        # Untied: count_params holds the table twice. The output projection is
        # read whole for the logits; the input embedding is a row gather, so
        # one copy comes off. Tied: the single copy *is* the output
        # projection, so it stays.
        total -= dims.vocab * dims.hidden
    if dims.is_moe and dims.num_experts:
        _, _, routed = _ffn_param_split(dims)
        active = min(dims.num_experts_per_tok or dims.num_experts, dims.num_experts)
        total -= routed - routed * active // dims.num_experts
    return max(int(total), 0)


def _attention_layers(dims: ModelDims) -> tuple[int, int]:
    """(full-attention, sliding-window) layer counts, with a safe fallback."""
    full = dims.num_full_attn_layers
    sliding = dims.num_sliding_attn_layers
    if not (full or sliding):
        # A ModelDims built directly by a caller instead of parsed.
        full = dims.num_layers if not dims.num_recurrent_layers else 0
    return full, sliding


def _kv_elements(dims: ModelDims, full: bool) -> int:
    """KV elements one token stores in one layer of the given kind."""
    if dims.mla_kv_dim:
        return dims.mla_kv_dim
    heads, dim = dims.num_kv_heads, dims.head_dim
    if full and dims.full_head_dim:
        heads, dim = (dims.full_kv_heads or heads), dims.full_head_dim
    return 2 * heads * dim


def _kv_layer_counts(dims: ModelDims) -> tuple[int, int]:
    """(full, sliding) layers that allocate a KV cache of their own."""
    full, sliding = _attention_layers(dims)
    return (max(full - dims.kv_shared_full, 0),
            max(sliding - dims.kv_shared_sliding, 0))


def kv_bytes_per_token(dims: ModelDims, kv_dtype_bytes: float) -> int:
    """Cache bytes each additional context token adds, across all layers.

    This is the roofline's per-token read cost, so it counts every layer that
    caches at all and ignores the window bound: it answers "how much more does
    one more token cost", not "how big is the pool". Use kv_bytes() for pool
    size. Recurrent and KV-shared layers are excluded: their cost does not
    grow with context.
    """
    full, sliding = _kv_layer_counts(dims)
    return int((full * _kv_elements(dims, True)
                + sliding * _kv_elements(dims, False)) * kv_dtype_bytes)


def _kv_elements_per_token_per_layer(dims: ModelDims) -> int:
    return _kv_elements(dims, False)


def kv_shards(dims: ModelDims, tp: int) -> int:
    """How many ways the KV cache actually splits across `tp` ranks.

    Each rank holds max(1, kv_heads // tp) KV heads, so with fewer KV heads
    than ranks vLLM replicates them. MLA's latent is not split at all.
    """
    if tp <= 1 or dims.mla_kv_dim:
        return 1
    kv = max(int(dims.num_kv_heads or 1), 1)
    return max(1, kv // max(1, kv // tp))


def tp_problems(dims: ModelDims, tp: int) -> list[str]:
    """Reasons vLLM would refuse to start this model at `tp`, or [].

    Mirrors model-can-it-fit's head checks. Its FFN-width check is left out:
    intermediate_size here is not the expert width on an MoE, so applying it
    would reject TP degrees that launch.
    """
    if tp <= 1:
        return []
    problems = []
    if dims.num_attn_heads % tp:
        problems.append(f"{dims.num_attn_heads} attention heads do not divide "
                        f"evenly across TP={tp}")
    kv = dims.num_kv_heads
    if not dims.mla_kv_dim and kv and (kv % tp if kv >= tp else tp % kv):
        problems.append(f"{kv} KV heads neither divide TP={tp} nor are "
                        f"divisible by it")
    if dims.num_recurrent_layers and dims.state_heads_divide_tp:
        for count, what in ((dims.num_state_heads, "linear-attention heads"),
                            (dims.state_groups, "linear-attention key heads")):
            if count % tp:
                problems.append(f"{count} {what} do not divide evenly "
                                f"across TP={tp}")
                break
    if dims.num_recurrent_layers and dims.state_kind == "mamba2":
        # MambaMixer2.__init__ asserts `n_groups % tp == 0 or n_groups == 1`;
        # the shape calculator's group padding never gets to run otherwise.
        if dims.num_state_heads % tp:
            problems.append(f"{dims.num_state_heads} Mamba2 heads do not "
                            f"divide evenly across TP={tp}")
        elif dims.state_groups % tp and dims.state_groups != 1:
            problems.append(f"{dims.state_groups} Mamba2 groups neither divide "
                            f"TP={tp} nor are a single group")
    if (dims.num_recurrent_layers and dims.state_kind in ("mamba1", "short_conv")
            and dims.conv_dim % tp):
        problems.append(f"recurrent width {dims.conv_dim} does not divide "
                        f"evenly across TP={tp}")
    return problems


def default_max_num_batched_tokens(device_vram_gb: float) -> int:
    """vLLM's default --max-num-batched-tokens, from its device-memory tiers."""
    if device_vram_gb >= VLLM_LARGE_DEVICE_GIB:
        return VLLM_MAX_NUM_BATCHED_TOKENS_LARGE
    return VLLM_MAX_NUM_BATCHED_TOKENS_SMALL


def max_in_flight_tokens(device_vram_gb: float) -> int:
    """Tokens vLLM keeps addressable beyond a sliding window: two batches."""
    return 2 * default_max_num_batched_tokens(device_vram_gb)


def sliding_ctx(dims: ModelDims, ctx: int, runtime: str,
                in_flight_tokens: int) -> int:
    """Tokens one sliding-window layer must hold per request.

    vLLM pages the window and cannot free a block until every token in it has
    left the window, so it holds `window - 1` plus the tokens still in flight
    (a chunked-local layer: the whole chunk plus them). SGLang sizes the
    window pool as a fraction of the full-attention pool, and only for the
    architectures it runs as hybrid SWA. PyTorch allocates exactly the window.
    """
    window = dims.sliding_window
    if not window:
        return ctx
    if runtime.startswith("vllm"):
        held = window if dims.window_is_chunked else window - 1
        return min(ctx, held + in_flight_tokens)
    if runtime.startswith("sglang"):
        if dims.architecture not in SGLANG_HYBRID_SWA_ARCHS:
            return ctx
        pool = int(SGLANG_SWA_FULL_TOKENS_RATIO * ctx)
        return min(ctx, max(window, pool))
    return min(ctx, window)


def kv_bytes(dims: ModelDims, ctx: int, concurrency: int,
             kv_dtype_bytes: float, runtime: str = "vllm-xpu",
             in_flight_tokens: int = 0, block_tokens: int = 0) -> int:
    """KV-cache pool bytes before the tensor-parallel divide.

    Charged per layer: a sliding-window layer holds a bounded number of tokens
    however long the context is, a recurrent layer holds no KV cache, a
    KV-shared layer reuses another's, and a full-attention layer may use a
    wider head (Gemma-4). `block_tokens` rounds each request up to whole
    blocks (vLLM's enlarged hybrid block); 0 skips the rounding.
    """
    full, sliding = _kv_layer_counts(dims)
    full_ctx = ctx
    windowed = sliding_ctx(dims, ctx, runtime, in_flight_tokens) if sliding else 0
    if block_tokens:
        full_ctx = _ceil_div(ctx, block_tokens) * block_tokens
        if sliding and windowed < ctx:
            windowed = _ceil_div(windowed, block_tokens) * block_tokens + block_tokens
        windowed = min(windowed, full_ctx)
    elements = (full * full_ctx * _kv_elements(dims, True)
                + sliding * windowed * _kv_elements(dims, False))
    return int(elements * kv_dtype_bytes * concurrency)


def _state_layer_elements(dims: ModelDims, tp: int) -> tuple[int, int]:
    """(conv, ssm) element counts one recurrent layer holds on one rank."""
    tp = max(tp, 1)
    columns = max(dims.conv_kernel - 1, 1)
    if dims.state_kind == "mamba2":
        groups = dims.state_groups or 1
        if groups % tp:
            groups = tp   # a single group is replicated to every rank
        conv_dim = dims.state_inner + 2 * groups * dims.state_head_dim_k
        conv = _ceil_div(conv_dim, tp) * columns
        ssm = (_ceil_div(dims.num_state_heads, tp)
               * dims.state_head_dim_v * dims.state_head_dim_k)
        return conv, ssm
    if dims.state_kind == "mamba1":
        inner = _ceil_div(dims.state_inner, tp)
        return inner * columns, inner * dims.state_head_dim_k
    if dims.state_kind == "short_conv":
        return _ceil_div(dims.conv_dim, tp) * columns, 0
    conv = _ceil_div(dims.conv_dim, tp) * columns
    ssm = (_ceil_div(dims.num_state_heads, tp)
           * dims.state_head_dim_k * dims.state_head_dim_v)
    return conv, ssm


def _state_layer_bytes(dims: ModelDims, tp: int, runtime: str) -> int:
    conv, ssm = _state_layer_elements(dims, tp)
    ssm_dtype = "float32" if runtime.startswith("sglang") else dims.ssm_dtype
    return int(conv * BYTES_PER_STATE_DTYPE[dims.conv_dtype]
               + ssm * BYTES_PER_STATE_DTYPE[ssm_dtype])


def state_page_bytes(dims: ModelDims, tp: int, runtime: str = "vllm-xpu") -> int:
    """Bytes of one unpadded copy of one request's recurrent state, per rank.

    Summed over the recurrent layers. Allocated per running request and never
    shrunk with context. Neither engine quantizes it; SGLang forces the SSM
    half to fp32. What vLLM reserves is more: see state_reserved_bytes().
    """
    if not (dims.num_recurrent_layers and (dims.num_state_heads or dims.conv_dim)):
        return 0
    return _state_layer_bytes(dims, tp, runtime) * dims.num_recurrent_layers


def _attn_bytes_per_token_per_layer(dims: ModelDims, tp: int,
                                    kv_dtype_bytes: float) -> int:
    if dims.mla_kv_dim:
        return int(dims.mla_kv_dim * kv_dtype_bytes)
    kv_per_rank = max(1, int(dims.num_kv_heads or 1) // max(tp, 1))
    return int(2 * kv_per_rank * dims.head_dim * kv_dtype_bytes)


def vllm_block_tokens(dims: ModelDims, tp: int, kv_dtype_bytes: float) -> int:
    """vLLM's attention block size for a hybrid model, or 0 if not hybrid."""
    full, sliding = _attention_layers(dims)
    if not (dims.num_recurrent_layers and (full or sliding)):
        return 0
    page = _state_layer_bytes(dims, tp, "vllm-xpu")
    per_token = _attn_bytes_per_token_per_layer(dims, tp, kv_dtype_bytes)
    if not (page and per_token):
        return 0
    align = VLLM_HYBRID_BLOCK_ALIGN
    return align * _ceil_div(page, align * per_token)


def state_reserved_bytes(dims: ModelDims, tp: int, runtime: str = "vllm-xpu",
                         kv_dtype_bytes: float = 2.0) -> int:
    """Recurrent-state bytes a runtime reserves per running request, per rank.

    vLLM pads each state page to one attention block and, with its default
    prefix caching, holds two pages per recurrent layer on a hybrid model;
    an attention-free model holds one unpadded page. SGLang holds one copy.
    """
    raw = state_page_bytes(dims, tp, runtime)
    if not raw or not runtime.startswith("vllm"):
        return raw
    block = vllm_block_tokens(dims, tp, kv_dtype_bytes)
    if not block:
        return raw
    padded = block * _attn_bytes_per_token_per_layer(dims, tp, kv_dtype_bytes)
    return padded * VLLM_ALIGN_MODE_STATE_PAGES * dims.num_recurrent_layers


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


def vision_profile_patches(dims: ModelDims, batched_tokens: int,
                           ctx: int = 0) -> int:
    """Patches vLLM's encoder profiling pass runs through the vision tower.

    A fixed-size tower is profiled with as many images as one batch of
    tokens holds -- or, when the family also takes video (InternVL), with one
    video of as many frames as fit the context: 31 frames of 256 tokens at
    `--max-model-len 8192` on InternVL3-2B, 7,936 encoder tokens, measured.
    """
    if not (dims.vision_params and dims.vision_patches_per_image):
        return 0
    if not dims.vision_tokens_per_image:
        return dims.vision_patches_per_image
    budget = max(batched_tokens, dims.vision_tokens_per_image)
    if dims.vision_naming == "intern" and ctx:
        budget = max(budget, ctx - dims.vision_tokens_per_image)
    return max(1, budget // dims.vision_tokens_per_image) * dims.vision_patches_per_image


def vllm_reservation(dims: ModelDims, bytes_per_param: float,
                     device_vram_gb: float,
                     batched_tokens: int | None = None,
                     ctx: int = 0) -> tuple[int, int, int]:
    """(runtime floor, held encoder buffers, profile peak) vLLM reserves.

    Fixed at startup by one profiling pass -- a batch of
    max_num_batched_tokens and, for a VLM, a maximum-size image -- so it
    never depends on the context or concurrency later served.
    """
    batched = batched_tokens or default_max_num_batched_tokens(device_vram_gb)
    floor = VLLM_RUNTIME_FLOOR_GB
    if bytes_per_param < 2.0:
        floor += VLLM_QUANT_WORKSPACE_GB
    dense_width = (dims.dense_intermediate if dims.is_moe else dims.intermediate) or 0
    dense = batched * VLLM_DENSE_PEAK_WIDTHS * max(dense_width, dims.conv_dim) * 2
    moe = (batched * dims.num_experts_per_tok * VLLM_MOE_PEAK_WIDTHS
           * dims.intermediate * 2 if dims.is_moe else 0)
    patches = vision_profile_patches(dims, batched, ctx)
    vision = int(patches * VLLM_VISION_PEAK_WIDTHS * dims.vision_intermediate * 2)
    held = patches * VLLM_VISION_HELD_BYTES_PER_PATCH
    return int(floor * GiB), int(held), int(max(dense, moe, vision))


def usable_bytes(device_vram_gb: float, gpu_memory_utilization: float) -> int:
    """vLLM's share: gpu_memory_utilization of what the driver reports."""
    return int(device_vram_gb * GiB * XPU_ALLOCATABLE_FRACTION * gpu_memory_utilization)


def docker_image_id(image: str) -> str:
    try:
        return subprocess.check_output(
            ["docker", "inspect", "--format={{.Id}}", image],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return image


def docker_container_exists(name: str) -> bool:
    out = subprocess.check_output(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        text=True,
    )
    return any(line.strip() == name for line in out.splitlines())
