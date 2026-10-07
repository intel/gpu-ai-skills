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
from dataclasses import dataclass
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
    num_experts: int = 0
    num_experts_per_tok: int = 0
    shared_intermediate: int = 0
    kv_layers: int = 0
    mtp_layers: int = 0


def parse_model_dims(cfg: dict) -> ModelDims:
    """Parse decoder-only LLM dimensions from a Transformers config."""
    family = cfg.get("model_type", "unknown")
    text_cfg = cfg.get("text_config") or cfg
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

    # MoE fields live under text_config for multimodal wrappers (Qwen3.5), so
    # check there before the root.
    def pick(*keys):
        for src in (text_cfg, cfg):
            for k in keys:
                if src.get(k):
                    return src[k]
        return 0

    num_experts = int(pick("num_local_experts", "num_experts", "n_routed_experts"))
    is_moe = num_experts > 1
    if is_moe:
        intermediate = pick("moe_intermediate_size") or text_cfg.get("intermediate_size")
        shared = int(pick("shared_expert_intermediate_size")
                     or pick("n_shared_experts") * (intermediate or 0))
    else:
        intermediate = text_cfg.get("intermediate_size")
        shared = 0
    intermediate = intermediate or 4 * (hidden or 0)

    # Hybrid models (Qwen3.5, Qwen3-Next) mix in linear-attention layers that
    # keep a small fixed-size state instead of a KV cache. Sliding-window
    # layers (GPT-OSS, Gemma) do hold KV, so only linear/Mamba layers are
    # dropped: under-counting KV is the direction that reports a false fit.
    layer_types = text_cfg.get("layer_types") or []
    kv_layers = (sum(1 for t in layer_types
                     if "linear" not in str(t) and "mamba" not in str(t))
                 if layer_types else num_layers)
    mtp_layers = int(pick("mtp_num_hidden_layers", "num_nextn_predict_layers"))

    missing = [k for k, v in {"hidden_size": hidden, "num_hidden_layers": num_layers,
                              "num_attention_heads": num_attn}.items() if not v]
    if missing:
        sys.exit(f"config.json missing {missing}; not a decoder-only LLM.")
    return ModelDims(family, int(hidden), int(num_layers), int(num_attn),
                     int(num_kv), int(head_dim), int(intermediate),
                     int(vocab), tied, is_moe,
                     num_experts=num_experts if is_moe else 0,
                     num_experts_per_tok=int(pick("num_experts_per_tok")) if is_moe else 0,
                     shared_intermediate=shared,
                     kv_layers=int(kv_layers or num_layers),
                     mtp_layers=mtp_layers)


def count_params(dims: ModelDims, experts: int | None = None) -> int:
    """Estimate decoder params from config shape.

    This is the same approximation used by the recommender. It is config
    driven, so it scales to larger models without hand-entered layer counts.
    For MoE models, ``experts`` is how many routed experts to count per layer:
    all of them (the default) for memory, ``num_experts_per_tok`` for the
    params one token touches.
    """
    h = dims.hidden
    q_proj = dims.head_dim * dims.num_attn_heads
    kv_proj = dims.head_dim * dims.num_kv_heads
    attn = h * q_proj + h * kv_proj + h * kv_proj + q_proj * h
    if dims.is_moe:
        n = dims.num_experts if experts is None else experts
        router = h * dims.num_experts
        ff = n * 3 * h * dims.intermediate + 3 * h * dims.shared_intermediate + router
    else:
        ff = 3 * h * dims.intermediate
    norms = 4 * h
    emb = dims.vocab * h
    head = 0 if dims.tied else dims.vocab * h
    return emb + head + dims.num_layers * (attn + ff + norms)


def count_active_params(dims: ModelDims) -> int:
    """Params one token runs through (equals count_params for dense models)."""
    if not dims.is_moe:
        return count_params(dims)
    return count_params(dims, experts=dims.num_experts_per_tok)


def params_touched(dims: ModelDims, tokens: int) -> int:
    """Params whose weights a forward pass over ``tokens`` tokens must read.

    Dense models read every weight. An MoE layer reads only the experts the
    tokens route to: with uniform routing, ``tokens`` tokens choosing k of E
    experts touch E * (1 - (1 - k/E) ** tokens) distinct experts.
    """
    total = count_params(dims)
    if not dims.is_moe or dims.num_experts_per_tok <= 0:
        return total
    e, k = dims.num_experts, dims.num_experts_per_tok
    touched = e * (1 - (1 - k / e) ** max(tokens, 1))
    return count_params(dims, experts=min(e, max(k, round(touched))))


def kv_bytes_per_token(dims: ModelDims, kv_dtype_bytes: float) -> int:
    layers = dims.kv_layers or dims.num_layers
    return int(2 * layers * dims.num_kv_heads * dims.head_dim * kv_dtype_bytes)


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
