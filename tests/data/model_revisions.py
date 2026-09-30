#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pinned Hugging Face revisions for the models `tests/test_fit.py` measures,
plus the on-disk cache layout shared by the test suite and the pre-fetch
helper (`fetch_configs.py`).

Why this file exists
--------------------
The repository does not redistribute upstream `config.json` files, and must not
start. Each one is a third-party file arriving under its model's licence
(Apache-2.0, MIT, the Llama 3.x Community Licence, or the Gemma Terms of Use).
Committing copies would make this project a redistributor and pull those inbound
obligations -- licence copies, modification notices, attribution, notice-file
text, use-policy pass-through -- into the release artifact, to get unit tests to
pass. There is no engineering benefit that trades against that.

So the suite fetches what it needs at run time and caches it under
`tests/data/.cache/configs/`, which `.gitignore` excludes (`.cache/` matches at
any depth). `tests/data/configs/` -- where the fixtures used to live -- is
ignored by name as well, so a stray re-fetch cannot be committed back into the
old location.

A config that cannot be fetched is always a *skip*, never a failure. Gated repo,
no `HF_TOKEN`, no network, HTTP 429 rate limit -- none of those say anything
about the calculator under test, so none of them should redden a run. If you are
looking at a skipped test, the fix is a token or a network route, never a
committed config.

Why the revisions are pinned
----------------------------
`test_fit.py` asserts on exact model dimensions (hidden size, head_dim,
num_key_value_heads, expert counts). Tracking `main` would let an upstream
config edit silently change what those assertions measure -- the failure would
look like a calculator regression. A pinned commit makes the fetched bytes
reproducible.

Refreshing a pin
----------------
Revisions are repository commit SHAs. The metadata endpoint is public even for
gated repositories, so no token is needed to resolve one:

    curl -s https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'

Bump the SHA here, re-run the suite, and fix up any dimension assertion that
legitimately changed.

What each model is here to cover
-------------------------------
This module is the single source of truth: both `test_fit.py` and
`fetch_configs.py` read it, so the two cannot drift on which models are needed.
Each entry earns its place by exercising a distinct calculator path -- do not
drop one without checking what stops being covered.

    Qwen3-30B-A3B, Qwen3-235B-A22B   MoE; moe_intermediate_size priority, head_dim
    Qwen2.5-7B, Qwen2.5-14B          dense regression baseline
    Mixtral-8x7B                     MoE with standard field names
    DeepSeek-V4-Flash                MoE with shared experts
    gpt-oss-20b, gpt-oss-120b        pre-quantized mxfp4, mixed-precision path
    Llama-3.1-8B, Llama-3.3-70B      dense; matched against the HF KV calculator
    Gemma-2-9B, Gemma-2-27B          dense with an explicit head_dim=256
    Qwen2-VL-7B, Qwen2.5-VL-7B       VLM; ViT width from embed_dim, gated tower
    Qwen2-VL-7B-AWQ                  VLM whose exclusion list is ["visual"]
    Qwen2.5-VL-7B-AWQ                VLM whose vision_config has no depth field

Expected results are recorded in the "Layer 3a" section of HOW_TO_TEST.md.
"""

from pathlib import Path

# model id -> pinned commit SHA. Comments record the revision's lastModified
# date as reported by the Hub when the pin was taken (2026-08-18).
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct":           "a09a35458c702b33eeacc393d103063234e8bc28",  # 2025-01-12
    "Qwen/Qwen2.5-14B-Instruct":          "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",  # 2024-09-25
    "Qwen/Qwen3-30B-A3B-Instruct-2507":   "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",  # 2025-09-17
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "ac9c66cc9b46af7306746a9250f23d47083d689e",  # 2025-09-17
    "deepseek-ai/DeepSeek-V4-Flash":      "60d8d70770c6776ff598c94bb586a859a38244f1",  # 2026-06-22
    "mistralai/Mixtral-8x7B-v0.1":        "fc7ac94680e38d7348cfa806e51218e6273104b0",  # 2025-07-24
    "openai/gpt-oss-20b":                 "6cee5e81ee83917806bbde320786a8fb61efebee",  # 2025-08-26
    "openai/gpt-oss-120b":                "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",  # 2025-08-26

    # Gated: fetching config.json returns HTTP 401 without an accepted licence
    # and HF_TOKEN. Tests that need these skip when the fetch fails.
    "meta-llama/Llama-3.1-8B-Instruct":   "0e9e39f249a16976918f6564b8830bc894c89659",  # 2024-09-25
    "meta-llama/Llama-3.3-70B-Instruct":  "6f6073b423013f6a7d4d9f39144961bfbfbc386b",  # 2024-12-21
    "google/gemma-2-9b-it":               "11c9b309abf73637e4b6f9a3fa1e92e615547819",  # 2024-08-27
    "google/gemma-2-27b-it":              "aaf20e6b9f4c0fcf043f6fb2a2068419086d77b0",  # 2024-08-27

    # Pinned for TestMeasuredOnDevice, which compares each config
    # against the weight figure vLLM-XPU reported when that exact
    # revision was served (see MEASUREMENT_DATE below -- the dates in
    # these comments are the revisions' own lastModified, per the
    # convention above, not when the measurement was taken). Bumping a
    # pin here invalidates the measurement it is compared against, so
    # re-measure rather than widening the tolerance.
    "Qwen/Qwen2.5-0.5B-Instruct":         "7ae557604adf67be50417f59c2c2f167def9a775",  # 2024-09-25
    "Qwen/Qwen2.5-1.5B-Instruct":         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",  # 2024-09-25
    "Qwen/Qwen3-0.6B":                    "c1899de289a04d12100db370d81485cdf75e47ca",  # 2025-07-26
    "Qwen/Qwen3-4B":                      "1cfa9a7208912126459214e8b04321603b3df60c",  # 2025-07-26
    "Qwen/Qwen3-8B":                      "b968826d9c46dd6066d109eabc6255188de91218",  # 2025-07-26
    "Qwen/Qwen3-14B":                     "40c069824f4251a91eefaf281ebe4c544efd3e18",  # 2025-07-26
    "Qwen/Qwen3.5-35B-A3B":               "59d61f3ce65a6d9863b86d2e96597125219dc754",  # 2026-04-24
    "microsoft/Phi-4-mini-instruct":      "cfbefacb99257ffa30c83adab238a50856ac3083",  # 2025-12-10
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",  # 2025-12-03
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "6e8885a6ff5c1dc5201574c8fd700323f23c25fa",  # 2025-05-29
    "NousResearch/Hermes-3-Llama-3.1-8B": "896ea440e5a9e6070e3d8a2774daf2b481ab425b",  # 2024-09-08
    "tiiuae/Falcon3-7B-Instruct":         "1e57a0ecd176c7c139f289c60a74e57f887c3dfb",  # 2025-05-31
    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": "54641c1611fcff44fa4865626462445e0a153fc7",  # 2025-10-15
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4": "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20",  # 2024-10-18
    "Qwen/Qwen2.5-7B-Instruct-AWQ":       "b25037543e9394b818fdfca67ab2a00ecc7dd641",  # 2024-10-09
    "meta-llama/Llama-3.2-3B-Instruct":   "0cb88a4f764b7a12671c53f0838cd831a0843b95",  # 2024-10-24

    # Pinned for TestVisionTowerPricing, which compares each estimate against
    # the root-level safetensors byte total of that exact revision, and for the
    # VLM rows of TestMeasuredOnDevice, which compare against what vLLM-XPU
    # allocated when these revisions were served (VLM_MEASUREMENT_DATE below).
    "Qwen/Qwen2-VL-7B-Instruct":          "eed13092ef92e448dd6875b2a00151bd3f7db0ac",  # 2025-02-06
    "Qwen/Qwen2-VL-7B-Instruct-AWQ":      "6ec2560b0afc3a618d4acc9b8e2967d1642f463d",  # 2024-09-25
    "Qwen/Qwen2.5-VL-7B-Instruct":        "cc594898137f460bfe9f0759e9844b3ce807cfb5",  # 2025-04-06
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ":    "536a35794df8831aa814970ee8f89eff577e7718",  # 2025-04-06
}

# When the TestMeasuredOnDevice figures were taken on hardware. Kept separate
# from the revision dates above: one says what the upstream config was, the
# other says when this project measured it. A pin refresh needs both updated.
MEASUREMENT_DATE = "2026-09-18"   # 2x Intel Arc B-series, vllm/vllm-openai-xpu:latest

# The VLM rows were served later, on the same SKU and the same image. Kept as
# its own constant so a pin refresh on one set does not silently re-date the
# other.
VLM_MEASUREMENT_DATE = "2026-09-23"   # Arc Pro B70 (0xe223), device 1, same image


# Models whose config.json needs an accepted licence + HF_TOKEN to fetch.
# Informational: the fetch path skips on any 401/403 rather than consulting
# this set, so a repo becoming gated (or un-gated) upstream does not need a
# code change here.
GATED_MODELS = frozenset({
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
})

# Fetched configs land here. `.gitignore` already ignores `.cache/` at any
# depth, so nothing under this directory can be committed by accident.
CACHE_DIR = Path(__file__).resolve().parent / '.cache' / 'configs'


def revision_for(model_id):
    """Pinned revision for a model id, or 'main' if it has no pin."""
    return MODEL_REVISIONS.get(model_id, 'main')


def cache_path(model_id, revision=None):
    """Cache file for a (model, revision) pair.

    The revision is part of the filename so that bumping a pin invalidates the
    cached copy instead of silently serving the old config.
    """
    if revision is None:
        revision = revision_for(model_id)
    stem = model_id.replace('/', '__')
    return CACHE_DIR / f"{stem}@{revision[:12]}.json"
