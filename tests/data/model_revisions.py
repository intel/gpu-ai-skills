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
}

# Models whose config.json needs an accepted licence + HF_TOKEN to fetch.
# Informational: the fetch path skips on any 401/403 rather than consulting
# this set, so a repo becoming gated (or un-gated) upstream does not need a
# code change here.
GATED_MODELS = frozenset({
    "meta-llama/Llama-3.1-8B-Instruct",
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
