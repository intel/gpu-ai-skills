#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive pytest test suite for fit.py

Tests all known bugs and verifies against HuggingFace KV cache calculator.

Run from repository root:
    cd intel-gpu-ai-skills
    python3 -m pytest tests/test_fit.py -v

Options:
    -v                    # Verbose output
    -s                    # Show print statements
    -k "bug3"             # Run only tests matching keyword
    -x                    # Stop on first failure
    --tb=short            # Short traceback format

Environment variables:
    HF_TOKEN=...           # HuggingFace token; needed for the gated models
    SKILLPACK_TESTS_OFFLINE=1
                           # Never touch the network. Tests whose config is not
                           # already cached skip instead of fetching.

Verifies:
1. Bug #1: moe_intermediate_size priority (parameter counts)
2. Bug #3: head_dim read from config (KV cache size)
3. KV cache correctly divided by TP (vLLM shards by attention heads)
4. Results match HuggingFace KV cache calculator
5. Dense models unchanged (regression test)
6. Gated models skipped when their config cannot be fetched

Model configs: fetched from the Hub at pinned revisions on first use and cached
under tests/data/.cache/configs/. This repository does not redistribute
upstream config.json files -- see tests/data/model_revisions.py for why and for
the pinned-revision refresh procedure. Pre-fetch them
with `python3 tests/data/fetch_configs.py` to make later runs offline-clean.
"""

import sys
import os
import json
import re
import tempfile
from pathlib import Path
import pytest

# Never reach the network; serve only what is already cached.
OFFLINE = os.environ.get('SKILLPACK_TESTS_OFFLINE', '0') == '1'

# Find git repo root
def find_git_root():
    """Find the git repository root by walking up directories"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent

    test_file = Path(__file__).resolve()
    test_dir = test_file.parent
    repo_candidates = [test_dir.parent, test_dir]

    for candidate in repo_candidates:
        fit_path = candidate / 'plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts'
        if fit_path.exists():
            return candidate

    raise RuntimeError(f"Could not find git repo root from {test_file}")

# Add fit.py directory to path
git_root = find_git_root()
fit_py_dir = git_root / 'plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts'
sys.path.insert(0, str(fit_py_dir))

# The pinned-revision map and cache layout are shared with the pre-fetch helper
# so the two cannot drift apart on which models the suite needs.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'data'))

import fit
from fit import fetch_config as fetch_config_live, parse_dims, count_params, kv_bytes, estimate
from model_revisions import (CACHE_DIR, MEASUREMENT_DATE, cache_path,
                             revision_for)

GB = 1024 ** 3
PARAM_TOLERANCE = 0.05  # 5% tolerance

def _write_cache(path, cfg):
    """Cache a fetched config, atomically and best-effort.

    Written via a temp file in the same directory plus os.replace so a parallel
    run (pytest-xdist) or an interrupted one can never leave a half-written
    file that the next run would read back as corrupt JSON. A cache write
    failure (read-only checkout, full disk) is not a test failure -- the config
    is already in hand, so the run continues uncached.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            # Do not leave the temp file behind on any exit path, including
            # KeyboardInterrupt.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _load_config(model_id):
    """Return a model's config.json, from cache if present, else from the Hub."""
    revision = revision_for(model_id)
    path = cache_path(model_id, revision)

    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            # A corrupt or truncated cache entry must not wedge the suite.
            # Drop it and fall through to a fresh fetch.
            try:
                path.unlink()
            except OSError:
                pass

    if OFFLINE:
        raise pytest.skip.Exception(
            f"SKILLPACK_TESTS_OFFLINE=1 and {model_id}@{revision[:12]} is not "
            f"cached at {path}. Run `python3 tests/data/fetch_configs.py` with "
            f"network access first."
        )

    # fit.fetch_config builds the URL from an https://huggingface.co literal;
    # this suite adds no new network call site of its own.
    cfg = fetch_config_live(model_id, revision)
    _write_cache(path, cfg)
    return cfg


def fetch_config(model_id):
    """
    Fetch a model's config.json, skipping the calling test if it is unavailable.

    Callers get a config back or never reach the next statement, so the result
    is always bound. Every reason a config can be missing -- gated repo without
    a token, no network, Hub rate limit, offline mode -- becomes a skip rather
    than a failure, because none of them says anything about the calculator
    under test.
    """
    try:
        return _load_config(model_id)
    except SystemExit as exc:
        # fit.fetch_config calls sys.exit() on HTTP 401/403 (gated or private
        # repo) -- reasonable for a CLI, but as a library call it has to become
        # a skip. SystemExit derives from BaseException, so a bare
        # `except Exception` would let it through and error the test.
        raise pytest.skip.Exception(
            f"Config for {model_id} is not accessible (gated repo, or HF_TOKEN "
            f"missing/unaccepted licence): {exc}"
        ) from exc
    except Exception as exc:
        # Network down, DNS failure, HTTP 429 rate limit, malformed response.
        # pytest.skip.Exception derives from BaseException and so is not caught
        # here -- an inner skip (offline mode) propagates unchanged.
        raise pytest.skip.Exception(
            f"Config for {model_id} not available: {type(exc).__name__}: {exc}"
        ) from exc


# Retained as an alias: fetch_config already skips rather than fails, so the
# two are now the same thing. Both names are in use across this file.
fetch_config_or_skip = fetch_config


class TestBug1_MoEIntermediateSize:
    """
    Test Bug #1 fix: moe_intermediate_size priority for parameter counts.

    Tests verify the bug is fixed by checking parameter counts are reasonable
    (not the wildly wrong values from the bug) rather than exact matches to
    model card numbers, since:
    - Model cards round parameter counts
    - Configs can change upstream
    - Exact values are less important than detecting the bug
    """

    def test_qwen3_30b_params_reasonable(self):
        """
        Qwen3-30B should have reasonable params (~20-40B).

        Bug #1 would cause ~233B (8x wrong due to wrong intermediate_size).
        """
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        params_b = count_params(parse_dims(cfg)) / 1e9

        # Reasonable range: model is called "30B", should be 20-40B
        assert 20.0 <= params_b <= 40.0, \
            f"Params {params_b:.2f}B out of reasonable range [20-40B]. " \
            f"Bug #1 NOT fixed if ~233B!"

    def test_qwen3_235b_params_reasonable(self):
        """
        Qwen3-235B should have reasonable params (~200-250B).

        Bug #1 would cause ~1821B (8x wrong due to wrong intermediate_size).
        """
        cfg = fetch_config("Qwen/Qwen3-235B-A22B-Instruct-2507")
        params_b = count_params(parse_dims(cfg)) / 1e9

        # Reasonable range: model is called "235B", should be 200-250B
        assert 200.0 <= params_b <= 250.0, \
            f"Params {params_b:.2f}B out of reasonable range [200-250B]. " \
            f"Bug #1 NOT fixed if ~1821B!"

    def test_deepseek_v4_params_reasonable(self):
        """
        DeepSeek-V4-Flash should have reasonable params (~260-300B).

        Bug #1 would cause ~11B (25x wrong due to wrong intermediate_size).
        """
        cfg = fetch_config("deepseek-ai/DeepSeek-V4-Flash")
        params_b = count_params(parse_dims(cfg)) / 1e9

        # Reasonable range: model card says ~284B, should be 260-300B
        assert 260.0 <= params_b <= 300.0, \
            f"Params {params_b:.2f}B out of reasonable range [260-300B]. " \
            f"Bug #1 NOT fixed if ~11B!"

    def test_deepseek_v4_shared_experts(self):
        """DeepSeek-V4-Flash should detect shared experts"""
        cfg = fetch_config("deepseek-ai/DeepSeek-V4-Flash")
        d = parse_dims(cfg)
        assert d.num_shared_experts == 1, \
            f"Expected 1 shared expert, got {d.num_shared_experts}"


class TestBug4_AttentionProjectionDimensions:
    """
    Test that Q and O projections use correct dimensions when head_dim differs from calculated.

    Bug: Previously used h×h for Q and O, which undercounts when head_dim ≠ hidden/num_attn_heads.
    Fix: Use h×(num_attn_heads×head_dim) for Q and (num_attn_heads×head_dim)×h for O.
    """

    def test_qwen3_attention_proj_dimensions(self):
        """Qwen3-30B: head_dim=128 vs calculated=64, Q/O should use correct dimensions"""
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        d = parse_dims(cfg)

        # Qwen3-30B: hidden=2048, num_attn_heads=32, head_dim=128 (vs calculated=64)
        calculated_head_dim = d.hidden // d.num_attn_heads
        assert d.head_dim > calculated_head_dim, \
            f"Test assumes head_dim ({d.head_dim}) > calculated ({calculated_head_dim})"

        # Q projection should be h × (num_attn_heads × head_dim)
        q_proj_expected = d.hidden * (d.num_attn_heads * d.head_dim)
        # Not h × h (old wrong formula)
        q_proj_wrong = d.hidden * d.hidden

        assert q_proj_expected > q_proj_wrong, \
            f"Q projection with correct head_dim should be larger than h×h"

        # Verify parameter count is reasonable (not undercounted)
        params_b = count_params(d) / 1e9
        assert 25.0 <= params_b <= 35.0, \
            f"Params {params_b:.2f}B out of range - may be using wrong Q/O dimensions"

    def test_gemma2_attention_proj_dimensions(self):
        """Gemma-2: head_dim=256 vs calculated=224, should use correct dimensions"""
        cfg = fetch_config("google/gemma-2-9b-it")
        d = parse_dims(cfg)

        # Gemma-2-9B: head_dim=256 (explicit) vs calculated
        calculated_head_dim = d.hidden // d.num_attn_heads
        assert d.head_dim != calculated_head_dim, \
            f"Test assumes head_dim ({d.head_dim}) differs from calculated ({calculated_head_dim})"

        # Calculate what params would be with correct vs wrong formula
        # (We can't easily test the exact value without reimplementing count_params,
        # but we can verify reasonable range)
        params_b = count_params(d) / 1e9
        assert 8.0 <= params_b <= 11.0, \
            f"Params {params_b:.2f}B out of range for Gemma-2-9B"

    def test_attention_proj_ratio_correctness(self):
        """When head_dim=2×calculated, count_params() should use correct Q/O dimensions"""
        # Create synthetic config where head_dim is exactly 2x calculated
        cfg = {
            'hidden_size': 2048,
            'num_hidden_layers': 12,
            'num_attention_heads': 32,
            'num_key_value_heads': 4,
            'head_dim': 128,  # 2× calculated (2048/32=64)
            'intermediate_size': 6144,
            'vocab_size': 32000,
            'tie_word_embeddings': False
        }

        d = parse_dims(cfg)

        # Independently compute expected parameters with correct formula
        h = d.hidden
        q_proj_dim = d.num_attn_heads * d.head_dim
        kv_proj_dim = d.num_kv_heads * d.head_dim

        # Per-layer attention block
        attn_correct = (
            h * q_proj_dim      # Q: 2048 × (32×128) = 2048 × 4096
            + h * kv_proj_dim   # K: 2048 × (4×128) = 2048 × 512
            + h * kv_proj_dim   # V: 2048 × (4×128) = 2048 × 512
            + q_proj_dim * h    # O: (32×128) × 2048 = 4096 × 2048
        )
        ff_block = 3 * h * d.intermediate
        norms = 4 * h
        per_layer_expected = attn_correct + ff_block + norms

        # Total expected params
        emb = d.vocab * h
        head = 0 if d.tied else d.vocab * h
        expected_params = emb + head + d.num_layers * per_layer_expected

        # Actual params from count_params()
        actual_params = count_params(d)

        # Should match exactly (or very close due to any rounding)
        diff_ratio = abs(actual_params - expected_params) / expected_params
        assert diff_ratio < 0.001, \
            f"count_params() should match expected calculation: " \
            f"got {actual_params:,}, expected {expected_params:,} ({diff_ratio*100:.2f}% diff)"

        # Also verify the ratio test: with 2× head_dim, Q+O should be 2× larger than h×h
        q_plus_o_correct = h * q_proj_dim + q_proj_dim * h
        q_plus_o_wrong = h * h + h * h
        ratio = q_plus_o_correct / q_plus_o_wrong
        assert abs(ratio - 2.0) < PARAM_TOLERANCE, \
            f"Q+O should be 2× with 2× head_dim, got {ratio:.2f}×"


class TestBug3_HeadDimFromConfig:
    """Test Bug #3 fix: head_dim read from config (not calculated)"""

    def test_qwen3_30b_head_dim_explicit(self):
        """Qwen3-30B has explicit head_dim=128 in config (not 64 calculated)"""
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        d = parse_dims(cfg)

        # Config has head_dim: 128
        assert d.head_dim == 128, \
            f"Expected head_dim=128 from config, got {d.head_dim} (Bug #3 NOT fixed if 64!)"

        # NOT the calculated value (hidden / num_attn_heads = 2048 / 32 = 64)
        calculated = d.hidden // d.num_attn_heads
        assert d.head_dim != calculated, \
            f"head_dim should be from config (128), not calculated ({calculated})"

    def test_gemma2_9b_head_dim_explicit(self):
        """Gemma-2-9B has explicit head_dim=256 in config (not 224 calculated)"""
        cfg = fetch_config("google/gemma-2-9b-it")
        d = parse_dims(cfg)

        # Config has head_dim: 256
        assert d.head_dim == 256, \
            f"Expected head_dim=256 from config, got {d.head_dim}"

        # NOT the calculated value (hidden / num_attn_heads = 3584 / 16 = 224)
        calculated = d.hidden // d.num_attn_heads
        assert d.head_dim != calculated, \
            f"head_dim should be from config (256), not calculated ({calculated})"

    def test_qwen25_7b_head_dim_calculated(self):
        """Qwen2.5-7B has no explicit head_dim, should calculate it"""
        cfg = fetch_config("Qwen/Qwen2.5-7B-Instruct")
        d = parse_dims(cfg)

        # Should calculate: hidden / num_attn_heads = 3584 / 28 = 128
        expected = d.hidden // d.num_attn_heads
        assert d.head_dim == expected, \
            f"Expected calculated head_dim={expected}, got {d.head_dim}"

    def test_head_dim_affects_kv_cache(self):
        """Verify head_dim affects KV cache calculation (relationship test)"""
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        d = parse_dims(cfg)

        ctx = 32768
        concurrency = 8
        kv_dtype = "bf16"

        # Calculate KV cache with correct head_dim from config
        kv_correct = kv_bytes(d, ctx, concurrency, kv_dtype)

        # Calculate what it would be using calculated head_dim (hidden / num_attn_heads)
        # For Qwen3-30B: 2048 / 32 = 64, but config has explicit head_dim=128
        calculated_head_dim = d.hidden // d.num_attn_heads

        # Simulate KV cache with calculated head_dim
        # per_token = 2 * layers * kv_heads * head_dim
        per_token_calculated = 2 * d.num_layers * d.num_kv_heads * calculated_head_dim
        kv_calculated = per_token_calculated * ctx * concurrency * fit.BYTES_PER_KV[kv_dtype]

        # The ratio should match head_dim ratio (128/64 = 2x for this model)
        ratio = kv_correct / kv_calculated
        expected_ratio = d.head_dim / calculated_head_dim

        assert abs(ratio - expected_ratio) < PARAM_TOLERANCE, \
            f"KV cache should scale with head_dim ratio: " \
            f"head_dim={d.head_dim} vs calculated={calculated_head_dim}, " \
            f"expected ratio {expected_ratio:.2f}x, got {ratio:.2f}x"


class TestKVCacheSharding:
    """Test that KV cache IS divided by TP (correct vLLM behavior)"""

    @pytest.mark.parametrize("model_id", [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mixtral-8x7B-v0.1",
    ])
    def test_kv_cache_divided_by_tp(self, model_id):
        """KV cache should be divided by TP (sharded by attention heads)"""
        cfg = fetch_config(model_id)

        ctx = 32768
        concurrency = 8
        kv_dtype = "bf16"

        # Calculate KV cache for different TP values
        result_tp1 = estimate(cfg, "bf16", kv_dtype, ctx, concurrency, 1, "vllm", 32.0)
        result_tp4 = estimate(cfg, "bf16", kv_dtype, ctx, concurrency, 4, "vllm", 32.0)

        kv_tp1 = result_tp1['kv']
        kv_tp4 = result_tp4['kv']

        # KV cache should be 4x smaller at TP=4
        ratio = kv_tp1 / kv_tp4
        assert abs(ratio - 4.0) < PARAM_TOLERANCE, \
            f"KV cache should be divided by TP! TP=1: {kv_tp1/GB:.2f}GB, TP=4: {kv_tp4/GB:.2f}GB, ratio: {ratio:.1f}x (expected 4x)"

    def test_kv_cache_formula_with_tp(self):
        """KV cache formula should include TP division"""
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        d = parse_dims(cfg)

        ctx = 16384
        concurrency = 4
        kv_dtype = "bf16"
        tp = 4
        bytes_per = 2

        # Calculate expected KV cache WITH TP division
        kv_total = 2 * d.num_layers * d.num_kv_heads * d.head_dim * ctx * concurrency * bytes_per
        kv_expected_per_gpu = kv_total // tp  # Divided by TP

        # Get actual from estimate
        result = estimate(cfg, "bf16", kv_dtype, ctx, concurrency, tp, "vllm", 32.0)
        kv_actual = result['kv']

        assert kv_actual == kv_expected_per_gpu, \
            f"KV cache calculation wrong! Expected {kv_expected_per_gpu/GB:.2f}GB, got {kv_actual/GB:.2f}GB"


class TestHFCalculatorMatch:
    """
    Test that fit.py results match HuggingFace KV cache calculator formula.

    Tests verify formula correctness rather than exact values, since:
    - Configs can change upstream (mutable revision="main")
    - Hardcoded expected values become stale
    - Formula relationships are what matter, not specific numbers
    """

    def calc_hf_formula(self, cfg, ctx_len=32768, num_users=8, dtype="bf16"):
        """Calculate KV cache using HF calculator formula"""
        if 'text_config' in cfg:
            cfg = cfg['text_config']

        num_layers = cfg['num_hidden_layers']
        num_kv_heads = cfg['num_key_value_heads']
        num_attn_heads = cfg['num_attention_heads']
        hidden_size = cfg['hidden_size']
        head_dim = cfg.get('head_dim', hidden_size // num_attn_heads)

        nelems_per_token = num_layers * num_kv_heads * head_dim * 2

        # Reuse fit.BYTES_PER_KV for consistent dtype mapping
        # Maps bf16/fp16=2, fp8/int8=1 (fit.py doesn't support quantized KV below int8)
        if dtype not in fit.BYTES_PER_KV:
            raise ValueError(f"Unsupported KV dtype '{dtype}'. Must be one of {list(fit.BYTES_PER_KV.keys())}")
        nbytes_per_elem = fit.BYTES_PER_KV[dtype]

        kv_cache_gb = nelems_per_token * ctx_len * num_users * nbytes_per_elem / 1e9

        return kv_cache_gb

    @pytest.mark.parametrize("model_id", [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
    ])
    def test_open_models_match_hf_calc(self, model_id):
        """
        Open models should match HF calculator formula.

        Verifies formula correctness rather than hardcoded values.
        """
        cfg = fetch_config(model_id)

        # HF formula (ground truth)
        hf_gb = self.calc_hf_formula(cfg)

        # fit.py result
        result = estimate(cfg, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)
        fitpy_gib = result['kv'] / GB
        fitpy_gb = fitpy_gib * 1.073741824  # Convert GiB to GB

        # Should match within PARAM_TOLERANCE (5% as ratio, not percentage)
        diff_ratio = abs(hf_gb - fitpy_gb) / hf_gb
        assert diff_ratio < PARAM_TOLERANCE, \
            f"fit.py doesn't match HF calculator formula! " \
            f"Model: {model_id}, HF: {hf_gb:.2f}GB, fit.py: {fitpy_gb:.2f}GB ({diff_ratio*100:.2f}% diff)"

    @pytest.mark.parametrize("model_id", [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ])
    def test_gated_models_match_hf_calc(self, model_id):
        """
        Gated models should match HF calculator formula.

        Verifies formula correctness rather than hardcoded values.
        """
        # No explicit HF_TOKEN guard: a cached config works without one, and
        # when the fetch does need a token fetch_config turns the 401 into a
        # skip with the reason attached.
        cfg = fetch_config(model_id)
        if not cfg:
            pytest.skip(f"Could not fetch config for {model_id}")

        # HF formula (ground truth)
        hf_gb = self.calc_hf_formula(cfg)

        # fit.py result
        result = estimate(cfg, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)
        fitpy_gib = result['kv'] / GB
        fitpy_gb = fitpy_gib * 1.073741824

        # Should match within PARAM_TOLERANCE (5% as ratio, not percentage)
        diff_ratio = abs(hf_gb - fitpy_gb) / hf_gb
        assert diff_ratio < PARAM_TOLERANCE, \
            f"fit.py doesn't match HF calculator formula! " \
            f"Model: {model_id}, HF: {hf_gb:.2f}GB, fit.py: {fitpy_gb:.2f}GB ({diff_ratio*100:.2f}% diff)"

    def test_kv_cache_scales_with_context(self):
        """KV cache should scale linearly with context length"""
        cfg = fetch_config("Qwen/Qwen2.5-7B-Instruct")

        # Test at different context lengths
        ctx_4k = estimate(cfg, "bf16", "bf16", 4096, 8, 1, "vllm", 32.0)['kv']
        ctx_8k = estimate(cfg, "bf16", "bf16", 8192, 8, 1, "vllm", 32.0)['kv']
        ctx_16k = estimate(cfg, "bf16", "bf16", 16384, 8, 1, "vllm", 32.0)['kv']

        # Should scale linearly
        ratio_8k_4k = ctx_8k / ctx_4k
        ratio_16k_8k = ctx_16k / ctx_8k

        assert abs(ratio_8k_4k - 2.0) < PARAM_TOLERANCE, \
            f"KV cache should double with 2x context: 8K/4K = {ratio_8k_4k:.2f} (expected 2.0)"
        assert abs(ratio_16k_8k - 2.0) < PARAM_TOLERANCE, \
            f"KV cache should double with 2x context: 16K/8K = {ratio_16k_8k:.2f} (expected 2.0)"

    def test_kv_cache_scales_with_concurrency(self):
        """KV cache should scale linearly with concurrency (batch size)"""
        cfg = fetch_config("Qwen/Qwen2.5-7B-Instruct")

        # Test at different concurrency levels
        conc_1 = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)['kv']
        conc_4 = estimate(cfg, "bf16", "bf16", 4096, 4, 1, "vllm", 32.0)['kv']
        conc_8 = estimate(cfg, "bf16", "bf16", 4096, 8, 1, "vllm", 32.0)['kv']

        # Should scale linearly
        ratio_4_1 = conc_4 / conc_1
        ratio_8_4 = conc_8 / conc_4

        assert abs(ratio_4_1 - 4.0) < PARAM_TOLERANCE, \
            f"KV cache should 4x with 4x concurrency: {ratio_4_1:.2f} (expected 4.0)"
        assert abs(ratio_8_4 - 2.0) < PARAM_TOLERANCE, \
            f"KV cache should 2x with 2x concurrency: {ratio_8_4:.2f} (expected 2.0)"

    def test_kv_cache_monotonic_with_model_size(self):
        """Larger models should have more KV cache (monotonicity check)"""
        # Test models in increasing size order
        qwen_7b = estimate(fetch_config("Qwen/Qwen2.5-7B-Instruct"),
                          "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)['kv']
        qwen_14b = estimate(fetch_config("Qwen/Qwen2.5-14B-Instruct"),
                           "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)['kv']

        # 14B should have more KV cache than 7B (more layers/heads)
        assert qwen_14b > qwen_7b, \
            f"Larger model should have more KV cache: 14B={qwen_14b/GB:.2f}GB, 7B={qwen_7b/GB:.2f}GB"


class TestDenseModels:
    """
    Test that dense models are unaffected by MoE fixes.

    Verifies parameter counts are reasonable rather than exact values.
    """

    def test_qwen25_7b_params_reasonable(self):
        """Qwen2.5-7B should have reasonable params (~6-9B)"""
        cfg = fetch_config("Qwen/Qwen2.5-7B-Instruct")
        params_b = count_params(parse_dims(cfg)) / 1e9

        # Reasonable range for a "7B" model
        assert 6.0 <= params_b <= 9.0, \
            f"Params {params_b:.2f}B out of reasonable range [6-9B]"

    def test_qwen25_14b_params_reasonable(self):
        """Qwen2.5-14B should have reasonable params (~13-16B)"""
        cfg = fetch_config("Qwen/Qwen2.5-14B-Instruct")
        params_b = count_params(parse_dims(cfg)) / 1e9

        # Reasonable range for a "14B" model
        assert 13.0 <= params_b <= 16.0, \
            f"Params {params_b:.2f}B out of reasonable range [13-16B]"

    def test_dense_params_monotonic(self):
        """Larger dense models should have more parameters (monotonicity)"""
        qwen_7b = count_params(parse_dims(fetch_config("Qwen/Qwen2.5-7B-Instruct"))) / 1e9
        qwen_14b = count_params(parse_dims(fetch_config("Qwen/Qwen2.5-14B-Instruct"))) / 1e9

        assert qwen_14b > qwen_7b, \
            f"14B model should have more params than 7B: 14B={qwen_14b:.2f}B, 7B={qwen_7b:.2f}B"


class TestMixtralMoE:
    """Test Mixtral (uses standard field names)"""

    def test_mixtral_params_reasonable(self):
        """Mixtral-8x7B should have reasonable params (~40-50B)"""
        cfg = fetch_config("mistralai/Mixtral-8x7B-v0.1")
        params_b = count_params(parse_dims(cfg)) / 1e9

        # Reasonable range for 8x7B MoE model
        assert 40.0 <= params_b <= 50.0, \
            f"Params {params_b:.2f}B out of reasonable range [40-50B]"

    def test_mixtral_moe_detected(self):
        """Mixtral should be detected as MoE with 8 experts"""
        cfg = fetch_config("mistralai/Mixtral-8x7B-v0.1")
        d = parse_dims(cfg)
        assert d.is_moe, "MoE architecture not detected"
        assert d.num_experts == 8, f"Expected 8 experts, got {d.num_experts}"


class TestVLMMoE:
    """Test VLM-MoE models (MoE fields in text_config)"""

    def test_vlm_moe_detection_in_text_config(self):
        """VLM-MoE should detect MoE fields from text_config, not root"""
        # Simulated VLM-MoE config (like a hypothetical Qwen3-VL-MoE)
        cfg = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "num_local_experts": 64,
                "num_experts_per_tok": 4,
                "vocab_size": 32000,
                "tie_word_embeddings": False
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 14,
                "num_channels": 3
            }
        }

        d = parse_dims(cfg)

        # Should detect MoE from text_config
        assert d.is_moe, "VLM-MoE not detected"
        assert d.num_experts == 64, f"Expected 64 experts from text_config, got {d.num_experts}"
        assert d.num_experts_per_tok == 4, f"Expected 4 experts per token, got {d.num_experts_per_tok}"

        # Should be recognized as VLM
        assert d.is_vlm, "VLM not detected"
        assert d.vision_params > 0, "Vision parameters not counted"

    def test_vlm_moe_params_calculation(self):
        """VLM-MoE should use moe_intermediate_size from text_config"""
        cfg = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "num_local_experts": 64,
                "num_experts_per_tok": 4,
                "vocab_size": 32000,
                "tie_word_embeddings": False
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 14,
                "num_channels": 3
            }
        }

        d = parse_dims(cfg)
        params = count_params(d)
        params_b = params / 1e9

        # Should use moe_intermediate_size (768), not intermediate_size (6144)
        # With 64 experts × 4 active, should be reasonable (not wildly inflated)
        assert params_b < 50.0, \
            f"Params {params_b:.2f}B too high - likely using wrong intermediate_size"
        assert params_b > 5.0, \
            f"Params {params_b:.2f}B too low - MoE calculation may be wrong"

    def test_vlm_moe_with_shared_experts(self):
        """VLM-MoE should detect shared experts from text_config"""
        cfg = {
            "architectures": ["DeepSeekVLForConditionalGeneration"],
            "model_type": "deepseek_vl",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "vocab_size": 32000,
                "tie_word_embeddings": False
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 14,
                "num_channels": 3
            }
        }

        d = parse_dims(cfg)

        # Should detect both routed and shared experts from text_config
        assert d.is_moe, "VLM-MoE not detected"
        assert d.num_experts == 64, f"Expected 64 routed experts, got {d.num_experts}"
        assert d.num_shared_experts == 2, f"Expected 2 shared experts, got {d.num_shared_experts}"
        assert d.num_experts_per_tok == 6, f"Expected 6 experts per token, got {d.num_experts_per_tok}"

    def test_vlm_moe_fallback_to_root(self):
        """VLM should fall back to root config if MoE fields not in text_config"""
        # Edge case: VLM with MoE fields at root (unusual but should work)
        cfg = {
            "architectures": ["UnusualVLMForConditionalGeneration"],
            "model_type": "unusual_vlm",
            "num_local_experts": 32,  # At root, not in text_config
            "num_experts_per_tok": 2,
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "vocab_size": 32000,
                "tie_word_embeddings": False
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 14,
                "num_channels": 3
            }
        }

        d = parse_dims(cfg)

        # Should still detect MoE from root fallback
        assert d.is_moe, "VLM-MoE not detected (fallback failed)"
        assert d.num_experts == 32, f"Expected 32 experts from root fallback, got {d.num_experts}"
        assert d.num_experts_per_tok == 2, f"Expected 2 experts per token, got {d.num_experts_per_tok}"


class TestWeightCalculation:
    """
    Test weight calculation is correct.

    Tests verify formula correctness (weights scale with TP)
    rather than exact parameter counts.
    """

    @pytest.mark.parametrize("model_id", [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen2.5-7B-Instruct",
    ])
    @pytest.mark.parametrize("tp", [1, 2, 4])
    def test_weight_per_gpu(self, model_id, tp):
        """Weight per GPU should be total_params * bytes_per_param / TP"""
        cfg = fetch_config(model_id)
        d = parse_dims(cfg)
        params = count_params(d) + d.vision_params

        # Expected: params * 2 bytes (bf16) / TP
        expected_weight_bytes = int(params * 2 / tp)

        result = estimate(cfg, "bf16", "bf16", 4096, 1, tp, "vllm", 32.0)
        actual_weight_bytes = result['weights']

        # Should match exactly (integer division)
        assert abs(actual_weight_bytes - expected_weight_bytes) <= 1, \
            f"{model_id} at TP={tp}: expected {expected_weight_bytes}, got {actual_weight_bytes}"

    def test_weights_scale_inversely_with_tp(self):
        """Weights should scale inversely with TP (higher TP = lower per-GPU)"""
        cfg = fetch_config("Qwen/Qwen2.5-7B-Instruct")

        weights_tp1 = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)['weights']
        weights_tp2 = estimate(cfg, "bf16", "bf16", 4096, 1, 2, "vllm", 32.0)['weights']
        weights_tp4 = estimate(cfg, "bf16", "bf16", 4096, 1, 4, "vllm", 32.0)['weights']

        # Should be roughly half when TP doubles
        ratio_tp1_tp2 = weights_tp1 / weights_tp2
        ratio_tp2_tp4 = weights_tp2 / weights_tp4

        assert abs(ratio_tp1_tp2 - 2.0) < PARAM_TOLERANCE, \
            f"Weights should halve with TP=2: ratio={ratio_tp1_tp2:.2f} (expected 2.0)"
        assert abs(ratio_tp2_tp4 - 2.0) < PARAM_TOLERANCE, \
            f"Weights should halve with TP=4: ratio={ratio_tp2_tp4:.2f} (expected 2.0)"


class TestHybridMoE:
    """Test Hybrid MoE models (first_k_dense_replace)"""

    def test_deepseek_v3_hybrid_moe_detection(self):
        """DeepSeek-V3 should detect hybrid MoE with first_k_dense_replace"""
        # Create synthetic DeepSeek-V3 config (hybrid MoE)
        cfg = {
            "hidden_size": 7168,
            "num_hidden_layers": 61,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "n_shared_experts": 2,
            "first_k_dense_replace": 1,
            "vocab_size": 102400,
            "tie_word_embeddings": False
        }

        d = parse_dims(cfg)

        # Should detect hybrid MoE
        assert d.is_moe, "Hybrid MoE not detected"
        assert d.first_k_dense_replace == 1, f"Expected first_k_dense_replace=1, got {d.first_k_dense_replace}"
        assert d.num_experts == 256, f"Expected 256 experts, got {d.num_experts}"
        assert d.num_shared_experts == 2, f"Expected 2 shared experts, got {d.num_shared_experts}"
        assert d.dense_intermediate == 18432, f"Expected dense_intermediate=18432, got {d.dense_intermediate}"

    def test_hybrid_moe_params_calculation(self):
        """Hybrid MoE should count dense + MoE layers separately"""
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 12,  # 2 dense + 10 MoE
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,  # Dense FFN
            "moe_intermediate_size": 768,  # MoE FFN
            "num_local_experts": 64,
            "num_experts_per_tok": 4,
            "n_shared_experts": 2,
            "first_k_dense_replace": 2,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        }

        d = parse_dims(cfg)
        params = count_params(d)

        # Manual calculation
        h = 2048
        q_proj = 32 * 128  # num_attn_heads * head_dim
        kv_proj = 4 * 128   # num_kv_heads * head_dim

        # Attention (same for all layers)
        attn = h * q_proj + h * kv_proj + h * kv_proj + q_proj * h

        # Dense layers (first 2)
        dense_ff = 3 * h * 6144
        dense_norms = 4 * h
        dense_per_layer = attn + dense_ff + dense_norms

        # MoE layers (remaining 10)
        routed_ff = 64 * 3 * h * 768
        shared_ff = 2 * 3 * h * 768
        moe_ff = routed_ff + shared_ff
        moe_norms = 4 * h
        moe_per_layer = attn + moe_ff + moe_norms

        # Total
        emb = 32000 * h
        head = 32000 * h  # not tied
        expected = emb + head + (2 * dense_per_layer) + (10 * moe_per_layer)

        diff_ratio = abs(params - expected) / expected
        assert diff_ratio < 0.001, \
            f"Hybrid MoE param calculation wrong: got {params:,}, expected {expected:,}"

    def test_hybrid_moe_vs_pure_moe(self):
        """Hybrid MoE should differ from pure MoE due to dense layers"""
        cfg_base = {
            "hidden_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_local_experts": 64,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        }

        # Pure MoE
        cfg_pure = {**cfg_base}
        params_pure = count_params(parse_dims(cfg_pure))

        # Hybrid MoE (2 dense layers)
        cfg_hybrid = {**cfg_base, "first_k_dense_replace": 2}
        params_hybrid = count_params(parse_dims(cfg_hybrid))

        # Hybrid should have fewer params (replaces MoE layers with dense layers)
        # Dense FFN: 3 * 2048 * 6144 = 37.7M per layer
        # MoE FFN: 64 * 3 * 2048 * 768 = 301.9M per layer
        # So replacing 2 MoE layers with 2 dense layers reduces params
        assert params_hybrid < params_pure, \
            f"Hybrid MoE should have fewer params than pure MoE: hybrid={params_hybrid:,}, pure={params_pure:,}"

        # The difference should be roughly 2 * (MoE_FFN - Dense_FFN)
        expected_reduction = 2 * ((64 * 3 * 2048 * 768) - (3 * 2048 * 6144))
        actual_reduction = params_pure - params_hybrid
        diff_ratio = abs(actual_reduction - expected_reduction) / expected_reduction
        assert diff_ratio < 0.1, \
            f"Reduction doesn't match expected: actual={actual_reduction:,}, expected={expected_reduction:,}"


class TestParamsJsonFormat:
    """Test Mistral params.json format support"""

    def test_params_json_field_mapping(self):
        """params.json format should map fields correctly to config.json equivalents"""
        # Mistral params.json format
        cfg = {
            "dim": 8192,  # -> hidden_size
            "n_layers": 48,  # -> num_hidden_layers
            "n_heads": 64,  # -> num_attention_heads
            "n_kv_heads": 8,  # -> num_key_value_heads
            "vocab_size": 131072,
            "tied_embeddings": False,  # -> tie_word_embeddings
            "moe": {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "expert_hidden_dim": 28672  # -> moe_intermediate_size
            }
        }

        d = parse_dims(cfg)

        # Should parse params.json fields correctly
        assert d.hidden == 8192, f"Expected hidden=8192, got {d.hidden}"
        assert d.num_layers == 48, f"Expected num_layers=48, got {d.num_layers}"
        assert d.num_attn_heads == 64, f"Expected num_attn_heads=64, got {d.num_attn_heads}"
        assert d.num_kv_heads == 8, f"Expected num_kv_heads=8, got {d.num_kv_heads}"
        assert d.num_experts == 8, f"Expected num_experts=8, got {d.num_experts}"
        assert d.num_experts_per_tok == 2, f"Expected num_experts_per_tok=2, got {d.num_experts_per_tok}"
        assert d.intermediate == 28672, f"Expected intermediate=28672, got {d.intermediate}"

    def test_params_json_hybrid_moe(self):
        """params.json format should support hybrid MoE (Mistral-Large-3)"""
        cfg = {
            "dim": 12288,
            "n_layers": 88,
            "n_heads": 96,
            "n_kv_heads": 8,
            "vocab_size": 131072,
            "tied_embeddings": False,
            "hidden_dim": 12288 * 4,  # Dense FFN for first_k_dense_replace layers
            "moe": {
                "num_experts": 128,
                "num_experts_per_tok": 2,
                "num_shared_experts": 2,
                "expert_hidden_dim": 3584,
                "first_k_dense_replace": 3
            }
        }

        d = parse_dims(cfg)

        # Should detect hybrid MoE from params.json
        assert d.is_moe, "params.json MoE not detected"
        assert d.first_k_dense_replace == 3, f"Expected first_k_dense_replace=3, got {d.first_k_dense_replace}"
        assert d.num_experts == 128, f"Expected 128 experts, got {d.num_experts}"
        assert d.num_shared_experts == 2, f"Expected 2 shared experts, got {d.num_shared_experts}"
        assert d.dense_intermediate == 12288 * 4, f"Expected dense_intermediate={12288*4}, got {d.dense_intermediate}"
        assert d.intermediate == 3584, f"Expected moe_intermediate=3584, got {d.intermediate}"

    def test_params_json_params_reasonable(self):
        """params.json models should have reasonable parameter counts"""
        # Simulated Mistral-Large-3 config (scaled down for reasonable test)
        # Real Mistral-Large-3 is huge, this tests the formula works
        cfg = {
            "dim": 4096,
            "n_layers": 32,
            "n_heads": 32,
            "n_kv_heads": 8,
            "vocab_size": 32000,
            "tied_embeddings": False,
            "hidden_dim": 16384,  # Dense FFN for first_k_dense_replace
            "moe": {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "num_shared_experts": 1,
                "expert_hidden_dim": 2048,  # MoE FFN
                "first_k_dense_replace": 2
            }
        }

        d = parse_dims(cfg)
        params_b = count_params(d) / 1e9

        # Should be reasonable for a scaled-down hybrid MoE model (~8-10B range)
        assert 5.0 <= params_b <= 15.0, \
            f"Params {params_b:.2f}B out of reasonable range [5-15B]"


class TestKVCacheForNewModels:
    """
    Test KV cache calculations for models added in PR #14 against HF calculator.

    Ground truth values obtained from https://huggingface.co/spaces/gaunernst/kv-cache-calculator
    Formula source: https://huggingface.co/spaces/gaunernst/kv-cache-calculator/blob/main/app.py

    HF Calculator formula (standard MHA):
        nelems_per_token = num_layers × num_kv_heads × head_dim × 2
        kv_cache_gb = (nelems_per_token × ctx_len × num_users × nbytes_per_elem) / 1e9

    All tests use ctx_len=32768, num_users=8, dtype=bf16 unless specified.
    Validated on 2026-05-18 - all results match HF calculator with 0.00% difference.
    """

    def calc_hf_formula(self, cfg, ctx_len=32768, num_users=8, dtype="bf16"):
        """
        Calculate KV cache using exact HF calculator formula.

        This implements the formula from:
        https://huggingface.co/spaces/gaunernst/kv-cache-calculator/blob/main/app.py
        """
        if 'text_config' in cfg:
            cfg = cfg['text_config']
        elif 'llm_config' in cfg:
            cfg = cfg['llm_config']

        num_layers = cfg.get('num_hidden_layers') or cfg.get('n_layers')
        num_kv_heads = cfg.get('num_key_value_heads') or cfg.get('n_kv_heads')
        num_attn_heads = cfg.get('num_attention_heads') or cfg.get('n_heads')
        hidden_size = cfg.get('hidden_size') or cfg.get('dim')
        head_dim = cfg.get('head_dim', hidden_size // num_attn_heads)

        # Standard MHA formula (not MLA)
        nelems_per_token = num_layers * num_kv_heads * head_dim * 2

        if dtype not in fit.BYTES_PER_KV:
            raise ValueError(f"Unsupported KV dtype '{dtype}'")
        nbytes_per_elem = fit.BYTES_PER_KV[dtype]

        kv_cache_gb = nelems_per_token * ctx_len * num_users * nbytes_per_elem / 1e9

        return kv_cache_gb

    def test_hybrid_moe_kv_cache_vs_hf(self):
        """Hybrid MoE KV cache should match HF calculator (DeepSeek-V3 style)"""
        # Simulated DeepSeek-V3 config
        cfg = {
            "hidden_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 8192,
            "moe_intermediate_size": 1024,
            "n_routed_experts": 64,
            "num_experts_per_tok": 4,
            "n_shared_experts": 2,
            "first_k_dense_replace": 2,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        }

        # HF formula (ground truth)
        hf_gb = self.calc_hf_formula(cfg)

        # fit.py result
        result = estimate(cfg, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)
        fitpy_gib = result['kv'] / GB
        fitpy_gb = fitpy_gib * 1.073741824

        # Should match within 5% tolerance
        diff_ratio = abs(hf_gb - fitpy_gb) / hf_gb
        assert diff_ratio < PARAM_TOLERANCE, \
            f"Hybrid MoE KV cache doesn't match HF calculator! " \
            f"HF: {hf_gb:.2f}GB, fit.py: {fitpy_gb:.2f}GB ({diff_ratio*100:.2f}% diff)"

    def test_params_json_kv_cache_vs_hf(self):
        """params.json models KV cache should match HF calculator (Mistral style)"""
        # Simulated Mistral params.json config
        cfg = {
            "dim": 8192,
            "n_layers": 48,
            "n_heads": 64,
            "n_kv_heads": 8,
            "vocab_size": 131072,
            "tied_embeddings": False,
            "moe": {
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "expert_hidden_dim": 28672
            }
        }

        # HF formula (ground truth)
        hf_gb = self.calc_hf_formula(cfg)

        # fit.py result
        result = estimate(cfg, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)
        fitpy_gib = result['kv'] / GB
        fitpy_gb = fitpy_gib * 1.073741824

        # Should match within 5% tolerance
        diff_ratio = abs(hf_gb - fitpy_gb) / hf_gb
        assert diff_ratio < PARAM_TOLERANCE, \
            f"params.json KV cache doesn't match HF calculator! " \
            f"HF: {hf_gb:.2f}GB, fit.py: {fitpy_gb:.2f}GB ({diff_ratio*100:.2f}% diff)"

    def test_llm_config_kv_cache_vs_hf(self):
        """llm_config models KV cache should match HF calculator (Nemotron style)"""
        # Simulated Nemotron llm_config
        cfg = {
            "architectures": ["NemotronForConditionalGeneration"],
            "model_type": "nemotron",
            "llm_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "moe_intermediate_size": 768,
                "num_local_experts": 64,
                "num_experts_per_tok": 4,
                "n_shared_experts": 2,
                "vocab_size": 32000,
                "tie_word_embeddings": False
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096
            }
        }

        # HF formula (ground truth)
        hf_gb = self.calc_hf_formula(cfg)

        # fit.py result
        result = estimate(cfg, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)
        fitpy_gib = result['kv'] / GB
        fitpy_gb = fitpy_gib * 1.073741824

        # Should match within 5% tolerance
        diff_ratio = abs(hf_gb - fitpy_gb) / hf_gb
        assert diff_ratio < PARAM_TOLERANCE, \
            f"llm_config KV cache doesn't match HF calculator! " \
            f"HF: {hf_gb:.2f}GB, fit.py: {fitpy_gb:.2f}GB ({diff_ratio*100:.2f}% diff)"

    def test_hybrid_moe_kv_independent_of_first_k_dense(self):
        """KV cache should NOT depend on first_k_dense_replace (only affects FFN)"""
        cfg_base = {
            "hidden_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_local_experts": 64,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        }

        # Pure MoE
        result_pure = estimate(cfg_base, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)

        # Hybrid MoE (2 dense layers)
        cfg_hybrid = {**cfg_base, "first_k_dense_replace": 2}
        result_hybrid = estimate(cfg_hybrid, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)

        # KV cache should be identical (first_k_dense_replace only affects FFN, not attention)
        assert result_pure['kv'] == result_hybrid['kv'], \
            f"KV cache should not change with first_k_dense_replace: " \
            f"pure={result_pure['kv']}, hybrid={result_hybrid['kv']}"

    def test_kv_cache_scales_with_context_new_models(self):
        """New model types should have KV cache that scales linearly with context"""
        # Test hybrid MoE
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 12,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_local_experts": 64,
            "num_experts_per_tok": 4,
            "first_k_dense_replace": 2,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        }

        ctx_4k = estimate(cfg, "bf16", "bf16", 4096, 8, 1, "vllm", 32.0)['kv']
        ctx_8k = estimate(cfg, "bf16", "bf16", 8192, 8, 1, "vllm", 32.0)['kv']

        ratio = ctx_8k / ctx_4k
        assert abs(ratio - 2.0) < PARAM_TOLERANCE, \
            f"KV cache should double with 2x context: ratio={ratio:.2f} (expected 2.0)"

    @pytest.mark.parametrize("model_id,ctx_len,num_users,expected_gb", [
        # Ground truth from HF calculator: https://huggingface.co/spaces/gaunernst/kv-cache-calculator
        # Validated 2026-05-18, ctx_len=32768, num_users=8, dtype=bf16
        ("Qwen/Qwen2.5-7B-Instruct", 32768, 8, 15.03),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", 32768, 8, 25.77),
        ("mistralai/Mixtral-8x7B-v0.1", 32768, 8, 34.36),
        ("deepseek-ai/DeepSeek-V4-Flash", 32768, 8, 23.09),
        # Additional validation points with different parameters
        ("Qwen/Qwen2.5-7B-Instruct", 4096, 1, 0.23),
        ("Qwen/Qwen2.5-7B-Instruct", 8192, 4, 1.88),
        ("Qwen/Qwen2.5-7B-Instruct", 16384, 8, 7.52),
    ])
    def test_real_models_match_hf_ground_truth(self, model_id, ctx_len, num_users, expected_gb):
        """
        Validate against actual HF calculator ground truth values.

        These are real values obtained by running the HF calculator, not computed from formula.
        This ensures we match the calculator's behavior exactly, including any rounding or edge cases.
        """
        cfg = fetch_config(model_id)
        result = estimate(cfg, "bf16", "bf16", ctx_len, num_users, 1, "vllm", 32.0)
        fitpy_gib = result['kv'] / GB
        fitpy_gb = fitpy_gib * 1.073741824

        # Allow 0.01 GB tolerance for floating point precision
        diff = abs(fitpy_gb - expected_gb)
        assert diff < 0.01, \
            f"{model_id}: Expected {expected_gb:.2f} GB (HF ground truth), got {fitpy_gb:.2f} GB ({diff:.4f} GB diff)"


class TestLLMConfigSupport:
    """Test llm_config support for multimodal models"""

    def test_llm_config_detection(self):
        """Should read LLM fields from llm_config (Nemotron-3-Nano-Omni)"""
        cfg = {
            "architectures": ["NemotronForConditionalGeneration"],
            "model_type": "nemotron",
            "llm_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 4,
                "intermediate_size": 6144,
                "moe_intermediate_size": 768,
                "num_local_experts": 64,
                "num_experts_per_tok": 4,
                "n_shared_experts": 2,
                "vocab_size": 32000,
                "tie_word_embeddings": False
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096
            }
        }

        d = parse_dims(cfg)

        # Should read from llm_config
        assert d.hidden == 2048, f"Expected hidden=2048 from llm_config, got {d.hidden}"
        assert d.num_layers == 24, f"Expected num_layers=24 from llm_config, got {d.num_layers}"
        assert d.is_moe, "MoE not detected from llm_config"
        assert d.num_experts == 64, f"Expected 64 experts from llm_config, got {d.num_experts}"
        assert d.num_shared_experts == 2, f"Expected 2 shared experts from llm_config, got {d.num_shared_experts}"

    def test_llm_config_vlm_detection(self):
        """llm_config models with vision_config should be detected as VLM"""
        cfg = {
            "architectures": ["NemotronForConditionalGeneration"],
            "model_type": "nemotron",
            "llm_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "intermediate_size": 6144,
                "vocab_size": 32000
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 14,
                "num_channels": 3
            }
        }

        d = parse_dims(cfg)

        # Should detect as VLM
        assert d.is_vlm, "VLM not detected with llm_config"
        assert d.vision_params > 0, "Vision parameters not counted with llm_config"

    def test_llm_config_priority_over_root(self):
        """llm_config should take priority over root-level fields"""
        cfg = {
            "architectures": ["NemotronForConditionalGeneration"],
            "hidden_size": 999,  # Wrong value at root
            "num_hidden_layers": 999,  # Wrong value at root
            "llm_config": {
                "hidden_size": 2048,  # Correct value
                "num_hidden_layers": 24,  # Correct value
                "num_attention_heads": 32,
                "intermediate_size": 6144,
                "vocab_size": 32000
            },
            "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 16,
                "intermediate_size": 4096
            }
        }

        d = parse_dims(cfg)

        # Should use llm_config values, not root
        assert d.hidden == 2048, f"Expected hidden=2048 from llm_config, got {d.hidden}"
        assert d.num_layers == 24, f"Expected num_layers=24 from llm_config, got {d.num_layers}"


class TestRegressions:
    """Ensure bugs don't regress"""

    def test_bug1_qwen3_30b_not_233b(self):
        """Bug #1 regression: Qwen3-30B must NOT be 233B"""
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        params_b = count_params(parse_dims(cfg)) / 1e9
        assert params_b < 100, \
            f"Bug #1 REGRESSION! Calculated {params_b:.2f}B (should be ~30B)"

    def test_bug1_qwen3_235b_not_1821b(self):
        """Bug #1 regression: Qwen3-235B must NOT be 1821B"""
        cfg = fetch_config("Qwen/Qwen3-235B-A22B-Instruct-2507")
        params_b = count_params(parse_dims(cfg)) / 1e9
        assert params_b < 500, \
            f"Bug #1 REGRESSION! Calculated {params_b:.2f}B (should be ~235B)"

    def test_bug1_deepseek_not_11b(self):
        """Bug #1 regression: DeepSeek-V4 must NOT be 11B"""
        cfg = fetch_config("deepseek-ai/DeepSeek-V4-Flash")
        params_b = count_params(parse_dims(cfg)) / 1e9
        assert params_b > 100, \
            f"Bug #1 REGRESSION! Calculated {params_b:.2f}B (should be ~280B)"

    def test_bug3_qwen3_kv_not_12gb(self):
        """Bug #3 regression: Qwen3-30B KV cache must NOT be ~12GB"""
        cfg = fetch_config("Qwen/Qwen3-30B-A3B-Instruct-2507")
        result = estimate(cfg, "bf16", "bf16", 32768, 8, 1, "vllm", 32.0)
        kv_gib = result['kv'] / GB

        # Should be ~24 GiB (not ~12 GiB with wrong head_dim)
        assert kv_gib > 20.0, \
            f"Bug #3 REGRESSION! KV cache {kv_gib:.2f} GiB (should be ~24 GiB)"


class TestMixedPrecisionQuantization:
    """
    Test mixed-precision quantization support (quantization_config.modules_to_not_convert).

    Models like openai/gpt-oss-20b and openai/gpt-oss-120b use selective quantization:
    - Most parameters (MoE experts) are quantized to mxfp4 (0.55 bytes/param)
    - Critical components (embeddings, attention) stay at bf16 (2.0 bytes/param)

    This tests both auto-detection of pre-quantized models and accurate mixed-precision calculation.
    """

    def test_gpt_oss_20b_auto_detect_mxfp4(self):
        """GPT OSS 20B should auto-detect mxfp4 from quantization_config"""
        # Uses the cached config if present, otherwise fetches; skips if neither
        # is possible.
        cfg = fetch_config_or_skip("openai/gpt-oss-20b")

        # Check quantization_config exists
        assert "quantization_config" in cfg, "Model should have quantization_config"
        qcfg = cfg.get("quantization_config", {})
        assert qcfg.get("quant_method") == "mxfp4", "Should be quantized with mxfp4"

    def test_gpt_oss_20b_mixed_precision_calculation(self):
        """GPT OSS 20B should use mixed-precision calculation for accurate weights"""
        cfg = fetch_config_or_skip("openai/gpt-oss-20b")

        # Estimate with auto-detected mxfp4
        result = estimate(cfg, "mxfp4", "fp8", 4096, 1, 1, "vllm", 32.0)

        # Should have mixed-precision breakdown
        assert result.get("mixed_breakdown") is not None, \
            "Should have mixed-precision breakdown for this model"

        breakdown = result["mixed_breakdown"]
        weights_bytes = result["weights"]

        # Check component breakdown exists
        assert breakdown["embed_params"] > 0, "Should have embedding params"
        assert breakdown["attn_params"] > 0, "Should have attention params"
        assert breakdown["ffn_params"] > 0, "Should have FFN/expert params"

        # Embeddings and attention should be at bf16 (2.0 B/p)
        assert breakdown["embed_bpp"] == 2.0, "Embeddings should be bf16"
        assert breakdown["attn_bpp"] == 2.0, "Attention should be bf16"

        # FFN/experts should be at mxfp4 (0.55 B/p)
        assert breakdown["ffn_bpp"] == 0.55, "FFN/experts should be mxfp4"

        # Sum of components should match total weights
        component_sum = (breakdown["embed_bytes"] + breakdown["attn_bytes"] +
                        breakdown["router_bytes"] + breakdown["ffn_bytes"])
        assert abs(component_sum - weights_bytes) <= 1, \
            f"Component sum ({component_sum}) should match total weights ({weights_bytes})"

        # Mixed-precision should be more than naive uniform calculation
        params = result["params"]
        naive_weights = int(params * 0.55)  # Uniform mxfp4
        assert weights_bytes > naive_weights, \
            f"Mixed-precision ({weights_bytes/GB:.2f} GB) should be larger than " \
            f"naive uniform calculation ({naive_weights/GB:.2f} GB)"

    def test_gpt_oss_20b_weight_accuracy(self):
        """GPT OSS 20B mixed-precision should be ~13 GB, not ~11 GB"""
        cfg = fetch_config_or_skip("openai/gpt-oss-20b")

        result = estimate(cfg, "mxfp4", "fp8", 4096, 1, 1, "vllm", 32.0)
        weights_gib = result["weights"] / GB

        # Should be ~13 GiB (mixed-precision), not ~11 GiB (naive uniform)
        assert 12.0 <= weights_gib <= 14.0, \
            f"Weights should be ~13 GiB with mixed-precision, got {weights_gib:.2f} GiB"

    def test_gpt_oss_120b_auto_detect_mxfp4(self):
        """GPT OSS 120B should auto-detect mxfp4 from quantization_config"""
        cfg = fetch_config_or_skip("openai/gpt-oss-120b")

        # Check quantization_config exists
        assert "quantization_config" in cfg, "Model should have quantization_config"
        qcfg = cfg.get("quantization_config", {})
        assert qcfg.get("quant_method") == "mxfp4", "Should be quantized with mxfp4"

    def test_gpt_oss_120b_mixed_precision_calculation(self):
        """GPT OSS 120B should use mixed-precision calculation"""
        cfg = fetch_config_or_skip("openai/gpt-oss-120b")

        result = estimate(cfg, "mxfp4", "fp8", 4096, 1, 1, "vllm", 80.0)

        # Should have mixed-precision breakdown
        assert result.get("mixed_breakdown") is not None, \
            "Should have mixed-precision breakdown for this model"

        breakdown = result["mixed_breakdown"]

        # Check precision assignments
        assert breakdown["embed_bpp"] == 2.0, "Embeddings should be bf16"
        assert breakdown["attn_bpp"] == 2.0, "Attention should be bf16"
        assert breakdown["ffn_bpp"] == 0.55, "FFN/experts should be mxfp4"

    def test_gpt_oss_120b_weight_accuracy(self):
        """GPT OSS 120B mixed-precision should be ~63 GB, not ~60 GB"""
        cfg = fetch_config_or_skip("openai/gpt-oss-120b")

        result = estimate(cfg, "mxfp4", "fp8", 4096, 1, 1, "vllm", 80.0)
        weights_gib = result["weights"] / GB

        # Should be ~63 GiB (mixed-precision), not ~60 GiB (naive uniform)
        assert 61.0 <= weights_gib <= 65.0, \
            f"Weights should be ~63 GiB with mixed-precision, got {weights_gib:.2f} GiB"

    def test_gpt_oss_explicit_bf16_override(self):
        """GPT OSS models should respect explicit --quant bf16 override"""
        cfg = fetch_config_or_skip("openai/gpt-oss-20b")

        # Estimate with explicit bf16 (override auto-detection)
        result = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)
        weights_gib = result["weights"] / GB

        # Should be ~39 GiB (all bf16), not ~13 GiB (mxfp4)
        assert 37.0 <= weights_gib <= 41.0, \
            f"With explicit bf16, weights should be ~39 GiB, got {weights_gib:.2f} GiB"

    def test_mixed_precision_falls_back_gracefully(self):
        """Models without modules_to_not_convert should fall back to uniform calculation"""
        # Regular model without selective quantization
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "vocab_size": 32000,
            "tie_word_embeddings": False
        }

        result = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)

        # Should NOT have mixed-precision breakdown (no selective quantization)
        assert result.get("mixed_breakdown") is None, \
            "Regular model should not have mixed-precision breakdown"

        # Weights should be uniform calculation
        d = parse_dims(cfg)
        params = count_params(d)
        expected_weights = int(params * 2.0)  # Uniform bf16
        assert result["weights"] == expected_weights, \
            f"Should use uniform calculation: got {result['weights']}, expected {expected_weights}"

    def test_mixed_precision_params_sum_to_total(self):
        """Mixed-precision component params should sum to total params"""
        cfg = fetch_config_or_skip("openai/gpt-oss-20b")

        result = estimate(cfg, "mxfp4", "fp8", 4096, 1, 1, "vllm", 32.0)
        breakdown = result["mixed_breakdown"]

        # Sum of component params
        component_params = (breakdown["embed_params"] + breakdown["attn_params"] +
                           breakdown["router_params"] + breakdown["ffn_params"])

        # Should equal total params (allowing small rounding difference)
        total_params = result["params"]
        diff_ratio = abs(component_params - total_params) / total_params
        assert diff_ratio < 0.01, \
            f"Component params sum ({component_params/1e9:.2f}B) should match " \
            f"total ({total_params/1e9:.2f}B), diff: {diff_ratio*100:.2f}%"

    def test_router_params_counted_from_the_architecture(self):
        """Routers are hidden * num_experts per MoE layer, not a placeholder

        A `2 * hidden` approximation is 16x light on the 32-expert config
        below, which underprices exactly the bf16 uplift a router-only
        modules_to_not_convert list exists to charge for.
        """
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "num_local_experts": 32,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
        }
        d = parse_dims(cfg)
        assert fit.router_params(d) == 24 * 2048 * 32
        assert fit.router_params(d) == 16 * (24 * 2 * 2048), "16x the placeholder"

        # Counted in the total, so the component rows can add up to it.
        assert count_params(d) > 0
        no_router = count_params(d) - fit.router_params(d)
        assert no_router > 0

        # Hybrid MoE: only the expert layers carry a gate.
        hybrid = parse_dims(dict(cfg, first_k_dense_replace=4,
                                 intermediate_size=6144))
        assert fit.router_params(hybrid) == (24 - 4) * 2048 * 32

        # Dense, and MoE-with-no-expert-count, have none.
        assert fit.router_params(parse_dims({
            "hidden_size": 2048, "num_hidden_layers": 4,
            "num_attention_heads": 16, "intermediate_size": 8192,
            "vocab_size": 32000})) == 0

    def test_router_uplift_uses_the_real_count(self):
        """The bf16 router row is priced on hidden * num_experts"""
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "num_local_experts": 32,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "quantization_config": {
                "quant_method": "int4",
                "modules_to_not_convert": ["router"],
            },
        }
        d = parse_dims(cfg)
        breakdown = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm",
                             32.0)["mixed_breakdown"]
        assert breakdown["router_params"] == fit.router_params(d)
        assert breakdown["router_bytes"] == int(fit.router_params(d) * 2.0)

    @pytest.mark.parametrize("entry,is_router", [
        ("model.layers.*.mlp.router", True),           # gpt-oss
        ("model.layers.*.block_sparse_moe.gate", True),  # Mixtral
        ("model.layers.*.mlp.gate", True),             # DeepSeek / Qwen3-MoE
        ("gate", True),
        ("router", True),
        ("model.layers.*.mlp.gate_proj", False),       # dense SwiGLU
        ("model.layers.*.mlp.gate_up_proj", False),    # fused SwiGLU
        ("model.layers.*.self_attn", False),
        ("lm_head", False),
    ])
    def test_router_module_spellings(self, entry, is_router):
        """`gate` is the common spelling, and gate_proj is not a router

        Matching `gate` as a substring would sweep the dense SwiGLU
        projections into the router row -- most of a layer's params in the
        wrong place, at the wrong dtype.
        """
        assert fit.is_router_module(entry) is is_router

    def test_gate_only_list_is_honored_like_router(self):
        """A Mixtral-style gate entry holds the gate at bf16"""
        cfg = {
            "hidden_size": 2048, "num_hidden_layers": 24,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 6144, "num_local_experts": 32,
            "moe_intermediate_size": 768, "num_experts_per_tok": 4,
            "vocab_size": 32000, "tie_word_embeddings": False,
            "quantization_config": {
                "quant_method": "int4",
                "modules_to_not_convert": [
                    "model.layers.*.block_sparse_moe.gate"],
            },
        }
        breakdown = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm",
                             16.0)["mixed_breakdown"]
        assert breakdown is not None, "a gate-only list is still selective"
        assert breakdown["router_params"] == fit.router_params(parse_dims(cfg))
        assert breakdown["router_bpp"] == 2.0

        # A gate_proj-only list is not a router exclusion, so no router row.
        # (There is still a breakdown: int4 holds the embeddings at bf16.)
        dense_gate = dict(cfg, quantization_config={
            "quant_method": "int4",
            "modules_to_not_convert": ["model.layers.*.mlp.gate_proj"]})
        no_router = estimate(dense_gate, "int4", "fp8", 4096, 1, 1, "vllm",
                             16.0)["mixed_breakdown"]
        assert no_router["router_params"] == 0

    def test_router_only_modules_to_not_convert_is_honored(self):
        """A router-only list must hold the routers at bf16

        Routers are one of the three components the parser recognizes, so a
        list naming only routers must not drop through to the uniform path --
        that would price them at the very dtype the config excluded them from.
        """
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "num_local_experts": 32,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "quantization_config": {
                "quant_method": "int4",
                "modules_to_not_convert": ["model.layers.*.mlp.router"],
            },
        }
        result = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm", 32.0)
        breakdown = result["mixed_breakdown"]
        assert breakdown is not None, "a router-only list is still selective"
        assert breakdown["router_params"] > 0
        assert breakdown["router_bpp"] == 2.0, "routers stay bf16"
        # Attention was not named, so it is at the quant dtype. Embeddings are
        # bf16 regardless: int4 is sub-16-bit, and a quantizer does not convert
        # nn.Embedding whatever the exclusion list says.
        assert breakdown["attn_bpp"] == 0.55
        assert breakdown["embed_bpp"] == 2.0
        # Rows still total the weights line and cover every param once.
        component_sum = (breakdown["embed_bytes"] + breakdown["attn_bytes"] +
                         breakdown["router_bytes"] + breakdown["ffn_bytes"])
        assert abs(component_sum - result["weights"]) <= 1
        assert (breakdown["embed_params"] + breakdown["attn_params"] +
                breakdown["router_params"] + breakdown["ffn_params"]) == result["params"]

    @pytest.mark.parametrize("modules,label", [
        (["model.layers.*.self_attn"], "attention-only"),
        (["model.layers.*.mlp.gate"], "router-only"),
        (["model.layers.*.self_attn", "model.layers.*.mlp.gate"], "attn+router"),
        (["model.embed_tokens", "lm_head"], "embeddings named"),
        ([], "no exclusion list"),
    ])
    def test_embeddings_are_wide_for_every_exclusion_shape(self, modules, label):
        """A sub-16-bit dtype never prices the embedding matrix at that dtype

        Whether the config names embeddings, names something else, or names
        nothing, `nn.Embedding` is not what a quantizer converts. An
        attention-only or router-only list reaching the selective branch with
        keep_embeddings False used to charge a 152064x3584 matrix at 0.55
        B/param -- over a gigabyte of understatement on a wide-vocab model.
        """
        qcfg = {"quant_method": "int4"}
        if modules:
            qcfg["modules_to_not_convert"] = modules
        cfg = {
            "hidden_size": 3584, "num_hidden_layers": 28,
            "num_attention_heads": 28, "num_key_value_heads": 4,
            "intermediate_size": 18944, "vocab_size": 152064,
            "tie_word_embeddings": False, "quantization_config": qcfg,
        }
        d = parse_dims(cfg)
        result = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm", 32.0)
        breakdown = result["mixed_breakdown"]
        assert breakdown is not None, label
        assert breakdown["embed_bpp"] == 2.0, label
        assert breakdown["embed_params"] == d.vocab * d.hidden * 2, label
        # And the total reflects it: a flat int4 pricing would be ~1.5 GiB less.
        flat = int(count_params(d) * 0.55)
        assert result["weights"] > flat + int(0.9 * GB), label

    def test_router_only_list_on_a_dense_model_stays_uniform(self):
        """A dense model has no routers to hold back, so no breakdown"""
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "quantization_config": {
                "quant_method": "int4",
                "modules_to_not_convert": ["router"],
            },
        }
        result = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm", 32.0)
        breakdown = result["mixed_breakdown"]
        assert breakdown["router_params"] == 0, "no router row"
        # Embeddings are still held at bf16: quantizers convert Linear layers,
        # not nn.Embedding, whatever modules_to_not_convert says.
        d = parse_dims(cfg)
        embed_p = d.vocab * d.hidden * 2
        assert breakdown["embed_params"] == embed_p
        assert breakdown["embed_bpp"] == 2.0
        assert result["weights"] == int(embed_p * 2.0
                                        + (count_params(d) - embed_p) * 0.55)

    def test_modules_to_not_convert_detection(self):
        """Should detect modules_to_not_convert and apply correct precision"""
        # Simulated config with selective quantization
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "num_local_experts": 32,
            "moe_intermediate_size": 768,
            "num_experts_per_tok": 4,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "quantization_config": {
                "quant_method": "int4",
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.embed_tokens",
                    "lm_head"
                ]
            }
        }

        result = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm", 32.0)
        breakdown = result["mixed_breakdown"]

        # Should keep embeddings and attention at bf16
        assert breakdown["embed_bpp"] == 2.0, "Embeddings in modules_to_not_convert should be bf16"
        assert breakdown["attn_bpp"] == 2.0, "Attention in modules_to_not_convert should be bf16"

        # Should quantize FFN/experts to int4
        assert breakdown["ffn_bpp"] == 0.55, "FFN/experts should be quantized to int4"


class TestExpertDtype:
    """
    A separate `expert_dtype` for the MoE FFNs (DeepSeek-V4 style).

    The config declares `quantization_config.quant_method: fp8` *and*
    `expert_dtype: fp4`, which makes the checkpoint a three-way split: the
    experts (~96% of the params) at 4-bit, embeddings and the untied LM head
    at bf16, and only what is left over at 8-bit. Pricing everything at
    quant_method roughly doubles the weight figure, which is enough to flip a
    multi-XPU verdict, so these paths are pinned here.

    Synthetic configs throughout: no network, no pinned revision, and the
    proportions (experts dominating) are what the arithmetic turns on rather
    than any one checkpoint's exact dimensions.
    """

    @staticmethod
    def _moe_cfg(**overrides):
        """MoE config with fp8 non-experts and fp4 experts."""
        cfg = {
            "hidden_size": 4096,
            "num_hidden_layers": 8,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1024,
            "n_routed_experts": 64,
            "num_experts_per_tok": 4,
            "n_shared_experts": 1,
            "vocab_size": 128000,
            "tie_word_embeddings": False,
            "expert_dtype": "fp4",
            "quantization_config": {"quant_method": "fp8"},
        }
        cfg.update(overrides)
        return cfg

    def _run_cli(self, cfg, capsys, extra_args=()):
        """Run fit.main() against a config written to a temp file.

        The matching-vs-differing --quant decision lives in main(), not in
        estimate(), so it can only be covered through the CLI entry point.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            argv = [
                "--model", str(path),
                "--device-vram-gb", "48",
            ] + list(extra_args)
            rc = fit.main(argv)
        return rc, capsys.readouterr().out

    def test_expert_dtype_detected_and_normalized(self):
        """expert_dtype is read top-level or from text_config, and aliased"""
        assert fit.expert_dtype_of(self._moe_cfg()) == "mxfp4"
        assert fit.expert_dtype_of({"text_config": {"expert_dtype": "fp4"}}) == "mxfp4"
        # NVFP4 is its own row: it scales per 16 values, not per 32, so it
        # cannot be folded into `fp4`.
        assert fit.expert_dtype_of({"expert_dtype": "nvfp4"}) == "nvfp4"
        assert fit.BYTES_PER_PARAM["nvfp4"] >= 0.5625
        assert fit.BYTES_PER_PARAM["nvfp4"] > fit.BYTES_PER_PARAM["fp4"]
        assert fit.expert_dtype_of({"expert_dtype": "MXFP4"}) == "mxfp4"
        # Absent, or a dtype with no price, must not silently become one
        assert fit.expert_dtype_of({}) is None
        assert fit.expert_dtype_of({"expert_dtype": "fp6"}) is None

    def test_unpriceable_expert_dtype_refuses(self, capsys):
        """"Declared but unpriceable" must not read as "absent"

        Falling through prices ~96% of the model at the base dtype: an fp6 (or
        anything wider) expert dtype read as fp8 is a false FITS.
        """
        cfg = self._moe_cfg(expert_dtype="fp6")
        assert fit.raw_expert_dtype(cfg) == "fp6", "the raw value is preserved"
        assert fit.expert_dtype_of(cfg) is None, "and it cannot be priced"

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with pytest.raises(SystemExit) as exc:
                fit.main(["--model", str(path), "--device-vram-gb", "48"])
            assert exc.value.code != 0
            err = capsys.readouterr().err
            assert "expert_dtype 'fp6'" in err
            assert "cannot price" in err

            # An explicit uniform --quant does not depend on the unknown
            # dtype, so that request is still answered.
            rc = fit.main(["--model", str(path), "--device-vram-gb", "48",
                           "--quant", "bf16"])
            out = capsys.readouterr().out
        assert rc == 0
        assert "Verdict:" in out
        assert "Expert weights" not in out

    def test_torch_dtype_spellings_are_priced(self, capsys):
        """float32/bfloat16/float16 are how an expert_dtype names itself"""
        assert fit.expert_dtype_of({"expert_dtype": "float32"}) == "fp32"
        assert fit.expert_dtype_of({"expert_dtype": "bfloat16"}) == "bf16"
        assert fit.expert_dtype_of({"expert_dtype": "float16"}) == "fp16"
        assert fit.BYTES_PER_PARAM["fp32"] == 4.00

        # fp32 experts are priced, not refused -- and wider than the base.
        cfg = self._moe_cfg(expert_dtype="float32")
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Expert weights:    fp32  (4.00 bytes/param)" in out

    def test_expert_dtype_from_nested_backbone_configs(self):
        """Multimodal layouts nest the backbone; a null text_config is legal"""
        # llm_config is the Nemotron-Omni spelling parse_dims() already accepts
        assert fit.expert_dtype_of({"llm_config": {"expert_dtype": "fp4"}}) == "mxfp4"
        # An explicit null must not be dereferenced
        assert fit.expert_dtype_of({"text_config": None}) is None
        assert fit.expert_dtype_of(
            {"text_config": None, "expert_dtype": "fp4"}) == "mxfp4"

    def test_expert_dtype_nested_config_reaches_the_verdict(self, capsys):
        """A nested expert_dtype must price the experts, not be dropped"""
        inner = self._moe_cfg()
        expert_dtype = inner.pop("expert_dtype")
        qcfg = inner.pop("quantization_config")
        cfg = {"text_config": None,
               "llm_config": dict(inner, expert_dtype=expert_dtype),
               "quantization_config": qcfg}
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Expert weights:    mxfp4" in out

    def test_router_row_does_not_double_bill_experts(self):
        """keep-routers plus expert_dtype must not price the experts twice

        modules_to_not_convert gives routers their own row, which comes out of
        the FFN row's param count. The FFN row's bytes have to match the params
        it reports, or the total is inflated and ffn_bpp contradicts ffn_params.
        """
        cfg = self._moe_cfg(quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": ["model.embed_tokens", "router"],
        })
        result = estimate(cfg, "fp8", "fp8", 4096, 1, 1, "vllm", 48.0,
                          expert_dtype="fp4")
        breakdown = result["mixed_breakdown"]
        assert breakdown["router_params"] > 0, "routers should have their own row"

        # The FFN row is priced at fp4 for exactly the params it claims.
        assert breakdown["ffn_bytes"] == int(breakdown["ffn_params"] * 0.55)
        assert abs(breakdown["ffn_bpp"] - 0.55) < 1e-9

        # Rows still total the Weights line, and account for each param once.
        component_sum = (breakdown["embed_bytes"] + breakdown["attn_bytes"] +
                         breakdown["router_bytes"] + breakdown["ffn_bytes"])
        assert abs(component_sum - result["weights"]) <= 1
        component_params = (breakdown["embed_params"] + breakdown["attn_params"] +
                            breakdown["router_params"] + breakdown["ffn_params"])
        assert component_params == result["params"]

    def test_expert_params_are_the_ffn_tensors(self):
        """expert_params() counts only the expert FFNs, and only for MoE"""
        d = parse_dims(self._moe_cfg())
        e_params = fit.expert_params(d)
        expected = d.num_layers * (d.num_experts + d.num_shared_experts) * \
            3 * d.hidden * d.intermediate
        assert e_params == expected
        # Experts dominate an MoE of these proportions, and the remainder
        # (non-expert weights) must stay positive for the split to be sane.
        params = count_params(d)
        assert 0 < e_params < params
        assert e_params / params > 0.5

        dense = parse_dims({"hidden_size": 2048, "num_hidden_layers": 4,
                            "num_attention_heads": 16, "intermediate_size": 8192,
                            "vocab_size": 32000})
        assert fit.expert_params(dense) == 0

    def test_experts_priced_at_expert_dtype(self):
        """Breakdown rows: bf16 embeddings, fp8 non-expert, fp4 experts"""
        cfg = self._moe_cfg()
        result = estimate(cfg, "fp8", "fp8", 4096, 1, 1, "vllm", 48.0,
                          expert_dtype="fp4")
        breakdown = result["mixed_breakdown"]
        assert breakdown is not None, "expert_dtype should produce a breakdown"

        assert breakdown["embed_bpp"] == 2.0, "embeddings stay bf16"
        assert breakdown["attn_bpp"] == 1.0, "non-expert weights at quant_method"
        assert breakdown["attn_label"] == "Non-expert"
        assert breakdown["ffn_bpp"] == 0.55, "experts at expert_dtype fp4"
        assert breakdown["ffn_params"] == fit.expert_params(parse_dims(cfg))

        # The printed rows must total the Weights line, or the report
        # contradicts itself.
        component_sum = (breakdown["embed_bytes"] + breakdown["attn_bytes"] +
                         breakdown["router_bytes"] + breakdown["ffn_bytes"])
        assert abs(component_sum - result["weights"]) <= 1

        # And the rows must account for every param exactly once.
        component_params = (breakdown["embed_params"] + breakdown["attn_params"] +
                            breakdown["router_params"] + breakdown["ffn_params"])
        assert component_params == result["params"]

    def test_expert_split_is_smaller_than_uniform(self):
        """The split must not be silently equivalent to uniform quant_method"""
        cfg = self._moe_cfg()
        d = parse_dims(cfg)
        split = estimate(cfg, "fp8", "fp8", 4096, 1, 1, "vllm", 48.0,
                         expert_dtype="fp4")["weights"]
        uniform = estimate(cfg, "fp8", "fp8", 4096, 1, 1, "vllm", 48.0)["weights"]
        assert uniform > split, "uniform fp8 should be the larger figure"

        # Both figures hold the embeddings at bf16 (fp8 is sub-16-bit), so the
        # gap is exactly one effect: the experts drop fp8 -> fp4. Pinning the
        # identity rather than a ratio keeps this independent of the
        # dimensions above.
        expected = uniform - fit.expert_params(d) * (1.00 - 0.55)
        assert abs(split - expected) <= 2

    def test_expert_split_divides_by_tp(self):
        """Every breakdown row and the total shard by --tp"""
        cfg = self._moe_cfg()
        one = estimate(cfg, "fp8", "fp8", 4096, 1, 1, "vllm", 48.0,
                       expert_dtype="fp4")
        four = estimate(cfg, "fp8", "fp8", 4096, 1, 4, "vllm", 48.0,
                        expert_dtype="fp4")
        assert abs(four["weights"] - one["weights"] // 4) <= 4
        for key in ("embed", "attn", "ffn"):
            assert abs(four["mixed_breakdown"][f"{key}_bytes"] -
                       one["mixed_breakdown"][f"{key}_bytes"] // 4) <= 1

    def test_cli_auto_detect_honors_expert_dtype(self, capsys):
        """No --quant: auto-detected fp8 + config fp4 experts"""
        rc, out = self._run_cli(self._moe_cfg(), capsys)
        assert rc == 0
        assert "Expert weights:    mxfp4" in out
        assert "hypothetical" not in out
        assert "Non-expert" in out

    def test_cli_matching_quant_keeps_expert_dtype(self, capsys):
        """--quant fp8 restates the config, so the fp4 experts survive"""
        rc, out = self._run_cli(self._moe_cfg(), capsys, ["--quant", "fp8"])
        assert rc == 0
        assert "Expert weights:    mxfp4" in out
        assert "hypothetical" not in out

    def test_expert_weights_line_defers_to_the_breakdown(self, capsys):
        """The status line must not claim every non-expert tensor is --quant

        Embeddings are always bf16, and modules_to_not_convert can hold
        attention or routers there too, so the line says "base dtype" and
        points at the breakdown that shows where each dtype landed.
        """
        _, out = self._run_cli(self._moe_cfg(), capsys)
        assert "base dtype for the rest" in out
        assert "only non-expert tensors are" not in out
        # The breakdown it defers to has to disagree with a flat reading.
        assert "Embeddings" in out and "2.00 B/p" in out

    def test_explicit_bf16_restates_a_config_with_no_quant_method(self, capsys):
        """--quant bf16 on an unquantized-base config is not an override

        With no priceable quant_method the non-expert tensors ship bf16, so
        typing that dtype restates the config the same way --quant fp8 does
        against a quant_method: fp8 checkpoint. The fp4 experts must survive.
        """
        cfg = self._moe_cfg()
        del cfg["quantization_config"]
        rc, out = self._run_cli(cfg, capsys, ["--quant", "bf16"])
        assert rc == 0
        assert "Expert weights:    mxfp4" in out
        assert "hypothetical" not in out

        # It matches the figure the auto-detected run reports, and is well
        # below a uniform-bf16 pricing of the same model.
        _, auto_out = self._run_cli(cfg, capsys)
        weights = re.search(r"^  Weights\s+([\d.]+) GB", out, re.M).group(1)
        assert re.search(r"^  Weights\s+" + weights + " GB", auto_out, re.M)
        uniform = count_params(parse_dims(cfg)) * 2.0
        assert float(weights) < uniform / GB * 0.75

    def test_explicit_quant_still_overrides_a_declared_quant_method(self, capsys):
        """bf16 is only a restatement when the config declares nothing

        Against quant_method: fp8 it is a real re-quantization request, so the
        split is still suppressed there.
        """
        _, out = self._run_cli(self._moe_cfg(), capsys, ["--quant", "bf16"])
        assert "hypothetical" in out
        assert "as shipped (fp8 + mxfp4 experts)" in out

    def test_unpriceable_quant_method_is_not_an_unquantized_checkpoint(self, capsys):
        """"No method declared" and "method I cannot price" are different facts

        An absent quantization_config means the checkpoint ships 16-bit, so
        `--quant bf16` restates it. A declared-but-unpriceable method means the
        shipped base is *unknown*: nothing can restate it, so an explicit dtype
        is an override, and no as-shipped figure can honestly be quoted.
        """
        cfg = self._moe_cfg(quantization_config={
            "quant_method": "compressed-tensors"})
        assert fit.declared_quant_method(cfg) == "compressed-tensors"
        assert fit.quant_from_config(cfg) == (None, False)

        # Auto: the fp4 experts are still known, the base is not.
        rc, auto = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Expert weights:    mxfp4" in auto
        assert "cannot price" in auto and "unknown" in auto

        # Explicit bf16 is an override here, not a restatement -- and the
        # as-shipped line refuses to invent a figure.
        _, forced = self._run_cli(cfg, capsys, ["--quant", "bf16"])
        assert "overrides an unknown base" in forced
        assert "as-shipped size unknown" in forced
        assert "as shipped (bf16" not in forced

        # With no quantization_config at all, bf16 *is* a restatement.
        absent = self._moe_cfg()
        del absent["quantization_config"]
        _, restated = self._run_cli(absent, capsys, ["--quant", "bf16"])
        assert "Expert weights:    mxfp4" in restated
        assert "hypothetical" not in restated

    def test_unpriceable_method_honors_the_refusal_escape_hatch(self, capsys):
        """An explicit --quant must answer, as the refusal message promises

        With both an unpriceable quant_method and an unpriceable expert_dtype,
        the auto path refuses. The error tells the caller that an explicit
        --quant gives a uniform estimate that does not depend on either -- so
        that path has to actually work, which it did not while an unknown base
        was treated as bf16.
        """
        cfg = self._moe_cfg(expert_dtype="fp6", quantization_config={
            "quant_method": "compressed-tensors"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with pytest.raises(SystemExit):
                fit.main(["--model", str(path), "--device-vram-gb", "48"])
            assert "cannot price" in capsys.readouterr().err

            rc = fit.main(["--model", str(path), "--device-vram-gb", "48",
                           "--quant", "bf16"])
            out = capsys.readouterr().out
        assert rc == 0
        assert "Verdict:" in out

    def test_shipped_figure_never_uses_the_suppressed_quant(self, capsys):
        """With no config quant_method, "as shipped" means bf16 + fp4

        The base for an as-shipped figure comes from the config. Reusing the
        --quant being suppressed would describe a checkpoint that does not
        exist -- an int8 + fp4 footprint for a model that declares neither.
        """
        cfg = self._moe_cfg()
        del cfg["quantization_config"]
        _, out = self._run_cli(cfg, capsys, ["--quant", "int8"])
        assert "as shipped (bf16 + mxfp4 experts)" in out
        assert "as shipped (int8" not in out

        # The figure itself must be the bf16-based one, not the int8-based one.
        d = parse_dims(cfg)
        params = count_params(d)
        bf16_based, _ = fit.calculate_mixed_precision_weights(
            cfg, d, params, "bf16", 1, "fp4")
        assert fit.fmt_gb(bf16_based) in out

    def test_shipped_figure_uses_config_quant_when_present(self, capsys):
        """With quant_method fp8 declared, "as shipped" is fp8 + fp4"""
        _, out = self._run_cli(self._moe_cfg(), capsys, ["--quant", "bf16"])
        assert "as shipped (fp8 + mxfp4 experts)" in out

    def test_cli_differing_quant_suppresses_with_shipped_figure(self, capsys):
        """--quant bf16 posits a re-quantization: suppressed, but labelled"""
        _, out = self._run_cli(self._moe_cfg(), capsys, ["--quant", "bf16"])
        assert "hypothetical" in out, "a re-quantization must be marked as one"
        assert "expert_dtype mxfp4" in out
        # The as-shipped figure has to appear next to it, or the hypothetical
        # reads as this checkpoint's real size.
        assert "as shipped" in out
        assert "drop --quant" in out
        # No breakdown rows: bf16 everywhere is a uniform calculation.
        assert "Non-expert" not in out

    def test_cli_expert_dtype_without_quantization_config(self, capsys):
        """expert_dtype with no priceable quant_method is still honored

        args.quant falls back to bf16 here, which is not the (absent)
        config quant. That is not a caller override, so the fp4 experts
        must survive and nothing may blame a --quant the caller never passed.
        """
        cfg = self._moe_cfg()
        del cfg["quantization_config"]
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Expert weights:    mxfp4" in out
        assert "hypothetical" not in out
        assert "drop --quant" not in out

    def test_expert_dtype_does_not_shrink_the_default_kv(self, capsys):
        """KV dtype is a launch flag, not a property of the weights

        The cache lives on the attention path, which `--quant` describes; the
        experts are not in it. Defaulting an fp4-expert config to fp8 KV would
        halve the KV figure on an assumption the runtime guidance does not
        make -- and a launch without --kv-cache-dtype fp8 would then need 2x.
        """
        cfg = self._moe_cfg()
        del cfg["quantization_config"]          # bf16 base, fp4 experts
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Quantization:      bf16" in out
        assert "KV dtype:          bf16" in out
        assert "not auto-paired to fp8" in out
        # The lever is named rather than silently applied.
        assert "pass --kv-dtype fp8" in out

        # And it is a lever: asking for it works.
        _, forced = self._run_cli(cfg, capsys, ["--kv-dtype", "fp8"])
        assert "kv_dtype fp8" in forced
        assert "auto-paired" not in forced

    def test_mxfp4_follows_the_documented_auto_kv(self, capsys):
        """vllm-xpu-run documents `auto` KV for MXFP4 and AutoRound"""
        for method, quant in (("mxfp4", "mxfp4"), ("gptq", "int3")):
            cfg = self._moe_cfg(quantization_config={
                "quant_method": method, "bits": 3} if method == "gptq"
                else {"quant_method": method})
            _, out = self._run_cli(cfg, capsys)
            assert f"Quantization:      {quant}" in out, method
            assert "KV dtype:          bf16" in out, method
            assert "not auto-paired to fp8" in out, method

    def test_int4_and_fp8_still_auto_pair(self, capsys):
        """The two rows the pack does document with fp8 KV"""
        for method, bits, quant in (("gptq", 4, "int4"), ("fp8", None, "fp8")):
            qcfg = {"quant_method": method}
            if bits:
                qcfg["bits"] = bits
            _, out = self._run_cli(self._moe_cfg(quantization_config=qcfg),
                                   capsys)
            assert f"auto-paired with --quant {quant}" in out, method
            # And the report says what that assumption costs if unmet.
            assert "--kv-cache-dtype fp8 or the cache costs 2x" in out, method

    @pytest.mark.parametrize("method", [
        "autoround", "auto-round", "auto_round", "intel/auto-round"])
    @pytest.mark.parametrize("bits,quant", [(4, "int4"), (8, "int8")])
    def test_autoround_keeps_the_documented_auto_kv(self, method, bits, quant,
                                                    capsys):
        """The width an AutoRound checkpoint reduces to does not pair its KV

        quant_from_config() turns `bits` into int8/int4, and those widths pair
        with fp8 KV on a gptq/awq checkpoint. AutoRound has its own row with
        `auto` KV, so pairing on the reduced width alone would halve the cache
        against the launch vllm-xpu-run documents -- a false FITS. The
        quantization method has to survive into the pairing decision.
        """
        cfg = self._moe_cfg(quantization_config={
            "quant_method": method, "bits": bits})
        _, out = self._run_cli(cfg, capsys)
        assert f"Quantization:      {quant}" in out
        assert "KV dtype:          bf16" in out
        assert "auto-paired with --quant" not in out
        # The method is what blames the pairing, not the width: the same width
        # pairs one row above.
        assert f"quant_method '{method}' is not one of the rows" in out
        assert "pass --kv-dtype fp8" in out

    def test_autoround_kv_is_not_halved(self, capsys):
        """The estimate and the verdict, not just the wording"""
        cfg = self._moe_cfg(quantization_config={
            "quant_method": "auto-round", "bits": 4})
        gptq = self._moe_cfg(quantization_config={
            "quant_method": "gptq", "bits": 4})
        # 18 GB sits between the two totals (15.98 GB with an fp8 cache,
        # 19.98 GB with a bf16 one), so the pairing decides the verdict.
        args = ["--ctx", "32768", "--concurrency", "8", "--device-vram-gb", "18"]
        rc_ar, out_ar = self._run_cli(cfg, capsys, args)
        rc_gptq, out_gptq = self._run_cli(gptq, capsys, args)
        assert "kv_dtype bf16" in out_ar and "kv_dtype fp8" in out_gptq
        # Same weights, twice the cache: the AutoRound launch needs the bytes
        # the gptq row's fp8 pairing saves.
        kv_ar = float(out_ar.split("KV cache")[1].split("GB")[0])
        kv_gptq = float(out_gptq.split("KV cache")[1].split("GB")[0])
        assert kv_ar == pytest.approx(kv_gptq * 2)
        assert rc_ar == 1 and "DOES NOT FIT" in out_ar
        assert rc_gptq == 0 and "FITS" in out_gptq

        # Asking for the lever explicitly still works, and then the figures
        # match the paired row.
        _, forced = self._run_cli(cfg, capsys, args + ["--kv-dtype", "fp8"])
        assert "kv_dtype fp8" in forced
        assert float(forced.split("KV cache")[1].split("GB")[0]) == kv_gptq

    def test_explicit_quant_overrides_the_autoround_row(self, capsys):
        """A differing --quant is a hypothetical the config's method left

        `--quant fp8` on an AutoRound checkpoint asks what an fp8 build would
        cost, and the fp8 row does pair fp8 KV. Restating the shipped width
        (`--quant int4`) is still the AutoRound checkpoint, so it does not.
        """
        cfg = self._moe_cfg(quantization_config={
            "quant_method": "auto-round", "bits": 4})
        _, hypothetical = self._run_cli(cfg, capsys, ["--quant", "fp8"])
        assert "auto-paired with --quant fp8" in hypothetical
        _, restated = self._run_cli(cfg, capsys, ["--quant", "int4"])
        assert "KV dtype:          bf16" in restated
        assert "auto-paired with --quant" not in restated

    @pytest.mark.parametrize("runtime,expect_fp8,flag", [
        ("vllm", True, "--kv-cache-dtype fp8"),
        ("sglang", False, "--kv-cache-dtype fp8_e4m3"),
        ("torch", False, None),
    ])
    def test_kv_pairing_is_per_runtime(self, runtime, expect_fp8, flag, capsys):
        """Only vLLM's guidance pairs quantized weights with an fp8 cache

        sglang-xpu-run documents a BF16 cache for fp8 weights unless
        `--kv-cache-dtype fp8_e4m3` is set, and the torch path has no KV-dtype
        flag at all. Pairing on those runtimes halves the estimated cache
        against their own default -- the false-FITS direction -- and quotes a
        flag that is wrong or nonexistent.
        """
        cfg = self._moe_cfg()          # quant_method fp8
        _, out = self._run_cli(cfg, capsys, ["--runtime", runtime])
        if expect_fp8:
            assert "KV dtype:          fp8" in out
            assert f"Launch with {flag}" in out
        else:
            assert "KV dtype:          bf16" in out
            assert "auto-paired with --quant" not in out
            if flag:
                assert flag in out, "the lever must name this runtime's flag"
            else:
                assert "no KV-dtype flag" in out
                assert "--kv-cache-dtype" not in out

    def test_sglang_and_torch_kv_is_not_halved(self):
        """The estimate itself, not just the wording"""
        cfg = self._moe_cfg()
        vllm = estimate(cfg, "fp8", "fp8", 32768, 4, 1, "vllm", 48.0)
        sgl = estimate(cfg, "fp8", "bf16", 32768, 4, 1, "sglang", 48.0)
        assert sgl["kv"] == vllm["kv"] * 2, "bf16 cache is twice the fp8 one"

    def test_kv_pairing_names_the_base_quant(self, capsys):
        """An fp8 base pairs with fp8 KV, and the line names --quant"""
        _, out = self._run_cli(self._moe_cfg(), capsys)
        assert "KV dtype:          fp8" in out
        assert "auto-paired with --quant fp8" in out
        assert "expert_dtype" not in out.split("KV dtype:")[1].split("\n")[0]

    def test_candidate_matching_shipped_base_keeps_the_split(self, capsys):
        """A candidate that restates the config keeps its expert dtype

        From a failed --quant bf16 run, the suggested --quant fp8 restores the
        fp4 experts -- so pricing that candidate uniformly quotes a saving the
        suggested command does not reproduce. Reachable because the current
        dtype here is *wider* than the config's.
        """
        cfg = self._moe_cfg()          # quant_method fp8 + expert_dtype fp4
        rc, wide = self._run_cli(cfg, capsys,
                                 ["--quant", "bf16", "--device-vram-gb", "4"])
        assert rc == 1 and "Binding constraint: weights" in wide

        quoted = re.search(r"quant bf16 -> fp8 \(saves\s+([\d.]+) GB\)", wide)
        assert quoted, f"expected an fp8 mitigation in:\n{wide}"

        def weights_gb(text):
            return float(re.search(r"^  Weights\s+([\d.]+) GB", text,
                                   re.M).group(1))

        _, at_fp8 = self._run_cli(cfg, capsys,
                                  ["--quant", "fp8", "--device-vram-gb", "4"])
        assert "Expert weights:    mxfp4" in at_fp8, "fp8 restates the config"
        assert abs((weights_gb(wide) - weights_gb(at_fp8))
                   - float(quoted.group(1))) <= 0.02

        # A candidate that does differ is still priced uniformly.
        int4_quoted = re.search(r"quant bf16 -> int4 \(saves\s+([\d.]+) GB\)",
                                wide)
        _, at_int4 = self._run_cli(cfg, capsys,
                                   ["--quant", "int4", "--device-vram-gb", "4"])
        assert "hypothetical" in at_int4
        assert abs((weights_gb(wide) - weights_gb(at_int4))
                   - float(int4_quoted.group(1))) <= 0.02

    def test_quant_mitigation_saving_is_reproducible(self, capsys):
        """The suggested --quant must produce the saving that was quoted

        A narrower --quant is a differing --quant, which suppresses the
        expert split. Pricing the candidate with the split retained would
        quote a saving the suggested command cannot reproduce.
        """
        cfg = self._moe_cfg()
        # 4 GB cannot hold the 5.62 GB of weights, so the verdict is
        # weights-bound and the quant mitigations are printed.
        rc, out = self._run_cli(cfg, capsys, ["--device-vram-gb", "4"])
        assert rc == 1
        assert "Binding constraint: weights" in out

        def weights_gb(text):
            return float(re.search(r"^  Weights\s+([\d.]+) GB", text,
                                   re.M).group(1))

        before = weights_gb(out)
        quoted = re.search(r"quant fp8 -> int4 \(saves\s+([\d.]+) GB\)", out)
        assert quoted, f"expected an int4 mitigation in:\n{out}"

        # Run the command that was suggested and compare what it reports.
        _, after_out = self._run_cli(cfg, capsys,
                                     ["--device-vram-gb", "4", "--quant", "int4"])
        after = weights_gb(after_out)
        assert abs((before - after) - float(quoted.group(1))) <= 0.02

    @staticmethod
    def _dense_cfg(**overrides):
        cfg = {
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 6144,
            "vocab_size": 32000,
            "tie_word_embeddings": False,
            "expert_dtype": "fp4",
        }
        cfg.update(overrides)
        return cfg

    def test_mxfp4_experts_are_not_flagged_for_kernel_path(self, capsys):
        """DeepSeek's `expert_dtype: fp4` is MXFP4, which XPU does run

        vLLM's DeepSeek-V4 quant config reads `expert_dtype="fp4"` as "MXFP4
        experts with ue8m0 FP8 linear scales" and dispatches it to
        Mxfp4MoEMethod, which selects Mxfp4MoeBackend.XPU and the same
        XPUExpertsMxFp4 kernel `quant_method: mxfp4` reaches. A served
        gpt-oss-20b allocated 12.87 GiB against a 13.14 GiB mxfp4 estimate, so
        those weights measurably stayed packed (the DeepSeek spelling is
        verified by dispatch inspection, not by serving that checkpoint here).
        Warning that such a verdict is bytes-only would be false on the
        flagship MoE.
        """
        _, out = self._run_cli(self._moe_cfg(), capsys)
        assert "Expert weights:    mxfp4" in out
        assert "Kernel path:" not in out
        assert "no documented XPU kernel" not in out

        # NVFP4 is a different format (per-16 fp8 scales), is not on the
        # pack's XPU table, and has no measured run here -- still flagged.
        nvfp4 = self._moe_cfg(expert_dtype="nvfp4")
        _, nv_out = self._run_cli(nvfp4, capsys)
        assert "Expert weights:    nvfp4" in nv_out
        assert "nvfp4 has no documented XPU kernel" in nv_out

    def test_dense_model_ignores_expert_dtype(self):
        """A stray expert_dtype on a dense config must not create experts"""
        cfg = self._dense_cfg()
        result = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 32.0,
                          expert_dtype="fp4")
        assert result["mixed_breakdown"] is None
        params = count_params(parse_dims(cfg))
        assert result["weights"] == int(params * 2.0)

    def test_cli_reports_no_experts_for_a_dense_config(self, capsys):
        """A dense model has no expert weights to report, stray dtype or not

        The weight figure is right either way -- expert_params() is 0 -- but
        an "Expert weights" line for a model with no experts is not.
        """
        rc, out = self._run_cli(self._dense_cfg(), capsys)
        assert rc == 0
        assert "Expert weights" not in out
        assert "fp4" not in out

    def test_cli_no_expert_hypothetical_for_a_dense_config(self, capsys):
        """Nor may a differing --quant print a suppressed-expert hypothetical"""
        rc, out = self._run_cli(self._dense_cfg(), capsys, ["--quant", "int4"])
        assert rc == 0
        assert "Expert weights" not in out
        assert "hypothetical" not in out
        assert "as shipped" not in out

    def test_null_quantization_config_reaches_a_verdict(self, capsys):
        """An explicit "quantization_config": null must not crash the verdict

        quant_from_config() tolerates it, but the weight calculation reads the
        same key, so the CLI has to survive it end to end -- a helper-level
        assertion alone would miss that.
        """
        cfg = self._dense_cfg(quantization_config=None)
        cfg.pop("expert_dtype")
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Verdict:" in out
        assert "Quantization:      bf16" in out

        # Same for the library entry point, at both the uniform and the
        # expert-split paths.
        result = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 32.0)
        assert result["weights"] == int(count_params(parse_dims(cfg)) * 2.0)
        moe = self._moe_cfg(quantization_config=None)
        assert estimate(moe, "fp8", "fp8", 4096, 1, 1, "vllm", 48.0,
                        expert_dtype="fp4")["mixed_breakdown"] is not None

    def test_null_modules_to_not_convert_reaches_a_verdict(self, capsys):
        """A null modules_to_not_convert is the same class of config bug"""
        cfg = self._moe_cfg(quantization_config={
            "quant_method": "fp8", "modules_to_not_convert": None})
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Verdict:" in out


class TestKVReplicationAboveKVHeads:
    """
    KV divides by min(tp, num_key_value_heads), not by tp.

    Attention shards by KV head. Past the KV-head count the runtime
    replicates the group instead of splitting it further, so every rank keeps
    one head: a 4-KV-head model at TP 8 stores total_KV/4 per device. Dividing
    by TP there understates KV, and understating is the direction that reports
    FITS for a launch that OOMs.
    """

    @staticmethod
    def _cfg(num_kv_heads):
        return {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": 11008,
            "vocab_size": 32000,
        }

    def test_kv_shards_caps_at_the_kv_head_count(self):
        d = parse_dims(self._cfg(4))
        assert fit.kv_shards(d, 1) == 1
        assert fit.kv_shards(d, 2) == 2
        assert fit.kv_shards(d, 4) == 4
        assert fit.kv_shards(d, 8) == 4, "8 ranks still hold one head each"
        assert fit.kv_shards(d, 16) == 4

    def test_kv_stops_shrinking_past_the_kv_head_count(self):
        """TP 8 on a 4-KV-head model must report the same KV as TP 4"""
        cfg = self._cfg(4)
        kv_tp4 = estimate(cfg, "bf16", "bf16", 32768, 8, 4, "vllm", 32.0)["kv"]
        kv_tp8 = estimate(cfg, "bf16", "bf16", 32768, 8, 8, "vllm", 32.0)["kv"]
        assert kv_tp8 == kv_tp4

        # And it equals one KV head's worth of the model total.
        d = parse_dims(cfg)
        total = kv_bytes(d, 32768, 8, "bf16")
        assert kv_tp8 == total // 4
        assert kv_tp8 > total // 8, "dividing by TP would understate KV here"

    def test_kv_still_divides_by_tp_up_to_the_kv_head_count(self):
        """The common case is unchanged: 8 KV heads shard 8 ways at TP 8"""
        cfg = self._cfg(8)
        d = parse_dims(cfg)
        total = kv_bytes(d, 4096, 1, "bf16")
        for tp in (1, 2, 4, 8):
            kv = estimate(cfg, "bf16", "bf16", 4096, 1, tp, "vllm", 32.0)["kv"]
            assert kv == total // tp

    def test_capacity_ceilings_use_the_same_divisor(self):
        """max_context must be priced with the replicated per-token KV

        The ceiling is free VRAM divided by per-token KV, so it inherits the
        divisor. Deriving it here from the KV-head count is what catches the
        ceiling drifting away from the `kv` figure printed beside it.
        """
        cfg = self._cfg(4)
        d = parse_dims(cfg)
        at_tp8 = estimate(cfg, "bf16", "bf16", 4096, 1, 8, "vllm", 32.0)

        free = (at_tp8["usable_vram"] - at_tp8["weights"] - at_tp8["act"] -
                at_tp8["framework"])
        per_token = kv_bytes(d, 1, 1, "bf16") // 4   # 4 KV heads, not TP 8
        assert at_tp8["max_context"] == free // per_token
        assert at_tp8["max_concurrency"] == free // (per_token * 4096)

    def test_layer_shards_caps_at_the_layer_count(self):
        """device_map places whole layers, so gains stop at the layer count

        A 4-layer model on 8 devices still keeps a whole layer on each
        nonempty one: the busiest device holds total/4, and the spare devices
        are idle. An uneven split floors, so the estimate errs heavy.
        """
        def dims(layers):
            return parse_dims(dict(self._cfg(8), num_hidden_layers=layers))

        assert fit.layer_shards(dims(32), 8) == 8, "even split is unchanged"
        assert fit.layer_shards(dims(32), 1) == 1
        assert fit.layer_shards(dims(4), 8) == 4, "not 8: one layer per device"
        assert fit.layer_shards(dims(4), 16) == 4, "gains cap, they do not grow"
        # 33 over 8 leaves one device holding 5 layers, i.e. total/6.6; the
        # floored divisor of 6 overstates that device rather than understating.
        assert fit.layer_shards(dims(33), 8) == 6

    def test_torch_kv_and_weights_stop_shrinking_past_the_layer_count(self):
        """Both halves of the footprint respect layer granularity"""
        cfg = dict(self._cfg(8), num_hidden_layers=4)
        d = parse_dims(cfg)
        at_4 = estimate(cfg, "bf16", "bf16", 32768, 8, 4, "torch", 32.0)
        at_8 = estimate(cfg, "bf16", "bf16", 32768, 8, 8, "torch", 32.0)
        at_16 = estimate(cfg, "bf16", "bf16", 32768, 8, 16, "torch", 32.0)
        assert at_8["kv"] == at_4["kv"] == at_16["kv"]
        assert at_8["weights"] == at_4["weights"] == at_16["weights"]

        # And they equal one layer's share, not one eighth.
        assert at_8["kv"] == kv_bytes(d, 32768, 8, "bf16") // 4
        assert at_8["weights"] > int(count_params(d) * 2.0) // 8

        # A head-sharded runtime is unaffected by layer granularity: it splits
        # the matrices themselves.
        vllm_8 = estimate(cfg, "bf16", "bf16", 32768, 8, 8, "vllm", 32.0)
        assert vllm_8["weights"] == int(count_params(d) * 2.0) // 8

    def test_placement_floor_is_the_largest_unsplittable_tensor(self, capsys):
        """No device can hold less than the embedding matrix

        device_map assigns that matrix whole, so a wide-vocab, few-layer model
        divided by its layer count can land below a tensor it must
        materialise. Left unfloored this reports FITS for a launch that OOMs.
        """
        cfg = {
            "hidden_size": 8192, "num_hidden_layers": 4,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 11008, "vocab_size": 262144,
            "tie_word_embeddings": False,
        }
        d = parse_dims(cfg)
        embed_matrix = 262144 * 8192 * 2.0
        even_share = int(count_params(d) * 2.0) // 4
        assert even_share < embed_matrix, "premise: the share is below one matrix"

        torch_r = estimate(cfg, "bf16", "bf16", 4096, 1, 4, "torch", 16.0)
        assert torch_r["weights"] == int(embed_matrix)
        assert torch_r["weights_floor"] == int(embed_matrix)

        # Head-sharded runtimes do split the vocab dimension, so no floor.
        vllm_r = estimate(cfg, "bf16", "bf16", 4096, 1, 4, "vllm", 16.0)
        assert vllm_r["weights"] == even_share
        assert vllm_r["weights_floor"] == 0

        # It changes verdicts, not just numbers.
        assert not estimate(cfg, "bf16", "bf16", 4096, 1, 4,
                            "torch", 5.0)["fits"]
        unfloored_total = (torch_r["total"] - torch_r["weights"] + even_share)
        assert unfloored_total <= 5.0 * GB, "would have said FITS unfloored"

        # And the report explains where the number came from.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            fit.main(["--model", str(path), "--device-vram-gb", "16",
                      "--tp", "4", "--runtime", "torch"])
        out = capsys.readouterr().out
        assert "Placement floor:" in out
        assert "262144x8192 embedding matrix" in out

    def test_floored_weights_are_not_scaled_by_device_count(self, capsys):
        """Widening placement cannot split the embedding, so no saving

        Once weights are the floor, dividing that figure by a bigger divisor
        invents a saving no device count can deliver.
        """
        cfg = {
            "hidden_size": 8192, "num_hidden_layers": 4,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 11008, "vocab_size": 262144,
            "tie_word_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            base = ["--model", str(path), "--device-vram-gb", "3",
                    "--runtime", "torch", "--tp", "4"]
            rc = fit.main(base)
            out = capsys.readouterr().out

            assert rc == 1 and "Placement floor:" in out
            assert "Binding constraint: weights" in out
            assert "splits weights across" not in out, \
                "the floor cannot be divided by more devices"

            # Nor is a quant mitigation offered: the binding constraint is a
            # bf16 embedding matrix, and quantizing Linear layers cannot
            # shrink it, so every candidate reports the same floored figure.
            weights = float(re.search(r"^  Weights\s+([\d.]+) GB", out,
                                      re.M).group(1))
            assert "quant bf16 ->" not in out
            assert "too big for this device" in out

            for q in ("fp8", "int4", "int3"):
                fit.main(base + ["--quant", q])
                at_q = float(re.search(r"^  Weights\s+([\d.]+) GB",
                                       capsys.readouterr().out, re.M).group(1))
                assert abs(at_q - weights) <= 0.02, q

    def test_placement_floor_respects_the_component_dtype(self):
        """A quantized run's floor uses the dtype the embeddings are held at"""
        cfg = {
            "hidden_size": 8192, "num_hidden_layers": 4,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 11008, "vocab_size": 262144,
            "tie_word_embeddings": False,
            "quantization_config": {
                "quant_method": "int4",
                "modules_to_not_convert": ["model.embed_tokens", "lm_head"],
            },
        }
        # Embeddings excluded from quantization: the floor is the bf16 matrix.
        held_at_bf16 = estimate(cfg, "int4", "fp8", 4096, 1, 4, "torch", 32.0)
        assert held_at_bf16["weights"] == int(262144 * 8192 * 2.0)

        # And with no modules_to_not_convert at all: still bf16, because a
        # quantizer would not have touched the embedding matrix either way.
        uniform = estimate({k: v for k, v in cfg.items()
                            if k != "quantization_config"},
                           "int4", "fp8", 4096, 1, 4, "torch", 32.0)
        assert uniform["weights"] == int(262144 * 8192 * 2.0)
        assert uniform["weights"] == held_at_bf16["weights"]

        # A 16-bit run has nothing held wide, so the floor is that dtype.
        as_bf16 = estimate({k: v for k, v in cfg.items()
                            if k != "quantization_config"},
                           "bf16", "bf16", 4096, 1, 4, "torch", 32.0)
        assert as_bf16["weights_floor"] == int(262144 * 8192 * 2.0)

    def test_torch_wider_device_suggestion_is_bounded(self, capsys):
        """No doubling past the layer count, and the saving is reproducible"""
        cfg = {
            "hidden_size": 8192, "num_hidden_layers": 4,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 28672, "vocab_size": 128000,
            "tie_word_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            base = ["--model", str(path), "--device-vram-gb", "4",
                    "--runtime", "torch"]
            fit.main(base + ["--tp", "2"])
            at_2 = capsys.readouterr().out
            fit.main(base + ["--tp", "4"])
            at_4 = capsys.readouterr().out

        def weights_gb(text):
            return float(re.search(r"^  Weights\s+([\d.]+) GB", text,
                                   re.M).group(1))

        # TP 2 -> 4 buys a real split, and the quoted saving is what TP 4
        # actually reports.
        quoted = re.search(r"--tp 4 splits weights across 4 XPUs "
                           r"\(saves\s+([\d.]+) GB", at_2)
        assert quoted, f"expected a --tp 4 suggestion in:\n{at_2}"
        assert abs((weights_gb(at_2) - weights_gb(at_4))
                   - float(quoted.group(1))) <= 0.02

        # At TP 4 the model is out of layers, so no wider suggestion at all --
        # doubling to 8 would leave four devices idle and save nothing.
        assert "Binding constraint: weights" in at_4
        assert "splits weights across" not in at_4

    def test_kv_divisor_is_runtime_aware(self):
        """Head capping is a vLLM/SGLang rule; device_map splits by count"""
        d = self._cfg(4) and parse_dims(self._cfg(4))
        assert fit.kv_divisor(d, 8, "vllm") == 4
        assert fit.kv_divisor(d, 8, "sglang") == 4
        assert fit.kv_divisor(d, 8, "torch") == 8
        assert fit.kv_divisor(d, 2, "torch") == 2

    def test_kv_mitigations_quote_the_verdict_divisor(self, capsys):
        """A torch KV-bound verdict and its savings must use one divisor

        With 4 KV heads at TP 8 the two divisors differ by 2x, so a mitigation
        computed with the head-capped one would offer to save the entire cache
        twice over.
        """
        cfg = {
            "hidden_size": 4096, "head_dim": 128, "num_hidden_layers": 80,
            "num_attention_heads": 32, "num_key_value_heads": 4,
            "intermediate_size": 11008, "vocab_size": 32000,
        }
        d = parse_dims(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            fit.main(["--model", str(path), "--device-vram-gb", "12",
                      "--tp", "8", "--ctx", "65536", "--concurrency", "8",
                      "--kv-dtype", "bf16", "--runtime", "torch"])
        out = capsys.readouterr().out
        assert "Binding constraint: KV cache" in out

        kv = float(re.search(r"^  KV cache\s+([\d.]+) GB", out, re.M).group(1))
        halved = fit.kv_bytes(d, 32768, 8, "bf16") // 8   # tp divisor, torch
        quoted = float(re.search(r"drop ctx 65536 -> 32768 \(saves\s+([\d.]+) GB\)",
                                 out).group(1))
        assert abs(quoted - (kv - halved / GB)) < 0.02
        # Halving the context cannot save the whole cache.
        assert quoted < kv

    def test_single_kv_head_model_never_shards_kv(self):
        """MQA: one KV head, so KV is identical at every TP"""
        cfg = self._cfg(1)
        kvs = {estimate(cfg, "bf16", "bf16", 4096, 1, tp, "vllm", 32.0)["kv"]
               for tp in (1, 2, 4, 8)}
        assert len(kvs) == 1


class TestQuantMethodWidth:
    """
    Auto-detecting the weight dtype from `quantization_config`.

    AWQ, GPTQ and AutoRound are algorithm names, not widths -- each ships
    8-, 4-, 3- and 2-bit checkpoints -- so `bits` decides. Reading the method
    name alone prices an Int8 GPTQ checkpoint at int4, understating weights by
    nearly 2x, which is the direction that turns an OOM into a FITS verdict.
    """

    @staticmethod
    def _cfg(qcfg):
        return {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 128000,
            "tie_word_embeddings": False,
            "quantization_config": qcfg,
        }

    @pytest.mark.parametrize("method", ["gptq", "awq", "autoround"])
    @pytest.mark.parametrize("bits,expected", [(8, "int8"), (4, "int4"),
                                               (3, "int3"), (2, "int2")])
    def test_width_comes_from_bits(self, method, bits, expected):
        """Every width-parameterized method reads its width from bits"""
        cfg = self._cfg({"quant_method": method, "bits": bits})
        assert fit.quant_from_config(cfg) == (expected, False)

    def test_older_awq_w_bit_spelling(self):
        """Older AWQ configs spell the width w_bit"""
        cfg = self._cfg({"quant_method": "awq", "w_bit": 4})
        assert fit.quant_from_config(cfg) == ("int4", False)

    def test_int8_gptq_is_not_priced_as_int4(self):
        """The regression this guards: an Int8 GPTQ checkpoint at ~1 B/param"""
        cfg = self._cfg({"quant_method": "gptq", "bits": 8})
        d = parse_dims(cfg)
        params = count_params(d)
        embed_p = d.vocab * d.hidden * (1 if d.tied else 2)
        result = estimate(cfg, "int8", "fp8", 4096, 1, 1, "vllm", 32.0)
        # int8 for the Linear layers, bf16 for the embedding matrix.
        assert result["weights"] == int(embed_p * 2.0
                                        + (params - embed_p) * 1.00)
        # int4 would be ~55% of the Linear share; 8-bit must not read as 4-bit.
        assert result["weights"] > int(params * 0.9)

    def test_missing_width_assumes_the_widest(self):
        """No usable bits: assume int8 and flag that the width was assumed"""
        assert fit.quant_from_config(
            self._cfg({"quant_method": "gptq"})) == ("int8", True)
        # A width this script does not price is the same situation
        assert fit.quant_from_config(
            self._cfg({"quant_method": "gptq", "bits": 5})) == ("int8", True)
        assert fit.quant_from_config(
            self._cfg({"quant_method": "awq", "bits": "four"})) == ("int8", True)

    def test_cli_says_when_the_width_was_assumed(self, capsys):
        """An assumed width must be visible, with the override named"""
        cfg = self._cfg({"quant_method": "gptq"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            fit.main(["--model", str(path), "--device-vram-gb", "32"])
        out = capsys.readouterr().out
        assert "Quantization:      int8" in out
        assert "without a usable" in out
        assert "--quant" in out

    def test_cli_stays_quiet_when_bits_is_known(self, capsys):
        """A declared width is not an assumption, so no note"""
        cfg = self._cfg({"quant_method": "gptq", "bits": 4})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            fit.main(["--model", str(path), "--device-vram-gb", "32"])
        out = capsys.readouterr().out
        assert "Quantization:      int4" in out
        assert "without a usable" not in out

    def test_nvfp4_is_priced_above_its_packed_floor(self):
        """NVFP4 scales per 16 values, so 0.55 is below what it can be

        0.5 for the fp4 payload plus 1/16 for the fp8 block scale is 0.5625
        B/param before the per-tensor scale; pricing it at the `fp4` row would
        understate every NVFP4 checkpoint.
        """
        cfg = self._cfg({"quant_method": "nvfp4"})
        d = parse_dims(cfg)
        params = count_params(d)
        embed_p = d.vocab * d.hidden * (1 if d.tied else 2)
        result = estimate(cfg, "nvfp4", "fp8", 4096, 1, 1, "vllm", 32.0)
        linear = params - embed_p
        assert result["weights"] == int(embed_p * 2.0
                                        + linear * fit.BYTES_PER_PARAM["nvfp4"])
        assert result["weights"] > int(linear * 0.5625)

        # Strictly heavier than the 32-value-scaling fp4 row.
        fp4 = estimate(self._cfg({"quant_method": "fp4"}), "fp4", "fp8",
                       4096, 1, 1, "vllm", 32.0)
        assert result["weights"] > fp4["weights"]

    def test_no_xpu_kernel_path_is_flagged(self, capsys):
        """An fp4/nvfp4 verdict must say the kernel path is undocumented

        The byte math is worth reporting, but vllm-xpu-run's quant table has
        no row for either format, so a load refuses or upcasts -- and an
        upcast makes the printed weights 3-4x too small.
        """
        cfg = self._cfg({"quant_method": "modelopt_fp4"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            rc = fit.main(["--model", str(path), "--device-vram-gb", "16"])
        out = capsys.readouterr().out
        assert rc == 0, "still a verdict, just a qualified one"
        assert "Kernel path:" in out
        assert "nvfp4 has no documented XPU kernel" in out
        assert "vllm-xpu-run" in out

        # The upcast figure has to be the bf16 one, since that decides the fit
        # if the runtime does not refuse.
        d = parse_dims(cfg)
        upcast = int(count_params(d) * 2.0)
        assert fit.fmt_gb(upcast) in out

    def test_caveat_routes_to_the_selected_runtime(self, capsys):
        """A vLLM load says nothing about SGLang's or torch's kernels"""
        cfg = self._cfg({"quant_method": "nvfp4"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            for runtime in ("vllm", "sglang", "torch"):
                fit.main(["--model", str(path), "--device-vram-gb", "32",
                          "--runtime", runtime])
                out = capsys.readouterr().out
                assert f"**{runtime}-xpu-run**" in out, runtime
                for other in {"vllm", "sglang", "torch"} - {runtime}:
                    assert f"**{other}-xpu-run**" not in out, (runtime, other)

    def test_upcast_figure_respects_the_placement_floor(self, capsys):
        """The quoted bf16 upcast must not undercut a mandatory bf16 tensor"""
        cfg = {
            "hidden_size": 8192, "num_hidden_layers": 4,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 11008, "vocab_size": 262144,
            "tie_word_embeddings": False,
            "quantization_config": {"quant_method": "nvfp4"},
        }
        embed_bf16 = int(262144 * 8192 * 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            fit.main(["--model", str(path), "--device-vram-gb", "16",
                      "--tp", "4", "--runtime", "torch"])
        out = capsys.readouterr().out
        quoted = re.search(r"upcast to bf16 puts weights at\s+([\d.]+) GB", out)
        assert quoted, out
        # An unfloored total/4 would be ~2.82 GB, below the 4.00 GB matrix.
        assert abs(float(quoted.group(1)) - embed_bf16 / GB) < 0.02
        assert float(quoted.group(1)) == pytest.approx(
            estimate(cfg, "bf16", "fp8", 4096, 1, 4, "torch",
                     16.0)["weights"] / GB, abs=0.01)

    def test_supported_dtypes_are_not_flagged(self, capsys):
        """No caveat for a dtype the XPU quant table does cover"""
        for method in ("fp8", "mxfp4", "gptq"):
            cfg = self._cfg({"quant_method": method, "bits": 4})
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f)
                fit.main(["--model", str(path), "--device-vram-gb", "32"])
            assert "Kernel path:" not in capsys.readouterr().out, method

    def test_name_fixed_methods_still_normalize(self):
        """Methods whose name fixes the width are unchanged"""
        assert fit.quant_from_config(
            self._cfg({"quant_method": "nvfp4"})) == ("nvfp4", False)
        assert fit.quant_from_config(
            self._cfg({"quant_method": "modelopt_fp4"})) == ("nvfp4", False)
        assert fit.quant_from_config(
            self._cfg({"quant_method": "MXFP4"})) == ("mxfp4", False)
        assert fit.quant_from_config(
            self._cfg({"quant_method": "fp8"})) == ("fp8", False)

    def test_unpriceable_or_absent_method_is_none(self):
        """An unknown method defaults to bf16 via a None, not a guess"""
        assert fit.quant_from_config(
            self._cfg({"quant_method": "compressed-tensors"})) == (None, False)
        assert fit.quant_from_config(self._cfg({})) == (None, False)
        assert fit.quant_from_config({"hidden_size": 4096}) == (None, False)
        assert fit.quant_from_config({"quantization_config": None}) == (None, False)


class TestShardableTPSuggestion:
    """
    The `--tp N` mitigation on a weights-bound DOES NOT FIT verdict.

    Runtimes shard attention by head, so a TP that does not divide the head
    count is rejected at engine init. Doubling the current TP names such a
    launch whenever the head count is not a power of two.
    """

    @staticmethod
    def _cfg_dict():
        return {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "vocab_size": 32000,
        }

    @staticmethod
    def _dims(num_attn_heads, num_kv_heads=None, intermediate=12288):
        # 12288 divides by every TP these tests reach, so a case fails on the
        # head counts it is about rather than on the FFN width.
        return parse_dims({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": num_attn_heads,
            "num_key_value_heads": num_kv_heads or num_attn_heads,
            "intermediate_size": intermediate,
            "vocab_size": 32000,
        })

    def test_power_of_two_head_count_doubles(self):
        """The common case is unchanged: 32 heads at TP 1/2/4 -> 2/4/8"""
        assert fit.next_shardable_tp(self._dims(32), 1) == 2
        assert fit.next_shardable_tp(self._dims(32), 2) == 4
        assert fit.next_shardable_tp(self._dims(32, 8), 4) == 8

    def test_non_power_of_two_head_count_skips_invalid_tp(self):
        """12 heads is valid at TP 4 but not TP 8, so 6 is the next one"""
        assert fit.next_shardable_tp(self._dims(12, 2), 4) == 6
        assert fit.next_shardable_tp(self._dims(12, 2), 6) == 12

    def test_shardable_rule_needs_both_head_counts(self):
        """12 query heads with 8 KV heads is valid at 4, not at 6

        6 divides the query heads, but 8 is neither a multiple nor a divisor
        of 6, so the runtime cannot lay the KV group out across 6 ranks.
        """
        d = self._dims(12, 8)
        assert fit.tp_is_shardable(d, 4)
        assert not fit.tp_is_shardable(d, 6)
        assert fit.next_shardable_tp(d, 4) is None, "nothing above 4 qualifies"

        # Replication above the KV-head count is still allowed.
        assert fit.tp_is_shardable(self._dims(32, 4), 8)
        # And a TP that does not divide the query heads never is.
        assert not fit.tp_is_shardable(self._dims(28, 4), 8)
        assert not fit.tp_is_shardable(self._dims(32, 8), 0)

    def test_torch_runtime_warns_instead_of_refusing(self, capsys):
        """device_map places whole modules, so head divisibility is not its rule

        The refusal is a vLLM/SGLang contract. Under --runtime torch the same
        --tp is a legitimate planning request and must still produce a verdict.
        """
        cfg = {
            "hidden_size": 1536,
            "num_hidden_layers": 8,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 28672,
            "vocab_size": 32000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            argv = ["--model", str(path), "--device-vram-gb", "16", "--tp", "6"]
            rc = fit.main(argv + ["--runtime", "torch"])
            out = capsys.readouterr().out
            assert rc == 0
            assert "Verdict:" in out
            assert "does not divide this model's shard dimensions" in out
            assert "Estimating anyway" in out

            # A sweep is not filtered there either.
            fit.main(argv[:-2] + ["--runtime", "torch", "--tp-sweep", "1,6"])
            sweep = capsys.readouterr().out
            assert "dropped --tp-sweep" not in sweep
            assert re.search(r"^   6 ", sweep, re.M), "TP 6 row should be present"

            # vLLM, same config, still refuses.
            with pytest.raises(SystemExit):
                fit.main(argv + ["--runtime", "vllm"])

    def test_only_widths_a_layer_uses_are_checked(self):
        """parse_dims fills dense_intermediate for every MoE, hybrid or not

        A pure MoE has no dense-replacement layers, so its dense width belongs
        to no layer and must not veto a TP the expert width divides.
        """
        pure = parse_dims({
            "hidden_size": 2048, "num_hidden_layers": 24,
            "num_attention_heads": 12, "num_key_value_heads": 2,
            "intermediate_size": 28672,       # carried but unused
            "moe_intermediate_size": 1536,
            "num_local_experts": 32, "num_experts_per_tok": 4,
            "vocab_size": 32000})
        assert pure.dense_intermediate == 28672, "premise: parse_dims fills it"
        assert pure.first_k_dense_replace == 0
        assert fit.ffn_shard_widths(pure) == (1536,)
        assert fit.tp_is_shardable(pure, 6), "1536 divides by 6; 28672 is unused"

        # A hybrid MoE does have those layers, so both widths count.
        hybrid = parse_dims({
            "hidden_size": 2048, "num_hidden_layers": 24,
            "num_attention_heads": 12, "num_key_value_heads": 2,
            "intermediate_size": 28672, "moe_intermediate_size": 1536,
            "num_local_experts": 32, "num_experts_per_tok": 4,
            "first_k_dense_replace": 2, "vocab_size": 32000})
        assert fit.ffn_shard_widths(hybrid) == (1536, 28672)
        assert not fit.tp_is_shardable(hybrid, 6)

        dense = parse_dims({
            "hidden_size": 2048, "num_hidden_layers": 8,
            "num_attention_heads": 16, "intermediate_size": 8192,
            "vocab_size": 32000})
        assert fit.ffn_shard_widths(dense) == (8192,)

    def test_error_names_the_width_that_failed(self, capsys):
        """A hybrid MoE rejected on its dense width must say so

        Printing only the expert width would show dimensions that all divide
        cleanly next to a message claiming they do not.
        """
        cfg = {
            "hidden_size": 2048, "num_hidden_layers": 24,
            "num_attention_heads": 12, "num_key_value_heads": 2,
            "intermediate_size": 28672, "moe_intermediate_size": 1536,
            "num_local_experts": 32, "num_experts_per_tok": 4,
            "first_k_dense_replace": 2, "vocab_size": 32000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with pytest.raises(SystemExit):
                fit.main(["--model", str(path), "--device-vram-gb", "16",
                          "--tp", "6"])
        err = capsys.readouterr().err
        assert "FFN width(s) 1536/28672" in err

    def test_torch_wider_device_suggestion_is_not_gated(self, capsys):
        """Module placement takes any device count, so never withhold it

        This config has no shardable TP above 4, so a head-sharded runtime
        rightly omits the suggestion -- but torch must still get one.
        """
        cfg = {
            "hidden_size": 8192, "head_dim": 128, "num_hidden_layers": 40,
            "num_attention_heads": 12, "num_key_value_heads": 2,
            "intermediate_size": 28672, "vocab_size": 128000,
            "tie_word_embeddings": False,
        }
        d = parse_dims(cfg)
        assert fit.next_shardable_tp(d, 4) is None, "premise: nothing above 4"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            argv = ["--model", str(path), "--device-vram-gb", "16", "--tp", "4"]
            fit.main(argv + ["--runtime", "torch"])
            torch_out = capsys.readouterr().out
            fit.main(argv + ["--runtime", "vllm"])
            vllm_out = capsys.readouterr().out
        # 5 divides neither the 12 heads nor the 28672 FFN width, which is
        # the point: placement is not bound by either. It is also the smallest
        # count that helps -- 40 layers over 5 devices puts 8 on the busiest
        # one instead of 10, where 8 devices would be more hardware than the
        # next improvement needs.
        assert "--tp 5 splits weights across 5 XPUs" in torch_out
        assert "splits weights across" not in vllm_out

    def test_cli_refuses_an_unshardable_tp(self, capsys):
        """--tp the heads cannot support is not a memory question"""
        cfg = {
            "hidden_size": 1536,
            "num_hidden_layers": 8,
            "num_attention_heads": 12,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 32000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with pytest.raises(SystemExit) as exc:
                fit.main(["--model", str(path), "--device-vram-gb", "16",
                          "--tp", "6"])
            assert exc.value.code != 0
            err = capsys.readouterr().err
            assert "--tp 6 does not divide this model's shard dimensions" in err
            assert "Dimensions divide at, up to 16: 1, 2, 4" in err

            # A sweep drops the bad values instead of failing the run.
            rc = fit.main(["--model", str(path), "--device-vram-gb", "16",
                           "--tp-sweep", "1,2,4,6,8"])
            out = capsys.readouterr().out
        assert rc == 0
        assert "dropped --tp-sweep value(s) 6, 8" in out
        # The surviving rows are still reported.
        assert "TP sweep verdict" in out

    def test_ffn_width_must_divide_too(self):
        """Column-parallel MLP splits the intermediate dimension

        12 query heads and 2 KV heads pass both head checks at TP 6, but
        intermediate_size 28672 does not divide by 6, so the runtime still
        rejects it -- and a recommendation of TP 6 would be unlaunchable.
        """
        heads_ok = self._dims(12, 2, intermediate=28672)
        assert fit.tp_is_shardable(heads_ok, 4)
        assert not fit.tp_is_shardable(heads_ok, 6)
        assert fit.next_shardable_tp(heads_ok, 4) is None

        # Same heads, an FFN width that does divide by 6.
        divides = self._dims(12, 2, intermediate=27648)
        assert fit.tp_is_shardable(divides, 6)
        assert fit.next_shardable_tp(divides, 4) == 6

    def test_hybrid_moe_dense_width_must_divide_too(self):
        """A hybrid MoE has two FFN widths, and both are partitioned"""
        d = parse_dims({
            "hidden_size": 4096,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 10752,     # dense layers: divides by 6
            "moe_intermediate_size": 1536,  # expert layers: divides by 6
            "n_routed_experts": 16,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 2,
            "vocab_size": 32000,
        })
        assert d.dense_intermediate == 10752 and d.intermediate == 1536
        assert fit.tp_is_shardable(d, 2)
        assert fit.tp_is_shardable(d, 8)

        # Break only the dense width and the TP must be rejected.
        broken = parse_dims({
            "hidden_size": 4096,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 10751,     # odd: no longer divides by 2
            "moe_intermediate_size": 1536,
            "n_routed_experts": 16,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 2,
            "vocab_size": 32000,
        })
        assert not fit.tp_is_shardable(broken, 2)

    def test_next_placement_tp_takes_the_smallest_improving_count(self):
        """Doubling overstates the hardware an improvement needs

        Placement improves in steps. A 4-layer model at 3 devices already has
        2 layers on its busiest device; 4 devices drops that to 1, and so does
        6 -- but two of those six would be idle.
        """
        d4 = parse_dims(dict(self._cfg_dict(), num_hidden_layers=4))
        assert fit.layer_shards(d4, 3) == 2
        assert fit.next_placement_tp(d4, 3) == 4, "not 6"
        assert fit.next_placement_tp(d4, 4) is None, "out of layers"
        assert fit.next_placement_tp(d4, 1) == 2

        # 40 layers step 1, 2, 2, 4, 5, 5, 6, 8 -- doubling from 4 skips 5.
        d40 = parse_dims(dict(self._cfg_dict(), num_hidden_layers=40))
        assert fit.next_placement_tp(d40, 4) == 5
        assert fit.next_placement_tp(d40, 5) == 7, "6 is no better than 5"
        assert fit.next_placement_tp(d40, 40) is None

    def test_returns_none_when_no_tp_qualifies(self):
        """A prime head count past the limit has no wider valid TP"""
        assert fit.next_shardable_tp(self._dims(17), 17) is None
        assert fit.next_shardable_tp(self._dims(32), 32, limit=16) is None

    def test_kv_heads_must_shard_or_replicate(self):
        """TP must divide the KV group or be a multiple of it (vLLM's rule)"""
        d = self._dims(32, 6)
        tp = fit.next_shardable_tp(d, 1)
        assert tp is not None
        assert d.num_attn_heads % tp == 0
        assert d.num_kv_heads % tp == 0 or tp % d.num_kv_heads == 0

    def test_cli_suggests_only_a_shardable_tp(self, capsys):
        """A weights-bound verdict must not recommend an unlaunchable TP"""
        # 12 heads, explicit head_dim, and too big for 4x16 GB at bf16
        cfg = {
            "hidden_size": 8192,
            "head_dim": 128,
            "num_hidden_layers": 40,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "intermediate_size": 27648,   # divides by 6, unlike 28672
            "vocab_size": 128000,
            "tie_word_embeddings": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            rc = fit.main(["--model", str(path), "--device-vram-gb", "16",
                           "--tp", "4"])
        out = capsys.readouterr().out
        assert rc == 1, "this config should not fit on 4x16 GB"
        assert "weights" in out.lower()
        assert "--tp 6" in out, "6 divides 12 heads; 8 does not"
        assert "--tp 8" not in out
        # The device count the figure assumes is stated, since the script
        # cannot see how many are installed.
        assert "needs 6 devices" in out


class TestVisionTowerPricing:
    """
    A VLM's vision tower: how many params it holds, and at what dtype.

    Two independent decisions, and both were wrong. Found against on-disk
    bytes:

    - Qwen2-VL names the ViT width `embed_dim` (1280) and reuses
      `vision_config.hidden_size` for the *output* of its patch merger
      (3584). Reading `hidden_size` priced 32 blocks at 2.8x their width --
      4.94 B params against a 0.63 B tower -- which is +51% on the bf16
      checkpoint's weights.
    - A ViT block is Q/K/V/O and MLP Linear layers, which a quantizer
      converts. So the tower follows `--quant` unless `modules_to_not_convert`
      names it. Pricing it at the embedding rate instead held a whole tower at
      bf16 for every sub-16-bit dtype, which is the false DOES NOT FIT
      direction.

    Ground truth is the root-level safetensors byte total of each pinned
    revision. The residual is the patch merger, which this count omits: ~46 M
    params, 0.6% of a 7B VLM's bytes.
    """

    # model id -> GiB of root-level *.safetensors at the pinned revision, from
    # `curl -s "https://huggingface.co/api/models/<id>?blobs=true"`.
    ON_DISK = {
        "Qwen/Qwen2-VL-7B-Instruct": 15.44,
        "Qwen/Qwen2-VL-7B-Instruct-AWQ": 6.45,
        "Qwen/Qwen2.5-VL-7B-Instruct": 15.45,
    }

    @staticmethod
    def _qwen2vl_vision():
        """Qwen2-VL's vision_config, the fields that drive the count."""
        return {"depth": 32, "embed_dim": 1280, "mlp_ratio": 4, "num_heads": 16,
                "in_chans": 3, "hidden_size": 3584, "patch_size": 14,
                "spatial_merge_size": 2, "temporal_patch_size": 2}

    def _vlm_cfg(self, modules_to_not_convert=None):
        """Qwen2-VL-7B-shaped config, 4-bit AWQ, with a chosen exclusion list."""
        qcfg = {"quant_method": "awq", "bits": 4, "group_size": 128}
        if modules_to_not_convert is not None:
            qcfg["modules_to_not_convert"] = modules_to_not_convert
        return {
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "vocab_size": 152064,
            "tie_word_embeddings": False,
            "vision_config": self._qwen2vl_vision(),
            "quantization_config": qcfg,
        }

    def _weights(self, cfg):
        d = parse_dims(cfg)
        params = fit.count_params(d) + d.vision_params
        quant = fit.quant_from_config(cfg)[0] or "bf16"
        return fit.calculate_mixed_precision_weights(cfg, d, params, quant, 1)

    def _run_cli(self, cfg, capsys, extra_args=()):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            rc = fit.main(["--model", str(path), "--device-vram-gb", "24"]
                          + list(extra_args))
        return rc, capsys.readouterr().out

    @pytest.mark.parametrize("model_id", sorted(ON_DISK))
    def test_weights_match_the_bytes_on_disk(self, model_id):
        """Within 3% of the checkpoint's own root-level safetensors total"""
        cfg = fetch_config(model_id)
        quant = fit.quant_from_config(cfg)[0] or "bf16"
        est = estimate(cfg, quant, "bf16", 4096, 1, 1, "vllm", 31.89,
                       0.85)["weights"] / GB
        disk = self.ON_DISK[model_id]
        err = (est - disk) / disk
        assert abs(err) < 0.03, \
            f"{model_id}: estimated {est:.2f} GiB vs {disk} GiB on disk " \
            f"({err*100:+.1f}%)"

    def test_tower_width_is_embed_dim_not_the_merger_output(self):
        """0.63 B for Qwen2-VL's tower, against 4.94 B from hidden_size"""
        counted = fit._vision_param_count(self._qwen2vl_vision())
        assert 0.60e9 < counted < 0.68e9, f"{counted/1e9:.2f} B"
        # The published tower is 0.675 B; the gap is the patch merger.
        assert counted < 0.675e9

    def test_gated_tower_counts_three_matrices(self):
        """Qwen2.5-VL's SwiGLU tower has gate/up/down, not up/down

        Counting two matrices there is 22% light on the blocks, which put the
        bf16 checkpoint 2.3% below its bytes on disk.
        """
        clip = {"num_hidden_layers": 32, "hidden_size": 1280,
                "intermediate_size": 3420, "patch_size": 14,
                "hidden_act": "quick_gelu"}
        gated = dict(clip, hidden_act="silu")
        extra = fit._vision_param_count(gated) - fit._vision_param_count(clip)
        assert extra == 32 * 1280 * 3420, "exactly one more matrix per block"

    def test_tower_stays_quantized_when_the_list_does_not_name_it(self):
        """The embedding rule does not reach the vision tower

        An attention-only exclusion list says nothing about the ViT, whose
        blocks a 4-bit quantizer converts. Charging 0.63 B params at bf16 here
        adds 0.92 GB to a 7B VLM and can report DOES NOT FIT for a launch that
        fits.
        """
        _, breakdown = self._weights(self._vlm_cfg(["self_attn"]))
        assert breakdown["vision_bpp"] == fit.BYTES_PER_PARAM["int4"]
        # ...while the tensors the rule does cover stay wide.
        assert breakdown["embed_bpp"] == 2.0
        assert breakdown["attn_bpp"] == 2.0

    def test_visual_only_list_keeps_the_tower_wide(self):
        """`["visual"]` is the whole exclusion list Qwen2-VL AWQ ships

        It names no component the branch used to recognize, so it fell through
        to the uniform path and priced the tower at 4-bit -- 1.08 GiB under the
        checkpoint's bytes on disk.
        """
        wide_bytes, wide = self._weights(self._vlm_cfg(["visual"]))
        assert wide["vision_bpp"] == 2.0
        assert wide["vision_params"] == parse_dims(self._vlm_cfg()).vision_params

        flat_bytes, flat = self._weights(self._vlm_cfg())   # no list at all
        assert flat["vision_params"] == 0, "the tower is inside the flat row"
        assert wide_bytes > flat_bytes, "excluding the tower costs bytes"
        assert (wide_bytes - flat_bytes) / GB == pytest.approx(
            wide["vision_params"] * (2.0 - fit.BYTES_PER_PARAM["int4"]) / GB,
            rel=1e-6)

    def test_breakdown_rows_sum_to_the_weights_total(self):
        """A vision row, or the printed table does not add up"""
        total, breakdown = self._weights(self._vlm_cfg(["visual"]))
        rows = sum(breakdown[f"{k}_bytes"]
                   for k in ("embed", "attn", "router", "ffn", "vision"))
        assert abs(rows - total) <= 4, "integer division of each row only"

    def test_unsized_tower_is_named_instead_of_reported_as_zero(self, capsys):
        """A vision_config with no depth prices no tower, and says so

        Qwen2.5-VL-7B-Instruct-AWQ ships exactly this: hidden_size and nothing
        to count layers from. `0.00 B params` would read as "no tower" for a
        0.63 B one that is really there.
        """
        cfg = self._vlm_cfg(["visual"])
        cfg["vision_config"] = {"hidden_size": 1280, "in_chans": 3,
                                "spatial_patch_size": 14}
        rc, out = self._run_cli(cfg, capsys)
        assert rc == 0
        assert "Vision tower:      not sized" in out
        assert "0.00 B params" not in out
        assert "LLM backbone only" in out


class TestMeasuredOnDevice:
    """
    Weight estimates against figures vLLM-XPU reported on real hardware.

    Ground truth is the engine's own `Model loading took X GiB` line from a
    served deployment on 2x Intel Arc (Battlemage, 32656 MiB each), single
    device, `--enforce-eager --block-size 64 --max-model-len 4096
    --gpu-memory-utilization 0.85`. Stronger than a safetensors byte total: it
    is what the runtime actually allocated.

    Every figure comes from `vllm/vllm-openai-xpu:latest`, the one image
    vllm-xpu-run documents (digest
    sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901),
    taken on model_revisions.MEASUREMENT_DATE. Each model also served a real
    completion. The VLM rows were added later, on the same SKU and image
    (model_revisions.VLM_MEASUREMENT_DATE); each of those served a text
    completion and a base64 image request, so the tower was resident, not
    merely allocated.

    Tolerance is one-sided where it matters -- an estimate below the measured
    figure is the direction that reports FITS for a launch that OOMs.
    """

    # model id -> GiB the engine reported for weights
    MEASURED = {
        "Qwen/Qwen2.5-0.5B-Instruct": 0.93,
        "Qwen/Qwen2.5-1.5B-Instruct": 2.89,
        "Qwen/Qwen3-0.6B": 1.12,
        "Qwen/Qwen3-4B": 7.56,
        "meta-llama/Llama-3.2-3B-Instruct": 6.02,
        "microsoft/Phi-4-mini-instruct": 7.17,
        "Qwen/Qwen2.5-7B-Instruct": 14.25,
        "mistralai/Mistral-7B-Instruct-v0.3": 13.51,
        "meta-llama/Llama-3.1-8B-Instruct": 14.99,
        "Qwen/Qwen3-8B": 15.27,
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 15.29,
        "NousResearch/Hermes-3-Llama-3.1-8B": 14.99,
        "tiiuae/Falcon3-7B-Instruct": 13.93,
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": 14.99,
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4": 5.18,
        "Qwen/Qwen2.5-7B-Instruct-AWQ": 5.2,
        "openai/gpt-oss-20b": 12.87,
        # VLMs: the tower is part of every figure below.
        "Qwen/Qwen2-VL-7B-Instruct": 15.53,
        "Qwen/Qwen2.5-VL-7B-Instruct": 15.63,
        "Qwen/Qwen2-VL-7B-Instruct-AWQ": 6.49,
    }

    @pytest.mark.parametrize("model_id", sorted(MEASURED))
    def test_weights_match_the_engines_own_figure(self, model_id):
        """Within 5%, and never more than 2% below what the engine allocated"""
        cfg = fetch_config(model_id)
        quant = fit.quant_from_config(cfg)[0] or "bf16"
        est = estimate(cfg, quant, "bf16", 4096, 1, 1, "vllm", 31.89,
                       0.85)["weights"] / GB
        measured = self.MEASURED[model_id]
        err = (est - measured) / measured
        assert abs(err) < 0.05, \
            f"{model_id}: estimated {est:.2f} GiB vs measured {measured} GiB " \
            f"({err*100:+.1f}%)"
        assert err > -0.02, \
            f"{model_id}: estimate is {err*100:+.1f}% BELOW the measured " \
            f"figure -- that direction reports FITS for a launch that OOMs"

    @pytest.mark.parametrize("model_id,measured", [
        ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", 5.18),
        ("Qwen/Qwen2.5-7B-Instruct-AWQ", 5.20),
    ])
    def test_quantized_checkpoints_keep_embeddings_wide(self, model_id, measured):
        """The int4 case the wide-embedding rule was measured from

        Both are 4-bit, group_size 128, untied, 152064 vocab. Pricing their
        embedding matrix and LM head at int4 gives 3.90 GiB -- 25% under what
        the engine allocated. Holding those 1.09 B params at bf16, as GPTQ and
        AWQ actually do, gives 5.37 GiB.
        """
        cfg = fetch_config(model_id)
        assert fit.quant_from_config(cfg) == ("int4", False)
        result = estimate(cfg, "int4", "fp8", 4096, 1, 1, "vllm", 31.89, 0.85)

        est = result["weights"] / GB
        assert abs(est - measured) / measured < 0.05, f"{est:.2f} vs {measured}"
        assert est > measured, "the safe side of a 4-bit estimate"
        assert est > 3.90 * 1.2, "a flat int4 pricing would be far too small"

        breakdown = result["mixed_breakdown"]
        assert breakdown is not None, "the split has to be reported"
        assert breakdown["embed_bpp"] == 2.0
        assert breakdown["attn_bpp"] == 0.55

    def test_quantized_vlm_keeps_its_tower_wide(self):
        """The served AWQ VLM allocated more than a uniformly 4-bit model would

        `Qwen/Qwen2-VL-7B-Instruct-AWQ` lists exactly `["visual"]`, so its
        0.63 B tower stays bf16 while the backbone goes to 4-bit. The engine
        allocated 6.49 GiB. Pricing that tower at int4 instead lands near
        5.70 GiB -- 12% under what the runtime asked the driver for, in the
        direction that reports FITS for a launch that OOMs.
        """
        cfg = fetch_config("Qwen/Qwen2-VL-7B-Instruct-AWQ")
        measured = self.MEASURED["Qwen/Qwen2-VL-7B-Instruct-AWQ"]
        r = estimate(cfg, "int4", "bf16", 4096, 1, 1, "vllm", 31.89, 0.85)

        breakdown = r["mixed_breakdown"]
        assert breakdown is not None, "['visual'] has to reach the split path"
        assert breakdown["vision_bpp"] == 2.0, "the tower is the excluded one"

        quantized_tower = (r["weights"]
                           - breakdown["vision_params"] * (2.0 - 0.55)) / GB
        assert (measured - quantized_tower) / measured > 0.10, \
            f"a 4-bit tower gives {quantized_tower:.2f} GiB against " \
            f"{measured} GiB allocated"

    def test_unsized_tower_understates_by_the_measured_gap(self):
        """The documented gap, measured: 5.37 GiB estimated, 6.59 GiB allocated

        `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` ships a `vision_config` with no depth
        field, so the tower cannot be sized from it and the estimate covers the
        backbone only. Served, the engine allocated 6.59 GiB. The report says
        which field is missing rather than implying the tower is free -- this
        test pins the size of what it is warning about, so sizing the tower
        later must update the wording along with the number.
        """
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        measured = 6.59
        cfg = fetch_config(model_id)
        assert fit._vision_param_count(cfg["vision_config"]) == 0, \
            "this config is the unsized case; pick another if it gained depth"

        est = estimate(cfg, "int4", "bf16", 4096, 1, 1, "vllm", 31.89,
                       0.85)["weights"] / GB
        shortfall = measured - est
        assert 1.0 < shortfall < 1.5, \
            f"{est:.2f} GiB estimated vs {measured} GiB allocated " \
            f"({shortfall:.2f} GiB short)"

    def test_capacity_ceiling_matches_the_kv_pool(self):
        """`Capacity:` max_context vs the KV pool vLLM allocated

        Measured token counts from the same deployments. The ceiling is what a
        user sets --max-model-len against, so a wildly optimistic one is as
        useless as a wrong verdict.
        """
        # model id -> (GPU KV cache size in tokens, quant)
        pools = {
            "Qwen/Qwen2.5-1.5B-Instruct": (820352, "bf16"),
            "microsoft/Phi-4-mini-instruct": (142690, "bf16"),
            "Qwen/Qwen2.5-7B-Instruct": (194496, "bf16"),
            "Qwen/Qwen3-8B": (69632, "bf16"),
            "openai/gpt-oss-20b": (249633, "mxfp4"),
        }
        for model_id, (tokens, quant) in pools.items():
            cfg = fetch_config(model_id)
            kv_dtype = "bf16" if quant == "bf16" else "bf16"
            r = estimate(cfg, quant, kv_dtype, 4096, 1, 1, "vllm", 31.89, 0.85)
            err = (r["max_context"] - tokens) / tokens
            assert abs(err) < 0.10, \
                f"{model_id}: predicted {r['max_context']} tokens vs {tokens} " \
                f"allocated ({err*100:+.1f}%)"

    def test_does_not_fit_verdicts_were_real_ooms(self):
        """Both negative verdicts failed engine init on the hardware

        Qwen3-14B: bf16 weights 27.51 GiB against 27.11 GiB usable at
        gmu 0.85 -- the engine loaded them (reporting 27.52 GiB) and then
        raised "No available memory for the cache blocks". Qwen3.5-35B-A3B:
        64.11 GiB of weights, "XPU out of memory" during the load itself.
        """
        for model_id in ("Qwen/Qwen3-14B", "Qwen/Qwen3.5-35B-A3B"):
            cfg = fetch_config(model_id)
            r = estimate(cfg, "bf16", "bf16", 4096, 1, 1, "vllm", 31.89, 0.85)
            assert not r["fits"], model_id
            assert r["weights"] > r["usable_vram"] * 0.98, model_id


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
