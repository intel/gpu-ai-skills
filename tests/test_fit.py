#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive pytest test suite for fit.py

Tests all known bugs and verifies against HuggingFace KV cache calculator.

Run from repository root:
    cd intel-models-skillpack
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
        fit_path = candidate / 'plugins/intel-model-skillpack/skills/model-can-it-fit/scripts'
        if fit_path.exists():
            return candidate

    raise RuntimeError(f"Could not find git repo root from {test_file}")

# Add fit.py directory to path
git_root = find_git_root()
fit_py_dir = git_root / 'plugins/intel-model-skillpack/skills/model-can-it-fit/scripts'
sys.path.insert(0, str(fit_py_dir))

# The pinned-revision map and cache layout are shared with the pre-fetch helper
# so the two cannot drift apart on which models the suite needs.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'data'))

import fit
from fit import fetch_config as fetch_config_live, parse_dims, count_params, kv_bytes, estimate
from model_revisions import CACHE_DIR, cache_path, revision_for

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
