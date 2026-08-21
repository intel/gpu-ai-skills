#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Pre-fetch the model config.json files that tests/test_fit.py measures.

Usage:
    export HF_TOKEN="..."        # optional; needed for the gated models
    python3 tests/data/fetch_configs.py

Running this is OPTIONAL. The test suite fetches each config on first use
anyway. Pre-fetching is useful when you want to:

- populate the cache once, then run the suite repeatedly with no network
  (`SKILLPACK_TESTS_OFFLINE=1 python3 -m pytest tests/test_fit.py`);
- fetch the gated Llama/Gemma configs while HF_TOKEN is set, so later runs
  cover them without needing the token again;
- see up front which configs are reachable from this machine.

Configs are written to tests/data/.cache/configs/, which is git-ignored. They
are upstream third-party files carrying their own licences and are deliberately
not committed -- see the module docstring in model_revisions.py for why, and
never commit a fetched config to make a skipped test pass.

This script uses urllib from the standard library. Keep it that way: the release
artifact declares no Python dependencies at all, which is what makes its
supply-chain scanning story trivial. Do not add requests or huggingface_hub here.
"""
import json
import os
import sys
import urllib.error
import urllib.request

from model_revisions import CACHE_DIR, GATED_MODELS, MODEL_REVISIONS, cache_path

HF_BASE = "https://huggingface.co"


def fetch_one(model_id, revision, token):
    """Fetch one config.json at a pinned revision. Returns the parsed dict."""
    # Bandit B310 suppression justification: the URL is built from the HF_BASE
    # https:// literal above. Neither scheme nor host is reachable from any
    # parameter -- model_id and revision only ever extend the path.
    url = f"{HF_BASE}/{model_id}/raw/{revision}/config.json"
    headers = {"User-Agent": "intel-models-skillpack-tests/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310
        return json.loads(r.read())


def fetch_and_save_configs():
    """Fetch every pinned config into the cache. Returns (ok, skipped, errors)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        print("NOTE: HF_TOKEN not set. The gated models will be skipped:")
        for model_id in sorted(GATED_MODELS):
            print(f"        {model_id}")
        print("      Tests needing them will skip too, which is not a failure.")
        print()

    ok = skipped = errors = 0

    for model_id, revision in sorted(MODEL_REVISIONS.items()):
        path = cache_path(model_id, revision)

        if path.exists():
            print(f"cached  {model_id}@{revision[:12]}")
            ok += 1
            continue

        if model_id in GATED_MODELS and not token:
            print(f"skip    {model_id} (gated, no HF_TOKEN)")
            skipped += 1
            continue

        print(f"fetch   {model_id}@{revision[:12]} ...", end=" ", flush=True)
        try:
            cfg = fetch_one(model_id, revision, token)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                print("FAILED (HTTP %d: gated repo, or the licence has not been "
                      "accepted for this token)" % e.code)
            elif e.code == 429:
                print("FAILED (HTTP 429: Hub rate limit; retry later)")
            else:
                print(f"FAILED (HTTP {e.code})")
            errors += 1
            continue
        except Exception as e:  # noqa: BLE001 - report and continue to the next model
            print(f"FAILED ({type(e).__name__}: {e})")
            errors += 1
            continue

        # Same atomic write the test suite uses, for the same reason: never
        # leave a half-written file that a later run reads back as corrupt.
        tmp = path.with_suffix('.tmp')
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except OSError as e:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            print(f"FAILED (could not write {path}: {e})")
            errors += 1
            continue

        print("ok")
        ok += 1

    print()
    print(f"{ok} available, {skipped} skipped, {errors} errors")
    print(f"Cache: {CACHE_DIR}")

    return ok, skipped, errors


if __name__ == "__main__":
    print(f"Pre-fetching {len(MODEL_REVISIONS)} model configs at pinned revisions")
    print()

    ok, skipped, errors = fetch_and_save_configs()

    if errors:
        print()
        print("Some configs could not be fetched. Tests needing them will skip.")
        sys.exit(1)
    sys.exit(0)
