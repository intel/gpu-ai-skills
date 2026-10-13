#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# CPU-only regression checks for vllm-xpu-run's emit_launch.sh opt-in flags.

set -euo pipefail

cd "$(dirname "$0")/.."

launch="plugins/intel-gpu-ai-skills/skills/vllm-xpu-run/scripts/emit_launch.sh"

fail() { printf 'FAIL: %s\n%s\n' "$1" "$2" >&2; exit 1; }

# Default stays eager, with no graph env vars or optional serve flags.
out=$(bash "$launch" --model Qwen/Qwen2.5-1.5B-Instruct)
grep -q -- "--enforce-eager" <<<"$out" || fail "default launch is not eager" "$out"
for s in VLLM_XPU_ENABLE_XPU_GRAPH VLLM_USE_V2_MODEL_RUNNER --speculative-config --reasoning-parser; do
    if grep -q -- "$s" <<<"$out"; then fail "default launch carries $s" "$out"; fi
done

# --xpu-graph: dropping --enforce-eager alone leaves vLLM-XPU eager, so the
# env var must come with it, plus the v1 runner (v2 crashes with graphs).
out=$(bash "$launch" --model Qwen/Qwen3.5-35B-A3B --num-xpus 4 --tp 4 \
    --xpu-graph --mtp 2 --reasoning-parser qwen3)
if grep -q -- "--enforce-eager" <<<"$out"; then fail "--xpu-graph kept --enforce-eager" "$out"; fi
grep -q -- "-e VLLM_XPU_ENABLE_XPU_GRAPH=1" <<<"$out" || fail "--xpu-graph missing env var" "$out"
grep -q -- "-e VLLM_USE_V2_MODEL_RUNNER=0" <<<"$out" || fail "--xpu-graph missing v1 runner" "$out"
grep -q -- "--reasoning-parser qwen3" <<<"$out" || fail "--reasoning-parser not emitted" "$out"
grep -qF -- "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'" <<<"$out" ||
    fail "--mtp not emitted as JSON speculative config" "$out"

# The emitted command must be valid shell, with the JSON intact after quoting.
cmd=$(sed -n '/^docker run/,$p' <<<"$out")
bash -n <<<"$cmd" || fail "emitted command does not parse" "$cmd"

# Bad values are rejected rather than pasted into a shell command.
if bash "$launch" --model m --mtp two >/dev/null 2>&1; then
    fail "--mtp accepted a non-integer" ""
fi
if bash "$launch" --model m --reasoning-parser 'qwen3;id' >/dev/null 2>&1; then
    fail "--reasoning-parser accepted shell metacharacters" ""
fi

echo "OK vllm-xpu-run emit_launch.sh checks passed"
