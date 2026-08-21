#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# CPU-only regression checks for model-can-it-fit.

set -euo pipefail

cd "$(dirname "$0")/.."

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

cfg="$tmpdir/config.json"
cat >"$cfg" <<'JSON'
{
  "model_type": "qwen2",
  "hidden_size": 3584,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "intermediate_size": 18944,
  "vocab_size": 152064,
  "tie_word_embeddings": true
}
JSON

fit_py="plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py"

out="$tmpdir/fit.txt"
python3 "$fit_py" \
    --model "$cfg" \
    --quant bf16 --kv-dtype bf16 \
    --ctx 4096 --concurrency 4 \
    --runtime vllm --device-vram-gb 32 \
    --gpu-memory-utilization 1.0 \
    --tp-sweep 1,2 >"$out"

grep -q "Verdict:           FITS" "$out"
grep -q "Capacity:" "$out"
grep -q "TP sweep verdict: smallest requested TP that fits = 1" "$out"

oom="$tmpdir/oom.txt"
if python3 "$fit_py" \
    --model "$cfg" \
    --quant bf16 --kv-dtype bf16 \
    --ctx 4096 --concurrency 4 \
    --runtime vllm --device-vram-gb 32 \
    --gpu-memory-utilization 0.50 >"$oom" 2>&1; then
    echo "expected low gpu-memory-utilization run to fail" >&2
    cat "$oom" >&2
    exit 1
fi

grep -q "Usable VRAM:" "$oom"
grep -q "Verdict:           DOES NOT FIT" "$oom"

rec_py="plugins/intel-model-skillpack/skills/model-config-recommend/scripts/recommend.py"
rec="$tmpdir/recommend.txt"
python3 "$rec_py" \
    --model "$cfg" \
    --device arc-pro-b70 --num-devices 2 \
    --ctx 4096 --concurrency 4 \
    --gpu-memory-utilization 0.85 \
    --no-hub-search >"$rec"

grep -q "Memory:    27.2 GB usable per GPU" "$rec"
grep -q -- "--gpu-memory-utilization 0.85" "$rec"

echo "OK model-can-it-fit usable-memory checks passed"
