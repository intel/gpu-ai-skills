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

fit_py="plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py"

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

# A hybrid state-space model: recurrent layers must be reported as a state
# pool that does not grow with context, and must not be charged a KV cache.
hybrid_cfg="$tmpdir/hybrid.json"
cat >"$hybrid_cfg" <<'JSON'
{
  "model_type": "qwen3_next",
  "torch_dtype": "bfloat16",
  "hidden_size": 2048,
  "num_hidden_layers": 4,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "intermediate_size": 5120,
  "vocab_size": 151936,
  "tie_word_embeddings": false,
  "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "linear_value_head_dim": 128,
  "layer_types": ["linear_attention", "linear_attention",
                  "linear_attention", "full_attention"]
}
JSON

hybrid="$tmpdir/hybrid.txt"
python3 "$fit_py" \
    --model "$hybrid_cfg" \
    --quant bf16 --kv-dtype bf16 \
    --ctx 32768 --concurrency 4 \
    --runtime vllm --device-vram-gb 24 \
    --gpu-memory-utilization 0.90 >"$hybrid"

grep -q "hybrid SSM + attention" "$hybrid"
grep -q "4 total: 1 full attention, 3 recurrent" "$hybrid"
grep -q "Recurrent state " "$hybrid"
grep -q "independent of context" "$hybrid"

# A sliding-window model: windowed layers must report a bounded token count.
swa_cfg="$tmpdir/swa.json"
cat >"$swa_cfg" <<'JSON'
{
  "model_type": "mistral",
  "torch_dtype": "bfloat16",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 14336,
  "vocab_size": 32768,
  "tie_word_embeddings": false,
  "sliding_window": 4096
}
JSON

swa="$tmpdir/swa.txt"
python3 "$fit_py" \
    --model "$swa_cfg" \
    --quant bf16 --kv-dtype bf16 \
    --ctx 32768 --concurrency 4 \
    --runtime vllm --device-vram-gb 32 \
    --gpu-memory-utilization 0.90 >"$swa"

grep -q "sliding-window attention" "$swa"
grep -q "32 windowed layer(s) hold 8191 tok, not 32768" "$swa"

rec_py="plugins/intel-gpu-ai-skills/skills/model-config-recommend/scripts/recommend.py"
rec="$tmpdir/recommend.txt"
python3 "$rec_py" \
    --model "$cfg" \
    --device arc-pro-b70 --num-devices 2 \
    --ctx 4096 --concurrency 4 \
    --gpu-memory-utilization 0.85 \
    --no-hub-search >"$rec"

# 0.85 of the 30.29 GiB the B70 driver reports (vLLM requested 25.75 GiB
# on the card), not 0.85 of the 32 GB nameplate.
grep -q "Memory:    25.7 GB usable per GPU" "$rec"
grep -q -- "--gpu-memory-utilization 0.85" "$rec"

# The recommender must charge a hybrid model a state pool and report the layout.
rec_hybrid="$tmpdir/recommend-hybrid.txt"
python3 "$rec_py" \
    --model "$hybrid_cfg" \
    --device arc-pro-b70 --num-devices 2 \
    --ctx 32768 --concurrency 4 \
    --gpu-memory-utilization 0.85 \
    --no-hub-search >"$rec_hybrid"

grep -q "Layers:    4 total: 1 full attention, 3 recurrent" "$rec_hybrid"
grep -q "GB state + " "$rec_hybrid"

echo "OK model-config-recommend hybrid checks passed"

echo "OK model-can-it-fit usable-memory checks passed"
