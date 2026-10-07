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

rec_py="plugins/intel-gpu-ai-skills/skills/model-config-recommend/scripts/recommend.py"
rec="$tmpdir/recommend.txt"
python3 "$rec_py" \
    --model "$cfg" \
    --device arc-pro-b70 --num-devices 2 \
    --ctx 4096 --concurrency 4 \
    --gpu-memory-utilization 0.85 \
    --no-hub-search >"$rec"

grep -q "Memory:    27.2 GB usable per GPU" "$rec"
grep -q -- "--gpu-memory-utilization 0.85" "$rec"

# Multimodal MoE wrapper: expert, hybrid-attention and MTP fields live under
# text_config (Qwen3.5-35B-A3B shape). The recommender once read only the root
# and priced this as a 3.8 B dense model.
moe_cfg="$tmpdir/moe/config.json"
mkdir -p "$tmpdir/moe"
python3 - "$moe_cfg" <<'PY'
import json, sys
layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 10
json.dump({
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "text_config": {
        "hidden_size": 2048, "num_hidden_layers": 40,
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "num_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        "vocab_size": 248320, "layer_types": layer_types,
        "mtp_num_hidden_layers": 1,
    },
}, open(sys.argv[1], "w"))
PY

moe_rec="$tmpdir/moe-recommend.txt"
python3 "$rec_py" \
    --model "$moe_cfg" \
    --device arc-pro-b70 --num-devices 4 \
    --ctx 32768 --concurrency 8 \
    --no-hub-search >"$moe_rec"

grep -q "(34.13 B params, qwen3_5_moe)" "$moe_rec"
grep -q "MoE:       256 experts, 8 per token; 2.93 B active params per token" "$moe_rec"
grep -q "ships an MTP head" "$moe_rec"
grep -q -- "--xpu-graph" "$moe_rec"

# model-can-it-fit must agree with the recommender on hybrid-layer KV, since
# xpu-deploy-plan runs both for the same model.
python3 - "$moe_cfg" <<'PY'
import json, sys
sys.path.insert(0, "plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts")
import fit
d = fit.parse_dims(json.load(open(sys.argv[1])))
# 10 of 40 layers hold KV: 2 * 10 * 2 heads * 256 dim * 2 B.
assert fit.kv_bytes(d, 1, 1, "bf16") == 20480, fit.kv_bytes(d, 1, 1, "bf16")
PY

roof_py="plugins/intel-gpu-ai-skills/skills/model-config-recommend/scripts/llm_roofline.py"
roof_json=$(python3 "$roof_py" --model "$moe_cfg" --tensor-parallel 4 --concurrency 8 --json)
python3 - "$roof_json" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
# 10 of 40 layers hold KV: 2 * 10 * 2 heads * 256 dim * 2 B / TP 4.
assert r["kv_bytes_per_token_per_rank"] == 5120, r["kv_bytes_per_token_per_rank"]
# A batch of 8 routes to a fraction of the 256 experts, not all of them.
assert r["decode"]["bytes_moved"] < 0.4 * r["weights_per_rank_bytes"], r["decode"]
PY

# Sliding-window layers hold KV (GPT-OSS alternates sliding/full); counting
# only full_attention layers would halve KV and can report a false fit.
swa_cfg="$tmpdir/swa/config.json"
mkdir -p "$tmpdir/swa"
python3 - "$swa_cfg" <<'PY'
import json, sys
json.dump({
    "model_type": "gpt_oss", "hidden_size": 2880, "num_hidden_layers": 24,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
    "num_local_experts": 32, "num_experts_per_tok": 4,
    "intermediate_size": 2880, "vocab_size": 201088,
    "layer_types": ["sliding_attention", "full_attention"] * 12,
}, open(sys.argv[1], "w"))
PY
swa_json=$(python3 "$roof_py" --model "$swa_cfg" --json)
python3 - "$swa_json" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
# All 24 layers: 2 * 24 * 8 heads * 64 dim * 2 B.
assert r["kv_bytes_per_token_per_rank"] == 49152, r["kv_bytes_per_token_per_rank"]
PY

echo "OK model-can-it-fit usable-memory checks passed"
