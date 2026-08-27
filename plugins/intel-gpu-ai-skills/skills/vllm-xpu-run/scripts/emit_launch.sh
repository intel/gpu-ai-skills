#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Emit a vLLM-XPU docker launch command. This script is the source of truth
# for generated launch templates consumed by higher-level planner skills.

set -uo pipefail

model=""
config_file=""
target_gpu=0
num_xpus=1
ctx=4096
quant="bf16"
kv_dtype="bfloat16"
dp=1
tp=1
image="${VLLM_XPU_IMAGE:-vllm/vllm-openai-xpu:latest}"
port=8000
container_name="vllm-xpu"
trust_remote_code=0

usage() {
    cat <<'EOF'
usage: emit_launch.sh --model HF_ID [options]

Options:
  --config FILE         model-config-recommend output; top candidate overrides quant/KV/DP/TP
  --target-gpu N        First XPU id to expose (default: 0)
  --num-xpus N          Number of XPUs visible to the plan (default: 1)
  --ctx N               Max context length (default: 4096)
  --quant Q             bf16 (default) | fp8 | awq | gptq | mxfp4 | inc | auto-round | int4
  --kv-cache-dtype Q    bfloat16 (default) | fp8 | auto
  --dp N                Data-parallel replica count (default: 1)
  --tp N                Tensor-parallel size per replica (default: 1)
    --image TAG           vLLM XPU image (default: $VLLM_XPU_IMAGE or vllm/vllm-openai-xpu:latest)
  --port N              First host port to publish (default: 8000)
  --container-name N    Base container name (default: vllm-xpu)
  --trust-remote-code   Add --trust-remote-code
  -h, --help            Show this help
EOF
}

die() { printf '%s\n' "$1" >&2; usage >&2; exit 2; }

# This script's output is a command the caller pastes into a shell. Values
# scraped from a --config file are not authored by the caller, so they must not
# reach stdout carrying shell metacharacters. Arguments passed directly on the
# command line are the caller's own input and are not filtered here; the default
# --image value is itself a `<version>` placeholder the caller edits.
check_token() {
    case "$2" in
        '') die "$1 must not be empty" ;;
        *[!A-Za-z0-9._:/@-]*)
            die "$1 contains unsupported characters (allowed: alphanumerics . _ : / @ -): $2" ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)             model="${2:-}"; shift 2 ;;
        --config)            config_file="${2:-}"; shift 2 ;;
        --target-gpu)        target_gpu="${2:-}"; shift 2 ;;
        --num-xpus)          num_xpus="${2:-}"; shift 2 ;;
        --ctx)               ctx="${2:-}"; shift 2 ;;
        --quant)             quant="${2:-}"; shift 2 ;;
        --kv-cache-dtype)    kv_dtype="${2:-}"; shift 2 ;;
        --dp)                dp="${2:-}"; shift 2 ;;
        --tp)                tp="${2:-}"; shift 2 ;;
        --image)             image="${2:-}"; shift 2 ;;
        --port)              port="${2:-}"; shift 2 ;;
        --container-name)    container_name="${2:-}"; shift 2 ;;
        --trust-remote-code) trust_remote_code=1; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -n "$model" ] || die "--model is required"
for v in target_gpu num_xpus ctx dp tp port; do
    val="${!v}"
    case "$val" in
        ''|*[!0-9]*) die "--${v//_/-} must be a non-negative integer (got: $val)" ;;
    esac
done
[ "$num_xpus" -ge 1 ] || die "--num-xpus must be >= 1"
[ "$dp" -ge 1 ] || die "--dp must be >= 1"
[ "$tp" -ge 1 ] || die "--tp must be >= 1"

if [ -n "$config_file" ] && [ -f "$config_file" ]; then
    parsed=$(awk '
        /^--- Candidate 1:/ {
            in_top=1
            line=$0
            if (line ~ /bf16/) quant="bf16"
            if (line ~ /AutoRound/) quant="auto-round"
            if (match(line, /--quantization [^, -]+/)) {
                quant=substr(line, RSTART + 15, RLENGTH - 15)
            }
            if (match(line, /--kv-cache-dtype [^,]+/)) {
                kv=substr(line, RSTART + 17, RLENGTH - 17)
            }
            if (match(line, /dp=[0-9]+/)) {
                dp=substr(line, RSTART + 3, RLENGTH - 3)
            }
            if (match(line, /tp=[0-9]+/)) {
                tp=substr(line, RSTART + 3, RLENGTH - 3)
            }
            next
        }
        in_top && /^  Checkpoint: / {
            checkpoint=$0
            sub(/^  Checkpoint: /, "", checkpoint)
        }
        in_top && /^$/ { in_top=0 }
        END {
            printf "quant=%s\nkv=%s\ndp=%s\ntp=%s\ncheckpoint=%s\n",
                quant, kv, dp, tp, checkpoint
        }
    ' "$config_file")
    # Assign scraped values by name rather than with eval: the file is data the
    # caller did not necessarily author, so shell metacharacters in it must not
    # be interpreted. Numeric fields are re-validated here because they are set
    # after the --dp/--tp argument checks above have already run.
    while IFS='=' read -r key val; do
        case "$key" in
            quant)
                if [ -n "$val" ]; then
                    check_token "--config: quant" "$val"
                    quant="$val"
                fi ;;
            kv)
                if [ -n "$val" ]; then
                    check_token "--config: kv-cache-dtype" "$val"
                    kv_dtype="$val"
                fi ;;
            dp)
                case "$val" in
                    '') ;;
                    *[!0-9]*) die "--config: dp must be a non-negative integer (got: $val)" ;;
                    *) dp="$val" ;;
                esac ;;
            tp)
                case "$val" in
                    '') ;;
                    *[!0-9]*) die "--config: tp must be a non-negative integer (got: $val)" ;;
                    *) tp="$val" ;;
                esac ;;
            checkpoint)
                if [ -n "$val" ]; then
                    check_token "--config: Checkpoint" "$val"
                    model="$val"
                fi ;;
        esac
    done <<EOF
$parsed
EOF
    [ "$dp" -ge 1 ] || die "--config: dp must be >= 1 (got: $dp)"
    [ "$tp" -ge 1 ] || die "--config: tp must be >= 1 (got: $tp)"
fi

required_xpus=$((dp * tp))
if [ "$required_xpus" -gt "$num_xpus" ]; then
    die "topology requires $required_xpus XPU(s), but --num-xpus is $num_xpus"
fi

serve_flags="--enforce-eager \\
        --block-size=64 \\
        --max-model-len $ctx \\
        --gpu-memory-utilization 0.85"

case "$quant" in
    bf16|fp16)
        serve_flags="--dtype bfloat16 \\
        $serve_flags"
        ;;
    auto-round)
        serve_flags="$serve_flags \\
        --attention-backend TRITON_ATTN"
        ;;
    int4)
        serve_flags="$serve_flags \\
        --quantization awq \\
        --attention-backend TRITON_ATTN"
        ;;
    fp8|awq|gptq|mxfp4|inc)
        serve_flags="$serve_flags \\
        --quantization $quant \\
        --attention-backend TRITON_ATTN"
        ;;
    *) die "unsupported --quant: $quant" ;;
esac

if [ "$kv_dtype" != "bfloat16" ]; then
    serve_flags="$serve_flags \\
        --kv-cache-dtype $kv_dtype"
fi
if [ "$trust_remote_code" -eq 1 ]; then
    serve_flags="$serve_flags \\
        --trust-remote-code"
fi
if [ "$tp" -gt 1 ]; then
    serve_flags="$serve_flags \\
        --disable-custom-all-reduce \\
        --tensor-parallel-size $tp"
fi

printf '# Official image: https://hub.docker.com/r/vllm/vllm-openai-xpu; pin a digest for reproducibility.\n'
printf '# To pass host proxy settings into Docker, keep the proxy -e lines below; Docker ignores unset env names.\n'
if [ "$dp" -gt 1 ]; then
    printf '# DP=%s, TP=%s: launch %s replicas, then route traffic across ports %s..%s.\n' \
        "$dp" "$tp" "$dp" "$port" "$((port + dp - 1))"
fi

replica=0
while [ "$replica" -lt "$dp" ]; do
    start=$((target_gpu + replica * tp))
    end=$((start + tp - 1))
    mask=""
    gpu="$start"
    while [ "$gpu" -le "$end" ]; do
        if [ -z "$mask" ]; then
            mask="$gpu"
        else
            mask="$mask,$gpu"
        fi
        gpu=$((gpu + 1))
    done
    if [ "$dp" -eq 1 ]; then
        name="$container_name"
    else
        name="$container_name-$replica"
    fi
    host_port=$((port + replica))

    printf '\n'
    printf 'docker run -d --name %s \\\n' "$name"
    printf '    --device /dev/dri \\\n'
    printf '    -v /dev/dri/by-path:/dev/dri/by-path:ro \\\n'
    printf '    --group-add "$(getent group render | cut -d: -f3)" \\\n'
    printf '    --ipc=host \\\n'
    printf '    -e ZE_AFFINITY_MASK=%s \\\n' "$mask"
    printf '    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\\n'
    if [ "$tp" -gt 1 ]; then
        printf '    -e CCL_ZE_IPC_EXCHANGE=pidfd \\\n'
    fi
    printf '    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \\\n'
    printf '    -e http_proxy -e https_proxy -e no_proxy \\\n'
    printf '    -e HF_TOKEN="$HF_TOKEN" \\\n'
    printf '    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \\\n'
    printf '    -p %s:8000 \\\n' "$host_port"
    printf '    %s \\\n' "$image"
    printf '    %s \\\n' "$model"
    printf '        %s\n' "$serve_flags"

    replica=$((replica + 1))
done
