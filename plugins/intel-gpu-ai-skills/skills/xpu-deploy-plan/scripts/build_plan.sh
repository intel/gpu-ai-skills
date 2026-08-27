#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Build an end-to-end Intel XPU deployment PLAN.md by chaining skills.
#
# Chains: xpu-runtime-preflight -> model-can-it-fit -> model-config-recommend
# -> selected runtime skill (vllm-xpu-run | sglang-xpu-run | torch-xpu-run).
#
# This script never launches the container, pulls images, or modifies the
# system. It only writes PLAN.md and sibling artifacts under --out-dir.

set -uo pipefail

model=""
runtime="vllm"
target_gpu=0
ctx=4096
concurrency=1
quant="bf16"
device="arc-pro-b70"
device_vram_gb=32
num_xpus=1
image="${VLLM_XPU_IMAGE:-vllm/vllm-openai-xpu:latest}"
port=8000
container_name="vllm-xpu"
out_dir=".out/skills/xpu-deploy-plan"
skip_preflight=0
skip_fit=0
skip_config=0

usage() {
    cat <<'EOF'
usage: build_plan.sh --model HF_ID [options]

Required:
  --model ID            Hugging Face model id or local path

Options:
  --runtime NAME        vllm (default) | sglang | torch
  --target-gpu N        XPU id to deploy on (default: 0)
  --ctx N               Max context length (default: 4096)
  --concurrency N       Initial concurrency (default: 1)
  --quant Q             bf16 (default) | fp8 | int4
  --device NAME         Recommender device id (default: arc-pro-b70)
  --device-vram-gb N    VRAM per XPU in GiB (default: 32, Arc Pro B70)
  --num-xpus N          Number of XPUs to use (default: 1)
    --image TAG           Container image (default: $VLLM_XPU_IMAGE or vllm/vllm-openai-xpu:latest)
  --port N              Host port to publish (default: 8000)
  --container-name N    Container name (default: vllm-xpu)
  --out-dir DIR         Output dir (default: .out/skills/xpu-deploy-plan)
  --skip-preflight      Skip xpu-runtime-preflight call
  --skip-fit            Skip model-can-it-fit call
  --skip-config         Skip model-config-recommend call
  -h, --help            Show this help

The script does not pull images, restart services, stop containers, or
edit system configuration. Image policy, multi-GPU oneCCL setup, and OOM
recovery are owned by the selected runtime skill (vllm-xpu-run /
sglang-xpu-run / torch-xpu-run) and the runtime triage skill.
EOF
}

die() { printf '%s\n' "$1" >&2; usage >&2; exit 2; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)           model="${2:-}"; shift 2 ;;
        --runtime)         runtime="${2:-}"; shift 2 ;;
        --target-gpu)      target_gpu="${2:-}"; shift 2 ;;
        --ctx)             ctx="${2:-}"; shift 2 ;;
        --concurrency)     concurrency="${2:-}"; shift 2 ;;
        --quant)           quant="${2:-}"; shift 2 ;;
        --device)          device="${2:-}"; shift 2 ;;
        --device-vram-gb)  device_vram_gb="${2:-}"; shift 2 ;;
        --num-xpus)        num_xpus="${2:-}"; shift 2 ;;
        --image)           image="${2:-}"; shift 2 ;;
        --port)            port="${2:-}"; shift 2 ;;
        --container-name)  container_name="${2:-}"; shift 2 ;;
        --out-dir)         out_dir="${2:-}"; shift 2 ;;
        --skip-preflight)  skip_preflight=1; shift ;;
        --skip-fit)        skip_fit=1; shift ;;
        --skip-config)     skip_config=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -n "$model" ] || die "--model is required"
case "$runtime" in
    vllm|sglang|torch) ;;
    *) die "--runtime must be vllm, sglang, or torch" ;;
esac
for v in target_gpu ctx concurrency device_vram_gb num_xpus port; do
    val="${!v}"
    case "$val" in
        ''|*[!0-9]*) die "--${v//_/-} must be a non-negative integer (got: $val)" ;;
    esac
done
# The string options are interpolated into inputs.json below by the heredoc, so
# a value containing a double quote, a backslash, or a control character would
# emit malformed JSON — or, worse, silently add a key a reader would attribute
# to this script. Reject exactly those characters and nothing more: the charset
# has to stay wide enough for the default container image reference.
for v in model quant device image container_name; do
    val="${!v}"
    case "$val" in
        *'"'*|*'\'*|*[[:cntrl:]]*)
            die "--${v//_/-} must not contain quotes, backslashes, or control characters (got: $val)" ;;
    esac
done

mkdir -p "$out_dir"
log_file="$out_dir/build_plan.log"
plan_md="$out_dir/PLAN.md"
inputs_json="$out_dir/inputs.json"
: >"$log_file"

# Resolve skills_root. Sibling skills always live next to this skill dir at
# .../skills/<name>/scripts/, in both the in-repo layout
# (<repo>/plugins/intel-gpu-ai-skills/skills/) and the installed layout
# (~/.claude/skills/), because scripts/install.sh uses `cp -r` and preserves
# the directory shape. Override with $SKILLS_ROOT for tests.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SKILLS_ROOT:-}" ] && [ -d "$SKILLS_ROOT" ]; then
    skills_root="$SKILLS_ROOT"
elif [ -d "$script_dir/../../vllm-xpu-run" ]; then
    skills_root="$(cd "$script_dir/../.." && pwd)"
else
    printf 'build_plan.sh: cannot locate sibling skills (vllm-xpu-run/, model-can-it-fit/, ...) from %s. Set SKILLS_ROOT to override.\n' "$script_dir" >&2
    exit 2
fi

cat >"$inputs_json" <<EOF
{
  "model": "$model",
  "runtime": "$runtime",
  "target_gpu": $target_gpu,
  "ctx": $ctx,
  "concurrency": $concurrency,
  "quant": "$quant",
  "device": "$device",
  "device_vram_gb": $device_vram_gb,
  "num_xpus": $num_xpus,
  "image": "$image",
  "port": $port,
  "container_name": "$container_name"
}
EOF

# ---- Step 1: xpu-runtime-preflight ----
preflight_status="SKIPPED"
preflight_summary=""
preflight_fail_count=0
preflight_script="$skills_root/xpu-runtime-preflight/scripts/check_runtime_preflight.sh"
preflight_out="$out_dir/preflight"
if [ "$skip_preflight" -eq 0 ]; then
    if [ -x "$preflight_script" ]; then
        mkdir -p "$preflight_out"
        {
            printf '\n== xpu-runtime-preflight ==\n'
            "$preflight_script" --target-gpu "$target_gpu" --out-dir "$preflight_out"
            printf 'exit=%s\n' "$?"
        } >>"$log_file" 2>&1
        if [ -f "$preflight_out/status.tsv" ]; then
            preflight_fail_count=$(awk -F'\t' 'NR>1 && $1=="FAIL"{c++} END{print c+0}' "$preflight_out/status.tsv")
            if [ "$preflight_fail_count" -gt 0 ]; then
                preflight_status="FAIL"
            else
                preflight_status="PASS"
            fi
            preflight_summary="$preflight_out/SUMMARY.md"
        else
            preflight_status="MISSING"
        fi
    else
        printf 'preflight script not found at %s\n' "$preflight_script" >>"$log_file"
        preflight_status="NOT_FOUND"
    fi
fi

# ---- Step 2: model-can-it-fit ----
fit_status="SKIPPED"
fit_out="$out_dir/fit.txt"
fit_script="$skills_root/model-can-it-fit/scripts/fit.py"
if [ "$skip_fit" -eq 0 ]; then
    if [ -f "$fit_script" ]; then
        {
            printf '== model-can-it-fit ==\n'
            python3 "$fit_script" \
                --model "$model" \
                --quant "$quant" \
                --device-vram-gb "$device_vram_gb" \
                --ctx "$ctx" \
                --concurrency "$concurrency" \
                --tp "$num_xpus"
            printf 'exit=%s\n' "$?"
        } >"$fit_out" 2>&1
        fit_status="RAN"
    else
        printf 'fit script not found at %s\n' "$fit_script" >>"$log_file"
        fit_status="NOT_FOUND"
    fi
fi

# ---- Step 3: model-config-recommend (vLLM only) ----
config_status="SKIPPED"
config_out="$out_dir/config.txt"
config_script="$skills_root/model-config-recommend/scripts/recommend.py"
if [ "$skip_config" -eq 0 ] && [ "$runtime" = "vllm" ]; then
    if [ -f "$config_script" ]; then
        config_rc=0
        {
            printf '== model-config-recommend ==\n'
            python3 "$config_script" \
                --model "$model" \
                --device "$device" \
                --num-devices "$num_xpus" \
                --ctx "$ctx" \
                --concurrency "$concurrency" \
                --runtime vllm-xpu
            config_rc="$?"
            printf 'exit=%s\n' "$config_rc"
        } >"$config_out" 2>&1
        if [ "$config_rc" -eq 0 ]; then
            config_status="RAN"
        else
            config_status="FAIL"
        fi
    else
        printf 'config recommend script not found at %s\n' "$config_script" >>"$log_file"
        config_status="NOT_FOUND"
    fi
fi

# ---- Step 4: build the launch command from the selected runtime skill ----
case "$runtime" in
    vllm)
        emit_launch_script="$skills_root/vllm-xpu-run/scripts/emit_launch.sh"
        if [ -f "$emit_launch_script" ]; then
            launch_config_args=()
            if [ "$config_status" = "RAN" ]; then
                launch_config_args=(--config "$config_out")
            fi
            emit_rc=0
            launch_cmd=$(bash "$emit_launch_script" \
                --model "$model" \
                "${launch_config_args[@]}" \
                --target-gpu "$target_gpu" \
                --num-xpus "$num_xpus" \
                --ctx "$ctx" \
                --quant "$quant" \
                --image "$image" \
                --port "$port" \
                --container-name "$container_name") || emit_rc=$?
            if [ "$emit_rc" -ne 0 ]; then
                launch_cmd="vllm-xpu-run launch emitter at $emit_launch_script exited $emit_rc. Read plugins/intel-gpu-ai-skills/skills/vllm-xpu-run/SKILL.md and run it manually."
            fi
        else
            launch_cmd="vllm-xpu-run launch emitter not found at $emit_launch_script. Read plugins/intel-gpu-ai-skills/skills/vllm-xpu-run/SKILL.md."
        fi
        runtime_skill="vllm-xpu-run"
        bench_skill="vllm-xpu-bench"
        ;;
    sglang)
        launch_cmd="See plugins/intel-gpu-ai-skills/skills/sglang-xpu-run/SKILL.md for the exact docker run command for $model on XPU $target_gpu."
        runtime_skill="sglang-xpu-run"
        bench_skill="sglang-xpu-bench"
        ;;
    torch)
        launch_cmd="See plugins/intel-gpu-ai-skills/skills/torch-xpu-run/SKILL.md for the pure-PyTorch path for $model on XPU $target_gpu."
        runtime_skill="torch-xpu-run"
        bench_skill="torch-xpu-bench"
        ;;
esac

# ---- Step 5: write PLAN.md ----
{
    printf '# Deployment plan: %s on %d x XPU\n\n' "$model" "$num_xpus"
    printf '_Generated by `xpu-deploy-plan/scripts/build_plan.sh`. Do not edit by hand._\n\n'

    printf '## Inputs\n\n'
    printf -- '- Model: `%s`\n' "$model"
    printf -- '- Runtime: `%s` (skill: **%s**)\n' "$runtime" "$runtime_skill"
    printf -- '- Target XPU: `%s`\n' "$target_gpu"
    printf -- '- Context: `%s`\n' "$ctx"
    printf -- '- Concurrency (initial): `%s`\n' "$concurrency"
    printf -- '- Quant: `%s`\n' "$quant"
    printf -- '- Device: `%s`\n' "$device"
    printf -- '- XPU count: `%s` (VRAM/XPU: `%s GiB`)\n' "$num_xpus" "$device_vram_gb"
    if [ "$runtime" = "vllm" ]; then
        printf -- '- Image: `%s`\n' "$image"
    fi
    printf '\n'

    printf '## 1. Readiness (xpu-runtime-preflight)\n\n'
    case "$preflight_status" in
        PASS)
            printf 'Verdict: **PASS** (no FAIL entries).\n\n'
            printf 'Details: `%s`\n\n' "$preflight_summary"
            ;;
        FAIL)
            printf 'Verdict: **FAIL** \u2014 %d hard blocker(s). **Halt the plan.**\n\n' "$preflight_fail_count"
            printf 'Read `%s` and follow its Next Skill Routing section before re-running.\n\n' "$preflight_summary"
            ;;
        NOT_FOUND|MISSING)
            printf 'Verdict: **UNKNOWN** \u2014 `xpu-runtime-preflight` not found in this checkout. Install / merge the preflight skill and re-run.\n\n'
            ;;
        SKIPPED)
            printf 'Verdict: **SKIPPED** by `--skip-preflight`. Caller asserts readiness.\n\n'
            ;;
    esac

    printf '## 2. Fit (model-can-it-fit)\n\n'
    if [ "$fit_status" = "RAN" ]; then
        printf 'See `%s`. Headline:\n\n' "$fit_out"
        printf '```text\n'
        sed -n '1,40p' "$fit_out"
        printf '\n```\n\n'
    else
        printf 'Status: `%s`. Run **model-can-it-fit** manually before launch.\n\n' "$fit_status"
    fi

    if [ "$runtime" = "vllm" ]; then
        printf '## 3. Config (model-config-recommend)\n\n'
        if [ "$config_status" = "RAN" ]; then
            printf 'See `%s`. Headline:\n\n' "$config_out"
            printf '```text\n'
            sed -n '1,40p' "$config_out"
            printf '\n```\n\n'
        elif [ "$config_status" = "FAIL" ]; then
            printf 'Status: **FAIL**. The recommender errored; launch falls back to the requested quant/topology rather than a recommended config.\n\n'
            printf '```text\n'
            sed -n '1,40p' "$config_out"
            printf '\n```\n\n'
        else
            printf 'Status: `%s`. Run **model-config-recommend** manually for vLLM on Arc B-series.\n\n' "$config_status"
        fi
    fi

    printf '## 4. Launch (from **%s**)\n\n' "$runtime_skill"
    printf 'Image policy, proxy env propagation, multi-GPU oneCCL setup, and OOM recovery are owned by **%s**. Do not duplicate them here.\n\n' "$runtime_skill"
    printf '```sh\n%s\n```\n\n' "$launch_cmd"

    printf '## 5. Smoke test\n\n'
    printf 'Verify XPU is actually used, not just HTTP 200:\n\n'
    printf '```sh\n'
    printf 'curl -s http://localhost:%s/v1/models\n' "$port"
    printf 'curl -s http://localhost:%s/v1/chat/completions \\\n' "$port"
    printf "    -H 'Content-Type: application/json' \\\\\n"
    printf "    -d '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi.\"}],\"max_tokens\":32}'\n" "$model"
    printf "docker logs %s 2>&1 | grep -iE 'xpu|device|cpu fallback'\n" "$container_name"
    printf 'xpu-smi ps\n'
    printf '```\n\n'
    printf 'Pass criteria: `/v1/models` lists `%s`; one chat reply is non-empty plausible text; logs show the XPU device path; `xpu-smi ps` shows memory rising by approximately model-size scale.\n\n' "$model"

    printf '## 6. Benchmark\n\n'
    printf 'Run **%s** after warmup (first-run kernel compile is not representative).\n\n' "$bench_skill"

    printf '## 7. Rollback\n\n'
    printf '```sh\n'
    printf 'docker stop %s\n' "$container_name"
    printf 'docker rm %s\n' "$container_name"
    printf '```\n\n'
    printf 'The plan only stops/removes the container it created. It never touches other containers or caches.\n\n'

    printf '## Artifacts\n\n'
    printf -- '- Inputs: `%s`\n' "$inputs_json"
    if [ "$preflight_status" = "PASS" ] || [ "$preflight_status" = "FAIL" ]; then
        printf -- '- Preflight: `%s`\n' "$preflight_out"
    fi
    [ "$fit_status" = "RAN" ]    && printf -- '- Fit: `%s`\n' "$fit_out"
    [ "$config_status" = "RAN" ] && printf -- '- Config: `%s`\n' "$config_out"
    printf -- '- Log: `%s`\n' "$log_file"
} >"$plan_md"

printf 'Wrote %s\n' "$plan_md"
printf 'Preflight=%s Fit=%s Config=%s\n' "$preflight_status" "$fit_status" "$config_status"

if [ "$preflight_status" = "FAIL" ]; then
    exit 1
fi
exit 0
