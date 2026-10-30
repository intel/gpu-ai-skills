#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Emit a `docker run` command that serves a model with vLLM-Omni on Intel XPU.
# Prints the command only; nothing is executed. Notes about stage placement go
# to stderr so stdout stays a pasteable command.
#
# The image built from upstream docker/Dockerfile.xpu has
# ENTRYPOINT ["vllm", "serve", "--omni"], so serve flags follow the image name.

set -uo pipefail

model=""
image="${VLLM_OMNI_XPU_IMAGE:-vllm-omni-xpu}"
devices="0"
port=8091
name="vllm-omni-xpu"
deploy_config=""
stage_overrides=""
trust_remote_code=0
no_remote_code=0
privileged=0
host_network=0
first_launch=0

# The default is emitted verbatim so $HOME expands on the host at run time.
# A caller-supplied --hf-cache never takes this path: it goes through
# check_path below, which refuses anything the shell could interpret.
hf_cache_default='$HOME/.cache/huggingface'
hf_cache=""

# Single-card layout for Qwen2.5-Omni: thinker, talker and code2wav all on
# logical device 0, fractions summing to 0.93. Validated with the 7B on one
# 32 GB Arc Pro B70. The 8192-token caps keep the AR stages' profiling peak
# small enough to leave KV cache; at the default 32768 both AR stages report
# negative KV memory. See references/deploy-configs.md before changing these.
QWEN25_OMNI_SINGLE_CARD='{"0":{"devices":"0","gpu_memory_utilization":0.62,"max_model_len":8192,"max_num_batched_tokens":8192},"1":{"devices":"0","gpu_memory_utilization":0.19,"max_model_len":8192,"max_num_batched_tokens":8192},"2":{"devices":"0","gpu_memory_utilization":0.12}}'

usage() {
    cat <<'EOF'
usage: emit_launch.sh --model HF_ID [options] [-- extra vllm serve args]

Options:
  --image NAME[:TAG]      vLLM-Omni XPU image (default: $VLLM_OMNI_XPU_IMAGE or
                          vllm-omni-xpu, the name upstream's guide builds)
  --devices LIST          Physical XPU ids exposed via ZE_AFFINITY_MASK, e.g. 0 or
                          0,1 (default: 0). Stage `devices` are logical indices
                          into this list.
  --port N                API port inside and outside the container (default: 8091)
  --name NAME             Container name (default: vllm-omni-xpu)
  --deploy-config FILE    Host deploy YAML; mounted read-only and passed to
                          --deploy-config. Does not replace the remote-code
                          decision below.
  --stage-overrides JSON  Per-stage overrides passed to --stage-overrides
  --trust-remote-code     Add --trust-remote-code (runs code from the model repo)
  --no-remote-code        Disable trust_remote_code on every stage. For a model
                          whose stage layout is not known here, list the stages
                          in --stage-overrides, since there is otherwise
                          nothing to disable it on.
  --privileged            Add --privileged (upstream-docs fallback; not default)
  --host-network          Use --net=host instead of publishing --port
  --first-launch          Raise stage init timeouts for a first run that still
                          has to download weights
  --hf-cache DIR          Host Hugging Face cache, absolute path
                          (default: $HOME/.cache/huggingface)
  -h, --help              Show this help

One of --trust-remote-code or --no-remote-code is REQUIRED, with or without
--deploy-config: most bundled deploy YAMLs enable trust_remote_code per stage,
and omitting --trust-remote-code does NOT turn that off.
EOF
}

die() {
    printf '%s\n' "$1" >&2
    usage >&2
    exit 2
}

note() {
    printf '# note: %s\n' "$1" >&2
}

check_token() {
    case "$2" in
        '')
            die "$1 must not be empty"
            ;;
        *[!A-Za-z0-9._:/@-]*)
            die "$1 contains unsupported characters (allowed: alphanumerics . _ : / @ -): $2"
            ;;
    esac
}

# Host paths land inside a double-quoted -v argument of the emitted command.
# Rather than escape what the shell would interpret there (quotes, backslashes,
# $, backticks, whitespace), refuse it; and require an absolute path, which is
# what Docker needs for a bind mount anyway. ':' is excluded because it
# delimits the mount spec.
check_path() {
    case "$2" in
        '')
            die "$1 must not be empty"
            ;;
        /*)
            ;;
        *)
            die "$1 must be an absolute path: $2"
            ;;
    esac

    case "$2" in
        *[!A-Za-z0-9._/-]*)
            die "$1 contains unsupported characters (allowed: alphanumerics . _ / -): $2"
            ;;
    esac
}

# Every value-taking option reads $2. A trailing flag with no value must fail
# with a clear message, not with `shift: shift count out of range`.
# Called with "$@" in the main shell, never in a subshell, so die() exits.
need_value() {
    [ "$#" -ge 2 ] || die "$1 requires a value"
}

extra=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            need_value "$@"
            model=$2
            shift 2
            ;;
        --image)
            need_value "$@"
            image=$2
            shift 2
            ;;
        --devices)
            need_value "$@"
            devices=$2
            shift 2
            ;;
        --port)
            need_value "$@"
            port=$2
            shift 2
            ;;
        --name)
            need_value "$@"
            name=$2
            shift 2
            ;;
        --deploy-config)
            need_value "$@"
            deploy_config=$2
            shift 2
            ;;
        --stage-overrides)
            need_value "$@"
            stage_overrides=$2
            shift 2
            ;;
        --trust-remote-code)
            trust_remote_code=1
            shift
            ;;
        --no-remote-code)
            no_remote_code=1
            shift
            ;;
        --privileged)
            privileged=1
            shift
            ;;
        --host-network)
            host_network=1
            shift
            ;;
        --first-launch)
            first_launch=1
            shift
            ;;
        --hf-cache)
            need_value "$@"
            hf_cache=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra=("$@")
            break
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[ -n "$model" ] || die "--model is required"

# Options this script emits itself must not come back through `--`. Upstream's
# parser keeps the last occurrence (logging only "Found duplicate keys") and
# accepts underscores and unambiguous prefixes, so a passthrough
# --stage-overrides would silently drop the remote-code disable and the
# placement checked below.
for arg in "${extra[@]}"; do
    case "$arg" in
        --?*) ;;
        *) continue ;;
    esac
    opt=${arg%%=*}
    opt=${opt//_/-}
    for managed in --model --port --stage-overrides --deploy-config --trust-remote-code; do
        case "$managed" in
            "$opt"*)
                die "$arg after -- would replace the launcher's own $managed; pass it as a launcher option instead"
                ;;
        esac
    done
done

check_token "--model" "$model"
check_token "--image" "$image"
check_token "--name" "$name"

if [ "$trust_remote_code" = 1 ] && [ "$no_remote_code" = 1 ]; then
    die "--trust-remote-code and --no-remote-code are mutually exclusive"
fi

case "$devices" in
    ''|*[!0-9,]*|,*|*,|*,,*)
        die "--devices must be a comma-separated list of integers: $devices"
        ;;
esac

case "$port" in
    ''|*[!0-9]*)
        die "--port must be an integer: $port"
        ;;
esac

case "$stage_overrides" in
    *"'"*)
        die "--stage-overrides must not contain single quotes"
        ;;
esac

if [ -n "$hf_cache" ]; then
    check_path "--hf-cache" "$hf_cache"
else
    hf_cache="$hf_cache_default"
fi

ndev=$(printf '%s' "$devices" | awk -F, '{print NF}')

# Model-specific stage placement. Upstream deploy YAMLs carry a `platforms.xpu`
# section that is applied automatically; these notes only cover layouts whose
# default device count exceeds what the caller exposed.
case "$model" in
    Qwen/Qwen2.5-Omni-*)
        if [ "$ndev" -lt 2 ] && [ -z "$deploy_config" ]; then
            # Caller values win key by key, but they do not displace the whole
            # preset: a stage the caller did not mention keeps the bundled
            # layout, which puts the talker and code2wav on logical device 1 --
            # not exposed here -- so taking any override as a complete
            # placement emits a launch that cannot start.
            caller_overrides="$stage_overrides"
            stage_overrides=$(
                python3 - "$QWEN25_OMNI_SINGLE_CARD" "$stage_overrides" <<'PY'
import json
import sys

preset, raw = sys.argv[1:]
merged = json.loads(preset)

try:
    caller = json.loads(raw) if raw else {}
except json.JSONDecodeError as exc:
    raise SystemExit(f"--stage-overrides is not valid JSON: {exc}")

if not isinstance(caller, dict):
    raise SystemExit("--stage-overrides must be a JSON object")

for stage, overrides in caller.items():
    if not isinstance(overrides, dict):
        raise SystemExit(
            f"--stage-overrides entry {stage!r} must be a JSON object"
        )
    merged.setdefault(stage, {}).update(overrides)

print(json.dumps(merged, separators=(",", ":"), sort_keys=True))
PY
            ) || die "could not merge --stage-overrides into the single-card preset"

            note "Qwen2.5-Omni XPU default puts the talker + code2wav on logical device 1; colocating all 3 stages on device 0"
            [ -n "$caller_overrides" ] &&
                note "your --stage-overrides were merged over that preset, so the stages you did not name stay on device 0"
        elif [ "$ndev" -ge 2 ]; then
            note "Qwen2.5-Omni XPU default layout: thinker on logical 0, talker + code2wav on logical 1"
        fi
        ;;
    Qwen/Qwen3-Omni-*)
        # No preset to fall back on here, unlike Qwen2.5-Omni above: the
        # default layout addresses logical devices 0-4, so with fewer exposed
        # the talker's placement is out of range and the container cannot
        # start. Emitting it anyway would only produce a failed startup.
        #
        # --stage-overrides is not accepted as the fix. Fitting this pipeline
        # on fewer cards also means shrinking the thinker's
        # tensor_parallel_size, which the bundled YAML sets to 4; a placement
        # that moves the stages but leaves TP at 4 fails just as surely, and
        # this script cannot read the YAML to tell the two apart. An overlay
        # states the whole layout in one file, which is what the skill's
        # references/deploy-configs.md prescribes for this model.
        if [ "$ndev" -lt 5 ] && [ -z "$deploy_config" ]; then
            die "Qwen3-Omni's XPU default layout needs 5 devices (thinker tensor_parallel_size=4 on logical 0-3, talker + code2wav on 4) but $ndev is exposed, so its stages would reference devices that do not exist. Pass --deploy-config with an overlay that both shrinks the thinker's tensor_parallel_size and places every stage within 0-$((ndev - 1)); --stage-overrides alone cannot be checked for that here. Size it with the model-can-it-fit skill and treat the layout as unvalidated until the smoke test passes"
        fi
        ;;
esac

# Some upstream deploy YAMLs enable trust_remote_code at the stage level.
# Merely omitting the top-level --trust-remote-code flag does not disable those
# stage defaults. Require the caller to choose explicitly.
#
# The choice is required for EVERY model, not only the layouts listed below.
# On main, 79 of the 141 bundled deploy YAMLs set trust_remote_code per
# stage (71 of 110 in v0.30.0), and these four families are 4 of those 79 --
# lance, bagel and the audex family are not. Gating only on the known list
# would emit a silent launch for the other ~75. Reading the YAML here would mean running docker,
# which this script never does, so the decision is the caller's.
#
# --deploy-config is not a decision either. A copy of the bundled YAML keeps
# its trust_remote_code: true, and an overlay can inherit it through
# base_config from inside the image, so the file alone says nothing about
# whether anyone agreed to run the model repository's code.
remote_code_stages=""
case "$model" in
    Qwen/Qwen2.5-Omni-*|Qwen/Qwen3-Omni-*)
        remote_code_stages="0,1,2"
        ;;
    Qwen/Qwen3-TTS-*|mistralai/Voxtral-*-TTS-*)
        remote_code_stages="0,1"
        ;;
esac

if [ "$trust_remote_code" = 0 ] && [ "$no_remote_code" = 0 ]; then
    if [ -n "$remote_code_stages" ]; then
        die "$model has bundled stages that enable trust_remote_code; omitting --trust-remote-code does not turn it off. Pass --trust-remote-code or --no-remote-code explicitly, with or without --deploy-config"
    fi
    die "$model may have a bundled deploy YAML that enables trust_remote_code on its stages, and omitting --trust-remote-code does not turn it off. Pass --trust-remote-code or --no-remote-code explicitly, with or without --deploy-config"
fi

# Merge trust_remote_code=false into each known stage without discarding
# caller-provided overrides or the single-card memory preset.
if [ "$no_remote_code" = 1 ] && [ -z "$remote_code_stages" ]; then
    # Unknown stage layout. Guessing stage ids is not an option (an override
    # for a stage the pipeline does not have fails at startup), and emitting
    # nothing would make --no-remote-code a silent no-op on exactly the models
    # whose bundled YAML cannot be read here -- the flag would promise a
    # disable it never applied. So the caller has to name the stages in
    # --stage-overrides, the only form this script can emit and the only one
    # that beats both a bundled YAML and a --deploy-config.
    #
    # A --deploy-config is not an acceptable substitute: an overlay inherits
    # stages from a base_config inside the image, so no amount of reading the
    # host file proves every stage ends up disabled.
    [ -n "$stage_overrides" ] ||
        die "--no-remote-code cannot be enforced for $model: its stage layout is unknown, so there are no stage ids to disable, and a top-level value loses to the per-stage ones. Pass --stage-overrides listing this pipeline's stages (e.g. '{\"0\":{},\"1\":{}}'). A --deploy-config does not substitute: it can inherit stages from a base_config inside the image, so this script cannot confirm the value you asked for"

    remote_code_stages=$(
        python3 - "$stage_overrides" <<'PY'
import json
import sys

try:
    overrides = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"--stage-overrides is not valid JSON: {exc}")

if not isinstance(overrides, dict) or not overrides:
    raise SystemExit("--stage-overrides must be a non-empty JSON object")

# Stage ids are joined with commas below, so they must not contain one.
if not all(key.isdigit() for key in overrides):
    raise SystemExit("--stage-overrides keys must be stage ids (integers)")

print(",".join(overrides))
PY
    ) || die "could not read the stage ids to disable from --stage-overrides"
fi

# Every --no-remote-code now has stage ids to apply to, so the flag is always
# enforced in the emitted command rather than only asserted.
if [ "$no_remote_code" = 1 ]; then
    stage_overrides=$(
        python3 - "$stage_overrides" "$remote_code_stages" <<'PY'
import json
import sys

raw, stage_ids = sys.argv[1:]
try:
    overrides = json.loads(raw) if raw else {}
except json.JSONDecodeError as exc:
    raise SystemExit(f"--stage-overrides is not valid JSON: {exc}")

if not isinstance(overrides, dict):
    raise SystemExit("--stage-overrides must be a JSON object")

for stage_id in stage_ids.split(","):
    stage = overrides.setdefault(stage_id, {})
    if not isinstance(stage, dict):
        raise SystemExit(
            f"--stage-overrides entry {stage_id!r} must be a JSON object"
        )
    stage["trust_remote_code"] = False

print(json.dumps(overrides, separators=(",", ":"), sort_keys=True))
PY
    ) || die "could not merge --no-remote-code into --stage-overrides"

    note "trust_remote_code disabled on stages $remote_code_stages"
elif [ "$trust_remote_code" = 1 ]; then
    note "--trust-remote-code allows model-repository Python to execute inside the server"
fi

# The JSON is emitted inside single quotes. The raw input was checked for
# apostrophes, but re-serialising merged JSON decodes an escaped ' into a
# literal one, which would close the quote and let the rest run as shell.
case "$stage_overrides" in
    *"'"*)
        die "--stage-overrides must not contain single quotes"
        ;;
esac

note "confirm the resolved value after startup: docker logs $name 2>&1 | grep -o 'trust_remote_code=[A-Za-z]*'"

# Stage `devices` are logical indices into ZE_AFFINITY_MASK, so an index at or
# beyond the number of exposed cards names a device that does not exist and
# fails at startup. Only placements visible here can be checked -- a stage that
# takes its devices from a bundled YAML or a deploy config is not -- but one
# the caller or a preset put out of range is caught before the command is run.
if [ -n "$stage_overrides" ]; then
    if ! python3 - "$stage_overrides" "$ndev" <<'PY'
import json
import sys

raw, ndev = sys.argv[1], int(sys.argv[2])

try:
    overrides = json.loads(raw)
except json.JSONDecodeError as exc:
    raise SystemExit(f"--stage-overrides is not valid JSON: {exc}")

if not isinstance(overrides, dict):
    raise SystemExit("--stage-overrides must be a JSON object")

for stage, config in sorted(overrides.items()):
    if not isinstance(config, dict) or "devices" not in config:
        continue
    for index in str(config["devices"]).split(","):
        index = index.strip()
        if not index.isdigit() or int(index) >= ndev:
            raise SystemExit(
                f"stage {stage} is placed on logical device {index!r}, but "
                f"--devices exposes {ndev} "
                f"(valid logical {'index 0' if ndev == 1 else f'indices 0-{ndev - 1}'})"
            )
PY
    then
        die "--stage-overrides places a stage outside the exposed devices"
    fi
fi

deploy_mount=""
deploy_arg=""
if [ -n "$deploy_config" ]; then
    [ -f "$deploy_config" ] ||
        die "--deploy-config file not found: $deploy_config"

    deploy_abs="$(cd "$(dirname "$deploy_config")" && pwd)/$(basename "$deploy_config")"
    check_path "--deploy-config" "$deploy_abs"
    deploy_mount="/workspace/deploy/$(basename "$deploy_config")"
    deploy_arg="$deploy_mount"
fi

printf 'docker run -d --name %s \\\n' "$name"
printf '    --device /dev/dri \\\n'
printf '    -v /dev/dri/by-path:/dev/dri/by-path:ro \\\n'
printf '    --group-add "$(getent group render | cut -d: -f3)" \\\n'
printf '    --ipc=host \\\n'

[ "$privileged" = 1 ] &&
    printf '    --privileged \\\n'

if [ "$host_network" = 1 ]; then
    printf '    --net=host \\\n'
else
    printf '    -p %s:%s \\\n' "$port" "$port"
fi

printf '    -e ZE_AFFINITY_MASK=%s \\\n' "$devices"
printf '    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\\n'
printf '    -e HF_TOKEN \\\n'
printf '    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \\\n'
printf '    -e http_proxy -e https_proxy -e no_proxy \\\n'
printf '    -v "%s:/root/.cache/huggingface" \\\n' "$hf_cache"

[ -n "$deploy_mount" ] &&
    printf '    -v "%s:%s:ro" \\\n' "$deploy_abs" "$deploy_mount"

printf '    %s \\\n' "$image"
printf '    --model %s --port %s' "$model" "$port"

[ -n "$deploy_arg" ] &&
    printf ' \\\n    --deploy-config %s' "$deploy_arg"

[ -n "$stage_overrides" ] &&
    printf " \\\\\n    --stage-overrides '%s'" "$stage_overrides"

[ "$trust_remote_code" = 1 ] &&
    printf ' \\\n    --trust-remote-code'

[ "$first_launch" = 1 ] &&
    printf ' \\\n    --stage-init-timeout 1200 --init-timeout 2400'

if [ "${#extra[@]}" -gt 0 ]; then
    printf ' \\\n   '
    printf ' %q' "${extra[@]}"
fi

printf '\n'
