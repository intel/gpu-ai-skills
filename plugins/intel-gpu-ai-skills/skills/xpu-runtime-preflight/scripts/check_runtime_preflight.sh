#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Safe readiness preflight for Intel GPU/XPU skillpack work.

set -uo pipefail

target_gpu=0
out_dir=".out/skills/xpu-runtime-preflight"
dev_dri_dir="${XPU_PREFLIGHT_DEV_DRI_DIR:-/dev/dri}"
env_file=""
image=""
image_command=""
image_network="bridge"
network_check=0

usage() {
    cat <<'EOF'
usage: check_runtime_preflight.sh [options]

Options:
  --target-gpu ID       XPU device ID to gate on (default: 0)
  --out-dir DIR         Output directory (default: .out/skills/xpu-runtime-preflight)
  --env-file FILE       Load KEY=VALUE assignments (strict dotenv parser; not sourced)
  --network-check       Check Docker Hub, Hugging Face, and XPU package reachability
  --image IMAGE         Verify an already-local container image can see XPU
  --image-command CMD   Override the default in-container XPU visibility command
  --image-network MODE  Docker network mode for --image preflight (default: bridge)
  -h, --help            Show this help

The script does not pull images, restart services, stop containers, or
edit system configuration.
EOF
}

die_usage() {
    printf '%s\n' "$1" >&2
    usage >&2
    exit 2
}

require_value() {
    local opt="$1"
    local value="${2-}"
    if [ -z "$value" ]; then
        die_usage "$opt requires a value"
    fi
    case "$value" in
        --*)
            die_usage "$opt requires a value"
            ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --target-gpu)
            require_value "$1" "${2-}"
            target_gpu="${2:-}"
            shift 2
            ;;
        --out-dir)
            require_value "$1" "${2-}"
            out_dir="${2:-}"
            shift 2
            ;;
        --env-file)
            require_value "$1" "${2-}"
            env_file="${2:-}"
            shift 2
            ;;
        --network-check)
            network_check=1
            shift
            ;;
        --image)
            require_value "$1" "${2-}"
            image="${2:-}"
            shift 2
            ;;
        --image-command)
            require_value "$1" "${2-}"
            image_command="${2:-}"
            shift 2
            ;;
        --image-network)
            require_value "$1" "${2-}"
            image_network="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$target_gpu" in
    ''|*[!0-9]*)
        die_usage "--target-gpu must be a non-negative numeric device ID"
        ;;
esac

case "$image_network" in
    ''|*[[:space:]]*)
        die_usage "--image-network must be a Docker network mode without whitespace"
        ;;
esac

mkdir -p "$out_dir"
status_tsv="$out_dir/status.tsv"
log_file="$out_dir/preflight.log"
discovery_json="$out_dir/xpu-smi-discovery.json"
summary_md="$out_dir/SUMMARY.md"
: >"$log_file"
printf 'status\tcheck\tdetail\n' >"$status_tsv"

# out_dir persists across runs: stamp every evidence file, including the
# flag-gated ones, so a skipped probe cannot leave the previous run's output
# behind.
for evidence in xpu-smi-discovery.json xpu-smi-discovery.err \
    xpu-smi-discovery-lookup.err xpu-smi-health.txt xpu-smi-stats-target.txt \
    target-driver.txt kernel-log.txt kernel-log.err kernel-log-review.txt \
    dev-dri.txt dev-dri-stat.txt docker-info.txt docker-buildx.txt docker-ps.txt \
    df-dev-shm.txt df-workdir.txt curl-dockerhub.txt curl-huggingface.txt \
    curl-xpu-wheels.txt image-inspect.txt image-preflight.txt; do
    printf 'not collected in this run; see status.tsv\n' >"$out_dir/$evidence"
done

pass_count=0
warn_count=0
fail_count=0

record() {
    local status="$1"
    local check="$2"
    local detail="$3"
    printf '%s\t%s\t%s\n' "$status" "$check" "$detail" | tee -a "$status_tsv"
    case "$status" in
        PASS) pass_count=$((pass_count + 1)) ;;
        WARN) warn_count=$((warn_count + 1)) ;;
        FAIL) fail_count=$((fail_count + 1)) ;;
    esac
}

log_run() {
    local label="$1"
    shift
    {
        printf '\n== %s ==\n' "$label"
        "$@"
        printf 'exit=%s\n' "$?"
    } >>"$log_file" 2>&1
}

# Strict dotenv parser. Reads KEY=VALUE lines and exports them; never
# evaluates the file as shell, so a hostile --env-file cannot run code.
# Accepts: blank lines, `# comment` lines, `export KEY=VALUE`, and values
# that are bare, double-quoted, or single-quoted. Rejects anything else
# (command substitution, redirections, multi-token values, etc.) with
# FAIL — partial loads are not permitted.
load_env_file() {
    if [ -z "$env_file" ]; then
        return
    fi
    if [ ! -f "$env_file" ]; then
        record WARN env-file "not found: $env_file"
        return
    fi

    local lineno=0
    local line key value first last inner
    local count=0
    while IFS= read -r line || [ -n "$line" ]; do
        lineno=$((lineno + 1))
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        case "$line" in
            ''|'#'*) continue ;;
        esac
        case "$line" in
            'export '*)
                line="${line#export }"
                line="${line#"${line%%[![:space:]]*}"}"
                ;;
        esac
        if [[ ! "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            record FAIL env-file "line $lineno not a KEY=VALUE assignment; refusing to load $env_file"
            return
        fi
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        first="${value:0:1}"
        last="${value: -1}"
        if [ "${#value}" -ge 2 ] && [ "$first" = '"' ] && [ "$last" = '"' ]; then
            inner="${value:1:${#value}-2}"
            case "$inner" in
                *'"'*)
                    record FAIL env-file "line $lineno embedded double quote in quoted value; refusing to load $env_file"
                    return
                    ;;
            esac
            value="$inner"
        elif [ "${#value}" -ge 2 ] && [ "$first" = "'" ] && [ "$last" = "'" ]; then
            inner="${value:1:${#value}-2}"
            case "$inner" in
                *"'"*)
                    record FAIL env-file "line $lineno embedded single quote in quoted value; refusing to load $env_file"
                    return
                    ;;
            esac
            value="$inner"
        else
            case "$value" in
                *[[:space:]]*)
                    record FAIL env-file "line $lineno unquoted value contains whitespace; quote it (refusing to load $env_file)"
                    return
                    ;;
                *'#'*)
                    record FAIL env-file "line $lineno unquoted value contains '#'; quote it (refusing to load $env_file)"
                    return
                    ;;
            esac
        fi
        export "$key=$value"
        count=$((count + 1))
    done <"$env_file"

    record PASS env-file "loaded $count assignment(s) from $env_file"
}

has_command() {
    command -v "$1" >/dev/null 2>&1
}

http_probe() {
    local label="$1"
    local url="$2"
    local out="$out_dir/curl-${label}.txt"
    if ! has_command curl; then
        record WARN "network-$label" "curl not found"
        return
    fi
    if curl -sSI --max-time 15 "$url" >"$out" 2>&1; then
        first_line=$(sed -n '1p' "$out")
        record PASS "network-$label" "$first_line"
    else
        first_line=$(sed -n '1p' "$out")
        record WARN "network-$label" "request failed: ${first_line:-see $out}"
    fi
}

# Resolve the target from `xpu-smi discovery -j`; the table is display-only.
# Prints "<bdf>\t<drm>" and returns 0, or prints a reason and
# returns 3 bad JSON, 4 no devices, 5 missing or invalid fields, 6 target
# absent, 7 xpu-smi reported an error object.
discovery_lookup() {
    python3 - "$discovery_json" "$target_gpu" <<'PY'
import json
import re
import sys

path, target = sys.argv[1], sys.argv[2]
try:
    with open(path) as handle:
        doc = json.load(handle)
except (OSError, ValueError) as exc:
    print(f"{type(exc).__name__}: {exc}")
    raise SystemExit(3)

if isinstance(doc, dict) and "error" in doc:
    print(str(doc["error"])[:200])
    raise SystemExit(7)
if not isinstance(doc, dict):
    print("expected a discovery object")
    raise SystemExit(3)

devices = doc.get("device_list")
if not isinstance(devices, list):
    print("device_list must be an array")
    raise SystemExit(3)
if not devices:
    print("device_list is empty")
    raise SystemExit(4)

device_ids = []
for device in devices:
    device_id = device.get("device_id") if isinstance(device, dict) else None
    if (
        not isinstance(device_id, (int, str))
        or isinstance(device_id, bool)
        or not re.fullmatch(r"[0-9]+", str(device_id))
    ):
        print("each device must have a non-negative numeric device_id")
        raise SystemExit(3)
    device_ids.append(int(device_id))
if len(set(device_ids)) != len(device_ids):
    print("duplicate device_id values")
    raise SystemExit(3)

for device, device_id in zip(devices, device_ids):
    if device_id != int(target):
        continue
    fields = {
        "pci_bdf_address": r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]",
        "drm_device": r"/dev/dri/card[0-9]+",
    }
    invalid = [
        name for name, pattern in fields.items()
        if not isinstance(device.get(name), str) or not re.fullmatch(pattern, device[name])
    ]
    if invalid:
        print(", ".join(invalid))
        raise SystemExit(5)
    print("%s\t%s" % (device["pci_bdf_address"], device["drm_device"]))
    raise SystemExit(0)

print(", ".join(map(str, device_ids)))
raise SystemExit(6)
PY
}

# Kernel-log review is triage evidence, not a health verdict: matches mean
# "read these". Only xe/i915 lines count as driver activity; IOMMU and DRM
# core lines are kept for review but prove nothing about the GPU driver. The
# leading `\b` keeps `fault` from matching `Default`; no trailing boundary,
# so `errors`, `Resetting` and `Timedout` still match.
kernel_log_selector='guc|huc|iommu|drm|\bxe\b|i915|level.?zero'
kernel_log_driver='\b(xe|i915)\b'
kernel_log_faults='\b(error|fail|warn|timed? ?out|reset|hang|wedged|fault)'

check_kernel_log_review() {
    local driver_lines match_lines

    if ! has_command journalctl; then
        record WARN kernel-log-review "journalctl not found; kernel log not reviewed"
        return
    fi

    # A read failure is not the same as zero matches.
    if ! journalctl -k --no-pager >"$out_dir/kernel-log.txt" 2>"$out_dir/kernel-log.err"; then
        record WARN kernel-log-review "could not read the kernel log; see kernel-log.err"
        return
    fi

    driver_lines=$(grep -icE "$kernel_log_driver" "$out_dir/kernel-log.txt" || true)
    grep -iE "$kernel_log_selector" "$out_dir/kernel-log.txt" \
        | grep -iE "$kernel_log_faults" >"$out_dir/kernel-log-review.txt" || true
    match_lines=$(wc -l <"$out_dir/kernel-log-review.txt" | tr -d ' ')
    if [ "$driver_lines" -eq 0 ]; then
        record WARN kernel-log-review "no xe/i915 driver log lines; review kernel-log.txt and kernel-log.err"
    elif [ "$match_lines" -eq 0 ]; then
        record INFO kernel-log-review "no matching messages in $driver_lines GPU driver log line(s)"
    else
        record INFO kernel-log-review "$match_lines message(s) to review; see kernel-log-review.txt"
    fi
}

check_target_driver() {
    local bdf="$1"
    local drm_device="$2"
    local driver=""
    local drm_name=""
    local sysfs_driver=""

    : >"$out_dir/target-driver.txt"
    {
        printf 'target_gpu=%s\n' "$target_gpu"
        printf 'pci_bdf=%s\n' "${bdf:-unknown}"
        printf 'drm_device=%s\n' "${drm_device:-unknown}"
    } >>"$out_dir/target-driver.txt"

    if [ -n "$drm_device" ]; then
        drm_name=$(basename "$drm_device")
        sysfs_driver="/sys/class/drm/$drm_name/device/driver"
        if [ -e "$sysfs_driver" ]; then
            driver=$(basename "$(readlink -f "$sysfs_driver")")
            printf 'sysfs_driver=%s\n' "$driver" >>"$out_dir/target-driver.txt"
        fi
    fi

    if [ -z "$driver" ] && [ -n "$bdf" ] && has_command lspci; then
        if lspci -k -s "$bdf" >>"$out_dir/target-driver.txt" 2>&1; then
            driver=$(awk -F': ' '/Kernel driver in use:/ {print $2; exit}' "$out_dir/target-driver.txt")
        fi
    fi

    case "$driver" in
        xe|i915)
            record PASS target-driver "target GPU $target_gpu bound to kernel driver $driver"
            ;;
        vfio*)
            record FAIL target-driver "target GPU $target_gpu bound to $driver, not an Intel DRM driver"
            ;;
        "")
            record WARN target-driver "could not determine target GPU $target_gpu kernel driver; see target-driver.txt"
            ;;
        *)
            record WARN target-driver "target GPU $target_gpu bound to unexpected kernel driver $driver"
            ;;
    esac
}

load_env_file

record PASS host "$(hostname 2>/dev/null || printf unknown)"
record PASS user "$(id 2>/dev/null || printf unknown)"
log_run uname uname -a

target_gpu_found=0
if has_command xpu-smi; then
    if ! has_command python3; then
        record FAIL preflight-dependency "python3 is required to parse xpu-smi discovery JSON"
    elif xpu-smi discovery -j >"$discovery_json" 2>"$out_dir/xpu-smi-discovery.err"; then
        lookup_out=$(discovery_lookup 2>"$out_dir/xpu-smi-discovery-lookup.err")
        lookup_rc=$?
        case "$lookup_rc" in
            0)
                record PASS xpu-discovery "Intel GPU inventory found"
                target_gpu_found=1
                record PASS target-gpu "target GPU $target_gpu appears in discovery"
                check_target_driver "${lookup_out%%	*}" "${lookup_out##*	}"
                ;;
            6)
                record PASS xpu-discovery "Intel GPU inventory found"
                record FAIL target-gpu "target GPU $target_gpu not in discovery; present: $lookup_out"
                ;;
            5)
                record PASS xpu-discovery "Intel GPU inventory found"
                record FAIL target-gpu "target GPU $target_gpu has missing or invalid field(s): $lookup_out"
                ;;
            4)
                record FAIL xpu-discovery "discovery reported no devices ($lookup_out)"
                ;;
            3)
                record FAIL xpu-discovery "discovery JSON invalid ($lookup_out); see xpu-smi-discovery.json"
                ;;
            7)
                record FAIL xpu-discovery "xpu-smi reported an error ($lookup_out); see xpu-smi-discovery.json"
                ;;
            *)
                record FAIL preflight-dependency "discovery parser exited $lookup_rc; see xpu-smi-discovery-lookup.err"
                ;;
        esac
    else
        record FAIL xpu-discovery "xpu-smi discovery -j failed; see xpu-smi-discovery.err"
    fi
    # Captured, never scored: neither the output nor the exit status is a
    # health verdict. Do not add a PASS/WARN row.
    xpu-smi health -l >"$out_dir/xpu-smi-health.txt" 2>&1
    health_rc=$?
    record INFO xpu-health "health command exit=$health_rc; output captured, not assessed; see xpu-smi-health.txt"
    if [ "$target_gpu_found" -eq 1 ]; then
        # xpu-smi 2.0 to 2.2 reject `-n`; `--samples` bounds the run.
        if has_command timeout; then
            stats_cmd=(timeout -k 5s 10s xpu-smi stats -d "$target_gpu" --samples 1)
        else
            stats_cmd=(xpu-smi stats -d "$target_gpu" --samples 1)
        fi
        if "${stats_cmd[@]}" >"$out_dir/xpu-smi-stats-target.txt" 2>&1; then
            record PASS xpu-stats "stats -d $target_gpu --samples 1 completed"
        else
            stats_rc=$?
            if [ "$stats_rc" -eq 124 ]; then
                record WARN xpu-stats "stats probe for target $target_gpu timed out after 10s; partial output in xpu-smi-stats-target.txt"
            else
                record WARN xpu-stats "stats probe for target $target_gpu exited $stats_rc; see xpu-smi-stats-target.txt"
            fi
        fi
    else
        record WARN xpu-stats "skipped because target GPU $target_gpu was not resolved from discovery"
    fi
    log_run xpu-smi-ps xpu-smi ps
else
    record FAIL xpu-smi "xpu-smi not found"
fi
check_kernel_log_review

if [ -d "$dev_dri_dir" ]; then
    ls -l "$dev_dri_dir" >"$out_dir/dev-dri.txt" 2>&1
    stat -c '%a %U %G %n' "$dev_dri_dir"/* >"$out_dir/dev-dri-stat.txt" 2>/dev/null || true
    if compgen -G "$dev_dri_dir/renderD*" >/dev/null; then
        record PASS dev-dri "render nodes exist"
    else
        record FAIL dev-dri "no $dev_dri_dir/renderD* nodes"
    fi
else
    record FAIL dev-dri "$dev_dri_dir does not exist"
fi

if getent group render >/dev/null 2>&1; then
    render_gid=$(getent group render | cut -d: -f3)
    if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx render; then
        record PASS render-group "current user is in render group ($render_gid)"
    else
        record WARN render-group "current user is not in render group ($render_gid)"
    fi
else
    render_gid=""
    record WARN render-group "render group not found"
fi

if getent group video >/dev/null 2>&1; then
    video_gid=$(getent group video | cut -d: -f3)
    record PASS video-group "video group exists ($video_gid)"
else
    video_gid=""
    record WARN video-group "video group not found"
fi

if has_command docker; then
    if docker info >"$out_dir/docker-info.txt" 2>&1; then
        record PASS docker "daemon reachable by current user"
    else
        record FAIL docker "docker info failed; see docker-info.txt"
    fi
    if docker buildx version >"$out_dir/docker-buildx.txt" 2>&1; then
        record PASS docker-buildx "$(sed -n '1p' "$out_dir/docker-buildx.txt")"
    else
        record WARN docker-buildx "buildx unavailable; image builds may fail"
    fi
    docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' \
        >"$out_dir/docker-ps.txt" 2>&1 || true
else
    record FAIL docker "docker not found"
fi

if df -Pk /dev/shm >"$out_dir/df-dev-shm.txt" 2>&1; then
    shm_kb=$(awk 'NR==2 {print $2}' "$out_dir/df-dev-shm.txt")
    if [ "${shm_kb:-0}" -ge 16777216 ]; then
        record PASS shm "/dev/shm is at least 16 GiB"
    else
        record WARN shm "/dev/shm below 16 GiB; use --ipc=host or --shm-size for servers"
    fi
else
    record WARN shm "could not inspect /dev/shm"
fi

if df -Pk . >"$out_dir/df-workdir.txt" 2>&1; then
    avail_kb=$(awk 'NR==2 {print $4}' "$out_dir/df-workdir.txt")
    if [ "${avail_kb:-0}" -ge 52428800 ]; then
        record PASS disk "working filesystem has at least 50 GiB free"
    else
        record WARN disk "less than 50 GiB free; large image builds may fail"
    fi
else
    record WARN disk "could not inspect working filesystem"
fi

proxy_seen=0
for name in HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY http_proxy https_proxy ftp_proxy all_proxy; do
    value="${!name:-}"
    if [ -n "$value" ]; then
        proxy_seen=1
    fi
done
if [ "$proxy_seen" -eq 1 ]; then
    record PASS proxy-env "proxy variables are set"
else
    record WARN proxy-env "no proxy variables set; fine on direct networks"
fi

combined_no_proxy=",${NO_PROXY:-${no_proxy:-}},"
case "$combined_no_proxy" in
    *,localhost,*|*,127.0.0.1,*)
        record PASS no-proxy-local "NO_PROXY includes localhost or 127.0.0.1"
        ;;
    *)
        record WARN no-proxy-local "NO_PROXY should include localhost and 127.0.0.1"
        ;;
esac

if [ "$network_check" -eq 1 ]; then
    http_probe dockerhub https://registry-1.docker.io/v2/
    http_probe huggingface https://huggingface.co/
    http_probe xpu-wheels https://download.pytorch.org/whl/xpu/
fi

if [ -n "$image" ]; then
    if ! has_command docker; then
        record FAIL image-preflight "docker not found"
    elif ! docker image inspect "$image" >"$out_dir/image-inspect.txt" 2>&1; then
        record FAIL image-preflight "image not local: $image"
    else
        default_image_command='
set -e
if [ -f /opt/intel/oneapi/setvars.sh ]; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null || true
fi
id
ls -l /dev/dri
probe_ok=0
if command -v sycl-ls >/dev/null 2>&1; then
  sycl-ls && probe_ok=1 || true
fi
if command -v python3 >/dev/null 2>&1; then
  if python3 - <<PY
import torch
print("torch", torch.__version__)
print("xpu_available", torch.xpu.is_available())
count = torch.xpu.device_count()
print("xpu_count", count)
assert torch.xpu.is_available(), "torch.xpu.is_available() is False"
assert count > 0, "no XPU devices visible"
print("device0", torch.xpu.get_device_name(0))
PY
  then
    probe_ok=1
  else
    echo "python XPU visibility probe failed or unavailable"
  fi
fi
if [ "$probe_ok" -eq 0 ]; then
  echo "no XPU visibility probe succeeded"
  exit 1
fi
'
        cmd_to_run="${image_command:-$default_image_command}"
        if [ "$image_network" = "host" ]; then
            record WARN image-network "using host network because --image-network host was explicitly requested"
        else
            record PASS image-network "using Docker network mode: $image_network"
        fi
        docker_args=(run --rm --network "$image_network" --ipc=host --shm-size=16g
            --device "$dev_dri_dir"
            -e "ZE_AFFINITY_MASK=$target_gpu"
            -e SYCL_UR_USE_LEVEL_ZERO_V2=0
            -e HTTP_PROXY -e HTTPS_PROXY -e FTP_PROXY -e ALL_PROXY
            -e http_proxy -e https_proxy -e ftp_proxy -e all_proxy
            -e NO_PROXY -e no_proxy)
        if [ -n "${render_gid:-}" ]; then
            docker_args+=(--group-add "$render_gid")
        fi
        if [ -n "${video_gid:-}" ]; then
            docker_args+=(--group-add "$video_gid")
        fi
        docker_args+=(--entrypoint /bin/bash "$image" -lc "$cmd_to_run")
        if docker "${docker_args[@]}" >"$out_dir/image-preflight.txt" 2>&1; then
            record PASS image-preflight "container image sees XPU: $image"
        else
            record FAIL image-preflight "container image failed XPU visibility check; see image-preflight.txt"
        fi
    fi
fi

{
    printf '# xpu-runtime-preflight\n\n'
    printf 'Target GPU: `%s`\n\n' "$target_gpu"
    if [ "$fail_count" -gt 0 ]; then
        verdict="BLOCKED"
    elif [ "$warn_count" -gt 0 ]; then
        verdict="READY WITH WARNINGS"
    else
        verdict="READY"
    fi
    printf 'Verdict: `%s`\n\n' "$verdict"
    printf 'Scope: configuration prerequisites and optional XPU visibility only; workload execution and device health are not certified.\n\n'
    printf 'Review sensor output and kernel-log evidence before treating this report as a workload go-ahead; INFO rows do not affect the verdict.\n\n'
    printf 'Result counts: PASS `%s`, WARN `%s`, FAIL `%s`.\n\n' \
        "$pass_count" "$warn_count" "$fail_count"
    first_fail=$(awk -F '\t' 'NR > 1 && $1 == "FAIL" {print $2 ": " $3; exit}' "$status_tsv")
    if [ -n "$first_fail" ]; then
        printf 'First blocker: %s\n\n' "$first_fail"
    fi
    printf '## Status\n\n'
    printf '| Status | Check | Detail |\n'
    printf '|---|---|---|\n'
    tail -n +2 "$status_tsv" | while IFS="$(printf '\t')" read -r status check detail; do
        printf '| `%s` | `%s` | %s |\n' "$status" "$check" "$detail"
    done
    printf '\n## Next Skill Routing\n\n'
    printf -- '- GPU inventory or diagnostics failures: **xpu-discover**.\n'
    printf -- '- Container GPU visibility or group failures: **xpu-container-run**.\n'
    printf -- '- Network, proxy, Docker pull, or model download failures: review proxy-env, no-proxy-local, docker-daemon, and container env checks; then use the failing runtime/container skill troubleshooting section.\n'
    printf -- '- Runtime, benchmark, or profile logs after launch: return to the skill that produced the log and follow its troubleshooting guidance.\n'
    printf -- '- Model capacity concerns after readiness passes: **model-can-it-fit**.\n'
} >"$summary_md"

printf '\nWrote %s\n' "$summary_md"
printf 'PASS=%s WARN=%s FAIL=%s\n' "$pass_count" "$warn_count" "$fail_count"

if [ "$fail_count" -gt 0 ]; then
    exit 1
fi
exit 0
