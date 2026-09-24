#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Mocked integration checks for xpu-runtime-preflight.

set -euo pipefail

cd "$(dirname "$0")/.."

script="plugins/intel-gpu-ai-skills/skills/xpu-runtime-preflight/scripts/check_runtime_preflight.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

mock_bin="$tmp/bin"
mkdir -p "$mock_bin"
mock_dev_dri="$tmp/dev-dri"
mkdir -p "$mock_dev_dri"
: >"$mock_dev_dri/renderD128"

cat >"$mock_bin/xpu-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >>"${MOCK_XPU_SMI_LOG:?}"

case "${1:-}" in
    discovery)
        # The table exists so the suite can prove preflight never requests it.
        if [ "$*" = discovery ]; then
            printf '| Device ID | Device Information |\n| 0         | Device Name: Intel Mock XPU |\n'
            exit 0
        fi
        [ "$*" = 'discovery -j' ] || exit 2
        case "${MOCK_DISCOVERY_MODE:-ok}" in
            fail)
                printf 'mock discovery failure\n' >&2
                exit 1
                ;;
            malformed)
                printf 'not json at all\n'
                ;;
            root-array) printf '[]\n' ;;
            root-null) printf 'null\n' ;;
            error) printf '{"error":"Level Zero initialization failed"}\n' ;;
            wrong-list) printf '{"device_list":{}}\n' ;;
            wrong-device) printf '{"device_list":[null]}\n' ;;
            wrong-id) printf '{"device_list":[{"device_id":false}]}\n' ;;
            duplicate) printf '{"device_list":[{"device_id":0},{"device_id":0}]}\n' ;;
            wrong-fields)
                printf '{"device_list":[{"device_id":0,"pci_bdf_address":42,"drm_device":["/dev/dri/card1"]}]}\n'
                ;;
            invalid-fields)
                printf '{"device_list":[{"device_id":0,"pci_bdf_address":"not-a-bdf","drm_device":"/dev/dri/card1\\n"}]}\n'
                ;;
            trailing-junk)
                printf '{"device_list":[{"device_id":0,"pci_bdf_address":"0000:18:00.0","drm_device":"/dev/dri/card1 junk"}]}\n'
                ;;
            multiple)
                printf '{"device_list":[{"device_id":0},{"device_id":"2","pci_bdf_address":"0000:28:00.0","drm_device":"/dev/dri/card2"}]}\n'
                ;;
            empty)
                printf '{"device_list":[]}\n'
                ;;
            nofields)
                printf '{"device_list":[{"device_id":0}]}\n'
                ;;
            *)
                cat <<'OUT'
{"device_list":[{"device_id":0,"device_name":"Intel Mock XPU","pci_bdf_address":"0000:18:00.0","drm_device":"/dev/dri/card1"}]}
OUT
                ;;
        esac
        ;;
    health)
        [ "$*" = 'health -l' ] || exit 2
        case "${MOCK_HEALTH_MODE:-ok}" in
            fail)
                printf '[Error] ZE_RESULT_ERROR_UNSUPPORTED_FEATURE\n' >&2
                exit 1
                ;;
            zero-errors)
                printf '[Error] Failed to get core temperature\n' >&2
                printf 'Power: OK\nCore temperature: Unknown\n'
                ;;
            critical) printf 'Power: Critical\nMemory: Warning\nCore temperature: Unknown\n' ;;
            *) printf 'Power: OK\nCore temperature: Unknown\n' ;;
        esac
        ;;
    stats)
        [ "$#" -eq 5 ] && [ "$2" = -d ] && [[ "$3" =~ ^[0-9]+$ ]] \
            && [ "$4" = --samples ] && [ "$5" = 1 ] || exit 2
        printf 'mock xpu-smi stats\n'
        ;;
    diag)
        # Arc/Battlemage builds have no diag subcommand.
        printf "'diag' is not a valid subcommand.\n" >&2
        exit 2
        ;;
    ps)
        printf 'mock xpu-smi ps\n'
        ;;
    *)
        printf 'unexpected xpu-smi args: %s\n' "$*" >&2
        exit 1
        ;;
esac
EOF

cat >"$mock_bin/journalctl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >>"${MOCK_JOURNALCTL_LOG:?}"

case "${MOCK_JOURNAL_MODE:-ok}" in
    denied)
        printf 'Permission denied\n' >&2
        exit 1
        ;;
    empty) exit 0 ;;
    no-driver)
        printf 'mockhost kernel: unrelated system message\n'
        exit 0
        ;;
    faults)
        printf '%s\n' 'xe: GuC load failed' 'xe: page fault detected' 'drm: *ERROR* device unavailable'
        ;;
    benign-reset) printf 'xe: reset completed successfully\n' ;;
esac

# Three driver lines, no faults. "Default" trips an unanchored `fault` regex.
cat <<'OUT'
Sep 08 11:50:52 mockhost kernel: iommu: Default domain type: Translated
Sep 08 11:50:53 mockhost kernel: xe 0000:18:00.0: [drm] GuC firmware version 70.29.2
Sep 08 11:50:53 mockhost kernel: xe 0000:18:00.0: [drm] HuC firmware authenticated
OUT
EOF

cat >"$mock_bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >>"${MOCK_DOCKER_LOG:?}"

case "${1:-}" in
    info)
        printf 'Server Version: mock\n'
        ;;
    buildx)
        if [ "${2:-}" = version ]; then
            printf 'github.com/docker/buildx mock\n'
        else
            printf 'unexpected docker buildx args: %s\n' "$*" >&2
            exit 1
        fi
        ;;
    ps)
        printf 'NAMES IMAGE STATUS PORTS\n'
        ;;
    image)
        if [ "${2:-}" = inspect ]; then
            printf 'mock image inspect ok\n'
        else
            printf 'unexpected docker image args: %s\n' "$*" >&2
            exit 1
        fi
        ;;
    run)
        printf 'mock container sees XPU\n'
        ;;
    *)
        printf 'unexpected docker args: %s\n' "$*" >&2
        exit 1
        ;;
esac
EOF

chmod +x "$mock_bin/xpu-smi" "$mock_bin/docker" "$mock_bin/journalctl"

cat >"$mock_bin/clinfo" <<'EOF'
#!/usr/bin/env bash
printf 'Device Name Intel Mock XPU\n'
EOF
chmod +x "$mock_bin/clinfo"

export PATH="$mock_bin:$PATH"
export MOCK_DOCKER_LOG="$tmp/docker.log"
export MOCK_XPU_SMI_LOG="$tmp/xpu-smi.log"
export MOCK_JOURNALCTL_LOG="$tmp/journalctl.log"
export XPU_PREFLIGHT_DEV_DRI_DIR="$mock_dev_dri"

fail() {
    printf 'FAIL %s\n' "$1" >&2
    exit 1
}

require_file() {
    [ -f "$1" ] || fail "missing file: $1"
}

require_grep() {
    local pattern="$1"
    local file="$2"
    if ! grep -Eq -- "$pattern" "$file"; then
        printf '%s\n' "--- $file ---" >&2
        sed 's/^/  /' "$file" >&2
        fail "pattern not found: $pattern"
    fi
}

require_no_grep() {
    local pattern="$1"
    local file="$2"
    if grep -Eq -- "$pattern" "$file"; then
        printf '%s\n' "--- $file ---" >&2
        sed 's/^/  /' "$file" >&2
        fail "unexpected pattern found: $pattern"
    fi
}

require_rc() {
    local out_dir="$1"
    local expected="$2"
    local actual

    actual=$(cat "$out_dir.rc")
    if [ "$actual" != "$expected" ]; then
        printf '%s\n' "--- $out_dir.stdout ---" >&2
        sed 's/^/  /' "$out_dir.stdout" >&2
        printf '%s\n' "--- $out_dir.stderr ---" >&2
        sed 's/^/  /' "$out_dir.stderr" >&2
        fail "$out_dir exited $actual, expected $expected"
    fi
}

require_single_network_mode() {
    local mode="$1"
    local exact_count network_flag_count

    exact_count=$(grep -Ec -- "(^| )--network $mode( |$)" "$MOCK_DOCKER_LOG")
    network_flag_count=$(grep -Eo -- '(^| )--network( |$)' "$MOCK_DOCKER_LOG" | wc -l | tr -d ' ')
    [ "$exact_count" = 1 ] || fail "expected exactly one --network $mode in docker log"
    [ "$network_flag_count" = 1 ] || fail "expected exactly one --network flag in docker log"
}

run_preflight() {
    local out_dir="$1"
    shift

    run_preflight_target "$out_dir" 0 "$@"
}

run_preflight_target() {
    local out_dir="$1"
    local target="$2"
    shift 2

    set +e
    "$script" --target-gpu "$target" --out-dir "$out_dir" "$@" >"$out_dir.stdout" 2>"$out_dir.stderr"
    local rc=$?
    set -e
    printf '%s\n' "$rc" >"$out_dir.rc"
}

help_text=$("$script" --help)
case "$help_text" in
    *"--image-network MODE"*) ;;
    *) fail "--help does not document --image-network" ;;
esac

default_out="$tmp/default"
run_preflight "$default_out" --image mock-image --image-command 'echo ok'
require_rc "$default_out" 0
require_file "$default_out/SUMMARY.md"
require_file "$default_out/status.tsv"
require_file "$default_out/preflight.log"
require_grep $'^PASS\timage-network\tusing Docker network mode: bridge$' "$default_out/status.tsv"
require_grep $'^PASS\timage-preflight\tcontainer image sees XPU: mock-image$' "$default_out/status.tsv"
require_single_network_mode bridge
require_grep $'^PASS\ttarget-gpu\ttarget GPU 0 appears in discovery$' "$default_out/status.tsv"
# The stats probe must use flags the real binary accepts.
require_grep $'^PASS\txpu-stats\t' "$default_out/status.tsv"
require_grep '\-\-samples 1' "$MOCK_XPU_SMI_LOG"
require_no_grep '(^| )-n( |$)' "$MOCK_XPU_SMI_LOG"
# The JSON fields must reach the driver check.
require_grep 'pci_bdf=0000:18:00\.0' "$default_out/target-driver.txt"
require_grep 'drm_device=/dev/dri/card1' "$default_out/target-driver.txt"
require_no_grep '^diag' "$MOCK_XPU_SMI_LOG"
# Only the JSON inventory is parsed; the table is never requested.
require_grep '^discovery -j$' "$MOCK_XPU_SMI_LOG"
require_no_grep '^discovery$' "$MOCK_XPU_SMI_LOG"
require_file "$default_out/xpu-smi-discovery.json"
# Health is captured, never scored.
require_grep $'^INFO\txpu-health\t' "$default_out/status.tsv"
require_no_grep $'^(PASS|WARN|FAIL)\txpu-health' "$default_out/status.tsv"
require_file "$default_out/xpu-smi-health.txt"
# Kernel log is advisory only.
require_grep $'^INFO\tkernel-log-review\tno matching messages in 2 GPU driver log line' "$default_out/status.tsv"
require_no_grep $'^(PASS|WARN|FAIL)\tkernel-log-review' "$default_out/status.tsv"
require_file "$default_out/kernel-log.txt"
require_grep '^-k --no-pager$' "$MOCK_JOURNALCTL_LOG"

: >"$MOCK_DOCKER_LOG"
host_out="$tmp/host"
run_preflight "$host_out" --image mock-image --image-command 'echo ok' --image-network host
require_rc "$host_out" 0
require_file "$host_out/SUMMARY.md"
require_file "$host_out/status.tsv"
require_grep $'^WARN\timage-network\tusing host network because --image-network host was explicitly requested$' "$host_out/status.tsv"
require_single_network_mode host

missing_out="$tmp/missing-target"
: >"$MOCK_XPU_SMI_LOG"
run_preflight_target "$missing_out" 2
require_rc "$missing_out" 1
require_grep $'^FAIL\ttarget-gpu\ttarget GPU 2 not in discovery; present: 0$' "$missing_out/status.tsv"
require_grep $'^WARN\txpu-stats\tskipped because target GPU 2 was not resolved from discovery$' "$missing_out/status.tsv"
require_no_grep '^stats ' "$MOCK_XPU_SMI_LOG"
require_grep 'not collected in this run' "$missing_out/xpu-smi-stats-target.txt"

multi_out="$tmp/multiple"
MOCK_DISCOVERY_MODE=multiple run_preflight_target "$multi_out" 2
require_rc "$multi_out" 0
require_grep 'pci_bdf=0000:28:00\.0' "$multi_out/target-driver.txt"
require_grep 'drm_device=/dev/dri/card2' "$multi_out/target-driver.txt"

# Each discovery failure must FAIL without falling back to the table.
for mode in fail malformed root-array root-null error wrong-list wrong-device wrong-id duplicate empty nofields wrong-fields invalid-fields trailing-junk; do
    mode_out="$tmp/discovery-$mode"
    : >"$MOCK_XPU_SMI_LOG"
    MOCK_DISCOVERY_MODE="$mode" run_preflight_target "$mode_out" 0
    require_rc "$mode_out" 1
    case "$mode" in
        fail)      require_grep $'^FAIL\txpu-discovery\txpu-smi discovery -j failed' "$mode_out/status.tsv" ;;
        empty)     require_grep $'^FAIL\txpu-discovery\tdiscovery reported no devices' "$mode_out/status.tsv" ;;
        error)     require_grep $'^FAIL\txpu-discovery\txpu-smi reported an error \\(Level Zero initialization failed\\)' "$mode_out/status.tsv" ;;
        trailing-junk)
            # Full-string matching: a suffix on an otherwise valid field is rejected.
            require_grep $'^FAIL\ttarget-gpu\ttarget GPU 0 has missing or invalid field\\(s\\): drm_device$' "$mode_out/status.tsv" ;;
        nofields|wrong-fields|invalid-fields)
            require_grep $'^FAIL\ttarget-gpu\ttarget GPU 0 has missing or invalid field' "$mode_out/status.tsv" ;;
        *) require_grep $'^FAIL\txpu-discovery\tdiscovery JSON invalid' "$mode_out/status.tsv" ;;
    esac
    require_no_grep '^discovery$' "$MOCK_XPU_SMI_LOG"
    require_no_grep '^stats ' "$MOCK_XPU_SMI_LOG"
    require_no_grep 'Traceback' "$mode_out.stderr"
done

# A minimal PATH proves the dependency failure without removing host packages.
no_python_bin="$tmp/no-python-bin"
mkdir -p "$no_python_bin"
for tool in bash mkdir tee hostname id uname grep sed wc tr ls stat getent cut df tail; do
    tool_path=$(command -v "$tool" 2>/dev/null) || continue
    ln -s "$tool_path" "$no_python_bin/$tool"
done
for tool in xpu-smi docker journalctl; do
    ln -s "$mock_bin/$tool" "$no_python_bin/$tool"
done
no_python_out="$tmp/no-python"
PATH="$no_python_bin" run_preflight "$no_python_out"
require_rc "$no_python_out" 1
require_grep $'^FAIL\tpreflight-dependency\tpython3 is required' "$no_python_out/status.tsv"

# A python3 that cannot run the parser is a dependency failure, not bad JSON.
broken_python_bin="$tmp/broken-python-bin"
mkdir -p "$broken_python_bin"
cat >"$broken_python_bin/python3" <<'EOF'
#!/usr/bin/env bash
printf 'SyntaxError: invalid syntax\n' >&2
exit 1
EOF
chmod +x "$broken_python_bin/python3"
broken_python_out="$tmp/broken-python"
PATH="$broken_python_bin:$PATH" run_preflight "$broken_python_out"
require_rc "$broken_python_out" 1
require_grep $'^FAIL\tpreflight-dependency\tdiscovery parser exited 1' "$broken_python_out/status.tsv"
require_grep 'SyntaxError' "$broken_python_out/xpu-smi-discovery-lookup.err"
require_no_grep 'discovery JSON invalid' "$broken_python_out/status.tsv"

for mode in fail zero-errors critical; do
    health_out="$tmp/health-$mode"
    MOCK_HEALTH_MODE="$mode" run_preflight "$health_out"
    require_rc "$health_out" 0
    expected_rc=0
    [ "$mode" != fail ] || expected_rc=1
    require_grep "^INFO"$'\txpu-health\t'".*exit=$expected_rc;.*not assessed" "$health_out/status.tsv"
    require_no_grep $'^(PASS|WARN|FAIL)\txpu-health' "$health_out/status.tsv"
    require_grep 'Error|Critical' "$health_out/xpu-smi-health.txt"
    require_grep 'workload execution and device health are not certified' "$health_out/SUMMARY.md"
done

# A probe skipped this run must not leave the previous run's evidence behind.
no_xpu_smi_bin="$tmp/no-xpu-smi-bin"
mkdir -p "$no_xpu_smi_bin"
for tool in bash awk basename cat cut df env getent grep head hostname id ls \
    lspci mkdir python3 readlink sed sh sort stat tail tee timeout tr uname wc; do
    tool_path=$(command -v "$tool" 2>/dev/null) || continue
    ln -s "$tool_path" "$no_xpu_smi_bin/$tool"
done
for tool in docker journalctl; do
    ln -s "$mock_bin/$tool" "$no_xpu_smi_bin/$tool"
done

stale_out="$tmp/stale-evidence"
MOCK_HEALTH_MODE=critical run_preflight "$stale_out"
require_grep 'Error|Critical' "$stale_out/xpu-smi-health.txt"
PATH="$no_xpu_smi_bin" run_preflight "$stale_out"
require_grep $'^FAIL\txpu-smi\txpu-smi not found' "$stale_out/status.tsv"
require_no_grep 'Error|Critical' "$stale_out/xpu-smi-health.txt"

# Flag-gated probes must not leave the previous run's evidence behind either.
stale_image_out="$tmp/stale-image"
run_preflight "$stale_image_out" --image mock-image --image-command 'echo ok'
require_grep $'^PASS\timage-preflight\t' "$stale_image_out/status.tsv"
require_no_grep 'not collected in this run' "$stale_image_out/image-preflight.txt"
run_preflight "$stale_image_out"
require_no_grep 'image-preflight' "$stale_image_out/status.tsv"
require_grep 'not collected in this run' "$stale_image_out/image-preflight.txt"
require_grep 'not collected in this run' "$stale_image_out/image-inspect.txt"

stale_errors_out="$tmp/stale-errors"
MOCK_DISCOVERY_MODE=fail MOCK_JOURNAL_MODE=denied run_preflight "$stale_errors_out"
require_grep 'mock discovery failure' "$stale_errors_out/xpu-smi-discovery.err"
require_grep 'Permission denied' "$stale_errors_out/kernel-log.err"
rm "$no_xpu_smi_bin/journalctl"
PATH="$no_xpu_smi_bin" run_preflight "$stale_errors_out"
require_rc "$stale_errors_out" 1
require_grep $'^FAIL\txpu-smi\txpu-smi not found' "$stale_errors_out/status.tsv"
require_grep $'^WARN\tkernel-log-review\tjournalctl not found' "$stale_errors_out/status.tsv"
require_grep 'not collected in this run' "$stale_errors_out/xpu-smi-discovery.err"
require_grep 'not collected in this run' "$stale_errors_out/kernel-log.err"

for mode in faults benign-reset denied empty no-driver; do
    journal_out="$tmp/journal-$mode"
    MOCK_JOURNAL_MODE="$mode" run_preflight "$journal_out"
    require_rc "$journal_out" 0
    case "$mode" in
        faults|benign-reset)
            matches=3
            [ "$mode" != benign-reset ] || matches=1
            require_grep "^INFO"$'\tkernel-log-review\t'"$matches message" "$journal_out/status.tsv"
            [ "$(wc -l <"$journal_out/kernel-log-review.txt")" -eq "$matches" ] || fail 'wrong log match count'
            ;;
        denied)
            require_grep $'^WARN\tkernel-log-review\tcould not read the kernel log' "$journal_out/status.tsv"
            ;;
        *)
            require_grep $'^WARN\tkernel-log-review\tno xe/i915 driver log lines' "$journal_out/status.tsv"
            ;;
    esac
done

# Reusing an output directory must not leave old matches after a denied read.
MOCK_JOURNAL_MODE=denied run_preflight "$tmp/journal-faults"
require_grep 'not collected in this run' "$tmp/journal-faults/kernel-log-review.txt"
require_grep 'Permission denied' "$tmp/journal-faults/kernel-log.err"
require_no_grep '^diag' "$MOCK_XPU_SMI_LOG"

bad_out="$tmp/bad"
set +e
"$script" --out-dir "$bad_out" --image-network 'bad value' >"$tmp/bad.stdout" 2>"$tmp/bad.stderr"
bad_rc=$?
set -e
[ "$bad_rc" -eq 2 ] || fail "invalid --image-network exited $bad_rc, expected 2"
require_grep '--image-network must be a Docker network mode without whitespace' "$tmp/bad.stderr"

printf 'OK xpu-runtime-preflight mocked integration checks passed\n'
