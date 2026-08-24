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
        cat <<'OUT'
| Device ID | Device Name |
| 0         | Intel Mock XPU |
OUT
        ;;
    diag)
        printf 'mock diag ok\n'
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

chmod +x "$mock_bin/xpu-smi" "$mock_bin/docker"

export PATH="$mock_bin:$PATH"
export MOCK_DOCKER_LOG="$tmp/docker.log"
export MOCK_XPU_SMI_LOG="$tmp/xpu-smi.log"
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
require_grep $'^FAIL\ttarget-gpu\ttarget GPU 2 not found in discovery$' "$missing_out/status.tsv"
require_grep $'^WARN\txpu-diag\tskipped because target GPU 2 was not found$' "$missing_out/status.tsv"
require_no_grep 'diag -d 2' "$MOCK_XPU_SMI_LOG"

bad_out="$tmp/bad"
set +e
"$script" --out-dir "$bad_out" --image-network 'bad value' >"$tmp/bad.stdout" 2>"$tmp/bad.stderr"
bad_rc=$?
set -e
[ "$bad_rc" -eq 2 ] || fail "invalid --image-network exited $bad_rc, expected 2"
require_grep '--image-network must be a Docker network mode without whitespace' "$tmp/bad.stderr"

printf 'OK xpu-runtime-preflight mocked integration checks passed\n'
