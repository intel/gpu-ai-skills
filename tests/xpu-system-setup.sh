#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

count_sycl_records() {
    awk '
        function flush_record() {
            if (!in_record) {
                return
            }
            if (record_intel) {
                intel++
            }
            if (record_intel && record_gpu) {
                intel_gpu++
            }
            in_record = 0
            record_intel = 0
            record_gpu = 0
        }
        NF == 0 {
            flush_record()
            next
        }
        /^\[/ {
            flush_record()
        }
        {
            in_record = 1
            line = tolower($0)
            if (line ~ /intel/) {
                record_intel = 1
            }
            if (line ~ /^\[[^]]*:gpu:[^]]*\]/ ||
                line ~ /^[[:space:]]*(device[[:space:]]+)?type[[:space:]]*:[[:space:]]*gpu([[:space:]]|$)/) {
                record_gpu = 1
            }
        }
        END {
            flush_record()
            printf "%d %d\n", intel + 0, intel_gpu + 0
        }
    '
}

repo_is_configured_from_text() {
    local codename="$1" path="$2"
    if grep -RhsE '^[[:space:]]*deb[[:space:]]+' "$path" 2>/dev/null \
        | awk -v codename="$codename" '
            {
                line = $0
                sub(/^[[:space:]]*deb[[:space:]]+/, "", line)
                sub(/^\[[^]]+\][[:space:]]+/, "", line)
                if (line ~ "^https?://repositories\\.intel\\.com/gpu/ubuntu/?[[:space:]]+" codename "/intel-omix[[:space:]]+unified([[:space:]]+.*)?$") {
                    found = 1
                    exit
                }
            }
            END {
                exit !found
            }
        '; then
        return 0
    fi

    awk -v codename="$codename" '
        BEGIN {
            RS = ""
            IGNORECASE = 1
        }
        {
            block = tolower($0)
            if (block ~ /(^|\n)types:[^\n]*([[:space:]]|^)deb([[:space:]]|$)/ &&
                block ~ /(^|\n)uris:[^\n]*https?:\/\/repositories\.intel\.com\/gpu\/ubuntu\/?([[:space:]]|$)/ &&
                block ~ "(^|\n)suites:[^\n]*" tolower(codename) "/intel-omix([[:space:]]|$)" &&
                block ~ /(^|\n)components:[^\n]*([[:space:]]|^)unified([[:space:]]|$)/) {
                found = 1
                exit
            }
        }
        END {
            exit !found
        }
    ' "$path"
}

assert_counts() {
    local name="$1" input="$2" expected_intel="$3" expected_gpu="$4" actual
    actual=$(printf '%s\n' "$input" | count_sycl_records)
    if [[ "$actual" != "$expected_intel $expected_gpu" ]]; then
        printf 'FAIL %s: expected "%s %s", got "%s"\n' "$name" "$expected_intel" "$expected_gpu" "$actual" >&2
        exit 1
    fi
}

assert_repo_match() {
    local name="$1" codename="$2" content="$3" suffix="$4" tmp
    tmp=$(mktemp "/tmp/xpu-system-setup-repo-${suffix}.XXXXXX")
    printf '%s\n' "$content" >"$tmp"
    if ! repo_is_configured_from_text "$codename" "$tmp"; then
        printf 'FAIL %s: repo entry was not detected\n' "$name" >&2
        rm -f "$tmp"
        exit 1
    fi
    rm -f "$tmp"
}

assert_counts \
    "bracketed records" \
    "[opencl:cpu:0] Intel(R) Xeon(R)
[level_zero:gpu:0] Intel(R) Arc(TM) Pro B70" \
    "2" "1"

assert_counts \
    "type field records" \
    "Device:
  Vendor: Intel(R) Corporation
  Type: GPU

Device:
  Vendor: Intel(R) Corporation
  Type: CPU" \
    "2" "1"

assert_counts \
    "mixed multiline records" \
    "[opencl:cpu:0]
  Vendor : Intel(R) Corporation
  Device : Intel(R) Xeon(R)

[level_zero:gpu:0]
  Vendor : Intel(R) Corporation
  Device : Intel(R) Arc(TM) Pro B70" \
    "2" "1"

assert_repo_match \
    "list repo entry with trailing components" \
    "noble" \
    "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble/intel-omix unified main" \
    "list"

assert_repo_match \
    "deb822 repo entry" \
    "noble" \
    "Types: deb
URIs: https://repositories.intel.com/gpu/ubuntu
Suites: noble/intel-omix
Components: unified
Signed-By: /usr/share/keyrings/intel-graphics.gpg" \
    "sources"

# Kernel-log review. The block is extracted from the shipped script, not
# copied, so these cases follow the script; journalctl is mocked.
setup_script="$(cd "$(dirname "$0")/.." && pwd)/plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

kernel_log_block=$(awk '
    /^    local log_selector=/ { in_block = 1 }
    in_block { print }
    in_block && /^    fi$/ { exit }
' "$setup_script")
if [[ "$kernel_log_block" != *'journalctl -k'* ]]; then
    printf 'FAIL kernel-log block: not found in %s\n' "$setup_script" >&2
    exit 1
fi

# Same shell options as the script: no `set -e`, so `((warn_count++))` from 0
# does not abort.
{
    cat <<'EOF'
set -uo pipefail
warn_count=0
info() { printf 'INFO %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*"; }
kernel_log_block() {
EOF
    printf '%s\n' "$kernel_log_block"
    cat <<'EOF'
}
kernel_log_block
printf 'warn_count=%s\n' "$warn_count"
EOF
} >"$tmp/kernel-log-block.sh"

mkdir -p "$tmp/bin" "$tmp/empty-bin"
cat >"$tmp/bin/journalctl" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${MOCK_JOURNAL_FAIL:-}" ]]; then
    printf 'mock journalctl: no access\n' >&2
    exit 1
fi
cat "${MOCK_JOURNAL:?}"
EOF
chmod +x "$tmp/bin/journalctl"

run_kernel_log_case() {
    local name="$1" journal="$2" path="${3:-$tmp/bin:$PATH}"
    mkdir -p "$tmp/$name"
    printf '%s\n' "$journal" >"$tmp/$name/journal.txt"
    if ! MOCK_JOURNAL="$tmp/$name/journal.txt" OUT_DIR="$tmp/$name" PATH="$path" \
        "$BASH" "$tmp/kernel-log-block.sh" >"$tmp/$name/stdout.txt" 2>&1; then
        printf 'FAIL %s: kernel-log block exited non-zero\n' "$name" >&2
        cat "$tmp/$name/stdout.txt" >&2
        exit 1
    fi
}

assert_kernel_log_stdout() {
    local name="$1" expected="$2"
    if ! grep -qF -- "$expected" "$tmp/$name/stdout.txt"; then
        printf 'FAIL %s: expected output containing "%s", got:\n' "$name" "$expected" >&2
        cat "$tmp/$name/stdout.txt" >&2
        exit 1
    fi
}

# Compares the review file with the `uniq -c` count padding removed.
assert_kernel_log_review() {
    local name="$1" expected="$2" actual
    actual=$(sed 's/^ *//' "$tmp/$name/kernel-log-review.txt")
    if [[ "$actual" != "$expected" ]]; then
        printf 'FAIL %s: expected review file:\n%s\ngot:\n%s\n' "$name" "$expected" "$actual" >&2
        exit 1
    fi
}

# Repeats collapse across timestamps, a space-padded day and a non-`kernel:`
# tag, including repeats that are not adjacent in the log. A second BDF stays
# its own line, `kernel: ` inside a message is kept, and `Default` and a clean
# driver line are not selected.
run_kernel_log_case "repeats" \
    "Sep  1 09:00:00 host kernel: xe 0000:03:00.0: [drm] Found battlemage
Sep  1 09:00:01 host kernel: iommu: Default domain type: Translated
Sep  1 09:00:02 host kernel: xe 0000:03:00.0: [drm] *ERROR* GT0: TLB invalidation timed out
Sep  2 10:11:12 host kernel: xe 0000:03:00.0: [drm] GT0: Engine reset: kernel: nested tag text
Sep 12 23:59:58 host kernel: xe 0000:03:00.0: [drm] *ERROR* GT0: TLB invalidation timed out
Sep 12 23:59:59 host kernel: xe 0000:04:00.0: [drm] *ERROR* GT0: TLB invalidation timed out
Sep 13 00:00:01 host kernel: xe 0000:03:00.0: [drm] GT0: Engine reset: kernel: nested tag text
Sep 13 00:00:02 host unknown: xe 0000:03:00.0: [drm] *ERROR* GT0: TLB invalidation timed out"
assert_kernel_log_review "repeats" \
    "3 xe 0000:03:00.0: [drm] *ERROR* GT0: TLB invalidation timed out
2 xe 0000:03:00.0: [drm] GT0: Engine reset: kernel: nested tag text
1 xe 0000:04:00.0: [drm] *ERROR* GT0: TLB invalidation timed out"
assert_kernel_log_stdout "repeats" "INFO Verification: 3 distinct message(s) (6 matching line(s)) to review in 7 xe/i915 driver log line(s)"
assert_kernel_log_stdout "repeats" "warn_count=0"

run_kernel_log_case "no-matches" \
    "Sep  1 09:00:00 host kernel: xe 0000:03:00.0: [drm] Found battlemage"
assert_kernel_log_review "no-matches" ""
assert_kernel_log_stdout "no-matches" "INFO Verification: 0 distinct message(s) (0 matching line(s)) to review in 1 xe/i915 driver log line(s)"
assert_kernel_log_stdout "no-matches" "warn_count=0"

run_kernel_log_case "no-driver-lines" \
    "Sep  1 09:00:00 host kernel: usb 1-1: reset high-speed USB device number 2"
assert_kernel_log_review "no-driver-lines" ""
assert_kernel_log_stdout "no-driver-lines" "WARN Verification: no xe/i915 driver log lines"
assert_kernel_log_stdout "no-driver-lines" "warn_count=1"

MOCK_JOURNAL_FAIL=1 run_kernel_log_case "unreadable" ""
assert_kernel_log_stdout "unreadable" "WARN Verification: could not read the kernel log"
assert_kernel_log_stdout "unreadable" "warn_count=1"

run_kernel_log_case "no-journalctl" "" "$tmp/empty-bin"
assert_kernel_log_stdout "no-journalctl" "WARN Verification: journalctl not found; kernel log not reviewed"
assert_kernel_log_stdout "no-journalctl" "warn_count=1"

echo "OK xpu-system-setup sycl parser, repo entry and kernel-log checks passed"
