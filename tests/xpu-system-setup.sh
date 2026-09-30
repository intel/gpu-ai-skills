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

echo "OK xpu-system-setup sycl parser checks passed"
