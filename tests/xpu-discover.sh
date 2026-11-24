#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Mocked sysfs checks for xpu-discover's scripts/check_pcie_link.sh.

set -euo pipefail

cd "$(dirname "$0")/.."

script="plugins/intel-gpu-ai-skills/skills/xpu-discover/scripts/check_pcie_link.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fail() {
    printf 'FAIL %s\n' "$1" >&2
    exit 1
}

# dev <dir> <vendor> <device> <class> [cur_speed cur_width max_speed max_width]
dev() {
    mkdir -p "$1"
    printf '%s\n' "$2" >"$1/vendor"
    printf '%s\n' "$3" >"$1/device"
    printf '%s\n' "$4" >"$1/class"
    if [ "$#" -ge 8 ]; then
        printf '%s GT/s PCIe\n' "$5" >"$1/current_link_speed"
        printf '%s\n' "$6" >"$1/current_link_width"
        printf '%s GT/s PCIe\n' "$7" >"$1/max_link_speed"
        printf '%s\n' "$8" >"$1/max_link_width"
    fi
}

# b70 <root> <bus> <slot max speed> <slot max width> <cur speed> <cur width> <card max speed> <card max width> [gpu id]
# Arc Pro B70 layout: root port -> card upstream port -> card downstream port
# -> GPU, whose own link is the card's internal x1 @ 2.5 GT/s.
b70() {
    local root=$1 b=$2 rp="$1/devices/pci0000:$2/0000:$2:01.0"
    dev "$rp" 0x8086 0x352a 0x060400 "$5" "$6" "$3" "$4"
    dev "$rp/0000:$b:02.0" 0x8086 0xe2ff 0x060400 "$5" "$6" "$7" "$8"
    dev "$rp/0000:$b:02.0/0000:$b:03.0" 0x8086 0xe2f0 0x060400 2.5 1 2.5 1
    dev "$rp/0000:$b:02.0/0000:$b:03.0/0000:$b:04.0" 0x8086 "${9:-0xe223}" 0x030000 2.5 1 2.5 1
    mkdir -p "$root/bus"
    ln -s "$rp/0000:$b:02.0/0000:$b:03.0/0000:$b:04.0" "$root/bus/0000:$b:04.0"
}

# expect <root> <exit status> <expected output>
expect() {
    local out rc=0
    out=$(bash "$script" --sysfs-root "$1/bus" 2>&1) || rc=$?
    [ "$out" = "$3" ] || fail "output differs for $1:
--- expected
$3
--- got
$out"
    [ "$rc" = "$2" ] || fail "$1 exited $rc, expected $2"
}

# 1. Healthy cards, and an Arc Pro B70 whose card port is capped at Gen4 in a Gen5 slot.
b70 "$tmp/a" 10 32.0 16 32.0 16 32.0 16
b70 "$tmp/a" 20 32.0 16 16.0 16 16.0 16
expect "$tmp/a" 1 "OK    0000:10:04.0  x16 @ 32.0 GT/s
WARN  0000:20:04.0  x16 @ 16.0 GT/s; the slot allows 32.0 GT/s but the card port is limited to 16.0 GT/s"

# 2. Link trained below what both ends allow, in speed and in width.
b70 "$tmp/b" 30 32.0 16 16.0 16 32.0 16
b70 "$tmp/b" 31 32.0 16 32.0 8 32.0 16
expect "$tmp/b" 1 "WARN  0000:30:04.0  x16 @ 16.0 GT/s, but card and slot allow x16 @ 32.0 GT/s (link trained low)
WARN  0000:31:04.0  x8 @ 32.0 GT/s, but card and slot allow x16 @ 32.0 GT/s (link trained low)"

# 3. No warning when nothing is wrong: a Gen5 card in a Gen4 slot, and a card
#    built for Gen4 (not an Arc Pro B60/B70) in a Gen5 slot.
b70 "$tmp/c" 40 16.0 16 16.0 16 32.0 16
b70 "$tmp/c" 41 32.0 16 16.0 16 16.0 16 0x56a0
expect "$tmp/c" 0 "INFO  0000:40:04.0  x16 @ 16.0 GT/s; the slot limits the link (the card allows x16 @ 32.0 GT/s)
INFO  0000:41:04.0  x16 @ 16.0 GT/s; the card port is limited to 16.0 GT/s (slot allows 32.0 GT/s), normal for a card built for an older PCIe generation"

# 4. A GPU with no switch on the card, and a non-Intel board switch between the
#    root port and the card: the board switch's port is the slot.
rp="$tmp/d/devices/pci0000:50/0000:50:01.0"
dev "$rp" 0x8086 0x352a 0x060400 32.0 16 32.0 16
dev "$rp/0000:51:00.0" 0x8086 0xe223 0x030000 32.0 16 32.0 16
rp="$tmp/d/devices/pci0000:60/0000:60:01.0"
dev "$rp" 0x8086 0x352a 0x060400 32.0 16 32.0 16
dev "$rp/0000:61:00.0" 0x10b5 0x8796 0x060400 32.0 16 32.0 16
dev "$rp/0000:61:00.0/0000:62:08.0" 0x10b5 0x8796 0x060400 16.0 16 16.0 16
card="$rp/0000:61:00.0/0000:62:08.0/0000:63:00.0"
dev "$card" 0x8086 0xe2ff 0x060400 16.0 16 32.0 16
dev "$card/0000:64:01.0" 0x8086 0xe2f0 0x060400 2.5 1 2.5 1
dev "$card/0000:64:01.0/0000:65:00.0" 0x8086 0xe223 0x030000 2.5 1 2.5 1
mkdir -p "$tmp/d/bus"
ln -s "$tmp/d/devices/pci0000:50/0000:50:01.0/0000:51:00.0" "$tmp/d/bus/0000:51:00.0"
ln -s "$card/0000:64:01.0/0000:65:00.0" "$tmp/d/bus/0000:65:00.0"
expect "$tmp/d" 0 "OK    0000:51:00.0  x16 @ 32.0 GT/s
INFO  0000:65:00.0  x16 @ 16.0 GT/s; the slot limits the link (the card allows x16 @ 32.0 GT/s)"

# 5. Skipped: integrated GPU, SR-IOV virtual function, non-Intel GPU, Intel
#    non-display device. An unreadable link is reported, never guessed.
b70 "$tmp/e" 70 32.0 16 32.0 16 32.0 16
printf 'Unknown\n' >"$tmp/e/devices/pci0000:70/0000:70:01.0/0000:70:02.0/current_link_speed"
dev "$tmp/e/devices/pci0000:00/0000:00:02.0" 0x8086 0x7d55 0x030000
ln -s "$tmp/e/devices/pci0000:00/0000:00:02.0" "$tmp/e/bus/0000:00:02.0"
vf="$tmp/e/devices/pci0000:70/0000:70:01.0/0000:70:02.0/0000:70:03.0/0000:70:04.1"
dev "$vf" 0x8086 0xe223 0x038000 2.5 1 2.5 1
ln -s "${vf%.1}.0" "$vf/physfn"
ln -s "$vf" "$tmp/e/bus/0000:70:04.1"
dev "$tmp/e/devices/pci0000:90/0000:90:01.0" 0x8086 0x352a 0x060400 32.0 16 32.0 16
dev "$tmp/e/devices/pci0000:90/0000:90:01.0/0000:91:00.0" 0x10de 0x2684 0x030000 32.0 16 32.0 16
ln -s "$tmp/e/devices/pci0000:90/0000:90:01.0/0000:91:00.0" "$tmp/e/bus/0000:91:00.0"
ln -s "$tmp/e/devices/pci0000:90/0000:90:01.0" "$tmp/e/bus/0000:90:01.0"
expect "$tmp/e" 0 "INFO  0000:70:04.0  PCIe link not readable"

# 6. No discrete Intel GPU at all.
mkdir -p "$tmp/f/bus"
expect "$tmp/f" 2 "No discrete Intel GPU found under $tmp/f/bus"

echo "OK xpu-discover PCIe link checks passed"
