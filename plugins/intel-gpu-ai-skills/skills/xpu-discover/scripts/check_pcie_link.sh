#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# check_pcie_link.sh
# Read-only: does each Intel GPU card's PCIe link run as fast and as wide as
# the card and its slot allow? Reads sysfs; needs no root; changes nothing.
#
# Usage: bash check_pcie_link.sh
# Prints one line per discrete Intel GPU: OK, WARN or INFO. Exit status 1 when
# any line is WARN, 2 when no discrete Intel GPU is found.
#
# Cards with a PCIe switch (Arc Pro B-series) put the GPU function behind it,
# so the GPU's own link (x1 @ 2.5 GT/s) is internal. The script walks up
# through the card's Intel switch ports to the port that links the card to
# the slot, and reads that link instead.

set -uo pipefail

sysfs=/sys/bus/pci/devices
[ "${1:-}" = --sysfs-root ] && sysfs=$2        # test hook: a fake sysfs tree

speed() { local v; v=$(cat "$1" 2>/dev/null) || return 1; v=${v%% *}     # "32.0 GT/s PCIe" -> 32.0
          case "$v" in ''|*[!0-9.]*) return 1 ;; esac; echo "$v"; }
width() { local v; v=$(cat "$1" 2>/dev/null) || return 1
          case "$v" in ''|0|*[!0-9]*) return 1 ;; esac; echo "$v"; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 < b + 0) }'; }

warned=0 found=0
for dev in "$sysfs"/*; do
    [ "$(cat "$dev/vendor" 2>/dev/null)" = 0x8086 ] || continue
    case "$(cat "$dev/class" 2>/dev/null)" in 0x03*) ;; *) continue ;; esac   # display devices only
    [ -e "$dev/physfn" ] && continue                                          # skip SR-IOV virtual functions
    gpu=$(readlink -e "$dev") || continue
    [ -f "${gpu%/*}/vendor" ] || continue                                     # skip integrated GPUs (no slot)
    found=1
    bdf=${dev##*/}

    # Climb while the parent is an Intel bridge that is not the root port
    # (the root port's own parent has no vendor file).
    port=$gpu
    while [ -f "${port%/*}/vendor" ] && [ -f "${port%/*/*}/vendor" ] \
        && [ "$(cat "${port%/*}/vendor")" = 0x8086 ]; do
        port=${port%/*}
    done
    slot=${port%/*}

    if ! cur=$(speed "$port/current_link_speed") || ! curw=$(width "$port/current_link_width") \
        || ! card=$(speed "$port/max_link_speed") || ! cardw=$(width "$port/max_link_width") \
        || ! slotmax=$(speed "$slot/max_link_speed") || ! slotw=$(width "$slot/max_link_width"); then
        echo "INFO  $bdf  PCIe link not readable"
        continue
    fi

    best=$card;   lt "$slotmax" "$best" && best=$slotmax     # fastest both ends allow
    bestw=$cardw; lt "$slotw" "$bestw" && bestw=$slotw       # widest both ends allow
    link="x$curw @ $cur GT/s"

    if lt "$cur" "$best" || lt "$curw" "$bestw"; then
        echo "WARN  $bdf  $link, but card and slot allow x$bestw @ $best GT/s (link trained low)"
        warned=1
    elif lt "$card" "$slotmax"; then
        case "$(cat "$gpu/device" 2>/dev/null)" in
            0xe211|0xe212|0xe222|0xe223)                       # Arc Pro B60 / B50 / B65 / B70: PCIe 5.0 cards
                echo "WARN  $bdf  $link; the slot allows $slotmax GT/s but the card port is limited to $card GT/s"
                warned=1 ;;
            *)
                echo "INFO  $bdf  $link; the card port is limited to $card GT/s (slot allows $slotmax GT/s), normal for a card built for an older PCIe generation" ;;
        esac
    elif lt "$slotmax" "$card" || lt "$slotw" "$cardw"; then
        echo "INFO  $bdf  $link; the slot limits the link (the card allows x$cardw @ $card GT/s)"
    else
        echo "OK    $bdf  $link"
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No discrete Intel GPU found under $sysfs" >&2
    exit 2
fi
exit "$warned"
