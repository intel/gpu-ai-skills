#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Consistency checks for model-config-recommend/data/hardware.json.
#
# Every published peak must be reproducible as units x ops_per_clk x clock,
# so a new card with a mistyped total or the wrong clock fails here instead
# of silently skewing the roofline.

set -euo pipefail

cd "$(dirname "$0")/.."

python3 - plugins/intel-gpu-ai-skills/skills/model-config-recommend/data/hardware.json <<'EOF'
import json
import re
import sys

hw = json.load(open(sys.argv[1]))
rates = hw["arch_rates"]

TOLERANCE = 0.01

failures = []
seen_ids = {}


def peak(dev, tier):
    rate = rates[dev["arch_id"]][tier]
    return dev[rate["unit"]] * rate["ops_per_clk"] * dev["peak_clock_mhz"] * 1e6 / 1e12


for dev in hw["devices"]:
    did = dev["id"]
    for key in ("arch_id", "xve_engines", "xmx_engines", "graphics_clock_mhz",
                "peak_clock_mhz", "pci_device_id", "pci_ids_name", "published"):
        if key not in dev:
            failures.append(f"{did}: missing {key}")
    if any(f.startswith(did + ":") for f in failures):
        continue
    if dev["arch_id"] not in rates:
        failures.append(f"{did}: arch_id {dev['arch_id']!r} has no arch_rates entry")
        continue

    # The peak clock is Intel's max dynamic clock where one is listed. It is
    # never below the graphics clock; the published-peak check below pins it.
    max_clock = dev.get("max_dynamic_clock_mhz")
    if max_clock and dev["peak_clock_mhz"] != max_clock:
        failures.append(f"{did}: peak_clock_mhz {dev['peak_clock_mhz']} is not "
                        f"the max dynamic clock {max_clock}")
    if dev["peak_clock_mhz"] < dev["graphics_clock_mhz"]:
        failures.append(f"{did}: peak_clock_mhz {dev['peak_clock_mhz']} is below "
                        f"the graphics clock {dev['graphics_clock_mhz']}")

    for field, tier in (("fp32_tflops", "xve_fp32"), ("int8_tops", "xmx_int8")):
        published = dev["published"].get(field)
        if published is None:
            continue
        computed = peak(dev, tier)
        off = abs(computed - published) / published
        if off > TOLERANCE:
            failures.append(f"{did}: {field} computes to {computed:.2f}, "
                            f"Intel publishes {published} ({off:.1%} off)")

    pci = dev["pci_device_id"]
    if not re.fullmatch(r"0x[0-9a-f]{4}", pci):
        failures.append(f"{did}: pci_device_id {pci!r} is not lowercase 0xNNNN")
    if pci in seen_ids:
        failures.append(f"{did}: pci_device_id {pci} already used by {seen_ids[pci]}")
    seen_ids[pci] = did
    if dev["name"] not in dev["pci_ids_name"]:
        failures.append(f"{did}: pci_ids_name {dev['pci_ids_name']!r} does not "
                        f"name {dev['name']!r}")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"OK hardware table: {len(hw['devices'])} devices reconcile with "
      f"units x rate x clock")
EOF
