---
name: xpu-discover
description: Inventory Intel GPUs (Arc, Arc Pro) on a Linux host. Detect devices, collect driver, firmware and component-health evidence, list processes using each XPU, and read live utilisation.
---

# xpu-discover

`xpu-smi` is Intel's `nvidia-smi`. Sees only Intel GPUs (Arc, Arc
Pro, Battlemage, Flex, Max).

## Quickstart

Subcommands and flags differ between builds; `xpu-smi -h` lists what the
installed one supports (prefer `-h`: some builds reject `help`). Report an
absent subcommand rather than retrying it.

Run in order. If step 1 fails or has no devices, stop.

```sh
xpu-smi discovery                 # 1. inventory
xpu-smi health -l                 # 2. component health (power, frequency, temp)
journalctl -k --no-pager | grep -iE 'guc|huc|iommu|drm|\bxe\b|i915|level.?zero'
                                  # 3. driver/firmware log evidence
xpu-smi ps                        # 4. processes using each GPU
xpu-smi stats -d 0                # 5. utilisation snapshot
xpu-smi dump --device 0 --metrics MEMORY,POWER --interval 1
                                  # 6. live CSV stream (Ctrl-C)
```

The `xpu-smi` commands accept `-j` for JSON output (use when parsing).

Steps 2 and 3 are both needed: `health` covers power and thermals, the kernel
log covers driver and firmware. Neither certifies health, and a command that
runs is not a GPU that works.

## CUDA -> Intel cheat sheet

| CUDA | Intel |
|---|---|
| `nvidia-smi` | `xpu-smi discovery` |
| `nvidia-smi -L` | `xpu-smi discovery -j` |
| `nvidia-smi pmon -c 1` | `xpu-smi ps` |
| `nvidia-smi dmon` | `xpu-smi dump --device <id> --metrics MEMORY,POWER --interval 1` |
| `nvidia-smi --query-gpu=...` | `xpu-smi stats -d <id> -j` |
| `nvidia-smi topo -m` | `xpu-smi topology -m` |
| `CUDA_VISIBLE_DEVICES=0` | `ZE_AFFINITY_MASK=0` |
| `cuda-memcheck` | no Arc equivalent: run a real workload (**torch-xpu-run**) |

**CUDA refugee footgun**: `CUDA_VISIBLE_DEVICES=99` silently hides
all GPUs; `ZE_AFFINITY_MASK=99` **crashes** the Level Zero loader
with an assertion. Always check `xpu-smi discovery` for valid IDs
(start at 0) before setting the mask.

## What each subcommand returns

### `discovery` — inventory

One stanza per Intel GPU. Key fields:

- **Device ID** — small integer, used as `-d` and as
  `ZE_AFFINITY_MASK` value.
- **PCI BDF Address** — stable across reboots (e.g. `0000:36:00.0`).
- **DRM Device** — `/dev/dri/card0`, used in `--device` for Docker.
- **Device Name** — Battlemage shows `Intel(R) Graphics [0xe2XX]`
  rather than the marketing name; driver quirk, not a problem.
  Map the PCI device ID in brackets to the product SKU:

  | PCI device ID | Product SKU | Confirmed |
  |---|---|---|
  | `0xe20b` | Arc B580 | yes (lspci on hardware) |
  | `0xe211` | Arc Pro B60 | yes (lspci on hardware) |
  | `0xe220` | Arc Pro B50 | yes (pci.ids) |
  | `0xe221` | Arc Pro B65 | yes (pci.ids) |
  | `0xe223` | Arc Pro B70 | yes (lspci on hardware) |

  Full table provided above. Cross-check with `lspci -d 8086: -nn` (prints `[8086:XXXX]`). 

Empty output -> kernel didn't enumerate any Intel GPU. See
"Troubleshooting".

### `health -l` — component health

Per-device status for `power_health`, `frequency_health`, `core_temperature`,
`memory_temperature`, `memory_health`. Read the statuses, not the exit code:
exit status and sensor support vary independently by build.

- `OK` -> no problem reported in this sample; not a workload pass.
- `Unknown`, empty, or unsupported -> **inconclusive, not healthy**. Usual on
  Arc Pro B60/B70, where only power and frequency report `OK`.
- `Warning` / `Critical` -> report the component and its value. The routing
  table below covers kernel-log categories, not sensor components.

If the aggregate call fails (unsupported temperature sensors can abort it),
query components one at a time:

```sh
xpu-smi health -d 0 -c 3           # power
xpu-smi health -d 0 -c 4           # memory
xpu-smi health -d 0 -c 6           # frequency
```

### Kernel-log scan — driver/firmware errors

Step 3 lists the GPU driver lines. To select messages for review, append
`| grep -iE '\b(error|fail|warn|timed? ?out|reset|hang|wedged|fault)'`
(the leading word boundary stops `fault` matching `Default`; no trailing
boundary, so `errors`, `Resetting` and `Timedout` match). This is triage, not
a fault detector: it selects benign resets and misses faults worded without
these keywords. No matches means only that nothing matched.

Read `journalctl`'s stderr, and run it unpiped to see its exit status (step 3's
pipeline reports `grep`'s): a failed read or no GPU driver lines means the log
is unavailable, not that the host is clean. Prefer
`journalctl -k` over `dmesg`, which can return nothing unprivileged under
`kernel.dmesg_restrict=1`. `-k` is the current boot; `--since` narrows it
and can hide earlier errors.

Correctable PCIe/AER events are usually benign; check
`/sys/bus/pci/devices/<bdf>/aer_dev_correctable` before treating them as a
fault.

### `ps` — what's using each GPU

Lists processes holding Level Zero handles + shared/device memory
in MiB. Desktop processes (`plasmashell`, `xauth_*`) are normal on
a workstation; only worry about a stale model server still
holding memory.

### `stats -d <id>` — utilisation snapshot

Many fields show `N/A` on consumer Battlemage drivers (Arc Pro
B70 included) — counter-wiring limitation, not a bug. Memory and
power figures generally populate; GPU-utilisation ones do not. For
utilisation while running, prefer `xpu-smi dump`. For a bounded probe
use `xpu-smi stats -d <id> --samples 1` (`-n` is not a stats flag).

### `dump` — live CSV stream

On Arc builds use long flags (`--device`, `--metrics`, `--interval`,
`--number`); `-m` / `-i` / `-n` are rejected with `The following arguments
were not expected`. `-d <id>` is still accepted (xpu-smi 2.2.0), so
`xpu-smi dump -d 0 --metrics MEMORY --interval 1` also works. Select metrics by group name: `MEMORY`, `UTILIZATION`,
`POWER`, `TEMPERATURE`, `CLOCK`, `PCI`, `ECC`, `EU_ARRAY`, `FAN`, `ALL`.

```sh
xpu-smi dump --device 0 --metrics MEMORY,POWER --interval 1 > xpu.csv &
# ... run your model ...
kill %1
```

For an exact column set:
`xpu-smi --query-gpu=memory.used,power.draw --id=0 --format=csv`.
It is single-shot. For repeated sampling use `dump --interval`, not
`--query-gpu --loop`: on a non-TTY it writes `tcsetattr` warnings to stderr and
does not exit when `--count` is reached.

The Data Center GPU build (Intel XPU Manager) takes short flags and numeric
IDs instead, e.g. `xpu-smi dump -d 0 -m 18,1 -i 1 -n 5`; `xpu-smi dump --help`
prints each field with its numeric ID.

**Privilege note**: unprivileged, `memory.used` / `total` / `free`,
`utilization.memory`, `power.draw` and `power.limit` populate. The other
`utilization.*` fields, `eu.*`, the `memory.*bandwidth*` fields,
`power.draw.gpu`, `power.max_limit` and `energy.consumed` read `N/A`, which is
missing telemetry, not zero.

### `topology -m` — multi-GPU connectivity

Matrix of Xe Link / PCIe switch / hostbridge between XPU pairs.
Only useful on multi-XPU systems.

## Error pattern routing

When the kernel-log scan flags one of these categories:

| Category | Cause + fix |
|---|---|
| Level Zero Init Error | Driver/userspace mismatch. Confirm `xe` (Battlemage) or `i915` (older) modules loaded: `lsmod \| grep -E 'i915\|xe'`. Reload or reboot. |
| GuC / HuC Not Running | Missing firmware blob. Check `journalctl -k \| grep -i 'GuC\|HuC'`; install `linux-firmware`. |
| IOMMU Catastrophic | Kernel cmdline. On consumer boards: `intel_iommu=on iommu=pt`. |
| PCIe Error | Check `/sys/bus/pci/devices/<bdf>/aer_dev_correctable` and `lspci -vv` `LnkSta` vs `LnkCap`; correctable counts are usually benign. Reseat card / check slot if uncorrectable. |
| DRM Error | Stuck context from a crashed desktop session. Logout/login (or reboot) clears it. |
| i915 Not Loaded on Battlemage | Battlemage uses `xe`, not `i915`. Confirm `modinfo xe`; an `i915`-missing message is misleading on this generation. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `xpu-smi: command not found` | Install Intel level-zero packages (Ubuntu / RHEL / Arch all package `xpu-smi`). Binary lands at `/usr/bin/xpu-smi`. |
| `discovery` empty but card present | (1) Kernel didn't bind: `lspci -k -s <bdf>` should show `Kernel driver in use:`. (2) Bound to vfio: `lsmod \| grep vfio`. (3) Inside container without `/dev/dri`: add `--device /dev/dri --privileged`. |
| `'diag' is not a valid subcommand` | Expected on Arc/Battlemage. Use `xpu-smi health -l` plus the kernel-log scan (steps 2 and 3). |
| Two XPUs present, one visible | `printenv ZE_AFFINITY_MASK`; unset for full inventory, re-export for workloads. |

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) a process sees (`0`, `0,1`, ...). Invalid IDs crash the L0 loader (unlike CUDA's silent-hide). |
| `ZE_FLAT_DEVICE_HIERARCHY` | `FLAT` exposes tiles as separate root devices; `COMPOSITE` (default) groups under one. Battlemage is single-tile, doesn't matter. |
| `ZE_ENABLE_VALIDATION_LAYER=1` | L0 loader prints API misuse — useful when something silently returns wrong device count. |

## References

- `xpu-smi` source: <https://github.com/intel/xpumanager>
- Level Zero spec: <https://oneapi-src.github.io/level-zero-spec/>
- Battlemage / Xe driver: <https://docs.kernel.org/gpu/xe/index.html>
