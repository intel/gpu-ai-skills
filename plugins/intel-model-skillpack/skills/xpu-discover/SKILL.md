---
name: xpu-discover
description: Inventory Intel GPUs (Arc, Arc Pro, Data Center GPU Max) on a Linux host. Detect devices, check driver health, list processes using each XPU, run a quick diagnostic, and read live utilisation.
---

# xpu-discover

`xpu-smi` is Intel's `nvidia-smi`. Sees only Intel GPUs (Arc, Arc
Pro, Battlemage, Flex, Max).

## Quickstart

Run in order. If step 1 is empty, stop.

```sh
xpu-smi discovery                 # 1. inventory
xpu-smi diag --precheck           # 2. driver/firmware health
xpu-smi ps                        # 3. processes using each GPU
xpu-smi diag -d 0 -l 1            # 4. quick functional test
xpu-smi stats -d 0                # 5. utilisation snapshot
xpu-smi dump -d 0 -m 0,5,18 -i 1  # 6. live CSV stream (Ctrl-C)
```

All commands accept `-j` for JSON output (use when parsing).

## CUDA -> Intel cheat sheet

| CUDA | Intel |
|---|---|
| `nvidia-smi` | `xpu-smi discovery` |
| `nvidia-smi -L` | `xpu-smi discovery -j` |
| `nvidia-smi pmon -c 1` | `xpu-smi ps` |
| `nvidia-smi dmon` | `xpu-smi dump -d <id> -m 0,5,18 -i 1` |
| `nvidia-smi --query-gpu=...` | `xpu-smi stats -d <id> -j` |
| `nvidia-smi topo -m` | `xpu-smi topology -m` |
| `CUDA_VISIBLE_DEVICES=0` | `ZE_AFFINITY_MASK=0` |
| `cuda-memcheck` | `xpu-smi diag -d 0 -l 1` |

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
  | `0xe211` | Arc Pro B60 | yes (pci.ids) |
  | `0xe220` | Arc Pro B50 | yes (pci.ids) |
  | `0xe221` | Arc Pro B65 | yes (pci.ids) |
  | `0xe223` | Arc Pro B70 | yes (lspci on hardware) |

  Full table provided above. Cross-check with `lspci -d 8086: -nn` (prints `[8086:XXXX]`). 

Empty output -> kernel didn't enumerate any Intel GPU. See
"Troubleshooting".

### `diag --precheck` — driver health

Scans `journalctl` for known Intel-GPU error categories (GuC/HuC
firmware, IOMMU, PCIe, DRM, i915/Xe, Level Zero init).

- All Pass -> proceed.
- Any Critical -> see "Error pattern routing" below.

Scope with `--since today` / `--since yesterday` /
`--listtypes` (show every error category).

### `ps` — what's using each GPU

Lists processes holding Level Zero handles + shared/device memory
in MiB. Desktop processes (`plasmashell`, `xauth_*`) are normal on
a workstation; only worry about a stale model server still
holding memory.

### `diag -d <id> -l <level>` — does it compute

| Level | Time | Impact |
|---|---|---|
| `-l 1` | seconds | safe on a busy box |
| `-l 2` | medium | impacts other workloads |
| `-l 3` | minutes | impacts performance |

**Reading on a workstation**: the `Software Permission` sub-test
fails when other processes already hold the device, so the
*overall* line says `Fail` even when `Computation Check: Pass`.
Read per-sub-test rows. For a clean run, log out of the GUI
session and run from TTY, or run inside a privileged container.

Pick individual tests with `--singletest`:

| ID | Test | When |
|---|---|---|
| 1 | Computation | does it compute |
| 2 | Memory Error | suspected ECC / bit-flip |
| 3 | Memory Bandwidth | sanity-check HBM/GDDR |
| 4 | Media Codec | video pipelines |
| 5 | PCIe Bandwidth | suspected slot/cable issue |
| 6 | Power | thermal/TDP investigation |
| 7 | Computation functional | quick sanity (lighter than 1) |
| 8 | Media Codec functional | lighter media check |
| 9 | Xe Link Throughput | multi-GPU peer link |
| 10 | Xe Link all-to-all | multi-GPU; needs `-d -1` |

Example: `xpu-smi diag -d 0 --singletest 1,3 -j`.

### `stats -d <id>` — utilisation snapshot

Many fields show `N/A` on consumer Battlemage drivers (Arc Pro
B70 included) — counter-wiring limitation, not a bug. Memory
(column 5) and power (column 18) generally work. For
utilisation while running, prefer `xpu-smi dump`.

### `dump -d <id> -m <metrics> -i <interval>` — live CSV stream

Useful metric IDs for LLM serving:

- `0` GPU utilisation (%)
- `5` GPU memory used (MiB)
- `18` GPU power (W)

```sh
xpu-smi dump -d 0 -m 0,5,18 -i 1 > xpu.csv &
# ... run your model ...
kill %1
```

**Privilege note**: metric `0` reads MEI telemetry, restricted on
consumer parts. Without `sudo`, that column shows `N/A`. Metrics
`5` and `18` work unprivileged.

### `topology -m` — multi-GPU connectivity

Matrix of Xe Link / PCIe switch / hostbridge between XPU pairs.
Only useful on multi-XPU systems.

## Error pattern routing

When `diag --precheck` flags a critical error:

| Category | Cause + fix |
|---|---|
| Level Zero Init Error | Driver/userspace mismatch. Confirm `xe` (Battlemage) or `i915` (older) modules loaded: `lsmod \| grep -E 'i915\|xe'`. Reload or reboot. |
| GuC / HuC Not Running | Missing firmware blob. Check `dmesg \| grep -i 'GuC\|HuC'`; install `linux-firmware`. |
| IOMMU Catastrophic | Kernel cmdline. On consumer boards: `intel_iommu=on iommu=pt`. |
| PCIe Error | Reseat card / check slot; re-run `xpu-smi diag -d <id> --singletest 5`. |
| DRM Error | Stuck context from a crashed desktop session. Logout/login (or reboot) clears it. |
| i915 Not Loaded on Battlemage | Battlemage uses `xe`, not `i915`. Confirm `modinfo xe`; precheck error is misleading on this generation. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `xpu-smi: command not found` | Install Intel level-zero packages (Ubuntu / RHEL / Arch all package `xpu-smi`). Binary lands at `/usr/bin/xpu-smi`. |
| `discovery` empty but card present | (1) Kernel didn't bind: `lspci -k -s <bdf>` should show `Kernel driver in use:`. (2) Bound to vfio: `lsmod \| grep vfio`. (3) Inside container without `/dev/dri`: add `--device /dev/dri --privileged`. |
| `diag` fails inside container | Container needs `--privileged` for diag ioctls beyond the standard render-node interface. |
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
