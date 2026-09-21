---
name: xpu-system-setup
description: "First-time setup for Intel XPU/GPU hosts. Installs the Intel OMIX (Open Middleware Xe) stack — Level Zero, OpenCL, SYCL compiler, oneMKL/oneDNN — plus clinfo, xpu-smi, user groups (render), and Docker, then runs a post-setup verification gate (including sycl-ls). Prompts before each installation by default (use --auto for unattended). Also handles Battlemage (Arc Pro B60/B70) prerequisites on Ubuntu: nomodeset removal, OEM kernel upgrade, and compute runtime 26.18+ — use check_battlemage_prerequisites.sh when xpu-smi shows No device discovered or clinfo shows 0 platforms. Use when a bare-metal or minimally-configured machine needs to be prepared for XPU model work."
---

# xpu-system-setup

First-time system setup for Intel GPU/XPU workloads on a bare OS install.
Detects what's already configured and only installs what's missing.

This is a **standalone skill** — it has no dependencies on other skills
and can be run independently.

## Install path: Intel OMIX

This skill installs **Intel OMIX (Open Middleware Xe)** — a single
pinned bundle covering Level Zero, OpenCL, the SYCL/DPC++ compiler, and
oneMKL/oneDNN — as its only install mechanism.

**Do not also add the legacy per-package PPA
(`ppa:kobuk-team/intel-graphics`) on a host this skill has set up.**
Intel's OMIX docs explicitly call for "a clean system without
preinstalled Intel GPU user-mode packages from the PPA" — mixing the
two causes apt dependency conflicts (an exact-pinned OMIX dependency
like `libze1` fighting a newer PPA-provided version of the same
package). If a host already has PPA packages installed, remove them
and the PPA repo first (see the PPA doc's Uninstallation section:
https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html#uninstallation),
*then* run this skill.

The `intel-omix` runtime depends on `intel-gpu-compute`, which already
includes `libze-dev`, `intel-ocloc`, Level Zero, OpenCL, and `xpu-smi`.
Use `--include-dev` when full development headers and libraries are
needed; it installs `intel-omix-dev` and its `intel-gpu-compute-dev`
dependency. Media/VAAPI provisioning is outside this compute-focused
skill.

## When To Use

**Run this on a fresh machine to prepare it for Intel GPU/XPU workloads.**

Typical scenarios:
- New bare-metal or VM with Intel GPUs that hasn't been configured
  for compute workloads yet — the standard customer-onboarding entry point.
- System is missing packages, user groups, or Docker needed for GPU work.
- `xpu-smi discovery` shows "No device discovered" or `clinfo` reports 0
  platforms after a fresh install — common on Arc Pro B60/B70 (Battlemage)
  with Ubuntu's stock kernel. Run `check_battlemage_prerequisites.sh`
  to diagnose and fix the underlying kernel/runtime issues, then re-run
  this skill.

The script is idempotent: it detects what's already installed and
only acts on what's missing, so re-running it on a configured host
is safe and finishes quickly.

## How to Invoke

**Always invoke this skill by running the script.** Do not run `apt install`,
`add-apt-repository`, or `usermod` directly — even if the dry-run reports
exactly which package is missing. Confirm with the user before running —
the script installs system packages and modifies group membership.

The script:
1. Detects what's installed (idempotent — safe to re-run)
2. Installs only what's missing (with prompts unless `--auto`)
3. Runs the post-setup verification gate (7 checks)

The raw commands shown in the table below are what the script runs
internally — they are descriptive, not a manual checklist. Step 3 only
runs when you go through the script. Manual installs leave the
verification step skipped, which can hide problems (e.g., a package
installed but the driver not loadable, or render group not effective).

If only one component is missing, use `--only`:

```sh
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --only xpu-smi --auto
```

This still runs the full verification gate at the end.

## What It Covers

Based on https://dgpu-docs.intel.com/installation-guides/installing-omix.html:

| Component | Installed by default? | Check | What the script does if missing |
|-----------|:---:|-------|---------------------------------|
| OMIX repo | Yes | `intel-omix` in apt sources | Fetch GPG key, write `/etc/apt/sources.list.d/intel-gpu-<codename>.list` pointing at `intel-omix/<series>` |
| OMIX runtime | Yes | `dpkg -l intel-omix` | `apt install intel-omix` (Level Zero, OpenCL, SYCL compiler, oneMKL/oneDNN) |
| clinfo | Yes | `command -v clinfo` | `apt install clinfo` (not bundled by OMIX) |
| xpu-smi | Yes | `command -v xpu-smi` | Verify/install `xpu-smi` from the OMIX repo if the meta-package installation is incomplete |
| User groups | Yes | Current user in `render` | `gpasswd -a $TARGET_USER render` |
| Docker | Yes | `command -v docker` + daemon reachable | Install via `get.docker.com` convenience script |
| Docker group | Yes | Current user in `docker` group | `usermod -aG docker $USER` |
| OMIX dev | Opt-in (`--include-dev` or `--only omix-dev`) | `dpkg -l intel-omix-dev` | `apt install intel-omix-dev` (SYCL/oneMKL/oneDNN build headers) |

## Quick Start

```sh
# Interactive mode (default) — prompts before each step
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh

# Auto mode — install all without prompts (requires sudo)
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --auto

# Also install the OMIX dev package (SYCL/oneMKL/oneDNN build headers)
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --auto --include-dev

# Dry-run — show what would be done without changing anything
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --dry-run

# Setup specific components only
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --only xpu-smi,groups

# Skip Docker install (e.g., if using podman)
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --skip docker
```

**Interactive vs Auto Mode:**
- **Default (no flags):** Interactive — prompts "Install X? [y/N]" before each component
- **`--auto` or `--yes`:** Unattended — installs all missing components without prompts

## Script Output

```text
~/.out/skills/xpu-system-setup/SUMMARY.md
~/.out/skills/xpu-system-setup/setup.log
~/.out/skills/xpu-system-setup/status.tsv
```

`status.tsv` columns: `component | before | action | after | result`

## Post-Setup Verification

After setup, the script runs a verification gate:

1. `ls -l /dev/dri` — GPU device files exist with correct permissions
2. `id -nG` — user is in render group (active in current session)
3. `clinfo` — Intel OpenCL devices detected
4. `xpu-smi discovery` — Intel GPUs visible
5. `xpu-smi diag --precheck` (or `xpu-smi health -l` on newer xpu-smi
   releases that dropped `diag`) — driver health check
6. `source /opt/intel/oneapi/setvars.sh && sycl-ls` — SYCL compiler sees
   Intel device(s)
7. `docker info` — Docker daemon reachable

If verification requires a re-login (group changes), the script
reports `READY AFTER RELOGIN` and prints the command to verify
after re-login.

## Reporting back to the user

When you report the outcome to the user, always state two things
explicitly, **even when the host is already fully configured and
nothing was installed**:

1. **Mode** — the script prompts before each install by default
   (interactive); pass `--auto` (or its alias `--yes`) for unattended
   runs, and `--dry-run` to preview without changing anything. Name
   these so the user knows how to drive a real install.
2. **Verification gate** — report the result of the post-setup
   verification gate (the 7-check gate above). If you ran `--dry-run`
   on an already-configured host, say the verification gate would run
   at the end of a real invocation and summarize the detected state.

Do not reduce the answer to a bare "already installed / nothing to
do" table — the mode explanation and the verification-gate result
must appear regardless of host state.

## Keeping this current

Supported Ubuntu codenames, the repo/version string, and the GPG key URL
are **not fetched live at
runtime** — they are constants declared near the top of
`scripts/setup_xpu_system.sh` (`OMIX_CODENAMES`, `OMIX_VERSION_SERIES`,
`OMIX_GPG_KEY_URL`, `OMIX_RUNTIME_PKG`, `OMIX_DEV_PKG`), each tagged with the
date they were last checked. The OMIX doc has changed its content
between revisions before, so **before relying on this skill, or
whenever an install fails, a package isn't found, or the
distro-codename warning fires, fetch this page and reconcile:**

https://dgpu-docs.intel.com/installation-guides/installing-omix.html

- Supported Ubuntu codenames — the doc's own install snippet embeds them:
  `if [[ ! " <codenames> " =~ " ${VERSION_CODENAME} " ]]` → maps to
  `OMIX_CODENAMES`.
- Repo version series — from the repo line
  `.../intel-omix/<series> unified` → maps to `OMIX_VERSION_SERIES`.
- GPG key URL — `https://repositories.intel.com/gpu/intel-graphics.key`
  (Intel rotates signing keys periodically) → maps to `OMIX_GPG_KEY_URL`.
- Package names — confirm `intel-omix` (runtime) and `intel-omix-dev`
  (dev) are still named this way → maps to `OMIX_RUNTIME_PKG` /
  `OMIX_DEV_PKG`.

If any of these have drifted from what's declared in the script, update
the constants (and the "last verified" date comment next to them)
before running the skill, and mention the drift to the user.

## Supported Hardware

**Intel client discrete GPUs:** Arc, Arc Pro (all generations, including
Battlemage B-series). For the complete list, see the hardware table
linked from the OMIX doc above.

## Battlemage (Arc Pro B60/B70) Prerequisites

On Ubuntu with the stock GA kernel, Battlemage GPUs (`0xe211` Arc Pro
B60, `0xe223` Arc Pro B70) have three silent failure modes that prevent
`xpu-smi`, `clinfo`, and `torch.xpu` from seeing any devices — even
after this skill completes successfully. All three must be fixed before
re-running this skill.

**Quick diagnosis:**

```sh
bash scripts/check_battlemage_prerequisites.sh
# --fix      apply remediations interactively (requires sudo)
# --dry-run  preview changes without applying
```

**The three layers:**

**Layer 1 — Remove `nomodeset`**

`nomodeset` prevents the `xe` driver from binding. Set by some installers
or cloud images as a framebuffer fallback.

```sh
grep nomodeset /proc/cmdline          # present = broken
sudo sed -i 's/\bnomodeset\b//g' /etc/default/grub
sudo update-grub && sudo reboot
```

**Layer 2 — Upgrade to OEM kernel 6.17**

The Ubuntu 24.04 GA kernel (6.8) has no PCI alias for `0xe223`/`0xe211`
in the `xe` module — the driver will not bind even without `nomodeset`.

```sh
modinfo xe | grep -E 'd0000[Ee]2(11|23)'   # empty = kernel too old
sudo apt install -y linux-oem-24.04
sudo reboot
# After reboot: dmesg | grep -i battlemage  →  "Found battlemage (device ID e223)"
```

Ubuntu HWE kernel 6.11+ also works; OEM 6.17 is preferred for Arc Pro
because it ships matching GuC/HuC firmware blobs.

**Layer 3 — Run `xpu-system-setup`**

Older Intel GPU repos can ship a compute runtime that predates
Battlemage support (`device_family: unknown`, `clinfo` reports 0
platforms). Running this skill installs a current-enough runtime via
OMIX automatically — verify with:

```sh
clinfo | grep "Number of platforms"   # should be 1
xpu-smi discovery                     # should show Arc Pro B60/B70
```

If a host has packages from an older/different Intel GPU repo (or the
legacy PPA) installed *before* running `xpu-system-setup`, remove them
first — see "Install path: Intel OMIX" above for why mixing repos
causes dependency conflicts.

**Offline environments:** If the OMIX repo is unreachable, download the
compute runtime directly from the
[Intel compute-runtime GitHub releases](https://github.com/intel/compute-runtime/releases)
and the
[Intel graphics compiler releases](https://github.com/intel/intel-graphics-compiler/releases),
then install with `dpkg -i`.

**PCIe topology note:** `lspci` shows x1 downstream ports below the B70.
This is not a slot wiring problem — the B70 has an on-card PCIe switch
(`0xe2ff`) between the host link and the GPU die. On capable platforms
the host-to-GPU link negotiates PCIe 5.0 x16; verify with
`xpu-smi diag -d 0 --singletest 5` (older xpu-smi) or
`xpu-smi listpciinfo` (xpu-smi 2.x, which dropped `diag`'s `--singletest`).

## Supported Distributions

Ubuntu only, codenames per the live OMIX doc content (see "Keeping this
current" above) — currently declared in `OMIX_CODENAMES`
(e.g. `noble` = 24.04, `resolute` = 26.04).

Ubuntu 22.04 requires a different installation method not covered by
this skill; Debian and other distributions require different
installation methods entirely — see the OMIX doc's introduction for
pointers.

## What This Skill Does (Standalone)

This skill performs one-time system-level setup. It installs packages,
configures user groups, and prepares Docker. After running this skill,
your system will have:

- Intel OMIX repo configured
- `intel-omix` installed (Level Zero, OpenCL, SYCL compiler, oneMKL/oneDNN)
- `intel-omix-dev` installed, if `--include-dev` was passed
- `clinfo` and `xpu-smi` installed
- User added to `render` group for GPU access
- Docker installed and configured (optional)

This is system provisioning — you run it once on a fresh machine.
For runtime checks, GPU discovery, or running workloads, those are
handled by other tools or skills, but this skill has no dependencies
on them.

## Important Notes

- Requires `sudo` access for package installation and group changes.
- Group changes (render, docker) take effect on next login session.
  To activate them immediately without logging out, run `newgrp render`.
  This gives you a subshell with the render group active; `exit` returns
  to the original shell. For permanent activation across all future
  sessions, log out and log back in.
- Do not add the legacy PPA (`ppa:kobuk-team/intel-graphics`) alongside
  OMIX on the same host — see "Install path: Intel OMIX" above.
- Docker install uses the official convenience script from
  `get.docker.com`. For air-gapped environments, pre-install Docker
  and use `--skip docker`.


