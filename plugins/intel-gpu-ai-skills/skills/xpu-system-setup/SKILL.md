---
name: xpu-system-setup
description: "First-time setup for Intel XPU/GPU hosts. Detects what's missing and installs xpu-smi, configures user groups (render), sets up Intel GPU PPA repository, installs Level Zero runtime, installs Docker, and runs a post-setup verification gate. Prompts before each installation by default (use --auto for unattended). Also handles Battlemage (Arc Pro B60/B70) prerequisites on Ubuntu 24.04: nomodeset removal, OEM kernel upgrade, and compute runtime 26.18+ — use check_battlemage_prerequisites.sh when xpu-smi shows No device discovered or clinfo shows 0 platforms. Use when a bare-metal or minimally-configured machine needs to be prepared for XPU model work."
---

# xpu-system-setup

First-time system setup for Intel GPU/XPU workloads on a bare OS install.
Detects what's already configured and only installs what's missing.

This is a **standalone skill** — it has no dependencies on other skills
and can be run independently.

## When To Use

**Run this on a fresh machine to prepare it for Intel GPU/XPU workloads.**

Typical scenarios:
- New bare-metal or VM with Intel GPUs that hasn't been configured
  for compute workloads yet — the standard customer-onboarding entry point.
- System is missing packages, user groups, or Docker needed for GPU work.
- `xpu-smi discovery` shows "No device discovered" or `clinfo` reports 0
  platforms after a fresh install — common on Arc Pro B60/B70 (Battlemage)
  with Ubuntu 24.04 stock kernel. Run `check_battlemage_prerequisites.sh`
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
3. Runs the post-setup verification gate (6 checks)

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

Based on the official Intel GPU driver installation guide
(https://dgpu-docs.intel.com/driver/client/overview.html):

| Component | Check | What the script does if missing |
|-----------|-------|---------------------------------|
| Intel GPU PPA | `ppa:kobuk-team/intel-graphics` in apt sources | `add-apt-repository ppa:kobuk-team/intel-graphics` |
| Compute packages | `libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc` | `apt install` from PPA |
| Media packages | `intel-media-va-driver-non-free libmfx-gen1[.2] libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo` | `apt install` (VAAPI video; skip with `--skip media` on compute-only servers) |
| PyTorch extras | `libze-dev intel-ocloc` | `apt install` (needed for PyTorch/XPU workloads) |
| xpu-smi | `command -v xpu-smi` | `apt install xpu-smi` |
| User groups | Current user in `render` | `gpasswd -a $TARGET_USER render` |
| Docker | `command -v docker` + daemon reachable | Install via `get.docker.com` convenience script |
| Docker group | Current user in `docker` group | `usermod -aG docker $USER` |


## Quick Start

```sh
# Interactive mode (default) — prompts before each installation
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh

# Auto mode — install all without prompts (requires sudo)
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --auto

# Dry-run — show what would be done without changing anything
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --dry-run

# Setup specific components only
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --only xpu-smi,groups

# Skip Docker install (e.g., if using podman)
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --skip docker

# Skip media packages (compute-only server)
plugins/intel-gpu-ai-skills/skills/xpu-system-setup/scripts/setup_xpu_system.sh --skip media
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
5. `xpu-smi diag --precheck` — driver health check
6. `docker info` — Docker daemon reachable

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
   verification gate (the 6-check gate above). If you ran `--dry-run`
   on an already-configured host, say the verification gate would run
   at the end of a real invocation and summarize the detected state.

Do not reduce the answer to a bare "already installed / nothing to
do" table — the mode explanation and the verification-gate result
must appear regardless of host state.

## Supported Hardware

**Intel client discrete GPUs:**

This skill installs drivers from the Intel client GPU PPA
(`ppa:kobuk-team/intel-graphics`) which supports Intel Arc discrete GPUs
(all generations including Arc Pro).

For the complete list of supported hardware, see:
https://dgpu-docs.intel.com/driver/client/overview.html

## Battlemage (Arc Pro B60/B70) Prerequisites on Ubuntu 24.04

On Ubuntu 24.04 with the stock GA kernel (6.8), Battlemage GPUs
(`0xe211` Arc Pro B60, `0xe223` Arc Pro B70) have three silent failure
modes that prevent `xpu-smi`, `clinfo`, and `torch.xpu` from seeing any
devices — even after this skill completes successfully. All three must be
fixed before re-running this skill.

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

The Intel GPU client repo (`repositories.intel.com`) ships runtime 24.39
which does not recognise Battlemage (`device_family: unknown`).
`clinfo` reports 0 platforms. `xpu-system-setup` adds the kobuk-team PPA
which ships runtime 26.18+ — simply running this skill installs the correct
version automatically.

If you installed packages from the Intel client repo *before* running
`xpu-system-setup`, the kobuk-team PPA will upgrade them on the next run.
You can also upgrade explicitly:

```sh
sudo apt-get install -y --only-upgrade \
  libze-intel-gpu1 intel-opencl-icd xpu-smi
clinfo | grep "Number of platforms"   # should be 1
xpu-smi discovery                     # should show Arc Pro B60/B70
```

**Offline environments:** If the PPA is unreachable, download runtime 26.18
directly from the [Intel compute-runtime GitHub releases](https://github.com/intel/compute-runtime/releases/tag/26.18.38308.1)
and the [Intel graphics compiler releases](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.34.4),
then install with `dpkg -i`.

**PCIe topology note:** `lspci` shows x1 downstream ports below the B70.
This is not a slot wiring problem — the B70 has an on-card PCIe switch
(`0xe2ff`) between the host link and the GPU die. On capable platforms
the host-to-GPU link negotiates PCIe 5.0 x16; verify with
`xpu-smi diag -d 0 --singletest 5`.

## Supported Distributions

- **Ubuntu 24.04 LTS (Noble Numbat)**
- **Ubuntu 25.10 (Oracular Oriole)**

Per the official Intel documentation, the `ppa:kobuk-team/intel-graphics` PPA
supports Ubuntu 24.04 and 25.10. Ubuntu 25.10 provides native support for
recent Intel graphics including Battlemage. Ubuntu 24.04 supports newer GPUs
with the HWE (hardware enablement) kernel.

**Note:** Ubuntu 22.04 uses a different installation method (not this PPA).
See https://dgpu-docs.intel.com/driver/client/overview.html for Ubuntu 22.04 instructions.

This skill uses the Intel client GPU PPA (`ppa:kobuk-team/intel-graphics`),
which is Ubuntu-specific. Debian and other distributions require different
installation methods and are not supported by this skill.

For Debian or other distros, install drivers manually following:
https://dgpu-docs.intel.com/driver/client/overview.html

## What This Skill Does (Standalone)

This skill performs one-time system-level setup. It installs packages,
configures user groups, and prepares Docker. After running this skill,
your system will have:

- Intel GPU PPA configured
- GPU compute packages installed (Level Zero, OpenCL)
- xpu-smi tool installed
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
- The Intel GPU PPA (`ppa:kobuk-team/intel-graphics`) is the official
  Intel-maintained PPA per https://dgpu-docs.intel.com/driver/client/overview.html.
- Docker install uses the official convenience script from
  `get.docker.com`. For air-gapped environments, pre-install Docker
  and use `--skip docker`.
