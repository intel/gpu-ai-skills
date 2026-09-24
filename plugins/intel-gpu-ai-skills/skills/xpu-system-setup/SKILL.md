---
name: xpu-system-setup
description: "First-time setup for Intel XPU/GPU hosts. Installs the Intel OMIX (Open Middleware Xe) stack — Level Zero, OpenCL, SYCL compiler, oneMKL/oneDNN — plus clinfo, xpu-smi, user groups (render), and Docker, then runs a post-setup verification gate (including sycl-ls). Prompts before each installation by default (use --auto for unattended). Also handles Battlemage (Arc Pro B60/B70) kernel/runtime prerequisites: nomodeset removal and kernel/runtime upgrade guidance — use check_battlemage_prerequisites.sh when xpu-smi shows No device discovered or clinfo shows 0 platforms. Use when a bare-metal or minimally-configured machine needs to be prepared for XPU model work."
---

# xpu-system-setup

First-time system setup for Intel GPU/XPU workloads on a bare OS install.
Detects what's already configured and only installs what's missing.

This is a **standalone skill** — it has no dependencies on other skills
and can be run independently.

## Source of Truth

Everything in this skill about supported operating systems, supported
GPUs, and Intel OMIX installation, upgrade, verification, and
uninstallation procedures is derived from one live page:

https://dgpu-docs.intel.com/installation-guides/installing-omix.html

Treat every such fact in this file (tables, package names, supported
codenames, the layer-by-layer instructions below) as a **cached
summary, not the authority**. Before advising a user on any of these
topics — what's supported, how to install, how to upgrade, how to
verify, or how to uninstall — fetch that page and answer from its
current content. If it has drifted from what's written here, say so
and follow the live page.

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
3. Runs post-setup configuration verification and captures sensor/log evidence

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
| OMIX repo | Yes | `intel-omix` in apt sources | Fetch GPG key, write `/etc/apt/sources.list.d/intel-gpu-<codename>.list` pointing at `intel-omix` (no version pin — always resolves the latest release for the codename) |
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
~/.out/skills/xpu-system-setup/xpu-smi-health.txt      # evidence, not scored
~/.out/skills/xpu-system-setup/kernel-log.txt           # journalctl -k capture
~/.out/skills/xpu-system-setup/kernel-log.err
~/.out/skills/xpu-system-setup/kernel-log-review.txt    # selected lines to read
```

`status.tsv` columns: `component | before | action | after | result`

## Post-Setup Verification

After setup, the script runs a verification gate:

1. `ls -l /dev/dri` — GPU device files exist with correct permissions
2. `id -nG` — user is in render group (active in current session)
3. `clinfo` — Intel OpenCL devices detected
4. `xpu-smi discovery` — Intel GPUs visible
5. `source /opt/intel/oneapi/setvars.sh && sycl-ls` — SYCL compiler sees
   Intel GPU device(s), not just an Intel CPU backend
6. `docker info` — Docker daemon reachable

The script also saves `xpu-smi health -l` to `xpu-smi-health.txt` and a
GPU-driver kernel-log selection to `kernel-log-review.txt`, in the output
directory above. Neither is scored: `Warning` / `Critical` health statuses need
follow-up and `Unknown` is inconclusive; log matches are lines to read, not
faults. When `journalctl` is missing, the log is unreadable, or it has no
xe/i915 driver lines, the run records a warning and reports
`READY WITH WARNINGS`. See **xpu-discover** for the per-component health
probes. `READY` means configuration checks passed, not that device health or
workload execution is certified.

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
   verification gate and the separate evidence above. If you ran `--dry-run`
   on an already-configured host, say the verification gate would run
   at the end of a real invocation and summarize the detected state.

Do not reduce the answer to a bare "already installed / nothing to
do" table — the mode explanation and the verification-gate result
must appear regardless of host state.

## Keeping this current

Supported Ubuntu codenames and the GPG key URL are **not fetched live at
runtime** — they are constants declared near the top of
`scripts/setup_xpu_system.sh` (`OMIX_CODENAMES`, `OMIX_GPG_KEY_URL`,
`OMIX_RUNTIME_PKG`, `OMIX_DEV_PKG`), each tagged with the date they were
last checked. The repo line itself omits the version segment
(`.../intel-omix unified`, not `.../intel-omix/<series> unified`), so apt
always resolves the latest OMIX release compliant with the detected codename
— there is no version constant to keep in sync. The OMIX doc has changed its
content between revisions before, so **always fetch this page and reconcile
before relying on this skill** (see "Source of Truth" above) — do not wait
for an install failure, a missing package, or the distro-codename warning
to prompt the check:

https://dgpu-docs.intel.com/installation-guides/installing-omix.html

- Supported Ubuntu codenames — the doc's own install snippet embeds them:
  `if [[ ! " <codenames> " =~ " ${VERSION_CODENAME} " ]]` → maps to
  `OMIX_CODENAMES`.
- GPG key URL — `https://repositories.intel.com/gpu/intel-graphics.key`
  (Intel rotates signing keys periodically) → maps to `OMIX_GPG_KEY_URL`.
- Package names — confirm `intel-omix` (runtime) and `intel-omix-dev`
  (dev) are still named this way → maps to `OMIX_RUNTIME_PKG` /
  `OMIX_DEV_PKG`.
- Version pinning — confirm the doc still documents that omitting the
  version segment installs the latest release; if Intel changes that
  default behavior, this script's repo-line construction needs updating too.

If any of these have drifted from what's declared in the script, update
the constants (and the "last verified" date comment next to them)
before running the skill, and mention the drift to the user. The setup
script now stops before installing anything when the detected Ubuntu
codename is not in `OMIX_CODENAMES`, so this reconciliation step has to
happen before a newly-supported release can be installed.

## Supported Hardware

**Intel client discrete GPUs:** Arc, Arc Pro (all generations, including
Battlemage B-series) — per the last-verified OMIX doc content. This list
changes as Intel validates new cards; fetch the live doc (see "Source of
Truth" above) rather than treating this line as exhaustive.

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

**Layer 2 — Upgrade the kernel**

The stock GA kernel on some Ubuntu releases has no PCI alias for
`0xe223`/`0xe211` in the `xe` module — the driver will not bind even
without `nomodeset`.

```sh
modinfo xe | grep -E 'd0000[Ee]2(11|23)'   # empty = kernel too old
# Install the newest HWE or OEM kernel available for your Ubuntu release, then:
sudo reboot
# After reboot: dmesg | grep -i battlemage  →  "Found battlemage (device ID e223)"
```

No specific kernel package or version number is asserted here — Intel's
OMIX install guide does not document one (see "Source of Truth" above).
`check_battlemage_prerequisites.sh` trusts the live `modinfo` alias
check and driver-binding state instead of a hardcoded package name or
version threshold.

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
This is not a slot wiring problem — these cards carry an on-card PCIe switch
(`0xe2ff` on the B70) between the host link and the GPU die.

`xpu-smi listpciinfo` reports what the card **advertises** (generation, max
width), not what it negotiated. Add `sudo` to populate the PCI Slot column;
without it that column reads `N/A`, `DMI reader failed` warnings go to stderr,
and the command still exits 0. For the negotiated host link, walk up from the
GPU's BDF: the endpoint and the switch's downstream port both report the
on-card link.

```sh
d=0000:18:00.0                      # replace with the GPU's discovery BDF
if path=$(readlink -e "/sys/bus/pci/devices/$d"); then
    while [ -f "$path/vendor" ]; do
        width=$(cat "$path/current_link_width") || break
        speed=$(cat "$path/current_link_speed") || break
        max_width=$(cat "$path/max_link_width") || break
        printf '%s  x%s @ %s (max x%s)\n' "${path##*/}" "$width" "$speed" "$max_width"
        path=${path%/*}
    done
else
    printf 'Unknown PCI BDF: %s\n' "$d" >&2
fi
```

The two can diverge: one Arc Pro B60 host advertised Gen 5 x16 while its
bridges negotiated `x8 @ 5.0 GT/s`. Neither figure is a measured transfer rate.

## Supported Distributions

Ubuntu only, codenames per the live OMIX doc content (see "Source of
Truth" above) — currently cached in `OMIX_CODENAMES`
(e.g. `noble` = 24.04, `resolute` = 26.04). Fetch the live doc rather
than treating this cached list as exhaustive or permanent.

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

## Upgrading Intel OMIX

`setup_xpu_system.sh` only performs first-time installs — it does not
upgrade an existing Intel OMIX install. Do not invent `apt-get
install --only-upgrade` commands from memory: fetch the live doc's
Upgrade section (see "Source of Truth" above) and follow it as
written, since the repo line and package names it gives can change
between revisions.

## Uninstalling Intel OMIX

This skill has no uninstall path. If a user asks to remove Intel OMIX,
fetch the live doc's Uninstallation section (see "Source of Truth"
above) rather than guessing `apt remove` targets — it names the exact
runtime/dev packages to remove and the `apt autoremove` follow-up.

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
