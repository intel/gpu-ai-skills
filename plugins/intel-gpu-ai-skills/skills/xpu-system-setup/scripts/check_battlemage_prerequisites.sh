#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# check_battlemage_prerequisites.sh
# Read-only diagnostic for Intel Arc Pro B60/B70 (Battlemage) hosts.
# Checks the three prerequisite layers and reports what needs fixing
# before xpu-system-setup can succeed.
#
# Usage:
#   bash check_battlemage_prerequisites.sh            # check + print remediation
#   bash check_battlemage_prerequisites.sh --fix      # apply fixes (requires sudo)
#   bash check_battlemage_prerequisites.sh --dry-run  # show what --fix would do

set -uo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' BOLD='\033[1m' NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

PASS="${GREEN}[PASS]${NC}"
FAIL="${RED}[FAIL]${NC}"
WARN="${YELLOW}[WARN]${NC}"
INFO="${BOLD}[INFO]${NC}"

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE=check

usage() {
    cat <<'EOF'
Usage: check_battlemage_prerequisites.sh [--fix | --dry-run]

  (no flags)   Read-only diagnostic. Reports what is broken and how to fix it.
  --fix        Apply fixes interactively (requires sudo). Confirms before each
               destructive step (GRUB edit, kernel install, reboot).
  --dry-run    Show what --fix would do without changing anything.
  -h, --help   Show this help and exit 0.

In check mode the script is completely read-only.
--fix aborts if run non-interactively (stdin not a terminal).

Target hardware: Intel Arc Pro B60 (0xe211) / B70 (0xe223).
Kernel and runtime remediation below is generic (no specific kernel
package or runtime version number is asserted) -- Intel's OMIX install
guide does not document either, so this script trusts live signals
(modinfo, clinfo, xpu-smi) instead of a hardcoded version threshold.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --fix)        MODE=fix;     shift ;;
        --dry-run)    MODE=dryrun;  shift ;;
        -h|--help)    usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$MODE" = fix ]; then
    if [ "$(id -u)" -ne 0 ]; then
        printf '%b\n' "${RED}ERROR:${NC} --fix requires root. Run with sudo." >&2
        exit 1
    fi
    # Refuse to run --fix non-interactively: destructive commands need confirmation
    if [ ! -t 0 ]; then
        printf '%b\n' "${RED}ERROR:${NC} --fix requires an interactive terminal (stdin is not a tty)." >&2
        exit 1
    fi
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
ISSUES=0

# Track which fix categories are needed (deduplicated by layer)
NEED_GRUB=false
NEED_KERNEL=false
NEED_RUNTIME=false

fail() {
    printf '%b %s\n' "$FAIL" "$1"
    ISSUES=$((ISSUES + 1))
}
pass()     { printf '%b %s\n' "$PASS" "$1"; }
warn()     { printf '%b %s\n' "$WARN" "$1"; }
info()     { printf '%b %s\n' "$INFO" "$1"; }

confirm() {
    printf '%b %s [y/N] ' "${YELLOW}?${NC}" "$1"
    read -r answer
    case "$answer" in [yY]*) return 0 ;; *) return 1 ;; esac
}

dryrun_note() { printf '%b  DRY-RUN: would run: %s\n' "${YELLOW}»${NC}" "$*"; }

# Run a fix command; abort if it fails rather than silently continuing
run_fix() {
    local desc="$1"; shift
    printf '%b Running: %s\n' "${YELLOW}»${NC}" "$*"
    if ! "$@"; then
        printf '%b %s failed — aborting --fix to avoid leaving system in bad state.\n' \
            "${RED}ERROR:${NC}" "$desc" >&2
        exit 1
    fi
}

# ── Distro detection (informational only) ───────────────────────────────────
DISTRO_ID="unknown"
DISTRO_VERSION_ID="unknown"
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO_ID="${ID:-unknown}"
    DISTRO_VERSION_ID="${VERSION_ID:-unknown}"
fi

# ── Banner ────────────────────────────────────────────────────────────────────
printf '\n%b\n' "${BOLD}═══ Battlemage (Arc Pro B60/B70) Prerequisites Check ═══${NC}"
printf 'Mode: %s | Distro: %s %s\n\n' "$MODE" "$DISTRO_ID" "$DISTRO_VERSION_ID"

# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 — Hardware presence
# ─────────────────────────────────────────────────────────────────────────────
printf '%b\n' "${BOLD}── Layer 0: Hardware Detection ──${NC}"

if ! command -v lspci &>/dev/null; then
    fail "lspci not found — install pciutils first: sudo apt install pciutils"
    exit 1
fi

# Match [8086:e211] or [8086:e223] exactly — avoids false-positives on other IDs
# shellcheck disable=SC2046
BMG_BDFS=$(lspci -D -d 8086: -nn 2>/dev/null \
    | grep -E '\[8086:e2(11|23)\]' \
    | awk '{print $1}' || true)

if [ -z "$BMG_BDFS" ]; then
    fail "No Battlemage GPU found ([8086:e211] / [8086:e223]). Is the card seated?"
    printf '  Intel devices visible: %s\n' "$(lspci -d 8086: -nn 2>/dev/null | head -5)"
    printf '\n%bNothing to fix — physical hardware issue.%b\n' "$RED" "$NC"
    exit 1
fi

GPU_COUNT=$(printf '%s\n' "$BMG_BDFS" | wc -l | tr -d ' ')
pass "Found $GPU_COUNT Battlemage GPU(s): $(printf '%s\n' "$BMG_BDFS" | tr '\n' ' ')"

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — nomodeset in GRUB
# ─────────────────────────────────────────────────────────────────────────────
printf '\n%b\n' "${BOLD}── Layer 1: GRUB nomodeset ──${NC}"

LAYER1_OK=true
if grep -q 'nomodeset' /proc/cmdline 2>/dev/null; then
    fail "nomodeset is active in the current kernel cmdline — xe driver will not bind"
    printf '  Current cmdline: %s\n' "$(cat /proc/cmdline)"
    LAYER1_OK=false
    NEED_GRUB=true
else
    pass "nomodeset not present in kernel cmdline"
fi

if grep -q 'nomodeset' /etc/default/grub 2>/dev/null; then
    fail "nomodeset found in /etc/default/grub — it may reappear after the next grub update"
    LAYER1_OK=false
    NEED_GRUB=true
fi

$LAYER1_OK && pass "GRUB clean"

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Kernel version and xe module PCI alias
# ─────────────────────────────────────────────────────────────────────────────
printf '\n%b\n' "${BOLD}── Layer 2: Kernel / xe Driver ──${NC}"

KERNEL=$(uname -r)
info "Running kernel: $KERNEL"

LAYER2_OK=true

# Ground truth first: check driver binding for each GPU.
# A GPU bound to xe means the driver is working regardless of lsmod output
# (xe may be built into the kernel and not appear in lsmod).
BOUND_COUNT=0
# shellcheck disable=SC2086
for BDF in $BMG_BDFS; do
    DRIVER=$(readlink "/sys/bus/pci/devices/$BDF/driver" 2>/dev/null \
        | xargs basename 2>/dev/null || echo "none")
    case "$DRIVER" in
        xe)
            pass "GPU $BDF bound to xe driver"
            BOUND_COUNT=$((BOUND_COUNT + 1))
            ;;
        vfio*) warn "GPU $BDF bound to $DRIVER (SR-IOV/passthrough mode)" ;;
        none)
            fail "GPU $BDF has no driver bound"
            NEED_KERNEL=true
            LAYER2_OK=false
            ;;
        *)     warn "GPU $BDF bound to unexpected driver: $DRIVER" ;;
    esac
done

# xe in lsmod is informational — may be absent when built into the kernel
if lsmod | grep -q '^xe'; then
    pass "xe kernel module loaded (as module)"
elif [ "$BOUND_COUNT" -gt 0 ]; then
    info "xe not in lsmod — likely built-in to kernel $KERNEL (driver is working)"
else
    warn "xe not in lsmod and no GPUs bound — kernel may not support Battlemage"
    info "A newer kernel is likely needed — no specific package/version is asserted here; see Layer 2 remediation below"
    NEED_KERNEL=true
fi

# Check modinfo alias as a recommendation, not a hard failure.
# If GPUs are already bound, a missing alias is informational only.
XE_HAS_BMG=$(modinfo xe 2>/dev/null | grep -cE 'd0000[Ee]2(11|23)' || true)
if [ "$XE_HAS_BMG" -eq 0 ]; then
    if [ "$BOUND_COUNT" -gt 0 ]; then
        # Built-in or alias not needed — driver is working
        info "xe module has no Battlemage alias in modinfo (built-in path — OK since GPUs are bound)"
    else
        warn "xe module on kernel $KERNEL has no Battlemage alias — kernel upgrade recommended"
        info "No specific kernel package/version is asserted here; see Layer 2 remediation below"
        NEED_KERNEL=true
    fi
else
    pass "xe module has Battlemage (0xe223/0xe211) PCI aliases"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Compute runtime (OpenCL / Level Zero)
# ─────────────────────────────────────────────────────────────────────────────
printf '\n%b\n' "${BOLD}── Layer 3: Compute Runtime ──${NC}"

LAYER3_OK=true
OPENCL_PLATFORMS=0

# Check the Intel OMIX repo is configured (needed for --fix runtime upgrade).
# Also recognizes the legacy kobuk-team PPA in case a host still has it from
# before xpu-system-setup switched to OMIX-only. Filter commented-out lines
# to avoid false positives.
REPO_CONFIGURED=false
if grep -Rhs -E 'kobuk-team/intel-graphics|intel-omix' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null \
    | grep -vE '^[[:space:]]*#' | grep -q .; then
    REPO_CONFIGURED=true
fi

# Live OpenCL check is the ground truth — check this first.
OPENCL_PLATFORMS=0
if command -v clinfo &>/dev/null; then
    OPENCL_PLATFORMS=$(clinfo 2>/dev/null | grep "^Number of platforms" | awk '{print $NF}' || echo 0)
    if [ "${OPENCL_PLATFORMS:-0}" -gt 0 ]; then
        pass "clinfo reports $OPENCL_PLATFORMS OpenCL platform(s) — runtime is working"
        clinfo 2>/dev/null | grep "Device Name" | while read -r line; do
            info "  $line"
        done
    else
        fail "clinfo reports 0 OpenCL platforms — runtime does not see the GPU"
        NEED_RUNTIME=true
        LAYER3_OK=false
    fi
else
    warn "clinfo not installed — skipping live OpenCL check (install with: sudo apt install clinfo)"
fi

# Runtime version is informational only. No specific minimum version is
# asserted here -- Intel's OMIX doc does not document one -- so clinfo above
# (and xpu-smi discovery below) is the ground truth for whether the runtime
# actually works, not the installed package's version number.
if dpkg -s intel-opencl-icd &>/dev/null; then
    RUNTIME_VERSION=$(dpkg -s intel-opencl-icd 2>/dev/null | grep '^Version:' | awk '{print $2}')
    info "intel-opencl-icd installed: $RUNTIME_VERSION"
else
    if [ "${OPENCL_PLATFORMS:-0}" -gt 0 ]; then
        warn "intel-opencl-icd package not found via dpkg but clinfo shows GPU — may be installed outside apt"
    else
        fail "intel-opencl-icd not installed and clinfo sees no GPU"
        NEED_RUNTIME=true
        LAYER3_OK=false
    fi
fi

printf '\n%b\n' "${BOLD}── Layer 3b: xpu-smi Version ──${NC}"

if command -v xpu-smi &>/dev/null; then
    XPU_SMI_VER=$(xpu-smi --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    XPU_SMI_VER=${XPU_SMI_VER:-unknown}
    info "xpu-smi version: $XPU_SMI_VER"

    if [ "$XPU_SMI_VER" = "unknown" ]; then
        fail "Unable to parse xpu-smi version — need xpu-smi >= 1.3"
        NEED_RUNTIME=true
        LAYER3_OK=false
    else
        XPU_SMI_MAJOR=$(printf '%s' "$XPU_SMI_VER" | cut -d. -f1)
        XPU_SMI_MINOR=$(printf '%s' "$XPU_SMI_VER" | cut -d. -f2)
        if [ "$XPU_SMI_MAJOR" -lt 1 ] \
            || { [ "$XPU_SMI_MAJOR" -eq 1 ] && [ "$XPU_SMI_MINOR" -lt 3 ]; }; then
            warn "xpu-smi $XPU_SMI_VER may not enumerate B70 (1.3+ recommended; install via xpu-system-setup)"
        else
            pass "xpu-smi version $XPU_SMI_VER >= 1.3"
            DISCOVERY=$(xpu-smi discovery 2>/dev/null || true)
            if printf '%s' "$DISCOVERY" | grep -q "No device discovered"; then
                fail "xpu-smi discovery: No device discovered"
                LAYER3_OK=false
            else
                pass "xpu-smi discovery: GPU(s) visible"
            fi
        fi
    fi
else
    fail "xpu-smi not installed — install xpu-smi >= 1.3 via xpu-system-setup"
    NEED_RUNTIME=true
    LAYER3_OK=false
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary + remediation
# ─────────────────────────────────────────────────────────────────────────────
printf '\n%b\n' "${BOLD}=== Summary ===${NC}"

if [ "$ISSUES" -eq 0 ]; then
    if [ "${OPENCL_PLATFORMS:-0}" -gt 0 ]; then
        printf '%b All prerequisites met — GPU is operational (clinfo: %s platform(s)).\n' \
            "$PASS" "${OPENCL_PLATFORMS:-0}"
    else
        printf '%b All prerequisites met. Run xpu-system-setup to complete host configuration.\n' "$PASS"
    fi
    exit 0
fi

printf '%b %d issue(s) found.\n\n' "$FAIL" "$ISSUES"

# ── Fix: Layer 1 — nomodeset ──────────────────────────────────────────────────
if $NEED_GRUB; then
    printf '%b Remove nomodeset from GRUB:\n' "$BOLD"
    printf "    sudo sed -i 's/\\\\bnomodeset\\\\b//g' /etc/default/grub\n"
    printf '    sudo update-grub && sudo reboot\n\n'
    if [ "$MODE" = fix ]; then
        if confirm "Remove nomodeset from /etc/default/grub and run update-grub?"; then
            run_fix "sed (remove nomodeset)" sed -i 's/\bnomodeset\b//g' /etc/default/grub
            run_fix "update-grub" update-grub
            printf '%b nomodeset removed and GRUB updated.\n' "$PASS"
            confirm "Reboot now?" && reboot
        fi
    elif [ "$MODE" = dryrun ]; then
        dryrun_note "sed -i 's/\\bnomodeset\\b//g' /etc/default/grub && update-grub"
    fi
fi

# ── Fix: Layer 2 — kernel upgrade ────────────────────────────────────────────
if $NEED_KERNEL; then
    printf '%b Kernel upgrade needed (no PCI alias/binding for this Battlemage GPU on the running kernel):\n' "$BOLD"
    printf '  No specific kernel package or version is asserted here — Intel'"'"'s OMIX\n'
    printf '  install guide does not document one. Install the newest HWE or OEM kernel\n'
    printf '  available for %s %s, reboot, then re-run this script:\n' "$DISTRO_ID" "$DISTRO_VERSION_ID"
    printf '  https://dgpu-docs.intel.com/installation-guides/installing-omix.html\n\n'
    if [ "$MODE" = fix ] || [ "$MODE" = dryrun ]; then
        printf '%b No automatic kernel remediation available — resolve manually and reboot.\n' "$WARN"
    fi
fi

# ── Fix: Layer 3 — runtime upgrade ───────────────────────────────────────────
if $NEED_RUNTIME; then
    printf '%b Install the latest compute runtime via xpu-system-setup:\n' "$BOLD"
    printf '    bash scripts/setup_xpu_system.sh --auto\n\n'
    printf '  This adds the Intel OMIX repo (repositories.intel.com), which always\n'
    printf '  resolves the latest release for your Ubuntu codename, and will upgrade\n'
    printf '  any previously installed Intel GPU packages.\n\n'
    if [ "$MODE" = fix ]; then
        if $REPO_CONFIGURED; then
            if confirm "Install or upgrade libze-intel-gpu1, intel-opencl-icd, and xpu-smi now?"; then
                run_fix "apt-get install runtime" \
                    env DEBIAN_FRONTEND=noninteractive \
                    apt-get install -y libze-intel-gpu1 intel-opencl-icd xpu-smi
                printf '%b Runtime packages installed/upgraded. Re-run this script to verify.\n' "$PASS"
            fi
        else
            printf '%b No Intel GPU repo configured yet — run xpu-system-setup first:\n' "$WARN"
            printf '    bash scripts/setup_xpu_system.sh --auto\n\n'
        fi
    elif [ "$MODE" = dryrun ]; then
        if $REPO_CONFIGURED; then
            dryrun_note "apt-get install -y libze-intel-gpu1 intel-opencl-icd xpu-smi"
        else
            printf '%b  DRY-RUN: no Intel GPU repo configured — would need to run setup_xpu_system.sh first\n' \
                "${YELLOW}»${NC}"
        fi
    fi
fi

if [ "$MODE" = check ]; then
    printf 'Re-run with --fix to apply remediations, or follow the steps above manually.\n'
fi

exit "$ISSUES"
