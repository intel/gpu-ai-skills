#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# First-time Intel XPU/GPU system setup.
# Detects what's missing and installs only what's needed.

set -uo pipefail

# --- Defaults ---
AUTO=false
DRY_RUN=false
INTERACTIVE=true
ONLY=""
SKIP=""
TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
TARGET_HOME="${TARGET_HOME:-$HOME}"
OUT_DIR="${TARGET_HOME}/.out/skills/xpu-system-setup"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

usage() {
    cat <<'EOF'
Usage: setup_xpu_system.sh [OPTIONS]

Options:
  --auto            Run all checks and install missing components without prompts
  --dry-run         Show what would be done without making changes
  --yes             Same as --auto (non-interactive mode)
  --only LIST       Comma-separated list of components to setup:
                    intel-ppa, compute-packages, media-packages, pytorch-extras,
                    xpu-smi, groups, docker
                    Note: compute-packages, media-packages, pytorch-extras, and
                    xpu-smi require intel-ppa. Include it in --only if not already
                    configured (e.g. --only intel-ppa,xpu-smi).
  --skip LIST       Comma-separated list of components to skip
  --out-dir DIR     Output directory (default: ~/.out/skills/xpu-system-setup)
  --user USER       Target user for group membership (default: current user)
  -h, --help        Show this help

Default mode is interactive - prompts before each installation.

Examples:
  setup_xpu_system.sh                      # Interactive mode, prompts for each component
  setup_xpu_system.sh --auto               # Non-interactive, installs all
  setup_xpu_system.sh --dry-run            # Show what would be done
  setup_xpu_system.sh --only xpu-smi,groups
  setup_xpu_system.sh --auto --skip docker
EOF
    exit 0
}

# --- Argument parsing ---
require_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Error: option '$1' requires a value" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto|--yes|-y) AUTO=true; INTERACTIVE=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --only) require_value "$1" "${2:-}"; ONLY="$2"; shift 2 ;;
        --skip) require_value "$1" "${2:-}"; SKIP="$2"; shift 2 ;;
        --out-dir) require_value "$1" "${2:-}"; OUT_DIR="$2"; shift 2 ;;
        --user) require_value "$1" "${2:-}"; TARGET_USER="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Setup output ---
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/setup.log"
STATUS_TSV="$OUT_DIR/status.tsv"
SUMMARY="$OUT_DIR/SUMMARY.md"
: > "$LOG"
echo -e "component\tbefore\taction\tafter\tresult" > "$STATUS_TSV"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG" >&2; }
ok() { echo -e "${GREEN}[OK]${NC} $*" | tee -a "$LOG" >&2; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG" >&2; }
fail() { echo -e "${RED}[FAIL]${NC} $*" | tee -a "$LOG" >&2; }

record() {
    local component="$1" before="$2" action="$3" after="$4" result="$5"
    echo -e "${component}\t${before}\t${action}\t${after}\t${result}" >> "$STATUS_TSV"
}

# Component aliases — short forms users may pass to --only / --skip.
# Keeps the interface forgiving: e.g. `--skip media` → skips `media-packages`.
normalize_component_list() {
    local list="$1"
    list="${list//,media,/,media-packages,}"
    list="${list//,compute,/,compute-packages,}"
    list="${list//,pytorch,/,pytorch-extras,}"
    list="${list//,ppa,/,intel-ppa,}"
    echo "$list"
}

# Returns 0 if the component is allowed to install (not filtered by --only/--skip).
# Detection always runs regardless — only the install/fix step is gated.
should_install() {
    local component="$1"
    if [[ -n "$ONLY" ]]; then
        local only_norm
        only_norm=$(normalize_component_list ",$ONLY,")
        echo "$only_norm" | grep -q ",$component," || return 1
    fi
    if [[ -n "$SKIP" ]]; then
        local skip_norm
        skip_norm=$(normalize_component_list ",$SKIP,")
        echo "$skip_norm" | grep -q ",$component," && return 1
    fi
    return 0
}

is_ppa_configured() {
    find /etc/apt/sources.list.d/ -name "*kobuk*" -o -name "*intel-graphics*" 2>/dev/null | grep -q . || \
    grep -rls "kobuk-team/intel-graphics" /etc/apt/sources.list.d/ 2>/dev/null | grep -q .
}

run_or_dry() {
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY-RUN] Would execute: $*"
        return 0
    fi
    "$@"
}

ask_confirm() {
    local prompt="$1"
    if [[ "$INTERACTIVE" != "true" ]]; then
        return 0  # Auto mode, always yes
    fi

    echo -e "${YELLOW}[?]${NC} $prompt [y/N] " >&2
    read -r response
    case "$response" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

need_sudo() {
    if [[ $EUID -ne 0 ]]; then
        if [[ "$INTERACTIVE" == "true" ]]; then
            if ! sudo -v 2>/dev/null; then
                fail "This operation requires sudo. Please provide your password or run with sudo."
                return 1
            fi
        else
            if ! sudo -n true 2>/dev/null; then
                fail "This operation requires sudo. In non-interactive mode, passwordless sudo is required."
                return 1
            fi
        fi
    fi
    return 0
}

sudo_cmd() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

# --- Detect GPU type ---
detect_gpu_type() {
    # Check if this is a Gaudi system (data center accelerator)
    # Gaudi uses hl-smi instead of xpu-smi and has different driver stack
    if command -v hl-smi &>/dev/null || lspci 2>/dev/null | grep -qi "habanalabs"; then
        fail "Detected Intel Gaudi (Habana Labs) accelerator"
        fail "This skill is for Intel client GPUs (Arc, Arc Pro, Battlemage)."
        fail "Gaudi uses different drivers and management tools (hl-smi, not xpu-smi)."
        fail "For Gaudi setup, see: https://docs.habana.ai/"
        exit 1
    fi

    # Check for Intel Data Center GPU Max (Ponte Vecchio)
    # These require different driver packages from the Data Center GPU repository
    if lspci 2>/dev/null | grep -qi "Data Center GPU Max\|Ponte Vecchio"; then
        fail "Detected Intel Data Center GPU Max (Ponte Vecchio)"
        fail "This skill is for Intel client GPUs (Arc, Arc Pro, Battlemage)."
        fail "Data Center GPU Max requires different drivers from the Intel Data Center GPU repository."
        fail "See: https://dgpu-docs.intel.com/driver/installation.html"
        exit 1
    fi

    # Check that at least one Intel GPU is visible in lspci
    if ! lspci 2>/dev/null | grep -Ei 'VGA|Display|3D' | grep -qi intel; then
        fail "No Intel GPU detected in lspci"
        fail "Check that the GPU card is properly seated and powered."
        fail "If this is a new install, verify the card appears in BIOS/UEFI."
        fail "Run 'lspci | grep VGA' to see what devices are visible."
        exit 1
    fi
}

# --- Detect distro ---
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO_ID="${ID:-unknown}"
        DISTRO_VERSION="${VERSION_ID:-unknown}"
        DISTRO_CODENAME="${VERSION_CODENAME:-unknown}"
    else
        fail "Cannot detect distribution (no /etc/os-release)"
        exit 1
    fi

    case "$DISTRO_ID" in
        ubuntu)
            # Per https://dgpu-docs.intel.com/driver/client/overview.html
            # The ppa:kobuk-team/intel-graphics PPA supports Ubuntu 24.04 and 25.10
            case "$DISTRO_VERSION" in
                24.04|25.10)
                    # Officially supported versions
                    ;;
                22.04)
                    fail "Ubuntu 22.04 requires a different installation method"
                    fail "The ppa:kobuk-team/intel-graphics PPA is for Ubuntu 24.04 and 25.10 only"
                    fail "For Ubuntu 22.04 instructions, see: https://dgpu-docs.intel.com/driver/client/overview.html"
                    exit 1
                    ;;
                20.04|21.*|23.*)
                    fail "Ubuntu $DISTRO_VERSION is not supported"
                    fail "This skill requires Ubuntu 24.04 or 25.10"
                    fail "See: https://dgpu-docs.intel.com/driver/client/overview.html"
                    exit 1
                    ;;
                *)
                    # Newer versions than tested - might work
                    warn "Ubuntu $DISTRO_VERSION is not officially tested"
                    warn "This skill is tested on Ubuntu 24.04 and 25.10"
                    warn "Newer versions may work but are untested"
                    ;;
            esac
            ;;
        *)
            fail "Unsupported distribution: $DISTRO_ID $DISTRO_VERSION"
            fail "This skill requires Ubuntu 24.04 or 25.10."
            fail "The Intel client GPU PPA (ppa:kobuk-team/intel-graphics) is Ubuntu-specific."
            fail "For other distros, see: https://dgpu-docs.intel.com/driver/client/overview.html"
            exit 1
            ;;
    esac
    info "Detected: $DISTRO_ID $DISTRO_VERSION ($DISTRO_CODENAME)"
}

# --- Component: Intel GPU PPA (ppa:kobuk-team/intel-graphics) ---
check_intel_ppa() {
    info "Checking Intel GPU PPA (ppa:kobuk-team/intel-graphics)..."

    local before="missing"
    if find /etc/apt/sources.list.d/ -name "*kobuk*" -o -name "*intel-graphics*" 2>/dev/null | grep -q . || \
       grep -rls "kobuk-team/intel-graphics" /etc/apt/sources.list.d/ 2>/dev/null | grep -q .; then
        before="present"
        ok "Intel GPU PPA (kobuk-team/intel-graphics) already configured"
        record "intel-ppa" "$before" "none" "present" "PASS"
        return 0
    fi

    if ! should_install "intel-ppa"; then
        info "Intel GPU PPA missing — install skipped (filtered by --only/--skip)"
        record "intel-ppa" "$before" "filtered-by-only" "missing" "FILTERED"
        return 0
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install Intel GPU PPA (ppa:kobuk-team/intel-graphics)?"; then
            info "Skipped Intel GPU PPA installation"
            record "intel-ppa" "$before" "skipped-by-user" "missing" "SKIPPED"
            return 0
        fi
    fi

    info "Adding Intel GPU PPA (ppa:kobuk-team/intel-graphics)..."
    need_sudo || return 1

    run_or_dry sudo_cmd apt-get update -qq
    run_or_dry sudo_cmd apt-get install -y -qq software-properties-common
    run_or_dry sudo_cmd add-apt-repository -y ppa:kobuk-team/intel-graphics
    run_or_dry sudo_cmd apt-get update -qq

    if [[ "$DRY_RUN" == "true" ]]; then
        record "intel-ppa" "$before" "would-add" "pending" "DRY-RUN"
    else
        ok "Intel GPU PPA added"
        record "intel-ppa" "$before" "added" "present" "PASS"
    fi
}

# --- Component: Compute packages (official Intel list) ---
# Per https://dgpu-docs.intel.com/driver/client/overview.html
COMPUTE_PACKAGES="libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc"

check_compute_packages() {
    info "Checking compute packages..."

    local missing=""
    for pkg in $COMPUTE_PACKAGES; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            missing="$missing $pkg"
        fi
    done

    if [[ -z "$missing" ]]; then
        local ver
        ver=$(dpkg -l libze-intel-gpu1 2>/dev/null | awk '/^ii/{print $3}')
        ok "All compute packages installed (libze-intel-gpu1 $ver)"
        record "compute-packages" "present" "none" "present" "PASS"
        return 0
    fi

    local before="missing:$missing"

    if ! should_install "compute-packages"; then
        info "Compute packages missing — install skipped (filtered by --only/--skip)"
        record "compute-packages" "$before" "filtered-by-only" "incomplete" "FILTERED"
        return 0
    fi

    if ! is_ppa_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel GPU PPA is not configured — compute packages require it."
        fail "Re-run with: --only intel-ppa,compute-packages"
        record "compute-packages" "$before" "missing-ppa" "incomplete" "FAIL"
        return 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install compute packages ($COMPUTE_PACKAGES)?"; then
            info "Skipped compute packages installation"
            record "compute-packages" "$before" "skipped-by-user" "incomplete" "SKIPPED"
            return 0
        fi
    fi

    info "Installing compute packages:$missing"
    need_sudo || return 1
    # shellcheck disable=SC2086
    run_or_dry sudo_cmd apt-get install -y -qq $COMPUTE_PACKAGES

    if [[ "$DRY_RUN" == "true" ]]; then
        record "compute-packages" "$before" "would-install" "pending" "DRY-RUN"
    else
        local still_missing=""
        for pkg in $COMPUTE_PACKAGES; do
            if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
                still_missing="$still_missing $pkg"
            fi
        done
        if [[ -z "$still_missing" ]]; then
            ok "Compute packages installed"
            record "compute-packages" "$before" "installed" "present" "PASS"
        else
            fail "Compute install incomplete — still missing:$still_missing"
            record "compute-packages" "$before" "install-incomplete" "missing:$still_missing" "FAIL"
        fi
    fi
}

# --- Component: Media packages ---
# VAAPI stack used for hardware video encode/decode. Compute-only servers
# can skip with `--skip media`.
#
# Ubuntu 25.10 renamed `libmfx-gen1` to `libmfx-gen1.2`; we resolve the actual
# package name from the apt cache at runtime so we work across releases.
MEDIA_PACKAGES_BASE="intel-media-va-driver-non-free libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo"

resolve_media_packages() {
    # Pick whichever libmfx-gen variant the apt cache offers on this distro.
    local libmfx=""
    for cand in libmfx-gen1.2 libmfx-gen1; do
        if apt-cache show "$cand" &>/dev/null; then
            libmfx="$cand"; break
        fi
    done
    if [[ -z "$libmfx" ]]; then
        # Last-resort fallback to keep the historical name in error messages.
        libmfx="libmfx-gen1"
    fi
    echo "$MEDIA_PACKAGES_BASE $libmfx"
}

check_media_packages() {
    info "Checking media packages..."

    local media_pkgs
    media_pkgs=$(resolve_media_packages)

    local missing=""
    for pkg in $media_pkgs; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            missing="$missing $pkg"
        fi
    done

    if [[ -z "$missing" ]]; then
        ok "Media packages installed (vainfo, libvpl2, etc.)"
        record "media-packages" "present" "none" "present" "PASS"
        return 0
    fi

    local before="missing:$missing"

    if ! should_install "media-packages"; then
        info "Media packages missing — install skipped (filtered by --only/--skip)"
        record "media-packages" "$before" "filtered-by-only" "incomplete" "FILTERED"
        return 0
    fi

    if ! is_ppa_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel GPU PPA is not configured — media packages require it."
        fail "Re-run with: --only intel-ppa,media-packages"
        record "media-packages" "$before" "missing-ppa" "incomplete" "FAIL"
        return 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install media packages (VAAPI video encode/decode support)?"; then
            info "Skipped media packages installation"
            record "media-packages" "$before" "skipped-by-user" "incomplete" "SKIPPED"
            return 0
        fi
    fi

    info "Installing media packages:$missing"
    need_sudo || return 1
    # shellcheck disable=SC2086
    run_or_dry sudo_cmd apt-get install -y -qq $media_pkgs

    if [[ "$DRY_RUN" == "true" ]]; then
        record "media-packages" "$before" "would-install" "pending" "DRY-RUN"
        return 0
    fi

    # Re-verify per-package — apt-get's exit code can be 0 even when an
    # individual package was unavailable (silent skip on `Unable to locate`).
    local still_missing=""
    for pkg in $media_pkgs; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            still_missing="$still_missing $pkg"
        fi
    done

    if [[ -z "$still_missing" ]]; then
        ok "Media packages installed"
        record "media-packages" "$before" "installed" "present" "PASS"
    else
        fail "Media install incomplete — still missing:$still_missing"
        record "media-packages" "$before" "install-incomplete" "missing:$still_missing" "FAIL"
    fi
}

# --- Component: PyTorch extras (libze-dev, intel-ocloc) ---
# Per official docs: "if you plan to use PyTorch, install libze-dev and intel-ocloc additionally"
PYTORCH_PACKAGES="libze-dev intel-ocloc"

check_pytorch_extras() {
    info "Checking PyTorch extras..."

    local missing=""
    for pkg in $PYTORCH_PACKAGES; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            missing="$missing $pkg"
        fi
    done

    if [[ -z "$missing" ]]; then
        ok "PyTorch extras installed (libze-dev, intel-ocloc)"
        record "pytorch-extras" "present" "none" "present" "PASS"
        return 0
    fi

    local before="missing:$missing"

    if ! should_install "pytorch-extras"; then
        info "PyTorch extras missing — install skipped (filtered by --only/--skip)"
        record "pytorch-extras" "$before" "filtered-by-only" "incomplete" "FILTERED"
        return 0
    fi

    if ! is_ppa_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel GPU PPA is not configured — PyTorch extras require it."
        fail "Re-run with: --only intel-ppa,pytorch-extras"
        record "pytorch-extras" "$before" "missing-ppa" "incomplete" "FAIL"
        return 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install PyTorch extras (libze-dev, intel-ocloc)?"; then
            info "Skipped PyTorch extras installation"
            record "pytorch-extras" "$before" "skipped-by-user" "incomplete" "SKIPPED"
            return 0
        fi
    fi

    info "Installing PyTorch extras:$missing"
    need_sudo || return 1
    # shellcheck disable=SC2086
    run_or_dry sudo_cmd apt-get install -y -qq $PYTORCH_PACKAGES

    if [[ "$DRY_RUN" == "true" ]]; then
        record "pytorch-extras" "$before" "would-install" "pending" "DRY-RUN"
    else
        local still_missing=""
        for pkg in $PYTORCH_PACKAGES; do
            if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
                still_missing="$still_missing $pkg"
            fi
        done
        if [[ -z "$still_missing" ]]; then
            ok "PyTorch extras installed"
            record "pytorch-extras" "$before" "installed" "present" "PASS"
        else
            fail "PyTorch extras install incomplete — still missing:$still_missing"
            record "pytorch-extras" "$before" "install-incomplete" "missing:$still_missing" "FAIL"
        fi
    fi
}

# --- Component: xpu-smi ---
check_xpu_smi() {
    info "Checking xpu-smi..."

    local before="missing"
    if command -v xpu-smi &>/dev/null; then
        before="present"
        local ver
        ver=$(xpu-smi --version 2>/dev/null | head -1 || echo "installed")
        ok "xpu-smi already installed ($ver)"
        record "xpu-smi" "$before" "none" "present" "PASS"
        return 0
    fi

    if ! should_install "xpu-smi"; then
        info "xpu-smi missing — install skipped (filtered by --only/--skip)"
        record "xpu-smi" "$before" "filtered-by-only" "missing" "FILTERED"
        return 0
    fi

    if ! is_ppa_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel GPU PPA is not configured — xpu-smi requires it."
        fail "Re-run with: --only intel-ppa,xpu-smi"
        record "xpu-smi" "$before" "missing-ppa" "missing" "FAIL"
        return 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install xpu-smi (GPU monitoring tool)?"; then
            info "Skipped xpu-smi installation"
            record "xpu-smi" "$before" "skipped-by-user" "missing" "SKIPPED"
            return 0
        fi
    fi

    info "Installing xpu-smi..."
    need_sudo || return 1
    run_or_dry sudo_cmd apt-get install -y -qq xpu-smi

    if [[ "$DRY_RUN" == "true" ]]; then
        record "xpu-smi" "$before" "would-install" "pending" "DRY-RUN"
    else
        if command -v xpu-smi &>/dev/null; then
            ok "xpu-smi installed"
            record "xpu-smi" "$before" "installed" "present" "PASS"
        else
            fail "xpu-smi installation failed — package may not be in configured repos"
            record "xpu-smi" "$before" "install-failed" "missing" "FAIL"
        fi
    fi
}

# --- Component: User groups ---
check_groups() {
    info "Checking user groups for '$TARGET_USER'..."

    local user_groups
    user_groups=$(id -nG "$TARGET_USER" 2>/dev/null)

    if echo "$user_groups" | grep -qw "render"; then
        ok "User '$TARGET_USER' in render group"
        record "groups" "render" "none" "render" "PASS"
        return 0
    fi

    local before="no-render"

    if ! should_install "groups"; then
        info "User '$TARGET_USER' not in render group — change skipped (filtered by --only/--skip)"
        record "groups" "$before" "filtered-by-only" "$before" "FILTERED"
        return 0
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Add user '$TARGET_USER' to render group (for GPU access)?"; then
            info "Skipped render group configuration"
            record "groups" "$before" "skipped-by-user" "$before" "SKIPPED"
            return 0
        fi
    fi

    need_sudo || return 1

    info "Adding '$TARGET_USER' to render group (gpasswd -a)..."
    run_or_dry sudo_cmd gpasswd -a "$TARGET_USER" render

    if [[ "$DRY_RUN" == "true" ]]; then
        record "groups" "$before" "would-add" "pending" "DRY-RUN"
    else
        ok "Render group added. Run 'newgrp render' or re-login to activate."
        record "groups" "$before" "added" "render (relogin)" "PASS-RELOGIN"
    fi
}

# --- Component: Docker ---
check_docker() {
    info "Checking Docker..."

    local before="missing"
    if command -v docker &>/dev/null; then
        before="present"
        if docker info &>/dev/null; then
            local ver
            ver=$(docker --version | awk '{print $3}' | tr -d ',')
            ok "Docker installed and reachable ($ver)"

            # Check docker group
            local user_groups
            user_groups=$(id -nG "$TARGET_USER" 2>/dev/null)
            if ! echo "$user_groups" | grep -qw "docker"; then
                if ! should_install "docker"; then
                    warn "User '$TARGET_USER' not in docker group — change skipped (filtered by --only/--skip)"
                    record "docker" "$before" "filtered-by-only" "present (no group)" "FILTERED"
                elif [[ "$AUTO" == "true" || "$DRY_RUN" == "true" ]]; then
                    info "Adding '$TARGET_USER' to docker group..."
                    need_sudo && run_or_dry sudo_cmd usermod -aG docker "$TARGET_USER"
                    record "docker" "$before" "added-group" "present (relogin)" "PASS-RELOGIN"
                else
                    if ! ask_confirm "Add user '$TARGET_USER' to docker group?"; then
                        info "Skipped docker group addition"
                        record "docker" "$before" "skipped-by-user" "present (no group)" "SKIPPED"
                    else
                        need_sudo && run_or_dry sudo_cmd usermod -aG docker "$TARGET_USER"
                        record "docker" "$before" "added-group" "present (relogin)" "PASS-RELOGIN"
                    fi
                fi
            else
                record "docker" "$before" "none" "present ($ver)" "PASS"
            fi
            return 0
        else
            warn "Docker installed but daemon not reachable (permission or service issue)"
            before="present-no-access"
        fi
    fi

    if ! should_install "docker"; then
        info "Docker needs install/fix — skipped (filtered by --only/--skip)"
        record "docker" "$before" "filtered-by-only" "$before" "FILTERED"
        return 0
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install Docker (container runtime)?"; then
            info "Skipped Docker installation"
            record "docker" "$before" "skipped-by-user" "$before" "SKIPPED"
            return 0
        fi
    fi

    if [[ "$before" == "missing" ]]; then
        info "Installing Docker via official convenience script..."
        need_sudo || return 1
        if [[ "$DRY_RUN" == "true" ]]; then
            info "[DRY-RUN] Would download and run https://get.docker.com"
            record "docker" "$before" "would-install" "pending" "DRY-RUN"
            return 0
        fi

        # mktemp, not /tmp/get-docker-$$.sh: a PID is guessable, so another
        # account on the host could pre-create that path as a symlink or win the
        # race between the download and the `sh` below -- and this file is run
        # under sudo. mktemp -d gives an unpredictable name and 0700, so only
        # this user can substitute the contents.
        local docker_dir docker_script
        docker_dir=$(mktemp -d) || {
            fail "Could not create a temp dir for the Docker install script"
            record "docker" "missing" "install-failed" "missing" "FAIL"
            return 1
        }
        docker_script="$docker_dir/get-docker.sh"
        curl -fsSL https://get.docker.com -o "$docker_script"
        local script_hash
        script_hash=$(sha256sum "$docker_script" | awk '{print $1}')
        # Recorded for the audit trail, NOT verified: get.docker.com is
        # re-published often enough that a pinned digest would fail closed on
        # every upstream refresh. Trust here rests on TLS to docker.com alone.
        # Installing from Docker's apt repository with their signing key pinned
        # is the verifiable alternative -- see the skill's SKILL.md.
        log "Docker install script SHA256 (recorded, not verified): $script_hash"
        warn "Running https://get.docker.com under sudo; its content is trusted via TLS only."
        sudo_cmd sh "$docker_script" >> "$LOG" 2>&1
        # Non-recursive on purpose: remove the one file we created, then the
        # now-empty directory. A recursive force-delete on a variable-expanded
        # path is the pattern this repo forbids (and tests/static.sh scans for
        # it), because an empty or mistyped variable turns it into a recursive
        # delete of the wrong tree. rmdir also fails loudly rather than silently
        # discarding anything unexpected left in the directory.
        rm -f "$docker_script"
        rmdir "$docker_dir" 2>/dev/null || \
            warn "Left $docker_dir in place: not empty after the Docker install"
        if command -v docker &>/dev/null; then
            sudo_cmd systemctl enable --now docker >> "$LOG" 2>&1
            sudo_cmd usermod -aG docker "$TARGET_USER"
            ok "Docker installed. Re-login required for non-root access."
            record "docker" "missing" "installed" "present (relogin)" "PASS-RELOGIN"
        else
            fail "Docker installation failed"
            record "docker" "missing" "install-failed" "missing" "FAIL"
        fi
    elif [[ "$before" == "present-no-access" ]]; then
        info "Fixing Docker access..."
        need_sudo || return 1
        run_or_dry sudo_cmd systemctl enable --now docker
        run_or_dry sudo_cmd usermod -aG docker "$TARGET_USER"
        if [[ "$DRY_RUN" == "true" ]]; then
            record "docker" "$before" "would-fix" "pending" "DRY-RUN"
        else
            ok "Docker service started, user added to docker group. Re-login required."
            record "docker" "$before" "fixed" "present (relogin)" "PASS-RELOGIN"
        fi
    fi
}


# --- Verification gate ---
run_verification() {
    info ""
    info "=== Post-Setup Verification ==="

    local pass=0 warn_count=0 fail_count=0 needs_relogin=false

    # If running as root but TARGET_USER is different, run verification probes as TARGET_USER.
    local run_as=""
    if [[ $EUID -eq 0 && "$TARGET_USER" != "root" ]]; then
        run_as="sudo -u $TARGET_USER"
        info "Running verification as '$TARGET_USER' (script is running as root)"
    fi

    # Pre-compute environment facts that other checks depend on.
    local current_groups render_active=false dri_present=false dri_accessible=false
    current_groups=$($run_as id -nG "$TARGET_USER" 2>/dev/null)
    if echo "$current_groups" | grep -qw "render"; then render_active=true; fi
    if ls /dev/dri/renderD* &>/dev/null; then
        dri_present=true
        # Check if at least one device is readable (indicates proper permissions)
        for dev in /dev/dri/renderD*; do
            if [[ -r "$dev" ]]; then
                dri_accessible=true
                break
            fi
        done
    fi

    # /dev/dri (run first — explains downstream visibility failures)
    if [[ "$dri_present" == "true" ]]; then
        if [[ "$dri_accessible" == "true" ]]; then
            ok "Verification: /dev/dri/renderD* exists and is readable"
            ((pass++))
        else
            warn "Verification: /dev/dri/renderD* exists but not readable (render group may be inactive)"
            ((warn_count++))
            needs_relogin=true
        fi
    else
        fail "Verification: /dev/dri/renderD* not found (no Intel GPU or driver not loaded)"
        ((fail_count++))
    fi

    # Group membership (effective)
    if [[ "$render_active" == "true" ]]; then
        ok "Verification: current session has render group"
        ((pass++))
    else
        warn "Verification: render group not active in current session (re-login required)"
        ((warn_count++))
        needs_relogin=true
    fi

    # Device visibility checks — clinfo and xpu-smi need render group to see GPUs.
    # If render isn't active but the device nodes exist and the driver is healthy,
    # 0-device output is expected and should not block; treat as relogin-pending.
    local visibility_relogin_pending=false
    if [[ "$render_active" != "true" && "$dri_present" == "true" ]]; then
        visibility_relogin_pending=true
    fi

    # clinfo verification (per official docs)
    if command -v clinfo &>/dev/null; then
        local devices
        devices=$($run_as clinfo 2>/dev/null | grep "Device Name" || true)
        if [[ -n "$devices" ]]; then
            ok "Verification: clinfo sees Intel GPU(s)"
            ((pass++))
        elif [[ "$visibility_relogin_pending" == "true" ]]; then
            warn "Verification: clinfo sees 0 devices (expected — render group pending re-login)"
            ((warn_count++))
            needs_relogin=true
        else
            fail "Verification: clinfo sees no Intel devices"
            ((fail_count++))
        fi
    else
        warn "Verification: clinfo not available (skipping OpenCL check)"
        ((warn_count++))
    fi

    # xpu-smi discovery
    if command -v xpu-smi &>/dev/null; then
        local gpu_count
        gpu_count=$($run_as xpu-smi discovery 2>/dev/null | grep -c "Device ID" || true)
        gpu_count=${gpu_count:-0}
        if [[ "$gpu_count" -gt 0 ]]; then
            ok "Verification: xpu-smi sees $gpu_count GPU(s)"
            ((pass++))
        elif [[ "$visibility_relogin_pending" == "true" ]]; then
            warn "Verification: xpu-smi sees 0 GPUs (expected — render group pending re-login)"
            ((warn_count++))
            needs_relogin=true
        else
            fail "Verification: xpu-smi sees 0 GPUs"
            ((fail_count++))
        fi
    else
        warn "Verification: xpu-smi not available (skipping GPU check)"
        ((warn_count++))
    fi

    # Driver health (independent of group membership — uses sysfs)
    if command -v xpu-smi &>/dev/null; then
        if $run_as xpu-smi diag --precheck &>/dev/null; then
            ok "Verification: driver precheck passed"
            ((pass++))
        else
            warn "Verification: driver precheck had warnings"
            ((warn_count++))
        fi
    fi

    # Docker
    if command -v docker &>/dev/null && docker info &>/dev/null; then
        ok "Verification: Docker daemon reachable"
        ((pass++))
    elif command -v docker &>/dev/null; then
        if systemctl is-active docker &>/dev/null; then
            warn "Verification: Docker daemon running but not reachable (re-login for docker group)"
            ((warn_count++))
            needs_relogin=true
        else
            fail "Verification: Docker installed but daemon is not running (check: systemctl status docker)"
            ((fail_count++))
        fi
    else
        warn "Verification: Docker not installed"
        ((warn_count++))
    fi

    info ""
    info "=== Verification Summary: PASS=$pass WARN=$warn_count FAIL=$fail_count ==="

    local verdict="READY"
    if [[ "$fail_count" -gt 0 ]]; then
        verdict="BLOCKED"
    elif [[ "$needs_relogin" == "true" ]]; then
        verdict="READY AFTER RELOGIN"
    elif [[ "$warn_count" -gt 0 ]]; then
        verdict="READY WITH WARNINGS"
    fi

    info "Verdict: $verdict"

    if [[ "$needs_relogin" == "true" ]]; then
        info ""
        info "Action required — pick the render group up in the current shell:"
        info "  newgrp render"
        info "  xpu-smi discovery && clinfo -l"
        info ""
        info "Alternative: log out and log back in (permanent for all future sessions)."
    fi

    echo "$verdict"
}

# --- Generate summary ---
generate_summary() {
    local verdict="$1"
    cat > "$SUMMARY" <<EOF
# XPU System Setup Summary

**Date:** $(date '+%Y-%m-%d %H:%M:%S')
**Host:** $(hostname)
**User:** $TARGET_USER
**Distribution:** $DISTRO_ID $DISTRO_VERSION ($DISTRO_CODENAME)
**Mode:** $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY-RUN"; elif [[ "$AUTO" == "true" ]]; then echo "AUTO"; else echo "INTERACTIVE"; fi)

## Verdict: $verdict

## Component Status

$(column -t -s$'\t' "$STATUS_TSV" 2>/dev/null || cat "$STATUS_TSV")

## Next Steps

EOF

    case "$verdict" in
        "READY")
            echo "System is ready for XPU workloads. You can verify GPU access with \`xpu-smi discovery\` or \`clinfo\`." >> "$SUMMARY"
            ;;
        "READY AFTER RELOGIN")
            cat >> "$SUMMARY" <<'EOF'
The render group has been added; activate it before running GPU workloads.

**Recommended:**

```sh
newgrp render
xpu-smi discovery && clinfo -l
```

**Alternative** — log out and log back in. This makes the render group
active for all future sessions (the `newgrp` form gives you a subshell
that ends when you `exit`).
EOF
            ;;
        "READY WITH WARNINGS")
            cat >> "$SUMMARY" <<'EOF'
System is usable but some checks reported warnings (see the verification
log). Common causes:

- A component was skipped (`SKIPPED` / `FILTERED` in the status table) —
  re-run with `--auto` or without `--only` to install it.
- An optional verification probe was skipped because its tool isn't
  installed (e.g., xpu-smi missing causes the GPU-count check to be skipped).

Re-run the script after addressing the warnings to confirm a clean PASS.
EOF
            ;;
        "BLOCKED")
            cat >> "$SUMMARY" <<'EOF'
Review failed components in the status table above. Common fixes:
- Missing Intel PPA: Check network connectivity to repositories.intel.com
- xpu-smi install failed: Ensure Intel GPU PPA is configured first
- Docker failed: Check systemd service status (`systemctl status docker`)
EOF
            ;;
    esac

    info "Summary written to: $SUMMARY"
    info "Full log: $LOG"
    info "Status: $STATUS_TSV"
}

# --- Main ---
main() {
    info "=== Intel XPU System Setup ==="
    info "Mode: $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY-RUN"; elif [[ "$AUTO" == "true" ]]; then echo "AUTO"; else echo "INTERACTIVE"; fi)"
    info "Target user: $TARGET_USER"
    info ""

    detect_gpu_type
    detect_distro

    check_intel_ppa
    check_compute_packages
    check_media_packages
    check_pytorch_extras
    check_xpu_smi
    check_groups
    check_docker

    local verdict
    if [[ "$DRY_RUN" == "true" ]]; then
        verdict="DRY-RUN COMPLETE"
    else
        verdict=$(run_verification)
    fi

    generate_summary "$verdict"
}

main
