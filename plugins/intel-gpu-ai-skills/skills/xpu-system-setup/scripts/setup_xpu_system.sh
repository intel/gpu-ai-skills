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

# Intel OMIX (Open Middleware Xe) is the sole install path: a single pinned
# bundle of Level Zero, OpenCL, the SYCL compiler, and oneMKL/oneDNN. Do not
# also add the legacy per-package PPA (ppa:kobuk-team/intel-graphics) on the
# same host -- Intel's OMIX docs call for "a clean system without preinstalled
# Intel GPU user-mode packages from the PPA", and mixing them causes apt
# dependency conflicts (an OMIX-pinned package fighting a newer PPA version
# of the same package).
INCLUDE_DEV=false

# --- Facts sourced from dgpu-docs.intel.com ---
# Last verified against the live pages on 2026-09-17. NOT re-fetched at
# runtime by this script. Before relying on this skill, or if anything below
# looks wrong (install fails, package not found, distro rejected), an agent
# running this skill should fetch the two URLs below and reconcile these
# constants against the current page content -- see "Keeping this current"
# in SKILL.md for exactly what to check and where.
OMIX_DOC_URL="https://dgpu-docs.intel.com/installation-guides/installing-omix.html"
OMIX_CODENAMES="resolute noble"                 # Ubuntu codenames OMIX doc lists as supported
OMIX_VERSION_SERIES="0.3"                        # .../intel-omix/<series> in the repo line
OMIX_GPG_KEY_URL="https://repositories.intel.com/gpu/intel-graphics.key"
OMIX_RUNTIME_PKG="intel-omix"
OMIX_DEV_PKG="intel-omix-dev"

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
  --include-dev     Also install the OMIX dev package (intel-omix-dev): SYCL/
                    oneMKL/oneDNN headers for building from source. Runtime-only
                    (intel-omix) is installed by default and is enough to run
                    PyTorch/XPU workloads.
  --only LIST       Comma-separated list of components to setup:
                    omix-repo, omix-runtime, omix-dev, clinfo, xpu-smi, groups,
                    docker
                    Note: omix-runtime/omix-dev require omix-repo. Include it
                    in --only if not already configured.
  --skip LIST       Comma-separated list of components to skip
  --out-dir DIR     Output directory (default: ~/.out/skills/xpu-system-setup)
  --user USER       Target user for group membership (default: current user)
  -h, --help        Show this help

Default mode is interactive - prompts before each installation.

Supported Ubuntu versions and package/repo details are verified against
dgpu-docs.intel.com as of the date noted near the top of this script -- see
"Keeping this current" in SKILL.md if that verification needs to be redone.

Examples:
  setup_xpu_system.sh                      # Interactive mode, installs OMIX
  setup_xpu_system.sh --auto               # Non-interactive, installs all
  setup_xpu_system.sh --dry-run            # Show what would be done
  setup_xpu_system.sh --auto --include-dev # Also install intel-omix-dev
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
        --include-dev) INCLUDE_DEV=true; shift ;;
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

is_omix_repo_configured() {
    grep -rls "intel-omix" /etc/apt/sources.list.d/ 2>/dev/null | grep -q .
}

# Component aliases — short forms users may pass to --only / --skip.
# Keeps the interface forgiving: e.g. `--skip media` → skips `media-packages`.
normalize_component_list() {
    local list="$1"
    list="${list//,repo,/,omix-repo,}"
    list="${list//,runtime,/,omix-runtime,}"
    list="${list//,dev,/,omix-dev,}"
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

# True only if the component was explicitly named in --only.
is_explicitly_requested() {
    local component="$1"
    [[ -n "$ONLY" ]] || return 1
    local only_norm
    only_norm=$(normalize_component_list ",$ONLY,")
    echo "$only_norm" | grep -q ",$component,"
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
    [[ "$DRY_RUN" == "true" ]] && return 0
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

    if [[ "$DISTRO_ID" != "ubuntu" ]]; then
        fail "Unsupported distribution: $DISTRO_ID $DISTRO_VERSION"
        fail "This skill requires Ubuntu; supported versions are verified against dgpu-docs.intel.com."
        fail "For other distros, see: https://dgpu-docs.intel.com/installation-guides/index.html"
        exit 1
    fi

    # Supported-codename list is the OMIX_CODENAMES constant declared near
    # the top of this script (last verified against dgpu-docs.intel.com --
    # see SKILL.md "Keeping this current") — never hardcoded inline here,
    # since the doc's supported-version list has changed over time.
    info "Supported codenames per OMIX doc: $OMIX_CODENAMES"
    if [[ " $OMIX_CODENAMES " != *" $DISTRO_CODENAME "* ]]; then
        warn "Ubuntu codename '$DISTRO_CODENAME' ($DISTRO_VERSION) is not in the last-verified OMIX supported list ($OMIX_CODENAMES)"
        warn "OMIX install may fail or behave unexpectedly; re-verify against $OMIX_DOC_URL"
    fi
    info "Detected: $DISTRO_ID $DISTRO_VERSION ($DISTRO_CODENAME)"
}

# --- Component: OMIX repo (sole install path) ---
check_omix_repo() {
    info "Checking Intel OMIX repo (intel-omix/$OMIX_VERSION_SERIES)..."

    local before="missing"
    if is_omix_repo_configured; then
        before="present"
        ok "Intel OMIX repo already configured"
        record "omix-repo" "$before" "none" "present" "PASS"
        return 0
    fi

    if ! should_install "omix-repo"; then
        info "Intel OMIX repo missing — install skipped (filtered by --only/--skip)"
        record "omix-repo" "$before" "filtered-by-only" "missing" "FILTERED"
        return 0
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Add Intel OMIX repo (intel-omix/$OMIX_VERSION_SERIES) and GPG key?"; then
            info "Skipped Intel OMIX repo setup"
            record "omix-repo" "$before" "skipped-by-user" "missing" "SKIPPED"
            return 0
        fi
    fi

    info "Adding Intel OMIX repo for $DISTRO_CODENAME..."
    need_sudo || return 1

    run_or_dry sudo_cmd apt-get update -qq
    run_or_dry sudo_cmd apt-get install -y -qq gnupg wget

    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY-RUN] Would fetch $OMIX_GPG_KEY_URL and write /etc/apt/sources.list.d/intel-gpu-$DISTRO_CODENAME.list"
        record "omix-repo" "$before" "would-add" "pending" "DRY-RUN"
        return 0
    fi

    wget -qO - "$OMIX_GPG_KEY_URL" | sudo_cmd gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
    local repo_line="deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${DISTRO_CODENAME}/intel-omix/${OMIX_VERSION_SERIES} unified"
    echo "$repo_line" | sudo_cmd tee "/etc/apt/sources.list.d/intel-gpu-${DISTRO_CODENAME}.list" >/dev/null
    sudo_cmd apt-get update -qq

    if is_omix_repo_configured; then
        ok "Intel OMIX repo added"
        record "omix-repo" "$before" "added" "present" "PASS"
    else
        fail "Intel OMIX repo setup failed"
        record "omix-repo" "$before" "add-failed" "missing" "FAIL"
        return 1
    fi
}

# --- Component: OMIX runtime (intel-omix) ---
check_omix_runtime() {
    info "Checking Intel OMIX runtime ($OMIX_RUNTIME_PKG)..."

    local before="missing"
    if dpkg -l "$OMIX_RUNTIME_PKG" 2>/dev/null | grep -q "^ii"; then
        before="present"
        local ver
        ver=$(dpkg -l "$OMIX_RUNTIME_PKG" 2>/dev/null | awk '/^ii/{print $3}')
        ok "$OMIX_RUNTIME_PKG already installed ($ver)"
        record "omix-runtime" "$before" "none" "present" "PASS"
        return 0
    fi

    if ! should_install "omix-runtime"; then
        info "$OMIX_RUNTIME_PKG missing — install skipped (filtered by --only/--skip)"
        record "omix-runtime" "$before" "filtered-by-only" "missing" "FILTERED"
        return 0
    fi

    if ! is_omix_repo_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel OMIX repo is not configured — $OMIX_RUNTIME_PKG requires it."
        fail "Re-run with: --only omix-repo,omix-runtime"
        record "omix-runtime" "$before" "missing-repo" "missing" "FAIL"
        return 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install Intel OMIX runtime ($OMIX_RUNTIME_PKG)?"; then
            info "Skipped $OMIX_RUNTIME_PKG installation"
            record "omix-runtime" "$before" "skipped-by-user" "missing" "SKIPPED"
            return 0
        fi
    fi

    info "Installing $OMIX_RUNTIME_PKG..."
    need_sudo || return 1
    run_or_dry sudo_cmd apt-get install -y -qq "$OMIX_RUNTIME_PKG"

    if [[ "$DRY_RUN" == "true" ]]; then
        record "omix-runtime" "$before" "would-install" "pending" "DRY-RUN"
    elif dpkg -l "$OMIX_RUNTIME_PKG" 2>/dev/null | grep -q "^ii"; then
        ok "$OMIX_RUNTIME_PKG installed"
        record "omix-runtime" "$before" "installed" "present" "PASS"
    else
        fail "$OMIX_RUNTIME_PKG install failed"
        record "omix-runtime" "$before" "install-failed" "missing" "FAIL"
    fi
}

# --- Component: OMIX dev (intel-omix-dev, opt-in via --include-dev) ---
check_omix_dev() {
    info "Checking Intel OMIX dev package ($OMIX_DEV_PKG)..."

    local before="missing"
    if dpkg -l "$OMIX_DEV_PKG" 2>/dev/null | grep -q "^ii"; then
        before="present"
        ok "$OMIX_DEV_PKG already installed"
        record "omix-dev" "$before" "none" "present" "PASS"
        return 0
    fi

    if [[ "$INCLUDE_DEV" != "true" ]] && ! is_explicitly_requested "omix-dev"; then
        info "$OMIX_DEV_PKG not requested (pass --include-dev or --only omix-dev to install)"
        record "omix-dev" "$before" "not-requested" "missing" "FILTERED"
        return 0
    fi

    if ! should_install "omix-dev"; then
        info "$OMIX_DEV_PKG missing — install skipped (filtered by --only/--skip)"
        record "omix-dev" "$before" "filtered-by-only" "missing" "FILTERED"
        return 0
    fi

    if ! is_omix_repo_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel OMIX repo is not configured — $OMIX_DEV_PKG requires it."
        fail "Re-run with: --only omix-repo,omix-dev"
        record "omix-dev" "$before" "missing-repo" "missing" "FAIL"
        return 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install Intel OMIX dev package ($OMIX_DEV_PKG: SYCL/oneMKL/oneDNN build headers)?"; then
            info "Skipped $OMIX_DEV_PKG installation"
            record "omix-dev" "$before" "skipped-by-user" "missing" "SKIPPED"
            return 0
        fi
    fi

    info "Installing $OMIX_DEV_PKG..."
    need_sudo || return 1
    run_or_dry sudo_cmd apt-get install -y -qq "$OMIX_DEV_PKG"

    if [[ "$DRY_RUN" == "true" ]]; then
        record "omix-dev" "$before" "would-install" "pending" "DRY-RUN"
    elif dpkg -l "$OMIX_DEV_PKG" 2>/dev/null | grep -q "^ii"; then
        ok "$OMIX_DEV_PKG installed"
        record "omix-dev" "$before" "installed" "present" "PASS"
    else
        fail "$OMIX_DEV_PKG install failed"
        record "omix-dev" "$before" "install-failed" "missing" "FAIL"
    fi
}

# --- Component: clinfo (OpenCL diagnostic CLI; standalone Ubuntu package) ---
check_clinfo() {
    info "Checking clinfo..."

    local before="missing"
    if command -v clinfo &>/dev/null; then
        before="present"
        ok "clinfo already installed"
        record "clinfo" "$before" "none" "present" "PASS"
        return 0
    fi

    if ! should_install "clinfo"; then
        info "clinfo missing — install skipped (filtered by --only/--skip)"
        record "clinfo" "$before" "filtered-by-only" "missing" "FILTERED"
        return 0
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        if ! ask_confirm "Install clinfo (OpenCL diagnostic CLI)?"; then
            info "Skipped clinfo installation"
            record "clinfo" "$before" "skipped-by-user" "missing" "SKIPPED"
            return 0
        fi
    fi

    info "Installing clinfo..."
    need_sudo || return 1
    run_or_dry sudo_cmd apt-get install -y -qq clinfo

    if [[ "$DRY_RUN" == "true" ]]; then
        record "clinfo" "$before" "would-install" "pending" "DRY-RUN"
    elif command -v clinfo &>/dev/null; then
        ok "clinfo installed"
        record "clinfo" "$before" "installed" "present" "PASS"
    else
        fail "clinfo install failed"
        record "clinfo" "$before" "install-failed" "missing" "FAIL"
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

    # xpu-smi is published from the OMIX repo (repositories.intel.com).
    if ! is_omix_repo_configured && [[ "$DRY_RUN" != "true" ]]; then
        fail "Intel OMIX repo is not configured — xpu-smi requires it."
        fail "Re-run with: --only omix-repo,xpu-smi"
        record "xpu-smi" "$before" "missing-repo" "missing" "FAIL"
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
        gpu_count=$($run_as xpu-smi discovery 2>/dev/null | grep -c "Device Name:" || true)
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

    # Driver health. xpu-smi 2.x removed the legacy diag subcommand.
    if command -v xpu-smi &>/dev/null; then
        if xpu-smi help 2>/dev/null | grep -qw diag; then
            if $run_as xpu-smi diag --precheck &>/dev/null; then
                ok "Verification: driver precheck passed"
                ((pass++))
            else
                warn "Verification: driver precheck had warnings"
                ((warn_count++))
            fi
        elif $run_as xpu-smi health -l &>/dev/null; then
            ok "Verification: xpu-smi health check passed"
            ((pass++))
        else
            warn "Verification: xpu-smi health telemetry is unsupported or unavailable"
            ((warn_count++))
        fi
    fi

    # SYCL compiler stack (part of the OMIX runtime install).
    local oneapi_setvars="/opt/intel/oneapi/setvars.sh"
    if [[ -f "$oneapi_setvars" ]]; then
        local sycl_devices
        sycl_devices=$($run_as bash -c "source '$oneapi_setvars' >/dev/null 2>&1 && sycl-ls 2>/dev/null" | grep -ci "intel" || true)
        if [[ "$sycl_devices" -gt 0 ]]; then
            ok "Verification: sycl-ls sees $sycl_devices Intel device(s)"
            ((pass++))
        elif [[ "$visibility_relogin_pending" == "true" ]]; then
            warn "Verification: sycl-ls sees 0 devices (expected — render group pending re-login)"
            ((warn_count++))
            needs_relogin=true
        else
            fail "Verification: sycl-ls sees no Intel devices"
            ((fail_count++))
        fi
    else
        warn "Verification: $oneapi_setvars not found (skipping SYCL check — is intel-omix installed?)"
        ((warn_count++))
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
- Missing Intel OMIX repo: check network connectivity to repositories.intel.com
- xpu-smi install failed: ensure omix-repo is configured first
- Docker failed: check systemd service status (`systemctl status docker`)
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

    check_omix_repo
    check_omix_runtime
    check_omix_dev
    check_clinfo
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
