#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Install the skill pack into every agent's standard skills directory
# from a local clone. No GitHub push or marketplace registration
# required.
#
# Usage:
#   bash scripts/install.sh                # install into every agent
#                                          # whose dir already exists
#   bash scripts/install.sh --all          # also create dirs that
#                                          # do not yet exist
#   bash scripts/install.sh --agent NAME   # just one agent (claude,
#                                          # codex, opencode, copilot,
#                                          # qwen, kimi, hermes,
#                                          # openclaw, gemini, cursor)
#   bash scripts/install.sh --uninstall    # remove the skills from
#                                          # every dir we may have
#                                          # installed into
#
# By default this is a "merge" install: it copies the skill folders
# next to whatever is already in your agent's skills directory; it
# does not delete other skills you have installed.

set -euo pipefail

cd "$(dirname "$0")/.."
SRC="plugins/intel-model-skillpack/skills"

agents=(claude codex opencode copilot qwen kimi hermes openclaw gemini cursor)
declare -A path
path[claude]="$HOME/.claude/skills"
path[codex]="${CODEX_HOME:-$HOME/.codex}/skills"
path[opencode]="$HOME/.config/opencode/skills"
path[copilot]="$HOME/.copilot/skills"
path[qwen]="$HOME/.qwen/skills"
path[kimi]="$HOME/.kimi/skills"
path[hermes]="$HOME/.hermes/skills"
path[openclaw]="$HOME/.openclaw/skills"
path[gemini]=""        # installed via `gemini extensions install`
path[cursor]="$HOME/.cursor/skills"

mode=existing
only=
do_uninstall=0
while [ $# -gt 0 ]; do
    case "$1" in
        --all)        mode=all ;;
        --agent)
            if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
                echo "--agent requires one of: ${agents[*]}" >&2
                exit 2
            fi
            only="$2"
            shift
            ;;
        --uninstall)  do_uninstall=1 ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        *)            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

names=()
for d in "$SRC"/*/; do names+=("$(basename "$d")"); done
backup_root=
backup_count=0

ensure_backup_root() {
    if [ -n "$backup_root" ]; then return; fi
    mkdir -p "$HOME/.cache/intel-model-skillpack/install-backups"
    backup_root=$(mktemp -d "$HOME/.cache/intel-model-skillpack/install-backups/install.XXXXXX")
}

backup_existing() {
    local agent="$1" dir="$2" n="$3"
    if [ ! -d "$dir/$n" ]; then return; fi
    ensure_backup_root
    local bdir="$backup_root/$agent"
    mkdir -p "$bdir"
    mv "$dir/$n" "$bdir/$n"
    backup_count=$((backup_count+1))
}

print_backup_note() {
    if [ "$backup_count" -gt 0 ]; then
        printf 'Backed up %d existing skill dirs to %s\n' "$backup_count" "$backup_root"
    fi
}

install_dir() {
    local agent="$1" dir="$2"
    if [ -z "$dir" ]; then return; fi
    if [ "$mode" != all ] && [ ! -d "$dir" ]; then
        printf '  %-7s skip (%s does not exist; pass --all to create)\n' "$agent" "$dir"
        return
    fi
    mkdir -p "$dir"
    for n in "${names[@]}"; do
        backup_existing "$agent" "$dir" "$n"
        cp -r "$SRC/$n" "$dir/$n"
    done
    printf '  %-7s installed %d skills into %s\n' "$agent" "${#names[@]}" "$dir"
}

uninstall_dir() {
    local agent="$1" dir="$2"
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then return; fi
    local removed=0
    for n in "${names[@]}"; do
        if [ -d "$dir/$n" ]; then
            backup_existing "$agent" "$dir" "$n"
            removed=$((removed+1))
        fi
    done
    printf '  %-7s removed %d skills from %s\n' "$agent" "$removed" "$dir"
}

if [ "$do_uninstall" = 1 ]; then
    echo "Uninstalling intel-model-skillpack:"
    for a in "${agents[@]}"; do uninstall_dir "$a" "${path[$a]}"; done
    print_backup_note
    exit 0
fi

if [ -n "$only" ]; then
    case "$only" in
        gemini)
            command -v gemini >/dev/null || { echo "gemini CLI not on PATH" >&2; exit 1; }
            gemini extensions install . --consent
            print_backup_note
            ;;
        claude|codex|opencode|copilot|qwen|kimi|hermes|openclaw|cursor)
            echo "Installing intel-model-skillpack for $only:"
            install_dir "$only" "${path[$only]}"
            print_backup_note
            ;;
        *)
            echo "unknown agent: $only" >&2
            echo "valid: ${agents[*]}" >&2
            exit 2
            ;;
    esac
    exit 0
fi

echo "Installing intel-model-skillpack into every detected agent:"
for a in claude codex opencode copilot qwen kimi hermes openclaw cursor; do
    install_dir "$a" "${path[$a]}"
done

if command -v gemini >/dev/null 2>&1; then
    printf '  gemini  installing extension...\n'
    gemini extensions install . --consent || true
else
    printf '  gemini  skip (gemini CLI not on PATH)\n'
fi

echo
print_backup_note
echo "Done. Verify by listing your agent's skills directory, e.g.:"
echo "  ls ~/.claude/skills | grep -E 'xpu-|model-can-it-fit'"
