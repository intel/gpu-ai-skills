#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Regenerate agents/AGENTS.md from SKILL.md frontmatter and re-stage it.
#
# Invoked by .pre-commit-config.yaml when a staged file matches
# plugins/intel-gpu-ai-skills/skills/*/SKILL.md. Safe to run manually
# from the repo root as well.

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

python3 scripts/generate_agents.py
git diff --quiet -- agents/AGENTS.md || git add agents/AGENTS.md
