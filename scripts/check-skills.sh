#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Required validation gate before adding or changing skills.

set -euo pipefail

cd "$(dirname "$0")/.."

# static.sh runs guardrails/check.py itself (and reports a missing guardrails/
# correctly for the tree it is running against), so invoking it again here was a
# duplicate run -- and an unguarded one, which failed outright on any tree that
# does not carry guardrails/.
bash tests/static.sh

if ! command -v skill-validator >/dev/null 2>&1; then
    if [ "${REQUIRE_SKILL_VALIDATOR:-0}" = 1 ]; then
        echo "skill-validator is required but is not installed" >&2
        echo "install: https://github.com/agent-ecosystem/skill-validator" >&2
        exit 1
    fi
    echo "skill-validator not found; skipped external Agent Skills validation"
    echo "install: https://github.com/agent-ecosystem/skill-validator"
    exit 0
fi

for d in plugins/intel-gpu-ai-skills/skills/*/; do
    skill-validator check --strict --skip links --skip-orphans --allow-dirs=data "$d"
done
