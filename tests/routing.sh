#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Routing-determinism static checks for the xpu-port <->
# cuda-to-xpu-migration boundary (one-directional: assess -> execute).
#
# These greps pin the load-bearing prose facts in the SHIPPED files —
# not a self-contained routing table — so any drift that could reopen
# the routing loop fails the gate. Spec:
# "xpu-migration routing redesign, Option C-lite" (PR #86).
#
# Run from the repo root:
#     bash tests/routing.sh

set -euo pipefail
cd "$(dirname "$0")/.."

XPU_PORT="plugins/intel-model-skillpack/skills/xpu-port/SKILL.md"
MIGRATION="plugins/intel-model-skillpack/skills/cuda-to-xpu-migration/SKILL.md"

pass=0
fail=0
ok()  { printf '  [OK]   %s\n' "$1"; pass=$((pass+1)); }
err() { printf '  [FAIL] %s\n' "$1"; fail=$((fail+1)); }

# The description: frontmatter line of a SKILL.md (single line by
# repo convention; static.sh section 2 already enforces presence).
desc_line() {
    awk '/^description:/{print; exit}' "$1"
}

# 1. xpu-port's description CLAIMS the "port" verb — any request that
#    says "port" routes to the executor directly (the model selects
#    xpu-port on this phrasing regardless). A mis-scoped repo is caught
#    at runtime by xpu-port's own Backstop + empty-scan advisory, not by
#    the description.
if desc_line "$XPU_PORT" | grep -q "port my repo"; then
    ok "xpu-port description claims the port verb"
else
    err "xpu-port description no longer claims the port verb ('port my repo')"
fi

# 2. xpu-port's description disclaims the bare workflow "migrate" verb and
#    forward-routes it to the assessment skill.
if desc_line "$XPU_PORT" | grep -q "migrate my repo"; then
    ok "xpu-port description disclaims the migrate verb"
else
    err "xpu-port description missing workflow disclaimer ('migrate my repo')"
fi

# 3. cuda-to-xpu-migration's description claims the migrate verb but NOT
#    the port verb — "port" belongs to xpu-port (check 1).
if desc_line "$MIGRATION" | grep -q "asks to migrate it to Intel"; then
    ok "migration description claims the migrate verb"
else
    err "migration description does not claim the migrate verb"
fi
# 3a. ...and does NOT claim the port verb.
if desc_line "$MIGRATION" | grep -q "port my repo"; then
    err "migration description still claims the port verb ('port my repo') — belongs to xpu-port"
else
    ok "migration description free of the port verb"
fi
# 3b. ...and the convert/move workflow phrasings (spec §5 Place 1 names
#     them alongside migrate; without this, "convert this to XPU" routes
#     on semantic luck rather than a pinned trigger).
if desc_line "$MIGRATION" | grep -q "convert this to XPU"; then
    ok "migration description claims the convert verb"
else
    err "migration description does not claim the convert verb"
fi

# 4. The two-way tiebreaker is gone: no Exception paragraph in the
#    migration skill (it existed only to break the loop this redesign
#    removes at the root, and it must not resurface).
if grep -q "Exception — multi-surface" "$MIGRATION"; then
    err "migration SKILL.md still carries the 'Exception — multi-surface' tiebreaker"
else
    ok "migration SKILL.md has no reciprocal-gate exception"
fi

# 5. The migration skill never bounces a request away before the report
#    exists (the old Stop-first gate said "this skill does not handle it").
if grep -q "this skill does not handle it" "$MIGRATION"; then
    err "migration SKILL.md still carries a pre-report execute-verb bounce"
else
    ok "migration SKILL.md never bounces before the report"
fi

# 6. Exactly one outbound gate exists and it lives in xpu-port: the
#    Backstop section replaces the old bidirectional "When to assess
#    first" guidance.
if grep -q "^## Backstop" "$XPU_PORT"; then
    ok "xpu-port carries the single outbound backstop"
else
    err "xpu-port SKILL.md missing its '## Backstop' section"
fi
if grep -q "When to assess first" "$XPU_PORT"; then
    err "xpu-port SKILL.md still carries the old 'When to assess first' section"
else
    ok "xpu-port old bidirectional section removed"
fi

# 7. The backstop redirect is guarded on plan-in-hand: without this
#    guard, a multi-surface repo routed here BY a migration report
#    would bounce straight back (the loop, rebuilt).
if grep -q "no migration report is in hand" "$XPU_PORT"; then
    ok "xpu-port backstop carries the plan-in-hand guard"
else
    err "xpu-port backstop missing the plan-in-hand guard"
fi

# 8. Final-scan guidance must not unconditionally route back to the
#    assessment skill. That would contradict the plan-in-hand guard and
#    reopen the assess -> execute -> assess loop after a report already
#    scoped the Python port.
if awk '
    /^### 5\. Final scan/ { in_final = 1 }
    /^### 6\. Verify/ { in_final = 0 }
    in_final && /hand those files to/ { found = 1 }
    END { exit found ? 0 : 1 }
' "$XPU_PORT"; then
    err "xpu-port final-scan guidance unconditionally routes back to assessment"
else
    ok "xpu-port final-scan guidance respects plan-in-hand routing"
fi

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
