#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Static validation for the skill pack.
#
# Exits non-zero on the first failure. Run from the repo root:
#     bash tests/static.sh

set -euo pipefail

cd "$(dirname "$0")/.."

fail=0
pycache_root=$(mktemp -d)
# Scratch for the grep-then-report checks below. mktemp, not a fixed /tmp name:
# a predictable path is one another account on the host can pre-create as a
# symlink, so the redirect would clobber whatever that link points at with the
# runner's own privileges. mktemp -d also gives us 0700, so the captured hits
# (which include repo paths and matched lines) are not world-readable.
scratch=$(mktemp -d)
trap 'rm -rf "$pycache_root" "$scratch"' EXIT
note() { printf '  %s\n' "$1"; }
ok()   { printf '\033[32mOK\033[0m   %s\n' "$1"; }
err()  { printf '\033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

# Some gates are development-tree tooling and are deleted from the published
# artifact, so this script has to run against two different file sets. Guarding
# each call with a plain [ -f ] would do that, but it also turns deleting a gate
# into a silent pass -- the fail-open shape this suite exists to prevent.
#
# So distinguish "absent by design" from "vanished". The internal buckets
# (evaluation/, health-reports/, tickets/) are what the release deletions
# remove and are never part of a published tree, so whether they are TRACKED
# says which file set we are in. Untracked-and-absent is also the right answer
# for an extracted tarball with no .git: no repo, no internal tooling.
internal_tree=0
for marker in evaluation health-reports tickets; do
    if [ -n "$(git ls-files -- "$marker" 2>/dev/null | head -1)" ]; then
        internal_tree=1
        break
    fi
done

# Run a gate that only the internal tree carries. Missing is fine there and only
# there; missing on the internal tree is a failure, not a skip.
run_internal_gate() {
    local path="$1" label="$2"
    shift 2
    if [ -e "$path" ]; then
        if "$@"; then
            ok "$label"
        else
            err "$label failed"
        fi
    elif [ "$internal_tree" = 1 ]; then
        err "$label is missing from a tree that still tracks the internal buckets; an internal gate cannot just disappear"
    else
        note "$label not present in the published artifact (skipped)"
    fi
}

# 1. JSON manifests parse.
printf '\n== JSON manifests ==\n'
for f in .claude-plugin/plugin.json \
         .claude-plugin/marketplace.json \
         .cursor-plugin/plugin.json \
         gemini-extension.json \
         configs/hermes/taps.json \
         configs/openclaw/openclaw.json; do
    if [ -f "$f" ]; then
        if python3 -c "import json,sys; json.load(open('$f'))" 2>/dev/null; then
            ok "$f"
        else
            err "$f does not parse as JSON"
        fi
    else
        note "$f not present (skipped)"
    fi
done

# 2. SKILL.md frontmatter is valid (name + description present, name
#    matches directory).
printf '\n== SKILL.md frontmatter ==\n'
for d in plugins/intel-model-skillpack/skills/*/; do
    name=$(basename "$d")
    sk="$d/SKILL.md"
    if [ ! -f "$sk" ]; then
        err "$d missing SKILL.md"; continue
    fi
    fm_name=$(awk '/^name:/{print $2; exit}' "$sk")
    has_desc=$(awk '/^description:/{print "y"; exit}' "$sk")
    if [ "$fm_name" != "$name" ]; then
        err "$sk frontmatter name '$fm_name' != dir '$name'"
        continue
    fi
    if [ "$has_desc" != "y" ]; then
        err "$sk missing description"
        continue
    fi
    ok "$name"
done

# 3. Python helpers compile.
printf '\n== Python helpers compile ==\n'
while IFS= read -r f; do
    if PYTHONPYCACHEPREFIX="$pycache_root" python3 -m py_compile "$f" 2>/dev/null; then
        ok "$f"
    else
        err "$f does not py_compile"
    fi
done < <(find plugins scripts -name '*.py' -not -path '*/__pycache__/*' 2>/dev/null)

# 4. Shell helpers parse.
printf '\n== Shell helpers parse ==\n'
while IFS= read -r f; do
    if bash -n "$f" 2>/dev/null; then
        ok "$f"
    else
        err "$f does not parse"
    fi
done < <(find scripts tests plugins/intel-model-skillpack/skills -name '*.sh' 2>/dev/null)

# 4a. Mocked integration checks.
printf '\n== Mocked integration checks ==\n'
if tests/xpu-runtime-preflight.sh; then
    ok "tests/xpu-runtime-preflight.sh"
else
    err "tests/xpu-runtime-preflight.sh failed"
fi
if tests/model-fit.sh; then
    ok "tests/model-fit.sh"
else
    err "tests/model-fit.sh failed"
fi
if tests/xpu-port.sh; then
    ok "tests/xpu-port.sh"
else
    err "tests/xpu-port.sh failed"
fi
if tests/routing.sh; then
    ok "tests/routing.sh"
else
    err "tests/routing.sh failed"
fi
# secure-config.sh enumerates tracked files via `git ls-files` and refuses to
# run outside a git worktree (an extracted tree can't distinguish "clean" from
# "nothing scanned" -- see its own header comment). static.sh already treats a
# missing .git the same way it treats an absent internal/ bucket: skip rather
# than fail, since neither can be evaluated for a tree that isn't a git clone.
if git rev-parse --git-dir >/dev/null 2>&1; then
    run_internal_gate tests/secure-config.sh "tests/secure-config.sh" tests/secure-config.sh
else
    note "tests/secure-config.sh not present in a git worktree (skipped)"
fi

# The bandit CI step names its target directories explicitly rather than using
# `-r .`, because `.` also sweeps up any in-tree virtualenv. The cost of being
# explicit is that a new top-level directory of Python would go unscanned and
# nothing would say so. Compare the two sets so it cannot happen quietly.
#
# Expected set = top-level dirs holding tracked Python, MINUS the single-segment
# dirs .bandit excludes on purpose (`tests`). Deriving the exclusion from the
# config rather than hardcoding it means the two files cannot disagree.
printf '\n== Bandit CI coverage ==\n'
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    note "skipped (not a git worktree; derives tracked Python roots via git ls-files)"
elif [ -f .github/workflows/static.yml ] && [ -f .bandit ]; then
    bandit_targets=$(grep -oE 'bandit --ini \.bandit -r [a-z ]+' .github/workflows/static.yml |
        head -1 | sed 's/.*-r //' | tr ' ' '\n' | sed '/^$/d' | sort -u)
    bandit_excluded=$(grep -E '^exclude:' .bandit | sed 's/^exclude:[[:space:]]*//' |
        tr ',' '\n' | sed 's#^[[:space:]]*/##; s#[[:space:]]*$##' |
        grep -E '^[A-Za-z0-9._-]+$' | sort -u)
    py_roots=$(git ls-files '*.py' | awk -F/ 'NF>1{print $1}' | sort -u)
    expected=$(comm -23 <(printf '%s\n' "$py_roots") <(printf '%s\n' "$bandit_excluded"))
    missing=$(comm -23 <(printf '%s\n' "$expected") <(printf '%s\n' "$bandit_targets"))
    extra=$(comm -13 <(printf '%s\n' "$expected") <(printf '%s\n' "$bandit_targets"))
    if [ -z "$missing" ] && [ -z "$extra" ]; then
        ok "bandit CI targets == tracked Python roots minus .bandit exclusions ($(printf '%s' "$bandit_targets" | tr '\n' ' '))"
    else
        [ -n "$missing" ] && err "bandit CI step does not scan: $(printf '%s' "$missing" | tr '\n' ' ')"
        # An extra target is not cosmetic: naming an excluded dir explicitly
        # overrides the ini exclusion and re-admits its findings.
        [ -n "$extra" ] && err "bandit CI targets a dir .bandit excludes (this defeats the exclusion): $(printf '%s' "$extra" | tr '\n' ' ')"
    fi
fi

# 4b. Internal research Python files are not published, but if we keep
#     executable scratch helpers around they should at least parse.
printf '\n== Research Python helpers compile ==\n'
found_research_py=0
while IFS= read -r f; do
    found_research_py=1
    if PYTHONPYCACHEPREFIX="$pycache_root" python3 -m py_compile "$f" 2>/dev/null; then
        ok "$f"
    else
        err "$f does not py_compile"
    fi
done < <(find research -name '*.py' \
    -not -path '*/__pycache__/*' \
    -not -path '*/.venv/*' \
    -not -path '*/.pytest_cache/*' \
    -not -path '*/*.egg-info/*' 2>/dev/null)
if [ "$found_research_py" = 0 ]; then
    note "no research Python files"
fi

# 5. agents/AGENTS.md is up to date with skills.
printf '\n== agents/AGENTS.md is current ==\n'
if [ -x scripts/generate_agents.py ] || [ -f scripts/generate_agents.py ]; then
    tmp=$(mktemp)
    cp agents/AGENTS.md "$tmp" 2>/dev/null || true
    python3 scripts/generate_agents.py >/dev/null
    if diff -q "$tmp" agents/AGENTS.md >/dev/null 2>&1; then
        ok "agents/AGENTS.md matches generator output"
    else
        err "agents/AGENTS.md drifted; run python3 scripts/generate_agents.py"
        cp "$tmp" agents/AGENTS.md 2>/dev/null || true
    fi
    rm -f "$tmp"
fi

# 5a. catalog/bundles.yaml covers every skill.
printf '\n== catalog/bundles.yaml covers every skill ==\n'
if [ -f catalog/bundles.yaml ]; then
    missing=$(python3 -c "
import os, re, sys
disk = sorted(d for d in os.listdir('plugins/intel-model-skillpack/skills')
              if os.path.isdir('plugins/intel-model-skillpack/skills/' + d))
refs = set(re.findall(r'^\s*-\s+([a-z][a-z0-9-]*)\s*\$',
                      open('catalog/bundles.yaml').read(), re.M))
print(','.join(s for s in disk if s not in refs))
")
    if [ -z "$missing" ]; then
        ok "every skill appears in at least one bundle"
    else
        err "skills missing from catalog/bundles.yaml: $missing"
    fi
else
    err "catalog/bundles.yaml not found"
fi

# 6. marketplace.json plugin count matches skills/ directory count.
printf '\n== marketplace.json covers every skill ==\n'
mk_count=$(python3 -c "import json; print(len(json.load(open('.claude-plugin/marketplace.json'))['plugins']))")
sk_count=$(find plugins/intel-model-skillpack/skills -mindepth 1 -maxdepth 1 -type d | awk 'END{print NR}')
if [ "$mk_count" = "$sk_count" ]; then
    ok "$mk_count plugins in marketplace.json, $sk_count skill dirs"
else
    err "$mk_count plugins in marketplace.json but $sk_count skill dirs"
fi

# 7. Published docs should not point at internal scratchpads.
printf '\n== Published docs are self-contained ==\n'
if rg -n 'research/|SCRATCH\.md' README.md HOW_TO_TEST.md agents plugins/intel-model-skillpack/skills >"$scratch/docs" 2>/dev/null; then
    err "published docs reference internal research paths"
    sed 's/^/  /' "$scratch/docs"
else
    ok "no internal research links in published docs"
fi

# 8. No destructive process/container cleanup in shipped scripts.
printf '\n== No destructive cleanup patterns ==\n'
if rg -n 'pkill|docker rm -f|rm -rf' scripts plugins/intel-model-skillpack/skills >"$scratch/destructive" 2>/dev/null; then
    err "destructive cleanup pattern found"
    sed 's/^/  /' "$scratch/destructive"
else
    ok "no pkill, docker rm -f, or rm -rf"
fi

# 9. No documented commands that advertise unimplemented TODO behavior.
printf '\n== No TODO command promises ==\n'
if rg -n '`[^`]*(--table|--mode|--verify|--calibrate)[^`]*`.*TODO|TODO: implement' README.md HOW_TO_TEST.md plugins/intel-model-skillpack/skills >"$scratch/todo" 2>/dev/null; then
    err "docs advertise unimplemented command behavior"
    sed 's/^/  /' "$scratch/todo"
else
    ok "no TODO command promises"
fi

# 10. Generated AGENTS.md descriptions should be bullets, not inline-code
# wrapped blobs that break on embedded backticks.
printf '\n== agents/AGENTS.md markdown shape ==\n'
if awk '/<available_skills>/{inside=1; next} /<\/available_skills>/{inside=0} inside && /^[^-[:space:]]/ {bad=1} END{exit bad}' agents/AGENTS.md; then
    ok "available skills are bullet formatted"
else
    err "agents/AGENTS.md has malformed available_skills entries"
fi
if rg -n '^- [^:]+: "' agents/AGENTS.md >"$scratch/agents-quotes" 2>/dev/null; then
    err "agents/AGENTS.md leaked YAML quote wrappers"
    sed 's/^/  /' "$scratch/agents-quotes"
else
    ok "available skill descriptions are unquoted"
fi

# 11. Guardrails static checks. guardrails/ is development-tree tooling and is
#     not part of the published artifact, so this is an internal gate.
printf '\n== Guardrails (static) ==\n'
run_internal_gate guardrails/check.py "guardrails/check.py" \
    python3 guardrails/check.py

printf '\n'
if [ "$fail" = 0 ]; then
    printf '\033[32mAll checks passed.\033[0m\n'
    exit 0
else
    printf '\033[31mOne or more checks failed.\033[0m\n'
    exit 1
fi
