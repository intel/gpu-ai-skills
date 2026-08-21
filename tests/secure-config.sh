#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Secure-configuration checks: asserts the pack's hardened defaults have not
# drifted.
#
# One section per requirement. Every assertion is static or runs a
# shipped script offline -- no GPU, no container, no network -- so the gate is
# the same on a developer box and in CI.
#
# Wired into tests/static.sh. Pass -v for verbose output.

set -u

cd "$(dirname "$0")/.." || {
    printf 'secure config: FATAL -- cannot cd to the repo root\n' >&2
    exit 2
}

# Every tracked-tree assertion below enumerates files with `git ls-files`. Outside
# a git worktree that call fails, and a failed enumeration is indistinguishable
# from "no matches" -- the gate would print [OK] for rules that never ran. That
# matters because .gitattributes ships this pack as an archive (export-ignore),
# so an archive consumer running the gate must be told it cannot work here rather
# than shown a green run. Refuse to start instead.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    printf 'secure config: FATAL -- not a git worktree.\n' >&2
    printf '  This gate enumerates tracked files with `git ls-files`; outside a\n' >&2
    printf '  worktree it cannot distinguish "clean" from "nothing scanned".\n' >&2
    printf '  Run it from a git clone.\n' >&2
    exit 2
fi

VERBOSE=0
[ "${1:-}" = "-v" ] && VERBOSE=1

SKILL="plugins/intel-model-skillpack/skills"
LAUNCH="$SKILL/vllm-xpu-run/scripts/emit_launch.sh"

pass=0
fail=0
note() { [ "$VERBOSE" = 1 ] && printf '  %s\n' "$1"; return 0; }
ok()   { printf '  [OK]   %s\n' "$1"; pass=$((pass + 1)); }
err()  { printf '  [FAIL] %s\n' "$1"; fail=$((fail + 1)); }

# Assert a pattern has no matches in the tracked tree. Uses git ls-files so
# untracked local scratch (evidence dirs, venvs) cannot fail the gate.
#
# This file is excluded from its own search. The patterns below are literal
# strings in a tracked *.sh file, so without the exclusion the gate reports
# every rule it enforces as a violation of itself.
#
# A pathspec that matches nothing is a FAILURE, not a pass: a typo, a renamed
# directory, or a file that stopped being tracked would otherwise turn the rule
# into a silent no-op that still prints [OK].
assert_absent() {
    local label="$1" pattern="$2"
    shift 2
    local -a files=()
    mapfile -t -d '' files < <(git ls-files -z -- "$@" ':!tests/secure-config.sh')
    if [ "${#files[@]}" -eq 0 ]; then
        err "$label -- pathspec matched 0 tracked files; the rule did not run"
        return 0
    fi
    note "$label: scanned ${#files[@]} tracked file(s)"
    local hits
    hits=$(grep -nE -- "$pattern" "${files[@]}" 2>/dev/null)
    if [ -z "$hits" ]; then
        ok "$label"
    else
        err "$label"
        printf '%s\n' "$hits" | sed 's/^/         /'
    fi
}

# ---------------------------------------------------------------------------
# TC1 -- 3rd party software and services are securely configured: restrictive
#        permissions, no unnecessary features, debug off in the release.
# ---------------------------------------------------------------------------
printf '\n== TC1: restrictive permissions, no debug in the release ==\n'

# Scoped to tracked files, like every other rule here: an untracked .venv or
# artifacts/ directory cannot ship, so letting it fail the gate would make a
# developer run disagree with CI for a file that is not part of the deliverable.
assert_perm_absent() {
    local label="$1" perm="$2"
    local -a files=()
    mapfile -t -d '' files < <(git ls-files -z)
    if [ "${#files[@]}" -eq 0 ]; then
        err "$label -- git ls-files returned nothing; the rule did not run"
        return 0
    fi
    local hits
    hits=$(find "${files[@]}" -maxdepth 0 -type f -perm "$perm" -print 2>/dev/null)
    if [ -z "$hits" ]; then
        ok "0 $label"
    else
        err "$(printf '%s\n' "$hits" | wc -l) $label"
        printf '%s\n' "$hits" | sed 's/^/         /'
    fi
}

assert_perm_absent "world-writable tracked files" -0002
assert_perm_absent "setuid/setgid tracked files" /6000

assert_absent "0 shell=True subprocess calls" 'shell=True' '*.py'
assert_absent "no permission widening at runtime" 'os\.chmod|chmod 777|umask ' '*.py' '*.sh'

# `FOO=1`, `FOO="1"` and `FOO='1'` are the same setting to the shell, so a rule
# that matches only the bare form is bypassed by adding a quote. Every
# value-matching pattern below allows an optional opening quote.
Q='["'"'"']?'

# Debug and trace switches are documented in knob tables, which is fine; what
# must not ship is a default that turns one on.
assert_absent "no debug/trace switch enabled by default" \
    "^[[:space:]]*(export[[:space:]]+)?(GGML_SYCL_DEBUG|PYTORCH_ENABLE_XPU_FALLBACK)=${Q}1|^[[:space:]]*(export[[:space:]]+)?(VLLM_LOGGING_LEVEL|SGLANG_LOGGING_LEVEL|KINETO_LOG_LEVEL)=${Q}DEBUG" \
    '*.sh' '*.py' '.env.example'

assert_absent "no development/test mode enabled by default" \
    "^[[:space:]]*export[[:space:]]+(APP_ENV=${Q}development|Testing=${Q}TRUE)" '.env.example'

# ---------------------------------------------------------------------------
# TC2 -- the software supports secure-configuration best practices: secure
#        defaults, less-secure features require customer action.
# ---------------------------------------------------------------------------
printf '\n== TC2: secure defaults, opt-in for less-secure features ==\n'

# The one flag that grants arbitrary code execution must default to off and
# must not appear in any copy-paste launch invocation.
if grep -q '^trust_remote_code=0' "$LAUNCH"; then
    ok "vLLM launcher defaults trust_remote_code=0"
else
    err "vLLM launcher no longer defaults trust_remote_code=0"
fi

# Documenting the flag is how a user learns it exists, so a backticked mention
# in prose or a knob table is allowed. What must not ship is the flag as a live
# command-line token anywhere a reader can copy it -- on its own continuation
# line, mid-invocation, or inside a quoted shell string.
#
# Matching an *unbackticked* occurrence covers all three and still permits the
# documented form. The scan spans every tracked text artifact, not just *.md;
# the two files that implement the opt-in are excluded here and asserted
# functionally below instead, and evaluation/ must contain the literal string
# because its assertions and fixtures test for exactly this.
TRC_INVOCATION='(^|[^`])--trust-remote-code([^`]|$)'
assert_absent "no shipped artifact carries --trust-remote-code as a live flag" \
    "$TRC_INVOCATION" \
    '*.md' '*.sh' '*.py' '*.txt' '*.yaml' '*.yml' '*.json' \
    ":!$LAUNCH" \
    ":!$SKILL/model-config-recommend/scripts/recommend.py" \
    ':!evaluation/'

# Functional check: the emitted command must not carry the flag unless asked.
if out=$(bash "$LAUNCH" --model Qwen/Qwen2.5-1.5B-Instruct 2>&1); then
    if printf '%s' "$out" | grep -q -- '--trust-remote-code'; then
        err "emit_launch.sh emits --trust-remote-code without being asked"
    else
        ok "emit_launch.sh omits --trust-remote-code by default"
    fi
    # Positive control: the opt-in must still work, or the check above is vacuous.
    if bash "$LAUNCH" --model Qwen/Qwen2.5-1.5B-Instruct --trust-remote-code 2>&1 |
            grep -q -- '--trust-remote-code'; then
        ok "emit_launch.sh honors the --trust-remote-code opt-in"
    else
        err "emit_launch.sh ignores the --trust-remote-code opt-in"
    fi
else
    err "emit_launch.sh failed to run: $out"
fi

# The recommender used to add --trust-remote-code whenever the *model's own*
# config.json declared auto_map. That let the untrusted artifact decide a
# code-execution opt-in, which is the inverse of the TC2 requirement. Exercise
# both paths against a synthetic auto_map config so a regression is caught
# behaviourally, not just by the text scan above.
RECOMMEND="$SKILL/model-config-recommend/scripts/recommend.py"
trc_tmp=$(mktemp -d) || trc_tmp=""
if [ -n "$trc_tmp" ]; then
    cat > "$trc_tmp/config.json" <<'JSON'
{
  "architectures": ["LlamaForCausalLM"],
  "auto_map": {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"},
  "hidden_size": 2048, "intermediate_size": 5632,
  "num_attention_heads": 16, "num_hidden_layers": 22,
  "num_key_value_heads": 4, "vocab_size": 32000,
  "max_position_embeddings": 4096, "torch_dtype": "bfloat16",
  "model_type": "llama"
}
JSON
    rec_args=(--model "$trc_tmp/config.json" --device arc-pro-b70 --no-hub-search)
    if out=$(python3 "$RECOMMEND" "${rec_args[@]}" 2>&1); then
        # Strip comment lines before looking for the flag: the security note
        # names the flag on purpose, and only a live flag is a finding.
        live=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*#')
        if printf '%s' "$live" | grep -q -- '--trust-remote-code'; then
            err "recommend.py emits --trust-remote-code from the model's own auto_map"
        else
            ok "recommend.py omits --trust-remote-code for an auto_map model by default"
        fi
        if printf '%s' "$out" | grep -qi 'SECURITY'; then
            ok "recommend.py states why the flag was withheld"
        else
            err "recommend.py withholds the flag silently, with no stated reason"
        fi
        # Positive control: the explicit opt-in must still work.
        if out=$(python3 "$RECOMMEND" "${rec_args[@]}" --trust-remote-code 2>&1); then
            live=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*#')
            if printf '%s' "$live" | grep -q -- '--trust-remote-code'; then
                ok "recommend.py honors the explicit --trust-remote-code opt-in"
            else
                err "recommend.py ignores the explicit --trust-remote-code opt-in"
            fi
        else
            err "recommend.py failed with --trust-remote-code: $out"
        fi
        # Granting code execution against an unpinned Hub repo means the engine
        # runs the publisher's latest push, not the code the user reviewed -- so
        # the opt-in must be refused without a revision. Asserted on the
        # argument parsing alone (a Hub id that never resolves), so the check
        # needs no network and cannot pass by failing for some other reason.
        out=$(python3 "$RECOMMEND" --model no-such-org/no-such-model \
            --device arc-pro-b70 --no-hub-search --trust-remote-code 2>&1)
        rc=$?
        if [ "$rc" -eq 2 ] && printf '%s' "$out" | grep -q -- '--trust-remote-code requires --revision'; then
            ok "recommend.py refuses --trust-remote-code without --revision for a Hub model"
        else
            err "recommend.py accepts --trust-remote-code with no revision pin (rc=$rc)"
        fi
    else
        err "recommend.py failed to run: $out"
    fi
    rm -rf "$trc_tmp"
else
    err "could not create a temp dir for the recommend.py opt-in check"
fi

# The emitted servers are unauthenticated and bind every host interface. That is
# a deliberate disposition, not a defect -- but it is only a disposition while it
# is written down somewhere the release carries. The guidance used to live in
# .env.example; removing that block left the finding with no trace in any tracked
# file, and `git grep -- '--api-key'` is 0 hits pack-wide, so a text scan cannot
# rediscover it. Pin the statement to SECURITY.md, which ships, so deleting it
# fails the gate instead of silently dropping the disposition.
if [ -f SECURITY.md ]; then
    sdl304_missing=""
    for phrase in 'unauthenticated and reachable' '127\.0\.0\.1' 'reverse proxy'; do
        grep -qE -- "$phrase" SECURITY.md || sdl304_missing="$sdl304_missing $phrase"
    done
    if [ -z "$sdl304_missing" ]; then
        ok "SECURITY.md states the unauthenticated-endpoint exposure and its mitigation"
    else
        err "SECURITY.md no longer states the unauthenticated-endpoint disposition; missing:$sdl304_missing"
    fi
else
    err "SECURITY.md is missing; the unauthenticated-endpoint disposition has no home"
fi

# Credentials are referenced by variable, never carried as a value. Obvious
# placeholders are the point of an example file, so they are allowed; what must
# not appear is a real-looking secret or an inline URL credential.
#
# Scanned over the whole tracked tree, not just .env.example: a leaked token is
# a leak wherever it lands, and the likelier places are a pasted transcript
# under results/, a curl example in a SKILL.md, or a ticket note -- none of
# which the old .env.example-only rule looked at. -I skips binaries.
CRED_SHAPES='hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(Bearer|Authorization:)[[:space:]]+[A-Za-z0-9._-]{25,}|://[A-Za-z0-9_.-]+:[^@/"[:space:]]+@'
CRED_PLACEHOLDER='your-|not-needed|changeme|CHANGE_?ME|example|dummy|placeholder|redacted|<[^>]+>|\$\{?[A-Z_]+'

# Self-test the pattern before trusting a clean result. A regex that silently
# stopped matching would otherwise report [OK] over the entire repository --
# the same fail-open shape as an empty pathspec.
if printf 'token=hf_%s\nurl=https://user:s3cr3t@host/\n' 'AbCdEfGhIjKlMnOpQrStUv' |
        grep -qE -- "$CRED_SHAPES"; then
    ok "credential pattern still matches a known-bad sample (rule is live)"
else
    err "credential pattern matched nothing in its own positive control"
fi

cred_files=()
mapfile -t -d '' cred_files < <(git ls-files -z -- ':!tests/secure-config.sh')
if [ "${#cred_files[@]}" -eq 0 ]; then
    err "credential scan enumerated 0 tracked files; the rule did not run"
else
    note "credential scan: ${#cred_files[@]} tracked file(s)"
    # -o so the allowlist is tested against the matched credential ALONE.
    # Applied to grep's full "file:line:text" output it also saw the path and
    # the surrounding prose, so a real token under an examples/ path -- or
    # merely on a line that happened to contain the word "dummy" -- was
    # discarded. -H keeps the filename even when one file is scanned.
    cred_hits=$(grep -nHoIE -- "$CRED_SHAPES" "${cred_files[@]}" 2>/dev/null |
        while IFS= read -r hit; do
            match=${hit#*:}        # drop "file:"
            match=${match#*:}      # drop "line:" -- what is left is the match
            printf '%s' "$match" | grep -qE -- "$CRED_PLACEHOLDER" ||
                printf '%s\n' "$hit"
        done || true)
    if [ -z "$cred_hits" ]; then
        ok "no credential value anywhere in the tracked tree"
    else
        err "credential-shaped value in a tracked file"
        printf '%s\n' "$cred_hits" | sed 's/^/         /'
    fi
fi

if git check-ignore -q .env; then
    ok ".env is gitignored, so a filled-in copy cannot be committed"
else
    err ".env is not gitignored"
fi

# ---------------------------------------------------------------------------
# TC3 -- compatibility settings are appropriate for the platform in use;
#        default paths and directories are correct and not shared.
# ---------------------------------------------------------------------------
printf '\n== TC3: default paths are per-user, not shared ==\n'

assert_absent "no fixed shared /tmp path as a default cache/output dir" \
    '(CACHE_DIR|OUTPUT_DIR|LOG_DIR|artifact_prefix)=["'"'"']?/tmp/' '*.sh' '*.py' '.env.example'

# Shell and Python helpers must allocate scratch with mktemp rather than a
# guessable fixed name another account can pre-create as a symlink.
#
# The old rule only caught `>` redirection and `touch`, and only looked under
# skills/ -- so it missed the repo's own tooling entirely. Widened on both axes:
#
#   contexts -- redirection, touch, mkdir, tee, `-o` (curl/wget), --output-file,
#               libFuzzer's --artifact_prefix, plain assignment, and a backticked
#               path in prose (docs telling a reader to use a fixed name are how
#               the fixed name spreads).
#   scope    -- the shipped skills plus scripts/ and tests/, i.e. everything the
#               gate can actually hold to the rule.
#
# The trailing `([^A-Za-z0-9._$-]|$)` is what keeps this usable: it requires the
# /tmp path to END at a non-path character, so a *parameterized* path such as
# /tmp/bench-${RUN_TAG}.jsonl does not match while a fixed /tmp/bench.jsonl does.
# Interpolating a per-run tag is the fix, so the rule must not punish it.
TMP_FIXED='(>[[:space:]]*|touch[[:space:]]+|mkdir[[:space:]]+(-p[[:space:]]+)?|tee[[:space:]]+(-a[[:space:]]+)?|-o[[:space:]]+|--output-file[[:space:]]+|--?artifact.prefix=|=["'"'"']?|`)/tmp/[A-Za-z0-9._-]+([^A-Za-z0-9._$-]|$)'
assert_absent "no guessable fixed temp path in skills, scripts, or tooling" \
    "$TMP_FIXED" \
    "$SKILL" 'scripts' 'tests' '*.md' \
    ':!evaluation/'

# ---------------------------------------------------------------------------
# TC4 -- legacy and unnecessary features are deactivated.
# ---------------------------------------------------------------------------
printf '\n== TC4: legacy stacks not installed or enabled ==\n'

# The pack deliberately targets upstream stacks only. These may be named in
# prose explaining the exclusion, but must never be an install target.
assert_absent "no legacy Intel stack is an install target" \
    '(pip|uv pip) install[^|]*(ipex-llm|intel-extension-for-pytorch|llm-scaler-vllm)' \
    '*.md' '*.sh'

assert_absent "silent CPU fallback is not enabled anywhere" \
    "PYTORCH_ENABLE_XPU_FALLBACK=${Q}1" '*.sh' '*.py' '.env.example'

# ---------------------------------------------------------------------------
printf '\n'
printf 'secure config: %d passed, %d failed\n' "$pass" "$fail"
if [ "$fail" -gt 0 ]; then
    exit 1
fi
exit 0
