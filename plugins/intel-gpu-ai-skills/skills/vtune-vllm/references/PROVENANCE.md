# Provenance of this skill's contents

Part of this skill is **vendored third-party content**. This file records what came from
upstream, what this pack changed, and why, as Apache-2.0 §4(b) asks for modified files.

## Vendored from vllm-xpu-profiling-toolkit (Apache-2.0, Copyright 2026 Vasudha)

- **Upstream:** https://github.com/Vasud-ha/vllm-xpu-profiling-toolkit
- **Source path:** `vtune/`
- **Commit vendored:** `4d93a4d0ba5a44727407b7af6c81ec31d5da356a` (2026-07-09)
- **License:** Apache-2.0 (upstream `LICENSE`; no upstream `NOTICE` file). Listed in this
  pack's `NOTICE` and `BOM.md`.

Copied **verbatim, byte-identical**:

```
references/itt-api.md            (upstream docs/itt-api.md)
references/vllm-integration.md   (upstream docs/vllm-integration.md)
scripts/vtune_itt.py
```

Copied and **modified** by this pack (see the tables below):

```
SKILL.md                            (upstream SKILL.md)
references/troubleshooting.md       (upstream docs/troubleshooting.md)
references/validation-and-flow.md   (upstream docs/validation-and-flow.txt, renamed)
scripts/run_vtune_vllm.sh
scripts/serve_with_vtune.py
references/commands-and-workflow.md (sections of upstream SKILL.md, moved out and edited)
```

**Original to this pack** (not from upstream):

```
references/PROVENANCE.md
scripts/sitecustomize.py
```

**Not copied** from upstream: `vtune/vtune-vllm-profiling-report.html` (a rendered report),
`pytorch-profiler/` and `unitrace/` (other profilers), and the toolkit's repository-level
material (`README.md`, `PREREQUISITES.md`, `scripts/check_prereqs.sh`,
`profiler_comparison*`, `direct_invocation.txt`, `verification_log_vllm_0.21.md`).

## Adaptations when the skill was first imported

| Change | Why |
|---|---|
| `docs/` → `references/`, `validation-and-flow.txt` → `.md` | Upstream's `SKILL.md` already pointed at `references/…`; the rename makes those links resolve and the file readable as Markdown |
| frontmatter `name: vtune-vllm-profiling` → `vtune-vllm` | The directory name must equal the frontmatter `name` |
| frontmatter `description` rewritten | Scopes the triggers to vLLM / LLM-server profiling so the skill does not compete with `vtune-gpu-hotspots` on generic kernel questions |
| `> **Scope.**` block after the H1 | States the boundary against `vtune-gpu-hotspots` / `vtune-xpu-offload`; the installed VTune's `-help` output wins over recorded knob and report names |
| Path strings in `SKILL.md`, `references/validation-and-flow.md`, `scripts/run_vtune_vllm.sh` | Point at `references/…` instead of the upstream file names |

## Changes after testing on vLLM 0.33 and VTune 2026.6 (2026-11-17)

An A/B test (the same agent with and without this skill, Opus / Sonnet / Haiku, Qwen3-8B
on one Arc Pro B70) found answers that went wrong because of gaps in this skill, and
launcher steps that do not work on vLLM 0.33 / VTune 2026.6. The changed skill was rerun on
the same questions; none of those wrong answers came back.

| File | Change | Why |
|---|---|---|
| `SKILL.md` | New §0 "Read first": default XPU graphs hide decode kernels; profiled server is slower; GPU busy = kernel time per step ÷ un-profiled TPOT; bytes ÷ time for the memory check | Two wrong answers in the test came from these gaps (a graph-mode kernel table read as "GPU 12.5% busy"; a profiled eager run compared with an un-profiled default run) |
| `SKILL.md` | §3: `-- vllm serve` fails, module / interpreter forms; §3.6 container recipe; knob table and §3.3 example updated for 2026.6 | `vllm` is a Python script; 6 of the listed knobs or values are rejected by 2026.6 |
| `SKILL.md` | §4.2 note: VTune 2026.6 ships no `libittnotify.so` | The ctypes fallback has nothing to load |
| `SKILL.md` | §8: the EngineCore is spawned, so the patch must be loaded in every process; the outer resume records every step | Checked: the parent-process patch never reached the EngineCore |
| `SKILL.md` | §9 graph-mode and bandwidth notes; §11.1 report names for 2026.6 | `gpu-hotspots`, `top-tasks` reports do not exist on 2026.6 |
| `SKILL.md` | §1, §3.1-3.5, §4, §5.1, §7, §10, §11.2 and the §11.1 design rules / hand commands moved to `references/commands-and-workflow.md` (§1-§8) with short pointers left | Keep `SKILL.md` under this pack's 500-line limit and the Agent Skills 5,000-token body; the A/B diff recipe now uses 2026.6 column names and sorts by time |
| `PROVENANCE.md` | Moved to `references/` | Agent Skills layout: no extra files at the skill root |
| `references/troubleshooting.md` | §11: vLLM 0.33 kernel names and measured ranges; IPEX-era red flags marked; VTune-only env fallback in the environment check | `xetla_*` names do not occur on vLLM 0.33; one `gemm_kernel` at 85% is healthy there |
| `references/validation-and-flow.md` | Check 3 and section F: vLLM 0.33 names, bandwidth from bytes ÷ time; VTune-only env fallback | Same |
| `scripts/run_vtune_vllm.sh` | Falls back to `vtune/latest/env/vars.sh`; exports `VTUNE_VLLM_PATCH=1` and puts `scripts/` on `PYTHONPATH`; reports via `hotspots -group-by computing-task` (sorted) and `-group-by task-domain,task`, comma CSV, stderr kept out of the files; verification reads the value column, fails without `hotspots.csv`, compares GPU Time with the benchmark window, warns when no `vllm.*` ITT tasks exist | Each fixes a failure seen on 2026.6 (preflight refused to start; reports failed; `Verification: PASS` with reports missing; a tab CSV parsed with `-F','`) |
| `scripts/run_vtune_vllm.sh` | The outer `vtune -command pause` after the benchmark is non-fatal in `per_step` / `per_step_isolate` ROI modes (still fatal in `window` mode) | Found in a live end-to-end test run: the worker's own `itt_pause()` had already paused collection, the outer pause was rejected, and `set -e` aborted the script before `vtune -command stop` and the reports |
| `scripts/serve_with_vtune.py` | Classifier handles `CachedRequestData` (`req_ids`); guard against double patching; pid in the log line | vLLM ≥ 0.10 changed `scheduled_cached_reqs`; the patch can now be imported twice (sitecustomize + `__main__`) |

Fixed launcher, checked on the test machine: exit 0, the patch active in the API server and
both spawned processes ("VTune ROI window OPENED"), all three CSVs written, GPU Time parsed.

## Re-syncing from upstream

```bash
UP=<path-to>/vllm-xpu-profiling-toolkit/vtune
DST=plugins/intel-gpu-ai-skills/skills/vtune-vllm

# Verbatim files — should diff clean
diff -u "$DST/references/itt-api.md"          "$UP/docs/itt-api.md"
diff -u "$DST/references/vllm-integration.md" "$UP/docs/vllm-integration.md"
diff -u "$DST/scripts/vtune_itt.py"           "$UP/scripts/vtune_itt.py"

# Modified files — expect exactly the changes in the tables above
diff -u "$DST/SKILL.md"                          "$UP/SKILL.md"
diff -u "$DST/references/troubleshooting.md"     "$UP/docs/troubleshooting.md"
diff -u "$DST/references/validation-and-flow.md" "$UP/docs/validation-and-flow.txt"
diff -u "$DST/scripts/run_vtune_vllm.sh"         "$UP/scripts/run_vtune_vllm.sh"
diff -u "$DST/scripts/serve_with_vtune.py"       "$UP/scripts/serve_with_vtune.py"
# Moved sections: compare by section title against the upstream SKILL.md
diff -u <(cat "$DST/SKILL.md" "$DST/references/commands-and-workflow.md") "$UP/SKILL.md"
```

Then update the commit hash above and re-apply the tables. Record every further edit to a
vendored file here; offer fixes that are not specific to this pack upstream as well.
