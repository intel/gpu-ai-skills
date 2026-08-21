# How to test this skill pack

Each layer takes more time than the last; stop at the deepest one you
care about.

## Layer 0 — Install targets and manual layout

The fastest broad install from a local checkout is:

```sh
bash scripts/install.sh --all
```

That copies every skill directory into the known personal skills
locations and leaves unrelated skills alone. Existing skill dirs
with the same names are moved into
`~/.cache/intel-model-skillpack/install-backups/`.

Manual install paths, if you want to test one agent at a time:

| Agent | Personal install | Project install / fallback |
|---|---|---|
| Claude Code | `~/.claude/skills/<skill>/SKILL.md` | `<repo>/.claude/skills/<skill>/SKILL.md` |
| OpenAI Codex | `${CODEX_HOME:-~/.codex}/skills/<skill>/SKILL.md` | `<repo>/.agents/skills/<skill>/SKILL.md` plus `agents/AGENTS.md` if needed |
| opencode | `~/.config/opencode/skills/<skill>/SKILL.md` | `<repo>/.opencode/skills/<skill>/SKILL.md`, `<repo>/.claude/skills/<skill>/SKILL.md`, or `<repo>/.agents/skills/<skill>/SKILL.md` |
| Gemini CLI | `gemini extensions install . --consent` from this repo | `gemini extensions install <repo-url> --consent`; plain skills can live in `<repo>/.gemini/skills/` |
| Qwen Code | `~/.qwen/skills/<skill>/SKILL.md` | `<repo>/.qwen/skills/<skill>/SKILL.md` |
| Kimi Code CLI | `~/.kimi/skills/<skill>/SKILL.md` | `<repo>/.agents/skills/<skill>/SKILL.md` |
| GitHub Copilot CLI | `~/.copilot/skills/<skill>/SKILL.md` | `<repo>/.agents/skills/<skill>/SKILL.md` |
| Cursor | `~/.cursor/skills/<skill>/SKILL.md` | `<repo>/.cursor/skills/<skill>/SKILL.md` |
| Hermes Agent | `~/.hermes/skills/<skill>/SKILL.md` | skills tap from `intel/gpu-ai-skills` |
| OpenClaw | `~/.openclaw/skills/<skill>/SKILL.md` | `<workspace>/skills/<skill>/SKILL.md` or `skills.load.extraDirs` |
| Generic AGENTS.md agent | Copy `agents/AGENTS.md` into the project | Keep this repo checkout available so paths in `agents/AGENTS.md` resolve |

The file layout must stay intact:

```text
<skills-root>/
  model-config-recommend/
    SKILL.md
    scripts/
    data/
```

Do not flatten the files. Agents discover the folder from
`SKILL.md` frontmatter, then load scripts and data by relative path.

## Layer 1 — Static checks (~10 seconds, no GPU)

Run from the repo root:

```sh
bash tests/static.sh
```

This validates:

- Every JSON manifest parses (`.claude-plugin/plugin.json`,
  `.claude-plugin/marketplace.json`, `.cursor-plugin/plugin.json`,
  `gemini-extension.json`).
- Every `SKILL.md` has the required `name` + `description`
  frontmatter, and the `name` matches its parent directory (Agent
  Skills spec hard requirement).
- Every Python helper compiles (`scripts/generate_agents.py`,
  `plugins/.../scripts/*.py`).
- `agents/AGENTS.md` is in sync with `scripts/generate_agents.py`
  (regenerates and diffs).
- Generated `agents/AGENTS.md` does not leak YAML quoting or
  internal-only paths.
- `marketplace.json` plugin count matches the skill directory
  count.

Expected output: `All checks passed.` Anything else is a bug.

Every check above is stdlib-only, so on the release branch this layer runs on a
bare interpreter. On `main` the gate additionally runs `guardrails/check.py`,
which needs PyYAML; without it that step prints `PyYAML is required: pip install
pyyaml` and the gate exits non-zero. On `main`, install it first:

```sh
python3 -m pip install pyyaml
```

It is not made a skip when absent on purpose — a missing package must not
silently switch a gate off.

One further optional package: `tests/xpu-port.sh` needs `libcst` to exercise its
scanner and rewriter sections. Absent it, that script announces `SKIP: libcst not
importable` and the rest of the suite still passes. `pip install libcst` if you
are changing anything under the `xpu-port` skill.

## Layer 2 — Install round-trip (~30 seconds, no GPU)

Confirms the install script puts files in the right places without
breaking other skills the user has installed.

```sh
# Install for whichever agents you have on this machine
bash scripts/install.sh

# Or, force-create dirs for agents you have not used yet
bash scripts/install.sh --all

# Verify one or more target dirs
ls ~/.claude/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls "${CODEX_HOME:-$HOME/.codex}/skills/" 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.config/opencode/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.qwen/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.kimi/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.copilot/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.cursor/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.hermes/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true
ls ~/.openclaw/skills/ 2>/dev/null | grep -E 'xpu-|model-can-it-fit' || true

# Reverse it
bash scripts/install.sh --uninstall
```

What to check: every skill directory under
`plugins/intel-model-skillpack/skills/` should appear under each
agent's skills path during install, and disappear cleanly during
uninstall. Any other skills you had previously installed (HF
skills, vllm-skills, your own) should be untouched on both
operations.

Hermes tap smoke:

```sh
hermes skills tap add intel/gpu-ai-skills
hermes skills install intel/gpu-ai-skills/xpu-discover
hermes skills list | grep xpu-discover
```

OpenClaw extra-dir smoke:

```sh
mkdir -p ~/.openclaw
cp configs/openclaw/openclaw.json ~/.openclaw/openclaw.json
# Edit the extraDirs path first if this repo is not at the sample path.
openclaw
```

## Layer 3 — VRAM calculator (~5 seconds, no GPU, internet to HF)

`model-can-it-fit` is a stand-alone calculator. Verify it returns
plausible numbers and refuses cleanly on diffusion:

```sh
# Decoder-only LLM
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --quant bf16 --device-vram-gb 32

# VLM (architecture detected, vision tower folded into weights)
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct --quant bf16 --device-vram-gb 32

# Diffusion (refused cleanly with component list)
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 --device-vram-gb 32
```

Expected: LLM and VLM print a VRAM breakdown and FITS / DOES NOT
FIT verdict. Diffusion exits with a list of pipeline components and
a pointer to `torch-xpu-bench`.

## Layer 3a — Calculator unit tests (~5 seconds, no GPU, internet to HF)

This layer needs `pytest` installed. It is the only third-party package the
calculator tests require — everything they exercise is stdlib — so a clean
checkout will report `No module named pytest` until you install it:

```sh
python3 -m pip install pytest
```

Then:

```sh
python3 -m pytest tests/test_fit.py -q
```

Expected on a machine with network and no `HF_TOKEN`:

```
71 passed, 6 skipped
```

The 6 skips are the four gated models (Llama 3.1/3.3, Gemma 2 9B/27B) and two
Gemma-specific dimension tests. Across situations:

| Situation | Result |
|---|---|
| Network, no `HF_TOKEN` | 71 passed, 6 skipped |
| Network, valid `HF_TOKEN` with the Llama + Gemma licences accepted | 77 passed |
| Cache warm, no network | same as the run that warmed it |
| No network, cold cache | 21 passed, 56 skipped — nothing fails |

This suite needs each model's upstream `config.json`. Those files are **not
committed, and must not be** — they are third-party files under their own
licences (Apache-2.0, MIT, Llama Community, Gemma Terms of Use), and this
project does not redistribute them. The suite fetches them at pinned revisions
on first use and caches them under `tests/data/.cache/configs/`, which is
git-ignored. The rationale, the pins, and how to refresh one are documented in
[`tests/data/model_revisions.py`](tests/data/model_revisions.py).

A config that cannot be fetched is always a **skip**, never a failure — gated
repo, missing token, no network, and HTTP 429 rate limits all skip. The fix for
a skipped test is a token or a network route, never a committed config.

Optionally pre-fetch, then run sealed off from the network:

```sh
export HF_TOKEN="..."        # optional; only the gated models need it
python3 tests/data/fetch_configs.py

SKILLPACK_TESTS_OFFLINE=1 python3 -m pytest tests/test_fit.py -q
```

`SKILLPACK_TESTS_OFFLINE=1` serves the cache only and makes no network call, so
it is also how you prove the suite is hermetic. With a warm cache it finishes in
about 0.1 s.

## Layer 3b — Recommender static smoke (~10 seconds, no GPU, internet to HF)

`model-config-recommend` should use model config from HF, prefer DP
when the model fits per GPU, and emit the current vLLM-XPU launch
shape.

```sh
python3 plugins/intel-model-skillpack/skills/model-config-recommend/scripts/recommend.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device arc-pro-b70 --num-devices 2 \
    --runtime vllm-xpu --ctx 4096 --concurrency 4 \
    --no-hub-search
```

Expected output:

- `Layout: dp=2, tp=1` or equivalent DP preference for this small
  model.
- Fit breakdown includes weights / KV / activations / framework.
- TTFT, decode-step latency, and aggregate decode throughput are
  printed as bands, not single measured numbers.
- Launch line includes `--block-size=64`.
- Launch line does **not** include `--trust-remote-code` for Qwen2.5
  unless the fetched config declares `auto_map`.
- A `# NOTE:` block says no `--revision` was passed, so the default
  branch is resolved at launch. Re-run with `--revision <sha>` and the
  launch line carries `--revision <sha>` instead of the note.
- Passing `--trust-remote-code` for a Hub model without `--revision`
  exits 2 with `--trust-remote-code requires --revision`.
- Quant caveats describe `--attention-backend TRITON_ATTN` and
  W4A8 verification via logs, not legacy env-var-only guidance.

Host guard path (no real XPU required; uses test override):

```sh
INTEL_SKILLPACK_FAKE_XPU_COUNT=1 \
python3 plugins/intel-model-skillpack/skills/model-config-recommend/scripts/recommend.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device arc-pro-b70 --num-devices 4 \
    --discover-host --no-hub-search
```

Expected: exits non-zero before fetching model config, with a
message saying `--num-devices 4 exceeds host-visible XPUs (1)`.
For real deployment on the current host, agents should add
`--discover-host`; omit it only for what-if planning for a different
host.

## Layer 4 — Per-skill agent acceptance (~5 minutes per skill, GPU)

For each skill, the way to test that an agent following the skill
actually succeeds is: hand a fresh agent (no prior context) just
the skill file(s) and a task that invokes it.

The pattern we used while authoring is documented as a sub-agent
prompt template:

> Role-play as a developer fluent in CUDA who just got their first
> Intel GPU. NO prior context. Your only resource is the SKILL.md
> at `<path>`. Read it. Then complete `<concrete task on this
> machine>`. Paste the actual commands and outputs. Report under
> "VERDICT" what worked, what was misleading, and what edits you
> would make.

Worked-example pairings used during authoring:

| Skill | Task |
|-------|------|
| `xpu-system-setup` (Battlemage on Ubuntu 24.04) | On a host with Ubuntu 24.04 stock kernel (6.8) and an Arc Pro B70, run `check_battlemage_prerequisites.sh` (no flags) and verify it identifies all three broken prerequisite layers: nomodeset present, xe alias missing, runtime too old. Then run `--fix` and follow the prompts; after reboot confirm `clinfo` shows 1 platform and `xpu-smi discovery` shows the B70. Re-run the script with no flags — it should report all OK. |
| `xpu-discover` | Answer the six pre-flight questions on this host. |
| `xpu-container-run` | Launch `intel/vllm:<tag>-xpu` interactively and confirm GPU visibility. `xpu-smi` is absent from the vLLM image (it ships only in the standalone `intel/intel-extension-for-pytorch` image); use `python -c "import torch; print(torch.xpu.device_count())"` as the fallback verification. |
| `torch-xpu-run` | Load `Qwen/Qwen2.5-0.5B-Instruct` and generate one sentence on XPU. |
| `vllm-xpu-run` | Serve `Qwen/Qwen2.5-0.5B-Instruct`, curl `/v1/chat/completions`, get a non-empty reply. |
| `sglang-xpu-run` | Run the pre-flight gate, then serve a model and smoke-test `/v1/chat/completions`. **Two build steps required:** (1) build `sglang-xpu:local` from `docker/xpu.slim.Dockerfile` (~28 min; add `ENV MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8` to avoid OOM on many-core hosts); (2) apply the UMD-pin patch: write the patch Dockerfile into a private scratch dir and build from it — `d=$(mktemp -d); ...write "$d/Dockerfile"...; docker build -t sglang-xpu:local-b580 "$d"` (<5 s). Use `mktemp -d`, not a fixed `/tmp` name: the path is otherwise guessable, and `/tmp` as a build context also uploads every readable file in it to the daemon. — this replaces the too-new PPA driver (26.05) with the kernel-compatible version from `intel/vllm:0.17.0-xpu` (25.48). Also add `--group-add "$(stat -c '%g' /dev/dri/renderD128)"` to all `docker run` commands — the sglang image runs as non-root user `sdp` which is not in the render group by default. If the image is absent, the skill correctly redirects to `vllm-xpu-run`; that redirect counts as PASS for the skill's guard logic. |
| `vllm-xpu-bench` | Bench the running vLLM server, check TTFT/TPOT/throughput land in the predicted shape. |
| `torch-xpu-bench` | Run `bench.py` on a small causal LM and on diffusion / encoder snippets. |
| `sglang-xpu-bench` | Run `sglang.bench_serving` against a sglang server (only reachable if the sglang container builds clean on your kernel). |
| `model-can-it-fit` | See Layer 3 above. |
| `model-config-recommend` | Ask for the best vLLM-XPU config for Qwen2.5-7B at 8K on B70/B60; verify it reports fit, DP/TP, TTFT, decode-step latency, aggregate decode, and a launch command. |

Run any pairing as a sub-agent task in your agent of choice; the
skill is "OK" if the sub-agent completes the task without asking
for help beyond the skill body.

## Layer 4b — User-shaped prompt acceptance

After installing the skills into a real agent, start a new session
with no extra context and ask natural prompts. These are the prompts
users will actually type:

| User prompt | Expected skill behavior |
|---|---|
| "What Intel GPUs do I have and are they healthy?" | Uses `xpu-discover`; reports devices, driver, render group, quick diagnostic. |
| "Will Qwen2.5-32B int4 fit on my Arc Pro B70 at 8K context and concurrency 4?" | Uses `model-can-it-fit`; fetches config, prints weights/KV/activation/framework breakdown and FITS / DOES NOT FIT. |
| "What is the expected performance of Qwen2.5-7B on Arc Pro B70 at 8K with vLLM?" | Uses `model-config-recommend`; reports config-derived TTFT, decode-step latency, aggregate decode band, and explains the MFU/BWE efficiency band. |
| "What is the best configuration for this model on 4x Arc Pro B70 cards?" | Uses `model-config-recommend`; evaluates DP/TP, prefers DP when the model fits per card, emits one launch command per DP replica or TP when needed. |
| "Deploy Qwen2.5-7B on my 4 Intel cards." | Uses `xpu-discover` first, then `model-config-recommend`, then `vllm-xpu-run`; should not guess device count without checking. |
| "Run this model for inference with the OpenAI API." | Uses `vllm-xpu-run` if the user wants a server; asks PyTorch vs vLLM if ambiguous. |
| "Run the same model in pure PyTorch, no server." | Uses `torch-xpu-run`; does not emit vLLM container commands. |
| "Benchmark this model." | Asks whether the target is PyTorch, vLLM, or SGLang unless a server/framework is already obvious. |
| "Why is TPOT worse than the recommender predicted?" | Uses `vllm-xpu-bench` / `xpu-profile-unitrace` framing; checks quant path, KV dtype, concurrency, kernel fallback, and explains roofline miss. |

Acceptance: the agent should choose the right skill, mention any
required preflight checks, avoid destructive cleanup, and distinguish
analytical predictions from measured throughput.

## Layer 5 — End-to-end smoke (~15 minutes, real GPU)

A single user-shaped flow that exercises Run + Bench + Fit
together. No script — type these into your agent and verify each
step.

1. **Health.** "Is my Intel GPU detected and healthy?" — agent
   activates `xpu-discover`, reports device 0, driver pass.
2. **Sizing.** "Will Qwen2.5-7B in BF16 fit on this device at 4K
   context, concurrency 4?" — agent activates `model-can-it-fit`,
   prints a verdict.
3. **Run.** "Start a vLLM server with that model on this device."
   — agent activates `vllm-xpu-run`, picks an `intel/vllm:<tag>-xpu`
   image, sets the four required env vars, server reaches
   `Application startup complete`.
4. **Smoke test.** "Curl `/v1/chat/completions` with a one-liner
   prompt." — agent returns a sample completion.
5. **Bench.** "Bench TTFT and TPOT on this server, concurrency
   sweep 1, 4, 8." — agent activates `vllm-xpu-bench`, runs
   `vllm bench serve` three times, prints the medians.
6. **Cleanup.** "Stop and remove the container."

If all six steps complete without the user pasting any link or
flag from the SKILL.md bodies, the pack is doing its job.

## Layer 6 — Routing spot-check (~5 minutes, no GPU)

Confirms the agent picks the right skill from a battery of
prompts. Used as part of authoring; reviewers can re-run if they
suspect description drift.

Prompt set: use the example prompts in the README tables plus at
least one ambiguous prompt for each gate, such as "benchmark this
model" without naming PyTorch / vLLM / SGLang. Acceptance: clear
cases all route to a single skill; ambiguous cases trigger an "ask
for clarification" response from the agent; adversarial cases route
to NONE or to a redirect skill.

## What is NOT tested at any layer

- Multi-XPU correctness (this workstation has one Intel GPU).
- The PROFILE gate skills (see `xpu-profile-*` once landed).
- The FIX gate skills (Python -> Triton -> SYCL escalation, planned
  for v0.2).
- Production-load benches (the skill bodies cite shapes for small
  bench windows; full-scale serving load is out of scope here).

## Reporting an issue

Open a GitHub issue with:

- Output of `bash tests/static.sh` (or note that it failed).
- Which layer hit the issue.
- For Layer 4 / 5: the verbatim agent prompt and the verbatim
  agent response that contains the failure.
