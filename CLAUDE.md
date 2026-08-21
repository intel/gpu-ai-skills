# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A collection of [Agent Skills](https://agentskills.io/specification) for running, benchmarking, and profiling arbitrary Hugging Face safetensors models on Intel GPUs (Arc, Arc Pro, Battlemage, Data Center GPU Max). Covers PyTorch + Transformers, vLLM-XPU, and SGLang-XPU stacks.

The skills follow the Agent Skills standard and work with any compliant agent (Claude Code, opencode, Cursor, GitHub Copilot CLI, Gemini CLI, Qwen Code, Kimi Code, Hermes Agent, OpenClaw).

## Repository structure

```
plugins/intel-model-skillpack/skills/   # One directory per skill
  <skill-name>/
    SKILL.md                            # Skill definition (required)
    scripts/                            # Helper scripts referenced by SKILL.md
    data/                               # Supporting data files
scripts/                                # Repo tooling (install, validation)
agents/                                 # Generated AGENTS.md bundle
tests/                                  # Static validation suite
template/                               # SKILL.md template for contributors
.claude-plugin/                         # Claude Code plugin marketplace manifests
.cursor-plugin/                         # Cursor plugin manifests
```

## Commands

### Validation (required before any skill change)

```bash
# Run full static validation suite
bash tests/static.sh

# Run repo checker + external skill-validator (if installed)
bash scripts/check-skills.sh

# External validator (optional, brew-only)
brew tap agent-ecosystem/tap && brew install skill-validator
```

The static gate validates:
- JSON manifests parse correctly
- Every SKILL.md has `name` and `description` frontmatter
- Skill name matches parent directory (Agent Skills spec requirement)
- Python helpers compile
- Shell scripts parse
- `agents/AGENTS.md` is in sync with generator
- Claude marketplace entries cover every skill

### Testing

```bash
# Layer 1: Static checks (~10 seconds, no GPU required)
bash tests/static.sh

# Layer 3: Test VRAM calculator
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --quant bf16 --device-vram-gb 32

# See HOW_TO_TEST.md for deeper acceptance testing layers
```

### Installation

```bash
# Install into all detected agent skills directories
bash scripts/install.sh

# Install and create dirs for agents not yet used
bash scripts/install.sh --all

# Install for specific agent only
bash scripts/install.sh --agent claude

# Uninstall from all locations
bash scripts/install.sh --uninstall
```

### Regenerate bundled AGENTS.md

```bash
python3 scripts/generate_agents.py
```

This is automatically handled by the pre-commit hook if installed (`pre-commit install`).

## Architecture

### Skill discovery model

Each skill is self-contained with:
1. **YAML frontmatter** in SKILL.md with `name` and `description` fields
   - `name`: kebab-case, matches parent directory (hard requirement)
   - `description`: tells agents when to use this skill (contextual loading)
2. **Markdown body** with instructions, commands, and rationales
3. **Supporting files** in `scripts/` and `data/` subdirectories

Agents read all skill descriptions at startup and load matching SKILL.md bodies when user requests align with the description.

### Three-gate workflow

Skills are organized around three gates:

1. **Run** — Get models on the GPU
   - `xpu-discover`, `xpu-runtime-preflight`, `xpu-container-run`
   - `torch-xpu-run`, `vllm-xpu-run`, `sglang-xpu-run`

2. **Benchmark** — Measure performance
   - `torch-xpu-bench`, `vllm-xpu-bench`, `sglang-xpu-bench`

3. **Profile** — Find bottlenecks
   - `torch-xpu-profile`, `vllm-xpu-profile`, `xpu-profile-unitrace`

Plus tooling skills: `model-can-it-fit` (VRAM estimation), `model-config-recommend` (config optimization), `cuda-to-xpu-migration` (translation guide).

### Validation philosophy

**What ships vs what is internal:**
- `research/` is excluded from published archives (`export-ignore` in `.gitattributes`)
- Skills must not reference `research/` paths
- Every claim must be verifiable from public sources or inline "verify locally" recipes

**Destructive operations are forbidden:**
- No `pkill`, `docker rm -f` of unowned containers, `rm -rf` outside workspace
- Skills accept container names/images as input; never assume one is running
- Scripts must be safe to run without prior context

### Conventions

- Skill names: kebab-case, match parent directory
- Shell: POSIX `sh` or explicit `bash`, never fish syntax
- Intel-specific knobs live in skills, not tooling (stays current without code changes)
- Skills are orchestrator-agnostic (no Kintsugi or private container runner dependencies)
- Assume Docker (or compatible runtime) and `xpu-smi` on host

### Python and shell structure

Helper scripts under `skills/<name>/scripts/`:
- Must compile/parse cleanly (`python3 -m py_compile` / `bash -n`)
- No hardcoded paths outside the skill directory
- Accept flags/inputs rather than reading environment state

## Adding or modifying skills

1. Use `template/SKILL.md` as starting point
2. Keep directory name and `name` frontmatter identical
3. Keep `description` specific about what the skill does and when to use it
4. Focus SKILL.md content; move detail into supporting files
5. Add the new skill to `catalog/bundles.yaml` under the right category
6. Run `bash scripts/check-skills.sh` before commit
7. If changing frontmatter, run `python3 scripts/generate_agents.py` (or let pre-commit hook handle it)

## Key files not to edit manually

- `agents/AGENTS.md` — generated from skill frontmatter, regenerate via `scripts/generate_agents.py`
- `.claude-plugin/marketplace.json` — must list every skill under `plugins/intel-model-skillpack/skills/`

## Relationship to other skill packs

Complementary to:
- [vllm-project/vllm-skills](https://github.com/vllm-project/vllm-skills) — NVIDIA-focused
- [huggingface/skills](https://github.com/huggingface/skills) — Hub workflows

This pack uniquely covers running arbitrary HF safetensors models on local Intel GPUs.

## Supported stacks

**Included:**
- Upstream PyTorch + Transformers with `torch.xpu`
- Upstream vLLM with XPU backend
- SGLang with XPU backend

**Explicitly avoided:**
- `ipex-llm`, `intel-extension-for-pytorch`, `intel/llm-scaler-vllm` — prefer upstream paths
