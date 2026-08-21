# intel-model-skillpack

A collection of [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) for running, benchmarking, and profiling arbitrary Hugging Face safetensors models on Intel GPUs (Arc, Arc Pro, Battlemage, Data Center GPU Max). Covers the PyTorch + Transformers, vLLM-XPU, and SGLang-XPU stacks.

Complementary to [vllm-project/vllm-skills](https://github.com/vllm-project/vllm-skills) (NVIDIA-focused) and [huggingface/skills](https://github.com/huggingface/skills) (Hub workflows). Neither covers running arbitrary HF safetensors models on local Intel GPUs; this pack does.

Format follows the [Agent Skills specification](https://agentskills.io/specification).

## Installing

These skills work with any agent that supports the Agent Skills standard,
including Claude Code, opencode, OpenAI Codex, Cursor, GitHub Copilot CLI,
Gemini CLI, Qwen Code, Kimi Code, Hermes Agent, and OpenClaw.

### Local clone, all agents at once (recommended)

```sh
git clone https://github.com/intel/gpu-ai-skills.git intel-model-skillpack
cd intel-model-skillpack
bash scripts/install.sh
```

Installs into every agent skills directory it detects. Pass `--all` to also create dirs for agents you haven't used, `--agent <name>` for one agent, `--uninstall` to reverse.

### Claude Code

```text
/plugin marketplace add intel/gpu-ai-skills
/plugin install <skill-name>@intel-model-skillpack
```

### Gemini CLI

This repo ships [`gemini-extension.json`](gemini-extension.json):

```sh
gemini extensions install . --consent
# or from URL:
gemini extensions install https://github.com/intel/gpu-ai-skills.git --consent
```

### Clone / Copy

For any other agent, clone this repo and copy the skill folders into the agent's skills directory:

| Agent | Skill Directory | Docs |
|-------|-----------------|------|
| Claude Code | `~/.claude/skills/` | [docs](https://code.claude.com/docs/en/skills) |
| opencode | `~/.config/opencode/skills/` | [docs](https://opencode.ai/docs/skills/) |
| OpenAI Codex | `${CODEX_HOME:-~/.codex}/skills/` | [docs](https://developers.openai.com/codex/skills/) |
| GitHub Copilot CLI | `~/.copilot/skills/` | [docs](https://docs.github.com/en/copilot) |
| Cursor | `~/.cursor/skills/` | [docs](https://cursor.com/docs/context/skills) |
| Qwen Code | `~/.qwen/skills/` | [docs](https://github.com/QwenLM/qwen-code) |
| Kimi Code | `~/.kimi/skills/` | [docs](https://www.kimi.com/code/docs/en/) |
| Hermes Agent | `~/.hermes/skills/` or `hermes skills tap add intel/gpu-ai-skills` | [docs](https://hermes-agent.nousresearch.com/docs/guides/work-with-skills/) |
| OpenClaw | `~/.openclaw/skills/`, `<workspace>/skills/`, or `skills.load.extraDirs` | [docs](https://open-claw.bot/docs/tools/skills/) |
| Generic / `AGENTS.md` | `<repo>/.agents/skills/` or `~/.config/agents/skills/` | uses [`agents/AGENTS.md`](agents/AGENTS.md) |

```sh
cp -r plugins/intel-model-skillpack/skills/* <skill-directory>/
```

opencode also auto-loads from `~/.claude/skills/` and `~/.agents/skills/`, so any of those paths works. GitHub Copilot CLI also accepts `gh skill install intel/gpu-ai-skills --agent github-copilot --scope user` (gh ≥ v2.90).

## Layout

```
plugins/intel-model-skillpack/skills/   # one directory per skill
scripts/               # repo tooling (install, publish, generate AGENTS.md)
agents/                # generated AGENTS.md fallback bundle
.claude-plugin/        # Claude Code plugin marketplace manifests
tests/                 # static validation
template/              # SKILL.md template for contributors
```

The `agents/AGENTS.md` bundle is generated from the individual `SKILL.md` files; re-run `scripts/` tooling after adding a skill to keep it fresh.

### Ubuntu 24.04 + Arc Pro B60/B70 (Battlemage)

On Ubuntu 24.04 with the stock kernel, Battlemage GPUs require three
prerequisites before the skills work. Run the diagnostic first:

```sh
bash plugins/intel-model-skillpack/skills/xpu-system-setup/scripts/check_battlemage_prerequisites.sh
```

It checks for `nomodeset` in GRUB, the OEM kernel 6.17 requirement, and
compute runtime >=26.18. Pass `--fix` to apply remediations, or follow
the steps in `plugins/intel-model-skillpack/skills/xpu-system-setup/SKILL.md`
→ **Battlemage Prerequisites**.

## Skills

Skills are contextual and auto-loaded based on your conversation. The agent reads each skill's description at startup and loads the matching `SKILL.md` body when a request matches.

Three gates: **Run** the model, **Benchmark** it, then **Profile** to find why it's slow.

### Run — get a model on the GPU

| Skill | Useful for | Example prompt |
|-------|------------|----------------|
| `xpu-discover` | Inventory Intel GPUs and check driver health (`xpu-smi` wrapper). CUDA analogue is `nvidia-smi`. | "Is my Intel GPU detected? Run a quick health check." |
| `xpu-runtime-preflight` | Check shared host, device, Docker, proxy, storage, and optional container readiness before using GPU/XPU skills in this pack. | "Before using the XPU skills on this host, run the preflight and tell me what blocks it." |
| `xpu-container-run` | Launch a Docker container with Intel GPU access (`/dev/dri`, render group, `ZE_AFFINITY_MASK`, `--ipc=host`). | "How do I launch a Docker container that can see my Intel GPU?" |
| `torch-xpu-run` | Run any HF safetensors model via upstream PyTorch + Transformers + `torch.xpu`. CUDA → XPU code translation. | "Run gemma-3 in pure PyTorch on my Intel GPU." |
| `vllm-xpu-run` | Serve a model with vLLM-XPU's OpenAI-compatible HTTP API. Image picker, flag rationales, multi-GPU patterns. | "Start a vLLM server with Qwen2.5-7B on my Intel GPU." |
| `sglang-xpu-run` | Serve a model with SGLang's XPU backend (`--device xpu --attention-backend intel_xpu`). RadixAttention prefix caching. | "Serve Qwen3 with sglang on my Battlemage GPU." |

### Benchmark — measure how fast

| Skill | Useful for | Example prompt |
|-------|------------|----------------|
| `torch-xpu-bench` | Single-process bench of an HF model via pure PyTorch (no server). TTFT, decode rate, peak XPU memory. | "Bench Qwen3-8B forward pass on Intel without any server." |
| `vllm-xpu-bench` | Bench a running vLLM-XPU server (`vllm bench serve` / `throughput`). TTFT, TPOT, ITL, throughput at concurrency. | "Benchmark TTFT and TPOT on my running vLLM-XPU server." |
| `sglang-xpu-bench` | Bench a running SGLang server (`sglang.bench_serving`). Includes prefix-cache hit-rate measurement. | "Measure RadixAttention prefix-cache hit rate on my sglang server." |

### Profile — find why it's slow

| Skill | Useful for | Example prompt |
|-------|------------|----------------|
| `torch-xpu-profile` | Profile an HF model with `torch.profiler` + Kineto. Export Chrome trace; find hot ops + idle gaps. | "Why is my Qwen2.5 generate() slow on Intel?" |
| `vllm-xpu-profile` | Profile a running vLLM server via `/start_profile` and `/stop_profile`, or offline `vllm bench --profile`. | "Capture a vLLM-XPU profile around a real-traffic window." |
| `xpu-profile-unitrace` | SYCL / Level Zero kernel-level profiling with `unitrace` (PTI-GPU; built from source). Per-kernel timing, oneCCL events, HW counters. | "Show me the actual SYCL kernel names taking the time." |

### Tooling — sizing and config

| Skill | Useful for | Example prompt |
|-------|------------|----------------|
| `model-can-it-fit` | Estimate VRAM (weights + KV + activations + framework) from HF `config.json`. Decoder-only LLM, MoE, VLM; refuses diffusion. | "Will Qwen2.5-32B in int4 fit on my Arc Pro B70 at 8K context, concurrency 4?" |
| `model-config-recommend` *(experimental)* | Recommend a vLLM-XPU deployment config (quant, KV dtype, DP/TP, capacity) using roofline math against Intel Arc B-series specs. Predict → calibrate → verify. | "What's the best config to serve Qwen2.5-7B on my B70 at 8K?" |

The descriptions are designed to disambiguate by *deployment shape*: "benchmark X" alone is intentionally ambiguous (which framework?), so a good agent will ask whether you mean PyTorch / vLLM / sglang before picking. If you want to be explicit, mention the skill by name: *"Use the **vllm-xpu-bench** skill to ..."*.

## Contributing

New skills go under `plugins/intel-model-skillpack/skills/<skill-name>/` with a `SKILL.md`. Start from [`template/SKILL.md`](template/SKILL.md).

Before committing any skill change:

```sh
bash scripts/check-skills.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the validation rules.

## Validate

```sh
bash tests/static.sh
```

The checks themselves are stdlib-only. On this development branch the gate also
runs `guardrails/check.py`, which needs PyYAML — without it that one step exits
`PyYAML is required: pip install pyyaml` and the gate fails:

```sh
python3 -m pip install pyyaml
```

Unit tests for the VRAM calculator, which need `pytest`:

```sh
python3 -m pip install pytest
python3 -m pytest tests/test_fit.py -q      # 71 passed, 6 skipped without HF_TOKEN
```

These fetch each model's upstream `config.json` from the Hub at a pinned
revision on first use and cache it under `tests/data/.cache/`. The configs are
third-party files under their own licences and are deliberately not committed to
this repository; a config that cannot be fetched skips its tests rather than
failing them. To pre-fetch the configs and then run with no network, see
[Layer 3a in HOW_TO_TEST.md](HOW_TO_TEST.md#layer-3a--calculator-unit-tests-5-seconds-no-gpu-internet-to-hf).

For deeper testing — install round-trip, per-skill acceptance, end-to-end smoke — see [HOW_TO_TEST.md](HOW_TO_TEST.md).

## Resources

- [Agent Skills specification](https://agentskills.io/specification)
- [Anthropic — Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [vllm-project/vllm-skills](https://github.com/vllm-project/vllm-skills) — NVIDIA equivalents
- [huggingface/skills](https://github.com/huggingface/skills) — Hub workflows
- [PyTorch XPU getting started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [vLLM XPU installation](https://docs.vllm.ai/en/latest/getting_started/xpu-installation.html)

## License

Apache-2.0. See `LICENSE`.
