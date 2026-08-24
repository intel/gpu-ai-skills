<skills>

You have additional SKILLs documented in directories containing a "SKILL.md" file.

These skills are:
 - cuda-to-xpu-migration -> "plugins/intel-gpu-ai-skills/skills/cuda-to-xpu-migration/SKILL.md"
 - llamacpp-xpu-run -> "plugins/intel-gpu-ai-skills/skills/llamacpp-xpu-run/SKILL.md"
 - model-can-it-fit -> "plugins/intel-gpu-ai-skills/skills/model-can-it-fit/SKILL.md"
 - model-config-recommend -> "plugins/intel-gpu-ai-skills/skills/model-config-recommend/SKILL.md"
 - sglang-xpu-bench -> "plugins/intel-gpu-ai-skills/skills/sglang-xpu-bench/SKILL.md"
 - sglang-xpu-run -> "plugins/intel-gpu-ai-skills/skills/sglang-xpu-run/SKILL.md"
 - torch-xpu-bench -> "plugins/intel-gpu-ai-skills/skills/torch-xpu-bench/SKILL.md"
 - torch-xpu-profile -> "plugins/intel-gpu-ai-skills/skills/torch-xpu-profile/SKILL.md"
 - torch-xpu-run -> "plugins/intel-gpu-ai-skills/skills/torch-xpu-run/SKILL.md"
 - vllm-xpu-bench -> "plugins/intel-gpu-ai-skills/skills/vllm-xpu-bench/SKILL.md"
 - vllm-xpu-profile -> "plugins/intel-gpu-ai-skills/skills/vllm-xpu-profile/SKILL.md"
 - vllm-xpu-run -> "plugins/intel-gpu-ai-skills/skills/vllm-xpu-run/SKILL.md"
 - xpu-container-run -> "plugins/intel-gpu-ai-skills/skills/xpu-container-run/SKILL.md"
 - xpu-deploy-plan -> "plugins/intel-gpu-ai-skills/skills/xpu-deploy-plan/SKILL.md"
 - xpu-discover -> "plugins/intel-gpu-ai-skills/skills/xpu-discover/SKILL.md"
 - xpu-model-type-detect -> "plugins/intel-gpu-ai-skills/skills/xpu-model-type-detect/SKILL.md"
 - xpu-port -> "plugins/intel-gpu-ai-skills/skills/xpu-port/SKILL.md"
 - xpu-profile-unitrace -> "plugins/intel-gpu-ai-skills/skills/xpu-profile-unitrace/SKILL.md"
 - xpu-runtime-preflight -> "plugins/intel-gpu-ai-skills/skills/xpu-runtime-preflight/SKILL.md"
 - xpu-system-setup -> "plugins/intel-gpu-ai-skills/skills/xpu-system-setup/SKILL.md"

IMPORTANT: You MUST read the SKILL.md file whenever the description of the skills matches the user intent, or may help accomplish their task.

IMPORTANT: When a SKILL.md points at files under its `references/` directory, you MUST read the referenced file before acting on the topic it covers — those files hold the detailed flag tables, command examples, and gotchas the SKILL.md trims out.

<available_skills>

- cuda-to-xpu-migration: Create a CUDA-to-XPU migration assessment for an existing AI repo. Identify CUDA-specific assumptions, route to the right XPU skills, produce a migration report. Use when the user has a CUDA repo, notebook, Dockerfile, launch script, HF / vLLM / SGLang workload, or Triton kernel and asks to migrate it to Intel Arc / Arc Pro / Battlemage / XPU — including "convert this to XPU" / "move it to XPU" and the bare "migrate this repo" request where scope is not yet set. A request that says "port" routes to xpu-port. Plans and routes only. Not for executing an already-scoped rewrite (use xpu-port), running migrated code, or measuring it.

- llamacpp-xpu-run: Run a GGUF model on an Intel GPU using llama.cpp's SYCL backend (Level Zero) with the official intel.Dockerfile. Covers building the Docker image from source at a pinned tag, launching llama-server with an OpenAI-compatible API, device selection, multi-GPU layer splitting, all recommended runtime env vars, flash-attention, and quantisation selection. Use when the user has a GGUF model and wants fast local inference or an OpenAI-compatible endpoint on Intel GPU without Python/PyTorch. The CUDA analogue is llama.cpp built with `-DGGML_CUDA=ON`. Use **vllm-xpu-run** instead for safetensors models with continuous batching at scale; use **torch-xpu-run** for Hugging Face Transformers direct.

- model-can-it-fit: Estimate whether a Hugging Face decoder-only LLM, MoE, or VLM fits in Intel GPU VRAM for a quantization, context length, concurrency, runtime, and tensor-parallel setting. Use for memory-fit or max-model-len planning before launch. Reports weights, KV cache, activations, framework overhead, and first mitigation. Not for diffusion. Memory-only — does NOT predict throughput, tokens/sec, latency, or runtime config; route those to bench/deploy/recommend skills.

- model-config-recommend: Recommend a vLLM-XPU deployment config (quant, KV dtype, DP/TP, max concurrency, max context) for a Hugging Face decoder-only LLM on Intel Arc B-series GPUs using roofline math against published hardware specs. Experimental; predictions are physics-bounded ranges, not measured throughput. Use after xpu-discover and before vllm-xpu-run.

- sglang-xpu-bench: Benchmark a **running SGLang-XPU server** on an Intel GPU using `sglang.bench_serving`. Measures TTFT, TPOT, ITL, end-to-end latency, and throughput against the OpenAI-compatible endpoint. Use after sglang-xpu-run. Not for vLLM servers (use vllm-xpu-bench) or no-server PyTorch (use torch-xpu-bench).

- sglang-xpu-run: Serve a Hugging Face safetensors model on an Intel GPU using SGLang's XPU backend with the OpenAI-compatible API. Covers pulling the pre-built `intel/sglang-dev:latest` image, fixing the render-group and UMD/kernel compatibility issues that affect non-root sglang images, the SYCL_UR / Level Zero env vars needed on Battlemage, the `--device xpu --attention-backend intel_xpu` flag set, multimodal serving, and how to validate output content (not just HTTP 200). Use when the user needs SGLang's RadixAttention prefix caching or grammar-constrained output; for broad-coverage serving on Intel today prefer vllm-xpu-run, and for benchmarking a running server use sglang-xpu-bench.

- torch-xpu-bench: Benchmark a Hugging Face model on an Intel GPU through pure PyTorch + Transformers, **single-process, no HTTP server**. Measures generate() throughput in tokens/sec, time-to-first-token, decode-step latency, and peak XPU memory. Also covers diffusion and encoder-only models via `references/non-llm-snippets.md`. Use after **model-can-it-fit** to validate predicted memory against `torch.xpu.max_memory_allocated()`.

- torch-xpu-profile: Profile a Hugging Face model on Intel GPU at the **PyTorch level** with `torch.profiler` and Kineto. Captures CPU + XPU timeline, exports Chrome trace, identifies hottest kernels and async-overlap gaps. Use when the user asks why a model is slow, which op is the bottleneck, or where the GPU is idle. Not for profiling inside a running vLLM server (use vllm-xpu-profile) or for SYCL-kernel-level signal beneath the PyTorch op layer (use xpu-profile-unitrace).

- torch-xpu-run: Run an arbitrary Hugging Face safetensors model on an Intel GPU using **upstream PyTorch** (>= 2.8) with the built-in `torch.xpu` device. Covers loading from the Hub, picking the right dtype, autocast, multi-GPU with accelerate's `device_map`, and the CUDA -> XPU code translation a user has to do once. Use for the Transformers / Accelerate / Diffusers path. Not for OpenAI-compatible serving (use vllm-xpu-run); explicitly not via intel-extension-for-pytorch (ipex) or ipex-llm — those paths are end-of-life and upstream PyTorch supersedes them.

- vllm-xpu-bench: Benchmark a **running vLLM-XPU OpenAI-compatible server** on an Intel GPU using `vllm bench`. Measures TTFT (time-to-first-token), TPOT (time-per-output-token), ITL (inter-token latency), end-to-end latency, and throughput under concurrency. Covers online (`vllm bench serve`) and offline (`vllm bench throughput`) modes; concurrency sweeps and quant comparison live in `references/sweep-and-compare.md`. Use after **vllm-xpu-run** when the user asks "how fast is this?".

- vllm-xpu-profile: Profile a running vLLM-XPU server with torch.profiler around a window of real requests, either via /start_profile and /stop_profile HTTP endpoints or via vllm bench --profile for offline runs. Use to find the dominant op under real concurrent traffic. Not for pure PyTorch (use torch-xpu-profile), SYCL kernel-level signal (use xpu-profile-unitrace), throughput numbers (use vllm-xpu-bench), or non-vLLM servers.

- vllm-xpu-run: Serve a Hugging Face safetensors model on an Intel GPU with upstream vLLM-XPU's OpenAI-compatible API. Covers image choice, container launch, the right vllm serve flags (dtype, enforce-eager, model-impl fallback, attention backend, quant + KV-cache pairing), and the transformers-backend fallback for unsupported architectures. Use for /v1/chat/completions or /v1/completions on an Intel GPU. Not for pure PyTorch without a server (use torch-xpu-run), throughput numbers (use vllm-xpu-bench), or NVIDIA (use vllm-project/vllm-skills).

- xpu-container-run: Launch a Docker container with Intel GPU access on Linux. Encodes the correct combination of `--device /dev/dri`, render-group access, `--ipc=host`, `ZE_AFFINITY_MASK` pinning, Hugging Face cache mount, and `--entrypoint /bin/bash` for interactive use. Use when running any Intel-XPU container (vLLM-XPU, sglang-xpu, torch-XPU, llama.cpp SYCL, etc.) and the device must be visible inside. The CUDA analogue is `docker run --gpus all` — Intel has no `--gpus` flag, you pass the Direct Rendering Manager (DRM) nodes directly.

- xpu-deploy-plan: Plan an end-to-end Intel XPU model deployment by chaining existing skills. Calls xpu-runtime-preflight (readiness), model-can-it-fit (sizing), model-config-recommend (flags), and the selected runtime skill (vllm-xpu-run / sglang-xpu-run / torch-xpu-run), then writes a single PLAN.md with one exact launch command, smoke test, and rollback to .out/skills/xpu-deploy-plan/. Use when the user asks for a coordinated plan (not a direct deploy/serve request) — wants the orchestration across preflight, fit, config, launch, smoke test, and rollback, or asks which skills to run and in what order.

- xpu-discover: Inventory Intel GPUs (Arc, Arc Pro, Data Center GPU Max) on a Linux host. Detect devices, check driver health, list processes using each XPU, run a quick diagnostic, and read live utilisation.

- xpu-model-type-detect: Before loading a Hugging Face model on Intel XPU, detect its actual type (text generation, text encoder, seq2seq, vision classification, vision-language, audio encoder, audio seq2seq, multimodal VL, diffusion, time-series, reward model, masked LM) so the agent picks the right `AutoModel` class and input kwargs. Prevents "got unexpected keyword argument 'pixel_values'" and "empty logits" errors from mis-routing. Use before `torch-xpu-run` or `vllm-xpu-run` when the user gives a model id the agent hasn't seen before, or when a smoke test fails with a wrong-input signature.

- xpu-port: Execute a single-target CUDA-to-XPU port of a PyTorch repo with libcst-based scan, mechanical rewrite, and CPU FP64 vs target-dtype correctness verify on one forward pass. Use when the request says "port" — "port my repo to XPU", "port my repo at <path> to XPU", "rewrite the CUDA calls to XPU", "apply the mechanical transforms", "run the scan and rewrite", "make the port changes now". Not for the "migrate" verb ("migrate my repo", "migrate this repo to XPU") or a bare whole-repo workflow request where scope is not yet set — those start with cuda-to-xpu-migration, whose plan routes here. Not for assessment-only, throughput (torch-xpu-bench), op-level slowness (torch-xpu-profile), custom CUDA C++ extensions, or dual-target CUDA+XPU codebases.

- xpu-profile-unitrace: Profile Intel-XPU workloads at the SYCL / Level Zero kernel level via Intel pti-gpu's unitrace. Captures per-API-call and per-kernel timing, memory transfers, oneCCL / MPI events, and hardware counters PyTorch-level profilers cannot see. Use when a hot op is already known at the torch.profiler layer and the user needs the SYCL kernel beneath, or when profiling oneCCL collectives in multi-GPU runs. Not for PyTorch-level signal (use torch-xpu-profile / vllm-xpu-profile). Requires building unitrace from source.

- xpu-runtime-preflight: Run a read-only go/no-go preflight before any Intel GPU/XPU skillpack work. Checks driver health, /dev/dri permissions, render/video groups, Docker, /dev/shm, disk, proxy, and optional container-level XPU visibility. Use when the user asks whether a machine is ready for XPU model work or needs a reusable lab readiness report. Not for launching workloads, pulling images, editing system config, or verifying model output.

- xpu-system-setup: First-time setup for Intel XPU/GPU hosts. Detects what's missing and installs xpu-smi, configures user groups (render), sets up Intel GPU PPA repository, installs Level Zero runtime, installs Docker, and runs a post-setup verification gate. Prompts before each installation by default (use --auto for unattended). Also handles Battlemage (Arc Pro B60/B70) prerequisites on Ubuntu 24.04: nomodeset removal, OEM kernel upgrade, and compute runtime 26.18+ — use check_battlemage_prerequisites.sh when xpu-smi shows No device discovered or clinfo shows 0 platforms. Use when a bare-metal or minimally-configured machine needs to be prepared for XPU model work.

</available_skills>

</skills>

Paths referenced within SKILL folders are relative to that SKILL. For example the `xpu-discover/scripts/x.py` would be referenced as `xpu-discover/scripts/x.py`.
