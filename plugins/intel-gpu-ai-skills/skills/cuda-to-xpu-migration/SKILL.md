---
name: cuda-to-xpu-migration
description: Create a CUDA-to-XPU migration assessment for an existing AI repo. Identify CUDA-specific assumptions, route to the right XPU skills, produce a migration report. Use when the user has a CUDA repo, notebook, Dockerfile, launch script, HF / vLLM / SGLang workload, or Triton kernel and asks to migrate it to Intel Arc / Arc Pro / Battlemage / XPU — including "convert this to XPU" / "move it to XPU" and the bare "migrate this repo" request where scope is not yet set. A request that says "port" routes to xpu-port. Plans and routes only. Not for executing an already-scoped rewrite (use xpu-port), running migrated code, or measuring it.
---

# cuda-to-xpu-migration

Create a migration assessment for CUDA-oriented repos that want an Intel XPU path.

## Scope

This skill is for assessment and planning only.

- Identify CUDA-specific assumptions and likely migration surfaces.
- Reuse existing skills for code translation, runtime setup, serving, profiling, and sizing.
- Produce a clean migration report with recommended edits, blockers, and next steps.
- Do not execute or validate migrated code.
- Do not duplicate detailed commands or framework runbooks that are already covered by other skills.

## Always assess; route forward once

This skill assesses the workflow verbs — "assess", "migrate",
"convert", "move" — with the same read-only pass: inventory,
classify, report. A request that says "port" (or otherwise names
the rewrite explicitly) belongs to **xpu-port**, not here; route
it there. For the workflow verbs, do not bounce the request away
before the report exists: a mis-routed execution request costs
one cheap read-only assessment, while the reverse mis-route would
rewrite a user's repo unasked.

The report's **Next steps** (with the "Non-namespace CUDA
surfaces → Route" table) is the single onward-routing authority.
It names the executor for each surface — **xpu-port** for
portable Python, **xpu-deploy-plan** for API-first / serving
repos, container and runtime skills per tier. Routing flows one
way, assess → execute; nothing here routes backward.

## Use with

- **xpu-discover** for an XPU environment / driver preflight check before recommending a migration target.
- **xpu-runtime-preflight** for a read-only go/no-go check of host/container XPU readiness (drivers, /dev/dri, permissions, runtime, and essentials) before any XPU workload.
- **torch-xpu-run** for PyTorch / Transformers CUDA-to-XPU code translation guidance.
- **xpu-container-run** for container runtime setup on Intel GPU.
- **vllm-xpu-run** for vLLM launch and runtime guidance.
- **sglang-xpu-run** for SGLang launch and runtime guidance.
- **model-can-it-fit** and **model-config-recommend** for fit and config decisions.
- **xpu-deploy-plan** for a coordinated end-to-end serving plan (preflight → fit → config → launch → smoke test) once the assessment routes an API-first / serving-stack repo to a local XPU endpoint.

## Migration flow

1. **Inventory**
   - If the request is a **git URL** (not a local path), shallow-clone it to a stated scratch location first (e.g. `git clone --depth 1 <url> /tmp/<repo>-assess`) and assess that path. State the clone target before scanning — do not assess a URL from memory.
   - If the user's request does not include a repo path, search the local workspace for the relevant scripts or repo (e.g. `find ~/workspace -name "*.sh" -path "*vllm*"`). State which path you are assessing before running the scan — do not silently assume a location.
   - Run `scripts/inventory.sh` from the target repo root to surface CUDA/NVIDIA references and common launch/runtime surfaces.
   - Group findings by area: framework code, runtime/container, serving stack, Triton kernels, and CUDA-native dependencies.
   - **Notebooks:** `inventory.sh` greps `.ipynb` files as text, so CUDA references in notebook cells are surfaced. Notebook code cells are not directly rewritable: for a notebook-heavy repo, flag that code cells need extraction to `.py` (or hand-porting) before the port executes, and note it under "Needs version-aware review".

2. **Classify**
   - Safe translation candidates: portable PyTorch / Transformers code.
   - Runtime changes: container flags, environment variables, launch scripts.
   - Version-sensitive changes: vLLM / SGLang internals and plugin code.
   - Likely blockers: CUDA extensions, CuPy, NCCL, TensorRT, flash-attn, bitsandbytes, cuBLAS / CUTLASS / cuTENSOR / cuSPARSE / cuDNN consumers, custom `.cu` / `.cuh` code.
   - **Cloud-API / provider references (not CUDA):** many "NVIDIA" hits are HTTP-client calls, not local GPU code — NVIDIA NIM (`integrate.api.nvidia.com`, `langchain-nvidia-ai-endpoints`, `build.nvidia.com`), or unrelated cloud services (ElevenLabs, OpenAI). These do **not** need a CUDA port. Classify them as: **provider swap** (retarget a local vLLM-XPU / SGLang-XPU OpenAI-compatible endpoint and change the `api_base` / model name — no Python rewrite) or **unaffected** (GPU-agnostic cloud dependency, e.g. a TTS API). Do not count these toward the CUDA migration surface — the pdf-to-podcast blueprint had ~170 grep hits at the time of assessment (upstream is unpinned, so the exact count drifts) and zero actual CUDA code.

3. **Route**
   - For XPU environment / driver preflight, use **xpu-discover**.
   - For PyTorch / Transformers code translation, use **torch-xpu-run**.
   - For container changes, use **xpu-container-run**.
   - For vLLM / SGLang serving paths, use **vllm-xpu-run** or **sglang-xpu-run**.
   - For deployment and capacity planning, use **model-can-it-fit**, and **model-config-recommend**.
   - For non-namespace CUDA constructs, route per the per-construct guidance in the "Non-namespace CUDA surfaces" section of `templates/migration_report.md`: `torch.compile` and `pin_memory` → **xpu-port** (execution) / **torch-xpu-run** (reference); `ProfilerActivity.CUDA` → **torch-xpu-profile**; `set_float32_matmul_precision` / TF32 toggles → delete (no-op on XPU).

4. **Report**
   - When ready to write the final assessment, read `templates/migration_report.md` and fill it in.
   - The verdict line, copied verbatim from the template, must appear as the **first line of your final message** — `Migration result: <COMPLETE | PARTIAL | BLOCKED>`. Keep the literal prefix `Migration result:` intact — do not abbreviate it to `Result:`, and do not fold it into a heading (e.g. `## Migration Assessment` followed by `**Result: PARTIAL**`). If you save the report to a file, do that step **first**, then make the verdict line the opening of the closing message — do not lead with the report and trail off in a paraphrased summary that drops the line. It is the headline of the assessment and the marker a reader uses to confirm an assessment — not a refusal — was produced.
   - Summarize what appears directly portable, what needs version-aware review, and what is blocked on CUDA-native components.
   - Recommend the next skill(s) to use for each category.

## Migration tiers

| Tier | Surface | Action |
|---|---|---|
| 1 | PyTorch / HF / Diffusers Python | Route to **torch-xpu-run** for code translation guidance. |
| 2 | Docker / shell / benchmark scripts | Route to **xpu-container-run** for runtime/container guidance. |
| 3 | vLLM / SGLang launch scripts | Route to **vllm-xpu-run** / **sglang-xpu-run**. |
| 4 | Triton kernels | Check whether equivalent kernels already exist in the [`intel/intel-xpu-backend-for-triton`](https://github.com/intel/intel-xpu-backend-for-triton) repo. If a matching kernel exists, plan to reuse it; otherwise flag for XPU review and correctness/perf follow-up. |
| 5 | CUDA-native libraries and custom ops | Mark as blockers or redesign candidates; do not claim mechanical migration. |

## CUDA-native blockers and routes

| Component | Why it blocks | Route |
|---|---|---|
| CUDA C++ / `.cu` custom op | CUDA kernels do not run on XPU directly. | Replace with torch native ops, Triton XPU, SYCL/oneAPI extension, or fallback path. |
| CuPy | CUDA array runtime. | Replace with torch, NumPy CPU fallback, dpctl/SYCL path, or isolate. |
| NCCL | NVIDIA collective library. | Use XPU distributed guidance from the appropriate runtime skill. |
| TensorRT | NVIDIA inference runtime. | Route to vLLM-XPU, SGLang-XPU, OpenVINO, or PyTorch XPU depending on target. |
| flash-attn | CUDA-specific optimized attention package. | Use runtime-native attention backend or fallback recommended by the runtime skill. |
| bitsandbytes | CUDA-centric quantization/runtime kernels. | Route to an XPU-supported quantized path or alternative checkpoint/runtime. |
| cuBLAS / CUTLASS / cuTENSOR / cuSPARSE / cuDNN | NVIDIA-only compute libraries (GEMM, attention building blocks, sparse ops, DNN primitives). | Replace with torch native ops on XPU (oneDNN-backed), oneMKL, or rewrite via Triton XPU / SYCL. |
| CUDA Graphs | CUDA execution feature. | Disable, replace, or branch only if the current XPU runtime supports the path. |

## Files in this skill

- `scripts/inventory.sh` — read-only CUDA/NVIDIA inventory scan; run from the target repo root.
- `templates/migration_report.md` — final report template; load only when writing the migration report.

## What this skill does NOT cover

- Executing or validating migrated code.
- Repeating detailed code-translation tables already covered by **torch-xpu-run**.
- Repeating detailed container command guidance already covered by **xpu-container-run**.
- Claiming feature parity for NVIDIA-only libraries.
