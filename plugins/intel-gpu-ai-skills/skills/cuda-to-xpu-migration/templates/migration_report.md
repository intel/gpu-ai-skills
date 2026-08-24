# Migration report template

Use this template only when producing the final migration assessment.
This skill does not run or validate migrated code; the report is an
assessment and a routing plan, not a test result.

The verdict line is the literal string `Migration result: <COMPLETE |
PARTIAL | BLOCKED>` — keep the word `Migration` and the word `result`
together, exactly as written. Do not shorten it to `Result:` and do not
fold it into a heading (`## Migration Assessment` then `**Result: …**`).
It must be reproducible verbatim as the first line of the reply.

```text
Migration result: <COMPLETE | PARTIAL | BLOCKED>

Repo class: <training-script | api-first | serving-launch | kernel | mixed>
- training-script — dense torch.cuda in .py training/inference code; xpu-port is the primary executor
- api-first — GPU touched only via cloud/HTTP clients (NIM, OpenAI); this is a provider swap, not a code port
- serving-launch — vLLM / SGLang launch scripts + containers dominate; route to the serving skills
- kernel — custom .cu / Triton / CUDA-native libs dominate; expect blockers, not mechanical migration
- mixed — more than one of the above; call out which surface dominates

Portable with existing skill guidance:
- <path/to/file>
  - <short reason it looks portable>
  - Next skill: <torch-xpu-run | xpu-container-run | vllm-xpu-run | sglang-xpu-run>

Needs version-aware review:
- <component or path>
  - <why it needs manual / version-aware review>
  - Next skill: <relevant runtime skill>

Non-namespace CUDA surfaces:
- torch.compile — XPU Inductor support is build-dependent; may need compile=False or device guard
  - Route: xpu-port (execution) / torch-xpu-run (reference)
- torch.profiler.ProfilerActivity.CUDA → .XPU
  - Route: torch-xpu-profile
- set_float32_matmul_precision / TF32 toggles — CUDA-only no-ops on XPU
  - Route: delete or leave (no-op on XPU)
- pin_memory — host-pinned transfer hint tied to CUDA host memory; a no-op on XPU and may be dropped
  - Route: xpu-port (execution) / torch-xpu-run (reference)

Blocked / redesign candidates:
- <path or component>
  - <CUDA-native dependency or assumption>
  - Route: <torch native replacement | Triton XPU | SYCL/oneAPI | alternative runtime>

Not assessed:
- Runtime validation and performance testing
  - This skill does not run or verify migrated code

Next steps:
- <ordered list of next skills to invoke>
- For an API-first / serving-stack repo, chain xpu-deploy-plan after the runtime skill for a coordinated preflight → fit → config → launch → smoke-test plan.
```
