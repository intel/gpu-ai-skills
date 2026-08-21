---
name: torch-xpu-profile
description: Profile a Hugging Face model on Intel GPU at the **PyTorch level** with `torch.profiler` and Kineto. Captures CPU + XPU timeline, exports Chrome trace, identifies hottest kernels and async-overlap gaps. Use when the user asks why a model is slow, which op is the bottleneck, or where the GPU is idle. Not for profiling inside a running vLLM server (use vllm-xpu-profile) or for SYCL-kernel-level signal beneath the PyTorch op layer (use xpu-profile-unitrace).
---

# torch-xpu-profile

`torch.profiler` with `ProfilerActivity.XPU` captures CPU + XPU op
timeline on Kineto. Use for "which `aten::*` op is slow?", "is
the GPU idle?", "is decode kernel-bound?". For server-side
profiling use **vllm-xpu-profile**; for SYCL kernel level use
**xpu-profile-unitrace**.

## CUDA -> XPU translation

| CUDA | XPU |
|---|---|
| `ProfilerActivity.CPU, ProfilerActivity.CUDA` | `ProfilerActivity.CPU, ProfilerActivity.XPU` |
| `prof.export_chrome_trace("trace.json")` | identical |
| `record_shapes=True, with_stack=True` | identical |
| `prof.key_averages().table(sort_by="cuda_time_total")` | `sort_by="xpu_time_total"` |

## Quickstart

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(
    mid, dtype=torch.bfloat16, device_map="xpu"
).eval()
inp = tok("Tell me a joke.", return_tensors="pt").to("xpu")

# Warmup so first-run kernel compile doesn't pollute the trace
with torch.no_grad():
    model.generate(**inp, max_new_tokens=8, do_sample=False,
                   pad_token_id=tok.eos_token_id)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=64, do_sample=False,
                       pad_token_id=tok.eos_token_id)

print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

Output: `trace.json` (drag into <https://ui.perfetto.dev>) plus a
top-20 ops table sorted by XPU time.

## Reading the trace

1. **Top of `key_averages` table** — first 5 rows are usually
   70–90% of XPU time. Top = matmul/attention -> GPU-bound (good).
   Top = `aten::copy_` / H2D-D2H transfers -> bandwidth-bound or
   paying for unnecessary moves.
2. **Gaps in XPU timeline** — wide white bands = GPU waiting for
   host. Causes: tokenizer on CPU, Python overhead, per-step sync.
   Decode TPOT spikes correlate with these gaps.
3. **Per-step decode structure** — same kernel sequence per
   token; divergence often means KV reallocation or SWA edge effects.
4. **Memory peaks** (with `profile_memory=True`) — peak before
   first generated token is prefill; subsequent peaks are KV
   growth. Compare against **model-can-it-fit** prediction.

## Longer-window profiling with `schedule`

For multi-step capture without giant files:

```python
sched = schedule(skip_first=2, wait=1, warmup=1, active=3, repeat=1)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
             schedule=sched,
             on_trace_ready=lambda p: p.export_chrome_trace(
                 f"trace_step_{p.step_num}.json")) as prof:
    for step in range(10):
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=16, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        prof.step()
```

One trace per repeat -> easier to diff than one giant file.

## Common errors

- `ProfilerActivity.XPU not supported` -> torch build doesn't
  include XPU profiler. Confirm `torch.__version__` ends in
  `+xpu` and is recent enough.
- Empty trace / only CPU ops -> kernel launches not synced before
  profiler exit. Add `torch.xpu.synchronize()` at end of `with
  profile(...)`.
- Trace > 500 MiB -> use `schedule` form above for short active
  windows.
- "Cannot find any XPU devices" mid-profile -> an op fell back to
  CPU. Set `PYTORCH_ENABLE_XPU_FALLBACK=0` so unsupported ops
  error loudly.

## What to do with the bottleneck

- Standard transformer op (attention, MLP, layer norm) at
  unsurprising time -> hardware limit for this dtype. Try a
  quantized path (see **vllm-xpu-run** Quantization).
- Memory copy / H2D-D2H -> input pipeline is the bottleneck. Move
  tokenizer to a background thread, use pinned memory.
- Unfamiliar dominating kernel -> SYCL-level investigation; use
  **xpu-profile-unitrace**.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Pin to one XPU. |
| `PYTORCH_ENABLE_XPU_FALLBACK=0` | Error on CPU fallbacks instead of hiding them. |
| `KINETO_LOG_LEVEL=INFO` | Diagnose incomplete traces. |

## What this skill does NOT cover

- Profiling inside a running vLLM server -> **vllm-xpu-profile**.
- SYCL kernel level / hardware metrics -> **xpu-profile-unitrace**.
- Fixing a slow kernel — out of scope.

## References

- PyTorch profiler tutorial: <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- `torch.profiler` API: <https://docs.pytorch.org/docs/stable/profiler.html>
- Perfetto: <https://ui.perfetto.dev>
