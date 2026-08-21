---
name: torch-xpu-bench
description: Benchmark a Hugging Face model on an Intel GPU through pure PyTorch + Transformers, **single-process, no HTTP server**. Measures generate() throughput in tokens/sec, time-to-first-token, decode-step latency, and peak XPU memory. Also covers diffusion and encoder-only models via `references/non-llm-snippets.md`. Use after **model-can-it-fit** to validate predicted memory against `torch.xpu.max_memory_allocated()`.
---

# torch-xpu-bench

Pure-PyTorch benchmarks of HF models on XPU. Measures bare
`model.generate()` with no scheduler, no continuous batching, no
API overhead. For serving-shape numbers (TTFT/TPOT/ITL under
concurrency) use **vllm-xpu-bench**.

**Important:** This skill provides a ready-to-use benchmark script at
`{base_dir}/scripts/bench.py`. Do not write your own benchmark script;
use the provided one.

Use cases:

- TTFT and decode rate of a single sequence at a given dtype.
- A/B dtypes (bf16 vs fp16), compile on/off, model size.
- Confirm `--enforce-eager` is hurting in vLLM before flipping it.
- Validate **model-can-it-fit** predictions against `max_memory_allocated`.

For diffusion and encoder-only models, see
`references/non-llm-snippets.md`.

## CUDA → XPU translation

| CUDA | XPU |
|---|---|
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` |
| `torch.cuda.Event(enable_timing=True)` | `torch.xpu.Event(enable_timing=True)` |
| `torch.cuda.max_memory_allocated()` | `torch.xpu.max_memory_allocated()` |
| `torch.cuda.reset_peak_memory_stats()` | `torch.xpu.reset_peak_memory_stats()` |
| `torch.cuda.empty_cache()` | `torch.xpu.empty_cache()` |

Two gotchas:

- **Synchronise before timing.** XPU kernels launch async; without
  `torch.xpu.synchronize()` you measure launch latency.
- **Warm up.** First call compiles XPU Triton kernels; first
  compiled-graph call triggers capture. Discard the first 1–3 runs.

## Hosts with non-Intel GPUs

If `nvidia-smi` or `rocm-smi` reports hardware, bench code that
doesn't pin a device may silently land on CUDA/ROCm. Defences:

```sh
export CUDA_VISIBLE_DEVICES=     # hide all NVIDIA
export HIP_VISIBLE_DEVICES=      # hide all AMD ROCm
export ZE_AFFINITY_MASK=0        # pick a specific Intel device
```

`scripts/bench.py` asserts model parameters land on `xpu` after
`from_pretrained`; replicate in your own benches. Cross-check with
`xpu-smi dump -d 0 -m 5` — if XPU memory doesn't climb, the bench
is on the wrong device.

These all load onto `xpu:0` when only an Intel XPU is visible:
`device_map="xpu" / "xpu:0" / 0 / "auto" / {"": "xpu"} / {"": 0}`.
`hf_device_map` is empty for whole-model placement; populates only
on multi-GPU splits.

## Quickstart

**Always use the provided benchmark script** at `{base_dir}/scripts/bench.py`:

```sh
python3 {base_dir}/scripts/bench.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype bfloat16 \
    --prompt-tokens 512 --gen-tokens 128 \
    --warmup 2 --runs 5
```

`--revision` pins the Hugging Face repo revision (branch, tag, or commit sha;
default `main`). Pass a commit sha when a benchmark number has to stay
reproducible — otherwise the run silently follows whatever `main` points at.

Output:

```
Run  TTFT (ms)  Decode (ms/tok)  Total (s)  Throughput (tok/s)
1      ...           ...           ...           ...
...

Median: TTFT ... ms, decode ... ms/tok, throughput ... tok/s
Peak XPU memory: ... GiB
```

Use this **relative**: capture a baseline median, change one knob,
re-run. Throughput won't match anyone else's; the *delta* between
two runs on your hardware is the signal.

## Per-run metrics

- **TTFT** — wall time from `generate()` entry to first token,
  dominated by prefill.
- **Decode (ms/tok)** — per-token time for tokens 2..N, averaged.
- **Total (s)** — full call wall time.
- **Throughput** — `gen_tokens / Total_s`.
- **Peak XPU memory** — `torch.xpu.max_memory_allocated()`.

Median across runs is robust to jitter; mean isn't. Median delta
within ~3% is noise; >5% is signal; >10% is a clear win/regression.

## A/B comparisons

```sh
python3 {base_dir}/scripts/bench.py --model "$M" --dtype bfloat16 --runs 5 > bench-bf16.txt
python3 {base_dir}/scripts/bench.py --model "$M" --dtype float16  --runs 5 > bench-fp16.txt
diff -u <(grep Median bench-bf16.txt) <(grep Median bench-fp16.txt)
```

Same shape for `--compile` on/off, model A vs B.

## When `torch.compile` actually helps

Confirm the compile speedup pattern on Battlemage before flipping
`--enforce-eager` in vLLM:

```sh
python3 {base_dir}/scripts/bench.py --model "$M" --runs 5             # eager
python3 {base_dir}/scripts/bench.py --model "$M" --compile --runs 5   # compiled
```

If compiled wins on decode (ms/tok) and content is correct, drop
`--enforce-eager` in **vllm-xpu-run** for that model.

Compile gain scales with model size — sub-1B models see little win
because launch overhead and kernel work are similar in magnitude.
First compiled run includes graph capture cost; default warmup is
post-capture. If a 0.5B sees no win, retest at 7B+ before
concluding "compile doesn't help."

## Feedback loop with model-can-it-fit

Run against the model you tested with **model-can-it-fit**. Compare
the bench's "Peak XPU memory" against the calculator's "Total" line:

- Decoder-only LLM, gap >20% -> calculator bug; file an issue.
- VLM / diffusion -> gap is expected (the calculator covers
  decoder-only). Use the measurement to refine your headroom budget.
- Lower than predicted -> KV cache may not have grown to the
  configured `max_seq_len`; rerun with longer `--gen-tokens`.

## Bench-procedure checklist (any workload)

1. Pin device (`device_map="xpu"`); verify
   `next(model.parameters()).device`.
2. `reset_peak_memory_stats(0)` + `empty_cache()` once.
3. Warm: 1–3 unmeasured iterations.
4. Synchronise around the timer:
   ```python
   torch.xpu.synchronize(); t = time.perf_counter()
   <work>
   torch.xpu.synchronize(); times.append(time.perf_counter() - t)
   ```
5. N=5 timed iterations (raise to 10–20 if >10% spread).
6. Report median, not mean. Add p99 for latency-tail concerns.
7. Cross-check with `xpu-smi dump -d 0 -m 5,18 -i 1` from host.

## Common errors

- `Cannot find any XPU devices` → check **xpu-container-run** /
  **xpu-discover** before running.
- `XPU out of memory` → reduce `--prompt-tokens` / `--gen-tokens`,
  use `float16`, or quantize. Run **model-can-it-fit** first.
- TTFT >> decode → expected on long prompts.
- Suspiciously fast decode + non-language output → garbage
  generation. Validate sample output before trusting throughput.
- `--compile` first run much slower → graph capture cost; raise
  `--warmup`.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Pin to one XPU. |
| `TRITON_CACHE_DIR` | Persist XPU Triton kernels across runs. |
| `PYTORCH_ENABLE_XPU_FALLBACK=0` | Make CPU fallbacks loud during a bench. |
| `HF_TOKEN` | Gated checkpoints. |

## What this skill does NOT cover

- vLLM serving benches → **vllm-xpu-bench**.
- SGLang serving benches → **sglang-xpu-bench**.
- Profiling → **torch-xpu-profile**.
- Training throughput — out of scope.

## References

- `references/non-llm-snippets.md` — diffusion + encoder-only
- `torch.xpu` API: <https://docs.pytorch.org/docs/stable/xpu.html>
- HF `generate()`: <https://huggingface.co/docs/transformers/main_classes/text_generation>
- `torch.compile`: <https://docs.pytorch.org/docs/stable/torch.compiler.html>
