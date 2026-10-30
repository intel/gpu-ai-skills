# Model recipes: hybrid (Qwen3.5) models, larger TP=4/8 models, and the SGLang Cookbook

All commands run inside the SGLang-XPU release image with the container flag
set from SKILL.md. `$IMG` is the tag resolved in SKILL.md Step 1. "Measured
here" means `v0.5.22-xpu` on 2× Arc Pro B70. Everything else is labelled with
its source and is **not validated by this pack**.

## Hybrid linear-attention models (Qwen3.5, Nemotron-3-Nano): different flags

Qwen3.5 mixes Gated DeltaNet (Mamba-style, linear-attention) layers with full
attention. Nemotron-3-Nano (`NemotronHForCausalLM`) mixes Mamba-2, MoE and
attention layers, and the image's code treats it the same way (it is in the
same hybrid-cache list), though that was not run here.

**1. The SKILL.md page-size flag kills startup while the radix cache is on.**
The Mamba state then uses the `no_buffer` strategy, which requires
`page_size=1`, but `intel_xpu` attention only takes 64/128:

| Flags (Qwen3.5-4B, radix cache on) | Result |
|---|---|
| `--attention-backend intel_xpu --page-size 64` | startup `AssertionError: no_buffer only supports page_size=1.` |
| `--attention-backend intel_xpu`, no `--page-size` | rewritten to 128, then `MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got 128` |
| `+ --mamba-radix-cache-strategy extra_buffer` | `extra_buffer is not supported for Qwen3_5ForConditionalGeneration; use no_buffer.` |

**2. Add `--linear-attn-backend intel_xpu`.** It is not the default: without
it the linear layers resolve to `triton`, and the first request spends over a
minute in JIT compilation. Upstream CI only smoke-tests this fused SYCL path
(`test_xpu_qwen3_5_9b.py`, accuracy threshold 0). Output was correct here, but
it has no upstream accuracy gate yet.

Two working configurations, both serving validated output:

| Qwen3.5-4B, `--tp 1`, measured here | first request | 256 tok, warm |
|---|---|---|
| cookbook: no attention flag (→ `triton`, page 1), default linear backend (→ `triton`) | **79 s** | 6.8 s |
| cookbook: no attention flag + `--linear-attn-backend intel_xpu` | 20 s | 6.3 s |
| `--attention-backend intel_xpu --linear-attn-backend intel_xpu --disable-radix-cache --page-size 64` | **6 s** | 6.2 s |

```sh
python3 -m sglang.launch_server --model Qwen/Qwen3.5-4B --device xpu --tp 1 \
    --attention-backend intel_xpu --linear-attn-backend intel_xpu \
    --disable-radix-cache --page-size 64 \
    --host 0.0.0.0 --port 30000
```

The second configuration is also the shape upstream's nightly XPU CI uses:
every model there runs `--attention-backend intel_xpu --disable-radix-cache`,
and Qwen3.5 runs at `--page-size 128`.

### Prefix caching on hybrid models: exact continuations only

Upstream disables the radix cache, and the cookbook configuration (no
attention flag) keeps it on. In that configuration it works, but reuses far
less than on a normal model. Same ~360-token shared prefix, measured here via
the `#cached-token` log field:

| Request pattern | Qwen3-0.6B (attention-only) | Qwen3.5-4B (cookbook flags) |
|---|---|---|
| raw `/generate` continuation: turn-1 prompt **+ its output** + new question | 384 cached | **425 cached** ✅ |
| same long system prompt, different question | 320 cached | **0** |
| exact repeat of an earlier prompt | 320 cached | **0** |
| multi-turn chat via `/v1/chat/completions` | 320 cached | **0** |

Why: a KV cache can be cut at any token, but a Mamba state is one running
summary that cannot be rewound. Under `no_buffer` it is saved only at the end
of a finished request, so a hit needs the new request to **extend that
request's exact token sequence**. Shared system prompts and repeats never do.
Multi-turn chat should, but Qwen3.5's chat template renders an empty
`<think>\n\n</think>` block into the generation prompt and **strips it from
assistant turns in the history**, so turn 2 diverges from turn 1 right after
`assistant\n` (checked with the tokenizer's `apply_chat_template`).

The mode that would save state at prefix boundaries, `extra_buffer`, is
**switched off for every hybrid model on XPU**:
`supports_mamba_cache_extra_buffer()` begins `if get_platform().is_xpu:
return False` (`sglang/srt/arg_groups/overrides.py`, v0.5.22). On CUDA the same
models get it with the Triton linear backend.

So for hybrid models on XPU:

- Use `--disable-radix-cache` with `intel_xpu` attention (fastest first
  request, upstream's shape). The cache is worth little here anyway.
- Keep the cache (cookbook flags) only for a client that sends raw-completion
  continuations, such as an agent appending to one growing prompt via
  `/generate` or `/v1/completions`.
- If prefix reuse across chat turns or shared system prompts is the reason
  for choosing SGLang, it does not hold for these models on XPU. Weigh
  **vllm-xpu-run**.

MTP (NEXTN) works with either configuration; see
`spec-decode-and-multimodal.md`. Both configurations also run on v0.5.21.

## Larger models (TP=4 / TP=8): recipes not validated by this pack

These need 4–8 cards, and the verified host has 2. They are included because
they are the best-sourced starting points available. Treat each as
unvalidated on your host: launch, then run the SKILL.md verify block before
trusting it.

**Source ranking.** Prefer the upstream nightly XPU CI test
(`test/registered/xpu/llm_models/*.py` at the image's tag, suite
`nightly-xpu-kernel-main-4-gpu`). It records exact flags and a GSM8K accuracy
bar. Use the SGLang Cookbook generator when no CI test exists; its commands
are bare and need translating (next section).

**Container wrapper**: the SKILL.md Multi-GPU launch, with
`-e ZE_AFFINITY_MASK=0,1,2,3` (or `0,…,7`) and `-e FI_TCP_IFACE=lo`. The
all-reduce hang in `multi-gpu-tp.md` is model-independent. Raise the verify
block's 120 s readiness cap: upstream's tests allow these launches up to an
hour (`timeout_for_server_launch = 3600`).

**Upstream CI base flags**, used by every row marked *CI* below. `<cmd>` is
the `python3 -m sglang.launch_server --model <model>` line inside that
wrapper:

```sh
<cmd> --device xpu --tp 4 --attention-backend intel_xpu --dtype bfloat16 \
    --disable-overlap-schedule --disable-radix-cache \
    --host 0.0.0.0 --port 30000
```

CI also passes `--trust-remote-code` on every model. Per SKILL.md, omit it and
add it back only if a launch fails without it.

| Model (BF16 weights) | Source | Extra flags on top of the base | Notes |
|---|---|---|---|
| `Qwen/Qwen3-30B-A3B` (~61 GB) | CI, TP=4, GSM8K ≥ 0.90 | `--max-total-tokens 65536 --mem-fraction-static 0.8` | Not hybrid. Drop `--disable-radix-cache` to get prefix caching (works on smaller Qwen3 here, not tested on this model) |
| `Qwen/Qwen3.5-35B-A3B` (~70 GB) | CI, TP=4, GSM8K ≥ 0.90 | `--page-size 128 --max-total-tokens 65536 --mem-fraction-static 0.85` | Hybrid: keep `--disable-radix-cache`. Cookbook adds `--linear-attn-backend intel_xpu` (faster start; smoke-tested only upstream) |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (~63 GB) | CI, TP=4, GSM8K ≥ 0.72 | `--page-size 64 --max-total-tokens 65536 --mem-fraction-static 0.85 --context-length 16384 --chunked-prefill-size 1024 --max-running-requests 8 --watchdog-timeout 1200 --model-impl sglang --tool-call-parser qwen3_coder --reasoning-parser nemotron_3` | Hybrid. `--model-impl sglang` picks SGLang's native NemotronH over the repo's `auto_map` code |
| `google/gemma-4-26B-A4B-it` (49 GB) | CI, TP=4, GSM8K ≥ 0.90 | `--page-size 64 --max-total-tokens 65536 --mem-fraction-static 0.9 --chunked-prefill-size -1` | **`--chunked-prefill-size -1` is required**: per the CI file, "Gemma-4 hybrid-attention kernels crash under chunked prefill on XPU". Measured here: at TP=2 BF16 does not fit and FP8 crashes, so 4 cards is the route |
| `Qwen/Qwen3.5-9B` (~19 GB) | CI, **TP=1**, GSM8K ≥ 0.90 | `--tp 1 --page-size 128 --max-total-tokens 65536 --mem-fraction-static 0.85` | Hybrid; same rules as Qwen3.5-4B above |
| `meta-llama/Llama-3.1-8B-Instruct` | CI, **TP=2**, GSM8K ≥ 0.80 | `--tp 2 --max-total-tokens 65536 --mem-fraction-static 0.8` | CI file: "TP=4 wedges the Level Zero driver during the first prefill batch on Arc/BMG". TP=1 measured here |
| `google/gemma-4-31B-it` (59 GB) | Cookbook only, TP=4 | `--mem-fraction-static 0.80` | No CI test. Add `--chunked-prefill-size -1` too: same Gemma-4 hybrid-attention kernels as the 26B (an inference, not run) |
| `Qwen/Qwen3-32B` (~65 GB) | Cookbook only, TP=4 | none beyond `--tp 4` | No CI test |
| `meta-llama/Llama-3.3-70B-Instruct` (~141 GB) | Cookbook only, **TP=8** | `--tp 8` | No CI test, and upstream's own Llama-3.1-8B wedges at TP=4 on Arc/BMG, so expect trouble. 141 GB needs 8 cards even at 32 GB each. If it hangs at the first prefill, nothing at TP>2 is known to work for Llama on this stack |
| `deepseek-ai/DeepSeek-OCR-2` | Cookbook TP=1; upstream has `test_deepseek_ocr_2_olmbench.py` | `--enable-multimodal --tp 1` | Single card |

Two hazards specific to TP≥4, both from upstream's CI code:

- **Concurrent prefill can wedge the driver.** The CI mixin runs single-stream
  by default because "intel_xpu attention at TP>=2 wedges the Level Zero driver
  on concurrent prefill". Only Qwen3-30B-A3B, Qwen3.5 and Nemotron raise it to
  4 concurrent clients. TP=2 here survived 96 concurrent requests, but ramp
  concurrency gradually at TP=4/8 and watch `dmesg` for engine resets.
- **These models expect `--max-total-tokens 65536`.** It caps the KV pool, so
  the default memory split does not starve activations. Raise it only after
  the model serves.

Both Qwen3.5-35B-A3B and Nemotron-3-Nano have 2 KV heads. At `--tp 4` that is
the case where decode context parallelism (`--dcp-size 2`, `multi-gpu-tp.md`)
can add KV capacity, unlike the TP=2 run measured there. Not tested.

## Translating an SGLang Cookbook BMG command

The [SGLang Cookbook](https://docs.sglang.io/cookbook/intro) model pages have
command generators with a **BMG** hardware option (`arc_b` in the page
source). The intro page lists no Intel hardware; look inside the model pages.
**Do not paste a cookbook BMG command as-is.** It is bare `sglang serve
--model-path <model> --device xpu [--tp N]`:

- **No `--attention-backend intel_xpu`**, so XPU resolves to `triton` with
  `page_size 1`. Single-stream decode is about the same, but under load it is
  much slower (measured here):

  | Llama-3.1-8B-Instruct, `--tp 1` | first request | 256 tok | 32 concurrent × 128 tok |
  |---|---|---|---|
  | cookbook (`triton`, page 1) | 15 s | 6.6 s | 12.9 s |
  | skill flags (`intel_xpu`, page 64) | 3 s | 6.6 s | 5.0 s |

- **No container shape, no `FI_TCP_IFACE=lo`**, and none of the CI-only flags
  in the table above (e.g. Gemma-4's `--chunked-prefill-size -1`).
- **`sglang serve --model-path`** is equivalent to `python3 -m
  sglang.launch_server --model` (both verified). The old form only prints a
  deprecation-style `UserWarning`.

Translate by keeping the model, `--tp` and `--mem-fraction-static`. Add the
SKILL.md container flags, `-e FI_TCP_IFACE=lo` when `--tp > 1`, and
`--attention-backend intel_xpu --page-size 64`. For hybrid models, either keep
the cookbook's attention default or add `--disable-radix-cache` with it (above).

The generators also **lock options on BMG**: BF16 only (FP8 greyed out),
base/instruct variants only for Qwen3 and Llama-3.1, speculative decoding off
(Qwen3.5, Gemma 4), and EP off ("Only Tensor Parallel (TP) is supported on
this hardware"). The EP lock matches this skill's finding that `--ep-size`
hangs the GPU. The FP8 and MTP locks are conservative rather than proven
failures: runtime FP8 (Qwen3-0.6B) and Qwen3.5 MTP serve validated output here.

The diffusion pages carry BMG recipes too: `FLUX.2-dev` with `--num-gpus 4
--tp-size 4 --component-residency dit=resident,text_encoder=layerwise-offload
--dit-cpu-offload False` (upstream also has `test_xpu_flux2_dev.py`), and
`Z-Image-Turbo` on one card (`test_xpu_zimage_turbo.py`). Those are SGLang
diffusion serving, outside this skill's text-generation scope, and not
verified here.
