# Model recipes: hybrid (Qwen3.5) models, SGLang Cookbook BMG recipes

All commands run inside the SGLang-XPU release image with the container flag
set from SKILL.md. `$IMG` is the tag resolved in SKILL.md Step 1. "Measured
here" means `v0.5.23-xpu` on 2× Arc Pro B70. "Cookbook" means the model is
listed with a **BMG** (Intel Arc Pro B-Series) option in the
[SGLang Cookbook](https://docs.sglang.io/cookbook/intro) command generator.

## Hybrid linear-attention models (Qwen3.5): different flags

Qwen3.5 mixes Gated DeltaNet (Mamba-style, linear-attention) layers with full
attention.

**1. The SKILL.md page-size flag kills startup while the radix cache is on.**
The Mamba state then uses the `no_buffer` strategy, which requires
`page_size=1`, but `intel_xpu` attention only takes 64/128:

| Flags (Qwen3.5-4B, radix cache on) | Result |
|---|---|
| `--attention-backend intel_xpu --page-size 64` | startup `AssertionError: no_buffer only supports page_size=1.` |
| `--attention-backend intel_xpu`, no `--page-size` | rewritten to 128, then `MambaComponent requires page_size=1 when mamba_extra_buffer is disabled, got 128` |
| `+ --mamba-radix-cache-strategy extra_buffer` | `extra_buffer is not supported for Qwen3_5ForConditionalGeneration; use no_buffer.` |

**2. Add `--linear-attn-backend intel_xpu`.** It is not the default: without
it the linear layers resolve to `triton`, and JIT compilation makes the first
request ~12× slower than with it. The Cookbook's BMG command for Qwen3.5 adds it too.

Two working configurations, both serving validated output:

| Qwen3.5-4B, `--tp 1`, measured here (time relative to the last row) | first request | 256 tok, warm |
|---|---|---|
| no attention flag (→ `triton`, page 1), default linear backend (→ `triton`) | **~12×** | 1.3× |
| Cookbook shape: no attention flag + `--linear-attn-backend intel_xpu` | 3× | 1.2× |
| `--attention-backend intel_xpu --linear-attn-backend intel_xpu --disable-radix-cache --page-size 64` | **1.0×** | 1.0× |

```sh
python3 -m sglang.launch_server --model Qwen/Qwen3.5-4B --device xpu --tp 1 \
    --attention-backend intel_xpu --linear-attn-backend intel_xpu \
    --disable-radix-cache --page-size 64 \
    --host 0.0.0.0 --port 30000
```

### Prefix caching on hybrid models: exact continuations only

The Cookbook shape (no attention flag) keeps the radix cache on. It works, but
reuses far less than on a normal model. Same ~360-token shared prefix,
measured here via the `#cached-token` log field:

| Request pattern | Qwen3-0.6B (attention-only) | Qwen3.5-4B (Cookbook shape) |
|---|---|---|
| raw `/generate` continuation: turn-1 prompt **+ its output** + new question | 384 cached | **412 cached** ✅ |
| same long system prompt, different question | 320 cached | **0** |
| exact repeat of an earlier prompt | 320 cached | **0** |
| multi-turn chat via `/v1/chat/completions` | 384 cached | **0** |

Why: a KV cache can be cut at any token, but a Mamba state is one running
summary that cannot be rewound. Under `no_buffer` it is saved only at the end
of a finished request, so a hit needs the new request to **extend that
request's exact token sequence**. Shared system prompts and repeats never do.
Multi-turn chat should, but Qwen3.5's chat template renders an empty
`<think>\n\n</think>` block into the generation prompt and **strips it from
assistant turns in the history**, so turn 2 diverges from turn 1 right after
`assistant\n` (checked with the tokenizer's `apply_chat_template`). The mode
that would save state at prefix boundaries, `extra_buffer`, is refused on XPU
(table above).

So for Qwen3.5 on XPU:

- Use `--disable-radix-cache` with `intel_xpu` attention (fastest first
  request). The cache is worth little here anyway.
- Keep the cache (Cookbook shape) only for a client that sends raw-completion
  continuations, such as an agent appending to one growing prompt via
  `/generate` or `/v1/completions`.
- If prefix reuse across chat turns or shared system prompts is the reason
  for choosing SGLang, it does not hold for these models on XPU. Weigh
  **vllm-xpu-run**.

MTP (NEXTN) works with either configuration; see
`spec-decode-and-multimodal.md`.

## SGLang Cookbook BMG recipes

The Cookbook lists these models for BMG. "Cookbook BMG flags" is what its
generator adds for BMG on top of `--model-path <model>`; translate them before
use (next section). Rows needing 3+ cards were not run on this 2-card host.

| Model (BF16) | Cookbook BMG flags | Measured here |
|---|---|---|
| `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-32B` | `--device xpu --tp 4 --attention-backend intel_xpu --page-size 64 --cuda-graph-backend-decode full --cuda-graph-backend-prefill breakable` (base variants only) | same flag set on Qwen3-0.6B, `--tp 1`: validated, see below |
| `Qwen/Qwen3.5-35B-A3B` | `--device xpu --linear-attn-backend intel_xpu --tp 4 --mem-fraction-static 0.8` | — |
| `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-4B` | `--device xpu --linear-attn-backend intel_xpu --mem-fraction-static 0.8` (`--tp 1`) | Qwen3.5-4B: section above |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | `--trust-remote-code --tp 4 --kv-cache-dtype fp8_e4m3 --device xpu` | — (note the FP8 KV cache; see translation) |
| `google/gemma-4-26B-A4B-it` | `--tp 4 --device xpu --mem-fraction-static 0.75` | at `--tp 2`: serves only with `--mem-fraction-static 0.85 --chunked-prefill-size -1` (16,896-token KV pool); 0.9 → `onecclCommInitRank failed with code 11`; default fraction refused at startup; `--quantization fp8` crashes (SKILL.md quant table) |
| `google/gemma-4-31B-it` | `--tp 4 --device xpu --mem-fraction-static 0.8` | — |
| `meta-llama/Llama-3.3-70B-Instruct` | `--device xpu --tp 8` (~141 GB: 8 cards even at 32 GB each) | — |
| `meta-llama/Llama-3.1-8B-Instruct` | `--device xpu --tp 1` (Instruct only) | yes — translation table below |
| `deepseek-ai/DeepSeek-OCR-2` | `--device xpu --enable-multimodal --tp 1` | — |

**Container wrapper**: the SKILL.md Multi-GPU launch, with
`-e ZE_AFFINITY_MASK=0,1,2,3` (or `0,…,7`) and `-e FI_TCP_IFACE=lo` — the
all-reduce hang in `multi-gpu-tp.md` does not depend on the model. Raise the
verify block's readiness cap for these weight sizes, then run the verify
block before trusting the server.

**TP=4/8 caveat — the Cookbook and the release's own test disagree:** the
Cookbook rows above use `--tp 4`, but the pinned release's
[Llama-3.1-8B XPU test](https://github.com/sgl-project/sglang/blob/v0.5.23/test/registered/xpu/llm_models/test_xpu_llama_3_1_8b.py)
stays at TP=2 because "TP=4 wedges the Level Zero driver during the first
prefill batch on Arc/BMG". That is a driver hang, not the transport hang —
`FI_TCP_IFACE=lo` does not fix it. Run TP=4/8 on cards nobody else is using,
send one request first, and watch `sudo dmesg` for `GT… reset` lines; if the
collective probe passes but serving hangs, retry at a lower `--tp`.

**XPU graphs (Qwen3 rows)**: the Cookbook warns that
`--cuda-graph-backend-decode full --cuda-graph-backend-prefill breakable`
produce corrupted output on older images and need one built with oneAPI
2026.1.3. The pinned image ships that bundle
(`intel-deep-learning-essentials 2026.1.3`). Measured here with both flags on
Qwen3-0.6B, `--tp 1`: 32 concurrent requests all valid, and decode ~2.2×
faster than without graphs.

## Translating an SGLang Cookbook BMG command

The Cookbook model pages have command generators with a **BMG** hardware
option (`arc_b` in the page source); the intro page lists no Intel hardware.
**Do not paste a BMG command as-is.** Most are just `--device xpu [--tp N]`
(table above):

- **No `--attention-backend intel_xpu`** (all rows except Qwen3), so XPU
  resolves to `triton` with `page_size 1`. Single-stream decode is about the
  same, but under load it is much slower (measured here):

  | Llama-3.1-8B-Instruct, `--tp 1` (time relative to skill flags) | first request | 256 tok | 32 concurrent × 128 tok |
  |---|---|---|---|
  | Cookbook (`triton`, page 1) | **5×** | 1.0× | **2.8×** |
  | skill flags (`intel_xpu`, page 64) | 1.0× | 1.0× | 1.0× |

- **No container shape and no `FI_TCP_IFACE=lo`.**
- **`sglang serve --model-path`** is equivalent to `python3 -m
  sglang.launch_server --model` (both verified). The old form only prints a
  deprecation-style `UserWarning`.

Translate by keeping the model, `--tp`, `--mem-fraction-static` and any
model-specific flags. Add the SKILL.md container flags, `-e FI_TCP_IFACE=lo`
when `--tp > 1`, and `--attention-backend intel_xpu --page-size 64`, with these
exceptions:

- **Gemma-4-26B-A4B**: also add `--chunked-prefill-size -1` with `intel_xpu`.
  The pinned release's XPU tests disable chunked prefill because the
  hybrid-attention kernels crash with it enabled.

- **Hybrid models**: with `intel_xpu`, also add `--disable-radix-cache`
  (section above) — or keep the Cookbook's attention default.
- **`--kv-cache-dtype fp8_e4m3`** (the Nemotron row): FP8 KV serves on `triton`
  (quality is model-dependent — `image-and-troubleshooting.md`) but kills
  `intel_xpu` at the first request (`fp8 KV cache requires k_descale
  and v_descale`). Either keep the Cookbook's attention default, or switch the
  KV cache to `bf16` before adding `intel_xpu`.

The generators also **lock options on BMG**: BF16 only (FP8 greyed out);
Qwen3 base variants (30B, 32B) and Llama-3.1 8B Instruct only; Qwen3.5 35B /
9B / 4B only; Gemma 4 31B / 26B-A4B standard checkpoints only; speculative
decoding off (Qwen3.5 MTP, Gemma 4, Llama-3.1's EAGLE3 "latency" mode); and
EP off ("Only Tensor Parallel (TP) is supported on this hardware"). The FP8
and MTP locks are conservative rather than proven failures: runtime FP8
(Qwen3-0.6B) and Qwen3.5 MTP serve validated output here.

The Gemma 4 page also notes for BMG: with `--attention-backend intel_xpu`,
image tokens get bidirectional attention within each image (as in Hugging
Face) only while the longest sequence in the prefill batch, cached prefix
included, fits the 1024-token sliding window; otherwise the whole batch falls
back to causal image attention. For document/OCR work it suggests
`--mm-process-config '{"image":{"max_soft_tokens":560}}'`.

The Cookbook's diffusion pages list BMG too: `FLUX.2-dev` (`--num-gpus 4
--tp-size 4 --component-residency dit=resident,text_encoder=layerwise-offload
--dit-cpu-offload False`) and `Z-Image-Turbo` (one card). SGLang diffusion
serving is outside this skill's text-generation scope.
