# Speculative decoding and multimodal serving

All commands below run inside the SGLang-XPU release image with the
container flag set from SKILL.md (no conda activation, no `setvars.sh`).
`$IMG` is the tag resolved in SKILL.md Step 1. Everything here is verified on
`v0.5.23-xpu` / Arc Pro B70.


## Speculative decoding (EAGLE, EAGLE3, NEXTN, DSpark)

**EAGLE3, EAGLE v1, NEXTN / MTP and DSpark all serve validated language.**
Every checkpoint must be local or the pull dominates startup. NGRAM is
refused at startup on this stack (`topk > 1 is unstable … only supported for
the 'flashinfer' backend`).

The proof that drafting is active is an `accept len` above 1.0 in the
`Decode batch` log lines (a silent fallback to normal decode sits at 1.0).
Those lines print every 40 decode steps, so generate a few hundred tokens —
a short smoke reply never prints one.

| Pair, `--tp 1` unless noted | `accept len` | decode speed-up vs. the same target without spec (512 tokens, warm) |
|---|---|---|
| EAGLE3: Llama-3.1-8B-Instruct + `lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B` | 2.15–3.85 | **2.1–2.7×** (prompt-dependent) |
| EAGLE v1: `NousResearch/Llama-2-7b-chat-hf` + `lmsys/sglang-EAGLE-llama2-chat-7B` | 1.75–2.08 | 1.4× |
| NEXTN: `Qwen/Qwen3.5-4B` (own MTP layer) | 2.5–3.0 | 1.5× |
| DSpark: `Qwen/Qwen3-14B` + `deepseek-ai/dspark_qwen3_14b_block7`, `--tp 2` | 2.4–3.5 | 1.7× |

Loading a draft makes startup ~10–20% slower than the same target alone.

### EAGLE3

```sh
# docker run ... -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 ... \
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --device xpu --tp 1 --attention-backend intel_xpu --dtype bfloat16 \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --page-size 64 \
    --disable-decode-cuda-graph
```

Three ways this launch fails, each reproduced:

- **No `--speculative-draft-model-path`** → dies at startup with a memory
  message (`Raise --mem-fraction-static above 0.941 … draft weights are now
  counted`) instead of naming the missing flag. EAGLE and EAGLE3 always need
  a draft checkpoint trained for the target.
- **No `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`** (a `docker run -e`) →
  startup refuses: the draft's config declares a 2048-token context, shorter
  than the target's 131072.
- `Warning: Target model's context_length (131072) is greater than the
  derived context_length (2048)` at startup is the expected side effect of
  that variable, not a fault.
- **No `--dtype bfloat16`** → the draft checkpoint is fp16; the server reports
  ready, then the scheduler crashes on the first request (`expected mat1 and
  mat2 to have the same dtype … BFloat16 != Half`) and the warmup times out.

### EAGLE v1

```sh
# base: an ungated mirror of Llama-2-7b-chat (see note); draft: LMSYS's
python3 -m sglang.launch_server \
    --model NousResearch/Llama-2-7b-chat-hf \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --device xpu --tp 1 --attention-backend intel_xpu --dtype bfloat16 \
    --speculative-algorithm EAGLE --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --page-size 64 --disable-decode-cuda-graph
```

Three things that cost time here:

- **The base `meta-llama/Llama-2-7b-chat-hf` is `gated=manual`.** A plain
  `HF_TOKEN` is not enough: a *fine-grained* token scoped only to your own
  namespace returns HTTP 200 on `/api/models/...` and **403 on
  `resolve/.../config.json`**, which looks like a broken download rather than a
  permission problem. Either grant the token the global "read access to public
  gated repos" scope (and accept the licence), or sidestep it with an ungated
  mirror such as `NousResearch/Llama-2-7b-chat-hf`.
- **Ungated mirrors often drop `tokenizer.chat_template`.** The NousResearch
  mirror does: `/v1/chat/completions` returns HTTP 400 (`Cannot use chat
  template functions because tokenizer.chat_template is not set`) while
  `/v1/completions` works. An empty `content` from the chat endpoint here is a
  template problem, **not** a spec-decode problem — check the raw body for an
  `error` object before blaming EAGLE. Smoke-test via `/v1/completions`.
- **Not every v1 draft works.** `yuhuili/EAGLE-LLaMA3.1-Instruct-8B` (a
  `pytorch_model.bin`-only repo) on `Llama-3.1-8B-Instruct` starts
  and serves correct text, but every `accept len` is **1.00** — all drafts
  rejected, so no speedup and no error. Only the accept-length check catches
  it. Prefer an `lmsys/sglang-EAGLE*`
  repackage; there is no `lmsys/sglang-EAGLE-LLaMA3.1-Instruct-8B` (HTTP 401),
  so for a Llama-3.1 base use **EAGLE3**, whose repackaged draft does exist.

### NEXTN / MTP (Qwen3.5) — with the hybrid-model flags, not the SKILL.md ones

`NEXTN` is an alias for `EAGLE` that drafts with the target's own MTP layer,
so no draft path is needed; `MTP` itself is not a valid value (`Unknown
speculative algorithm name: MTP`). `Qwen/Qwen3.5-4B` loads its MTP layer as a
1.4 GB `Qwen3_5ForCausalLMMTP` draft. Either Qwen3.5 configuration from
`model-recipes.md` serves it:

```sh
python3 -m sglang.launch_server --model Qwen/Qwen3.5-4B --device xpu --tp 1 \
    --attention-backend intel_xpu --linear-attn-backend intel_xpu \
    --disable-radix-cache --page-size 64 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --host 0.0.0.0 --port 30000
```

| Qwen3.5-4B, `--tp 1` | decode speed-up vs. no MTP (512 tokens, warm) | `accept len` |
|---|---|---|
| no MTP (same flags as the next row) | 1.0× (baseline) | — |
| MTP, `intel_xpu` + `--disable-radix-cache` (above) | 1.5× | 2.5–3.0 |
| MTP, cookbook flags (no attention flag, `--linear-attn-backend intel_xpu`) | 1.5× | 2.2–3.1 |

The SKILL.md page-size flags fail **any** Qwen3.5 launch, MTP or not
(`no_buffer only supports page_size=1` at `--page-size 64`; `MambaComponent
requires page_size=1 … got 128` without it). That is the Mamba radix-cache
rule in `model-recipes.md`, not an MTP limitation.

### DSpark (block-draft speculative decoding)

DSpark drafts a block of tokens with a small model trained for the target.
Run here at `--tp 2` on two cards (add the SKILL.md Multi-GPU wrapper,
including `-e FI_TCP_IFACE=lo`):

```sh
python3 -m sglang.launch_server --model Qwen/Qwen3-14B --device xpu --tp 2 \
    --dtype bfloat16 --attention-backend intel_xpu \
    --speculative-draft-attention-backend intel_xpu \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path deepseek-ai/dspark_qwen3_14b_block7 \
    --page-size 128 --mem-fraction-static 0.85 \
    --max-running-requests 2 --chunked-prefill-size 512 \
    --host 0.0.0.0 --port 30000
```

- No `--trust-remote-code` needed: the draft is a native `Qwen3DSparkModel`
  with no `auto_map`.

### The flag shape every run above used

`--speculative-eagle-topk 1` and decode graph capture off (the
`--cuda-graph-backend-decode` default, `disabled`; EAGLE/EAGLE3 also pass
`--disable-decode-cuda-graph`). Keep both: `topk > 1` is refused on the paged
`intel_xpu` path (the same check that refuses NGRAM), and spec decode was not
run with XPU graph capture.

Bench with and without on the same prompt set + concurrency. Keep
spec-decode only if TPOT improves without TTFT regression.

## Multimodal

- **Serve with the plain text-gen flag set** — verified for
  `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`,
  `Qwen/Qwen3-VL-4B-Instruct` and `google/gemma-4-E2B-it`.
  **`google/gemma-3-4b-it` also needs `--enable-multimodal`**: without it the
  server starts text-only and an `image_url` request returns HTTP 400 (`Model
  only supports text input; received unsupported content type 'image_url'`).
  With it, Gemma-3 vision answers correctly.
- A `data:image/png;base64,…` URL in `image_url` works, so the smoke test
  needs no network or image host.
- **Validate with a changed input, not a fixed one.** Generate a solid
  square, ask for its colour, swap red for blue, and confirm the answer
  *changes* with the image. A single fixed image can be answered by a lucky
  guess; two cannot. **Use 448×448 squares of `(200,20,20)` vs `(20,20,200)`**:
  on 64×64 red squares Qwen2.5-VL-7B (BF16 and AWQ) answers "pink" or
  "orange" while getting blue right, and pure `#FF0000` (or `#00FF00`) is
  "white" even at 448×448 — Transformers on CPU gives the same answer, so it
  is the model, not XPU. `(200,20,20)` at 448×448 answers "red".
  Qwen3-VL-4B is right at either size.
- **Qwen2-VL ignores image content** — `Qwen/Qwen2-VL-7B-Instruct` (BF16) and
  its AWQ variant serve text correctly, but give the same colour for every
  image ("Red" for both squares at 64 px, "Purple" for both at 448 px).
  `--mm-attention-backend sdpa` does not change it. Serve Qwen2.5-VL or
  Qwen3-VL instead.
- Keep the content check length-aware. A correct one-word vision answer
  ("Red") is 3 characters and trips naive `len(content) > 10` garbage
  heuristics — ask for two full sentences if you want the generic assert to
  apply unchanged.
- For any other vision model, validate with the changed-input pattern
  before relying on it. `deepseek-ai/DeepSeek-OCR-2` and Gemma 4's
  image-attention caveat on BMG: `model-recipes.md` (SGLang Cookbook).

## LoRA and two-batch overlap: both work

Both serve valid output on Arc Pro B70 — do not route these to vLLM.

**Static LoRA serving — works.** `meta-llama/Llama-3.1-8B-Instruct` with
`mkopecki/chess-lora-adapter-llama-3.1-8b` (`r=128`):

```sh
python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct \
    --device xpu --tp 1 --attention-backend intel_xpu \
    --disable-overlap-schedule --page-size 64 --dtype bfloat16 \
    --enable-lora --max-lora-rank 128 \
    --lora-paths chess=mkopecki/chess-lora-adapter-llama-3.1-8b
```

- `--max-lora-rank` must cover the adapter's `r`; read it from the adapter's
  `adapter_config.json` rather than guessing.
- `/v1/models` then lists **both** ids: the base and `chess`.
- **Address the adapter as `"model":"<base>:chess"`, or `"model":"<base>"`
  plus `"lora_path":"chess"`.** A bare `"model":"chess"` is accepted on
  `/v1/chat/completions` but **serves the base model** (word-for-word base
  output). The plain base request still works alongside them.
- **The adapter is live — verified.** A factual prompt proves nothing here
  (base and adapter answer it identically). Use the adapter's fine-tuned
  domain: *"Position after 1.e4 e5 2.Nf3. Give only the best next move"* →
  base replies in prose ("The best next move after 1.e4 e5 2.Nf3 is: …"),
  adapter replies `Nc6`. Differing output proves the weights are applied.
- **Runtime hot-swap works.** `POST /load_lora_adapter`
  `{"lora_name":"chess2","lora_path":"<repo>"}` → `success: true`; the new
  name is usable immediately as `lora_path` (answers `Nc6`);
  `POST /unload_lora_adapter` `{"lora_name":"chess2"}` → `success: true`.
- A correct terse answer like `Nc6` is 3 characters and will trip any
  `len(content) > 10` garbage assert — the same trap as a one-word vision
  answer. Judge LoRA output by *difference from base*, not length.

**Two-batch overlap — works, with a required companion flag.**
`--enable-two-batch-overlap` alone aborts at startup:

```
ValueError: When enabling two batch overlap without an EP a2a backend
(moe_a2a_backend='none'), attention DP (--attn-dp-size) is required
```

Adding `--attn-dp-size 2` to `openai/gpt-oss-20b --tp 2` starts cleanly and
returns valid text. The older `--enable-dp-attention --dp-size 2` also still
works, but logs that it is deprecated in favour of `--attn-dp-size`. No
throughput claim — that is `sglang-xpu-bench`'s job.

## Expert parallelism: not supported on BMG

The SGLang Cookbook allows only tensor parallelism on BMG ("Only Tensor
Parallel (TP) is supported on this hardware"), so do not pass `--ep-size`.
Plain `--tp 2` serves the MoE models below. For EP, route to
**vllm-xpu-run**.

## MoE on XPU: what fits on 2× 32 GB

- `openai/gpt-oss-20b` (32 experts, MXFP4, ~13 GB) — serves at `--tp 1`
  and `--tp 2`. The log prints `mxfp4 quantization is
  not fully optimized yet`; that is informational. Add `--reasoning-parser
  gpt-oss`, or the reasoning channel lands in `content` as raw
  `<|channel|>analysis<|message|>…` tokens.
- `google/gemma-4-26B-A4B-it` (128 experts, ~49 GB BF16) — **fits at
  `--tp 2` only with an explicit memory fraction**:
  `--mem-fraction-static 0.85 --chunked-prefill-size -1` serves validated
  text (16,896-token KV pool — short contexts only). The
  default fraction is refused at startup (`Raise --mem-fraction-static above
  0.802`); at `0.9` the server comes up, then the first request fails in
  `onecclCommInitRank failed with code 11` (no device memory left for
  oneCCL). `--quantization fp8` hits the `_scaled_mm` 16-divisibility crash
  (SKILL.md quant table). For real context lengths use the Cookbook's 4-card
  recipe (`model-recipes.md`).

Multi-GPU TP **does** work, but needs `FI_TCP_IFACE=lo` or the first
request might hang forever — see `multi-gpu-tp.md`.

## MLA / DeepSeek: works, but **only** at `--page-size 128`

`deepseek-ai/DeepSeek-V2-Lite-Chat` (MLA: `kv_lora_rank 512`,
`qk_nope_head_dim 128`, `v_head_dim 128`; also MoE, 64 routed + 2 shared
experts) serves at `--tp 2` with a correct multi-fact answer.

```sh
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --device xpu --tp 2 --attention-backend intel_xpu \
    --disable-overlap-schedule --page-size 128 \
    --host 0.0.0.0 --port 30000
```

Two things that cost a session to find:

- **`--page-size 128` is mandatory, and the wrong value fails late.** The XPU
  decode path calls `flash_mla_decode`, which asserts
  `block_num % (128 / PAGE_SIZE) == 0` (`sgl_kernel/attention.py:124`). At
  `page_size 128` the divisor is 1 and the assert cannot fire; at 64 (divisor
  2) and 32 (divisor 4) it does. Both wrong values are accepted by argument
  parsing, are **not** rewritten (`'page_size': 32` appears verbatim in
  `server_args`), serve `/v1/models` happily, and only then kill the
  scheduler on the first request — with a **bare `AssertionError` carrying no
  message** (the client sees HTTP 500 or a dropped connection). The parser
  accepts 16/32 for MLA; the kernel does not.
- **No `--trust-remote-code`, despite `auto_map`.** The repo declares
  `AutoConfig`/`AutoModel` remote classes, yet SGLang's native
  `DeepseekV2ForCausalLM` ran with `trust_remote_code: False`. Try without the
  flag first on any model SGLang implements natively; it is a security
  opt-in, not a formality.

No MLA model has a BMG recipe in the
[SGLang Cookbook](https://docs.sglang.io/cookbook/intro) (DeepSeek-OCR-2 there
has `use_mla: false`). Size with
**model-can-it-fit** first: on 24–32 GB cards only small MLA checkpoints fit —
DeepSeek-V2-Lite is ~31 GB BF16 and needs `--tp 2` on 32 GB cards.

## Opt-in features: XPU graph capture, memory saver, HiCache

None is on by default.

- **XPU graph capture** — less launch overhead, slightly slower start (a short
  capture step): `--cuda-graph-backend-decode full` (captures batch sizes
  1–48) and `--cuda-graph-backend-prefill breakable`. Each alone and both
  together (the Cookbook's Qwen3 BMG flags) return validated text; together,
  decode on Qwen3-0.6B was ~2.2× faster than without graphs, and 32
  concurrent requests were all valid. `tc_piecewise` is an argparse error (`tc_piecewise was removed; select
  breakable, full, or disabled`), and any other decode backend is rewritten
  to `disabled` with a warning (`XPU platform only supports decode backend
  'full'`). Do not combine with speculative decoding (above).
- **Memory saver** — `--enable-memory-saver`, then `POST
  /release_memory_occupation` / `/resume_memory_occupation` hands the device
  back without a restart. Two traps:
  - A **full** release (`{}` body) drops the weights; after resume the server
    answers `!!!!!!!!…`. Either add `--enable-weights-cpu-backup` (full
    release/resume then serves correct text) or release only the KV cache:
    `{"tags":["kv_cache"]}` (also correct after resume).
  - **Release only when idle.** With a request in flight the scheduler asserts
    `release_memory_occupation should be called only when server is idle`,
    the endpoint returns 500, and the server process dies. "Idle" includes the
    server's own startup warmup: run the verify block first, never release
    straight after `/v1/models` answers.
- **HiCache (hierarchical KV cache, device ↔ host RAM)** — works on XPU with
  `--enable-hierarchical-cache --hicache-size <GB>`. **Always pass
  `--hicache-size`** (or `--hicache-ratio`): a bare `--enable-hierarchical-cache`
  on Qwen3-0.6B tried to pin 2× the device KV pool and died at startup with
  `RuntimeError: Failed to allocate 64.00 GiB of pinned host memory`, on a
  125 GB host. With `--hicache-size 8` it starts (`Allocating … host pool`)
  and serves validated text. Proof the host tier is used: with
  `--max-total-tokens 1024 --hicache-ratio 8`, a ~380-token prompt, then a ~600-token unrelated
  prompt that evicts it, then the first prompt again → **320 cached tokens with
  HiCache, 0 without** (default `--hicache-io-backend` / `--hicache-mem-layout`).
