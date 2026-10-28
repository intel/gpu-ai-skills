# Speculative decoding and multimodal serving

All commands below run inside the SGLang-XPU release image with the
container flag set from SKILL.md (no conda activation, no `setvars.sh`).
`$IMG` is the tag resolved in SKILL.md Step 1. Everything here is verified on
`v0.5.22-xpu` / Arc Pro B70 unless a line says otherwise.


## Speculative decoding (EAGLE, EAGLE3, NEXTN)

**EAGLE3 verified on `v0.5.22-xpu`** (Arc Pro B70): the exact command below
came up and returned validated language. Both checkpoints must be local or
the pull dominates startup. NGRAM is unusable; NEXTN / MTP works on
Qwen3.5 with the hybrid-model flags (below).

**EAGLE v1 also verified on `v0.5.22-xpu`**, with upstream's own pair — ready
in 39 s, `accept len` 1.85–2.30 at `accept rate` 0.28–0.29 and ~63 tok/s, so
drafting is genuinely active (an accept length above 1.0 is the proof; a silent
fallback to normal decode would sit at 1.0):

```sh
# base: an ungated mirror of Llama-2-7b-chat (see note); draft: upstream's
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
- **Ungated mirrors often drop `tokenizer.chat_template`.** Then
  `/v1/chat/completions` returns HTTP 400 (`Cannot use chat template functions
  because tokenizer.chat_template is not set`) while `/v1/completions` works
  perfectly. An empty `content` from the chat endpoint here is a template
  problem, **not** a spec-decode problem — check the raw body for an `error`
  object before blaming EAGLE.
- **Not every v1 draft works.** `yuhuili/EAGLE-LLaMA3.1-Instruct-8B` on
  `Llama-3.1-8B-Instruct` hung after `Init parallel ends` — no exception, GPU
  idle ~53 W, draft weights never fetched — and never served in 12 minutes.
  Prefer an `lmsys/sglang-EAGLE*` repackage; note there is no
  `lmsys/sglang-EAGLE-LLaMA3.1-Instruct-8B` (HTTP 401), so for a Llama-3.1 base
  use **EAGLE3**, whose repackaged draft does exist.

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

EAGLE and EAGLE3 need a separate draft checkpoint trained for the target;
without `--speculative-draft-model-path` the launch above dies at startup
(no KV-cache memory left) instead of naming the missing flag. The pairs
upstream's XPU nightly runs are this one (`EAGLE3`) and
`meta-llama/Llama-2-7b-chat-hf` + `lmsys/sglang-EAGLE-llama2-chat-7B`
(`EAGLE`). Two knobs these drafts need, both from upstream's test fixture:

- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` (a `docker run -e`) — the
  draft's config declares a 2048-token context, and without it startup
  refuses the target's 131072.
- `--dtype bfloat16` — the draft checkpoint is fp16. Without it the server
  reports ready, then the scheduler crashes on the first request (`expected
  mat1 and mat2 to have the same dtype … BFloat16 != Half`) and the warmup
  times out.

Measured on one Arc Pro B70 with the command above: accept length ~2.6,
256 tokens in 3.2 s at v0.5.22 (v0.5.21: ~3.1, 3.0 s), vs 7.4 s for the same
target without spec decode (v0.5.21).

**NEXTN / MTP works — with the hybrid-model flags, not the SKILL.md ones.**
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

| Qwen3.5-4B, `--tp 1` | ready | 256 tok, warm | `accept len` |
|---|---|---|---|
| no MTP (same flags) | 31 s | 6.2 s | — |
| MTP, `intel_xpu` + `--disable-radix-cache` (above) | 34 s | 3.5 s | 2.3–3.0 |
| MTP, cookbook flags (no attention flag, `--linear-attn-backend intel_xpu`) | 34 s | 3.4 s | 2.7–2.9 |

That is ~1.8× faster decode, with accept length well above 1.0, so drafting
is really active. Same result on v0.5.21. **An earlier version of this file
said NEXTN had "no working configuration"; that was a misdiagnosis.** The
startup failures it recorded (`no_buffer only supports page_size=1` at
`--page-size 64`, and `MambaComponent requires page_size=1 … got 128`
otherwise) come from the Mamba radix-cache rule, which hits *any* Qwen3.5
launch with the SKILL.md page-size flags, MTP or not. They have nothing to do
with MTP. Other models shipping MTP layers (DeepSeek-V3 family, GLM-4.x MoE)
are too large for an Arc card anyway.

Constraints upstream's own XPU spec-decode CI runs under (EAGLE and
EAGLE3 on `intel_xpu`, nightly): `--speculative-eagle-topk 1`,
`--page-size 64`, and decode graph capture off. `topk > 1` is not on the
paged `intel_xpu` path, and spec decode is not combined with XPU graph
capture — keep `--cuda-graph-backend-decode` at its `disabled` default.

Bench with and without on the same prompt set + concurrency. Keep
spec-decode only if TPOT improves without TTFT regression.

## Multimodal

- **Verified on `v0.5.22-xpu`** (Arc Pro B70): `Qwen/Qwen2.5-VL-7B-Instruct`
  and `Qwen/Qwen3-VL-4B-Instruct`, ready in ~28 s with the plain text-gen flag set and no multimodal-specific
  flags. A `data:image/png;base64,…` URL in `image_url` works, so the smoke
  test needs no network or image host. Validate with a **changed** input, not
  a fixed one: ask for the colour of a generated solid square, swap red for
  blue, and confirm the answer tracks ("a solid, vibrant shade of blue"). A
  single fixed image can be answered by a lucky guess; two cannot. Use an
  off-pure red such as `(200,20,20)`: Qwen2.5-VL calls a 64×64 `#FF0000`
  square "entirely white" on both v0.5.21 and v0.5.22 (a model quirk; Qwen3-VL
  gets it right), which looks exactly like a broken vision path.
- **Qwen2-VL ignores image content** — `Qwen/Qwen2-VL-7B-Instruct` (BF16) and
  its AWQ variant serve text correctly, but give the same colour for every
  image ("completely white" at 64 px, "red"/"pink" at 448 px). Identical on
  v0.5.21 and v0.5.22, and `--mm-attention-backend sdpa` does not change it.
  Serve Qwen2.5-VL or Qwen3-VL instead.
- Keep the content check length-aware. A correct one-word vision answer
  ("Red") is 3 characters and trips naive `len(content) > 10` garbage
  heuristics — ask for two full sentences if you want the generic assert to
  apply unchanged.
- Exercised by upstream's XPU test suite: `google/gemma-4-E2B-it`,
  `Qwen/Qwen3-VL-2B-Thinking`, DeepSeek-OCR. Serve with the same flags as
  the text-gen quickstart; the OpenAI vision API works
  (`messages[].content[]` with `type: image_url`).
- Also covered on XPU: embedding, classification, rerank, reward, and
  encoder-decoder models — same launch shape, different endpoint.
- Anything else (`Llama-3.2-Vision`, `LLaVA`, …): try it with the
  smoke-test pattern and validate the output text.

## LoRA and two-batch overlap: both work at v0.5.22

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
  `/v1/chat/completions` but **serves the base model** — word-for-word base
  output on v0.5.21 and v0.5.22 — so an earlier claim that it routes was wrong.
  The plain base request still works alongside them.
- **The adapter is live — verified.** A factual prompt proves nothing here
  (base and adapter answer it identically). Use the adapter's fine-tuned
  domain: *"Position after 1.e4 e5 2.Nf3. Give only the best next move"* →
  base replies in prose ("The current position is: 1. e4 e5 …"), adapter
  replies `Nc6`. Differing output proves the weights are applied.
- **Runtime hot-swap works** (v0.5.22), which is the capability SKILL.md used
  to route to vLLM for. `POST /load_lora_adapter`
  `{"lora_name":"chess2","lora_path":"<repo>"}` → `success: true`; the new
  name is usable immediately as `lora_path`; `POST /unload_lora_adapter`
  `{"lora_name":"chess2"}` → `success: true`.
- A correct terse answer like `Nc6` is 3 characters and will trip any
  `len(content) > 10` garbage assert — the same trap as a one-word vision
  answer. Judge LoRA output by *difference from base*, not length.

**Two-batch overlap — works, with a required companion flag.**
`--enable-two-batch-overlap` alone aborts at startup:

```
ValueError: When enabling two batch overlap without an EP a2a backend
(moe_a2a_backend='none'), attention DP (--attn-dp-size) is required
```

Adding `--attn-dp-size 2` to `openai/gpt-oss-20b --tp 2` starts cleanly
(`enable_two_batch_overlap: True`, `attn_dp_size: 2` in `server_args`) and
returns valid text. The older `--enable-dp-attention --dp-size 2` still works
at v0.5.22 but logs a deprecation warning pointing to `--attn-dp-size`. No
throughput claim — that is `sglang-xpu-bench`'s job.

## Still not on this stack (route to vllm-xpu-run)

- **Expert parallelism on the MXFP4 MoE path.** `--ep-size 2` on
  `openai/gpt-oss-20b --tp 2` brings the EP ranks up (`TP0 EP0` / `TP1 EP1`
  in the log, past `Init parallel ends`) and reaches `/v1/models`, but no
  request returns a valid completion. At v0.5.21 it stalled silently; at
  v0.5.22 the first prefill dies with `UR_RESULT_ERROR_DEVICE_LOST` and
  `dmesg` shows a compute-engine reset plus a devcoredump on **both** cards —
  a GPU hang that other jobs sharing the cards will feel. Baseline
  `--tp 2` without `--ep-size` on the same model serves fine, so this is EP,
  not the MoE or the image.

## MoE on XPU: what fits on 2× 32 GB

- `openai/gpt-oss-20b` (32 experts, MXFP4, ~13 GB) — serves at `--tp 2`,
  ready in ~53 s. The log prints `mxfp4 quantization is not fully optimized
  yet`; that is informational.
- `google/gemma-4-26B-A4B-it` (128 experts, ~49 GB BF16) — **does not fit**.
  BF16 at `--tp 2` fails with `Loaded weights leave no GPU memory for the KV
  cache`, and `--quantization fp8` hits the `_scaled_mm` 16-divisibility
  crash (SKILL.md quant table). Needs more VRAM or a pre-quantized
  checkpoint, not a flag change.

Multi-GPU TP **does** work, but needs `FI_TCP_IFACE=lo` or the first
request might hang forever — see `multi-gpu-tp.md`.

## MLA / DeepSeek: works, but **only** at `--page-size 128`

**Validated on v0.5.22** (2× Arc Pro B70, 2026-10-16; first on v0.5.21):
`deepseek-ai/DeepSeek-V2-Lite-Chat` (MLA: `kv_lora_rank 512`,
`qk_nope_head_dim 128`, `v_head_dim 128`; also MoE, 64 routed + 2 shared
experts), ready in ~40 s, correct multi-fact answer.

```sh
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Lite-Chat \
    --device xpu --tp 2 --attention-backend intel_xpu \
    --disable-overlap-schedule --page-size 128 \
    --host 0.0.0.0 --port 30000
```

Two things that cost a session to find:

- **`--page-size 128` is mandatory, and the wrong value fails late.** The XPU
  decode path calls `flash_mla_decode`, which asserts
  `block_num % (128 / PAGE_SIZE) == 0` (`sgl_kernel/attention.py:112`). At
  `page_size 128` the divisor is 1 and the assert cannot fire; at 64 (divisor
  2) and 32 (divisor 4) it does. Both wrong values are accepted by argument
  parsing, are **not** rewritten (`'page_size': 32` appears verbatim in
  `server_args`), serve `/v1/models` happily, and only then kill the
  scheduler on the first decode — with a **bare `AssertionError` carrying no
  message**. So the earlier claim that MLA "additionally accepts
  `--page-size 16|32`" is wrong: the parser accepts them, the kernel does not.
- **No `--trust-remote-code`, despite `auto_map`.** The repo declares
  `AutoConfig`/`AutoModel` remote classes, so the SKILL.md rule ("required
  only if `config.json` has an `auto_map`") would say yes. It is not needed —
  SGLang has a native `DeepseekV2ForCausalLM` and ran with
  `trust_remote_code: False`. Try without the flag first on any model SGLang
  implements natively; it is a security opt-in, not a formality.

Still true: the image ships `flash_mla_decode` / `flash_mla_prefill`, DeepSeek
DSA models get `dsa_prefill_backend`/`dsa_decode_backend=intel_xpu` with a
forced BF16 KV cache, and no MLA model appears among the XPU models in the
[SGLang Cookbook](https://docs.sglang.io/cookbook/intro). Size with
**model-can-it-fit** first: on 24–32 GB cards only small MLA checkpoints fit —
DeepSeek-V2-Lite is ~31 GB BF16 and needs `--tp 2` on 32 GB cards.

## Opt-in features: XPU graph capture, memory saver, P/D disaggregation

All verified on `v0.5.22-xpu` / Arc Pro B70 unless noted. None is on by
default.

- **XPU graph capture** — less launch overhead, slower start:
  `--cuda-graph-backend-decode full` (captures bs 1–48 in ~2 s on Qwen3-0.6B)
  and `--cuda-graph-backend-prefill breakable` (~5 s). Both return validated
  text. `tc_piecewise` was **removed** in v0.5.22 — now an argparse error:
  `tc_piecewise was removed; select breakable, full, or disabled`. Do not
  combine with speculative decoding (above).
- **Memory saver** — `--enable-memory-saver`, then `POST
  /release_memory_occupation` / `/resume_memory_occupation` hands the device
  back without a restart. Two traps:
  - A **full** release (`{}` body) drops the weights; after resume the server
    answers `!!!!!!!!…`. Either add `--enable-weights-cpu-backup` (full
    release/resume then serves correct text) or release only the KV cache:
    `{"tags":["kv_cache"]}`.
  - **Release only when idle.** With requests in flight the scheduler asserts
    `release_memory_occupation should be called only when server is idle`,
    the endpoint returns 500, and the server process dies. "Idle" includes the
    server's own startup warmup: run the verify block first, never release
    straight after `/v1/models` answers.
- **P/D disaggregation** — via NIXL; needs `pip install nixl sglang-router`
  and `UCX_POSIX_USE_PROC_LINK=n` in the container. Not run here.

Upstream's current feature matrix and limitations:
<https://docs.sglang.io/docs/hardware-platforms/xpu>.
