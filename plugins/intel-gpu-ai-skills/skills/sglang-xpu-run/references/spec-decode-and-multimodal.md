# Speculative decoding and multimodal serving

All commands below run inside `lmsysorg/sglang:v0.5.20-xpu` with the
container flag set from SKILL.md (no conda activation, no `setvars.sh`).

## Speculative decoding (EAGLE, EAGLE3, NEXTN)

```sh
# docker run ... -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 ...
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

Measured on one Arc Pro B70 with the command above: accept length ~3.1,
256 tokens in 3.0 s vs 7.4 s for the same target without spec decode.

**NEXTN (a model's built-in MTP layer) does not work on this image.**
There is no `MTP` value — it fails with `Unknown speculative algorithm
name: MTP`. `NEXTN` is an alias for `EAGLE` that drafts with the target's
own MTP layer, so no draft path is needed: `Qwen/Qwen3.5-4B` loads it as a
1.4 GB `Qwen3_5ForCausalLMMTP` draft and reports ready. The first request
then crashes the scheduler (`TypeError: dynamic_func() missing 1 required
positional argument: 'stride_h0_source'`, in the Triton GDN verify kernel)
and the warmup times out — at `--speculative-num-steps` 1 and 3 alike.
`--linear-attn-backend intel_xpu` does not help: verify always runs the
Triton kernel on XPU. This matches the SGLang Cookbook, which offers no MTP
option for Arc B. Upstream `main` has since changed that XPU kernel
(sgl-project/sglang#37213); re-test on a newer image before recommending
NEXTN, and use EAGLE3 with a separate draft meanwhile. The other models
that ship MTP layers (DeepSeek-V3 family, GLM-4.x MoE, …) are far too
large for an Arc card anyway.

Constraints upstream's own XPU spec-decode CI runs under (EAGLE and
EAGLE3 on `intel_xpu`, nightly): `--speculative-eagle-topk 1`,
`--page-size 64`, and decode graph capture off. `topk > 1` is not on the
paged `intel_xpu` path, and spec decode is not combined with XPU graph
capture — keep `--cuda-graph-backend-decode` at its `disabled` default.

Bench with and without on the same prompt set + concurrency. Keep
spec-decode only if TPOT improves without TTFT regression.

## Multimodal

- Exercised by upstream's XPU test suite: `google/gemma-4-E2B-it`,
  `Qwen/Qwen3-VL-2B-Thinking`, DeepSeek-OCR. Serve with the same flags as
  the text-gen quickstart; the OpenAI vision API works
  (`messages[].content[]` with `type: image_url`).
- Also covered on XPU: embedding, classification, rerank, reward, and
  encoder-decoder models — same launch shape, different endpoint.
- Anything else (`Llama-3.2-Vision`, `LLaVA`, …): try it with the
  smoke-test pattern and validate the output text.

## Not on this stack today (route elsewhere)

- LoRA hot-swap for serving.
- Two-batch overlap (`--enable-two-batch-overlap`).
- Expert parallelism on the MXFP4 MoE path.

For these on Intel, use **vllm-xpu-run**.

DeepSeek / MLA is a separate case — kernels and code path ship in the
image but none of the SGLang Cookbook's XPU models uses MLA. See the MLA
bullet in SKILL.md before promising it to a user.

Multi-GPU TP **does** work, but needs `FI_TCP_IFACE=lo` or the first
request might hang forever — see `multi-gpu-tp.md`.

## MLA / DeepSeek: path present, not validated

`intel_xpu` has an MLA
branch (`AttentionArch.MLA`) and the image ships `sglang-kernel-xpu
0.2.0` with `flash_mla_decode` / `flash_mla_prefill`; DeepSeek DSA models
get `dsa_prefill_backend`/`dsa_decode_backend=intel_xpu` and a forced
BF16 KV cache automatically, and MLA additionally accepts
`--page-size 16|32`. But no MLA model is among the XPU models in the
[SGLang Cookbook](https://docs.sglang.io/cookbook/intro) — its
DeepSeek-OCR-2 page is a DeepSeek-V2 decoder with MLA off — and this
skillpack has not served one. Treat it as
"try it, then prove it" — run the content smoke test, and size it with
**model-can-it-fit** first: on a 24–32 GB Arc card only small MLA
checkpoints (e.g. DeepSeek-V2-Lite) fit at all.

