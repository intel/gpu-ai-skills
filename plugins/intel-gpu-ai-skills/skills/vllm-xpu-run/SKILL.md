---
name: vllm-xpu-run
description: Serve a Hugging Face safetensors model on an Intel GPU with upstream vLLM-XPU's OpenAI-compatible API, or check whether a model or architecture is currently documented on XPU. Covers live support lookup, image choice, container launch, known serve-flag requirements, model-impl fallback, and attention/quant compatibility. Use to launch /v1/chat/completions or /v1/completions, troubleshoot a launch, or check model support. Not for choosing the best quantization, KV dtype, DP/TP layout, context, or concurrency (use model-config-recommend).
---

# vllm-xpu-run

Use the official upstream `vllm/vllm-openai-xpu:latest` image.

Upstream vLLM has a first-class XPU backend. The CLI is identical
to the CUDA build (`vllm serve <model>`); device is detected from
`torch.xpu.is_available()`. There is no `--device xpu` flag.

The official image already sets `ENTRYPOINT ["vllm", "serve"]`.
Pass the model id and serve flags directly after the image name. Do
not append another `vllm serve`: that produces
`vllm serve vllm serve <model>` and the container exits with code 2.

## Current supported models and architectures

When the user asks which models vLLM supports on Intel XPU, fetch the
current upstream page at request time:

<https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/>

Do not answer from memory and do not copy a static model list into this
skill. Report both the explicitly listed **Model** rows and the
**Architecture** column, because the recommended model table is not an
exhaustive checkpoint allowlist.

For a specific unlisted Hugging Face model, read its current
`config.json` and compare every value in `architectures` with the live
page's Architecture column. Report the evidence precisely:

- Exact model row → explicitly documented on the fetched page.
- Architecture match only → the architecture is documented on XPU, but
  this exact checkpoint is not explicitly validated by the page; perform
  a generation smoke test before claiming full support.
- Neither matches → not documented by the current XPU page; this is not
  proof of impossibility.

Include the source URL and retrieval date in the answer. If the page
cannot be fetched, report that failure and offer to retry rather than
substituting a remembered list. Do not infer XPU support merely from
general vLLM, CUDA, or Transformers support.

## CUDA → XPU cheat sheet

| CUDA convention | XPU convention |
|---|---|
| `vllm/vllm-openai:latest` | `vllm/vllm-openai-xpu:latest` |
| `--gpus all` | `--device /dev/dri` + `-v /dev/dri/by-path:/dev/dri/by-path:ro` + `--ipc=host` |
| `--dtype auto` | `--dtype bfloat16` (explicit) |
| CUDA graphs default | `--enforce-eager` |
| `--tensor-parallel-size N` | same; pin N XPUs in `ZE_AFFINITY_MASK` |

## Quickstart (single GPU)

Confirm the target image tag and model with the user before running
the `docker run` command — container launches bind host devices and
download multi-GB weights.

Pull the official XPU image from
<https://hub.docker.com/r/vllm/vllm-openai-xpu>. The examples use
`:latest`; pin an immutable digest for reproducible deployments.
Generated plans should call `scripts/emit_launch.sh` instead of
copying the template manually — that keeps image policy, proxy env
propagation, quantization flags, and multi-XPU topology in one place.

```sh
docker run -d --name vllm-xpu \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -e http_proxy -e https_proxy -e no_proxy \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p 8000:8000 \
    vllm/vllm-openai-xpu:latest \
    Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 \
        --enforce-eager \
        --block-size=64 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85
```

Wait for `Application startup complete`, then test:

```sh
curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
         "messages":[{"role":"user","content":"Hi."}],"max_tokens":32}'
```

Return the response content or a short summary to the user. A running
HTTP server without a successful generation is not a validated
deployment.

Cleanup: `docker stop vllm-xpu && docker rm vllm-xpu`.

**Drop `--rm` on first launches** so logs survive a crashed init.

To serve on a remote Intel GPU host over ssh (the local machine →
remote-box workflow), see `references/remote-deploy.md`.

## Flag rationales

| Flag | Why |
|---|---|
| `--dtype bfloat16` | Battlemage runs bf16 better than fp16; `auto` may pick fp16 from the checkpoint config. Set for unquantized serving only — combining with `--quantization` produces conflicts. |
| `--enforce-eager` | Conservative default. Graph capture / `torch.compile` on XPU is experimental. After a stable eager run, drop it and re-bench; keep only if TPOT/TTFT improve and content stays correct. |
| `--block-size=64` | Validated default for the XPU paged-attention path on Battlemage. Bench higher (128, 256) once 64 is correct. |
| `--gpu-memory-utilization 0.85` | Default 0.92 fails on workstations with active GUI sessions. Drop to 0.70 with browsers open; raise to 0.92 on headless servers. |
| `-v /dev/dri/by-path:/dev/dri/by-path:ro` | Some oneCCL/device-discovery configurations scan the host's `/dev/dri/by-path` symlinks even for a single-GPU launch. Include this read-only mount to support those configurations; if it is omitted, affected images can abort at engine initialization with `opendir failed: could not open device directory`. |
| `--max-model-len 4096` | KV cache is allocated up-front. Start at 4096, raise in 2× steps until OOM, back off one step. |
| `--trust-remote-code` | **Security opt-in** — permits arbitrary Python from the model repo to run in your engine. Set only when you trust the publisher. |
| `--disable-sliding-window` | Workaround when SWA produces incorrect output for a specific (model, image) combination. Don't apply blindly — disabling SWA on a model designed for it inflates KV memory. |
| `--model-impl transformers` | Fallback for `Model architectures ['<X>'] are not supported`. Slower but correct. Upgrade transformers in the container if that also fails. |

For pooling / embedding / reranker, serve with `--dtype bfloat16`
or `--quantization fp8`.

## Env vars

vLLM's env-var surface is image-version-specific. After launch,
verify there are no silent rejects:

```sh
docker logs <name> 2>&1 | grep -i "Unknown vLLM environment"   # must be empty
```

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the server sees. |
| `HF_TOKEN` | HF auth. |
| `TRITON_CACHE_DIR` | Persist compiled XPU Triton kernels. |
| `CCL_ZE_IPC_EXCHANGE=pidfd` | oneCCL IPC over Docker PID namespace (multi-GPU). |
| `VLLM_LOGGING_LEVEL=DEBUG` | Verbose engine logs. |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Required — `fork` deadlocks oneCCL init on XPU. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Set when extending RoPE past the model card's default. |
| `VLLM_MLA_DISABLE=1` | Workaround for MLA models (DeepSeek-V2/V3, MiniMax-Text-01); no-op otherwise. |

## Quantization, attention backend selection

See `references/quantization.md` for the full
(quant × KV dtype × attention backend) pairing table and
live-kernel verification, plus FP8 / AWQ / GPTQ / MXFP4 / AutoRound
specifics.

## Multi-GPU, tuning, speculative decoding

See `references/multi-gpu-and-tuning.md` for tensor parallel,
oneCCL / XCCL collective env vars, `--block-size` /
`--max-num-batched-tokens` sweeps, speculative decoding
(EAGLE3 / MTP / n-gram), and legacy env-var aliases for older
images.

### Attention backend gotcha (read before quantising)

`--kv-cache-dtype fp8` and the W4A8 quant kernels (AWQ / GPTQ /
MXFP4) **require `--attention-backend TRITON_ATTN`**. The default
FA-XPU backend does not implement fp8 KV — vLLM exits with
`NotImplementedError` at engine init. Full pairing table in
`references/quantization.md`.

## Common errors

- `Model architectures ['<X>'] are not supported` → add
  `--model-impl transformers`. If still failing, upgrade
  transformers in the container or use a newer image.
- `RuntimeError: Cannot find any XPU devices` → container missing
  GPU access; verify with `xpu-smi discovery` inside the container.
- `Free memory on device xpu:0 ... is less than desired GPU memory
  utilization` → drop `--gpu-memory-utilization` to 0.85 or 0.70.
- Other OOM at engine init → `--max-model-len` too large; halve.
- OOM after a few requests → cap `--max-num-seqs 16`.
- Server hangs at `Detected platform: xpu` → oneCCL init. Check
  `--ipc=host`; for multi-GPU, check both XPUs in `ZE_AFFINITY_MASK`.
- Crash at init with `oneCCL: ze_fd_manager.cpp ... init_device_fds:
  opendir failed: could not open device directory` → `/dev/dri/by-path`
  not visible in the container. Add `-v /dev/dri/by-path:/dev/dri/by-path:ro`.
  Fires on single-GPU too (the worker `all_reduce`s at init). Setting
  `CCL_ZE_IPC_EXCHANGE=pidfd` alone does not fix it — the drmfd fallback
  still scans by-path.
- `tensor parallel size N is not allowed` → `ZE_AFFINITY_MASK` has
  fewer than N XPUs.
- HTTP 400 "model not found" → `model` field in JSON must match
  `/v1/models` exactly.
- Gibberish output → dtype mismatch. Force `--dtype bfloat16`. Last
  resort: `--override-attention-dtype float32`.
- Triton compile error on first request → set `TRITON_CACHE_DIR`
  to a mounted volume so the next run starts hot.

## Verifying device placement

```sh
xpu-smi dump -d 0 -m 5,18 -i 1 | head -5
```

Memory should sit at gigabytes once the engine is ready. <100 MiB
while the server reports ready means the model loaded on CPU.

## What this skill does NOT cover

- Choosing the best quantization, KV dtype, DP/TP layout, context, or
  concurrency → **model-config-recommend**. Questions such as "How
  should I configure vLLM on my Arc cards?" belong there even when the
  user also names a model.
- SGLang serving → **sglang-xpu-run**.
- Pure PyTorch / Transformers → **torch-xpu-run**.
- Throughput / TTFT / TPOT measurement → **vllm-xpu-bench**.
- Profile-level slowness → **vllm-xpu-profile**.
- SYCL kernel fixes — out of scope.
- `intel/llm-scaler-vllm` images — out of scope.

## References

- `references/quantization.md` — quant × KV × attention backend
- `references/multi-gpu-and-tuning.md` — TP, tuning, spec-decode, legacy envs
- `references/remote-deploy.md` — serve + verify on a remote Intel GPU host over ssh
- Image source: <https://hub.docker.com/r/vllm/vllm-openai-xpu>
- vLLM XPU installation: <https://docs.vllm.ai/en/latest/getting_started/xpu-installation.html>
- vLLM Arc Pro B-series blog: <https://blog.vllm.ai/2025/11/11/intel-arc-pro-b.html>
- vLLM #38064 (W4A8 fall-through): <https://github.com/vllm-project/vllm/issues/38064>
