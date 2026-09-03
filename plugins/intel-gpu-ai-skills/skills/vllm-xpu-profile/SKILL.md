---
name: vllm-xpu-profile
description: Profile a running vLLM-XPU server with torch.profiler around a window of real requests, either via /start_profile and /stop_profile HTTP endpoints or via vllm bench --profile for offline runs. Use to find the dominant op under real concurrent traffic. Not for pure PyTorch (use torch-xpu-profile), SYCL kernel-level signal (use xpu-profile-unitrace), throughput numbers (use vllm-xpu-bench), or non-vLLM servers.
---

# vllm-xpu-profile

Profile a vLLM-XPU server with `torch.profiler` to see scheduler,
KV manager, attention backend, and batching alongside XPU op
timeline. For pure-PyTorch traces use **torch-xpu-profile**; for
SYCL kernel level use **xpu-profile-unitrace**.

## Modes

| Mode | When |
|---|---|
| **A** — running server + HTTP `/start_profile` ... `/stop_profile` | Real-traffic capture; see scheduler / KV manager / batching behaviour. |
| **B** — `vllm bench latency / throughput --profile` | Offline; no network endpoint needed. |

## Mode A — server + HTTP bracket

Launch the server with profiler flags. Always launch from the official upstream
`vllm/vllm-openai-xpu:latest` image — even if a running container or host
process is using a different image (e.g. `intel/llm-scaler-vllm`), do
**not** reuse that image for the profiling server; those stacks are out
of scope (see **What this skill does NOT cover**).

```sh
docker run -d --name vllm-xpu-prof \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
   -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
   -e http_proxy -e https_proxy -e no_proxy \
   -e HF_TOKEN \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD/traces:/work/traces" \
    -p 8000:8000 \
   vllm/vllm-openai-xpu:latest \
   Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 --enforce-eager --max-model-len 4096 \
        --profiler-config.profiler=torch \
        --profiler-config.torch_profiler_dir=/work/traces
```

Wait for `Application startup complete`, then bracket your window:

```sh
# Warmup (don't profile this) — kernel cache hot
for i in 1 2 3; do
    curl -s http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
             "messages":[{"role":"user","content":"hi"}],"max_tokens":32}' >/dev/null
done

curl -X POST http://localhost:8000/start_profile

# Workload to characterise
for i in $(seq 1 8); do
    curl -s http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
             "messages":[{"role":"user","content":"Write a short paragraph."}],
             "max_tokens":128}' &
done; wait

curl -X POST http://localhost:8000/stop_profile
ls traces/
```

`Qwen/Qwen2.5-1.5B-Instruct` is public, so `HF_TOKEN` is optional;
it is forwarded for gated/private replacements and anonymous Hub rate
limits. `-e HF_TOKEN` (no `=value`) passes through the host's already-exported
variable without putting the token in the `docker` process argv, where
`ps`/`/proc` on a shared host could expose it. The proxy lines are
required on hosts whose containers cannot reach the Hub directly.
Docker ignores unset proxy variable names.

Cleanup after the trace has flushed:

```sh
docker stop vllm-xpu-prof && docker rm vllm-xpu-prof
```

Three artifacts written per profile (allow a few seconds for flush):

- `<timestamp>-rank-<N>.<timestamp>.pt.trace.json.gz` — per-worker
  Chrome trace. Drag-and-drop into <https://ui.perfetto.dev>;
  Perfetto handles `.json.gz` directly.
- `<container>_<N>.async_llm.<timestamp>.pt.trace.json.gz` — async
  engine trace (smaller).
- `profiler_out_<N>.txt` — **read this first**, sort-by-XPU-time
  table of top kernels. Most "what's slow" questions are answered
  by its top 5 rows:

  ```
  Name                     Self XPU %  XPU total   # of Calls
  gemm_kernel              28.50%      26.232ms    4753
  aten::mm                 21.79%      20.061ms    3577
  _C::fused_add_rms_norm   14.11%      12.988ms    2352
  _vllm_fa2_C::varlen_fwd   6.70%      17.979ms    1176
  _C::rotary_embedding      6.70%       6.172ms    1176
  Memcpy H2D                2.11%       1.945ms     412
  ```

## Mode B — offline bench with `--profile`

```sh
docker run --rm \
   --entrypoint vllm \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
   -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
   -e http_proxy -e https_proxy -e no_proxy \
   -e HF_TOKEN \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD/traces:/work/traces" \
   vllm/vllm-openai-xpu:latest \
   bench latency \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 \
        --enforce-eager \
        --num-iters-warmup 2 --num-iters 5 \
        --profile \
        --profiler-config.profiler=torch \
        --profiler-config.torch_profiler_dir=/work/traces
```

## What to look for

1. **Step boundary markers** — `EngineCore.step` spans. Time per
   step / batch size = TPOT contribution per request.
2. **Attention path** — confirm kernel name matches your
   `--attention-backend`. If you passed `TRITON_ATTN` but trace
   shows `flash_attention_xpu`, confirm the flag reached the
   engine:
   ```sh
   docker logs <name> 2>&1 | grep -E "attention_backend|Using.*backend"
   ```
3. **Gaps between steps** — wide white bands at boundaries mean
   scheduler / KV manager is on the critical path. Normal under
   `--enforce-eager`; tighter without it.
4. **Quant kernel names** — for `--quantization fp8` expect
   `fp8_gemm_w8a16`; for `awq` expect `int4_gemm_w4a8`. Seeing
   `int4_gemm_w4a16` with AWQ -> silent fall-through (vLLM #38064).
5. **Per-rank spans** — multi-GPU `-tp N` shows N timelines. Skew
   between ranks means a slow rank is gating throughput.

## Common errors

- `torch_profiler_dir is only applicable when profiler is set to 'torch'` -> add `--profiler-config.profiler=torch`.
- `/start_profile` 404 -> server launched without profiler-config flags; restart.
- Empty trace -> too few requests inside the bracket; send 5–10+.
- Trace doesn't appear after `/stop_profile` -> flush is async; wait a few seconds, or check server logs.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Pin to one XPU. |
| `PYTORCH_ENABLE_XPU_FALLBACK=0` | Error on CPU-fallback so it can't hide. |
| `KINETO_LOG_LEVEL=INFO` | Diagnose incomplete trace export. |

(vLLM profiler config is via CLI flags `--profiler-config.*`, not env vars.)

## What this skill does NOT cover

- Pure-PyTorch profiling (no server) -> **torch-xpu-profile**.
- SYCL-kernel events / hardware metrics -> **xpu-profile-unitrace**.
- Fixing a hot kernel — out of scope.
- `intel/llm-scaler-vllm` images — out of scope; launch the profiling
   server from upstream `vllm/vllm-openai-xpu:latest` instead.

## References

- vLLM profiling guide: <https://docs.vllm.ai/en/latest/contributing/profiling/>
- Perfetto trace viewer: <https://ui.perfetto.dev>
