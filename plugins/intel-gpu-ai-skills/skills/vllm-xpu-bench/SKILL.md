---
name: vllm-xpu-bench
description: Benchmark a **running vLLM-XPU OpenAI-compatible server** on an Intel GPU using `vllm bench`. Measures TTFT (time-to-first-token), TPOT (time-per-output-token), ITL (inter-token latency), end-to-end latency, and throughput under concurrency. Covers online (`vllm bench serve`) and offline (`vllm bench throughput`) modes; concurrency sweeps and quant comparison live in `references/sweep-and-compare.md`. Use after **vllm-xpu-run** when the user asks "how fast is this?".
---

# vllm-xpu-bench

`vllm bench` is the same CLI on Intel as on CUDA. The XPU-specific
levers are the serve-side flags from **vllm-xpu-run**
(`--enforce-eager`, `--max-model-len`, `--gpu-memory-utilization`,
`--block-size=64`). Pure measurement; for fixes see profiling
skills.

## Preflight — find the container and verify the server

Before benchmarking, identify the running vLLM container. The bench
client **must always run inside the container via `docker exec`** —
never on the host. This shares the engine's network namespace and
reuses the already-loaded tokenizer cache.

### 1. Find the vLLM container name

```sh
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}' | grep -iE 'vllm|8000'
```

This gives you `<container-name>`. If multiple containers appear, ask
the user which one to bench. **Do not proceed without a confirmed
container name.**

### 2. Verify the API is reachable and confirm the server is vLLM

```sh
docker exec <container-name> curl -s http://127.0.0.1:8000/v1/models
```

Check the `owned_by` field in the response:

```sh
docker exec <container-name> curl -s http://127.0.0.1:8000/v1/models | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['owned_by'])"
```

- `owned_by: "vllm"` → correct server, proceed.
- `owned_by: "sglang"` → **stop**. This is a SGLang server; use the
  **sglang-xpu-bench** skill instead.
- Any other value → ask the user to confirm the server type before
  proceeding.

Note the `id` field — you need it for `--model` and
`--served-model-name` in the bench command.

If the API is unreachable, check container logs:
```sh
docker logs <container-name> 2>&1 | tail -30
```

### 3. No server running — start one

If no vLLM container is running, launch one per **vllm-xpu-run**.
Confirm the model and image tag with the user before starting.
Use `--no-enable-prefix-caching` and `--disable-log-stats` for
fair benchmarks.

```sh
docker run -d --name <container-name> \
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
    <model-id> \
        --dtype bfloat16 \
        --enforce-eager \
        --block-size=64 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --no-enable-prefix-caching \
        --disable-log-stats
```

Wait for `Application startup complete` in `docker logs`, then
confirm:

```sh
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### 4. Verify XPU is in use (optional)

Only needed when starting a **new** server (step 3). If the server
was already running and `/v1/models` returned a model list, XPU is
confirmed — skip this step.

```sh
xpu-smi dump -d 0 -m 5,18 -i 1 -n 3
```

VRAM should be non-zero. If zero, the model loaded on CPU —
check `--device /dev/dri` and `ZE_AFFINITY_MASK`.

## Metrics

- **TTFT** — wall time to first generated token. Dominated by
  prefill; sensitive to prefix caching.
- **TPOT** — mean wall time per generated token after the first.
  Dominated by decode kernel + KV bandwidth.
- **ITL** — per-token inter-arrival; same source as TPOT but
  reported per-token so percentiles are meaningful.
- **E2EL** — wall time of one request, issue to last token.
- **Throughput** — `output_tokens / wall_seconds`.

## Modes

| Mode | Subcommand | When |
|---|---|---|
| Online (server up) | `vllm bench serve` | Default — client-observed metrics. |
| Offline (no server) | `vllm bench throughput` | Quick batch throughput; loads the model in the bench process. |
| Other | `vllm bench latency / sweep / startup / mm-processor` | `vllm bench --help`. |

## Canonical client flag set

For apples-to-apples decode/throughput numbers, pin these flags:

```sh
vllm bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --model <ID> --served-model-name <ID> \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 1024 \
    --num-prompts 10 --max-concurrency <C> \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 90,99 \
    --ignore-eos
```

Why each flag:

- `random` + fixed in/out lengths → removes prompt-distribution
  variance.
- `--num-prompts 10` → enough for median and p90/p99 at typical XPU
  throughput. Raise to 200+ only when specifically asked for tight
  p99 repeatability.
- `--request-rate inf` → saturate the server (throughput-at-
  saturation, not rate-limited TPOT).
- `--ignore-eos` → short completions otherwise terminate early and
  inflate per-token rate.
- `--percentile-metrics ttft,tpot,itl,e2el` -> the four numbers that
  describe a serving stack.
- `--backend openai` + `--endpoint /v1/completions` -> bypass the
  chat-template and any structured-output parser. See next note.

**Use `/v1/completions` for random-token datasets.** For gpt-oss
or any Harmony-based model, always use `/v1/completions` regardless
of dataset: the Harmony streaming parser can trip on
control-token-like substrings in any response. For non-Harmony
models with random tokens, the chat endpoint wraps each sequence
in system/role tokens, inflating input length and skewing throughput.
Use `--backend openai-chat` only for non-Harmony models *and* a
real-prompt dataset (`sharegpt`, `sonnet`) where you specifically
want to measure chat-template overhead.

Add `--trust-remote-code` only when the model declares repo-local
code and you trust the publisher. It executes Python from the model
repo in the bench client.

**Server-side prerequisites for fair benchmarks** (add to
`vllm serve` for the bench window only):

- `--no-enable-prefix-caching` — prefix cache hits report TTFT as
  scheduler-only (~25 ms) and contaminate the metric.
- `--disable-log-stats` — avoid per-step log IO.

## Online bench (server up)

**Always run the bench client inside the container via `docker exec`.**
Never search for or invoke `vllm` on the host — the binary lives
inside the container image.

```sh
docker exec <vllm-container-name> bash -c '
    mkdir -p /root/bench-out
    vllm bench serve \
        --backend openai \
        --endpoint /v1/completions \
        --host 127.0.0.1 --port 8000 \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --served-model-name Qwen/Qwen2.5-1.5B-Instruct \
        --dataset-name random \
        --random-input-len 1024 --random-output-len 1024 \
        --num-prompts 10 --max-concurrency 4 \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --ignore-eos \
        --save-result --result-dir /root/bench-out'
```

Headline rows in the output: *Total throughput (tok/s)*, *Mean
TTFT*, *p99 TTFT*, *Mean TPOT*, *p99 TPOT*. The JSON in
`/root/bench-out` keeps every per-request data point — preserve
for regression diffs.

## Offline throughput (no server)

```sh
docker run --rm \
  --entrypoint vllm \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    vllm/vllm-openai-xpu:latest \
    bench throughput \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dtype bfloat16 \
        --enforce-eager \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85 \
        --dataset-name random \
        --input-len 512 --output-len 128 \
        --num-prompts 200
```

## Concurrency sweep (find the knee)

```sh
docker exec <vllm-container-name> bash -c '
for c in 1 2; do
    mkdir -p /root/bench-out/c$c
    vllm bench serve \
        --backend openai --endpoint /v1/completions \
        --host 127.0.0.1 --port 8000 \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 512 --random-output-len 128 \
        --num-prompts 10 \
        --max-concurrency "$c" \
        --ignore-eos \
        --metric-percentiles 50,90,99 \
        --save-result --result-dir "/root/bench-out/c$c"
done'
```

The "knee" is the concurrency where throughput plateaus while
p99 TPOT starts climbing. On Arc Pro B70 with a 1–3B model that's
usually around 8–16. Beyond it you trade latency for nothing.

## Benchmarking quantised serving

Bench numbers are meaningful only when the quant kernel is
actually engaged. Most common silent failure: W4A8 falling
through to W4A16 (vLLM #38064) — int4 weights load, requests
succeed, activations are still FP16.

1. Confirm the kernel:
   ```sh
   docker logs <name> 2>&1 | grep -E "Selected.*Kernel|XPUFP8|gemm"
   docker logs <name> 2>&1 | grep -i "Unknown vLLM environment"
   ```
   If you asked for AWQ/GPTQ but see `int4_gemm_w4a16`, you're
   hitting the fall-through.
2. **Read 2–3 sample completions** from the saved JSON before
   trusting throughput. Some quant kernels return non-language
   output without raising; HTTP 200 doesn't prove correctness.

For apples-to-apples between quant kinds, keep
`--no-enable-prefix-caching` on the server — random prompts can
get artificial cache hits otherwise.

## What "good" looks like on Arc Pro B70 (BF16, single GPU, prompt 512 / gen 128, concurrency 1)

| Model size | TTFT (ms) | TPOT (ms/tok) | Single-stream tok/s |
|---|---|---|---|
| 0.5 B | tens | low single digits | low hundreds |
| 1.5 B | tens | ~10 | ~100 |
| 7–8 B | hundreds | tens | ~30 |
| 14 B BF16 | OOM at default | — | needs FP8 or `-tp` |

Order-of-magnitude only; real numbers move with each vLLM release.
Save a baseline JSON, diff against it on upgrades.

## Common XPU surprises

- First run slow, second fast → XPU Triton kernel compile. Mount
  `TRITON_CACHE_DIR` so the next start is hot.
- TTFT fine, TPOT awful → an op fell back to CPU. Set
  `PYTORCH_ENABLE_XPU_FALLBACK=0` for the bench; the engine errors
  and names the op.
- Throughput collapses past concurrency 8 → KV cache exhausted.
  Lower `--max-model-len` or raise `--gpu-memory-utilization`.
- Bench TPOT 30–60% worse than a single curl → bench is at
  `--max-concurrency`, curl is concurrency 1.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Pin offline-mode bench client to one XPU. |
| `TRITON_CACHE_DIR` | Persist XPU Triton kernels across runs. |
| `PYTORCH_ENABLE_XPU_FALLBACK=0` | Make CPU-fallback errors loud. |
| `HF_TOKEN` | Gated checkpoints. |

## What this skill does NOT cover

- Profiling → **vllm-xpu-profile**, **xpu-profile-unitrace**.
- Fixing unsupported ops — out of scope.
- SGLang benches → **sglang-xpu-bench**.
- Pure PyTorch benches → **torch-xpu-bench**.

## References

- `references/sweep-and-compare.md` — concurrency sweep, quant benches, run-vs-run diff
- vLLM bench docs: <https://docs.vllm.ai/en/latest/benchmarking/cli/>
