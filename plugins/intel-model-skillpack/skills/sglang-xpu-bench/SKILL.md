---
name: sglang-xpu-bench
description: Benchmark a **running SGLang-XPU server** on an Intel GPU using `sglang.bench_serving`. Measures TTFT, TPOT, ITL, end-to-end latency, and throughput against the OpenAI-compatible endpoint. Use after sglang-xpu-run. Not for vLLM servers (use vllm-xpu-bench) or no-server PyTorch (use torch-xpu-bench).
---

# sglang-xpu-bench

`sglang.bench_serving` is SGLang's online benchmark client (the
counterpart to `vllm bench serve`). Speaks the OpenAI-compatible API
exposed by **sglang-xpu-run**.

## Step 0 — verify SGLang server is running on Intel XPU

**REQUIRED**: Before benchmarking, you must confirm a SGLang server is running,
identify its port, and verify it's using Intel XPU (not CPU fallback, not NVIDIA).
This prevents benchmarking a server that silently fell back to CPU.

Run the checks below **in a single shell session** (later blocks reuse
`$SGLANG_PID`, `$SGLANG_PORT`, `$MODEL` from earlier ones).

Find the server process and its container:

```sh
# 1. Check if SGLang is running and find its container
SGLANG_PID=$(ps aux | grep -iE 'sglang|launch_server' | grep -v grep | awk 'NR==1 {print $2}')
if [ -z "$SGLANG_PID" ]; then
    echo "❌ No SGLang server found running. Start one with sglang-xpu-run."
    exit 1
fi
echo "✓ SGLang server found (PID: $SGLANG_PID)"

CONTAINER_NAME=$(docker ps --format '{{.Names}}' 2>/dev/null | while read name; do
    if docker top "$name" -o pid 2>/dev/null | awk 'NR>1' | grep -qxF "$SGLANG_PID"; then echo "$name"; break; fi
done)
if [ -z "$CONTAINER_NAME" ]; then
    echo "❌ No container matched PID $SGLANG_PID. This skill runs the bench client"
    echo "   via 'docker exec' because the sglang package is only installed inside"
    echo "   the server container. A host-only SGLang install is not supported here."
    exit 1
fi
echo "✓ SGLang container: $CONTAINER_NAME"
```

Read the launch args **once** — they give you both the container-internal port
and the `--device` flag. The bench client runs via `docker exec` inside the
container, so the port must be the one SGLang binds *inside* the container
(host `ss` can't see the container-namespaced socket). The `--device xpu` check
is REQUIRED — it catches a server that silently fell back to CPU:

```sh
# 2. Read port + device from the launch args in one pass (REQUIRED)
ARGS=$(ps -p "$SGLANG_PID" -o args=)
SGLANG_PORT=$(echo "$ARGS" | awk '{for(i=1;i<=NF;i++) if($i=="--port" && i<NF) print $(i+1)}' | head -1)
SGLANG_PORT=${SGLANG_PORT:-30000}   # sglang default
DEVICE_ARG=$(echo "$ARGS" | awk '{for(i=1;i<=NF;i++) if($i=="--device" && i<NF) print $(i+1)}' | head -1)
echo "DEVICE_ARG=${DEVICE_ARG:-none}"
if [ "$DEVICE_ARG" = "cuda" ]; then
    echo "❌ Server is on NVIDIA CUDA, not Intel XPU."; exit 1
elif [ "$DEVICE_ARG" = "xpu" ]; then
    echo "✓ SGLang server has --device xpu"
else
    echo "⚠️  Could not confirm --device xpu (got: '${DEVICE_ARG:-none}'); relying on XPU memory check."
fi

# 3. Verify reachable from inside the container and read model name
if ! docker exec "$CONTAINER_NAME" curl -s http://127.0.0.1:$SGLANG_PORT/v1/models >/dev/null 2>&1; then
    echo "❌ SGLang server not responding on container port $SGLANG_PORT."
    exit 1
fi
MODEL=$(docker exec "$CONTAINER_NAME" curl -s http://127.0.0.1:$SGLANG_PORT/v1/models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "✓ Responding on container port $SGLANG_PORT — model: $MODEL"
```

Reject NVIDIA GPU usage, then confirm XPU memory is in use (rules out CPU fallback):

```sh
# 4a. Reject NVIDIA GPU usage — scan only the compute-app PID list and match
# the whole line, so a short PID can't collide with memory/temp/other numbers
# elsewhere in nvidia-smi's output.
if command -v nvidia-smi >/dev/null 2>&1 && \
   nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
     | tr -d ' ' | grep -qxF "$SGLANG_PID"; then
    echo "❌ Server is running on NVIDIA GPU, not Intel XPU."; exit 1
fi

# 4b. Verify Intel XPU memory usage — -m 18 = "GPU Memory Used (MiB)", CSV output
GPU_COUNT=$(xpu-smi discovery 2>/dev/null | grep -cE "^\| +[0-9]|Device [0-9]+:")
[ "${GPU_COUNT:-0}" -gt 0 ] || GPU_COUNT=1
XPU_IN_USE=0
for id in $(seq 0 $((GPU_COUNT - 1))); do
    MEM_USED=$(xpu-smi dump -d "$id" -m 18 -i 1 -n 1 2>/dev/null | awk -F',' 'NR==2 {gsub(/ /,"",$NF); print int($NF)}')
    echo "  Device $id: ${MEM_USED:-0} MiB"
    [ "${MEM_USED:-0}" -gt 500 ] && XPU_IN_USE=1
done
if [ "$XPU_IN_USE" -eq 0 ]; then
    echo "❌ XPU memory near zero — server may have fallen back to CPU. Restart with --device xpu."
    exit 1
fi
echo "✓ SGLang is confirmed running on Intel XPU"
```

Everything needed for the benchmark is now confirmed:

```sh
# 5. Summary — CONTAINER_NAME, SGLANG_PORT, MODEL were set above
echo "=== Ready to benchmark === Port: $SGLANG_PORT  Model: $MODEL  Container: $CONTAINER_NAME"
```

If no SGLang server is found, if it's running on NVIDIA instead of Intel XPU,
or if it's fallen back to CPU, the checks exit with guidance.

## Metrics

- **TTFT** — wall time to first generated token.
- **TPOT** — mean per-token time after the first.
- **ITL** — per-token inter-arrival; percentiles meaningful for latency SLAs.
- **E2EL** — wall time of one request.
- **Throughput** — `output_tokens / wall_seconds`.

## Prerequisites

A running SGLang-XPU server (per **sglang-xpu-run**). Step 0 above will
verify the server is running, identify its port, and confirm it's using
Intel XPU.

Note on `ALL_PROXY`: if the host has `ALL_PROXY=socks://...` set, the
bench client's HTTP requests may be routed through the SOCKS proxy.
Unset it before benching a local server:

```sh
unset ALL_PROXY all_proxy
```

## Online bench

**Important:** The benchmark client must run **inside the same container**
where the SGLang server is running. The `sglang` package is only installed
in the container environment, not on the host.

**Always use `docker exec` (not `nsenter`) to run benchmarks.** The `nsenter`
approach is fragile and can hang when other processes are stalled.

**Verify from inside the container.** Output files are written inside the
container filesystem. To verify or read results, use
`docker exec "$CONTAINER_NAME" cat "$OUT"`, not host `cat`.

**Always write to a unique output file per run.** `--output-file` appends,
so reusing a name mixes runs — and a stale file from a prior run can trick you
into reading old results instead of running the benchmark. Derive a `RUN_TAG`
from the date and shell PID.

Use the `$SGLANG_PORT`, `$MODEL`, and `$CONTAINER_NAME` discovered in Step 0:

```sh
# Discover conda activate path (may differ on forked images)
CONDA_SH=$(docker exec "$CONTAINER_NAME" sh -c 'find /home /root /opt -maxdepth 5 -name activate -path "*/miniforge*/bin/activate" 2>/dev/null | head -1')

# Unique per-run output file so a stale file can't be mistaken for fresh output:
RUN_TAG=$(date +%Y%m%d-%H%M%S)-$$
OUT="/tmp/bench-${RUN_TAG}.jsonl"

# Run benchmark inside the server container (where sglang is installed):
docker exec -it "$CONTAINER_NAME" bash -c "
. $CONDA_SH && conda activate py3.12 &&
python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --host 127.0.0.1 --port $SGLANG_PORT \
    --model $MODEL \
    --dataset-name random \
    --random-input-len 512 --random-output-len 128 \
    --num-prompts 200 \
    --max-concurrency 8 \
    --output-file $OUT
"

# Read results from inside the container:
docker exec "$CONTAINER_NAME" cat "$OUT"
```

Flag rationales:

- `--backend sglang-oai-chat` → `/v1/chat/completions`. Use `sglang-oai`
  for `/v1/completions`; `sglang` for the native API. Match your server.
- `--dataset-name random` — synthetic, deterministic at the same `--seed`,
  no network. Use `sharegpt` for realistic prompt distribution.
- `--random-input-len` / `--random-output-len` — fix lengths for
  reproducible sweeps.
- `--num-prompts` — aim for `>= 5 × max-concurrency` for steady state.
- `--max-concurrency` — sweep to find the throughput knee.
- `--request-rate inf` (default) — issue all immediately. Pass a finite
  qps for Poisson arrivals (e.g. `--request-rate 4`).
- `--output-file` — append-only JSONL; always use a unique name per run
  (e.g. the `RUN_TAG` above) so stale files aren't mistaken for fresh results.

## Concurrency sweep

Auto-sizes from GPU count to find the throughput knee. Uses the
`$SGLANG_PORT`, `$MODEL`, and `$CONTAINER_NAME` from Step 0:

```sh
GPU_COUNT=$(xpu-smi discovery 2>/dev/null | grep -cE "^\| +[0-9]|Device [0-9]+:")
[ "${GPU_COUNT:-0}" -gt 0 ] || GPU_COUNT=1
CONDA_SH=$(docker exec "$CONTAINER_NAME" sh -c 'find /home /root /opt -maxdepth 5 -name activate -path "*/miniforge*/bin/activate" 2>/dev/null | head -1')
RUN_TAG=$(date +%Y%m%d-%H%M%S)-$$   # unique per sweep; files are /tmp/bench-${RUN_TAG}-c<N>.jsonl

docker exec -it "$CONTAINER_NAME" bash -c "
. $CONDA_SH && conda activate py3.12
for c in $(printf '%s\n' 1 2 4 8 $((GPU_COUNT * 8)) $((GPU_COUNT * 16)) | sort -nu | tr '\n' ' '); do
    echo \"--- concurrency \$c ---\"
    python3 -m sglang.bench_serving \
        --backend sglang-oai-chat \
        --host 127.0.0.1 --port $SGLANG_PORT \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 512 --random-output-len 128 \
        --num-prompts \$((c * 25)) \
        --max-concurrency \$c \
        --output-file /tmp/bench-${RUN_TAG}-c\${c}.jsonl
done
"
```

Knee = throughput plateaus while p99 TPOT climbs. Beyond it, latency
degrades without throughput gain.

## Comparing two runs

```sh
python3 - baseline.jsonl candidate.jsonl <<'PY'
import json, sys

def last(path):
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1])

a, b = last(sys.argv[1]), last(sys.argv[2])
for k in ("mean_ttft_ms","p99_ttft_ms","mean_tpot_ms","p99_tpot_ms",
          "request_throughput","output_throughput"):
    print(f"{k:28s}  base={a.get(k,0):.2f}  cand={b.get(k,0):.2f}"
          f"  delta={b.get(k,0)-a.get(k,0):+.2f}")
PY
```

>5% regression on `output_throughput` or >10% on `p99_tpot_ms` is
real. Smaller is noise.

## RadixAttention prefix-cache benchmark

The main sglang-vs-vllm differentiator on Intel. With caching on
(sglang's default), repeated prefixes hit the RadixAttention cache
and TTFT drops sharply on subsequent requests. Runs inside the server
container (where `sglang` is installed), using `$SGLANG_PORT`, `$MODEL`,
and `$CONTAINER_NAME` from Step 0:

```sh
CONDA_SH=$(docker exec "$CONTAINER_NAME" sh -c 'find /home /root /opt -maxdepth 5 -name activate -path "*/miniforge*/bin/activate" 2>/dev/null | head -1')
# Unique per-run tag so a rerun's files don't append onto a stale one:
RUN_TAG=$(date +%Y%m%d-%H%M%S)-$$

# Cache-on run (sglang default), inside the container:
docker exec -it "$CONTAINER_NAME" bash -c "
. $CONDA_SH && conda activate py3.12 &&
python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --host 127.0.0.1 --port $SGLANG_PORT \
    --model $MODEL \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 64 \
    --num-prompts 500 --max-concurrency 8 \
    --random-range-ratio 0.1 \
    --output-file /tmp/cache-on-${RUN_TAG}.jsonl
"

# Read results from inside the container:
docker exec "$CONTAINER_NAME" cat /tmp/cache-on-${RUN_TAG}.jsonl

# Cache-off baseline (restart server with --disable-radix-cache), then re-run
# the docker exec above with --output-file /tmp/cache-off-${RUN_TAG}.jsonl
```

`--random-range-ratio 0.1` → suffixes vary in only the last 10% of
tokens (high prefix overlap, simulates shared-prompt workloads). TTFT
delta between runs is the prefix-cache win.

## Validating quantized serving

HTTP 200 + plausible throughput don't imply correct output. For any
quantized model, before trusting numbers:

1. Capture per-request responses: add `--output-details` to the bench.
2. Read 2–3 sample completions; confirm they parse as language.
3. If non-language, see **sglang-xpu-run**'s Quantization table.

## Common errors

- Connection refused → server not running or on a different port.
  Run Step 0 to verify and discover the port.
- `model not found` HTTP 400 → `--model` must match `/v1/models` exactly.
  Step 0 auto-detects the correct model name into `$MODEL`.
- Suspiciously low TPOT on `sglang-oai-chat` → known sglang issue with
  TPOT computation for the chat backend (#10746). Cross-check with
  `--backend sglang` (native API).
- Throughput mismatch between bench output and server log → bench is
  wall-time-from-client; engine is decode-loop-internal. Both are valid;
  cite which you used.
- First bench slow → cold Triton cache on a freshly-started server. Send
  a warmup run first: `--num-prompts 20 --max-concurrency 1`.
- Bench hangs on connect → `ALL_PROXY` routing local traffic through a
  proxy. Unset `ALL_PROXY` and `all_proxy` before running.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the *server* sees. Bench client is a network process; GPU unaffected. |
| `TRITON_CACHE_DIR` | Persist XPU Triton kernels across server restarts. Set on the server, not the client. |
| `ALL_PROXY` / `all_proxy` | Unset if set to `socks://` — bench client uses plain HTTP. |

## What this skill does NOT cover

- vLLM serving benches → **vllm-xpu-bench**.
- Pure PyTorch benches → **torch-xpu-bench**.
- Profiling → out of scope.
- `sglang.bench_offline_throughput` (single-process, no server) —
  identical methodology to **torch-xpu-bench**; pick one for
  cross-run comparability.

## References

- `sglang.bench_serving` source: <https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_serving.py>
- SGLang benchmarking guide: <https://docs.sglang.io/developer_guide/benchmark_and_profiling.html>
- sglang #10746 (TPOT on `sglang-oai-chat`): <https://github.com/sgl-project/sglang/issues/10746>
