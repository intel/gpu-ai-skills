---
name: sglang-xpu-run
description: Serve a Hugging Face safetensors model on an Intel GPU using SGLang's XPU backend with the OpenAI-compatible API. Covers pulling the pre-built `intel/sglang-dev:latest` image, fixing the render-group and UMD/kernel compatibility issues that affect non-root sglang images, the SYCL_UR / Level Zero env vars needed on Battlemage, the `--device xpu --attention-backend intel_xpu` flag set, multimodal serving, and how to validate output content (not just HTTP 200). Use when the user needs SGLang's RadixAttention prefix caching or grammar-constrained output; for broad-coverage serving on Intel today prefer vllm-xpu-run, and for benchmarking a running server use sglang-xpu-bench.
---

# sglang-xpu-run

SGLang's XPU backend is functional. Use the pre-built
`intel/sglang-dev:latest` image. Verify GPU detection before
serving; SGLang silently falls back to CPU and you'll only notice when
throughput is 50× lower than expected.

If you don't specifically need RadixAttention prefix caching or
grammar-constrained output, prefer **vllm-xpu-run** — its Intel coverage
is broader today.

## Step 0 — discover GPUs and host RAM before anything else

Run **xpu-discover** first, or at minimum:

```sh
# GPU inventory
xpu-smi discovery

# Count cards and host RAM — used to size the build and ZE_AFFINITY_MASK
GPU_COUNT=$(xpu-smi discovery 2>/dev/null | grep -cE "^\| +[0-9]|Device [0-9]+:")
[ "${GPU_COUNT:-0}" -gt 0 ] || GPU_COUNT=1
RAM_GB=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}
echo "GPUs: $GPU_COUNT  RAM: ${RAM_GB} GB  render GID: $RENDER_GID"
```

Use `GPU_COUNT` to set `ZE_AFFINITY_MASK` and `--tp`; use `RAM_GB` to
set `MAX_JOBS` for the build; use `RENDER_GID` in every `docker run`
command below.

## CUDA → XPU cheat sheet

| CUDA | Intel |
|---|---|
| `lmsysorg/sglang:latest` | `intel/sglang-dev:latest` |
| `--gpus all` | `--privileged --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --group-add <render-gid>` |
| `--device cuda` (implicit) | `--device xpu` + `--attention-backend intel_xpu` |
| `--tp 2` | `--tp 2` + `ZE_AFFINITY_MASK=0,1` |
| `--quantization awq` | **silently broken on XPU** (HTTP 200, garbage content) |
| `--quantization fp8` | works (runtime BF16→FP8 weight conversion) |
| no env vars | needs `SYCL_UR_USE_LEVEL_ZERO_V2=0` on Battlemage |

## Pull the image

```sh
docker pull intel/sglang-dev:latest
```

Use `intel/sglang-dev:latest` in all `docker run` commands below.

## Pre-flight: verify XPU is visible inside the container

```sh
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}

docker run --rm \
    --privileged --network host --ipc=host --shm-size=32g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    --entrypoint /bin/bash \
    intel/sglang-dev:latest \
    -lc 'CONDA_SH=$(find /home /root /opt -maxdepth 5 -name activate -path "*/miniforge*/bin/activate" 2>/dev/null | head -1); \
         . "${CONDA_SH:-$HOME/miniforge3/bin/activate}" && \
         conda activate py3.12 && \
         source /opt/intel/oneapi/setvars.sh --force >/dev/null && \
         python -c "import torch; n=torch.xpu.device_count(); \
                    print(\"xpu count:\", n); \
                    assert n>0, \"NO XPU VISIBLE\""'
```

If `xpu count: 0` see Common errors below.

## Why each container flag

- `--privileged` — required on Battlemage with current images. Reducing
  to plain `--device /dev/dri --group-add render` is on the upstream
  roadmap but not yet stable.
- `--group-add "$RENDER_GID"` — the sglang image runs as non-root user
  `sdp`; `/dev/dri/renderD*` nodes are mode 660 owned by the `render`
  group. Without this, `zeInit` fails with `EACCES` even with
  `--privileged`. vLLM images run as root and skip this issue — that's
  why the flag isn't in most vLLM examples.
- `--network host` — simplifies port handling (sglang uses 30000 +
  internal RPC ports). Drop and add `-p 30000:30000` if you don't want
  host networking.
- `-v /dev/dri/by-path:/dev/dri/by-path` — Level Zero discovers devices
  via `/by-path` symlinks; without it some images report no XPU.
- `--shm-size=32g` (or `--ipc=host`) — sglang scheduler uses shared
  memory more aggressively than vLLM. Default Docker shm is too small.
- `SYCL_UR_USE_LEVEL_ZERO_V2=0` — SYCL Unified Runtime's v2 L0 adapter
  has a device-discovery bug on Battlemage; v1 is the verified
  workaround.
- `source /opt/intel/oneapi/setvars.sh --force` — sets oneCCL, oneMKL,
  Level Zero env paths inside the conda env.

## Quickstart — serve one text-gen model

```sh
# unset ALL_PROXY if set to a SOCKS URL — hf CLI doesn't support SOCKS
unset ALL_PROXY all_proxy
hf download Qwen/Qwen3-0.6B --local-dir "$HOME/models/Qwen3-0.6B"
```

Launch (single GPU — for multi-GPU see below):

```sh
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}

docker run -d --rm --name sglang-xpu \
    --privileged --network host --ipc=host --shm-size=32g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$HOME/models:/models" \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK=0 \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    --entrypoint /bin/bash \
    intel/sglang-dev:latest \
    -lc 'CONDA_SH=$(find /home /root /opt -maxdepth 5 -name activate -path "*/miniforge*/bin/activate" 2>/dev/null | head -1); \
         . "${CONDA_SH:-$HOME/miniforge3/bin/activate}" && \
         conda activate py3.12 && \
         source /opt/intel/oneapi/setvars.sh --force >/dev/null && \
         exec python -m sglang.launch_server \
             --model /models/Qwen3-0.6B \
             --device xpu \
             --tp 1 \
             --attention-backend intel_xpu \
             --disable-overlap-schedule \
             --page-size 64 \
             --host 0.0.0.0 --port 30000'
```

Wait for `Application startup complete` (`docker logs -f sglang-xpu`),
then confirm the API is reachable and the model is loaded:

```sh
# Programmatic readiness check — wait until /v1/models responds
for i in $(seq 1 60); do
    curl -sf http://127.0.0.1:30000/v1/models >/dev/null && break
    sleep 2
done
curl -s http://127.0.0.1:30000/v1/models | python3 -m json.tool
```

Send a warmup request before trusting latency numbers — the first
request compiles Triton kernels and is 10–60× slower:

```sh
curl -s http://127.0.0.1:30000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/models/Qwen3-0.6B",
         "messages":[{"role":"user","content":"warmup"}],
         "max_tokens":8}' > /dev/null
```

Now smoke-test — **validate content, not just HTTP 200**:

```sh
curl -s http://127.0.0.1:30000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/models/Qwen3-0.6B",
         "messages":[{"role":"user","content":"Say hi in one sentence."}],
         "max_tokens":64}' | python3 -c "
import sys, json
r = json.load(sys.stdin)
content = r['choices'][0]['message']['content']
print('content:', content[:120])
assert len(content) > 10 and not set(content).issubset(set('! ')), \
    'looks like garbage — check quant and XPU placement'
print('SMOKE TEST PASS')
"
```

Cleanup: `docker stop sglang-xpu`.

## Multi-GPU (TP)

First discover available XPUs and validate the requested TP degree:

```sh
GPU_COUNT=$(xpu-smi discovery 2>/dev/null | grep -cE "^\| +[0-9]|Device [0-9]+:")
[ "${GPU_COUNT:-0}" -gt 0 ] || GPU_COUNT=1

# Set desired TP — must not exceed available XPUs
TP=${TP:-$GPU_COUNT}
if [ "$TP" -gt "$GPU_COUNT" ]; then
    echo "ERROR: requested TP=$TP but only $GPU_COUNT XPU(s) available" >&2
    exit 1
fi

MASK=$(python3 -c "print(','.join(str(i) for i in range($TP)))")
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}
echo "Launching TP=$TP on XPUs: $MASK (of $GPU_COUNT available)"
```

Launch:

```sh
docker run -d --rm --name sglang-xpu-tp \
    --privileged --network host --ipc=host --shm-size=32g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$HOME/models:/models" \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK="$MASK" \
    -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    --entrypoint /bin/bash \
    intel/sglang-dev:latest \
    -lc "CONDA_SH=\$(find /home /root /opt -maxdepth 5 -name activate -path '*/miniforge*/bin/activate' 2>/dev/null | head -1); \
         . \"\${CONDA_SH:-\$HOME/miniforge3/bin/activate}\" && \
         conda activate py3.12 && \
         source /opt/intel/oneapi/setvars.sh --force >/dev/null && \
         exec python -m sglang.launch_server \
             --model /models/<model> \
             --device xpu --tp $TP \
             --attention-backend intel_xpu \
             --disable-overlap-schedule --page-size 64 \
             --host 0.0.0.0 --port 30000"
```

## SGLang flag rationales

- `--device xpu` — explicit. SGLang doesn't auto-detect XPU as cleanly
  as vLLM does.
- `--attention-backend intel_xpu` — verified-good SYCL kernel path.
  Without it you may land on `triton`, which is slower and has patchier
  coverage.
- `--disable-overlap-schedule` — sglang's overlapped CPU/GPU scheduler
  has known stalls on XPU; keep disabled until upstream fixes land.
- `--page-size 64` — KV-cache page size that holds up best on
  Battlemage; default may be smaller.
- `--tp N` — `ZE_AFFINITY_MASK` must expose exactly N XPUs.
- `--trust-remote-code` — **security opt-in, not in the launch lines above.**
  Permits arbitrary Python from the model repo to run in the engine. Add it
  only when the model declares repo-local code *and* you trust the publisher.
  Check before assuming you need it, per model: the flag is required only if
  that model's own `config.json` contains an `auto_map` entry. Most mainstream
  models do not. Decide from the config, not from a remembered example.
- `--disable-radix-cache` — disables RadixAttention prefix caching.
  Useful for A/B comparisons against vLLM or to isolate decode
  throughput without prefix-cache effects. Omit to keep caching on
  (the default and the main reason to use SGLang over vLLM).

## Quantization: what works, what silently fails

| Quant | Status |
|---|---|
| `fp8` | **works** — runtime BF16→FP8, ~half BF16 size, KV stays BF16 unless `--kv-cache-dtype fp8_e4m3` |
| `awq` | **silently broken on XPU** — HTTP 200, content is non-language |
| `gptq`, `marlin`, `awq_marlin`, `bitsandbytes`, `mxfp8`, `mxfp4`, `compressed-tensors`, `modelopt_*` | unverified — validate content |
| AutoRound | supported since Oct 2025; auto-detected via `quantization_config.quant_method=auto-round` |

Never accept HTTP 200 alone — read `choices[0].message.content` and
confirm it parses as language.

## Speculative decoding, multimodal

See `references/spec-decode-and-multimodal.md` for EAGLE / MTP
spec-decode flags and verified multimodal models.

## Not on this stack today

Route to **vllm-xpu-run** if you need any of:

- DeepSeek MLA path.
- MoE fused kernels.
- LoRA hot-swap.
- `torch.compile` for the model graph.

## Common errors

- `xpu count: 0` — **two independent causes**:
  1. *UMD/kernel mismatch*: image driver may be newer than the host
     kernel supports. Confirm with:
     `docker run --rm intel/sglang-dev:latest dpkg -l libze-intel-gpu1 | grep "^ii"`
     and compare to host `dpkg -l libze-intel-gpu1`.
  2. *Render group*: sglang image runs as non-root `sdp`; `/dev/dri/renderD*`
     require the render group. Fix: add
     `--group-add "$(getent group render | cut -d: -f3)"` to `docker run`.
     (vLLM images run as root and skip this issue.)
- `ZE_RESULT_ERROR_UNINITIALIZED` from `zeInit` → UMD mismatch (cause 1
  above).
- `EACCES` on `/dev/dri/renderD*` → render group missing (cause 2).
- `triton.compiler.errors.CompilationError` on first request → cold
  Triton cache. Mount `TRITON_CACHE_DIR` to a host volume.
- `RuntimeError: NCCL` → sglang prints "NCCL" on the XPU path; actual
  backend is oneCCL. Usually a `--tp` / `ZE_AFFINITY_MASK` mismatch.
- HTTP 200 with garbage (`!!!!!!!!...`) → AWQ quant. Use `fp8` or BF16.
- Server hangs at "Compiling kernel" → cold Triton cache on first run.
  Watch `xpu-smi dump -d 0 -m 18 -i 1` — high power = compiling, not
  stuck. Persist the cache for next run.
- `429 Too Many Requests` pulling image → Docker Hub rate-limiting
  anonymous pulls. Fix: `docker login` for higher limits, or retry
  after 60 seconds.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the server sees (`0`, `0,1`, …). |
| `SYCL_UR_USE_LEVEL_ZERO_V2=0` | Force L0 v1 adapter; v2 is buggy on Battlemage. **Required.** |
| `TRITON_CACHE_DIR` | Persist compiled kernels across runs. |
| `HF_TOKEN` | HF auth when downloading inside the container. |
| `SGLANG_LOGGING_LEVEL=DEBUG` | Verbose engine logs. |
| `ALL_PROXY` / `all_proxy` | Unset if set to `socks://` — `hf` CLI doesn't support SOCKS. |

## What this skill does NOT cover

- Pure PyTorch / Transformers → **torch-xpu-run**.
- vLLM serving → **vllm-xpu-run**.
- Benchmarking → **sglang-xpu-bench**.

## References

- `references/spec-decode-and-multimodal.md` — spec-decode + multimodal
- SGLang docs: <https://docs.sglang.ai/>
- Intel SGLang-XPU image: `docker pull intel/sglang-dev:latest`
- SYCL UR (Level Zero adapter): <https://github.com/oneapi-src/unified-runtime>
- Intel AutoRound: <https://github.com/intel/auto-round>
