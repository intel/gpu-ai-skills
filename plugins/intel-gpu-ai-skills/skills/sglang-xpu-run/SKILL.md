---
name: sglang-xpu-run
description: Serve a Hugging Face safetensors model on an Intel GPU using SGLang's XPU backend with the OpenAI-compatible API. Covers pulling the official `lmsysorg/sglang:v0.5.20-xpu` release image, the container flag set for Intel DRM devices, the UMD/kernel pairing the image pins, the `--device xpu --attention-backend intel_xpu` flag set, page-size and quantization constraints, multimodal serving, and how to validate output content (not just HTTP 200). Use when the user needs SGLang's RadixAttention prefix caching or grammar-constrained output; for broad-coverage serving on Intel today prefer vllm-xpu-run, and for benchmarking a running server use sglang-xpu-bench.
---

# sglang-xpu-run

SGLang's XPU backend is functional and ships as an official release image:
`lmsysorg/sglang:v0.5.20-xpu`. Verify GPU detection before serving; SGLang
silently falls back to CPU and you'll only notice when throughput is 50×
lower than expected.

If you don't specifically need RadixAttention prefix caching or
grammar-constrained output, prefer **vllm-xpu-run** — its Intel coverage
is broader today.

## Step 0 — discover GPUs and host RAM before anything else

Run **xpu-discover** first, or at minimum:

```sh
# GPU inventory
xpu-smi discovery

# Count cards and host RAM — used to size ZE_AFFINITY_MASK and --tp
GPU_COUNT=$(xpu-smi discovery 2>/dev/null | grep -cE "^\| +[0-9]|Device [0-9]+:")
[ "${GPU_COUNT:-0}" -gt 0 ] || GPU_COUNT=1
RAM_GB=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}
echo "GPUs: $GPU_COUNT  RAM: ${RAM_GB} GB  render GID: $RENDER_GID"
```

Use `GPU_COUNT` to set `ZE_AFFINITY_MASK` and `--tp`; use `RENDER_GID` in
every `docker run` command below.

## CUDA → XPU cheat sheet

| CUDA | Intel |
|---|---|
| `lmsysorg/sglang:latest` | `lmsysorg/sglang:v0.5.20-xpu` |
| `--gpus all` | `--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --group-add <render-gid>` |
| `--device cuda` (implicit) | `--device xpu` + `--attention-backend intel_xpu` |
| `--tp 2` | `--tp 2` + `ZE_AFFINITY_MASK=0,1` |
| `--quantization awq` | **validate content before trusting it on XPU** |
| `--quantization fp8` | works (runtime BF16→FP8 weight conversion) |
| conda env + `setvars.sh` | nothing to activate — see below |

## Pull the image

```sh
# Skip the pull when the image is already local — it is ~36 GB unpacked, and
# a slow or proxied registry otherwise eats the whole session in diagnostics.
IMG=lmsysorg/sglang:v0.5.20-xpu
docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG"
```

Use `lmsysorg/sglang:v0.5.20-xpu` in all `docker run` commands below.

Which tag to use:

- `lmsysorg/sglang:v<version>-xpu` — **prefer this.** Official per-release
  XPU image, built by the sgl-project release workflow from
  `docker/xpu.Dockerfile` at the matching tag. Pinned, reproducible.
  Substitute a newer `v<version>-xpu` tag when one ships.

What the image already contains, so you don't wrap the launch command:

- Python lives in a venv at `/opt/venv` that is **already on `PATH`** —
  run `python3 -m sglang.launch_server` directly. There is no conda env
  to activate.
- oneAPI env vars are baked into the image config
  (`SETVARS_COMPLETED=1`) — do **not** source `setvars.sh`.
- The container runs as **root** with `CMD ["bash"]` and no entrypoint,
  so `--entrypoint` is unnecessary.
- Level Zero UMD pinned + `apt-mark hold`-ed: compute-runtime
  `26.18.38308.1`, IGC `2.34.4`, gmmlib `22.10.0`; `torch 2.13.0+xpu`.

## Pre-flight: verify XPU is visible inside the container

Optional when a launch follows immediately — it costs a container start and
the verify block reports `DIED` with the same Level Zero error. Run it when
*diagnosing* (no XPU found, `zeInit` failure, a new host), not as a gate.

```sh
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}

docker run --rm \
    --network host --ipc=host --shm-size=16g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK=0 \
    lmsysorg/sglang:v0.5.20-xpu \
    python3 -c 'import torch; n = torch.xpu.device_count(); \
                print("xpu count:", n); \
                print(torch.xpu.get_device_name(0) if n else "NO DEVICE"); \
                assert n > 0, "NO XPU VISIBLE"'
```

If `xpu count: 0` see Common errors below.

## Why each container flag

- `--device /dev/dri` + `-v /dev/dri/by-path:/dev/dri/by-path` — Intel has
  no `--gpus` flag; you pass the DRM nodes directly, and Level Zero
  discovers devices through the `/by-path` symlinks. Without the bind
  mount some images report no XPU.
- `--group-add "$RENDER_GID"` — `/dev/dri/renderD*` is mode 660 owned by
  the `render` group. This image runs as root, which bypasses the group
  check, so the flag is not strictly required here — keep it anyway: it is
  what makes the same command work if you add `--user`, and upstream CI
  adds both the `video` and `render` GIDs.
- **`--privileged` is not required.** Earlier non-root sglang images
  needed it on Battlemage; the pinned-UMD release image does not (verified
  on Arc Pro B70). Only reach for it if `zeInit` fails with a permission
  error that `--group-add` doesn't fix.
- `--network host` — simplifies port handling (sglang uses 30000 +
  internal RPC ports). Swapping it for `-p 30000:30000` also sidesteps the
  multi-GPU interface trap below (bridge mode was verified unaffected).
- `--shm-size=16g` (or `--ipc=host`) — sglang's scheduler uses shared
  memory more aggressively than vLLM. Default Docker shm is too small.
- `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"` — reuse the
  host Hub cache. The container is root, so `/root/.cache/huggingface`
  is the path sglang reads.

## Quickstart — serve one text-gen model

**Skip this step unless you need it.** The container downloads weights
itself into the mounted cache, and `hf` is often absent on the host — do
not install it or hunt for an alternative CLI, just launch. Pre-download
only if you want the download outside the server's startup window:

```sh
# Only if `command -v hf` succeeds; unset ALL_PROXY if it is a SOCKS URL
# (hf CLI doesn't support SOCKS). Gated models need HF_TOKEN.
unset ALL_PROXY all_proxy
hf download Qwen/Qwen3-0.6B
```

Launch (single GPU — for multi-GPU see below):

```sh
RENDER_GID=$(getent group render 2>/dev/null | cut -d: -f3)
RENDER_GID=${RENDER_GID:-$(stat -c '%g' /dev/dri/renderD128 2>/dev/null)}

# Keep the container name literal (no shell variable): docker logs/stop and
# cleanup tooling identify the container by the name in this command.
# If the name is taken, stop here — never force-remove it (Common errors).
docker ps -a --format '{{.Names}}' | grep -qx sglang-xpu && {
    echo "sglang-xpu exists — reuse it, stop it if it is yours, or pick another literal name"; exit 1; }

# No --rm: a container that dies during init takes its log with it, and that
# log is the whole diagnosis. It is removed explicitly at cleanup instead.
docker run -d --name sglang-xpu \
    --network host --ipc=host --shm-size=16g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_TOKEN \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK=0 \
    lmsysorg/sglang:v0.5.20-xpu \
    python3 -m sglang.launch_server \
        --model Qwen/Qwen3-0.6B \
        --device xpu \
        --tp 1 \
        --attention-backend intel_xpu \
        --disable-overlap-schedule \
        --page-size 64 \
        --host 0.0.0.0 --port 30000
```

## Verify: readiness + warmup + smoke test, in one command

<a id="verify-block"></a>Run this **exactly once** after any launch, editing
only the three variables on the first line. It is one block on purpose:
readiness, warmup, and the content-validated smoke test are a single
round trip, so a slow start can't consume the whole session.

**Run it in the foreground.** It self-terminates (~30 s on a healthy
server, and it is bounded on every path), so backgrounding buys nothing
and moves the completion text out of your transcript — leaving you, and
anyone reading the run afterwards, with no evidence the server actually
answered. The printed `content:` line *is* the proof.

```sh
NAME=sglang-xpu; PORT=30000; MODEL=Qwen/Qwen3-0.6B

# 1. Readiness: poll the HTTP endpoint. Exits on ready, on container death,
#    or at the 120 s deadline; each probe is capped at 5 s so a stalled
#    endpoint can't hold the loop. Never loop on `docker logs`: the log is
#    for the post-mortem, not the readiness signal.
T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt 120 ]; do
    curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null \
        && { echo "READY after $(( $(date +%s) - T0 ))s"; break; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "DIED — run: docker logs --tail 50 $NAME"; break; }
    sleep 2
done
curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
    echo "NOT READY — run: docker logs --tail 50 $NAME"; exit 1; }

# 2. Warmup: the first request absorbs kernel compilation (10–60x slower).
# --max-time: a TP hang (see Multi-GPU) never returns; fail fast instead.
curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8}" \
    >/dev/null || { echo "WARMUP TIMED OUT — server accepted the request but never answered"; exit 1; }

# 3. Smoke test: validate CONTENT, not just HTTP 200.
curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one sentence.\"}],\"max_tokens\":64}" \
  | python3 -c "
import sys, json
m = json.load(sys.stdin)['choices'][0]['message']
content = (m.get('reasoning_content') or '') + (m.get('content') or '')
print('content:', content[:120])
assert len(content) > 10 and not set(content).issubset(set('! ')), \
    'looks like garbage — check quant and XPU placement'
assert '<|channel|>' not in content, 'raw gpt-oss tokens — add --reasoning-parser gpt-oss'
print('SMOKE TEST PASS')
"
```

Measured on Arc Pro B70 with a warm HF cache, Qwen3-0.6B: ready in ~28 s
at `--tp 1` and ~32 s at `--tp 2`, so the 120 s cap is ample for a small
model. Raise it for large weights or a cold cache. `/v1/models` also tells
you the exact `id` to send as `"model"`.

`MODEL` must match the `id` in `/v1/models` exactly — that is the value
you passed to `--model` (a Hub id here, a container path if you mounted
weights yourself).

Cleanup: `docker stop sglang-xpu && docker rm sglang-xpu` — if it died, read
`docker logs sglang-xpu` first, since `rm` discards the log for good.

## Multi-GPU (TP) — always pass `FI_TCP_IFACE=lo`

**For `--tp > 1` on one host, add `-e FI_TCP_IFACE=lo`. Unconditionally.**
Ranks all-reduce via torch-XPU → oneCCL → libfabric, which (no usable `shm`
provider) falls back to TCP and may bind an interface the ranks never reach
— a Docker bridge under `--network host` here — leaving the server *ready*
while the **first request never returns**. Loopback is always correct for
ranks on one machine, costs nothing (measured slightly faster than pinning a
real NIC), and is inert on single GPU and on images whose `shm` provider
works.

The one exception: **with `--nnodes > 1` do not use `lo`** — ranks on other
hosts become unreachable. Pin the NIC the nodes share instead.

If a TP launch still hangs with the variable set, isolate the collective
from the server in ~20 s with `scripts/tp_collective_probe.py` (usage and
verdicts in `references/multi-gpu-tp.md`). If the probe passes, retry at a
lower `--tp` — upstream reports engine-side hangs on some Arc/B-series hosts.

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
docker ps -a --format '{{.Names}}' | grep -qx sglang-xpu-tp && {
    echo "sglang-xpu-tp exists — reuse it, stop it if yours, or pick another literal name"; exit 1; }

docker run -d --name sglang-xpu-tp \
    --network host --ipc=host --shm-size=16g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK="$MASK" \
    -e FI_TCP_IFACE=lo \
    lmsysorg/sglang:v0.5.20-xpu \
    python3 -m sglang.launch_server \
        --model "<model>" \
        --device xpu --tp "$TP" \
        --attention-backend intel_xpu \
        --disable-overlap-schedule --page-size 64 \
        --host 0.0.0.0 --port 30000
```

**TP is not done until you have validated output.** Run the verify block
now, with `NAME=sglang-xpu-tp` — copy it as-is, do not substitute a
`docker logs` wait:

```sh
# MODEL must be quoted until you substitute it — bare <model> is a redirect.
NAME=sglang-xpu-tp; PORT=30000; MODEL="<model>"

T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt 120 ]; do
    curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null \
        && { echo "READY after $(( $(date +%s) - T0 ))s"; break; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "DIED — run: docker logs --tail 50 $NAME"; break; }
    sleep 2
done
curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
    echo "NOT READY — run: docker logs --tail 50 $NAME"; exit 1; }

# --max-time: a TP hang (see Multi-GPU) never returns; fail fast instead.
curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8}" \
    >/dev/null || { echo "WARMUP TIMED OUT — server accepted the request but never answered"; exit 1; }

curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one sentence.\"}],\"max_tokens\":64}" \
  | python3 -c "
import sys, json
m = json.load(sys.stdin)['choices'][0]['message']
content = (m.get('reasoning_content') or '') + (m.get('content') or '')
print('content:', content[:120])
assert len(content) > 10 and not set(content).issubset(set('! ')), \
    'looks like garbage — check tp/mask and XPU placement'
assert '<|channel|>' not in content, 'raw gpt-oss tokens — add --reasoning-parser gpt-oss'
print('SMOKE TEST PASS')
"
```

An unsmoke-tested TP server proves nothing: readiness is what the
all-reduce hang also looks like, and a `--tp`/`ZE_AFFINITY_MASK` mismatch
can serve HTTP 200 with garbage. If the smoke request hits its
`--max-time`, confirm `FI_TCP_IFACE=lo` reached the container
(`docker inspect -f '{{.Config.Env}}' sglang-xpu-tp`), then run the probe. Expect ~6 s for the first request and well under 1 s warm;
both cards should show comparable memory in `xpu-smi`.

Cleanup: `docker stop sglang-xpu-tp && docker rm sglang-xpu-tp`.

## SGLang flag rationales

- `--device xpu` — explicit. SGLang doesn't auto-detect XPU as cleanly
  as vLLM does.
- `--attention-backend intel_xpu` — verified-good SYCL kernel path, and
  **not the default**: with no flag, XPU resolves to `triton`, which is
  slower and has patchier coverage. The engine falls back to `triton`
  anyway on a device without XMX (pre-Xe2 hardware).
- `--disable-overlap-schedule` — conservative, as in upstream's XPU tests;
  not a known-bug fix (no stalls seen with overlap on: v0.5.20, 96
  concurrent requests), so dropping it is a fair A/B.
- `--page-size 64` — with `intel_xpu`, only `64` and `128` are accepted
  (plus `16`/`32` on the MLA path); anything else is silently rewritten
  to `128` with a warning.
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
| `mxfp4` MoE | native MXFP4 checkpoints (e.g. `openai/gpt-oss-20b`) run via `sgl-kernel-xpu` W4A16; auto-registered on `--device xpu`, no extra flag. Xe2 / BMG only. For gpt-oss add `--reasoning-parser gpt-oss`, or its reasoning channel lands in `content` as raw `<\|channel\|>…` tokens |
| `awq` | has produced HTTP 200 with non-language content on XPU — validate before trusting |
| `gptq`, `marlin`, `awq_marlin`, `bitsandbytes`, `mxfp8`, `compressed-tensors`, `modelopt_*` | unverified — validate content |
| AutoRound | auto-detected via `quantization_config.quant_method=auto-round` |

Never accept HTTP 200 alone — confirm `content` (and `reasoning_content`) is language.

## Also in this image (opt-in, v0.5.20)

- **XPU graph capture** — off by default: `--cuda-graph-backend-decode full`,
  `--cuda-graph-backend-prefill tc_piecewise|breakable`. Less launch
  overhead, slower startup.
- **Memory saver** — `--enable-memory-saver`, then
  `POST /release_memory_occupation` / `/resume_memory_occupation` to hand the
  device back without restarting (`torch_memory_saver` is prebuilt). **A full
  release drops the weights** (resume serves `!!!!…`) — add `--enable-weights-cpu-backup`,
  release only `{"tags":["kv_cache"]}`, or `POST /update_weights_from_disk` after resume.
- **P/D disaggregation** — NIXL backend; needs `pip install nixl
  sglang-router` + `UCX_POSIX_USE_PROC_LINK=n`.
- **MLA / DeepSeek** — kernels and code path ship in the image but no
  SGLang Cookbook XPU model uses MLA and this pack has not served one;
  see `references/spec-decode-and-multimodal.md` before promising it.

Details and current limitations: <https://docs.sglang.io/docs/hardware-platforms/xpu>.

## Speculative decoding, multimodal

See `references/spec-decode-and-multimodal.md` for EAGLE / EAGLE3 flags (NEXTN /
MTP crashes on this image) and verified multimodal models.

## Not on this stack today

Route to **vllm-xpu-run** if you need any of:

- LoRA hot-swap.
- Two-batch overlap (`--enable-two-batch-overlap`).
- Expert parallelism on the MXFP4 MoE path.

## Common errors

- `xpu count: 0` — **UMD/kernel mismatch.** The image pins
  compute-runtime `26.18.38308.1`; if the host `xe` KMD is older than
  that UMD supports, `zeInit` fails. Compare image vs host:

  ```sh
  # Note ^(ii|hi): the image apt-mark-holds these, so dpkg shows "hi", not "ii"
  docker run --rm lmsysorg/sglang:v0.5.20-xpu \
      dpkg -l libze-intel-gpu1 | grep -E "^(ii|hi)"
  dpkg -l libze-intel-gpu1 | grep -E "^(ii|hi)"
  ```

  Fix by upgrading the host stack (**xpu-system-setup**), or serve with
  **vllm-xpu-run** meanwhile.
- **Benign startup lines that look like failures.** Healthy startup prints
  `Ignore import error when loading sglang.srt.models.<arch>: No module
  named 'vllm'` (and similar for `cutlass`) — optional model-family plugins
  being skipped, ~4 lines per start. They are not errors. This is why
  readiness comes from polling `/v1/models`, never from grepping the log:
  an `error`-matching grep fires on a healthy server, and a grep for the
  ready banner costs a `docker logs` read per iteration and gives you no
  signal when the container dies. Read the log only after the verify block
  reports `DIED` or `NOT READY`.
- `Conflict. The container name "/sglang-xpu" is already in use` →
  **never force-remove it** (`docker rm` with `-f`): on a shared host that
  name may belong to someone else's run, and force-removal destroys their
  work. Inspect it first (`docker ps -a --filter name=sglang-xpu`): reuse it
  if it already serves your model, `docker stop` it if it is yours, else
  relaunch with a different **literal** name (`--name sglang-xpu-2`, not a
  shell variable). Same rule for clearing a name preemptively — don't.
  Your own `Exited` container from a failed run is the common case: read
  `docker logs sglang-xpu`, then free the name with `docker rm` (no `-f`).
- `EACCES` on `/dev/dri/renderD*` → you added `--user`; add
  `--group-add "$(getent group render | cut -d: -f3)"`.
- `ZE_RESULT_ERROR_UNINITIALIZED` from `zeInit` → same UMD mismatch as
  above, or `/dev/dri/by-path` not mounted.
- Segfault inside `libsycl` when a cached kernel is reloaded → set
  `SYCL_CACHE_PERSISTENT=0` (what upstream CI does on this runtime).
- `triton.compiler.errors.CompilationError` on first request → cold
  Triton cache. Mount `TRITON_CACHE_DIR` to a host volume.
- `RuntimeError: NCCL` → sglang prints "NCCL" on the XPU path; actual
  backend is oneCCL. Usually a `--tp` / `ZE_AFFINITY_MASK` mismatch.
- HTTP 200 with garbage (`!!!!!!!!...`) → unsupported quant path (use `fp8`
  or BF16), or a memory-saver resume that dropped the weights (see above).
- Server hangs at "Compiling kernel" → cold kernel cache on first run.
  Watch `xpu-smi dump -d 0 -m 18 -i 1` — high power = compiling, not
  stuck. Persist the cache for next run.
- `429 Too Many Requests` pulling image → Docker Hub rate-limiting
  anonymous pulls. Fix: `docker login` for higher limits, or retry
  after 60 seconds.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the server sees (`0`, `0,1`, …). |
| `FI_TCP_IFACE=lo` | **Set for every `--tp > 1` launch on one host.** Pins libfabric (oneCCL's OFI transport) to loopback; without it the TP all-reduce can bind an unreachable interface and the first request hangs forever. Wrong only for `--nnodes > 1` — pin the shared NIC there. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` | Keep SYCL's device view consistent with the `ZE_AFFINITY_MASK`-filtered set. |
| `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `NEO_CACHE_DIR` + `NEO_CACHE_PERSISTENT=1` | Persist compiled kernels across runs (mount to a host volume). |
| `SYCL_CACHE_PERSISTENT=0` | Disable the SYCL persistent kernel cache; upstream CI keeps it off to avoid a `libsycl` crash on reload. |
| `SGLANG_SERVER_LAUNCH_TIMEOUT` | Raise for large MoE loads from a cold Hub cache (upstream CI uses 36000). |
| `SYCL_UR_USE_LEVEL_ZERO_V2=0` | Force the L0 v1 adapter. Not needed with this image (device discovery verified without it); keep as a fallback if `zeInit`/enumeration misbehaves. |
| `HF_TOKEN` | HF auth when downloading inside the container. |
| `SGLANG_LOGGING_LEVEL=DEBUG` | Verbose engine logs. |
| `ALL_PROXY` / `all_proxy` | Unset if set to `socks://` — `hf` CLI doesn't support SOCKS. |

## What this skill does NOT cover

- Pure PyTorch / Transformers → **torch-xpu-run**.
- vLLM serving → **vllm-xpu-run**.
- Benchmarking → **sglang-xpu-bench**.

## References

- `references/spec-decode-and-multimodal.md` — spec-decode + multimodal
- `references/multi-gpu-tp.md` — the `--tp > 1` all-reduce hang: fix,
  root cause, py-spy recipe, and what does not fix it
- SGLang XPU docs: <https://docs.sglang.io/docs/hardware-platforms/xpu>
- Official image: `docker pull lmsysorg/sglang:v0.5.20-xpu`
  (<https://hub.docker.com/r/lmsysorg/sglang/tags?name=xpu>)
- Image recipe: <https://github.com/sgl-project/sglang/blob/main/docker/xpu.Dockerfile>
- Intel AutoRound: <https://github.com/intel/auto-round>
