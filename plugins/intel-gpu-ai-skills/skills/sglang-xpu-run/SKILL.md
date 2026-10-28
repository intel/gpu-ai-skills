---
name: sglang-xpu-run
description: Serve a Hugging Face safetensors model on an Intel GPU using SGLang's XPU backend with the OpenAI-compatible API. Pins the validated `lmsysorg/sglang:v<version>-xpu` release image and flags newer unvalidated releases, then covers the container flag set for Intel DRM devices, the UMD/kernel pairing the image pins, the `--device xpu --attention-backend intel_xpu` flag set, page-size and quantization constraints, multimodal serving, and how to validate output content (not just HTTP 200). Use when the user needs SGLang's RadixAttention prefix caching or grammar-constrained output; for broad-coverage serving on Intel today prefer vllm-xpu-run, and for benchmarking a running server use sglang-xpu-bench.
---

# sglang-xpu-run

SGLang's XPU backend is functional and ships as an official per-release image
(Step 1 pins the validated tag). Verify GPU detection before serving; SGLang
silently falls back to CPU and you'll only notice when throughput is 50×
lower than expected.

**Choose SGLang for RadixAttention prefix caching or grammar-constrained
output** — both v0.5.22-verified here (`#cached-token` reuse observed;
`json_schema` returns schema-valid JSON). For anything else prefer
**vllm-xpu-run**, whose Intel model coverage is broader today.

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
| `lmsysorg/sglang:latest` | `lmsysorg/sglang:v<version>-xpu` — no `latest-xpu` exists; use the validated `$PIN` (Step 1) |
| `--gpus all` | `--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --group-add <render-gid>` |
| `--device cuda` (implicit) | `--device xpu` + `--attention-backend intel_xpu` |
| `--tp 2` | `--tp 2` + `ZE_AFFINITY_MASK=0,1` |
| `--quantization awq` / `fp8` | both work with caveats — see the quantization table |
| conda env + `setvars.sh` | nothing to activate — see below |

## Step 1 — resolve the image tag, then pull

**There is no `latest-xpu` tag** — sgl-project ships one `v<version>-xpu` image per release.
**Serve `$PIN`, the tag this pack validated.** The lookup below only reports a newer release; it does not switch to it automatically.

```sh
PIN=v0.5.22-xpu                 # the only tag validated end-to-end by this pack
IMG=lmsysorg/sglang:$PIN        # always serve the validated tag
docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG"   # ~36 GB
echo "serving with $IMG"

# Informational only — report a newer release, never auto-adopt it.
NEW=$( { curl -fsS --max-time 10 \
    "https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags?page_size=100&name=xpu" \
    | grep -oE '"name"[[:space:]]*:[[:space:]]*"v[0-9]+\.[0-9]+\.[0-9]+-xpu"' | cut -d'"' -f4; echo "$PIN"; } \
    | sort -V | tail -1 )
[ "$NEW" = "$PIN" ] || echo "NOTE: $NEW is newer and UNVALIDATED here — see below."
```

`sort -V` is a version sort, the grep drops non-release tags, and an
unreachable registry prints no note. Everything below is **validated at
`v0.5.22-xpu`** (2× Arc Pro B70, 2026-10-16). **If the note fires, do not
silently switch.** To evaluate a newer tag: set `IMG`, check the pinned UMD
against the host first (`xpu count: 0` under Common errors — the likeliest
break), then serve flags and the quant table, then re-run the verify block.
Report the tag you served.

Nothing to wrap around the launch: `/opt/venv` is already on `PATH` (no conda
env), oneAPI vars are baked in (do **not** source `setvars.sh`), the container
is root with no entrypoint, and the Level Zero UMD is pinned and held
(compute-runtime `26.27.39122.11`, `torch 2.14.0+xpu` at this tag; each release
bumps them). Details: `references/image-and-troubleshooting.md`.

## Why each container flag

- `--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path` — Intel has no
  `--gpus`; Level Zero finds devices through the `/by-path` symlinks, so both
  are needed.
- `--group-add "$RENDER_GID"` — `renderD*` is mode 660, group `render`. Root
  bypasses it, but keep the flag so the same command survives `--user`.
  **`--privileged` is not required** (verified on Arc Pro B70).
- `--network host` — simplest port handling; `-p 30000:30000` also works and
  sidesteps the TP interface trap below.
- `--shm-size=16g` / `--ipc=host` — sglang's scheduler needs more shared memory
  than Docker's default.
- `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"` — the container is
  root, so this is the cache path sglang reads.

Full rationale and the optional in-container XPU pre-flight:
`references/image-and-troubleshooting.md`.

## Quickstart — serve one text-gen model

The container downloads weights itself into the mounted cache — do not install
`hf` or hunt for a CLI, just launch. Pre-download (`hf download <model>`, only
if `command -v hf` succeeds; unset a SOCKS `ALL_PROXY` first) only to keep the
download out of the server's startup window.

Launch (single GPU — for multi-GPU see below):

```sh
: "${RENDER_GID:?run Step 0 first}" "${IMG:?resolve the image first — see Step 1}"

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
    "$IMG" \
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

<a id="verify-block"></a>Run this **exactly once** after any launch, **in the
foreground**. It reads `NAME`/`PORT`/`MODEL` from the shell (quickstart
defaults otherwise), is bounded on every path (~30 s on a healthy server), and
does readiness, warmup and the content-validated smoke test in one round trip.
Backgrounding it only moves the proof — the printed `content:` line — out of
your transcript.

```sh
NAME=${NAME:-sglang-xpu}; PORT=${PORT:-30000}; MODEL=${MODEL:-Qwen/Qwen3-0.6B}

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

# 3. Smoke test: validate CONTENT, not just HTTP 200. The prompt demands
#    substance on purpose: never gate on length — "Hello!" is a valid reply.
curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France, then name two other European capitals.\"}],\"max_tokens\":80,\"temperature\":0}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert not (d.get('object') == 'error' or d.get('error')), d
m = d['choices'][0]['message']
content = ((m.get('reasoning_content') or '') + (m.get('content') or '')).strip()
print('content:', content[:120])
w = content.split()
assert content and not set(content).issubset(set('! ')), 'empty or !!!! — bad quant/placement'
'paris' in content.casefold() or print('NOTE: no Paris in reply (thinking/non-English model?) - read content above')
assert not (len(w) >= 2 and len(set(w)) == 1), 'one token repeated — broken quant path'
assert not (len(w) >= 8 and len(set(w)) <= max(2, len(w)//6)), 'degenerate repetition'
assert '<|channel|>' not in content, 'raw gpt-oss tokens — add --reasoning-parser gpt-oss'
print('SMOKE TEST PASS')
"
```

Measured (Arc Pro B70, warm cache, Qwen3-0.6B, `v0.5.22-xpu`): ready in ~26 s
at `--tp 1`, ~27 s at `--tp 2`; raise the 120 s cap for large weights or a cold
cache. `MODEL` must equal the `id` in `/v1/models` — the value you passed to
`--model`.

Cleanup: `docker stop sglang-xpu && docker rm sglang-xpu` — if it died, read
`docker logs sglang-xpu` first, since `rm` discards the log for good.

## Multi-GPU (TP) — always pass `FI_TCP_IFACE=lo`

**For `--tp > 1` on one host, add `-e FI_TCP_IFACE=lo`. Unconditionally.**
Without it oneCCL's TCP fallback can bind an interface the ranks never reach,
so the server reports *ready* and the **first request never returns**
(re-tested at `v0.5.22-xpu`). Loopback is always right for ranks on one
machine and inert on single GPU. Only exception: `--nnodes > 1` — pin the NIC
the nodes share instead. If a TP launch still hangs, isolate the collective in
~20 s with `scripts/tp_collective_probe.py`; root cause, probe verdicts, and
what does not fix it: `references/multi-gpu-tp.md`.

First discover available XPUs and validate the requested TP degree:

```sh
: "${GPU_COUNT:?run Step 0 first}"
# Set desired TP — must not exceed available XPUs
TP=${TP:-$GPU_COUNT}
if [ "$TP" -gt "$GPU_COUNT" ]; then
    echo "ERROR: requested TP=$TP but only $GPU_COUNT XPU(s) available" >&2
    exit 1
fi

MASK=$(python3 -c "print(','.join(str(i) for i in range($TP)))")
: "${RENDER_GID:?run Step 0 first}" "${IMG:?resolve the image first — see Step 1}"
echo "Launching TP=$TP on XPUs: $MASK (of $GPU_COUNT available) with $IMG"
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
    "$IMG" \
    python3 -m sglang.launch_server \
        --model "<model>" \
        --device xpu --tp "$TP" \
        --attention-backend intel_xpu \
        --disable-overlap-schedule --page-size 64 \
        --host 0.0.0.0 --port 30000
```

**TP is not done until you have validated output** — readiness is exactly
what the all-reduce hang looks like, and a `--tp`/`ZE_AFFINITY_MASK` mismatch
can serve HTTP 200 with garbage. Set these, then run the verify block above
unchanged (no `docker logs` wait); if the smoke request hits `--max-time`, see
"When a TP smoke request times out" in `references/multi-gpu-tp.md`:

```sh
# MODEL must be quoted until you substitute it — bare <model> is a redirect.
NAME=sglang-xpu-tp; PORT=30000; MODEL="<model>"
```

Cleanup: `docker stop sglang-xpu-tp && docker rm sglang-xpu-tp`.

## SGLang flag rationales

- `--device xpu` — explicit; SGLang doesn't auto-detect XPU as cleanly as vLLM.
- `--attention-backend intel_xpu` — verified-good SYCL kernel path, and
  **not the default**: without it XPU resolves to `triton` (slower, patchier;
  2.6× slower at 32 concurrent requests) — which is what the SGLang Cookbook's
  BMG recipes run, so add the flag to them. Pre-Xe2 devices (no XMX) fall back
  to `triton` regardless.
- `--disable-overlap-schedule` — conservative, as in upstream's XPU tests, but
  **not a known-bug fix**: with overlap *on*, 96 concurrent requests all
  returned valid content. Dropping it is a fair A/B.
- `--page-size 64` — `intel_xpu` accepts `64` and `128`; other values are
  rewritten to `128`, except on the MLA path, which takes `32` verbatim.
  **MLA needs `128`**: at 64 or 32, `flash_mla_decode` asserts on
  `block_num % (128/page_size)` and kills the scheduler at the first decode,
  after `/v1/models` had answered normally (v0.5.22). **Hybrid Mamba models
  (Qwen3.5, Nemotron-3-Nano) die at startup on any page size but 1 unless the
  radix cache is off** — flags and their weak prefix caching: `model-recipes.md`.
- `--tp N` — `ZE_AFFINITY_MASK` must expose exactly N XPUs.
- `--trust-remote-code` — **security opt-in, not in the launch lines above**:
  runs arbitrary Python from the model repo. An `auto_map` in `config.json` is
  not proof you need it (DeepSeek-V2-Lite ran without it) — try without first.
- `--disable-radix-cache` — turns off RadixAttention prefix caching (verified:
  `#cached-token` drops to 0). For A/B against vLLM or isolating decode
  throughput. Omit to keep caching on — the default, and the point of SGLang.

## Quantization: what works, what silently fails

| Quant | Status — content-validated on `v0.5.22-xpu` / Arc Pro B70 against the named checkpoint |
|---|---|
| `fp8` | **works** (`Qwen3-0.6B`) — runtime BF16→FP8; KV stays BF16 unless `--kv-cache-dtype fp8_e4m3`. **Every GEMM's trailing dim must be divisible by 16** (XPU uses `torch._scaled_mm`): `gemma-4-26B-A4B-it` (dim 2152) serves `/v1/models`, then kills the scheduler on the first request with `Expected trailing dimension of mat1 to be divisible by 16` |
| `mxfp4` MoE | **works** (`openai/gpt-oss-20b`) via `sgl-kernel-xpu` W4A16; auto-registered on `--device xpu`, no extra flag. Xe2 / BMG only. Add `--reasoning-parser gpt-oss` or the reasoning channel lands in `content` as raw `<\|channel\|>…` tokens |
| `gptq` | **works** (`Qwen2.5-7B-Instruct-GPTQ-Int4`) |
| `awq` | **works** — `Qwen2.5-7B-Instruct-AWQ`, `Qwen2.5-VL-7B-Instruct-AWQ` (text and vision). `Qwen2-VL-7B-Instruct-AWQ` answers text but ignores image content — a Qwen2-VL defect that BF16 shares, not AWQ (`references/spec-decode-and-multimodal.md`) |
| AutoRound | **broken on the asym path** (`kaitchup/Qwen2.5-7B-Instruct-AutoRound-GPTQ-asym-4bit`): HTTP 200 with degenerate repetition (`" Qwen Qwen…"`) at `temperature 0`, still at v0.5.22. Validate any AutoRound checkpoint before trusting it |
| `compressed-tensors` W4A16 | **works** (`RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16`) — new on XPU in v0.5.21 via the torch `int4pack` path (PR #40828) |
| `nvfp4` / `w4a4`, GGUF | **unsupported** — ModelOpt NVFP4 dies at startup (`native dense GEMM backends require SM100+`; `--fp4-gemm-backend marlin` then fails `Torch not compiled with CUDA enabled`); GGUF is CUDA/MUSA/NPU-only → **llamacpp-xpu-run** |
| compressed-tensors NVFP4, W8A8-Int8 | **unsupported** — refused at startup: `CompressedTensorsW4A16Fp4` / `CompressedTensorsW8A8Int8 is not supported on XPU (no XPU kernel implementation)` |
| anything not listed | validate `content` before trusting it |

Never accept HTTP 200 alone — confirm `content` (and `reasoning_content`) is language.

## Also available: spec-decode, multimodal, opt-in features

`references/spec-decode-and-multimodal.md` holds the verified EAGLE, EAGLE3
and **NEXTN / MTP** commands (MTP: Qwen3.5-4B, ~1.8× decode, needs the
hybrid-model flags), LoRA incl. hot-swap, two-batch overlap, MLA / DeepSeek
(`--page-size 128`), multimodal, and the opt-in features: **XPU graph
capture**, **memory saver** (two traps that serve `!!!!…` or kill the
server), **P/D disaggregation**. **Decode context parallelism** (`--dcp-size`,
new in v0.5.22) is in `references/multi-gpu-tp.md` — read it first, it only
helps when `tp_size` exceeds the model's KV heads.

## Common errors

- `xpu count: 0` / `ZE_RESULT_ERROR_UNINITIALIZED` from `zeInit` → **UMD/kernel
  mismatch**: the host `xe` KMD is older than the UMD the image pins. **Suspect
  this first when a newer tag replaces one that worked.** Compare:

  ```sh
  # ^(ii|hi): the image apt-mark-holds these, so dpkg shows "hi", not "ii"
  docker run --rm "$IMG" dpkg -l libze-intel-gpu1 | grep -E "^(ii|hi)"
  dpkg -l libze-intel-gpu1 | grep -E "^(ii|hi)"
  ```

  Fix by upgrading the host stack (**xpu-system-setup**) or serve with
  **vllm-xpu-run**; meanwhile pin `IMG=lmsysorg/sglang:<last-working-tag>`.
- **Benign startup lines that look like failures**: `Ignore import error when
  loading sglang.srt.models.<arch>: No module named 'vllm'` (and `cutlass`),
  `Only CUDA, MUSA and NPU support GGUF`, `'sglang serve' is the recommended
  entrypoint`. So never grep the log for readiness — poll `/v1/models`, and read
  the log only once the verify block reports `DIED` or `NOT READY`.
- `Conflict. The container name "/sglang-xpu" is already in use` → **never
  force-remove it** (`docker rm` with `-f`): on a shared host the name may be
  someone else's run. Inspect it (`docker ps -a --filter name=sglang-xpu`):
  reuse it if it serves your model, `docker stop` it if it is yours, else
  relaunch under another **literal** name (`--name sglang-xpu-2`); never clear
  a name preemptively either. Your own `Exited` container is the common case:
  read `docker logs sglang-xpu`, then `docker rm` (no `-f`).
- HTTP 200 with garbage → all-`!!!!` is a memory-saver resume that dropped the
  weights (`spec-decode-and-multimodal.md`); a **short token repeated**
  (`" Qwen Qwen…"`) is a broken quant path — fall back to BF16. That is why the
  verify block tests for repetition, not length.

Less common (`EACCES`, `libsycl` segfault, Triton `CompilationError`,
`RuntimeError: NCCL`, "Compiling kernel" hang, Docker Hub `429`):
`references/image-and-troubleshooting.md`.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the server sees (`0`, `0,1`, …); must expose exactly `--tp` devices. |
| `FI_TCP_IFACE=lo` | **Every `--tp > 1` launch on one host.** Pins oneCCL's libfabric transport to loopback; without it the first request can hang forever. Wrong only for `--nnodes > 1`. |
| `HF_TOKEN` | HF auth for gated models downloaded inside the container. |

Kernel-cache persistence (`TRITON_CACHE_DIR`, `NEO_CACHE_DIR`, …),
`SYCL_CACHE_PERSISTENT=0`, `SGLANG_SERVER_LAUNCH_TIMEOUT`, debug logging and
proxy notes: `references/image-and-troubleshooting.md`.

## What this skill does NOT cover

- Pure PyTorch / Transformers → **torch-xpu-run**.
- vLLM serving → **vllm-xpu-run**.
- Benchmarking → **sglang-xpu-bench**.

## References

- `references/spec-decode-and-multimodal.md` — spec-decode + multimodal
- `references/model-recipes.md` — hybrid-model flags, TP=4/8 recipes, Cookbook
- `references/multi-gpu-tp.md` — the `--tp > 1` all-reduce hang: fix,
  root cause, py-spy recipe, and what does not fix it
- `references/image-and-troubleshooting.md` — image contents, XPU pre-flight,
  container-flag rationale, less common errors, full env-var table
- SGLang XPU docs: <https://docs.sglang.io/docs/hardware-platforms/xpu>
- Tag list: <https://hub.docker.com/r/lmsysorg/sglang/tags?name=xpu>
- Image recipe: <https://github.com/sgl-project/sglang/blob/main/docker/xpu.Dockerfile>
- Intel AutoRound: <https://github.com/intel/auto-round>
