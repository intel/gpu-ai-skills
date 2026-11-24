---
name: sglang-xpu-run
description: Serve a Hugging Face safetensors model on an Intel GPU using SGLang's XPU backend with the OpenAI-compatible API. Pins the validated `lmsysorg/sglang:v<version>-xpu` release image and flags newer unvalidated releases, then covers the container flag set for Intel DRM devices, the UMD/kernel pairing the image pins, the `--device xpu --attention-backend intel_xpu` flag set, page-size and quantization constraints, multimodal serving, and how to validate output content (not just HTTP 200). Use when the user needs SGLang's RadixAttention prefix caching or grammar-constrained output; for broad-coverage serving on Intel today prefer vllm-xpu-run, and for benchmarking a running server use sglang-xpu-bench.
---

# sglang-xpu-run

SGLang's XPU backend is functional and ships as an official per-release image
(Step 1 pins the validated tag). Verify GPU detection before serving: a container
that sees no XPU dies at startup with a misleading `No module named
'tvm_ffi'` (Common errors).

**Choose SGLang for RadixAttention prefix caching or grammar-constrained
output** — both verified here (`#cached-token` reuse observed;
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
| `--tp 2` | `--tp 2` + `ZE_AFFINITY_MASK=0,1` + `FI_TCP_IFACE=lo` |
| `--quantization awq` / `fp8` | both work with caveats — see the quantization table |
| conda env + `setvars.sh` | nothing to activate — see below |

## Step 1 — resolve the image tag, then pull

**There is no `latest-xpu` tag** — sgl-project ships one `v<version>-xpu` image per release.
**Serve `$PIN`, the tag this pack validated.** The lookup below only reports a newer release; it does not switch to it automatically.

```sh
PIN=v0.5.23-xpu                 # the only tag validated end-to-end by this pack
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

An unreachable registry prints no note. Everything below is **validated at
`v0.5.23-xpu`** (on Arc Pro B70 cards). **If the note fires, do not
silently switch.** To evaluate a newer tag: set `IMG`, check the UMD/kernel
pairing first (`xpu count: 0` under Common errors), then serve flags and the
quant table, then re-run the verify block. Report the tag you served.

Nothing to wrap around the launch: `/opt/venv` is already on `PATH` (no conda
env), oneAPI vars are baked in (do **not** source `setvars.sh`), the container
is root with no entrypoint, and the Level Zero UMD is pinned and held
(compute-runtime `26.27.39122.11`, `torch 2.14.0+xpu` at this tag; a release
can bump them). Details: `references/image-and-troubleshooting.md`.

## Why each container flag

`--device /dev/dri` + `/by-path` mount (Intel has no `--gpus`; without the
mount `--tp 2` fails `onecclCommInitRank … code 11`). `--group-add`: needed
once you add `--user`; **`--privileged` is not**. `--network host`: simplest
(`-p 30000:30000` also works). `--shm-size`/`--ipc=host`: used by every
validated run, though text serving also worked without. Full rationale and the
XPU pre-flight: `references/image-and-troubleshooting.md`.

## Quickstart — serve one text-gen model

The container downloads weights itself into the mounted cache — do not install
`hf` or hunt for a CLI, just launch.

**Swap the model, keep the flags — except**: Qwen3.5/hybrid dies at startup
(use the `model-recipes.md` command), MLA/DeepSeek needs `--page-size 128`,
gpt-oss `--reasoning-parser gpt-oss`, Qwen3/Qwen3.5 `--reasoning-parser qwen3`
(else thinking lands in `content`). Recipes there omit
`--disable-overlap-schedule`; both work.

Launch (single GPU — for multi-GPU see below):

```sh
: "${RENDER_GID:?run Step 0 first}" "${IMG:?resolve the image first — see Step 1}"

# Keep the name literal (cleanup tooling matches it). If taken, stop here —
# never force-remove it (Common errors).
docker ps -a --format '{{.Names}}' | grep -qx sglang-xpu && {
    echo "sglang-xpu exists — reuse it, stop it if it is yours, or pick another literal name"; exit 1; }
# SGLang answers ANY model name, so a stranger's server on this port would pass
# the verify block. Pick another --port (and PORT=) if this fires.
ss -ltn | grep -q ':30000 ' && { echo "port 30000 is in use"; exit 1; }

# No --rm: a container that dies in init would take its log (the diagnosis).
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
foreground** with a ≥400 s tool timeout (worst case is 3 × 120 s;
backgrounding hides the proof, the `content:` line). It reads
`NAME`/`PORT`/`MODEL` from the shell (quickstart defaults otherwise) and is
bounded on every path.

```sh
NAME=${NAME:-sglang-xpu}; PORT=${PORT:-30000}; MODEL=${MODEL:-Qwen/Qwen3-0.6B}

# 1. Readiness: /v1/models must list $MODEL (any 200 could be another server).
#    Never loop on `docker logs` — the log is for the post-mortem.
up() { curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/v1/models" | grep -qF "\"$MODEL\""; }
T0=$(date +%s)
while [ $(( $(date +%s) - T0 )) -lt 120 ]; do
    up && { echo "READY after $(( $(date +%s) - T0 ))s"; break; }
    docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "DIED — run: docker logs --tail 50 $NAME"; break; }
    sleep 2
done
up || { echo "NOT READY (or port serves another model) — run: docker logs --tail 50 $NAME"; exit 1; }

# 2. Warmup absorbs kernel compilation; --max-time catches a TP hang, -f a
#    scheduler that died on its first request (the container stays Up).
curl -sf --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":8}" \
    >/dev/null || { echo "WARMUP FAILED (HTTP error or no answer) — run: docker logs --tail 50 $NAME"; exit 1; }

# 3. Smoke test: validate CONTENT, not HTTP 200 — and never gate on length.
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

With weights cached every launch here (0.6B–30 GB, `--tp 1`/`2`, with or
without a draft) was ready in 22–35 s; raise the 120 s cap only while weights
download. `MODEL` must equal the `--model` value. On thinking models the 80
tokens are all reasoning (`finish_reason` `length`) — that proves language,
not the answer; send a longer request to read one.

**PASS is necessary, not sufficient**: it catches empty, `!!!!` and repetition,
not subtly wrong text. Read the `content:` line — stray other-script tokens
(`счет`) or stutter (`Paris..`) mean a broken path, typically FP8 KV.

Cleanup: `docker stop sglang-xpu && docker rm sglang-xpu` — if it died, read
`docker logs sglang-xpu` first, since `rm` discards the log for good.

## Multi-GPU (TP) — always pass `FI_TCP_IFACE=lo`

**For `--tp > 1` on one host, add `-e FI_TCP_IFACE=lo`. Unconditionally.**
Without it oneCCL's TCP fallback can bind an interface the ranks never reach,
so the server reports *ready* and the **first request never returns**
(re-tested on the pinned tag). Wrong only for `--nnodes > 1` (pin the shared
NIC). If TP still hangs, run `scripts/tp_collective_probe.py` (self-bounding);
root cause and what does not fix it: `references/multi-gpu-tp.md`.

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
ss -ltn | grep -q ':30000 ' && { echo "port 30000 is in use"; exit 1; }

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

- `--device xpu` — always pass it; every validated run (and Cookbook BMG command) does.
- `--attention-backend intel_xpu` — verified-good SYCL kernel path, and
  **not the default**: without it XPU resolves to `triton` (slower, patchier;
  2.8× slower at 32 concurrent requests) — which is what most SGLang Cookbook
  BMG recipes run, so add the flag to them (exceptions: `model-recipes.md`).
- `--disable-overlap-schedule` — conservative, but
  **not a known-bug fix**: with overlap *on*, 96 concurrent requests all
  returned valid content. Dropping it is a fair A/B.
- `--page-size 64` — `intel_xpu` accepts `64` and `128`; other values are
  rewritten to `128`, except on the MLA path, which takes `32` verbatim.
  **MLA needs `128`**: at 64 or 32, `flash_mla_decode` asserts on
  `block_num % (128/page_size)` and kills the scheduler at the first decode,
  after `/v1/models` had answered normally. **Hybrid Mamba models
  (Qwen3.5) die at startup on any page size but 1 unless the radix cache is
  off** — flags and their weak prefix caching: `model-recipes.md`.
- `--trust-remote-code` — **security opt-in, not in the launch lines above**:
  runs arbitrary Python from the model repo. An `auto_map` in `config.json` is
  not proof you need it (DeepSeek-V2-Lite ran without it) — try without first.
- `--disable-radix-cache` — turns off RadixAttention prefix caching (verified:
  `#cached-token` drops to 0). For A/B against vLLM or isolating decode
  throughput. Omit to keep caching on — the default, and the point of SGLang.

## Quantization: what works, what silently fails

| Quant | Status — content-validated on Arc Pro B70 against the named checkpoint |
|---|---|
| `fp8` | **works** (`Qwen3-0.6B`) — runtime BF16→FP8 weights; KV stays BF16. **`--kv-cache-dtype fp8_e4m3` kills `intel_xpu` at the first request** (`fp8 KV cache requires k_descale and v_descale`); `triton` serves it, but **corrupts Qwen2.5** while the verify block passes (per-model results: `references/image-and-troubleshooting.md`) — compare against BF16 KV. **Every GEMM's trailing dim must be divisible by 16** (XPU uses `torch._scaled_mm`): `gemma-4-26B-A4B-it` (dim 2152) serves `/v1/models`, then kills the scheduler on the first request with `Expected trailing dimension of mat1 to be divisible by 16` |
| `mxfp4` MoE | **works** (`openai/gpt-oss-20b`) via `sgl-kernel-xpu`'s Xe2 W4A16 kernel (`xe20`; upstream lists Xe2/BMG), auto-registered on `--device xpu` — no extra flag. Add `--reasoning-parser gpt-oss` or the reasoning channel lands in `content` as raw `<\|channel\|>…` tokens |
| `gptq` | **works** (`Qwen2.5-7B-Instruct-GPTQ-Int4`) |
| `awq` | **works** — `Qwen2.5-7B-Instruct-AWQ`, `Qwen2.5-VL-7B-Instruct-AWQ` (text and vision). (Qwen2-VL ignores images in BF16 too, so not AWQ's fault — `references/spec-decode-and-multimodal.md`) |
| AutoRound | **broken on the asym path** (`kaitchup/Qwen2.5-7B-Instruct-AutoRound-GPTQ-asym-4bit`): HTTP 200 with degenerate repetition (`"Assistant Assistant…"`) at `temperature 0`. Validate any AutoRound checkpoint before trusting it |
| `compressed-tensors` W4A16 | **works** (`RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16`) via the torch `int4pack` path |
| `nvfp4` / `w4a4`, GGUF | **unsupported** — ModelOpt NVFP4 dies at startup (`native dense GEMM backends require SM100+`; `--fp4-gemm-backend marlin` then fails `Torch not compiled with CUDA enabled`); GGUF is CUDA/MUSA/NPU-only → **llamacpp-xpu-run** |
| compressed-tensors NVFP4, W8A8-Int8 | **unsupported** — refused at startup: `CompressedTensorsW4A16Fp4` / `CompressedTensorsW8A8Int8 is not supported on XPU (no XPU kernel implementation)` |
| anything not listed | validate `content` before trusting it |

Never accept HTTP 200 alone — confirm `content` (and `reasoning_content`) is language.

## Also available: spec-decode, multimodal, opt-in features

`references/spec-decode-and-multimodal.md` holds the verified EAGLE, EAGLE3
(2.1–2.7×), **NEXTN / MTP** (Qwen3.5-4B, hybrid-model flags) and **DSpark**
(Qwen3-14B, `--tp 2`, 1.7×) commands, LoRA incl. hot-swap, two-batch overlap,
MLA / DeepSeek (`--page-size 128`), multimodal (Gemma-3 needs
`--enable-multimodal`), and the opt-in features: **XPU graph capture**,
**memory saver** (two traps that serve `!!!!…` or kill the server),
**HiCache** (always pass `--hicache-size`). EP is not supported on BMG.
**Decode context parallelism** (`--dcp-size`) is in `references/multi-gpu-tp.md`
— read it first, it only helps when `tp_size` exceeds the model's KV heads.

## Common errors

- `xpu count: 0` / `ZE_RESULT_ERROR_UNINITIALIZED` from `zeInit` → **UMD/kernel
  mismatch**: the host `xe` KMD (the kernel, `uname -r`) is older than the
  UMD the image pins. **Suspect this first when a newer tag replaces one that
  worked.** The host's own `libze-intel-gpu1` is irrelevant: a host 26.05 UMD
  serves this image's 26.27 fine on kernel 6.17. Fix by upgrading the host stack (**xpu-system-setup**) or serve with
  **vllm-xpu-run**; meanwhile pin `IMG=lmsysorg/sglang:<last-working-tag>`.
- `ModuleNotFoundError: No module named 'tvm_ffi'` at startup → **the
  container sees no XPU** (no `--device /dev/dri`, a `ZE_AFFINITY_MASK` naming
  absent cards, or the mismatch above), so SGLang takes the CUDA import path.
  Not a missing package — run the pre-flight in
  `references/image-and-troubleshooting.md`.
- **Benign startup lines that look like failures**: `Ignore import error when
  loading sglang.srt.models.<arch>: No module named 'vllm'` (and `cutlass`),
  `Only CUDA, MUSA and NPU support GGUF`, `'sglang serve' is the recommended
  entrypoint`, `OpenAI Responses API (/v1/responses) disabled` (below). So
  never grep the log for readiness — poll `/v1/models`, and read the log only once the verify block reports `DIED` or `NOT READY`.
- `Conflict. The container name "/sglang-xpu" is already in use` → **never
  force-remove it** (`docker rm` with `-f`): on a shared host it may be
  someone else's run. Reuse it if it serves your model, stop it if yours, else relaunch under another
  **literal** name (`--name sglang-xpu-2`). Your own `Exited` one: read
  `docker logs sglang-xpu`, then `docker rm`.
- `/v1/responses` → HTTP 500 (the startup log already said it is disabled:
  `cannot import name 'ResponseFunctionShellToolCall'`): the image's `openai`
  SDK is too old for this sglang. Use `/v1/chat/completions` or `/v1/completions`.
- HTTP 200 with garbage → all-`!!!!` is a memory-saver resume that dropped the
  weights (`spec-decode-and-multimodal.md`); a **short token repeated**
  (`"Assistant Assistant…"`) is a broken quant path — fall back to BF16.

Less common (`xpu count: 0` under `--user`, `onecclCommInitRank … code 11`,
`Unsupported head size`, a very slow first request, Docker Hub `429`):
`references/image-and-troubleshooting.md`.

## Env vars

`ZE_AFFINITY_MASK` (must expose exactly `--tp` XPUs), `FI_TCP_IFACE=lo` (every
`--tp > 1`, above) and `HF_TOKEN` (gated downloads) are in the launch lines.
Kernel-cache persistence (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`;
Qwen3.5's first request ~3.8× faster on restart), `HF_HUB_OFFLINE` and the
spec-decode context override: `references/image-and-troubleshooting.md`.

## What this skill does NOT cover

- Pure PyTorch / Transformers → **torch-xpu-run**.
- vLLM serving → **vllm-xpu-run**.
- Benchmarking → **sglang-xpu-bench**.

## References

- `references/spec-decode-and-multimodal.md` — spec-decode + multimodal
- `references/model-recipes.md` — hybrid-model flags, SGLang Cookbook BMG recipes
- `references/multi-gpu-tp.md` — the `--tp > 1` all-reduce hang: fix,
  root cause, py-spy recipe, and what does not fix it
- `references/image-and-troubleshooting.md` — image contents, XPU pre-flight,
  container-flag rationale, less common errors, full env-var table
- SGLang Cookbook (pick **BMG** in a model page's generator): <https://docs.sglang.io/cookbook/intro>
- Tag list: <https://hub.docker.com/r/lmsysorg/sglang/tags?name=xpu>
- Image recipe: <https://github.com/sgl-project/sglang/blob/main/docker/xpu.Dockerfile>
- Intel AutoRound: <https://github.com/intel/auto-round>
