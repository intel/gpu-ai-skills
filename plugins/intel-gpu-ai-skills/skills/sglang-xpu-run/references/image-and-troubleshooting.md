# Image contents, XPU pre-flight, container flags, less common errors, env vars

Everything here is
verified on `lmsysorg/sglang:v0.5.23-xpu` / Arc Pro B70 unless a line says
otherwise.
`$IMG` and `$RENDER_GID` are the values from SKILL.md Steps 0–1.

## What the release image already contains

Nothing needs wrapping around `python3 -m sglang.launch_server`:

- Python lives in a venv at `/opt/venv` that is **already on `PATH`**. There is
  no conda env to activate.
- oneAPI env vars are baked into the image config (`SETVARS_COMPLETED=1`) — do
  **not** source `setvars.sh`.
- The container runs as **root** with `CMD ["bash"]` and no entrypoint, so
  `--entrypoint` is unnecessary and `/root/.cache/huggingface` is the Hub cache
  path sglang reads.
- Level Zero UMD pinned + `apt-mark hold`-ed: compute-runtime `26.27.39122.11`,
  IGC `2.38.2`, gmmlib `22.10.0`; `torch 2.14.0+xpu`, `sglang-kernel-xpu 0.5.0`,
  `triton-xpu 3.8.0`, oneAPI `2026.1`. A release can bump any of these — read
  them off the image you actually resolved:

  ```sh
  # ^(ii|hi): the image apt-mark-holds these, so dpkg shows "hi", not "ii"
  docker run --rm "$IMG" dpkg -l libze-intel-gpu1 intel-igc-core-2 | grep -E "^(ii|hi)"
  uname -r                                          # host kernel = the xe KMD
  ```

  The host's own UMD package does not matter — the container brings its own
  (verified: host `libze-intel-gpu1 26.05` with this image's `26.27` serves
  fine on kernel `6.17`). If the host `xe` KMD is older than the UMD the image
  pins, `zeInit` fails
  (`xpu count: 0`, `ZE_RESULT_ERROR_UNINITIALIZED`). Fix by upgrading the host
  stack (**xpu-system-setup**) or serve with **vllm-xpu-run**; meanwhile pin
  `IMG=lmsysorg/sglang:<last-working-tag>`.

## Pre-flight: verify XPU is visible inside the container

Optional when a launch follows immediately — it costs a container start, and
the verify block reports `DIED` with the same Level Zero error. Run it when
*diagnosing* (no XPU found, `zeInit` failure, a new host), not as a gate.

```sh
: "${RENDER_GID:?run Step 0 first}" "${IMG:?resolve the image first — see Step 1}"

docker run --rm \
    --network host --ipc=host --shm-size=16g \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --group-add "$RENDER_GID" \
    -e ZE_AFFINITY_MASK=0 \
    "$IMG" \
    python3 -c 'import torch; n = torch.xpu.device_count(); \
                print("xpu count:", n); \
                print(torch.xpu.get_device_name(0) if n else "NO DEVICE"); \
                assert n > 0, "NO XPU VISIBLE"'
```

`xpu count: 0` → the UMD/kernel mismatch above, or `/dev/dri/by-path` not
mounted. Expected on the verified host: `xpu count: 1`,
`Intel(R) Arc(TM) Pro B70 Graphics` (2 with `ZE_AFFINITY_MASK=0,1`).

## Why each container flag, in full

- `--device /dev/dri` + `-v /dev/dri/by-path:/dev/dri/by-path` — Intel has no
  `--gpus` flag; you pass the DRM nodes directly. Without the `/by-path` mount
  the pre-flight still finds one card, but a `--tp 2` server reports ready and
  its first request fails with `onecclCommInitRank failed with code 11` —
  oneCCL needs the symlinks. Keep both.
- `--group-add "$RENDER_GID"` — `/dev/dri/renderD*` is mode 660 owned by the
  `render` group. This image runs as root, which bypasses the group check, so
  the flag is not strictly required here — keep it anyway: it is what makes the
  same command work if you add `--user` (verified: `--user 1000:1000` alone
  gives `xpu count: 0`; adding `--group-add "$RENDER_GID"` gives 1).
- **`--privileged` is not required** — every run in this skill used none.
- `--network host` — simplifies port handling (sglang uses 30000 + internal RPC
  ports). Swapping it for `-p 30000:30000` also sidesteps the multi-GPU
  interface trap (`multi-gpu-tp.md`); bridge mode was verified unaffected.
- `--shm-size=16g` / `--ipc=host` — every run in this skill used both. Not
  required for text serving: Qwen3-0.6B at `--tp 1` and `--tp 2` served 32
  concurrent requests with neither (Docker's 64 MB default shm).
- `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"` — reuse the host Hub
  cache. The container is root, so `/root/.cache/huggingface` is the path
  sglang reads.
- No `--rm` on serving containers — a container that dies during init takes
  its log with it, and that log is the whole diagnosis. Remove explicitly at
  cleanup, after reading `docker logs` if it died.

## Less common errors

The four common ones (UMD mismatch, benign startup lines, container-name
conflict, HTTP 200 with garbage) are in SKILL.md.

- `xpu count: 0` after adding `--user` → the non-root user is not in the
  `render` group; add `--group-add "$(getent group render | cut -d: -f3)"`.
- `RuntimeError: xccl: onecclCommInitRank failed with code 11` on the first
  TP request → oneCCL could not set up its communicator. Two causes seen:
  the `/dev/dri/by-path` mount is missing (add it), or `--mem-fraction-static`
  left no device memory for it (Gemma-4-26B at 0.9 on 2× 32 GB; 0.85 works) —
  lower the fraction. The collective probe shows the
  same code for `FI_TCP_IFACE=docker0`, `CCL_ZE_ENABLE=0` and
  `FI_PROVIDER=shm` (`multi-gpu-tp.md`).
- A very slow first request on a server that reported ready is JIT
  compilation, not a hang: ~12× the `intel_xpu` time for Qwen3.5-4B on the
  default `triton` linear-attention backend, ~5× for Llama-3.1-8B on `triton`
  attention. The verify block's warmup absorbs it.
- `429 Too Many Requests` pulling the image → Docker Hub rate-limiting
  anonymous pulls. `docker login` for higher limits, or retry later.
- `/v1/responses` returns HTTP 500 (`AttributeError: 'State' object has no
  attribute 'openai_serving_responses'`) and startup logs `OpenAI Responses API
  (/v1/responses) disabled: … cannot import name 'ResponseFunctionShellToolCall'`.
  The image ships `openai 2.6.1` (`python/pyproject_xpu.toml`) while this
  sglang's Responses code needs `2.8.0` (main `pyproject.toml`). Use
  `/v1/chat/completions` or `/v1/completions`.
- `ModuleNotFoundError: No module named 'tvm_ffi'` at startup → the
  container sees no XPU. Reproduced with no `--device /dev/dri` and with
  `ZE_AFFINITY_MASK=7` on a 2-card host, with or without `--device xpu`:
  SGLang then imports its CUDA communication path. There is no CPU fallback.
- `RuntimeError: Unsupported head size for paged prefill attention: 80` on the
  first request (e.g. `microsoft/phi-2`) → the `intel_xpu` paged kernels do not
  cover that head size. Serve with `--attention-backend triton` (phi-2 then
  answers correctly via `/v1/completions`; it has no chat template).

## FP8 KV cache (`--kv-cache-dtype fp8_e4m3`): per-model results

`intel_xpu` refuses it at the first request (`fp8 KV cache requires k_descale
and v_descale`). `triton` serves it with unit scales (`Using FP8 KV cache but
no scaling factors provided`) and doubles the KV pool, but quality depends on
the model. `--attention-backend triton --kv-cache-dtype fp8_e4m3`, `temperature
0`, compared with the same model on BF16 KV:

| Model | Output |
|---|---|
| Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen3-0.6B, Qwen3-8B | clean |
| Qwen2.5-7B-Instruct | **corrupt**: `"Paris..\n\nTwo other European capitals are счет Europe, capitals are счет areSCORE"` — the verify block still prints PASS. `fp8_e5m2` too |
| Qwen2.5-1.5B-Instruct | degraded: stutter (`"body of of water"`) and wrong facts BF16 gets right |

Calibrated per-layer scales via `--quantization-param-path` do not rescue
Qwen2.5: the server logs `Loaded KV cache scaling factors`, then the first
request returns HTTP 500 (`AttributeError: 'NoneType' object has no attribute
'type'`) while the container stays `Up`. Use BF16 KV for Qwen2.5.

## Env vars, full table

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the server sees (`0`, `0,1`, …). Must expose exactly `--tp` devices. |
| `FI_TCP_IFACE=lo` | **Set for every `--tp > 1` launch on one host.** Pins libfabric (oneCCL's OFI transport) to loopback; without it the TP all-reduce can bind an unreachable interface and the first request hangs forever. Wrong only for `--nnodes > 1` — pin the shared NIC there. |
| `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` | Persist compiled kernels across runs: point both into a mounted host volume (`-v /path/kcache:/kcache -e TRITON_CACHE_DIR=/kcache/triton -e TORCHINDUCTOR_CACHE_DIR=/kcache/inductor`). Measured: Qwen3.5-4B with the default `triton` linear backend, first request ~3.8× faster on the next start than on a cold cache. The container writes the files as root, so clean the volume up from a container. |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | Needed by the EAGLE/EAGLE3 drafts whose config declares a shorter context than the target (`spec-decode-and-multimodal.md`). |
| `HF_TOKEN` | HF auth when downloading inside the container. |
| `HF_HUB_OFFLINE=1` | Serve only from the mounted cache; a missing file then fails fast instead of waiting on the network. |
