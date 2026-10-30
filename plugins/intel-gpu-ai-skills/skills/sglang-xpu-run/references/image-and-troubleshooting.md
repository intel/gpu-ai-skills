# Image contents, XPU pre-flight, container flags, less common errors, env vars

Detail moved out of SKILL.md so the body stays small. Everything here is for
`lmsysorg/sglang:v0.5.22-xpu` on Arc Pro B70 unless a line says otherwise.
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
  IGC `2.38.2`, gmmlib `22.10.0`; `torch 2.14.0+xpu`, `sglang-kernel-xpu 0.4.0`,
  `triton-xpu 3.8.0`. Each release bumps these (v0.5.21 pinned compute-runtime
  `26.18.38308.1`, IGC `2.34.4`, torch `2.13.0+xpu`) — read them off the image
  you actually resolved:

  ```sh
  # ^(ii|hi): the image apt-mark-holds these, so dpkg shows "hi", not "ii"
  docker run --rm "$IMG" dpkg -l libze-intel-gpu1 intel-igc-core-2 | grep -E "^(ii|hi)"
  dpkg -l libze-intel-gpu1 | grep -E "^(ii|hi)"     # host, for comparison
  ```

  If the host `xe` KMD is older than the UMD the image pins, `zeInit` fails
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
  `--gpus` flag; you pass the DRM nodes directly, and Level Zero discovers
  devices through the `/by-path` symlinks. Without the bind mount some images
  report no XPU.
- `--group-add "$RENDER_GID"` — `/dev/dri/renderD*` is mode 660 owned by the
  `render` group. This image runs as root, which bypasses the group check, so
  the flag is not strictly required here — keep it anyway: it is what makes the
  same command work if you add `--user`, and upstream CI adds both the `video`
  and `render` GIDs.
- **`--privileged` is not required.** Earlier non-root sglang images needed it
  on Battlemage; the pinned-UMD release image does not (verified on Arc Pro
  B70). Only reach for it if `zeInit` fails with a permission error that
  `--group-add` doesn't fix.
- `--network host` — simplifies port handling (sglang uses 30000 + internal RPC
  ports). Swapping it for `-p 30000:30000` also sidesteps the multi-GPU
  interface trap (`multi-gpu-tp.md`); bridge mode was verified unaffected.
- `--shm-size=16g` (or `--ipc=host`) — sglang's scheduler uses shared memory
  more aggressively than vLLM. Default Docker shm is too small.
- `-v "$HOME/.cache/huggingface:/root/.cache/huggingface"` — reuse the host Hub
  cache. The container is root, so `/root/.cache/huggingface` is the path
  sglang reads.
- No `--rm` on serving containers — a container that dies during init takes
  its log with it, and that log is the whole diagnosis. Remove explicitly at
  cleanup, after reading `docker logs` if it died.

## Less common errors

The four common ones (UMD mismatch, benign startup lines, container-name
conflict, HTTP 200 with garbage) are in SKILL.md.

- `EACCES` on `/dev/dri/renderD*` → you added `--user`; add
  `--group-add "$(getent group render | cut -d: -f3)"`.
- Segfault inside `libsycl` when a cached kernel is reloaded → set
  `SYCL_CACHE_PERSISTENT=0` (what upstream CI does on this runtime).
- `triton.compiler.errors.CompilationError` on first request → cold Triton
  cache. Mount `TRITON_CACHE_DIR` to a host volume.
- `RuntimeError: NCCL` → sglang prints "NCCL" on the XPU path; the actual
  backend is oneCCL. Usually a `--tp` / `ZE_AFFINITY_MASK` mismatch.
- Server hangs at "Compiling kernel" → cold kernel cache on first run. Watch
  `xpu-smi dump -d 0 -m 18 -i 1` — high power = compiling, not stuck. Persist
  the cache for next run (env vars below).
- `429 Too Many Requests` pulling the image → Docker Hub rate-limiting
  anonymous pulls. `docker login` for higher limits, or retry after 60 s.
- Multimodal models on older tags printed ~90 lines of benign
  `OSError: libavutil.so.NN` tracebacks from `torchcodec` probing FFmpeg; gone
  at v0.5.22. Harmless if seen on an older image.

## Env vars, full table

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the server sees (`0`, `0,1`, …). Must expose exactly `--tp` devices. |
| `FI_TCP_IFACE=lo` | **Set for every `--tp > 1` launch on one host.** Pins libfabric (oneCCL's OFI transport) to loopback; without it the TP all-reduce can bind an unreachable interface and the first request hangs forever. Wrong only for `--nnodes > 1` — pin the shared NIC there. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` | Keep SYCL's device view consistent with the `ZE_AFFINITY_MASK`-filtered set. |
| `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `NEO_CACHE_DIR` + `NEO_CACHE_PERSISTENT=1` | Persist compiled kernels across runs (mount to a host volume). |
| `SYCL_CACHE_PERSISTENT=0` | Disable the SYCL persistent kernel cache; upstream CI keeps it off to avoid a `libsycl` crash on reload. |
| `SGLANG_SERVER_LAUNCH_TIMEOUT` | Raise for large MoE loads from a cold Hub cache (upstream CI uses 36000). |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | Needed by the EAGLE/EAGLE3 drafts whose config declares a shorter context than the target (`spec-decode-and-multimodal.md`). |
| `SYCL_UR_USE_LEVEL_ZERO_V2=0` | Force the L0 v1 adapter. Not needed with this image (device discovery verified without it); keep as a fallback if `zeInit`/enumeration misbehaves. |
| `HF_TOKEN` | HF auth when downloading inside the container. |
| `HF_HUB_OFFLINE=1` | Serve only from the mounted cache; a missing file then fails fast instead of waiting on the network. |
| `SGLANG_LOGGING_LEVEL=DEBUG` | Verbose engine logs. |
| `ALL_PROXY` / `all_proxy` | Unset if set to `socks://` — the `hf` CLI doesn't support SOCKS. |
