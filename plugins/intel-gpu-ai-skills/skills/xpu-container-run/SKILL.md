---
name: xpu-container-run
description: Launch a Docker container with Intel GPU access on Linux. Encodes the correct combination of `--device /dev/dri`, render-group access, `--ipc=host`, `ZE_AFFINITY_MASK` pinning, Hugging Face cache mount, and `--entrypoint /bin/bash` for interactive use. Use when running any Intel-XPU container (vLLM-XPU, sglang-xpu, torch-XPU, llama.cpp SYCL, etc.) and the device must be visible inside. The CUDA analogue is `docker run --gpus all` — Intel has no `--gpus` flag, you pass the Direct Rendering Manager (DRM) nodes directly.
---

# xpu-container-run

Intel GPUs do not plug into Docker via `--gpus all`. There is no
`nvidia-container-toolkit` equivalent. Pass the kernel's Direct
Rendering Manager (DRM) character devices into the container and
grant the right group ownership.

## CUDA → Intel cheat sheet

| CUDA | Intel |
|---|---|
| `docker run --gpus all` | `--device /dev/dri --group-add "$(getent group render \| cut -d: -f3)"` |
| `docker run --gpus '"device=0"'` | `-e ZE_AFFINITY_MASK=0` |
| `--ipc=host` | same |
| `--shm-size=16g` | same (alternative to `--ipc=host`) |
| `--runtime nvidia` | nothing — `xe`/`i915` is in-kernel |
| `nvidia-smi` inside container | `xpu-smi discovery` |

No "Intel container toolkit" needed; passing the DRM nodes is enough.

## Image source

Comes from the runner skill:
- vLLM serving → **vllm-xpu-run** (`intel/vllm:<version>-xpu`)
- SGLang → **sglang-xpu-run** (built from upstream `docker/xpu.Dockerfile`)
- PyTorch / Transformers → **torch-xpu-run**

`<image>` below is whichever you picked.

## Quickstart — interactive shell, one GPU

Confirm the image name with the user before running — this binds
host GPU devices into the container.

```sh
docker run --rm -it \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint /bin/bash \
    <image>
```

| Flag | Why |
|---|---|
| `--device /dev/dri` | Pass every Intel GPU's DRM nodes. Use `--device /dev/dri/renderD128` for just the first GPU's render node (least privilege). |
| `--group-add "$(getent group render \| cut -d: -f3)"` | Joins the container user to the host's `render` group by **GID** (not name) so it works in images where a `render` group with a different GID — or no `render` group at all — exists. Required when nodes are mode 0660/0640. Skip causes `EACCES` on Level Zero init. |
| `--ipc=host` | vLLM and `torch.distributed` use `/dev/shm` and POSIX semaphores. `--shm-size=16g` is a private-IPC alternative. |
| `-e ZE_AFFINITY_MASK=0` | Pin to GPU 0. See **xpu-discover** for IDs. Always set explicitly. |
| `-v ~/.cache/huggingface:...` | Share the host model cache; avoid re-download. |
| `--entrypoint /bin/bash` | Override server-image autostart for interactive use. |

## When `--privileged` is needed

Exception, not rule. Required only for:

- `unitrace` / VTune collectors that read PMU MSRs.
- GPU firmware updates (`xpu-smi updatefw`).
- `xpu-smi diag --singletest 5` (PCIe bandwidth) and similar
  low-level diag tests.

For running models and most profiling, `--device /dev/dri` is
enough. Add `--privileged` only when you hit a specific permission
failure pointing at it.

## Server mode, multi-GPU, `--net=host`

See `references/server-and-multi-gpu.md` for daemon-style server
launches, one-process-per-GPU vs single-process TP / PP layouts,
the oneCCL `CCL_ZE_IPC_EXCHANGE=pidfd` setting for multi-XPU TP,
and when `--net=host` is actually needed.

## Verifying the container sees the GPU

```sh
xpu-smi discovery
```

| Symptom | Cause | Fix |
|---|---|---|
| `xpu-smi: command not found` | image lacks `xpu-smi` | use a different image or skip this check |
| empty `discovery` table | no `/dev/dri` passed | add `--device /dev/dri` |
| `Level Zero init failed` / `EACCES` | user not in `render` group | add `--group-add "$(getent group render | cut -d: -f3)"` |
| wrong GPU count | `ZE_AFFINITY_MASK` inherited from host | pass mask explicitly with `-e` |
| `diag` works on host, fails in container | container not privileged | add `--privileged`, or skip diag inside container |

## Common errors

- `failed to create shim task: permission denied` → container
  runtime can't open `/dev/dri/card0`. Add `--privileged` or check
  host file mode.
- `LIBZE_LOADER: Failed to load level-zero loader` → image missing
  `libze1` / `intel-level-zero-gpu`. Use a different image.
- `RuntimeError: Cannot find any XPU devices` (PyTorch) → Level
  Zero loaded but no device visible. Re-check `ZE_AFFINITY_MASK`
  and run `xpu-smi discovery` in the container.
- `bus error` early in vLLM/PyTorch startup → shared memory too
  small. Use `--ipc=host` or raise `--shm-size`.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) visible. |
| `HF_TOKEN` | Hugging Face auth. |
| `HF_HOME` | Override in-container HF cache path. |
| `HUGGINGFACE_HUB_CACHE` | Older alias; some images still use it. |
| `OMP_NUM_THREADS` | Cap CPU threads; `1` for single-process serving. |
| `CCL_ZE_IPC_EXCHANGE=pidfd` | Multi-GPU-friendly oneCCL IPC mechanism. |
| `IGC_EnableAluBinding=1` | Battlemage matmul-codegen hint; bench both. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:0` | Belt-and-suspenders pin alongside `ZE_AFFINITY_MASK`. |

## References

- `references/server-and-multi-gpu.md` — server mode, multi-GPU, `--net=host`
- Linux DRM device interface: <https://docs.kernel.org/gpu/drm-uapi.html>
- Level Zero loader: <https://oneapi-src.github.io/level-zero-spec/>
- Intel `xe` driver: <https://docs.kernel.org/gpu/xe/index.html>
