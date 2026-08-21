# Troubleshooting Reference

## Common errors

| Symptom | Cause | Fix |
|---|---|---|
| `level-zero` deb install fails with exit 100 during `docker build` | Base image already has a newer `libze-intel-gpu1` | Remove the Level Zero deb step from the Dockerfile — see `build-and-env.md`. |
| `libsycl.so: cannot open shared object` | oneAPI env not sourced | Use the official `intel.Dockerfile` — it sets up LD paths. Bare-metal: `source /opt/intel/oneapi/setvars.sh`. |
| `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` | Model + KV cache exceeds VRAM | Reduce `-c`, use a smaller quant, or split across GPUs. |
| `can't allocate >4GB on device` | Level Zero relaxed limits not active | Set `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. |
| `detect 0 SYCL GPUs` | Wrong `ONEAPI_DEVICE_SELECTOR` or iGPU-only | Run `--list-devices` (see below). Check device index order. |
| No dGPU visible — only iGPU appears | `ONEAPI_DEVICE_SELECTOR` not set or wrong index | Set `ONEAPI_DEVICE_SELECTOR="level_zero:N"` where N is the dGPU index from `--list-devices`. |
| `error: unknown value for --flash-attn: '--host'` | Bare `--flash-attn` flag used on b9494+ | Add explicit value: `--flash-attn on`. See `build-and-env.md`. |
| Garbled / incoherent output | FP16 precision loss | Rebuild with `GGML_SYCL_F16=OFF` (default). |
| `split-mode row` failure | Not implemented for SYCL | Use `--split-mode layer` or `--split-mode none`. |
| Server never becomes healthy (JIT hang) | First-load JIT compilation on large model | Normal — can take 60–120s. Use `start_period: 120s` in Compose healthcheck. Build with `GGML_SYCL_DEVICE_ARCH` for AOT to skip JIT. |
| Proxy errors during `docker build` | Proxy not forwarded into build steps | Pass `--build-arg http_proxy=... --build-arg https_proxy=...`. See `build-and-env.md`. |

## Diagnosing device visibility

```sh
# See the SYCL-visible device list and index order
docker run --rm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    llama-server-sycl:<tag> --list-devices

# Confirm render group membership inside the container
docker run --rm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --entrypoint id \
    llama-server-sycl:<tag>
```

## Checking logs

```sh
# Tail server logs for load progress and slot init
docker logs -f <container-name>

# Key lines to look for:
#   "model loaded"            — weights on GPU
#   "all slots are idle"      — ready to serve
#   "SYCL0: Intel(R) ..."     — correct device selected
#   "ggml_sycl_init"          — SYCL backend initialised
```

## AOT compilation (eliminate JIT delay)

Build with an arch-specific flag to pre-compile SYCL kernels:

| GPU family | `GGML_SYCL_DEVICE_ARCH` value |
|---|---|
| Arc A-series (Alchemist) | `intel_gpu_dg2_g10` |
| Arc B-series / B70 (Battlemage) | `intel_gpu_bmg_g21` |
| Data Center Max (Ponte Vecchio) | `intel_gpu_pvc` |
| Meteor Lake / Arrow Lake iGPU | `intel_gpu_mtl_h` / `intel_gpu_arl_h` |

Patch the cmake line in `.devops/intel.Dockerfile` or override at
build time. The Dockerfile does not accept `CMAKE_ARGS` as a build
arg natively — edit the `cmake` invocation directly:

```dockerfile
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL_DEVICE_ARCH=intel_gpu_bmg_g21 \
    ...
```
