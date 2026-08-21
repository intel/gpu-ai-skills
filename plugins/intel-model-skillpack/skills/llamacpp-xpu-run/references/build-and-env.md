# Build and Environment Reference

## Proxy-aware build

Docker `RUN` steps do not inherit the host proxy automatically.
Pass proxy build-args explicitly:

```sh
docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg no_proxy=$no_proxy \
    --build-arg GGML_SYCL_F16=OFF \
    --target server \
    -t llama-server-sycl:<tag> \
    -f .devops/intel.Dockerfile .
```

## Level Zero conflict fix

The `intel.Dockerfile` tries to install `level-zero` and
`level-zero-devel` debs. If the base image (`intel/deep-learning-essentials`)
already ships a newer `libze-intel-gpu1` those installs will fail with
`dpkg: error ... conflicting packages`.

**Diagnosis:** run `dpkg -l | grep -i 'level.zero\|libze'` inside
the base image. If `libze-intel-gpu1` is already present, the deb
install step is unnecessary.

**Fix:** remove the Level Zero deb download + install from the build
stage and rely on the base image's version. Edit lines 13–19 of
`.devops/intel.Dockerfile` to just install build tools:

```dockerfile
RUN apt-get update && \
    apt-get install -y git libssl-dev wget ca-certificates
```

The SYCL runtime only needs `libze1` / `libze-dev` at build time
(header inclusion); the base image supplies both.

## Runtime environment variables

Set all three required vars on every `docker run`:

| Env var | Value | Why |
|---|---|---|
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` (single) or `level_zero:0;level_zero:1` (multi) | Pin device(s). Without this the iGPU at index 0 may be selected instead of the dGPU. |
| `ZES_ENABLE_SYSMAN` | `1` | Enables `ext_intel_free_memory` — required for accurate layer split across GPUs. |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | Allows single allocations > 4 GiB via SYCL/UR. llama.cpp requests this natively since b5377, but set it as a belt-and-braces guard. |

Optional tuning:

| Env var | Default | Notes |
|---|---|---|
| `GGML_SYCL_ENABLE_FLASH_ATTN` | `1` | Disable only to isolate accuracy bugs. |
| `GGML_SYCL_DISABLE_DNN` | `0` | Keep `0` — oneDNN is the fast GEMM path. |
| `GGML_SYCL_ENABLE_LEVEL_ZERO` | `1` | Direct Level Zero allocation; reduces host RAM on dGPUs. |
| `GGML_SYCL_DEBUG` | `0` | Set `1` only for SYCL kernel tracing. |

## Device index mapping

`xpu-smi` device IDs do not map 1:1 to `level_zero:N` indices.
Use `llama-server --list-devices` inside the container to see the
SYCL-visible order:

```sh
docker run --rm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    llama-server-sycl:<tag> --list-devices
```

Example output on a host with 2 B70 dGPUs:
```
Available devices:
  SYCL0: Intel(R) Graphics [0xe223] (31023 MiB, 31023 MiB free)   ← B70 if ONEAPI_DEVICE_SELECTOR filters iGPU
```

Set `ONEAPI_DEVICE_SELECTOR="level_zero:0"` to pin the first
dGPU after iGPUs are filtered, or enumerate all devices without
the selector set first to learn the full index order.

## `--flash-attn` flag syntax (b9494+)

As of b9494 `--flash-attn` requires an explicit value:

```sh
--flash-attn on    # enable (recommended)
--flash-attn off   # disable (debugging only)
--flash-attn auto  # default — let the backend decide
```

The bare `--flash-attn` flag (no value) is rejected with a parse
error in b9494+. Earlier tags accepted it as a boolean toggle.
