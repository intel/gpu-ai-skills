---
name: llamacpp-xpu-run
description: "Run a GGUF model on an Intel GPU using llama.cpp's SYCL backend (Level Zero) with the official intel.Dockerfile. Covers building the Docker image from source at a pinned tag, launching llama-server with an OpenAI-compatible API, device selection, multi-GPU layer splitting, all recommended runtime env vars, flash-attention, and quantisation selection. Use when the user has a GGUF model and wants fast local inference or an OpenAI-compatible endpoint on Intel GPU without Python/PyTorch. The CUDA analogue is llama.cpp built with `-DGGML_CUDA=ON`. Use **vllm-xpu-run** instead for safetensors models with continuous batching at scale; use **torch-xpu-run** for Hugging Face Transformers direct."
---

# llamacpp-xpu-run

Run GGUF models on Intel GPUs via llama.cpp's SYCL backend.
The official `.devops/intel.Dockerfile` is the canonical build path.

Tested against tag **b9494** on Intel Arc B70 (Battlemage, `level_zero:0`).

**CRITICAL SAFETY RULE: When removing a docker container, always `docker stop` first, then `docker rm`. Never use `docker prune` or any system-wide process-kill command.**

## Performance note

When the Docker image and GGUF model are already present, launch the server immediately. Skip redundant image rebuilds and device checks — the server will fail fast if misconfigured. Always verify with the health endpoint after launch (see Validate section below).

**Use a pinned upstream tag (e.g. `:b9494`), not `:latest`.** `docker run` reuses any locally-tagged image with no upstream check, so a stale `llama-server-sycl:latest` will silently run. If the user requests `:latest`, rebuild with `--no-cache` first, or resolve `latest` to the current upstream tag and build with that pinned name.

## CUDA → XPU cheat sheet

| CUDA | Intel SYCL |
|---|---|
| `-DGGML_CUDA=ON` | `-DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx` |
| `CUDA_VISIBLE_DEVICES=0` | `ONEAPI_DEVICE_SELECTOR="level_zero:0"` |
| `--gpus all` (Docker) | `--device /dev/dri --group-add render` |
| `--n-gpu-layers 99` | same (`-ngl 99`) |

**Before running any docker commands below, confirm with the user that they want to proceed.** The following steps will launch a Docker container and may modify system state.


## Build

Confirm the tag and GGUF model with the user before running — the
build pulls source and creates a Docker image.

```sh
git clone --depth 1 --branch <tag> https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg GGML_SYCL_F16=OFF \
    --target server \
    -t llama-server-sycl:<tag> \
    -f .devops/intel.Dockerfile .
```

Three targets available — `server` (serving), `full` (bench + convert), `light`
(cli + batch). See `references/build-and-env.md` for proxy setup, the
Level Zero deb conflict fix, and full env var reference.

**Ask the user to confirm before proceeding with docker run.**

## Verify device

```sh
docker run --rm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    llama-server-sycl:<tag> --list-devices
# Expect: SYCL0: Intel(R) ... (NNNN MiB free)
```

## Run (single GPU)

**Quick-start workflow:** If the image exists and the model is downloaded, launch immediately with `docker run` below. The server will fail fast if there's a problem — trust its own validation.

Before launching, check if the container name is already in use:
```sh
docker ps -a --filter "name=llama-sycl"
# If it exists and is running: docker stop llama-sycl
# If it exists but is stopped: docker rm llama-sycl  # (without -f)
# Or use a different --name in the docker run command below
```

If port 8000 is already in use, check what's running:
```sh
docker ps --filter "publish=8000"
# Then stop it gracefully: docker stop <name>
```

Launch the server:
```sh
docker run -d --name llama-sycl \
    --rm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ONEAPI_DEVICE_SELECTOR="level_zero:0" \
    -e ZES_ENABLE_SYSMAN=1 \
    -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
    -v "$HOME/models:/models:ro" \
    -p 8000:8080 \
    llama-server-sycl:<tag> \
    -m /models/<model>.gguf \
    -ngl 99 -c 8192 \
    --flash-attn on \
    --host 0.0.0.0 --port 8080
```

Wait for `all slots are idle` in `docker logs`. First run may take
60–120 s for JIT kernel compilation.

## Run (multi-GPU, layer split)

```sh
-e ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1" \
... \
--split-mode layer
```

`--split-mode row` is not supported on SYCL; use `layer` or `none`.

## Validate

```sh
curl -f http://localhost:8000/health   # {"status":"ok"}

REPLY=$(curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2?"}],
         "max_tokens":16,"temperature":0}')
echo "$REPLY" | jq -r '.choices[0].message.content'
# Must contain "4"
```

## References

- `references/build-and-env.md` — proxy build-args, Level Zero conflict fix,
  `--flash-attn` syntax change (b9494+), device index mapping, full env var table.
- `references/quantisation-and-bench.md` — quant ladder, VRAM formula, GGUF
  sources, `llama-bench` usage, key server flags.
- `references/troubleshooting.md` — error table, AOT compilation for B70,
  log inspection, render group diagnosis.

## Cross-references

- **xpu-discover** / **xpu-runtime-preflight** — verify host readiness first.
- **model-can-it-fit** — VRAM sizing (uses HF config.json; for GGUF use the
  formula in `references/quantisation-and-bench.md`).
- **vllm-xpu-run** — safetensors + continuous batching at scale.