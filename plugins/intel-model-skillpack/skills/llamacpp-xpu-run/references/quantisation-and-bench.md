# Quantisation and Benchmarking Reference

## Quantisation ladder

| Quant | Bits/weight | Quality | 7B model size |
|---|---|---|---|
| Q8_0 | 8.5 | Excellent | ~7.7 GiB |
| Q6_K | 6.6 | Very good | ~5.5 GiB |
| Q5_K_M | 5.7 | Good | ~4.8 GiB |
| Q4_K_M | 4.8 | Good | ~4.1 GiB |
| Q4_0 | 4.5 | Acceptable | ~3.8 GiB |
| IQ4_XS | 4.3 | Acceptable | ~3.6 GiB |

Prefer `_K_M` variants (k-quants) over plain Q4_0/Q5_0 — better
quality at the same file size. Use the largest quant that fits in
GPU VRAM at your target context length.

## VRAM estimation

```
VRAM ≈ model_file_size
      + (ctx × n_layers × d_head × n_kv_heads × 2 bytes × 2)
        / (flash_attn ? 2 : 1)
```

Conservative rule of thumb: `model_file_size × 1.2 + 2 GiB` for
8K context. Use **model-can-it-fit** for safetensors models; for
GGUF use the formula above or measure empirically.

## Where to get GGUF files

- Hugging Face: search `<model-name> GGUF` — curated repos by
  bartowski, mradermacher, gguf-org.
- Convert from safetensors yourself with `convert_hf_to_gguf.py`
  (only available in the `full` image target).
- Verify before serving: `llama-cli --model <file> -ngl 0 -n 1`

## Benchmarking with llama-bench

`llama-bench` is only in the `full` image target. Build it first:

```sh
git clone --depth 1 --branch <tag> https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg GGML_SYCL_F16=OFF \
    --target full \
    -t llama-full-sycl:<tag> \
    -f .devops/intel.Dockerfile .
```

Then run:

```sh
docker run --rm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    -e ONEAPI_DEVICE_SELECTOR="level_zero:0" \
    -e ZES_ENABLE_SYSMAN=1 \
    -e UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 \
    -v "$HOME/models:/models:ro" \
    --entrypoint /app/llama-bench \
    llama-full-sycl:<tag> \
    -m /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -fa 1 \
    -p 512 -n 128
```

`-fa 1` enables flash-attention in llama-bench (equivalent to
`--flash-attn on` in llama-server). Reports pp (prompt processing)
and tg (token generation) in tokens/sec.

## Key server flags

| Flag | Recommended | Notes |
|---|---|---|
| `-ngl 99` | Always | Offload all layers to GPU. |
| `-c <ctx>` | 4096–32768 | KV cache allocated up-front; start at 8192, raise until OOM, back off. |
| `--flash-attn on` | Always (b5377+) | ~2× KV memory reduction. Requires explicit value since b9494. |
| `--mmap` | Always | Memory-map GGUF for fast cold start. |
| `--cont-batching` | Default ON | Continuous batching for concurrent requests. |
| `--parallel N` | 4–8 | Concurrent slots; each reserves KV. |
| `--metrics` | Production | Prometheus on `/metrics`. |
| `--split-mode layer` | Multi-GPU | Layer-split across devices. `row` not supported on SYCL. |
