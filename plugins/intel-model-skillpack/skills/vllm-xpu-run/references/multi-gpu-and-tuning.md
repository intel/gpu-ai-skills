# Multi-GPU, tuning knobs, and speculative decoding

## Discover XPUs and validate the tensor-parallel degree

Don't hardcode the GPU count. `--tensor-parallel-size N` requires N
Level-Zero-visible XPUs; launching with N greater than what's present
fails at engine init (typically `RuntimeError: device index out of
range`, though the exact message is image-dependent). Discover the count
first and refuse an over-subscribed TP up front:

```sh
# Count Level-Zero-enumerable XPUs. `discovery -j` emits one
# "device_id" per device; this counts what a TP launch can actually
# use, not DRM render nodes (a wedged GPU keeps its render node but
# drops out of Level Zero). Falls back to 1 if xpu-smi is absent.
GPU_COUNT=$(xpu-smi discovery -j 2>/dev/null | grep -c '"device_id"')
[ "${GPU_COUNT:-0}" -gt 0 ] || GPU_COUNT=1

TP=${TP:-$GPU_COUNT}                 # default to all visible XPUs
if [ "$TP" -gt "$GPU_COUNT" ]; then
    echo "ERROR: requested TP=$TP but only $GPU_COUNT XPU(s) visible" >&2
    exit 1
fi
MASK=$(seq -s, 0 $((TP - 1)))        # 0-based affinity mask, e.g. 0,1
echo "Launching TP=$TP on XPUs: $MASK (of $GPU_COUNT visible)"
```

## Tensor parallel

```sh
docker run -d --name vllm-xpu \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro --ipc=host \
    -e ZE_AFFINITY_MASK="$MASK" \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e CCL_ZE_IPC_EXCHANGE=pidfd \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p 8000:8000 \
    intel/vllm:<version>-xpu \
    vllm serve <model-id> \
        --dtype bfloat16 \
        --enforce-eager \
        --block-size=64 \
        --disable-custom-all-reduce \
        --tensor-parallel-size "$TP"
```

- `--tensor-parallel-size N` requires N XPUs in `ZE_AFFINITY_MASK`
  (the discovery block above enforces `N <= GPU_COUNT`).
- `CCL_ZE_IPC_EXCHANGE=pidfd` — oneCCL IPC mechanism that survives
  the Docker PID namespace (default since oneCCL 2021.14).
- `--disable-custom-all-reduce` — vLLM's CUDA-tuned path doesn't
  apply on XPU; flag silences probe noise.

For shared-memory issues on multi-XPU TP (hangs at oneCCL init,
`bus error`), see `xpu-container-run` — usually host `/dev/shm` is
too small.

For one-server-per-GPU instead of TP, launch separate containers
each pinned to a different `ZE_AFFINITY_MASK` and host port.

## Native XCCL collective env vars

The native XCCL backend in `libtorch_xpu.so` reads its own
`TORCH_XCCL_*` family. Worth knowing:

- `TORCH_XCCL_HIGH_PRIORITY=1` — high-priority XPU stream for
  collectives (when comm is on the critical path).
- `TORCH_XCCL_BLOCKING_WAIT=1` — block on completion with stack
  traces (debugging hangs only; not for production).

List all XCCL knobs your image accepts:

```sh
docker exec <name> bash -c \
    'strings $(python -c "import torch; print(torch.__file__)" \
              | xargs dirname)/lib/libtorch_xpu.so \
     | grep "^TORCH_XCCL_" | sort -u'
```

## Tuning knobs (bench, don't copy)

`--block-size`, `--max-num-batched-tokens`, `--max-num-seqs` are
model-, prompt-, and concurrency-dependent.

| Source | Model | `--block-size` | `--max-num-batched-tokens` |
|---|---|---:|---:|
| vLLM blog (2025-11-11), 4× B60 | GPT-OSS-120B | 64 | 8192 |
| vLLM XPU default | (any) | 16 | engine default |

Procedure: start at `--block-size=64` (correctness), bench, sweep
`--max-num-batched-tokens` in `[max-model-len, 4× max-model-len]`.
Lock per model.

## Speculative decoding

Methods: **n-gram** (cheap, no draft model), **EAGLE**, **EAGLE3**
(start here when an EAGLE3 head exists), **MTP** (DeepSeek-MTP,
GLM-OCR-MTP). CLI shape varies by vLLM version — check
`vllm serve --help=all | grep -i speculative`.

```sh
# n-gram
vllm serve <target> ... --speculative-method ngram --num-speculative-tokens 3
# EAGLE3 (newer JSON form)
vllm serve <target> ... --speculative-config '{"method":"eagle3","num_speculative_tokens":3}'
```

Bench with and without spec-decode on the same model and
concurrency. Keep only if TPOT improves without TTFT regression.

## Legacy env-var names (older images only)

On current `intel/vllm:*-xpu` images, prefer the CLI flags.
If pinned to an older image, verify the image's `vllm.envs`
surface and logs before assuming either spelling is accepted.

| Older recipe name | Current preferred path |
|---|---|
| `VLLM_USE_V1` | Drop. V1 is the default. |
| `VLLM_USE_TRITON_XPU_ATTN` | `--attention-backend TRITON_ATTN`. |
| `VLLM_XPU_FLASH_ATTN` | `--attention-backend FLASH_ATTN`. |
| `VLLM_XPU_USE_WFP8A16` | `--quantization fp8`. |
| `VLLM_XPU_USE_W4A8` | `--quantization awq`/`gptq`. |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | Drop. Loader handles weight placement. |
| `TORCH_LLM_ALLREDUCE` | Drop. `torch-ccl` is no longer in the dep chain on PyTorch 2.8+xpu; native XCCL backend is used instead. |
