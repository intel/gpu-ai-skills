# Server mode and multi-GPU

## Server mode (no entrypoint override)

```sh
docker run --rm -d --name xpu-llm \
    --device /dev/dri \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -p 8000:8000 \
    <image> \
    --model <hf-model-id>
```

`docker logs -f xpu-llm` to follow; OpenAI-compatible endpoint at
`http://localhost:8000/v1`.

## When `--net=host`

Switch to `--net=host` only when (a) following an Intel/vLLM blog
recipe verbatim, or (b) multi-GPU oneCCL is failing to bind to
localhost ports for inter-rank RPC. `-p 8000:8000` + Docker bridge
is fine for single-GPU.

## Multi-GPU: one process per GPU (preferred for inference)

```sh
for id in 0 1; do
  docker run --rm -d --name "xpu-llm-$id" \
    --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host -e ZE_AFFINITY_MASK="$id" \
    -p "$((8000 + id)):8000" \
    <image> --model <hf-model-id>
done
```

## Multi-GPU: one process, multiple GPUs (TP / PP)

```sh
docker run --rm -it \
    --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0,1 \
    -e CCL_ZE_IPC_EXCHANGE=pidfd \
    --entrypoint /bin/bash \
    <image>
```

vLLM: `--tensor-parallel-size 2`. PyTorch: `torchrun --nproc_per_node=2`.

`--ipc=host` mounts the host's `/dev/shm` (size is the host tmpfs
setting, typically 50% of RAM). Per Docker docs, `--shm-size` is
ignored under `--ipc=host`; set it only when running with private
IPC.

**If multi-XPU TP hangs at oneCCL init or `bus error`**, host
`/dev/shm` is too small. Either raise it
(`mount -o remount,size=32G /dev/shm`), or drop `--ipc=host` and
use `--shm-size=32g` (private IPC, explicit size).
