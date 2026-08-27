# Serving on a remote Intel GPU host over ssh

**Scope:** the serve + verify mechanics for an Intel GPU host you can
already reach over ssh — the local machine → remote-box workflow. For
*planning* a deployment end-to-end (sizing, host selection,
provisioning), see the **xpu-deploy-plan** skill. This reference is the
ssh-wrapping only; it does not turn `vllm-xpu-run` into a
remote-orchestration tool.

The pattern is simple: every command from the single-GPU quickstart —
launch, readiness check, verify, cleanup — runs verbatim, wrapped as
`ssh "$GPU_HOST" '<cmd>'`. The local agent never touches the GPU; the
remote host does. Inside single quotes, `$(getent group render …)` and
any `<placeholder>` are passed through and resolve **on the remote
host**, which is what you want — the render group id and the model
cache are facts about the GPU box, not the local machine.

Set the host once (an ssh alias with key auth keeps secrets off argv):

```sh
GPU_HOST=<user>@<gpu-host>
```

## Launch over ssh (serving from a warm host cache)

```sh
ssh "$GPU_HOST" 'docker run -d --name vllm-xpu \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_HOME=<host-cache> \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v <host-cache>:<host-cache> \
    -p 8000:8000 \
    vllm/vllm-openai-xpu:latest \
    <model-id> \
        --dtype bfloat16 \
        --enforce-eager \
        --block-size=64 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85'
```

`<host-cache>` is the pre-populated cache directory on the GPU host
(verify it first: `ssh "$GPU_HOST" 'ls <host-cache>'`). If the host
should download weights instead, drop the two offline lines
(`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) and add `-e HF_TOKEN`,
but keep `HF_HOME=<host-cache>` so downloaded weights persist to the
mounted cache rather than vanishing on `docker rm` — see the quickstart
in `SKILL.md`.

## Offline-cache gotcha

When serving from a pre-populated host cache, `HF_HOME=<host-cache>`
**alone is not enough** — vLLM may still reach the Hub instead of using
the cache as-is. Add **both** offline flags to force cache use:

```
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
```

This was a real relaunch on a verified run: the first launch set only
`HF_HOME` and had to be torn down and relaunched with the offline pair
before it served. Set all three from the start.

## Verify over ssh (do not skip)

A running server without a successful generation is not a validated
deployment. After `Application startup complete`, send a real request:

```sh
ssh "$GPU_HOST" 'curl -s localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"<model-id>\",
         \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: PONG\"}],
         \"max_tokens\":16}"'
```

Expect generated content back (e.g. `PONG`) and the model id echoed by
`ssh "$GPU_HOST" 'curl -s localhost:8000/v1/models'`. Prefer this
explicit curl over an open-ended `docker logs … | grep` poll loop — a
log poll burns wall-clock and never proves the server generates. Poll
the logs only to confirm readiness, then curl:

```sh
ssh "$GPU_HOST" 'docker logs vllm-xpu 2>&1 | grep "Application startup complete"'
```

## Cleanup

Always tear down a container you created:

```sh
ssh "$GPU_HOST" 'docker stop vllm-xpu && docker rm vllm-xpu'
ssh "$GPU_HOST" 'docker ps -a --filter name=vllm-xpu --format "{{.Names}}"'   # expect empty
```
