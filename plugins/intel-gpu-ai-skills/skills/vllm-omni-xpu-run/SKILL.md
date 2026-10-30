---
name: vllm-omni-xpu-run
description: Bring up a vLLM-Omni server (`vllm serve --omni`) on Intel Arc / Arc Pro B-series GPUs (XPU) for omni-modality and generative models — Qwen2.5-Omni / Qwen3-Omni chat, text-to-image (Qwen-Image, Z-Image, FLUX), video (Wan) and TTS (Qwen3-TTS). Covers building the image from upstream `docker/Dockerfile.xpu` on the matching `vllm/vllm-openai-xpu` base (no prebuilt XPU image or wheel exists), container flags, per-stage device placement with `--stage-overrides`, single-card layouts, and output-validated smoke tests for chat, image and speech endpoints. Use for any vLLM-Omni request on Intel XPU. Not for text-only LLM serving (use vllm-xpu-run).
---

# vllm-omni-xpu-run

vLLM-Omni extends vLLM to omni-modality models and multi-stage pipelines:
thinker → talker → code2wav for omni chat, DiT for diffusion, AR + codec for
TTS. On Intel XPU, upstream supports it through **Docker only**, validated
on Intel Arc B-Series, Linux only.

Facts to apply (re-verify live — version pins and model support both move
frequently):

- **No prebuilt XPU image or wheel.** Build `docker/Dockerfile.xpu` from a
  vllm-omni checkout. It layers onto `vllm/vllm-openai-xpu:<VLLM_VERSION>`.
  `vllm/vllm-omni` on Docker Hub is a **CUDA** image; never use it on XPU.
- The built image's entrypoint is `vllm serve --omni`. Put serve flags
  (`--model …`) directly after the image name.
- **Don't hardcode a vLLM base version.** Clone vllm-omni and take
  `VLLM_VERSION` from the checkout's own `ARG VLLM_VERSION=` in
  `docker/Dockerfile.xpu`, as upstream's guide does. The default branch's
  pin moves when upstream rebases, and a ref's name doesn't predict it,
  so any written-down number goes stale. Build `-t vllm-omni-xpu`, the
  name upstream's guide uses.
- `VLLM_VERSION` is a **vLLM core-engine** version (the base image
  `vllm/vllm-openai-xpu:<VLLM_VERSION>`), not a vLLM-Omni release
  number — the two projects have separate counters, so don't compare
  them. Details: `references/build-image.md`. Refs older than the
  layered-Dockerfile style use a self-contained Dockerfile this skill
  does not cover.
- The bundled deploy YAML auto-applies `platforms.xpu` overrides. The
  default Qwen2.5-Omni XPU layout needs **2 cards**; Qwen3-Omni-30B needs **5**.

Upstream install page (Intel XPU tab):
<https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/>

## Supported models: check live

When the user asks whether a model runs on XPU, fetch at request time:

<https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/>

Report rows with ✅ in the **Intel GPU** column, plus the source URL and
retrieval date. For the exact image version, read
`https://raw.githubusercontent.com/vllm-project/vllm-omni/<tag>/docs/models/supported_models.md`.
Do not answer from memory. A ✅ means the pipeline runs on XPU; still
validate the output content (step 6): Qwen2.5-Omni returns valid but
garbled speech unless the request carries its system prompt (see Known
issues). If the page cannot be fetched, say so and offer to retry.

## CUDA → XPU cheat sheet

| CUDA (upstream docs) | Intel XPU |
|---|---|
| `vllm/vllm-omni:v0.30.0` (prebuilt) | build `docker/Dockerfile.xpu` → `vllm-omni-xpu` |
| `--runtime nvidia --gpus 2` | `--device /dev/dri` + `-v /dev/dri/by-path:/dev/dri/by-path:ro` + render group + `--ipc=host` |
| `… vllm/vllm-omni:v0.30.0 vllm serve <model> --omni` | `… vllm-omni-xpu --model <model>` (entrypoint already has it) |
| `CUDA_VISIBLE_DEVICES=0,1` | `-e ZE_AFFINITY_MASK=0,1` |
| stage `devices: "1"` | same; logical index into `ZE_AFFINITY_MASK` |

## Workflow

### 1. Preflight (read-only)

```sh
xpu-smi discovery                                 # card count, model, VRAM
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'vllm-omni-xpu|vllm-openai-xpu'
docker ps -a --filter "name=vllm-omni-xpu" --format '{{.Names}} {{.Status}}'
ss -ltn | grep -E ':8091\b' || echo "port 8091 free"
df -h /var/lib/docker                             # keep ~30 GB free for base + image
```

Inspect the output before going on. If `xpu-smi discovery` shows no
device, stop and route to **xpu-runtime-preflight** / **xpu-system-setup**.
Never stop or remove containers you did not start; pick another
`--name` / `--port` instead.

### 2. Get the image

If a `vllm-omni-xpu` image already exists, check what it contains:

```sh
docker run --rm --entrypoint python3 vllm-omni-xpu -c \
    "import importlib.metadata as m; print('vllm', m.version('vllm')); print('vllm-omni', m.version('vllm-omni'))"
```

Otherwise build it, which is upstream's Intel XPU build verbatim.
**Confirm with the user first**: it pulls a multi-GB compressed base and
produces a ~20–30 GB image.

```sh
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
export VLLM_VERSION=$(sed -n 's/^ARG VLLM_VERSION=//p' docker/Dockerfile.xpu | head -1)
echo "vLLM base: vllm/vllm-openai-xpu:$VLLM_VERSION"   # sanity-check before pulling
docker pull "vllm/vllm-openai-xpu:$VLLM_VERSION"       # fails fast if unpublished
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g \
    --build-arg VLLM_VERSION="$VLLM_VERSION" \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy .
```

Read `VLLM_VERSION` from the checkout as above; never hardcode it. Then
re-run the version check at the top of this step to confirm what you
built. If `docker pull` 404s, the base tag isn't published yet — build it
from vLLM source first (`references/build-image.md`), which also covers
sizes and timings, the proxy args, and the Dockerfile walkthrough. If the
build itself fails, check whether the base fails the same way before
blaming the host; the default branch can be broken for XPU between
releases, and the fallback is the newest release tag (same reference).

### 3. Plan stage placement

Compare the card count from step 1 with the model's default layout. Read
the bundled YAML from the image rather than guessing:

```sh
docker run --rm --entrypoint cat vllm-omni-xpu \
    /workspace/vllm-omni/vllm_omni/deploy/qwen2_5_omni.yaml   # see `platforms: xpu:`
```

| Model | Default XPU layout | Fits on one 32 GB card? |
|---|---|---|
| Qwen2.5-Omni-7B | thinker on 0 (0.9); talker 0.5 + code2wav 0.3 on 1 | Yes, with the single-card overrides below |
| Qwen3-Omni-30B-A3B | thinker TP=4 on 0–3; talker + code2wav on 4 | No; needs a custom deploy YAML |
| Z-Image-Turbo (diffusion) | single stage | Yes (19.2 GiB loaded) |
| Qwen3-TTS-12Hz-1.7B | 2 stages on 0 (0.3 + 0.3) | Yes |

Rules (details in `references/deploy-configs.md`):

1. `devices` per stage = TP × DP × PP. Logical indices into `ZE_AFFINITY_MASK`.
2. Fractions of stages sharing a card sum to ≤ 1.0. Keep ≤ 0.93 so
   driver and desktop memory still fit.
3. Use `--stage-overrides` for per-stage TP. A top-level
   `--tensor-parallel-size` is broadcast to every stage.

For Qwen2.5-Omni on **one card**, `emit_launch.sh --devices 0` adds this
override automatically. It was validated with the 7B on an Arc Pro B70:

```text
{"0":{"devices":"0","gpu_memory_utilization":0.62,"max_model_len":8192,"max_num_batched_tokens":8192},
 "1":{"devices":"0","gpu_memory_utilization":0.19,"max_model_len":8192,"max_num_batched_tokens":8192},
 "2":{"devices":"0","gpu_memory_utilization":0.12}}
```

The 8192-token caps are load-bearing. At the default 32768, both AR stages
report negative KV memory. On a 24 GB B60 the 7B thinker does not fit in
0.62 of the card; use the 3B or a second card.

### 4. Launch

Confirm the image tag, model, cards and port with the user before running
`docker run`. Container launches bind host GPUs, and first runs download
multi-GB weights. Generate the command with `scripts/emit_launch.sh`,
which keeps flags, proxy passthrough and stage presets in one place:

Upstream's own quickstart serves `Qwen/Qwen2.5-Omni-3B`. This skill's
examples use the 7B instead — that's the one actually run end-to-end on
hardware, including the speech round trip. 3B is
expected to behave the same way.

```sh
bash scripts/emit_launch.sh --model Qwen/Qwen2.5-Omni-7B --devices 0,1 --image vllm-omni-xpu --no-remote-code
```

It prints, for example:

```sh
docker run -d --name vllm-omni-xpu \
    --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    -p 8091:8091 \
    -e ZE_AFFINITY_MASK=0,1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e HF_TOKEN \
    -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
    -e http_proxy -e https_proxy -e no_proxy \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    vllm-omni-xpu \
    --model Qwen/Qwen2.5-Omni-7B --port 8091 \
    --stage-overrides '{"0":{"trust_remote_code":false},"1":{"trust_remote_code":false},"2":{"trust_remote_code":false}}'
```

Run the printed `docker run` command itself, verbatim, as its own shell
command. Do not save it to a file and run `bash launch.sh`. Seen inline,
the exact flags stay visible to the user and in the session log, and so
does the container name that tooling uses to track and clean up what was
started. The build in step 2 is different: it runs as a script on purpose,
so its guards can stop a bad build.

Useful options: `--stage-overrides JSON` (merged over a model's single-card
preset, so naming one stage does not unplace the others), `--deploy-config
FILE` (mounted read-only), `--first-launch` (longer stage init timeouts
while weights download), and `-- <extra serve args>`.

**Remote code is decided per launch, and the CLI flag alone does not
decide it.** `--trust-remote-code` lets the server run arbitrary Python
from the model repo. vLLM-Omni's bundled deploy YAMLs set
`trust_remote_code: true` on every stage for most models (79 of 141 on `main`,
including Qwen2.5-Omni, Qwen3-Omni, Qwen3-TTS and Voxtral TTS), and
`vllm serve --omni` treats an *absent* `--trust-remote-code` as "not
specified", which leaves the YAML's value in force. Verified: a
Qwen2.5-Omni-7B launch with no flag logs `trust_remote_code=True` on all
three stages. So `emit_launch.sh` refuses to print a command for **any**
model until you pass one of:

- `--no-remote-code`: disables it on every stage through
  `--stage-overrides`, the only override that beats the YAML (a top-level
  value loses to per-stage ones). Use this unless the checkpoint needs
  remote code. Qwen2.5-Omni-7B and Qwen3-TTS-12Hz-1.7B-CustomVoice both
  serve with it off on `v0.30.0` and `main`; the TTS speech transcribes
  back word for word, so its bundled `true` is not needed. Works
  for the layouts the script knows (Qwen2.5-Omni, Qwen3-Omni, Qwen3-TTS,
  Voxtral TTS); for anything else it has no stage ids to set the value on,
  so it refuses until you add `--stage-overrides '{"0":{},"1":{}}'` naming
  that pipeline's stages — a flag it cannot apply would disable nothing.
- `--trust-remote-code`: keeps it on. Ask the user first, and state the
  trade-off in your answer. Qwen3-TTS's upstream launch uses it, though
  it is not required (above).

`--deploy-config FILE` does not replace that choice. A copy of a bundled
YAML keeps its `trust_remote_code: true`, and an overlay inherits it
through `base_config`, so pass one of the two flags with it.

The gate covers every model because the four families above are only 4 of
those 79 YAMLs — `lance`, `bagel` and the `audex` family enable it too.
Confirm the resolved value after launch either way (step 5).

Do not use `--rm` on first launches, so logs survive a crashed init.

### 5. Wait for readiness

Poll `/health` in a bounded loop instead of a blocking `docker logs -f`.
Each request is capped, and so is the whole wait: a stage can stall
while the container keeps running, or `/health` can accept a connection
and never answer. The default matches `--init-timeout 2400`; raise
`READY_TIMEOUT` for a cold weight download.

```sh
deadline=$(( $(date +%s) + ${READY_TIMEOUT:-2400} ))
until curl -sf --max-time 5 --noproxy '*' http://127.0.0.1:8091/health >/dev/null; do
    [ "$(docker inspect -f '{{.State.Status}}' vllm-omni-xpu)" = running ] || { echo "NOT READY: container exited"; break; }
    [ "$(date +%s)" -lt "$deadline" ] || { echo "NOT READY: no healthy /health after ${READY_TIMEOUT:-2400}s"; break; }
    sleep 10
done
docker logs vllm-omni-xpu 2>&1 | grep -E "Orchestrator ready|Application startup complete|StageEngineCoreProc_stage[0-9].*ERROR" | tail -5
docker logs vllm-omni-xpu 2>&1 | grep -o 'trust_remote_code=[A-Za-z]*' | sort -u   # must match the step-4 decision
```

On `NOT READY`, report it with those log lines and find the **first**
stage error; see `references/troubleshooting.md`.

The `trust_remote_code=` line is each stage's engine logging its resolved
config. That, not the command line, is what the server runs with. Report
the value to the user; a `True` after `--no-remote-code` means an override
did not reach that stage.

### 6. Validate output content, not just HTTP 200

```sh
python3 scripts/smoke_test.py --mode text --mode audio --roundtrip --expect paris   # omni chat
python3 scripts/smoke_test.py --mode image                                          # text-to-image
python3 scripts/smoke_test.py --mode speech --voice vivian --language English       # Qwen3-TTS
```

The script decodes every payload: text non-empty, WAV parseable with a
plausible duration, PNG with real dimensions. With `--roundtrip` it also
has the omni server transcribe its own speech. For omni chat it sends the
Qwen-Omni system prompt the talker needs (`--system` overrides it; `''`
reproduces the garbling in Known issues). Report the printed text,
durations and `heard:` transcript to the user. Open saved images to check
they match the prompt. A server that returns 200 without passing these
checks is **not** a validated deployment. Endpoint details and curl
equivalents: `references/smoke-tests.md`.

### 7. Verify device placement

```sh
xpu-smi dump -d 0 -m 18 -n 1     # repeat per card listed in ZE_AFFINITY_MASK
```

Every card that hosts a stage should hold multi-GB memory. Observed, 2-card
Qwen2.5-Omni-7B: card 0 25.4 GB, card 1 15.7 GB. Each stage logs
`device=xpu:0` because vLLM-Omni narrows `ZE_AFFINITY_MASK` per stage.

### 8. Stop

```sh
docker stop vllm-omni-xpu && docker rm vllm-omni-xpu
```

Only for containers this workflow started. Never force-remove.

## Flag rationales

| Flag | Why |
|---|---|
| `--device /dev/dri` + render group | Intel has no `--gpus`. Upstream XPU CI runs this way, without `--privileged` |
| `-v /dev/dri/by-path:/dev/dri/by-path:ro` | oneCCL scans by-path even on one card; missing it aborts init with `opendir failed` |
| `--ipc=host` | Inter-stage shared memory and oneCCL IPC. Also makes `--shm-size` (upstream example) moot |
| `-p 8091:8091` instead of `--net=host` | Stages talk over in-container localhost; publish only the API. `--host-network` for the stage-based multi-process CLI |
| `ZE_AFFINITY_MASK` | Picks physical cards; stage `devices` index into it |
| no `--privileged` | Least privilege. `emit_launch.sh --privileged` is a fallback only after the user accepts it |
| `--no-remote-code` or `--trust-remote-code` | Required for models whose bundled deploy YAML enables remote code, because omitting the flag keeps it on. Per-stage overrides are the opt-out; see step 4 |
| `--model` after the image | Entrypoint is `vllm serve --omni`; a second `vllm serve` exits 2 with `vllm: error: unrecognized arguments: vllm serve` |

## Known issues (XPU)

- **Qwen2.5-Omni speech is garbled without the model's system prompt.**
  A chat request with no system message gets correct text and a valid
  WAV whose transcript is unrelated ("我喝。" for "The capital of France
  is Paris."): the chat template's default system prompt leaves the
  talker's text conditioning out of distribution. Send the model card's
  prompt ("You are Qwen, a virtual human developed by the Qwen Team,
  Alibaba Group, capable of perceiving auditory and visual inputs, as well
  as generating text and speech."); `smoke_test.py` does by default, and
  the speech then transcribes back word for word on XPU. Not
  XPU-specific: upstream's e2e tests all send it. Evidence:
  `references/troubleshooting.md`.
- Qwen2.5-Omni AR stages must stay `enforce_eager: true` on torch 2.13 XPU
  (Dynamo assertion, per the bundled YAML).
- vLLM-Omni forces `VLLM_XPU_USE_SAMPLER_KERNEL=0` on XPU. Do not override it.

## Common errors

- `Available KV cache memory: -X GiB` → stage budget below weights plus
  profiling peak. Raise its fraction or cap `max_num_batched_tokens`.
- `… max seq len (8192), (0.38 GiB KV cache is needed …` → move ~0.02 of a
  card to that stage or lower its `max_model_len`.
- A stage wants logical device 1 on a 1-card host → apply single-card
  overrides.
- A diffusion or video pipeline OOMs on the cards you have → VAE tiling,
  then CPU offload via `--diffusion-offload-config`; see
  `references/deploy-configs.md`. `model-can-it-fit` cannot size these.
- `torch.xpu.device_count() == 0` in the container → `--device /dev/dri` or
  render group missing.
- First launch times out → weights still downloading; use `--first-launch`.

Full table: `references/troubleshooting.md`.

## What this skill does NOT cover

- Text-only LLM serving with upstream vLLM-XPU → **vllm-xpu-run**.
- Host driver or Docker readiness → **xpu-runtime-preflight**,
  **xpu-system-setup**.
- VRAM sizing for an LLM thinker → **model-can-it-fit** (decoder LLMs only;
  it refuses diffusion).
- Throughput or latency measurement. **vllm-xpu-bench** targets text
  endpoints of a vLLM server. Benchmarking omni audio, image or video
  output is not covered by this pack.
- Non-Docker installs: upstream ships no XPU wheel or pip path.
- Patching the vLLM-Omni image. Report upstream defects with the
  evidence instead.
- `intel/llm-scaler-vllm`, ipex, and third-party "omni XPU" images.

## References

- `references/build-image.md`: Dockerfile walkthrough, version pairing, base fallback, proxies
- `references/deploy-configs.md`: stage placement rules, layouts, single-card derivation, overlays, diffusion offload and step caching
- `references/smoke-tests.md`: endpoints per modality, curl equivalents, observed outputs
- `references/troubleshooting.md`: known issues with evidence, error table, env vars, privileges
- `scripts/emit_launch.sh`: prints the `docker run`, never executes it
- `scripts/smoke_test.py`: stdlib-only output validator
- Upstream: <https://github.com/vllm-project/vllm-omni>, base images <https://hub.docker.com/r/vllm/vllm-openai-xpu>
