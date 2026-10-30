# Stage placement on Intel XPU

vLLM-Omni runs a model as a pipeline of **stages**: for example Qwen2.5-Omni
has thinker (AR) → talker (AR) → code2wav (generation). Each stage is its own
engine process with its own devices and memory budget. Plan placement before
launch. Most first-launch failures on XPU come from the default layout
assuming more cards than the host has.

Upstream reference: <https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/>

## Where the layout comes from

- **Pipeline** (Python, fixed): stage count and topology. Not configurable.
- **Deploy YAML** (`vllm_omni/deploy/<model_type>.yaml`): per-stage
  `devices`, `gpu_memory_utilization`, `tensor_parallel_size`,
  `enforce_eager`, `max_model_len`, …. Auto-loaded when `--deploy-config` is
  omitted.
- **`platforms.xpu.stages`** inside that YAML is overlaid on the CUDA
  defaults automatically when the platform is XPU.

Read the bundled YAML from the image instead of guessing:

```sh
docker run --rm --entrypoint ls vllm-omni-xpu /workspace/vllm-omni/vllm_omni/deploy/
docker run --rm --entrypoint cat vllm-omni-xpu \
    /workspace/vllm-omni/vllm_omni/deploy/qwen2_5_omni.yaml
```

Look at both the top-level `stages:` and the `platforms: xpu:` block.

Precedence, highest first:

1. `--stage-overrides` JSON
2. Global CLI flags (`--gpu-memory-utilization`, `--max-model-len`, …)
3. `platforms.xpu.stages`
4. Overlay YAML via `base_config:`
5. Parser defaults

## Rules that must hold

1. A stage's `devices` count must equal its `tensor_parallel_size × DP ×
   PP`. A mismatch fails at startup and names the stage.
2. `devices` are **logical** indices into the container's
   `ZE_AFFINITY_MASK`. With `-e ZE_AFFINITY_MASK=2,3`, `devices: "1"` is
   physical card 3. vLLM-Omni then sets a per-stage `ZE_AFFINITY_MASK`, so
   every stage logs `device=xpu:0` even when it is on another card. That is
   expected.
3. On a shared card, the stages' `gpu_memory_utilization` values must sum to
   ≤ 1.0. Keep ~0.1 headroom. Each value is a fraction of the card's total
   memory, and each stage checks free memory ≥ its request at init.
4. A top-level `--tensor-parallel-size` is broadcast to **every** stage
   (upstream issue #5003). Set TP per stage in `--stage-overrides` instead.

## Qwen2.5-Omni (3 stages)

Default XPU layout (bundled YAML, `v0.30.0` and `main`). **Needs 2 cards:**

| Stage | Role | devices | gpu_memory_utilization | Measured weights (7B) |
|---|---|---|---|---|
| 0 | thinker | `"0"` | 0.9 | 16.74 GiB |
| 1 | talker | `"1"` | 0.5 | 3.57 GiB |
| 2 | code2wav | `"1"` (colocated with talker) | 0.3 | 1.46 GiB |

All stages run `enforce_eager: true` on XPU. The YAML notes that PyTorch 2.13
XPU Dynamo asserts while compiling the AR stages.

Observed on 2× Arc Pro B70 (32 GB) with `ZE_AFFINITY_MASK=0,1`: card 0 at
25.4 GB, card 1 at 15.7 GB.

### Single card

Colocate all stages on logical 0. `scripts/emit_launch.sh` applies this
automatically when `--devices` lists one card and no overrides or deploy
config were given:

```sh
--stage-overrides '{"0":{"devices":"0","gpu_memory_utilization":0.62,"max_model_len":8192,"max_num_batched_tokens":8192},"1":{"devices":"0","gpu_memory_utilization":0.19,"max_model_len":8192,"max_num_batched_tokens":8192},"2":{"devices":"0","gpu_memory_utilization":0.12}}'
```

How these numbers were found on one Arc Pro B70 (31.9 GiB, ~1 GiB baseline
use), with Qwen2.5-Omni-7B:

| Attempt | Result |
|---|---|
| 0.60 / 0.15 / 0.12, default 32768 batch tokens | stage 0: `Available KV cache memory: -3.87 GiB` |
| 0.66 (+8192 caps) / 0.15 / 0.12 | stage 0 OK (2.0 GiB KV); stage 1: `-3.77 GiB` |
| 0.64 / 0.17 (+8192 caps) / 0.12 | stage 1: 0.28 GiB KV, needs 0.38 GiB for 8192 tokens |
| **0.62 / 0.19 / 0.12, 8192 caps on stages 0–1** | **ready**; KV 10,176 tokens (thinker) and 23,936 (talker); card at 28.0 GB |

The same split still fits on `main` with vLLM 0.32.0 (torch 2.14): ready,
with 0.87 GiB of KV cache on the thinker and 0.93 GiB on the talker, and
the card at 27.3 GB.

Notes:

- The profiling peak, not the weights, is what breaks the default. At 32768
  batch tokens the thinker's multimodal profiling alone needs ~6 GiB above
  its 16.7 GiB of weights. Upstream CI instead sets `skip_mm_profiling:
  true` with 16384-token caps on both AR stages. That also works, but large
  video inputs then risk OOM at request time instead of at startup.
- The thinker margin is thin (10,176 KV tokens for an 8192 context). With a
  desktop session on the card, lower both `max_model_len` values to 4096.
- On a 24 GB card (Arc Pro B60), 0.62 of the card is 14.9 GiB: too small for
  the 7B thinker's 16.7 GiB of weights. Use Qwen2.5-Omni-3B there (untested
  on this skill's bench), or give the thinker its own card.

## Qwen3-Omni-30B-A3B (3 stages)

Default XPU layout needs **5 cards**: thinker `tensor_parallel_size: 4` on
`"0,1,2,3"`, talker and code2wav (0.3) on `"4"`. With fewer cards, write a
deploy overlay that shrinks TP and colocates stages, then size it with
**model-can-it-fit** for the thinker. Treat any such layout as unvalidated
until the smoke test passes.

## Overlay YAML

Prefer an overlay to a full copy, so upstream defaults keep flowing in:

```yaml
# my-omni.yaml — mounted read-only by emit_launch.sh --deploy-config
base_config: /workspace/vllm-omni/vllm_omni/deploy/qwen2_5_omni.yaml
platforms:
  xpu:
    stages:
      - stage_id: 1
        devices: "0"
        gpu_memory_utilization: 0.15
```

`stages:` and `platforms:` are deep-merged by `stage_id`. Other scalars:
the overlay wins.

## Diffusion and TTS models

Single-stage diffusion pipelines (Qwen-Image, Z-Image, FLUX, Wan, …) usually
need no overrides. With `distributed_executor_backend` unset, a single-GPU
diffusion stage uses the in-process `uni` executor. Tongyi-MAI/Z-Image-Turbo
loaded 19.2 GiB on one B70 with `--devices 0` and no overrides.
For a large DiT on a 24 GB card, check the model's recipe on
<https://recipes.vllm.ai> before launch.

### When the pipeline does not fit

All of these are serve flags, so pass them through the launcher after `--`:
`emit_launch.sh --model … -- --vae-use-tiling`. In increasing cost:

1. `--vae-use-tiling`, `--vae-use-slicing` — decode in tiles or slices
   instead of one full-size tensor. First thing to try when the OOM is in
   VAE decode.
2. `--diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}'`
   — hold those components in host memory and stream them per layer;
   `"mode":"module"` moves whole components instead. Setting
   `layer_options` `weight_transfer: allgather` or `resident_layers`
   switches to the distributed layerwise backend, whose code lists CUDA and
   NPU support only and is untested on XPU here. `--enable-layerwise-offload`
   and `--enable-cpu-offload` are compatibility aliases for the two modes,
   and the flags' own help points new work at the JSON form. Offload trades
   host bandwidth for capacity, so expect it to be slower than fitting
   natively.
3. Another card: raise `--devices` and set that stage's TP (rules above).

`model-can-it-fit` will not size these. Its `fit.py` exits on a diffusion
pipeline by design, because peak VRAM is dominated by denoising latents,
which depend on resolution, step count and scheduler. Confirm a fit by
launching with the flags above and reading the loaded footprint.

### Caching denoising steps

`--cache-backend` accepts `tea_cache`, `cache_dit`, `mag_cache`, `sea_cache`
or `step_cache` (default none) and reuses DiT output across similar steps;
`--cache-config` tunes thresholds and `--enable-cache-dit-summary` logs what
was skipped. **Diffusion only** — the flag's help reads "Cache backend for
diffusion models", so it does nothing for omni chat, TTS or text models.

`cache-dit` 1.5.0 is already installed in the image. The first 4 steps are
never cached, so the gain grows with step count. Z-Image-Turbo at 1024x1024
on one B70, same seed and prompt:

| Steps | `--cache-backend cache_dit` speedup |
|---|---|
| 8 | 1.0x |
| 20 | 1.6x |
| 50 | 2.2x |

Images matched the uncached ones at 30–35 dB PSNR, with no visible
difference.

**Send `num_inference_steps` in every request.** cache-dit has to be told
each request's step count. If a request omits it and the pipeline defines no
`default_num_inference_steps` (Z-Image, FLUX and Wan do not), the server logs
"currently requires num_inference_steps to be passed explicitly" and the
cache keeps the last request's setting. On a fresh server that is the
startup run's 2 steps, so the cache does nothing: a 50-step request with the
field omitted got no speedup. `smoke_test.py` omits it, so a smoke-test timing
says nothing about the cache.

TTS models ship their own YAMLs (`qwen3_tts.yaml`, `voxtral_tts.yaml`, …)
with `platforms.xpu` sections. Read them from the image the same way.
