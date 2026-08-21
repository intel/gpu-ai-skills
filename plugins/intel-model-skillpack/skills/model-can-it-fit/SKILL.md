---
name: model-can-it-fit
description: Estimate whether a Hugging Face decoder-only LLM, MoE, or VLM fits in Intel GPU VRAM for a quantization, context length, concurrency, runtime, and tensor-parallel setting. Use for memory-fit or max-model-len planning before launch. Reports weights, KV cache, activations, framework overhead, and first mitigation. Not for diffusion. Memory-only — does NOT predict throughput, tokens/sec, latency, or runtime config; route those to bench/deploy/recommend skills.
---

# model-can-it-fit

Use this for a pre-launch VRAM calculator: whether a Hugging Face model can fit on an Intel GPU at a requested quantization, context length, concurrency, runtime, and tensor-parallel degree. Input: HF model id, quantization, context length, concurrency, target VRAM. Output: a per-component breakdown and a verdict.

The skill runs a CPU-only calculator. It does not need to deploy the model on an Intel GPU, but it MUST check the Intel GPU VRAM memory space and may need Hub access unless the user provides a local `config.json` or `params.json`.

## Use And Route

Use this skill when the user asks:

- whether a model fits on an Intel GPU or XPU
- what max context or concurrency is memory-feasible
- how VRAM splits across weights, KV cache, activations, and runtime
- whether a vLLM/SGLang launch is likely to OOM before trying it

Use another skill instead when the user asks for:

- measured speed, TTFT, TPOT, or tokens/sec: use a benchmark skill
- an exact launch configuration or performance recommendation: use `model-config-recommend`
- diffusion fit: use `torch-xpu-bench` empirically
- live GPU readiness: use `xpu-runtime-preflight` or `xpu-discover`

## Inputs To Collect

Ask for or infer:

- model id or local config path
- target GPU VRAM per device -- if the target is this host, measure it
  (see "Measure VRAM first" below) instead of asking the user
- runtime: `vllm`, `sglang`, or `torch`
- quantization: `bf16`, `fp16`, `fp8`, `int8`, `int4`, `int3`, `int2`,
  or `mxfp4`
- context length and concurrency
- tensor parallel degree if multiple XPUs are planned
- vLLM `--gpu-memory-utilization` value if this is launch planning

For gated Hugging Face repos, use `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN`. For repeatable tests, prefer local config
snapshots.

## Measure VRAM First

`--device-vram-gb` is **required** and has no default. Confirm which card
the host actually has before choosing a value -- never assert VRAM from a
remembered spec sheet:

```sh
xpu-smi discovery -d 0 | grep -i 'Device Name\|Memory Physical Size'
```

Then pass that SKU's whole-GB figure from the table below (`24480.00 MiB`
confirms a 24 GB B60). Bare `xpu-smi discovery` (no `-d`) prints **no
memory field at all** -- it lists device names and PCI ids only. Do not
summarise it as though it reported VRAM.

Also read the free memory, because another workload may already own the
card:

```sh
xpu-smi discovery -d 0 -j | grep -E 'memory_(physical_size|free_size)_byte'
```

If the target hardware is not attached to this host, use the same table
and say in the answer that the card is a stated spec, not one you
confirmed.

| SKU | PCI id | `--device-vram-gb` |
|---|---|---|
| Arc Pro B70 | `0xe223` | 32 |
| Arc Pro B65 | `0xe221` | 32 |
| Arc Pro B60 | `0xe211` | 24 |
| Arc Pro B50 | `0xe220` | 16 |
| Arc B580    | `0xe20b` | 12 |

For **`--tp` greater than 1**, check every device in the intended set and
pass the **smallest** one -- a TP group is limited by its smallest card.
The same applies to `--tp-sweep`: measure enough devices to cover its
largest degree.

## Run

From the skillpack repo root:

```sh
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quant bf16 --ctx 8192 --concurrency 4 \
    --runtime vllm --device-vram-gb 24 \
    --gpu-memory-utilization 0.9 \
    --tp-sweep 1,2,4
```

When running from inside this skill directory, the shorter equivalent is:

```sh
python3 scripts/fit.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quant bf16 --ctx 8192 --concurrency 4 \
    --runtime vllm --device-vram-gb 24
```

For a local config:

```sh
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --model /path/to/model/config.json \
    --quant bf16 --ctx 8192 --concurrency 4 \
    --runtime vllm --device-vram-gb 24
```

For a quick common-model table:

```sh
python3 plugins/intel-model-skillpack/skills/model-can-it-fit/scripts/fit.py \
    --table --runtime vllm --device-vram-gb 32
```

The `24` in the examples is an Arc Pro B60; substitute the SKU you
confirmed. If the script errors with `--device-vram-gb is required`, that
is the guard working -- go check the device, do not pick a plausible
number.

## Interpret Results

Report the verdict and the binding constraint first.

State where the VRAM figure came from: measured on device *N*, or a spec
value for hardware you could not query. Never present a recalled number
as if it were measured.

Free memory can be far below physical VRAM on a shared host. The verdict
is computed against the figure you passed, so if `memory_free_size_byte`
showed the card already occupied, a FITS verdict is not a promise that a
launch right now will succeed -- say so, and point at an idle device with
`ZE_AFFINITY_MASK`.

For a pass, include:

- total VRAM estimate
- usable VRAM if `--gpu-memory-utilization` was supplied
- headroom in GB and percent
- max concurrency or max context memory ceiling if relevant

For a fail, include:

- deficit in GB
- largest component: weights, KV cache, activations, or framework
- the script's first mitigation: lower context, lower concurrency,
  lower KV dtype, quantize weights, or increase TP

Do not present the result as measured GPU memory. It is a config-derived
estimate intended to prevent obvious OOMs before launch.

## Runtime Planning Rules

Use the same `--gpu-memory-utilization` that the runtime launch will use.
For vLLM launch planning, `0.9` is a common starting point; the script's
default `1.0` answers only whether bytes fit in physical VRAM.

If quantization is omitted, the script auto-detects known
`quantization_config.quant_method` values from the model config. For
quantized weights, the script auto-pairs KV dtype with `fp8` unless the
user overrides `--kv-dtype`.

Tensor parallelism divides weights and KV cache per device in this
estimator, and `--device-vram-gb` is per device. The script does not know
which cards the launch will land on, so confirm N XPUs exist with
`xpu-smi discovery`, pass the smallest card's VRAM, and name the intended
devices in the answer via `ZE_AFFINITY_MASK`.


## Gotchas

- Do not calculate VRAM by hand in the final answer. Run `scripts/fit.py`;
  if a new repeated calculation is needed, add script support or a fixture.
- Never state a card's VRAM from memory, and never summarise
  `xpu-smi discovery` as though it reported memory -- the bare form does
  not print any memory field. `xpu-smi discovery -d <id>` does. Assert
  VRAM only from tool output you actually ran.
- Pass the SKU's whole-GB figure (B60 = 24), but treat it as *nameplate*.
  `fit.py` computes with `GB = 1024 ** 3`, so `24` is read as 24 GiB while
  the card holds 24480 MiB -- ~96 MiB optimistic -- and the driver's
  allocatable ceiling is ~5% below physical (22.71 GiB on a B60). Both
  gaps run the same direction. Omitting `--gpu-memory-utilization` makes
  the script print `Usable VRAM: all of it (physical-fit only)` plus a
  `Note:` spelling this out; relay it rather than reporting the verdict
  alone, and re-run with the launch's real value before trusting a tight
  FITS.
- An overstated VRAM figure is worse than an understated one: it yields a
  FITS verdict for a launch that OOMs at engine init. When unsure between
  two values, pass the lower.
- `ZE_AFFINITY_MASK` is not `CUDA_VISIBLE_DEVICES` -- an out-of-range
  index crashes the Level Zero loader rather than being ignored. Confirm
  device ids with `xpu-smi discovery` before recommending a mask.
- This is a config-derived estimate, not measured XPU memory or
  throughput. Use benchmark/profile skills for measured behavior.
- Match `--gpu-memory-utilization` to the planned runtime launch; the
  default `1.0` is only a physical-fit answer.
- Report auto-detected quantization and KV dtype so the user knows which
  assumptions drove the verdict.
- Route diffusion fit and tight VLM image-memory questions to empirical
  checks instead of treating this estimate as complete.

## References

- Read `references/coverage-and-formulas.md` when checking model class support, formula details, MoE/head-dim behavior, mixed precision, quick-reference verdicts on Arc Pro B70, or why an estimate differs from another calculator.
- Read `references/runtime-caveats.md` when planning dtype/KV choices, handling VLM or diffusion edge cases, or explaining what this skill intentionally does not predict.
