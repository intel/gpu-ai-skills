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
  `mxfp4`, `fp4`, or `nvfp4` -- but for an already-quantized checkpoint,
  prefer omitting `--quant` so the config's own dtypes are used (see
  "Pre-Quantized Checkpoints"). `fp4` is MXFP4's other spelling -- a
  config-declared `fp4` normalizes to `mxfp4`, a supported XPU path. `nvfp4`
  is a distinct format (per-16 rather than per-32 block scales) with no
  documented XPU kernel path today, so the script flags those verdicts as
  bytes-only
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
python3 plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py \
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
python3 plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py \
    --model /path/to/model/config.json \
    --quant bf16 --ctx 8192 --concurrency 4 \
    --runtime vllm --device-vram-gb 24
```

For a quick common-model table:

```sh
python3 plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py \
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
`quantization_config.quant_method` values from the model config. The KV dtype
default is **per runtime**, because KV dtype is a launch flag and each runtime
defaults differently -- assuming a smaller cache than a runtime's own guidance
supports is how a FITS verdict turns into an OOM:

| `--runtime` | Auto KV dtype | Flag that changes it |
|---|---|---|
| `vllm` | `fp8` for `fp8` weights and AWQ/GPTQ `int8`/`int4`, `bf16` otherwise | `--kv-cache-dtype fp8` |
| `sglang` | always `bf16` | `--kv-cache-dtype fp8_e4m3` |
| `torch` | always `bf16` | none -- the cache is allocated in the model dtype |

The weight dtype alone does not decide it, because **vllm-xpu-run** pairs rows,
not widths. An AutoRound checkpoint prices as `int8`/`int4` from
`quantization_config.bits`, but its own row is `auto` KV, so it keeps `bf16` --
pairing it would halve the cache against the launch that skill documents.

Only vLLM's table pairs quantized weights with an fp8 cache.
**sglang-xpu-run** documents that fp8 weights keep a BF16 cache unless
`fp8_e4m3` is requested, and the torch path has no KV-dtype flag at all. Where
the script does assume `fp8` KV it says what the launch must set; where it does
not, it names that runtime's lever -- or says there is none. `--kv-dtype`
overrides the default on any runtime.

## Pre-Quantized Checkpoints

For a checkpoint that ships quantized, **omit `--quant`** and let the
config drive the dtypes. `quant_method` is not always the whole story:
some MoE models give their experts a separate `expert_dtype`, and the
experts are almost all of the model.

DeepSeek-V4-Flash is the case to remember -- `quant_method: fp8` with
`expert_dtype: fp4`, so 278 B of its 291 B params are 4-bit. Pricing it
all at fp8 doubles the weight estimate and turns an 8-XPU byte fit into
DOES NOT FIT. That `fp4` is MXFP4 -- vLLM dispatches it to `Mxfp4MoEMethod`
on the XPU path, the same `XPUExpertsMxFp4` kernel `quant_method: mxfp4`
reaches -- so the script reports the experts as `mxfp4` and does not flag the
verdict. MXFP4 weights were measured staying packed on a served gpt-oss-20b
(12.87 GiB allocated against 13.14 GiB estimated); DeepSeek-V4-Flash can be deployed on 8 B70 cards with TP=8.

The script handles this automatically whenever the requested dtype still
matches the config -- auto-detected, or typed explicitly as the same
dtype. `--quant fp8` on a checkpoint whose `quant_method` is already fp8
keeps the fp4 experts.

Naming a *different* dtype (`--quant bf16`) is a re-quantization
hypothetical the config no longer describes, so the expert split is
suppressed. The output labels that line `hypothetical:` and quotes what
the as-shipped weights would be, so the larger number cannot be mistaken
for the checkpoint's real size. Always read the `Expert weights:` line
and report which dtype landed on the experts.

For `nvfp4` weights, also state the kernel caveat from
`references/runtime-caveats.md`: fitting the bytes is not the same as the
runtime keeping them 4-bit on Battlemage.

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
- Report auto-detected quantization and KV dtype -- including the
  `Expert weights:` line for MoE models -- so the user knows which
  assumptions drove the verdict.
- Use `--quant` only to ask a "what if I re-quantized this" question. To
  price a checkpoint as shipped, omit it. Naming a dtype that differs from
  the config's suppresses `expert_dtype` and can nearly double the weight
  estimate for models like DeepSeek-V4-Flash; the output says
  `hypothetical:` when that happens, so do not report such a run as the
  model's real footprint.
- Sanity-check a surprising weight estimate against the repo's actual
  size before reporting it. The Hub API gives it without downloading:
  `curl -s "https://huggingface.co/api/models/<id>?blobs=true"` and sum
  the root-level `*.safetensors` sizes. A large gap means a modeling gap
  -- fix the script and add a fixture, do not paper over it in prose.
- Route diffusion fit and tight VLM image-memory questions to empirical
  checks instead of treating this estimate as complete.
- For a VLM, relay the `Vision tower:` line. A quantized build is often not
  uniformly quantized -- the Qwen2-VL and Qwen2.5-VL AWQ checkpoints exclude
  `visual`, so the tower stays `bf16` and gets its own breakdown row. If that
  line says the tower is `not sized`, the verdict covers the LLM backbone only;
  say that rather than reporting the total as the model's footprint.

## References

- Read `references/coverage-and-formulas.md` when checking model class support, formula details, MoE/head-dim behavior, mixed precision, `expert_dtype`, validation numbers, quick-reference verdicts on Arc Pro B70, or why an estimate differs from another calculator.
- Read `references/runtime-caveats.md` when planning dtype/KV choices, weighing fp4 kernel support on Battlemage, handling VLM or diffusion edge cases, or explaining what this skill intentionally does not predict.
