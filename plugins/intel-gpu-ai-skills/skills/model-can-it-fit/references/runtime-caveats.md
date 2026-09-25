# Runtime Caveats

Use this reference when choosing dtype/KV options, explaining VLM or
diffusion behavior, or deciding whether the user needs a benchmark
instead of a fit estimate.

## Quick Arc Pro B70 Reference

Regenerate this table for the current script and model set:

```sh
python3 plugins/intel-gpu-ai-skills/skills/model-can-it-fit/scripts/fit.py \
    --table --runtime vllm --device-vram-gb 32
```

`--device-vram-gb 32` is deliberate here: this table is pinned to the B70
so it regenerates identically from any host. Do not swap in the VRAM of
whatever card the regenerating machine happens to have -- that silently
retargets the table. For the user's actual hardware, run the script
per-model with a measured value instead (see the skill's "Measure VRAM
First").

Typical Arc Pro B70 (32 GB) planning outcomes:

| Model | bf16 / 8K / c=1 | bf16 / 4K / c=4 | int4 / 32K / c=4 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | fits | fits | fits |
| Qwen2.5-7B-Instruct | fits | fits | fits |
| Llama-3.1-8B-Instruct | fits | fits | fits |
| Qwen2.5-14B-Instruct | fits or tight | tight | fits |
| Qwen2.5-32B-Instruct | OOM from weights | OOM from weights | fits |
| GPT-OSS-120B | OOM from weights | OOM from weights | needs multi-XPU TP |

Treat this table as orientation only. Use the script for the user's
actual context, concurrency, runtime, and memory-utilization target.

## Dtype And KV Choices

`bf16` is the safest default on XPU.

For vLLM launch planning, set `--gpu-memory-utilization` to the same
value as the planned `vllm serve` command. The script default of `1.0`
means physical-fit only.

For `fp8`, pair with `--kv-cache-dtype fp8` when serving. The vLLM launch may
also need an attention backend that supports fp8 KV on XPU.

For `int4` AWQ or GPTQ models, the script auto-pairs KV with `fp8`, which is
the pairing vllm-xpu-run documents for that path -- launch with
`--kv-cache-dtype fp8` and `--attention-backend TRITON_ATTN` or the cache
costs 2x the estimate. Validate live runtime logs for the expected int4
kernel path; if the runtime falls back to a wider activation path, activation
memory and throughput may differ from the estimate.

`mxfp4`, `int3`, `int2` and every AutoRound checkpoint are *not* auto-paired:
vllm-xpu-run documents `auto` KV for them, so the script keeps `bf16` KV and
names `--kv-dtype fp8` as an option. Pass it only if the launch will really set
`--kv-cache-dtype fp8`.

AutoRound is excluded by its `quant_method`, not by its width. A 4-bit
AutoRound checkpoint prices its weights as `int4` -- the same width the AWQ /
GPTQ row pairs with `fp8` -- but vLLM detects `quant_method=auto-round` and
routes it through the gptq/awq loader with `auto` KV, so the cache is 2x what
that pairing would suggest. Omit `--quantization` for those checkpoints, as
vllm-xpu-run says, and read the engine's own `GPU KV cache size` line before
trusting a tight verdict.

`int3` and `int2` are AutoRound-oriented planning modes. Validate output
quality before trusting a deployment based on those sizes.

MXFP4 is a supported XPU path, under both spellings. `quant_method: mxfp4`
and DeepSeek-V4's `expert_dtype: fp4` go through different quant configs and
MoE methods, but on XPU both select `Mxfp4MoeBackend.XPU` and the same
`XPUExpertsMxFp4` kernel, which consumes the checkpoint layout directly
instead of transforming it. Battlemage has no native fp4 *datapath*; the
kernel dequantizes on the fly rather than materialising wider weights.

Two different strengths of evidence, worth keeping apart:

- **Measured** on the `mxfp4` spelling: a served gpt-oss-20b allocated
  12.87 GiB against a 13.14 GiB estimate, so those weights stayed packed.
- **Inspected only** for the `expert_dtype: fp4` spelling: the dispatch to
  `Mxfp4MoEMethod` -> `Mxfp4MoeBackend.XPU` was read in
  `vllm/vllm-openai-xpu:latest` (vLLM 0.29.0), but DeepSeek-V4-Flash was not
  served here -- 155 GiB does not fit 2x32 GiB. Read the engine's own weight
  line before trusting a tight verdict on that checkpoint.

**NVFP4** is the case still to qualify. It is not on vllm-xpu-run's
quantization table and has no measured run here, so whether those bytes stay
4-bit on device is unknown. If a runtime upcasts to fp8 or bf16 at load, real
weight memory is 2x or 4x the estimate and a tight multi-XPU fit fails at
engine init. Say so on any nvfp4 verdict.

Architecture registration is the cheap precondition, and it is only that --
a registered architecture can still upcast the weights or fail during
engine init:

```sh
python3 -c "from vllm.model_executor.models.registry import ModelRegistry as M; \
print([a for a in M.get_supported_archs() if 'eepseek' in a])"
```

What settles it is a load, because only the loader reports what the weights
cost on device. Start the model through **vllm-xpu-run** at a small
`--max-model-len`, then read the engine's own weight figure out of the
startup log:

```sh
# from the host, against the container vllm-xpu-run started (-d --name)
docker logs <name> 2>&1 | grep -iE "model weights|loading model|KV cache size"
```

Compare that figure against the script's `Weights` line at the same `--tp`.
Within a few percent means the weights stayed 4-bit. A 2x or 4x gap means the
runtime upcast them, and every verdict derived from the nvfp4 estimate is void.
No engine-init line at all means the load failed, which is the same answer
arrived at the hard way.

A FITS verdict for an `nvfp4` model is a claim about bytes, not about kernels,
until that log line agrees with it. For MXFP4 that log line has already been
read: 12.87 GiB allocated against 13.14 GiB estimated on gpt-oss-20b.

## VLM Caveats

The script includes the vision tower weights when `vision_config`,
vision-language architecture names, or VLM model types are present.

The tower is priced at whatever `modules_to_not_convert` leaves it at, which on
the Qwen2-VL and Qwen2.5-VL AWQ builds (`["visual"]`) is `bf16` while the
backbone is 4-bit. A quantized VLM is therefore not uniformly quantized, and
the `Vision tower` row in the breakdown says which rate it got.

It does not fully model:

- image-token KV growth from resolution and image count
- vision encoder activation peaks
- processor-side memory spikes
- a `vision_config` stripped of its depth field, which cannot be sized at all.
  `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` ships one: the verdict there covers the LLM
  backbone only, and the report says so instead of implying the tower is free.
  Served, that build allocated 6.59 GiB against the 5.37 GiB estimated

For tight VLM fits, leave extra headroom. As a planning rule, add about
1-3 GiB per concurrent image-bearing request, then verify with
`torch-xpu-bench` at the user's image resolution.

That rule is what covers the non-weight gap, and the gap is measured. Serving
four Qwen-VL builds on an Arc Pro B70 at `--max-model-len 4096`, vLLM's own
startup line reported 1.51-1.70 GiB of non-torch memory plus 1.94-1.95 GiB of
peak activation -- 3.46-3.65 GiB of non-weight overhead, against the 2.55 GiB
this script carries at that setting (2.00 framework + 0.55 activations). The
extra ~0.9 GiB is the vision encoder's profiled peak, which the engine measures
at the model's maximum image size rather than at the image a request actually
sends. Weights are the part validated to ~2%; the overhead around them is a
planning allowance on a VLM, not an estimate.

## Diffusion Caveats

The script refuses diffusion pipelines when the root has
`model_index.json` but no top-level LLM config. Diffusion peak memory is
dominated by latent resolution, steps, scheduler, and pipeline component
activation peaks, so a config-only estimate is not reliable.

For a floor estimate, point the script at a component config such as the
UNet or transformer subdirectory. For the real answer, use
`torch-xpu-bench` with one run and read peak XPU memory.

## What This Skill Does Not Predict

- measured tokens/sec, TTFT, TPOT, or ITL
- runtime graph-capture or `torch.compile` buffers
- prefix-cache buffers when prefix caching is enabled
- pipeline-parallel sharding
- diffusion peak memory
- correctness or output quality after aggressive quantization

Use benchmark/profile skills for measured runtime behavior.

## External References

- apxml VRAM calculator, useful CUDA analogue: <https://apxml.com/tools/vram-calculator>
- vLLM XPU kernels: <https://github.com/vllm-project/vllm-xpu-kernels>
- Intel AutoRound: <https://github.com/intel/auto-round>
