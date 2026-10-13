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
- vision encoder activation peaks under `sglang` and `torch` (under `vllm`
  the encoder's profiling pass is charged; see below)
- processor-side memory spikes
- a `vision_config` stripped of its depth field, which cannot be sized at all.
  `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` ships one: the verdict there covers the LLM
  backbone only, and the report says so instead of implying the tower is free.
  Served, that build allocated 6.59 GiB against the 5.37 GiB estimated

For tight VLM fits, leave extra headroom. As a planning rule, add about
1-3 GiB per concurrent image-bearing request, then verify with
`torch-xpu-bench` at the user's image resolution.

Under `--runtime vllm` the largest part of that gap is now charged. vLLM
profiles the vision encoder at the model's maximum image size at startup,
whatever image a request later sends, and keeps the reservation: 3.8-4.1 GiB
of non-weight memory on Qwen2.5-VL-7B, Qwen3-VL-8B and Qwen3.5-4B on an Arc
Pro B70, 2.0 GiB on Gemma-3-4B. The estimate sizes that pass from the
tower's profiled patch count and MLP width (coverage-and-formulas.md) and
lands 0.3-0.7 GiB above it. What remains a planning allowance is the
per-request image traffic beyond that profiled image.

## Diffusion Caveats

The script refuses diffusion pipelines when the root has
`model_index.json` but no top-level LLM config. Diffusion peak memory is
dominated by latent resolution, steps, scheduler, and pipeline component
activation peaks, so a config-only estimate is not reliable.

For a floor estimate, point the script at a component config such as the
UNet or transformer subdirectory. For the real answer, use
`torch-xpu-bench` with one run and read peak XPU memory.

## Hybrid And Sliding-Window Caveats

For a model with recurrent (state-space) layers, the state pool is charged
per running request and does not shrink with context. Lowering `--ctx`
therefore does little for a hybrid model whose state pool dominates;
lowering `--concurrency` is the lever that works.

Under vLLM a hybrid's state costs more than one copy of it. The state
shares the attention block pool, so each state page is padded to one
attention block (vLLM grows the block until it holds a page -- 576 tokens
on Qwen3.5-4B) and default prefix caching keeps two pages per recurrent
layer per request. The report prints both figures: the bare state and what
vLLM reserves. An attention-free model (pure Mamba) has prefix caching off
and reserves one page.

The state pool is sensitive to `--tp`, and vLLM refuses some degrees
outright rather than padding: gated delta net, KDA and Mamba2 heads must
divide the TP degree, and Mamba2 groups must too unless there is a single
group, which is then replicated to every rank (the assertion in
`MambaMixer2.__init__`; the shape calculator's group padding only ever runs
for the single-group case). The TP check reports those launches as
unlaunchable instead of estimating them.

For a sliding-window model, the per-layer cache is bounded, so the KV
curve flattens once the context passes the window. Raising `--ctx` past
that point changes almost nothing, and the reported max context is found
by search rather than by division.

Switching `--runtime` changes both of these, because the engines size
their pools differently. SGLang provisions its window pool as a fraction
of the full-attention pool rather than from the window, and only for the
architectures it runs as hybrid SWA (gpt-oss, Llama-4, Gemma-4, ...);
Mistral, Qwen2 and Gemma-2/3 get a full pool there. It forces the SSM half
of recurrent state to fp32, and it fixes the split between its state pool
and its KV pool (0.9 : 1 by default), so whichever pool runs short first
sets the memory both need -- the report charges the difference as
`Pool split`. The same model can fit under vLLM and not under SGLang.

## What This Skill Does Not Predict

- measured tokens/sec, TTFT, TPOT, or ITL
- runtime graph-capture or `torch.compile` buffers
- prefix-cache reuse of attention blocks. (The extra recurrent-state page
  vLLM's prefix caching holds on a hybrid *is* charged.)
- activation memory under `sglang` and `torch`. Only vLLM's startup
  reservation is calibrated against the engine; the other two runtimes keep
  a per-token estimate that grows with `ctx * concurrency`, which is
  conservative at long contexts.
- the gap between vLLM's weight-load figure and the checkpoint. The
  `Weights` row is the checkpoint's bytes (exact to 0.1% on 16 served
  models, quantized packing aside); vLLM's `Model loading took` runs
  0.07 GiB above that on text models and up to 0.32 GiB on Qwen VLMs,
  which the runtime floor covers -- the KV budget still lands at or below
  the engine's.
- Gemma-4's encoder reservation. Its tower pools 3x3 before the language
  model, so vLLM's profiling peak is small (0.27 GiB on E4B); the
  estimate keeps the generic maximum-image figure and runs about 1.3 GiB
  conservative there.
- per-rank activation peaks under tensor parallelism. vLLM's profiled peak
  is charged whole on every rank, which errs heavy.
- a non-default `--max-num-batched-tokens` or `--max-num-seqs`. The
  reservation is sized for vLLM's defaults; a larger batch reserves more
  (linear in the batch: 0.29 GiB at 2048 tokens, 1.11 GiB at 8192 on
  Qwen2.5-7B).
- pipeline-parallel sharding
- diffusion peak memory
- correctness or output quality after aggressive quantization
- block-quantized and group-padded weight layouts. The bytes-per-parameter
  table folds in a typical scale and zero-point overhead for group size
  128. A checkpoint with a smaller group, per-block fp8 scales, or padded
  output dimensions carries a little more than the estimate.
- quantization recipes stated only in a sidecar file. NVIDIA modelopt
  checkpoints put the recipe in `hf_quant_config.json`, which the script
  does not read; pass `--quant` explicitly for those.
- replicated projector weights on a sharded vision encoder. A multimodal
  projector or merger that is not tensor-parallel is counted as sharded,
  which understates the per-device total by up to about 0.16 GB.
- decode-time bandwidth for hybrid models. A recurrent layer reads a fixed
  state instead of a growing cache, so its decode cost does not scale with
  context the way an attention layer's does. That is a throughput question,
  not a memory one; use a benchmark skill.
- SGLang's exact slot count. Its state pool is a fixed fraction of free
  memory (charged here as `Pool split`), so it holds as many slots as fit
  that fraction, which can be more than the requested concurrency.

Use benchmark/profile skills for measured runtime behavior.

## External References

- apxml VRAM calculator, useful CUDA analogue: <https://apxml.com/tools/vram-calculator>
- vLLM XPU kernels: <https://github.com/vllm-project/vllm-xpu-kernels>
- Intel AutoRound: <https://github.com/intel/auto-round>
