# Coverage And Formulas

Use this reference when the user asks how the estimate is calculated,
why it differs from another calculator, or whether a specific model class
is covered.

## Coverage

| Model class | Behavior | Expected accuracy |
|---|---|---|
| Decoder-only LLMs such as Qwen, Llama, Mistral, and Gemma text models | Full estimate | About 5% |
| MoE models such as Qwen3-MoE, Mixtral, and DeepSeek-V3 | Full estimate, including shared experts | About 10% |
| Hybrid MoE models such as DeepSeek-V2/V3/V4 and Mistral-Large-3 | Counts dense and MoE layers separately via `first_k_dense_replace` | About 10% |
| MoE with a separate expert dtype such as DeepSeek-V4-Flash | Three-way split: expert FFNs at `expert_dtype`, embeddings and untied LM head at `bf16`, remaining non-expert weights at `quant_method` | About 5% |
| VLMs such as Qwen2-VL, Gemma-3 vision, LLaVA, and Nemotron-Omni | Adds the vision tower at its own dtype and supports `text_config` or `llm_config` | About 2% on weights; runtime caveats apply |
| Mistral `params.json` models | Reads `params.json` when `config.json` is absent | About 10% |
| Diffusion models | Refuses as a full estimate | Use empirical benchmarking |

The script reads standard Hugging Face `config.json`, Mistral
`params.json`, or a local JSON path. It detects diffusion repos from
`model_index.json` and exits with a routing message.

## Core Formula

```text
VRAM = weights + kv_cache(ctx, concurrency) + activations + framework
usable_vram = physical_vram * gpu_memory_utilization
fits = VRAM <= usable_vram
```

Weights are estimated from config dimensions:

```text
head_dim = config.head_dim or hidden_size / num_attention_heads
q_proj_dim = num_attention_heads * head_dim
kv_proj_dim = num_key_value_heads * head_dim

attention_per_layer =
    hidden * q_proj_dim
  + hidden * kv_proj_dim
  + hidden * kv_proj_dim
  + q_proj_dim * hidden

dense_ffn_per_layer = 3 * hidden * intermediate_size
moe_ffn_per_layer =
    num_experts * 3 * hidden * moe_intermediate_size
  + num_shared_experts * 3 * hidden * moe_intermediate_size
  + hidden * num_experts                    # gate/router projection

kv_cache =
  2 * num_layers * num_key_value_heads * head_dim
  * bytes_per_kv_dtype * ctx * concurrency

vision_tower =                                  # when vision_config is present
    vw = vision_config.embed_dim or vision_config.hidden_size
    patch_size^2 * num_channels * vw            # patch embedding
  + vision_depth * (
        4 * vw * vw                             # ViT attention, square
      + vision_mlp_matrices * vw * vision_intermediate
      + 2 * vw )                                # two LayerNorms
```

The script includes embeddings and untied LM head when applicable. For
hybrid MoE models, dense replacement layers use dense FFN dimensions and
remaining layers use MoE expert dimensions.

Activation memory is a bounded estimate:

```text
activations ~= 2 * concurrency * ctx * hidden_size * bytes_per_param + 512 MiB
```

Runtime framework overhead is a floor estimate:

| Runtime | Overhead |
|---|---:|
| `vllm` | About 2.0 GiB |
| `sglang` | About 1.5 GiB |
| `torch` | About 0.8 GiB |

## Bytes Per Parameter

| Quant | Bytes per parameter |
|---|---:|
| `bf16` / `fp16` | 2.00 |
| `fp8` / `int8` | 1.00 |
| `int4` | 0.55 |
| `int3` | 0.42 |
| `int2` | 0.30 |
| `mxfp4` | 0.55 |
| `fp4` | 0.55 |
| `nvfp4` | 0.58 |
| `fp32` | 4.00 |

`int4` includes typical scale and zero overhead for grouped
quantization with group size 128. KV dtype bytes are 2 for `bf16` and
`fp16`, and 1 for `fp8` or `int8`.

These per-parameter figures apply to the Linear layers. **Embeddings and an
untied LM head stay at 16-bit for every sub-16-bit dtype**, whether or not the
config lists them in `modules_to_not_convert`: quantizers convert Linear
layers, not `nn.Embedding`. Measured on two served 4-bit builds of
Qwen2.5-7B-Instruct (group_size 128, untied, 152064 vocab):
`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` allocated 5.18 GiB and
`Qwen/Qwen2.5-7B-Instruct-AWQ` 5.20 GiB, against 3.90 GiB if their 1.09 B
embedding params are priced at int4. A wide-vocab 7B is 25% larger than a flat
int4 estimate suggests, which is enough to turn an OOM into a FITS verdict.

`fp4` is **MXFP4 under another name**, not a separate format: DeepSeek-V4
spells its MXFP4 experts `expert_dtype: fp4`, and vLLM's DeepSeek-V4 quant
config reads that value as "MXFP4 experts with ue8m0 (e8m0fnu) FP8 linear
scales" and dispatches it to `Mxfp4MoEMethod`. A config-declared `fp4`
therefore normalizes to `mxfp4`, inheriting its supported-kernel status; the
row survives only so an explicit `--quant fp4` still parses. The 0.55 figure is
deliberately conservative: DeepSeek-V4-Flash measures 0.531 effective bytes per
param (4 bits plus one `ue8m0` byte per 32), so the estimate runs ~4% heavy.

`nvfp4` is a separate row because NVFP4 scales in blocks of 16, not 32: one
fp8 (e4m3) scale per 16 fp4 values is 0.5625 bytes per param packed, before
the per-tensor fp32 scale and any block padding. 0.55 is below that floor,
so folding NVFP4 into `fp4` would understate every such checkpoint; 0.58
keeps the conservative margin the other rows carry.

`nvfp4` has no documented XPU kernel path: this pack documents bf16/fp16,
fp8, AWQ/GPTQ, MXFP4 and AutoRound on XPU, and NVFP4's per-16 fp8 block scales
are not on that list. The row exists because the bytes on disk are real and
worth sizing, but the script prints a `Kernel path:` caveat on such a verdict,
along with the bf16 weight figure that applies if the runtime upcasts instead
of refusing. The caveat names the runtime that was asked about --
`vllm-xpu-run`, `sglang-xpu-run` or `torch-xpu-run` -- because a successful load
on one says nothing about another's kernels. Treat those verdicts as bytes-only
until a load confirms otherwise.

**MXFP4 is not flagged**, `fp4` spelling included. Both spellings reach the
same XPU kernel: `quant_method: mxfp4` and `expert_dtype: fp4` take different
quant configs and MoE methods (`GptOssMxfp4MoEMethod` vs `Mxfp4MoEMethod`) but
both short-circuit on `is_xpu()` to `Mxfp4MoeBackend.XPU` -> `XPUExpertsMxFp4`,
which consumes the checkpoint layout directly. A served gpt-oss-20b allocated
12.87 GiB against a 13.14 GiB estimate, so those weights measurably stayed
packed; the `expert_dtype: fp4` dispatch is verified by inspection of vLLM
0.29.0 in the documented image, not by serving DeepSeek-V4-Flash, which does
not fit the hardware used here.

On `--runtime vllm`, `fp8`, `int8` and `int4` auto-pair KV with `fp8`
(`FP8_KV_PAIRED_QUANTS`), matching the only rows **vllm-xpu-run**'s
quantization table pairs with `--kv-cache-dtype fp8`. `mxfp4`, `fp4`, `nvfp4`,
`int3` and `int2` default to `bf16` KV there, because that table documents
`auto` KV for MXFP4 and AutoRound.

The dtype is not the whole test: a declared `quant_method` also has to name one
of those rows (`FP8_KV_PAIRED_METHODS` -- `fp8`, `awq`, `gptq`). AutoRound is
why. Its width comes from `bits`, so a 4-bit AutoRound checkpoint reduces to
`int4`, but AutoRound has its own `auto` KV row -- vLLM detects
`quant_method=auto-round` and routes it through the gptq/awq loader without
that pairing. Pairing on the reduced width would halve the cache against the
launch the table documents: the synthetic MoE in
`tests/test_fit.py::TestExpertDtype` moves from 4.00 to 8.00 GiB of KV at
32k/c=8, which is the difference between FITS and DOES NOT FIT on an 18 GB
budget. When a config declares no method -- or an explicit `--quant` differs
from what ships, which is a re-quantization the config's method no longer
describes -- the dtype decides alone, because a 4-bit re-quantization on XPU
means AWQ or GPTQ. The report names the method rather than the width when the
method is what withheld the pairing.

The pairing is vLLM's alone (`FP8_KV_AUTOPAIR_RUNTIMES`). **sglang-xpu-run**
documents that fp8 weights keep a BF16 cache unless `--kv-cache-dtype
fp8_e4m3` is set, so `--runtime sglang` defaults to `bf16` KV for every weight
dtype. The `torch` path has no KV-dtype flag -- transformers allocates the
cache in the model dtype -- so it defaults to `bf16` and the report says no
lever exists rather than naming one. Applying vLLM's pairing to either would
halve the estimated cache against that runtime's own default.

KV dtype is a launch flag, not a property of the checkpoint: nothing about
4-bit weights requires an 8-bit cache, and a `--quant` that is narrow says
only that the *weights* are. So the default is never made narrower than the
runtime guidance supports -- halving the cache on an unmet assumption is how a
FITS verdict becomes an OOM. Where `fp8` KV is assumed, the report says what
it costs if the launch omits `--kv-cache-dtype fp8`; where a narrow dtype is
*not* paired, it names `--kv-dtype fp8` as the lever instead of applying it.

An `expert_dtype` does not enter this decision. The cache lives on the
attention path and the experts are not in it, so an fp4-expert checkpoint with
a bf16 base gets bf16 KV.

`quant_method` spellings are normalized before lookup: `nvfp4` and
`modelopt_fp4` normalize to `nvfp4`; `awq`/`gptq`/AutoRound derive
`int8`/`int4`/`int3`/`int2` from `bits` (or `w_bit`) and conservatively
assume `int8` when no usable width is declared.

`awq`, `gptq` and `autoround` name an algorithm, not a width -- all three
ship 8-, 4-, 3- and 2-bit checkpoints -- so the dtype comes from
`quantization_config.bits` (`w_bit` on older AWQ configs): 8 to `int8`, 4
to `int4`, 3 to `int3`, 2 to `int2`. An Int8 GPTQ checkpoint read as
`int4` would understate weights by nearly 2x, which is enough to turn an
OOM into a FITS verdict.

When one of those methods declares no usable width, the script assumes
`int8` -- the widest it ships -- and says on the `Quantization` line that
the width was assumed, because understating weights is the direction that
produces a false FITS. Pass `--quant` explicitly to price a narrower
checkpoint.

## Important Modeling Details

Use explicit `head_dim` from config when present. Some models use a
larger head dimension than `hidden_size / num_attention_heads`; ignoring
that undercounts KV cache and Q/O projection parameters.

For MoE models, prefer `moe_intermediate_size`, `expert_hidden_dim`, or
the equivalent MoE-specific FFN field over dense `intermediate_size`.
Shared experts are always active in addition to routed experts.

Router (gate) projections are counted as `hidden * num_experts` per MoE
layer -- one linear layer scoring every routed expert, and no gate for the
always-on shared experts. That is 0.04% of DeepSeek-V4-Flash, so it barely
moves a uniform estimate, but it is what a `modules_to_not_convert` list
naming routers holds at `bf16` while the rest goes to 4-bit, and the uplift
is priced on this count.

A VLM's vision tower is sized from `vision_config`, whose field names do not
mean the same thing across families:

- **Width** is `embed_dim` where a config declares one, `hidden_size`
  otherwise. Qwen2-VL and Qwen2.5-VL name the ViT width `embed_dim` (1280) and
  reuse `vision_config.hidden_size` for the *output* of the patch merger
  (3584), so reading `hidden_size` there prices 32 blocks at 2.8x their width
  -- 4.94 B params against a ~0.68 B tower. CLIP-style configs (LLaVA,
  Mistral-Small) carry no `embed_dim`, and `hidden_size` is the width.
- **Depth** is `depth` or `num_hidden_layers`.
- **MLP width** is `intermediate_size`, or `mlp_ratio * width` where only
  Qwen2-VL's ratio is given.
- **MLP matrices** are three, not two, on a gated tower. Qwen2.5-VL's is
  SwiGLU and says so with `hidden_act: silu`; CLIP-style towers are
  `gelu`/`quick_gelu` with two. Counting two on a gated tower is 22% light on
  the blocks, and light is the direction that reports a false FITS.

A `vision_config` with no depth field cannot be sized -- Qwen2.5-VL-7B-AWQ
ships such a config. The report then names the missing field and says the
weights below cover the LLM backbone only, rather than printing a 0.00 B tower
that reads as "no vision weights".

A TP value is only estimated if the model's shard dimensions divide by it:

- TP divides `num_attention_heads` (28 query heads rule out TP 8).
- TP divides every FFN width a layer actually uses --
  `moe_intermediate_size` on an MoE, `intermediate_size` on a dense model,
  and both on a hybrid MoE, whose first `first_k_dense_replace` layers are
  dense (`intermediate_size` 28672 rules out TP 6). A pure MoE's
  `intermediate_size` is not checked: no layer uses it.
- TP divides `num_key_value_heads`, or is a multiple of it, since the runtime
  replicates the KV group above that count (8 KV heads rule out TP 6).

For `--runtime vllm` and `--runtime sglang`, `--tp` errors out and lists the
values that divide, and `--tp-sweep` drops such values with a note (a sweep
is exploratory). `--runtime torch` means accelerate's `device_map`, which
places whole modules per device rather than splitting a layer by head, so
those TP values are estimated with a warning instead of refused.

Divisibility is necessary, not sufficient: a runtime can reject a TP for
reasons this script does not model -- quantization group sizes, expert counts
per rank, its own padding rules. Treat a surviving TP as worth trying, and
confirm the launch itself with **vllm-xpu-run**.

How the footprint divides depends on what the runtime can split.

On `vllm` and `sglang`, weights divide by `--tp` exactly -- every matrix is
partitioned -- while KV divides by `min(tp, num_key_value_heads)`: attention
shards by KV head, so KV stops shrinking once TP passes the KV-head count and
the runtime replicates the group instead, every rank keeping one head. A
4-KV-head model at TP 8 stores `total_KV / 4` per device, not `total_KV / 8`;
dividing by TP past that point understates KV and can report FITS for a launch
that OOMs.

On `--runtime torch`, accelerate's `device_map` places whole transformer
layers, so it cannot split anything more finely than one layer. Weights *and*
KV divide by `num_layers // ceil(num_layers / tp)` -- the share held by the
busiest device. Two consequences:

- Gains stop at the layer count. A 4-layer model is estimated identically at
  `--tp 4`, 8 and 16, because the spare devices hold nothing.
- An uneven split is dominated by the fullest device. 33 layers over 8 devices
  leaves one device with 5, i.e. `total / 6.6`; the divisor floors to 6, so the
  figure errs heavy rather than light.

One thing the layer share cannot go below: the embedding matrix, and the LM
head when untied, are single module weights that `device_map` assigns whole.
`vocab * hidden * bytes_per_param` is therefore a floor on the per-device
weight figure -- without it a 262144-vocab, 4-layer model over 4 devices reports
2.82 GiB against a 4.00 GiB matrix it must materialise, and says FITS on a
5 GiB card. When the floor binds, the report prints a `Placement floor:` line
naming the matrix, because neither the component rows nor `total / divisor`
explains the number any more. Head-sharded runtimes are unaffected: they
partition the vocab dimension itself.

The model is still optimistic after that floor: the device holding the
embedding also holds layers, and accelerate balances by module size rather than
module count, so the fullest device can hold more than the figure shown. Treat
a tight torch verdict as needing confirmation from
`torch.xpu.max_memory_allocated()` via **torch-xpu-bench**.

Every weight figure the report quotes beside the verdict -- as-shipped, bf16
upcast, each `--quant` candidate, the wider-device option -- is produced by
re-running the same estimate with those arguments, not by scaling the printed
number. Scaling is wrong exactly where it matters: it divides the placement
floor that no device count can split, and it assumes gains past the layer count
that module placement cannot deliver. A wider-device option that reports no
saving is therefore dropped rather than printed, which is what happens once the
floor binds or the layers run out.

Which wider count gets suggested also follows the runtime. On `vllm`/`sglang`
it is the next TP whose shard dimensions divide. Under module placement it is
the next count that leaves the busiest device fewer layers, which is not a
doubling. For 40 layers the busiest device holds fewer layers at 1, 2, 4, 5,
7, 8 devices -- 3 is no better than 2 and **6 is no better than 5** -- so the
suggestion after `--tp 4` is 5, and after `--tp 5` it is 7. A 4-layer model at
3 devices is told 4 rather than 6: 6 would improve nothing that 4 does not,
with two devices idle. KV mitigations use the verdict's KV divisor for the same
reason.

## Mixed And Per-Component Precision

A checkpoint's weights are often not all one dtype. Two independent
config mechanisms express that, and a model may use both:

**1. `quantization_config.modules_to_not_convert`** (e.g.
openai/gpt-oss-20b) names components held at full precision while the
rest is quantized. The script recognizes embeddings, attention, router, and
vision entries. Routers are matched under both spellings in the wild --
`router` (gpt-oss's `mlp.router`) and `gate` as a whole path segment
(Mixtral's `block_sparse_moe.gate`, DeepSeek and Qwen3-MoE's `mlp.gate`) --
while `gate_proj` and `gate_up_proj` are left as the dense SwiGLU
projections they are.

The vision tower is priced by that list, not by the embedding rule: a ViT is
Linear layers, so a quantizer converts it unless the list excludes it. Qwen2-VL
and Qwen2.5-VL AWQ builds list exactly `["visual"]`, which holds the whole
tower at `bf16` while the backbone goes to 4-bit -- 2.92 GiB of the 6.55 GiB
estimate for `Qwen/Qwen2-VL-7B-Instruct-AWQ`, against 6.45 GiB on disk. A list
that names only the backbone's own components leaves the tower at the
quantized rate. `visual`, `vision_tower` and `vision_model` are all matched,
including as a `re:` pattern.

**2. A separate `expert_dtype`** (e.g. deepseek-ai/DeepSeek-V4-Flash)
gives the MoE expert FFNs their own dtype. DeepSeek-V4-Flash declares
`quantization_config.quant_method: fp8` *and* `expert_dtype: fp4`, which
makes it a three-way split: 278.1 B of experts at 4-bit, 1.1 B of
embeddings plus untied LM head at `bf16`, and only the 11.7 B of remaining
non-expert weights at fp8.

Reading `quant_method` alone and applying it uniformly prices 96% of that
model at double its real size -- 270.9 GiB of "weights" against 148.7 GiB
of actual safetensors -- which is enough to flip an 8-XPU verdict from
FITS to DOES NOT FIT. Always report which dtype landed on the experts.

Either way the script prints a component-level weight breakdown. With
`expert_dtype`, the rows are embeddings at `bf16`, non-expert weights at
`quant_method`, and expert FFNs at `expert_dtype`. A sized vision tower is its
own row, at whichever rate the exclusion list gave it, so the rows still sum to
`Weights total`.

An `expert_dtype` this table cannot price is refused, not ignored: the
experts are ~96% of an MoE, so falling back to the base dtype would be a
guess, and a guess in the false-FITS direction whenever the real dtype is
wider. `--quant <dtype>` still answers, because a uniform re-quantization does
not depend on the unknown value. Torch dtype spellings (`float32`, `bfloat16`,
`float16`) are normalized rather than refused.

`expert_dtype` applies **whenever the requested weight dtype still matches
the config** -- auto-detected, or typed explicitly as the same dtype.
`--quant fp8` against a `quant_method: fp8` checkpoint keeps its fp4
experts, because that argument restates the config rather than overriding
it.

"Matches the config" needs a declared base to match. Three cases, and they are
not the same:

- **No `quantization_config`**: the checkpoint ships 16-bit, so `--quant bf16`
  restates it and the expert split survives.
- **A priceable `quant_method`**: that dtype is what an explicit `--quant` is
  compared against.
- **A declared but unpriceable `quant_method`** (say `compressed-tensors`): the
  shipped base is *unknown*. Nothing can restate it, so every explicit `--quant`
  is an override, and no as-shipped figure is quoted -- the report says
  `as-shipped size unknown` and names the method instead of inventing a
  bf16-based number. Auto-detection prices it at `bf16`, the widest it could be,
  and says so.

Only a *differing* `--quant` suppresses the split, since it posits a
uniform re-quantization the config no longer describes. The same test decides
how a suggested mitigation is priced: on a failed `--quant bf16` run against an
fp8 + fp4-expert checkpoint, the suggested `--quant fp8` restates the config
and keeps the fp4 experts, so its quoted saving is the split figure, while a
suggested `--quant int4` is priced uniformly. That case prints
`hypothetical:` on the `Expert weights:` line along with the as-shipped
weight figure for comparison, so a re-quantization estimate cannot be
misread as the checkpoint's real size.

## Tested Model Families

Dense coverage includes Qwen2.5, Llama 3.1/3.3, Gemma-2, and
Nemotron-70B style configs.

MoE coverage includes Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral,
DeepSeek-V3/V4-style hybrid MoE, DeepSeek-V4-Flash (fp8 + fp4 experts),
and Mistral-Large-3 style `params.json` configs.

Multimodal coverage includes Qwen2-VL and Qwen2.5-VL (bf16 and AWQ builds),
Gemma vision configs, LLaVA-like configs, and Nemotron-Omni style `llm_config`
layouts.

## Validation

### Measured against served deployments

The strongest check available: the weight figure vLLM-XPU itself reports at
engine init (`Model loading took X GiB`), from serving each model on 2x Intel
Arc (Battlemage, 32656 MiB per device), one device, with the image
**vllm-xpu-run** documents -- `vllm/vllm-openai-xpu:latest`, digest
`sha256:96db42e248d48760a4937eb3d04c4878b39d13a9814efea95d510393e097a901` --
and `--enforce-eager --block-size 64 --max-model-len 4096
--gpu-memory-utilization 0.85`. Every row served a real completion, and each
VLM row also answered a base64 image request, so its tower was exercised rather
than merely allocated. `tests/test_fit.py::TestMeasuredOnDevice` pins these
figures against the revisions that produced them.

`Error` below is `(engine - estimate) / estimate`: **positive means the
estimate sat below what the engine allocated**, which is the direction that
reports FITS for a launch that OOMs. `TestMeasuredOnDevice` asserts the
inverse, `(estimate - engine) / engine`, bounded at 5% either way and at -2% on
the low side specifically; the two are the same comparison read from opposite
ends.

| Model | Estimate | Engine reported | Error (engine-est)/est |
|---|---:|---:|---:|
| Qwen/Qwen2.5-0.5B-Instruct | 0.92 GiB | 0.93 GiB | +1.1% |
| Qwen/Qwen2.5-1.5B-Instruct | 2.88 GiB | 2.89 GiB | +0.3% |
| Qwen/Qwen3-0.6B | 1.11 GiB | 1.12 GiB | +0.9% |
| Qwen/Qwen3-4B | 7.49 GiB | 7.56 GiB | +0.9% |
| meta-llama/Llama-3.2-3B-Instruct | 5.98 GiB | 6.02 GiB | +0.7% |
| microsoft/Phi-4-mini-instruct | 7.15 GiB | 7.17 GiB | +0.3% |
| Qwen/Qwen2.5-7B-Instruct | 14.19 GiB | 14.25 GiB | +0.4% |
| mistralai/Mistral-7B-Instruct-v0.3 | 13.50 GiB | 13.51 GiB | +0.1% |
| meta-llama/Llama-3.1-8B-Instruct | 14.96 GiB | 14.99 GiB | +0.2% |
| Qwen/Qwen3-8B | 15.26 GiB | 15.27 GiB | +0.1% |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 15.26 GiB | 15.29 GiB | +0.2% |
| NousResearch/Hermes-3-Llama-3.1-8B | 14.96 GiB | 14.99 GiB | +0.2% |
| tiiuae/Falcon3-7B-Instruct | 13.89 GiB | 13.93 GiB | +0.3% |
| nvidia/Llama-3.1-Nemotron-Nano-8B-v1 | 14.96 GiB | 14.99 GiB | +0.2% |
| Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 | 5.37 GiB | 5.18 GiB | -3.5% |
| Qwen/Qwen2.5-7B-Instruct-AWQ | 5.37 GiB | 5.20 GiB | -3.2% |
| openai/gpt-oss-20b | 13.14 GiB | 12.87 GiB | -2.1% |
| Qwen/Qwen2-VL-7B-Instruct | 15.36 GiB | 15.53 GiB | +1.1% |
| Qwen/Qwen2.5-VL-7B-Instruct | 15.36 GiB | 15.63 GiB | +1.8% |
| Qwen/Qwen2-VL-7B-Instruct-AWQ | 6.55 GiB | 6.49 GiB | -0.9% |

Mean error -0.05% over the 20 rows above, range -3.5% to +1.8%. Adding
Qwen3-14B -- which loaded its weights at 27.52 GiB against 27.51 estimated
before failing on KV, so it is listed under the verdicts below rather than in
this table -- gives -0.04% over 21. The two 4-bit text rows are the wide
embedding rule in action: pricing their embedding matrix and LM head at int4
instead of bf16 gives 3.90 GiB against the 5.18 GiB the engine allocated.

The AWQ VLM row is the vision-tower rule in action the same way. Its exclusion
list is `["visual"]`, so the 0.63 B tower stays bf16 -- 1.17 GiB of the
6.55 GiB estimate. Pricing that tower at int4 gives 5.70 GiB against the
6.49 GiB the engine allocated, 12% low.

A fourth VLM was served and is deliberately **not** in the table:
`Qwen/Qwen2.5-VL-7B-Instruct-AWQ`, whose `vision_config` carries no depth
field. The tower cannot be sized from it, so the estimate covers the backbone
only -- 5.37 GiB against **6.59 GiB allocated**, 1.22 GiB short. That is the
gap the `Vision tower: not sized` line exists to report, and
`TestMeasuredOnDevice::test_unsized_tower_understates_by_the_measured_gap`
pins its size so a later sizing fix has to update the wording with the number.

Two models were predicted DOES NOT FIT at that setting and failed engine init
as predicted: Qwen3-14B (27.51 GiB estimated, 27.52 GiB loaded, against
27.11 GiB usable -> `No available memory for the cache blocks`) and
Qwen3.5-35B-A3B (64.11 GiB -> `XPU out of memory` during the load).

The `Capacity:` ceiling was checked the same way, against the KV pool vLLM
allocated (`GPU KV cache size: N tokens`):

Same convention: `(allocated - predicted) / predicted`, so positive means
the prediction was the conservative one. The test bounds
`(predicted - allocated) / allocated` at +/-10%.

| Model | Predicted max ctx | KV pool allocated | Error (alloc-pred)/pred |
|---|---:|---:|---:|
| Qwen/Qwen2.5-0.5B-Instruct | 2068542 | 2105664 | +1.8% |
| Qwen/Qwen2.5-1.5B-Instruct | 812931 | 820352 | +0.9% |
| Qwen/Qwen3-0.6B | 219831 | 223680 | +1.8% |
| Qwen/Qwen3-4B | 124334 | 126400 | +1.7% |
| meta-llama/Llama-3.2-3B-Instruct | 173905 | 177920 | +2.3% |
| microsoft/Phi-4-mini-instruct | 142656 | 142690 | +0.0% |
| Qwen/Qwen2.5-7B-Instruct | 194108 | 194496 | +0.2% |
| mistralai/Mistral-7B-Instruct-v0.3 | 90464 | 93696 | +3.6% |
| meta-llama/Llama-3.1-8B-Instruct | 78528 | 80768 | +2.9% |
| Qwen/Qwen3-8B | 67626 | 69632 | +3.0% |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 67626 | 69568 | +2.9% |
| NousResearch/Hermes-3-Llama-3.1-8B | 78528 | 80768 | +2.9% |
| tiiuae/Falcon3-7B-Instruct | 99916 | 100480 | +0.6% |
| nvidia/Llama-3.1-Nemotron-Nano-8B-v1 | 78528 | 80768 | +2.9% |
| Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 | 718233 | 721920 | +0.5% |
| Qwen/Qwen2.5-7B-Instruct-AWQ | 718233 | 717184 | -0.1% |
| openai/gpt-oss-20b | 249585 | 249633 | +0.0% |

Mean +1.62%, range -0.1% to +3.6%. Non-weight overhead measured as
`usable - weights - KV pool` came to **2.08-2.76 GiB** (mean 2.33), against the
~2.5 GiB this script assumes -- 2.0 GiB vLLM floor plus the activation
estimate. That floor is calibrated for this image; a different vLLM build can
reserve substantially less, and recalibrating against one would produce false
FITS verdicts on the image the skill tells users to run.

### Measured against safetensors byte totals

Weight estimates are also checked against the real root-level safetensors byte
totals of each repo, which needs no hardware:

| Model | Estimate | On disk | Error |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct (bf16) | 14.19 GiB | 14.19 GiB | +0.0% |
| Qwen3-30B-A3B (bf16 MoE) | 56.87 GiB | 56.87 GiB | +0.0% |
| gpt-oss-20b (mxfp4 experts) | 13.14 GiB | 12.82 GiB | +2.5% |
| DeepSeek-V4-Flash (fp8 + fp4 experts) | 155.39 GiB | 148.66 GiB | +4.5% |
| Qwen2-VL-7B-Instruct (bf16 VLM) | 15.36 GiB | 15.44 GiB | -0.5% |
| Qwen2-VL-7B-Instruct-AWQ (int4 + bf16 tower) | 6.55 GiB | 6.45 GiB | +1.6% |
| Qwen2.5-VL-7B-Instruct (bf16 VLM, gated tower) | 15.36 GiB | 15.45 GiB | -0.6% |

The three VLM rows are the hardware-free check on the vision tower, since the
tower is a fixed share of those bytes; all three were later served as well, and
those figures are in the table above. What remains is the patch merger (~46 M
params), which the tower formula does not count -- the reason the bf16 rows sit
slightly low. `tests/test_fit.py::TestVisionTowerPricing` pins them at a 3%
bound.

Reproduce a row by summing the repo's root-level `*.safetensors` sizes
from `https://huggingface.co/api/models/<id>?blobs=true` and comparing
against the script's `Weights` line at `--tp 1`. Count root level only:
some repos ship a duplicate copy in a subdirectory (gpt-oss-20b has
`original/`), which doubles a naive sum.

The regression suite lives at the repo root and covers dimension
parsing, KV ground truth, and verdicts:

```sh
python3 -m pytest tests/test_fit.py -q
```

It also covers both mixed-precision paths above: `modules_to_not_convert`
against pinned gpt-oss configs, and `expert_dtype` against synthetic MoE
configs (`TestExpertDtype`), which pin the component rows, the TP split, and
the matching-vs-differing `--quant` behavior without a network fetch. Every
weight figure quoted in this file is reproducible from the recipe above or
from that suite.
