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
| VLMs such as Qwen2-VL, Gemma-3 vision, LLaVA, and Nemotron-Omni | Adds vision tower and supports `text_config` or `llm_config` | Weights are exact; runtime caveats apply |
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

kv_cache =
  2 * num_layers * num_key_value_heads * head_dim
  * bytes_per_kv_dtype * ctx * concurrency
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

`int4` includes typical scale and zero overhead for grouped
quantization with group size 128. KV dtype bytes are 2 for `bf16` and
`fp16`, and 1 for `fp8` or `int8`.

## Important Modeling Details

Use explicit `head_dim` from config when present. Some models use a
larger head dimension than `hidden_size / num_attention_heads`; ignoring
that undercounts KV cache and Q/O projection parameters.

For MoE models, prefer `moe_intermediate_size`, `expert_hidden_dim`, or
the equivalent MoE-specific FFN field over dense `intermediate_size`.
Shared experts are always active in addition to routed experts.

For mixed-precision quantized models, inspect
`quantization_config.modules_to_not_convert`. The script keeps
recognized embeddings, attention, or router modules at full precision
and prints a component-level weight breakdown.

## Tested Model Families

Dense coverage includes Qwen2.5, Llama 3.1/3.3, Gemma-2, and
Nemotron-70B style configs.

MoE coverage includes Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral,
DeepSeek-V3/V4-style hybrid MoE, and Mistral-Large-3 style
`params.json` configs.

Multimodal coverage includes Qwen2-VL, Gemma vision configs, LLaVA-like
configs, and Nemotron-Omni style `llm_config` layouts.
