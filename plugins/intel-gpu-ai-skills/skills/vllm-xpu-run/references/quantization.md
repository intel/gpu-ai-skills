# Quantization on vLLM-XPU

`--quantization` drives kernel selection; the engine logs the
chosen kernel (e.g. `Selected XPUFP8ScaledMMLinearKernel`).

| Quant | `--quantization` | `--kv-cache-dtype` | `--attention-backend` |
|---|---|---|---|
| BF16 / FP16 | (omit) | `auto` | `FLASH_ATTN` |
| FP8 (fp8 KV) | `fp8` | `fp8` | **`TRITON_ATTN`** |
| FP8 (bf16 KV) | `fp8` | `auto` | `FLASH_ATTN` or `TRITON_ATTN` |
| AWQ / GPTQ | `awq` / `gptq` | `fp8` | `TRITON_ATTN` |
| MXFP4 | `mxfp4` | `auto` | `TRITON_ATTN` |
| AutoRound checkpoint | (omit; auto-detected) | `auto` | `TRITON_ATTN` |

AutoRound is an algorithm, not a `--quantization` value — vLLM
auto-detects from `quantization_config.quant_method=auto-round` and
routes through the `gptq` or `awq` loader. Omit `--quantization`
for AutoRound checkpoints.

## Verify the live kernel

```sh
docker logs <name> 2>&1 | grep -E "Selected.*Kernel|XPUFP8|gemm"
# If you asked for AWQ/GPTQ but see int4_gemm_w4a16 -> silent
# fall-through (vLLM #38064).
```

**Validate output content** for any quant config — quant kernels
can produce non-language output without raising. Read 2–3 sample
completions before trusting throughput numbers.

## Attention backends

| Backend | Set with | When |
|---|---|---|
| Triton XPU | `--attention-backend TRITON_ATTN` | Required with `--kv-cache-dtype fp8` (FA-XPU does not implement fp8 KV — engine exits with `NotImplementedError`). Required by W4A8 quant kernels (`int4_gemm_w4a8`, `fp8_gemm_w8a16`). |
| Flash Attention (XPU) | `--attention-backend FLASH_ATTN` | Often faster than Triton with bf16/fp16 KV. Mutually exclusive with Triton. |
| `TORCH_SDPA` | `--attention-backend TORCH_SDPA` | Pure-PyTorch baseline; slow but portable. |

Set `--attention-backend` explicitly so the live path is known.
Confirm after launch:

```sh
docker logs <name> 2>&1 | grep -iE "Using.*backend|attention.*backend"
```
