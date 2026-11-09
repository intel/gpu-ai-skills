---
name: model-config-recommend
description: 'Recommend how to configure vLLM-XPU for a Hugging Face decoder-only LLM on Intel Arc B-series GPUs: choose quantization, KV dtype, DP/TP layout, max concurrency, and max context using roofline math against published hardware specs. Use when the user asks "How should I configure vLLM?", requests the best vLLM configuration, or asks how to use one or multiple Arc cards. Must run recommend.py; predictions are physics-bounded ranges, not measured throughput. Use after xpu-discover and before vllm-xpu-run.'
---

# model-config-recommend

**Status: experimental.** Emits physics-bounded predictions, not
measurements. Scoped to **vLLM-XPU**; for SGLang use
**model-can-it-fit** + **sglang-xpu-run** + **sglang-xpu-bench**.

## When to use

The user has chosen an HF decoder-only LLM and wants to know which
vLLM-XPU config (quant, KV dtype, DP/TP, max concurrency, max
context) to start with on Intel Arc B-series. Broad wording such as
"How should I configure vLLM for this model on my Arc cards?" still
activates this skill because it requires selecting layout and deployment
parameters rather than merely launching a server. Skip when:

- Exact throughput numbers required -> use **vllm-xpu-bench**.
- VLM or diffusion model -> use **model-can-it-fit**.
- Hardware not in `data/hardware.json` (other Intel families).
- MoE expert parallelism or speculative-decoding speedup — both
  workload-specific; the skill flags them as bench-only.

## Mandatory output contract (read first)

Every recommendation answer MUST do all three, even when the model
fits on one GPU — there is no "it's small, skip this" exception:

1. **Run `recommend.py`** this session against the user's actual
   model, device, context, and concurrency. Never answer from memory
   or from the "Worked example" numbers below.
2. **State the layout as the literal `dp=N, tp=M` token** (e.g.
   `dp=1, tp=1`). Prose like "no TP needed" does not count.
3. **Reproduce the full `docker run ... <model> ...` launch block**
   verbatim in a fenced code block. Never a flag table, a bare `vllm
   serve` line, or a "block above" pointer instead.

"Reporting the recommendation" below has the detail.

## Three tiers

| Tier | Script | What it does |
|---|---|---|
| 1 | `recommend.py` | Pulls config.json, applies roofline math against the spec table, emits candidates + launch line. Stdlib only, no GPU. ~2s. |
| 2 | `calibrate.py` | Runs a short BF16 bench on a reference model (`Qwen/Qwen2.5-1.5B-Instruct` by default), measures actual MFU/BWE, caches per (image, device). Requires Docker. |
| 3 | `verify.py` | Launches the recommended config on the target model, prints predicted-vs-measured with IN BAND / OUT OF BAND flags. |

```sh
# Tier 1
python3 scripts/recommend.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --device arc-pro-b70 \
    --num-devices 2 --discover-host --runtime vllm-xpu \
    --ctx 4096 --concurrency 4 \
    --gpu-memory-utilization 0.85

# Tier 2 (one-time per image)
python3 scripts/calibrate.py --image vllm/vllm-openai-xpu:latest --device arc-pro-b70
# Then re-run Tier 1 with --use-calibration vllm/vllm-openai-xpu:latest

# Tier 3 (verify a candidate)
python3 scripts/verify.py --image vllm/vllm-openai-xpu:latest --device arc-pro-b70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --quant fp8 --kv-dtype fp8 \
    --ctx 4096 --concurrency 4
```

All three tiers assume XPU graphs on, the vLLM v0.31.0+ default. For a
pre-v0.31.0 image, pass `--enforce-eager` to every tier so calibration,
prediction, and verification measure the same launch. Calibrations are cached
per mode, and `--use-calibration` refuses one taken in the other mode. It also
refuses a calibration measured against a different peak than the current
hardware table models (files from before BF16 moved to the XMX peak); re-run
`calibrate.py` to replace it.

For a plain roofline readout without a launch recommendation:

```sh
python3 scripts/llm_roofline.py --model <id> --device <id> \
    --quant fp8 --kv-dtype fp8 --ctx 8192 --concurrency 8
```

Use `--discover-host` whenever the recommendation is for the
current machine. It runs `xpu-smi discovery` first and fails if
`--num-devices` exceeds the Intel XPUs visible on the host. Omit it
only for what-if planning for a different host.

## What `recommend.py` outputs per candidate

- **Layout** `dp=N, tp=M`. TP shards one replica across GPUs (use
  when weights/KV don't fit on one GPU); DP runs independent
  replicas (use when the model fits per GPU). Recommender prefers
  DP when the model fits.
- **Fit breakdown** weights / KV / activations / framework per GPU.
  Fit is checked against `device_vram × --gpu-memory-utilization`,
  matching the vLLM launch flag emitted for the top candidate.
- **Capacity ceilings** max concurrency at target context, max
  context at target concurrency (memory-only).
- **Predicted TTFT band** (compute roofline, prefill).
- **Predicted decode band** step latency at target concurrency,
  aggregate decode tok/s.
- **Caveats** for the (quant, runtime) pair.
- **Available checkpoints** in each quant from a Hub search.
- **Launch line** for the top candidate.

## Reporting the recommendation

The launch block `recommend.py` prints begins with `docker run --rm
-d --name vllm-xpu ...` and ends with the model id plus serve flags.
Copy that whole block into a fenced ```bash block; the
recommender's stdout is not shown to the user, so a pointer like "the
block above" leaves them with nothing to run. A flag table is fine
*in addition*, never *instead*.

This skill only plans — it never launches anything itself. Present
the block as a command for the user to review and run, and have them
confirm the image tag and device mask before running it on their
host.

## Pinning the model revision

`--revision REF` takes a commit SHA, tag, or branch. It sets the
revision the config is read from *and* the revision the emitted launch
line pins, so the plan and the deployment describe the same artifact.

Left unset, vLLM resolves the repo's default branch when the server
starts: a later push by the publisher silently changes what is served,
and the run stops being reproducible. `recommend.py` prints a `# NOTE`
saying so. A commit SHA is the only value that cannot be moved.

For models whose `config.json` declares `auto_map`, the pin stops being
hygiene and becomes the mitigation. `recommend.py` therefore **requires
`--revision` alongside `--trust-remote-code`** for any Hub model and
exits 2 without it: granting arbitrary Python execution against a
moving branch means the engine runs the publisher's latest push, not
the code anyone reviewed. When both are given, the launch line carries
`--revision` and `--code-revision` — vLLM resolves repo-local modeling
code separately from the weights, so pinning one does not pin the
other. A local `--model /path/config.json` has no branch to move and
needs no pin.

## Quants and compute tiers on Battlemage

`--quantization` drives both kernel selection and which compute
tier the matmuls land on. The Xe2 XMX engines run **INT8 at 367
TOPS** and **BF16 at 183.5 TFLOPS** on an Arc Pro B70, a 2:1 ratio.
The 45.88 TFLOPS sometimes quoted for FP16 is the vector-engine rate,
not the XMX rate matmuls use. The roofline models W4A8 as INT8-tier
when the W4A8 kernel is active; FP8/BF16 stay on the BF16 tier.

| `--quantization` | Bytes/param | XMX tier | CLI pairing |
|---|---:|---|---|
| (omit; BF16/FP16) | 2.0 | XMX BF16 (183.5 TF on B70) | `--dtype bfloat16`, `--kv-cache-dtype auto` |
| `fp8` | 1.0 | XMX BF16 (no native FP8 DPAS — kernel dequantizes FP8->BF16) | `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN` |
| `awq` / `gptq` | 0.55 | INT8-modeled when W4A8 kernel is active | `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN` |
| `inc` | 0.55 | INT8-modeled when W4A8 kernel is active | INC-quantized pipelines only; most AutoRound checkpoints auto-detect via `gptq`/`awq` instead. |
| `mxfp4` | 0.55 | INT8-modeled W4A8-style path | Microscaling FP4 (GPT-OSS family). |
| AutoRound checkpoint | 0.55 | INT8-modeled when W4A8 kernel is active | Omit `--quantization` — vLLM auto-detects from `quantization_config.quant_method=auto-round` and routes through `gptq`/`awq` loader. |

### W4A8 decode win is workload-dependent on Battlemage

The roofline predicts INT4 quants win decode by ~4× over BF16 from
the weight-streaming reduction. On current vLLM-XPU images, how
close the measurement comes depends on:

- W4A8 kernel coverage on Battlemage's INT8 XMX tier is still
  maturing; some matmul shapes route through the BF16 path.
- Online dequant overhead the static formula treats as free.
- For MoE with low active-parameter ratios, routing cost can
  dominate weight-streaming savings.

Treat any INT4 candidate as "verify with bench" rather than
"preferred." The fit/headroom argument (smaller weights -> larger
context, larger concurrency) holds independently of throughput.

## Roofline model

```
weights     = params × bytes_per_param / tp      # everything in memory
roofline params = active_params / tp             # what one token reads: the
              # active experts only, no embedding table, no vision tower
kv_per_tok_per_layer = 2 × num_kv_heads × head_dim × kv_dtype_bytes
kv_cache    = kv_per_tok_per_layer × concurrency / tp
            × (num_full_attention_layers × ctx
               + num_windowed_layers × window_tokens)
state       = reserved_state_per_request × concurrency    # SSM layers only
activations = vLLM's profiling-pass peak: max of the dense MLP
              (batched_tokens × 4 × intermediate × 2 B), the routed experts,
              and a VLM's vision encoder at its profiled maximum input --
              fixed at startup and independent of concurrency; independent
              of ctx too, except an InternVL-style tower, whose profiled
              video grows with the maximum context
framework   = vLLM runtime floor (1.0 GiB, +0.15 quantized) + encoder buffers
usable      = gpu_memory_utilization × 0.946 × device_vram  # driver total

prefill_tok_s = peak_compute × MFU / (2 × params)         # compute-bound
TTFT          = prompt_tokens / prefill_tok_s

single_stream_decode_tok_s = (mem_bw / weights_bytes) × BWE   # memory-bound
TPOT_at_concurrency        = step time including KV traffic at (ctx, concurrency)
```

The cache is charged per layer because the layer kinds cost different
amounts. A windowed layer holds `window - 1` plus two scheduler batches
under vLLM, so its cache stops growing once the context passes the
window. A recurrent (state-space) layer holds no KV cache at all, only a
fixed state per running request that does not shrink with context -- so
for those models concurrency is the lever, not context. vLLM reserves more
than one bare copy of that state: it pads each state page to an attention
block (growing the block to fit the page) and, with its default prefix
caching, holds two pages per recurrent layer, about twice the bare state.
Because neither
curve is a straight line, the reported maximum context and maximum
concurrency are found by search rather than by division.

`peak_compute` is the quant's XMX tier, computed as
`xmx_engines × ops_per_clk × peak_clock` from `data/hardware.json`
(see "Hardware coverage"). MFU (compute efficiency) and BWE (bandwidth
efficiency) live there too, as ranges per runtime, and a runtime can
give a compute tier its own MFU band (`mfu_by_tier`). The XMX BF16
band is a stop-gap, a quarter of the INT8 band, chosen to keep BF16
predictions where they were when BF16 was modelled on the vector
peak; a `calibrate.py` run replaces it. See "Worked example" below for
why bands are wide. Roofline reference: Williams,
Waterman, Patterson, *Communications of the ACM* 52(4), 2008.

## Hardware coverage (`data/hardware.json`)

| Card | PCI device ID | Die | VRAM | Bus | Bandwidth | XVE / XMX engines | Peak clock |
|---|---|---|---|---|---|---|---|
| Arc B580 | `0xe20b` | BMG-G21 | 12 GB | 192-bit | 456 GB/s | 160 / 160 | 2850 MHz |
| Arc Pro B50 | `0xe212` | BMG-G21 (LP) | 16 GB | 128-bit | 224 GB/s | 128 / 128 | 2600 MHz |
| Arc Pro B60 | `0xe211` | BMG-G21 | 24 GB | 192-bit | 456 GB/s | 160 / 160 | 2400 MHz |
| Arc Pro B65 | `0xe222` | BMG-G31 (cut) | 32 GB | 256-bit | 608 GB/s | 160 / 160 | 2400 MHz |
| Arc Pro B70 | `0xe223` | BMG-G31 (full) | 32 GB | 256-bit | 608 GB/s | 256 / 256 | 2800 MHz |

The JSON stores unit counts, clocks and per-unit rates, not peak
totals. Each peak is `units × ops_per_clk × peak_clock`, with Xe2
rates per vector engine (XVE) of 32 FP32 and 64 FP16 FLOP/clk, and per
XMX engine of 256 BF16 FLOP/clk and 512 INT8 ops/clk:

| Card | FP32 vector | FP16 vector | BF16 XMX | INT8 XMX |
|---|---|---|---|---|
| Arc B580 | 14.59 TF | 29.18 TF | 116.7 TF | 233.5 TOPS |
| Arc Pro B50 | 10.65 TF | 21.30 TF | 85.2 TF | 170.4 TOPS |
| Arc Pro B60 | 12.29 TF | 24.58 TF | 98.3 TF | 196.6 TOPS |
| Arc Pro B65 | 12.29 TF | 24.58 TF | 98.3 TF | 196.6 TOPS |
| Arc Pro B70 | 22.94 TF | 45.88 TF | 183.5 TF | 367.0 TOPS |

Matmuls run on XMX, so the roofline uses the two XMX columns; the
vector columns are listed so a quoted FP16 figure can be recognised as
the vector rate. Intel's spec pages give per-card FP32 and INT8 totals;
`tests/hardware-table.sh` checks that the formula reproduces them to
within 1%. The per-unit INT8 and FP32 rates fall out of those totals;
BF16 at half the INT8 rate is the ratio Intel's oneAPI GPU
optimization guide gives for Xe-HPG and Xe-HPC. The peak clock is
Intel's Graphics Max Dynamic Clock where it lists one (B50, B70), and
its Graphics Clock for the B60 and B65. The B580 lists only a Graphics
Clock; its peak clock is 2850 MHz, the clock at which Intel's published
233 INT8 TOPS is reached.

Each row in the JSON cites Intel's SKU spec page directly, and every
PCI device ID matches both that page and the pci.ids database. The
PCI device ID is what `xpu-smi discovery` and `lspci -d 8086:` show in
brackets (e.g. `Intel(R) Graphics [0xe223]`).

## Worked example

This shows the *shape* of a `recommend.py` run, not real numbers —
the values are deliberately `<placeholders>` so they can't be pasted
into an answer. It is not a substitute for running the script:
checkpoints, spec rows, and fit math all change with the inputs. If
your answer has a number you didn't get from a `recommend.py` run
this session, it is wrong.

Qwen/Qwen2.5-7B-Instruct, 8K context, concurrency 8, Arc Pro B70
produces a block shaped like this:

```
--- Candidate 1: --quant <q>, --kv-cache-dtype <kv>, dp=<N>, tp=<M> ---
  Fit:       <W> GB weights + <K> GB KV + <A> GB act + <F> GB framework
             = <T> GB / <U> GB usable per GPU (headroom <H> GB)
  Capacity:  up to <C> concurrent at <ctx> tok per replica; ...
  Predicted: TTFT <lo>–<hi> ms; decode step <lo>–<hi> ms at concurrency <c>;
             aggregate decode <lo>–<hi> tok/s across DP=<N>
  ... (more candidates) ...

=== Launch line for top candidate ===
docker run --rm -d --name vllm-xpu \
    ...
    vllm/vllm-openai-xpu:latest \
    <checkpoint> \
        ...
```

Bands are wide because community-typical MFU/BWE ranges cover both
maturing and well-tuned kernels. Run `calibrate.py` to tighten to
your image, then `verify.py` to collapse to a measurement.

## What this skill does NOT predict

- Expert Parallelism (EP) for MoE — workload-specific. Bench
  `--enable-expert-parallel` against `--tensor-parallel-size N`.
- Speculative decoding speedup (n-gram, EAGLE, EAGLE3, MTP) —
  depends on draft accuracy on the target distribution.
- Tuning knobs `--block-size`, `--max-num-batched-tokens`,
  `--max-num-seqs` — emitted as starting points, sweep with
  **vllm-xpu-bench**.
- Tokens-per-second point estimates. Bands only.

## Common errors

- `HTTP 401` fetching `config.json` -> typo, gated repo, or rate
  limit. Set `HF_TOKEN` or pass `--config-path <local file>`.
- "config.json missing required keys" -> not a decoder-only LLM.
  Use **model-can-it-fit** for VLMs.
- Empty candidates list -> (model, device, ctx, concurrency) doesn't
  fit. Reduce concurrency / context, or use a more compressed quant.
- GGUF repo with no `config.json` -> use a safetensors checkpoint
  for vLLM-XPU, or a llama.cpp SYCL workflow for GGUF.

## What this skill does NOT cover

- Running the model — use **vllm-xpu-run** with the emitted line.
- Benchmarking — use **vllm-xpu-bench**.
- Fitting non-LLMs — use **model-can-it-fit**.

## References

- Roofline model (Williams, Waterman, Patterson, 2008):
  <https://dl.acm.org/doi/10.1145/1498765.1498785>
- vLLM Arc Pro B-series serving recipe (2025-11-11):
  <https://blog.vllm.ai/2025/11/11/intel-arc-pro-b.html>
- Intel Arc B-series SKU spec pages — one URL per row in
  `data/hardware.json`.
