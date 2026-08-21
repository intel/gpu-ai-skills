---
name: model-config-recommend
description: Recommend a vLLM-XPU deployment config (quant, KV dtype, DP/TP, max concurrency, max context) for a Hugging Face decoder-only LLM on Intel Arc B-series GPUs using roofline math against published hardware specs. Experimental; predictions are physics-bounded ranges, not measured throughput. Use after xpu-discover and before vllm-xpu-run.
---

# model-config-recommend

**Status: experimental.** Emits physics-bounded predictions, not
measurements. Scoped to **vLLM-XPU**; for SGLang use
**model-can-it-fit** + **sglang-xpu-run** + **sglang-xpu-bench**.

## When to use

The user has chosen an HF decoder-only LLM and wants to know which
vLLM-XPU config (quant, KV dtype, DP/TP, max concurrency, max
context) to start with on Intel Arc B-series. Skip when:

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
3. **Reproduce the full `docker run ... vllm serve ...` launch block**
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
python3 scripts/calibrate.py --image intel/vllm:0.17.0-xpu --device arc-pro-b70
# Then re-run Tier 1 with --use-calibration intel/vllm:0.17.0-xpu

# Tier 3 (verify a candidate)
python3 scripts/verify.py --image intel/vllm:0.17.0-xpu --device arc-pro-b70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --quant fp8 --kv-dtype fp8 \
    --ctx 4096 --concurrency 4
```

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
-d --name vllm-xpu ...` and ends with the `vllm serve <model> ...`
flag list. Copy that whole block into a fenced ```bash block; the
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
TOPS** vs **FP16/BF16 at 45.88 TFLOPS** (Arc Pro B70 spec). The
roofline models W4A8 as INT8-tier when the W4A8 kernel is active;
FP8/BF16 stay on the FP16 tier.

| `--quantization` | Bytes/param | XMX tier | CLI pairing |
|---|---:|---|---|
| (omit; BF16/FP16) | 2.0 | FP16/BF16 (45.88 TF on B70) | `--dtype bfloat16`, `--kv-cache-dtype auto` |
| `fp8` | 1.0 | FP16/BF16 (no native FP8 DPAS — kernel dequantizes FP8->BF16) | `--kv-cache-dtype fp8`, `--attention-backend TRITON_ATTN` |
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
weights     = params × bytes_per_param / tp
kv_cache    = 2 × num_layers × num_kv_heads × head_dim × kv_dtype_bytes × ctx × concurrency / tp
activations ≈ 2 × concurrency × ctx × hidden × dtype_bytes + 512 MiB
framework   = vllm-xpu overhead (~2.0 GB)

prefill_tok_s = peak_compute × MFU / (2 × params)         # compute-bound
TTFT          = prompt_tokens / prefill_tok_s

single_stream_decode_tok_s = (mem_bw / weights_bytes) × BWE   # memory-bound
TPOT_at_concurrency        = step time including KV traffic at (ctx, concurrency)
```

MFU (compute efficiency) and BWE (bandwidth efficiency) live in
`data/hardware.json` as ranges per runtime — see "Worked example"
below for why bands are wide. Roofline reference: Williams,
Waterman, Patterson, *Communications of the ACM* 52(4), 2008.

## Hardware coverage (`data/hardware.json`)

| Card | PCI device ID | Die | VRAM | Bandwidth | FP16/BF16 | INT8 |
|---|---|---|---|---|---|---|
| Arc B580 | `0xe20b` | BMG-G21 | 12 GB | 456 GB/s | 26.9 TF | 233 TOPS |
| Arc Pro B50 | `0xe220` ¹ | BMG-G21 (LP) | 16 GB | 224 GB/s | 21.30 TF | 170 TOPS |
| Arc Pro B60 | `0xe211` ¹ | BMG-G21 | 24 GB | 456 GB/s | 24.56 TF | 197 TOPS |
| Arc Pro B65 | `0xe221` ¹ | BMG-G31 (cut) | 32 GB | 608 GB/s | 24.56 TF | 197 TOPS |
| Arc Pro B70 | `0xe223` | BMG-G31 (full) | 32 GB | 608 GB/s | 45.88 TF | 367 TOPS |

¹ Provisional — derived from the xe kernel driver's `INTEL_BMG_IDS`
list; not yet verified by `lspci` on physical hardware.

Each row in the JSON cites Intel's SKU spec page directly. The PCI
device ID is what `xpu-smi discovery` and `lspci -d 8086:` show in
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
    intel/vllm:<tag>-xpu \
    vllm serve <checkpoint> \
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
