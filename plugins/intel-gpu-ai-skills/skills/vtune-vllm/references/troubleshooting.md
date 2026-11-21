# Troubleshooting: VTune GPU-Hotspots with vLLM on Xe GPU

## Quick Symptom Index

| Symptom | Section |
|---------|---------|
| GPU Time = 0 in results | §1 |
| Empty GPU Timeline tab | §2 |
| Huge CPU time, tiny GPU fraction | §3 |
| Truncated / incomplete kernel records | §4 |
| Result file is 10+ GB | §5 |
| ITT calls do nothing (no-ops) | §6 |
| `vtune -command` fails with "no collection" | §7 |
| Unreadable kernel names (`<unknown>`) | §8 |
| EU stall metrics missing | §9 |
| Multi-rank: one GPU rank not profiled | §10 |

---

## §1 — GPU Time = 0

**Root cause**: Collection paused before GPU kernels completed execution.

```
Timeline:  [resume]  ... GPU work submitted ...  [pause]  ... GPU kernels complete
                                                  ^             ^
                                             VTune stops    Too late!
                                             capturing      Not captured.
```

**Fix**:
```python
model.forward(input_ids)        # Submit GPU work
torch.xpu.synchronize()         # Wait for ALL kernels to complete
itt_pause()                     # Now safe to pause
```

**Verification**: After adding sync, GPU Time in Summary tab should appear
and roughly equal your ROI duration multiplied by GPU utilization.

---

## §2 — Empty GPU Timeline

**Root cause**: Level Zero tracing layer conflict — VTune cannot intercept
kernel submissions.

**Diagnosis**:
```bash
env | grep -E "ZE_|LEVEL_ZERO|PTI_|GPU_DEVICE_ORDINAL"
```

**Fix**: Unset conflicting env vars before launching VTune:
```bash
unset ZE_ENABLE_TRACING_LAYER
unset ZE_LOADER_LAYERS_ENABLE
unset PTI_ENABLE_COLLECTION
unset PTI_ENABLE_RUNTIME_TRACING

# Then launch:
vtune -collect gpu-hotspots -start-paused -result-dir ./out \
  -- python serve.py
```

**Also check**: Intel Extension for PyTorch (IPEX) may set ZE vars internally.
Launch via a clean environment:
```bash
env -i HOME=$HOME PATH=$PATH \
    LD_LIBRARY_PATH=/opt/intel/oneapi/vtune/latest/lib64:$LD_LIBRARY_PATH \
    vtune -collect gpu-hotspots -start-paused -result-dir ./out \
    -- python serve.py
```

**Secondary causes**:
- GPU driver incompatible with VTune version → upgrade VTune or driver
- Missing `/dev/dri` access → add user to `render` group (see §Environment Checklist)
- ZE_AFFINITY_MASK set to wrong tile → unset or correct the mask

---

## §3 — Huge CPU Time, Tiny GPU Fraction

**Root cause**: ROI is too wide — includes Python overhead, tokenization,
HTTP handling, asyncio event loop.

**Diagnosis**: In Bottom-up tab, sort by CPU Time descending. If Python
interpreter frames or asyncio/uvicorn frames dominate, your ROI is too wide.

**Fix — move ROI inward**:
```
TOO WIDE:   AsyncLLMEngine.generate()    <- includes HTTP, tokenizer, scheduling
BETTER:     LLMEngine.step()             <- includes scheduling
RECOMMENDED: Worker.execute_model()      <- GPU worker only
TIGHTEST:   model.forward()             <- pure GPU compute
```

For GPU-Hotspots analysis, `Worker.execute_model` is the sweet spot:
tight enough to exclude CPU noise, wide enough to capture full batch lifecycle.

---

## §4 — Truncated / Incomplete Kernels

**Root cause**: `pause()` called while GPU command queue still has in-flight work.

**Pattern to avoid**:
```python
# WRONG:
model.forward(input_ids)   # Async submission
for token in decode_loop:  # More async submissions
    model.forward(...)
itt_pause()                # Paused while kernels still executing!
```

**Correct pattern**:
```python
# CORRECT:
model.forward(input_ids)
for token in decode_loop:
    model.forward(...)
torch.xpu.synchronize()    # Drain ENTIRE queue
itt_pause()                # All kernels now captured
```

**Note**: `xpu.synchronize()` is device-global — it waits for all streams
on the XPU device, not just the default stream. This is what you want.

---

## §5 — Result File Too Large (10+ GB)

**Target size**: 100 MB – 2 GB for a meaningful short workload.

**Reduction strategies**:

> The CPU-collector knobs `enable-stack-collection` and `sampling-interval`
> are NOT valid for `gpu-hotspots`. Verify available knobs on your install
> with `vtune -help collect gpu-hotspots`.

```bash
# 1. Skip Level Zero / SYCL API timeline if you only care about kernels
vtune -collect gpu-hotspots -start-paused \
      -knob collect-programming-api=false \
      ...

# 2. Skip H2D / D2H transfer tracking
vtune -collect gpu-hotspots -start-paused \
      -knob collect-host-gpu-data-transfers=false \
      ...

# 3. Increase GPU EU sampling interval (coarser but smaller)
vtune -collect gpu-hotspots -start-paused \
      -knob gpu-sampling-interval=5 \    # 5ms instead of 1ms default
      ...

# 4. Reduce profiled requests (10-20 is usually enough)
python benchmark_with_vtune.py --profile 15   # not 200

# 5. Profile a single phase only
# Use prefill-only or decode-only patch from vllm-integration.md
```

---

## §6 — ITT Calls Do Nothing

**Symptom**: `itt_resume()` and `itt_pause()` are called without error,
but collection doesn't change — VTune collects continuously or stays paused.

**Diagnosis script**:
```python
from vtune_itt import itt_available
import os, ctypes.util

print(f"ITT available: {itt_available()}")
print(f"VTUNE_PROFILER_DIR: {os.environ.get('VTUNE_PROFILER_DIR', 'not set')}")
lib = ctypes.util.find_library("ittnotify")
print(f"ldconfig libittnotify: {lib}")
for p in ["/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so",
          "/opt/intel/vtune_profiler/lib64/libittnotify.so"]:
    print(f"  {p}: {'EXISTS' if os.path.exists(p) else 'MISSING'}")
```

**Common causes and fixes**:

1. **Library not found** → Set `VTUNE_PROFILER_DIR` or add to `LD_LIBRARY_PATH`:
   ```bash
   export LD_LIBRARY_PATH=/opt/intel/oneapi/vtune/latest/lib64:$LD_LIBRARY_PATH
   ```

2. **`-start-paused` not used** → Without this flag, VTune ignores ITT resume/pause.
   Always include `-start-paused` in the vtune command.

3. **vLLM launched outside VTune** → VTune only instruments processes it spawns
   (or attaches to via `-target-pid`). If vLLM was started separately, ITT is
   not functional unless VTune attached explicitly.

4. **LD_PRELOAD conflict** → Check `echo $LD_PRELOAD`. VTune injects `libittnotify`
   via LD_PRELOAD; a conflicting preload can prevent injection.

---

## §7 — `vtune -command` Fails

**Error**: `Error: No active collection found at <result-dir>`

**Causes and fixes**:

1. **Path mismatch** — must use exact same path as `-result-dir` at launch:
   ```bash
   # Use absolute paths to avoid ambiguity:
   RESULT=$(realpath ./vtune_results/run_001)
   vtune -collect gpu-hotspots -start-paused -result-dir $RESULT -- ...
   vtune -command resume -r $RESULT   # same $RESULT variable
   ```

2. **vLLM crashed at startup** — check the vLLM process is still running:
   ```bash
   ps aux | grep vllm
   vtune -command status -r $RESULT   # prints current collection state
   ```

3. **Collection already stopped** — `vtune -command stop` is irreversible.
   Verify status before issuing commands:
   ```bash
   vtune -command status -r $RESULT
   ```

---

## §8 — Unreadable Kernel Names

**Symptom**: Timeline shows `<unknown>` or hex offsets instead of
`xetla_gemm_f16_...`, `esimd_attention_...`.

**Cause**: AOT-compiled XPU kernels without debug symbols.

**Partial fix** — enable JIT compilation with debug info:
```bash
export TORCH_JIT_LOG_LEVEL=1
export IPEX_FP64_MATH_MODE=0
```

**Better fix** — use VTune's "Associate with Source" feature and provide
the build directory of PyTorch-XPU / IPEX:
```
VTune GUI -> Result -> Configure Analysis -> Search Directories
Add: /path/to/pytorch/build, /path/to/ipex/build
```

**For custom SYCL kernels** — compile with:
```bash
icpx -fsycl -gline-tables-only -O2 -o kernel.so kernel.cpp
```

---

## §9 — EU Stall Metrics Missing

**Symptom**: "EU Array Stalled %" column is greyed out or zero in Bottom-up tab.

**Cause**: Default GPU-Hotspots uses sampling mode, not instruction-level tracing.

**Fix** — use source-analysis profiling mode:
```bash
vtune -collect gpu-hotspots \
      -start-paused \
      -knob profiling-mode=source-analysis \
      -result-dir ./vtune_xe_stalls \
      -- python serve.py
```

**Caveat**: Source-analysis requires kernels compiled with debug info.
Pre-compiled PyTorch XPU kernels support limited EU stall analysis.
This mode is most useful for custom SYCL/OpenCL kernels in vLLM plugins.

---

## §10 — Multi-Rank: One GPU Rank Not Profiled

**Setup**: vLLM with `--tensor-parallel-size 2` (or more) spawns one process per rank.

**Problem**: Only rank 0 shows GPU activity in VTune.

**Cause A — ITT scope**: `ittapi.resume()` only resumes the calling process.
If only rank 0 calls resume, rank 1 stays paused.

**Fix A — per-rank ITT**: Each rank calls ITT in its `execute_model`:
```python
def execute_model(self, req):
    itt_resume()                   # This rank only
    result = _orig(self, req)
    torch.xpu.synchronize()
    itt_pause()                    # This rank only
    return result
```

**Cause B — external command scope**: `vtune -command resume` resumes ALL ranks
under the VTune umbrella — this is correct if all ranks were started by VTune.
If ranks were spawned via a Python multiprocessing method VTune doesn't track,
they won't be instrumented.

**Fix B**: Always start the top-level process under VTune (not a sub-subprocess).
vLLM's `--tensor-parallel-size` uses `ray` or `torch.multiprocessing` — VTune
tracks both when the parent process is launched under VTune.

---

## Environment Pre-Flight Checklist

Run this before every profiling session:

```bash
#!/bin/bash
echo "=== VTune Environment Check ==="

# 1. oneAPI sourced
source /opt/intel/oneapi/setvars.sh 2>/dev/null || source /opt/intel/oneapi/vtune/latest/env/vars.sh 2>/dev/null
echo "VTUNE_PROFILER_DIR: ${VTUNE_PROFILER_DIR:-NOT SET}"

# 2. No ZE conflicts
BAD_VARS=$(env | grep -E "^(ZE_ENABLE_TRACING|ZE_LOADER_LAYERS|PTI_ENABLE)" | head -5)
if [ -n "$BAD_VARS" ]; then
    echo "WARNING: Conflicting env vars found:"
    echo "$BAD_VARS"
    echo "Run: unset ZE_ENABLE_TRACING_LAYER ZE_LOADER_LAYERS_ENABLE"
else
    echo "ZE env vars: clean"
fi

# 3. GPU accessible
echo "GPU devices:"
sycl-ls 2>/dev/null | grep -i "gpu" || echo "  sycl-ls not found or no GPU"
python3 -c "import torch; print(f'  XPU count: {torch.xpu.device_count()}')" 2>/dev/null

# 4. VTune version
vtune --version 2>/dev/null | head -1 || echo "vtune not in PATH"

# 5. libittnotify
LIB=$(ldconfig -p 2>/dev/null | grep ittnotify | head -1)
echo "libittnotify (ldconfig): ${LIB:-NOT FOUND}"
for P in "/opt/intel/oneapi/vtune/latest/lib64/libittnotify.so" \
         "/opt/intel/vtune_profiler/lib64/libittnotify.so"; do
    [ -f "$P" ] && echo "  Found at: $P"
done

# 6. GPU device permissions
ls -la /dev/dri/render* 2>/dev/null | head -3
id | grep -o "render" && echo "  User in render group: YES" || \
    echo "  WARNING: User not in render group. Run: sudo usermod -aG render $USER && newgrp render"

echo "=== Check complete ==="
```

---

## VTune GUI Validation Checklist

After collection, open: `vtune-gui <result-dir>`

```
Summary tab:
  [ ] GPU Time > 0
  [ ] GPU Time matches expected ROI duration (±20%)
  [ ] GPU utilization percentage shown

Bottom-up tab (sort by GPU Time descending):
  [ ] Top kernels are GEMM or attention (not Python or asyncio)
  [ ] Kernel names are readable (not <unknown>)
  [ ] No single outlier kernel taking >80% (unless expected)

Timeline tab:
  [ ] GPU activity lanes show discrete bursts (not flat/empty)
  [ ] Bursts occur only during ROI window (not before/after)
  [ ] Gaps between bursts correspond to between-request idle (expected)
  [ ] No long continuous CPU thread blocking GPU submissions

Platform tab:
  [ ] Level Zero queue depth > 0 during bursts
  [ ] Command list submissions visible

For prefill/decode separation:
  [ ] Task domains "vllm.prefill" and "vllm.decode" visible in Timeline
  [ ] Prefill shows higher EU utilization than decode (expected for LLM)
  [ ] Decode shows higher memory bandwidth utilization (expected)

Red flags requiring action:
  [!] GPU Time = 0              -> Add torch.xpu.synchronize() before pause
  [!] Empty GPU lane            -> Check ZE env var conflicts (§2)
  [!] Python frames dominate    -> Move ROI to Worker.execute_model (§3)
  [!] All kernels at <1us       -> Sync issue, kernels captured but truncated (§4)
  [!] Result > 10 GB            -> Reduce requests or disable stack collection (§5)
```

---

## §11 — Healthy-Trace Cheat Sheet (BMG / Xe + vLLM)

The earlier checklist asks for "GEMM/attention" at the top. This section is
specific: what the top-of-pareto, EU/BW ranges, and red-flag patterns
*actually* look like for an `--enforce-eager` Llama-class run on BMG. Use it
to decide whether a trace is healthy in under a minute.

> **vLLM 0.33 (`vllm/vllm-openai-xpu`), Qwen3-8B on Arc Pro B70, measured.** Kernel
> names differ from the IPEX-era names below: all linear layers are one oneDNN
> `gemm_kernel` (decode: 85% of GPU time, 145 calls per step = 4 per layer × 36 +
> lm_head), attention is `XeFMHAFwdSplitKV*` / `cutlass::fmha::*` (7%), the sampler is
> `top_k_top_p_kernel` (1.1%), then Triton `triton_*rms_norm*`, `*fused_mul_silu*`,
> `reshape_and_cache_flash*` (≤ 0.8% each). One `gemm_kernel` row at 85% is healthy
> here — it is every layer's GEMV under one name.
>
> Only valid with graphs off (`--enforce-eager` or `cudagraph_mode=NONE`). In the
> default graph mode the decode kernels are missing from the table (SKILL.md §0).
>
> Sampled metrics at 1 ms on the decode GEMV: XVE Active ~5%, Stalled ~57%, Idle ~38%,
> read bandwidth ~394 GB/s (65% of 608) — while bytes ÷ kernel time is ~580 GB/s (95%).
> Judge bandwidth from bytes ÷ time, not from the sampled column.

### Prefill step (compute-bound, expected)

Top hotspots (Bottom-up, sort by GPU Time desc):
  1.  `xetla_gemm_*`            — QKV projection, FFN up/down, LM head
  2.  `flash_attn_varlen_*`     OR `xetla_paged_attention_*`
  3.  `xetla_gemm_*`            — second-largest GEMM (FFN gate or MLP)
  4.  `_layer_norm_*`           OR `_rms_norm_*`
  5.  `_silu_and_mul_*`         OR similar activation fuse

Metric ranges:
  EU Active                    50 – 80 %
  EU Stall                     10 – 25 %
  GPU Memory BW Used           30 – 60 %
  GPU Time per step (BS=1, 512 tok)  30 – 80 ms

Red flags:
  [!] `aten::copy_*` in top 5             → KV-cache layout mismatch / dtype
                                            mismatch on prompt feed
  [!] `_sycl_native_*` shadowing GEMM     → AOT pruning failed; recompile or
                                            check IPEX install
  [!] Single `xetla_gemm_*` >70 % of step → batch shape stuck on a slow path
                                            (IPEX-era kernels only; on vLLM 0.33
                                            one `gemm_kernel` covers all layers)
  [!] EU Active < 30 %                    → ROI captured warmup or stalled
                                            kernel; verify via Timeline tab

### Decode step (BW-bound, expected)

Top hotspots:
  1.  `xetla_gemv_*`            OR `_decode_gemm_*` (small-K GEMMs)
  2.  `_decode_attn_*`          OR `xetla_paged_attention_decode_*`
  3.  `xetla_gemv_*`            — second projection
  4.  Sampling kernels          (`_top_k_*`, `_softmax_*`, `_argmax_*`)
  5.  `_layer_norm_*`           OR `_rms_norm_*`

Metric ranges:
  EU Active                    10 – 30 %     ← LOW IS CORRECT
  EU Stall                     40 – 70 %
  GPU Memory BW Used           70 – 95 %     ← SHOULD BE HIGH
  GPU Time per step (BS=1)     15 – 40 ms

Red flags:
  [!] EU Active > 50 %                    → suspicious; either trace captured
                                            prefill mislabeled as decode, or
                                            BS jumped > 1 unexpectedly
  [!] GPU Memory BW < 50 %                → (compute it as bytes ÷ GEMV time; the
                                            sampled column reads low on sub-ms
                                            kernels) BW-bound work isn't fully feeding
                                            the EUs; KV-cache layout, sampler
                                            re-upload, or BW contention with
                                            another tile
  [!] Long single GEMV in decode          → batch fell to 1; check
                                            `--max-num-seqs`, scheduler logs
  [!] H2D / D2H transfers per step > 4    → sampler params / block-table
                                            re-uploaded each step; check
                                            sampling_params caching
  [!] `aten::index_*` or `aten::scatter_*` in top 10
                                          → KV-cache page indexing on host;
                                            paged-attention path not taken

### Metric → vLLM root-cause map (quick reference)

| VTune metric pattern              | Most likely vLLM root cause                    |
|-----------------------------------|------------------------------------------------|
| High EU stall + low BW            | Attention kernel bottleneck — try paged-attn variant or different block size |
| Low EU active + high BW           | Decode is BW-bound — expected; consider FP8 KV / quantization for headroom |
| Many H2D copies/step              | Block table / sampling params re-uploaded each step — cache them |
| Single long GEMM dominates decode | IPEX-era: batch fell to 1 — check `--max-num-seqs`. vLLM 0.33: normal (one `gemm_kernel` name for all layers) |
| Kernel time ≪ window in default mode | XPU graph replays are not listed — not an idle GPU (SKILL.md §0) |
| `<unknown>` symbols in top 10     | AOT kernels missing debug info — see §8 |
| `aten::copy_*` in prefill         | Dtype/layout mismatch on prompt feed — verify input is bf16 contiguous |
| Big kernel only in first profiled step | JIT first-touch leaked into ROI — add 1-2 throwaway requests after `vtune -command resume` |

### Phase ratios at a glance (steady state, BS=1)

  prefill_GPU_time : decode_GPU_time_per_token  ~= seq_len / 8  (rough)

  Example: 512-token prompt, 256-token response:
    prefill   ~ 50 ms          (one step)
    decode    ~ 25 ms × 256 = 6.4 s   (256 steps)
    ratio     prefill ≈ 0.8 % of total GPU time → decode dominates the trace,
              which is why decode-only profiling needs a tighter ROI than
              prefill-only does.
