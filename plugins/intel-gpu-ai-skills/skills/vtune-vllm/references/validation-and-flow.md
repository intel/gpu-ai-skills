================================================================================
  VTune Conditional ROI Profiling for vLLM
  Captured-Flow + Pre-flight Validation + Post-run Verification
================================================================================

This document explains, step by step, HOW a VTune Region-of-Interest (ROI) is
captured for a vLLM workload on Intel Xe GPU (BMG / PVC), what each stage
contributes, and the explicit checks that prove the capture is correct.

Read this when:
  - Setting up a new profiling host or a new vLLM version.
  - You opened a result and you are not sure whether the ROI is "good".
  - A reviewer asks "how do you know this trace measures what you claim?"

Pair this file with:
  - SKILL.md                          (commands + cheat sheet)
  - references/vllm-integration.md    (patch patterns)
  - references/troubleshooting.md     (failure modes)


--------------------------------------------------------------------------------
SECTION A - End-to-end flow (what actually happens, in order)
--------------------------------------------------------------------------------

Wall-clock                  Process / actor                     Collector state
--------------------------------------------------------------------------------
T0   user runs              shell                               (no collector)
       run_vtune_vllm.sh

T0+  shell exec's `vtune    vtune driver process                ATTACHED, PAUSED
       -collect gpu-hotspots                                    (because of
       -start-paused`                                            -start-paused)

T1   vtune fork/exec's      python (api_server parent)          PAUSED
       python -m vllm.
       entrypoints.openai.
       api_server

T1+  vLLM v1 spawns         EngineCore subprocess               PAUSED (umbrella)
       EngineCore subproc

T1++ EngineCore spawns      Worker (GPU rank 0)                 PAUSED (umbrella)
       Worker process(es)

T2   model load,            Worker                              PAUSED
       KV alloc,                                                (capture excluded)
       AOT/JIT first-touch                                      <- THIS IS WHY
       compiles                                                    -start-paused
                                                                   matters

T3   warmup requests        Worker                              PAUSED
       (10x throwaway)                                          (kernel JIT cache
                                                                 warms here)

T4   3-second drain         (idle)                              PAUSED
       (sleep 3)                                                (queue empties)

T5   benchmark script:                                          RESUMED
       `vtune -command       vtune driver                       <- ROI START
        resume -r $RESULT`

T6   benchmark loop:        Worker.execute_model                COLLECTING
       N profiled requests,    -> model.forward()                GPU samples,
       prefill+decode steps    -> xpu kernels submitted          L0 API timeline,
                                                                 EU sampling

T6a  per-step (optional):   Worker patch                        COLLECTING
       ITT task_begin/end                                       (phase tags
       with phase                                                appear in
       classification                                            Tasks tab)

T7   benchmark script:      Worker (last step finishes)         RESUMED
       sync barrier            -> torch.xpu.synchronize()        (drain queue)

T8   benchmark script:      vtune driver                        PAUSED
       `vtune -command                                          <- ROI END
        pause -r $RESULT`

T9   benchmark script:      vtune driver                        FINALIZING
       `vtune -command                                          (write traces,
        stop -r $RESULT`                                          resolve symbols,
                                                                  close result)

T10  result usable          on disk: $RESULT/                   STOPPED
       open: vtune-gui                                          (ready for
       or: vtune -report                                          analysis)


--------------------------------------------------------------------------------
SECTION B - Why each stage is necessary (the "why" behind the "what")
--------------------------------------------------------------------------------

-start-paused (T0+):
  Without it, VTune begins collecting at process exec. T2 (model load + JIT)
  produces 10-30 s of GPU activity that is NOT inference. It dominates the
  result by mass and pollutes hotspot rankings. -start-paused throws this
  noise away.

Process tree under one vtune driver (T1, T1+, T1++):
  vLLM v1 splits the engine across processes. Only descendants of the vtune
  driver are "under the umbrella" and reachable by `vtune -command resume`.
  This is why we launch via `vtune -- python ...` (NOT `vtune -target-pid`
  attaching after the fact) - to guarantee EngineCore and Worker subprocesses
  are tracked.

Warmup PAUSED (T3):
  First runs through any new shape compile SYCL/IPEX kernels. Those compile
  events appear as ENORMOUS unique kernels in the trace (sometimes >50 % of
  GPU time) and crowd out the steady-state hotspots. PAUSED warmup keeps
  the first-touch noise out.

Drain (T4):
  XPU work is asynchronous. Even after the warmup loop returns to Python,
  GPU command lists may still be executing. A 3 s sleep + idle gives the
  queue time to drain so it is not captured as part of the ROI.

`vtune -command resume` not ITT, for split topology (T5):
  ITT (`__itt_resume`) is per-process. The vLLM v1 worker is a SUBPROCESS,
  so an ITT call in the API parent does not reach it. The external command
  resumes the entire process tree at once - correct for our split topology.
  Use ITT only for in-worker phase tagging or fine-grained per-rank control.

torch.xpu.synchronize() before pause (T7):
  This is the SINGLE most common reason a trace looks empty. XPU kernel
  submissions return immediately; the kernels keep executing on the GPU.
  If `pause` fires before they finish, VTune stops sampling and misses
  the actual GPU work even though it was clearly "submitted in the ROI".
  The sync barrier guarantees every kernel submitted inside the ROI has
  completed (and been sampled) before the ROI closes.

`-command stop` (T9):
  Pause leaves the result open. Stop finalizes - writes index, resolves
  symbols, closes file handles. Without it, the result directory is in
  an indeterminate state and `vtune-gui` may refuse to open it.


--------------------------------------------------------------------------------
SECTION C - Pre-flight VALIDATION (run BEFORE the profiling run)
--------------------------------------------------------------------------------

If ANY of (1)..(5) fail, abort. The collection will produce a useless or
empty result.

(1) oneAPI sourced:
    $ command -v vtune
    expect: /opt/intel/oneapi/vtune/latest/bin64/vtune
    $ vtune --version
    expect: Intel(R) VTune(TM) Profiler 2024.x or later

(2) No tracing-layer conflict:
    $ env | grep -E '^(ZE_ENABLE_TRACING|ZE_LOADER_LAYERS|PTI_ENABLE)='
    expect: empty output.
    If non-empty -> unset before launching vtune:
      unset ZE_ENABLE_TRACING_LAYER ZE_LOADER_LAYERS_ENABLE \
            PTI_ENABLE_COLLECTION PTI_ENABLE_RUNTIME_TRACING

(3) GPU visible:
    $ sycl-ls | grep -i 'gpu'
    expect: at least one [ext_oneapi_level_zero:gpu]
    $ python3 -c "import torch; print(torch.xpu.device_count())"
    expect: >= 1.

(4) ITT lib resolvable (only matters when using ITT alongside external cmds):
    $ ls /opt/intel/oneapi/vtune/latest/lib64/libittnotify.so
    expect: file exists.

(5) /dev/dri permissions:
    $ id -nG | tr ' ' '\n' | grep -q '^render$' && echo OK || echo "NEEDS render"
    expect: OK.
    If NEEDS render: sudo usermod -aG render $USER && newgrp render

(6) Disk budget:
    $ df -h $(dirname $RESULT_DIR) | tail -1
    Budget per result: 100 MB - 2 GB. Refuse to start if free < 5 GB.

(7) vLLM image / version sanity (only first time on a host):
    $ python -c "import vllm, sys; print(vllm.__version__)"
    expect: >= 0.10 for the v1 paths used by this skill.
    $ python -c "from vllm.v1.worker.gpu_worker import Worker; print(Worker.__module__)"
    expect: vllm.v1.worker.gpu_worker  (no ImportError)


--------------------------------------------------------------------------------
SECTION D - Live VERIFICATION (DURING the run)
--------------------------------------------------------------------------------

While the benchmark loop is running, in another terminal:

  $ vtune -command status -r $RESULT_DIR

  Expected outputs and what they mean:

    "Collection paused"     -> we are in T2/T3/T4 or after T8.
                               GOOD if before T5 or after T8;
                               BAD  if you expect collection.

    "Collection running"    -> we are between T5 and T8.
                               GOOD during the benchmark window.

    "Error: No active       -> vLLM crashed at startup;
     collection found"         OR result-dir path mismatch.
                               Check `ps` for the vllm process and `realpath`
                               of the result dir.

Sanity-check that resume actually fired (GPU is being used):

  $ timeout 5 xpu-smi dump -d 0 -m 0,1,2 -i 500
    expect: >0 % GPU during the profiled window; ~0 % before/after.

If GPU is busy but `status` says paused, an ITT layer or env var is
overriding the external command. See troubleshooting.md section 6.


--------------------------------------------------------------------------------
SECTION E - Post-collection VERIFICATION (proves ROI was captured)
--------------------------------------------------------------------------------

Five checks in order. If any FAILS, do not trust the trace.

CHECK 1 - Result is finalized
  $ ls -la $RESULT_DIR/
  expect: at least  config.cfg, data.0/, log/, runss/  present.
  If only `data.0/` and no `runss/`, you forgot `-command stop`.
  Re-run:  vtune -command stop -r $RESULT_DIR

CHECK 2 - Non-zero GPU time
  $ vtune -report summary -r $RESULT_DIR | grep -i 'GPU Time'
  expect: GPU Time > 0 and within ~20 % of expected ROI duration.
          Example: 50 reqs x (~50 ms prefill + ~5 s decode) ~= 250 s GPU time.
  Zero GPU time -> almost always missing torch.xpu.synchronize() before pause.

CHECK 3 - Hotspots are real GPU kernels, not Python
  $ vtune -report hotspots -r $RESULT_DIR -group-by computing-task \
          -format csv -limit 10
  expect: top entries are GPU kernels: xetla_gemm_*, flash_attn_*, _decode_*
          (IPEX era) or gemm_kernel, XeFMHAFwd*, top_k_top_p_kernel (vLLM 0.33).
  Default (graph) mode lists only kernels outside graph replays; use
  --enforce-eager or cudagraph_mode=NONE for a full decode table.
  If top entries are python::*, asyncio::*, uvicorn::*  ->  ROI was too wide
          (HTTP/Python included). Move to Worker.execute_model patching.

CHECK 4 - ROI duration matches benchmark wall-clock
  Benchmark wall-clock between T5 and T8 vs:
  $ vtune -report summary -r $RESULT_DIR | grep -i 'Elapsed Time'
  expect: VTune elapsed time within +/- 20 % of benchmark wall-clock.
  Larger gap -> warmup contamination, premature pause, or stop command lost.

CHECK 5 - Phase tags present (only if ITT task tagging was enabled)
  $ vtune-gui $RESULT_DIR  ->  Tasks tab
  expect: domain "vllm.prefill" with task_count > 0,
          domain "vllm.decode"  with task_count > 0.
          (If batched ROI is used, expect "vllm.window" instead.)
  Missing -> patch did not load (subprocess didn't inherit), or phase
          classification rejected every step (check classify_step return).


--------------------------------------------------------------------------------
SECTION F - Quantitative health bounds (BMG / Llama-3.1-8B reference)
--------------------------------------------------------------------------------

Numerical ranges a HEALTHY trace shows. Values outside the bounds mean either
the model+config is unusual or the trace itself is wrong; investigate before
drawing conclusions.

PREFILL step (BS=1, seq_len ~512, --enforce-eager):
  GPU Time / step          : 30 - 80 ms
  EU Active                : 50 - 80 %
  EU Stall                 : 10 - 25 %
  GPU Memory BW            : 30 - 60 %
  Top kernel category      : xetla_gemm_* (QKV / FFN), flash_attn_varlen_*

DECODE step (BS=1, --enforce-eager):
  GPU Time / step          : 15 - 40 ms
  EU Active                : 10 - 30 %      (low is correct; decode is BW-bound)
  EU Stall                 : 40 - 70 %
  GPU Memory BW            : 70 - 95 %
  Top kernel category      : xetla_gemv_* OR _decode_attn_*
                             (vLLM 0.33: gemm_kernel ~85%, XeFMHAFwdSplitKV ~7%)
  GPU Memory BW            : compute as weight bytes / GEMV time per step; the
                             sampled VTune value reads low on sub-ms kernels

Per-step ITT/sync overhead (worker):
  itt_resume + itt_pause   : < 5 us
  torch.xpu.synchronize()  : 5 - 50 us
  Total overhead / step    : < 0.5 % of step time

Result file size:
  10-50 profiled requests  : 100 MB - 2 GB
  > 5 GB                   : programming-api timeline likely on; consider
                             -knob collect-programming-api=false


--------------------------------------------------------------------------------
SECTION G - Quick repro recipe (copy / paste)
--------------------------------------------------------------------------------

# 0) Pre-flight (Section C)
unset ZE_ENABLE_TRACING_LAYER ZE_LOADER_LAYERS_ENABLE \
      PTI_ENABLE_COLLECTION PTI_ENABLE_RUNTIME_TRACING
source /opt/intel/oneapi/setvars.sh 2>/dev/null || source /opt/intel/oneapi/vtune/latest/env/vars.sh   # VTune-only install

# 1) Launch vLLM under VTune (paused)
export RESULT_DIR=$(realpath ./vtune_results/run_$(date +%Y%m%d_%H%M%S))
vtune -collect gpu-hotspots -start-paused \
      -knob collect-programming-api=true \
      -result-dir $RESULT_DIR \
      -- python serve_with_vtune.py \
           --model meta-llama/Llama-3.1-8B \
           --port 8000 --enforce-eager \
           --disable-log-requests &

# 2) Wait for ready
until curl -sf http://localhost:8000/health >/dev/null; do sleep 2; done

# 3) Warmup (PAUSED)
python benchmark_with_vtune.py --warmup 10 --profile 0 \
       --result-dir $RESULT_DIR --endpoint http://localhost:8000

# 4) Drain
sleep 3

# 5) Profile (resume / run / pause / stop)
vtune -command resume -r $RESULT_DIR
python benchmark_with_vtune.py --warmup 0 --profile 30 \
       --result-dir $RESULT_DIR --endpoint http://localhost:8000
vtune -command pause  -r $RESULT_DIR
vtune -command stop   -r $RESULT_DIR

# 6) Post-run verification (Section E)
vtune -report summary  -r $RESULT_DIR
vtune -report hotspots -r $RESULT_DIR -group-by computing-task \
      -format csv -limit 10

================================================================================
End of document.
================================================================================
