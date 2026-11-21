#!/usr/bin/env bash
set -euo pipefail

# VTune GPU-Hotspots ROI profiling for vLLM (OpenAI API server).
#
# Phase isolation strategy (vLLM v1, multi-process engine core):
#   1. WORKLOAD SHAPING - INPUT_LEN/OUTPUT_LEN choose the dominant phase:
#        VTUNE_PHASE=prefill INPUT_LEN=2048 OUTPUT_LEN=1   ./run_vtune_vllm.sh
#        VTUNE_PHASE=decode  INPUT_LEN=128  OUTPUT_LEN=512 ./run_vtune_vllm.sh
#
#   2. WORKER PATCH - serve_with_vtune.py monkey-patches
#      vllm.v1.worker.gpu_worker.Worker.execute_model. classify_step buckets
#      each step as prefill | decode | mixed | cache_hit | empty using
#      num_scheduled_tokens; cache_hit / empty are excluded so they cannot
#      pollute the phase being measured. Sentinel-file gating works across
#      the v1 engine subprocess (env-var values are snapshotted at fork time
#      and can't be updated in-flight; a file the child polls can).
#
#   3. ROI MODE (VTUNE_ROI_MODE):
#        window           - default; one resume on first profiled step,
#                           pause at exit; ITT tasks tag every step.
#        per_step         - resume/pause every profiled step.
#        per_step_isolate - per_step semantics, mixed steps rejected too.
#
#   4. Outer `vtune -command resume/pause` brackets the benchmark window
#      so model load + warmup are excluded as a second line of defence.
#
# After collection we emit summary.csv / hotspots.csv / tasks.csv and run a
# 4-check post-run verification (see references/validation-and-flow.md section E).

# ---- Config ----
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Eager mode is recommended for profiling: torch.compile fuses ops into
# Inductor-named compiled regions which obscures kernel attribution. It also
# avoids an llvm-foreach SEGV seen on intel/vllm:0.14.1-xpu + oneAPI 2025.3
# during the SYCL JIT step. Set ENFORCE_EAGER=0 to use compiled mode anyway.
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
VTUNE_PHASE="${VTUNE_PHASE:-both}"
VTUNE_ROI_MODE="${VTUNE_ROI_MODE:-window}"
# TARGET_GPU - explicit GPU adapter selector. Required on multi-GPU hosts
# (e.g., systems with iGPU + BMG dGPU): without it VTune defaults to adapter
# 0 (often the iGPU) and the trace is empty. Set to the BDF of the GPU vLLM
# is using, e.g. TARGET_GPU=0000:03:00.0. Discover via:
#   sycl-ls --verbose | grep -E 'Name|BDF|PCI'
#   ls -l /sys/class/drm/card*/device
#   lspci -nn | grep -iE 'VGA|Display|3D'
# Leave empty to let VTune auto-select (correct only on single-GPU boxes).
TARGET_GPU="${TARGET_GPU:-}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-10}"
DRAIN_SECONDS="${DRAIN_SECONDS:-3}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-5}"

RESULT_ROOT="${RESULT_ROOT:-$(pwd)/vtune_results}"
# Absolute path: vtune -command needs the same path used at -collect time.
RESULT_DIR="$(realpath -m "$RESULT_ROOT")/$(date +%Y%m%d_%H%M%S)_${VTUNE_PHASE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAPPER="$SCRIPT_DIR/serve_with_vtune.py"
if [[ ! -f "$WRAPPER" ]]; then
  echo "ERROR: serve_with_vtune.py not found next to this script ($WRAPPER)"
  exit 1
fi

# Worker-side ROI gate via SENTINEL FILE: serve_with_vtune.py polls
# VTUNE_ROI_GATE for existence on each Worker.execute_model call.
export VTUNE_ROI_GATE="$RESULT_DIR/.roi_gate"
export VTUNE_PHASE
export VTUNE_ROI_MODE
rm -f "$VTUNE_ROI_GATE"   # ensure clean start

# ---- Environment ----
# Always (re-)source setvars.sh with --force. Reason: SETVARS_COMPLETED is an
# environment variable set once at container start (via /etc/environment or
# equivalent), but PATH additions from a prior interactive shell don't survive
# into a subshell launched by `docker exec bash -lc`. Skipping the source
# when SETVARS_COMPLETED=1 leaves `vtune` off PATH and preflight fails.
# setvars.sh is idempotent with --force, so re-sourcing is safe.
#
# We temporarily drop `set -euo pipefail` around the source because setvars.sh
# uses `local`/`return`/`unset` on nonempty vars in ways that a strict shell
# treats as failures — the sourced script then propagates a non-zero return
# and kills our wrapper. Isolating it protects the rest of the script.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
  set +euo pipefail
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -euo pipefail
else
  echo "WARN: /opt/intel/oneapi/setvars.sh not found - trying VTune's own env script"
fi
# A VTune-only install (e.g. the host's /opt/intel/oneapi/vtune mounted into a vLLM
# container) has no setvars.sh; VTune's own vars.sh puts it on PATH.
if ! command -v vtune >/dev/null 2>&1; then
  for _vars in /opt/intel/oneapi/vtune/latest/env/vars.sh /opt/intel/oneapi/vtune/latest/vtune-vars.sh; do
    if [[ -f "$_vars" ]]; then
      set +euo pipefail
      # shellcheck disable=SC1090
      source "$_vars" >/dev/null 2>&1 || true
      set -euo pipefail
      break
    fi
  done
fi

# Load the Worker.execute_model patch in every Python process of the server. vLLM-XPU
# starts the EngineCore with `spawn`, so a patch applied only in the API-server parent
# never reaches the process that runs the GPU work. sitecustomize.py in this folder is
# imported at start-up by every Python interpreter that has it on PYTHONPATH; it applies
# the patch only when VTUNE_VLLM_PATCH=1 and never in `vllm bench` clients.
export VTUNE_VLLM_PATCH=1
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# vLLM env - align with intel/vllm:0.21.0-ubuntu24.04-* defaults.
# HF_HOME: HuggingFace cache. Override if your cache lives elsewhere.
# TORCH_LLM_ALLREDUCE / VLLM_ALLOW_LONG_MAX_MODEL_LEN: recommended by the
#   intel/vllm image maintainers for XPU multi-GPU runs.
# VLLM_WORKER_MULTIPROC_METHOD=spawn: safer under VTune's collector
#   (`fork` inherits the collector's LD_PRELOAD and can confuse ITT).
# VLLM_ENGINE_READY_TIMEOUT_S=1800: vLLM 0.21 cold model load on NFS-mounted
#   HF caches can take 5-10 min; the 600s default trips before weights load.
export HF_HOME="${HF_HOME:-/hf_cache}"
export TORCH_LLM_ALLREDUCE="${TORCH_LLM_ALLREDUCE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
# VTune manages Level Zero tracing itself; manual settings cause empty GPU timelines.
for VAR in ZE_ENABLE_TRACING_LAYER ZE_LOADER_LAYERS_ENABLE \
           PTI_ENABLE_COLLECTION PTI_ENABLE_RUNTIME_TRACING; do
  unset "$VAR"
done

# ---- Pre-flight checks (references/validation-and-flow.md §C) ----
preflight() {
  local fail=0

  # 1. vtune in PATH
  if ! command -v vtune >/dev/null 2>&1; then
    echo "  [FAIL] vtune not in PATH"
    echo "         The oneAPI Base Toolkit alone does NOT include VTune Profiler."
    echo "         Install it from https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html"
    echo "         (or 'apt install intel-oneapi-vtune'). Then:"
    echo "           source /opt/intel/oneapi/setvars.sh --force"
    fail=1
  else
    echo "  [ OK ] vtune: $(command -v vtune)"
  fi

  # 2. No tracing-layer conflicts (already unset above; re-check)
  local bad
  bad=$(env | grep -E '^(ZE_ENABLE_TRACING|ZE_LOADER_LAYERS|PTI_ENABLE)=' || true)
  if [[ -n "$bad" ]]; then
    echo "  [FAIL] Conflicting env vars still set:"
    echo "$bad" | sed 's/^/         /'
    fail=1
  else
    echo "  [ OK ] No ZE/PTI tracing conflicts"
  fi

  # 3. GPU visible. Try sycl-ls first; fall back to torch.xpu (covers cases
  #    where sycl-ls isn't on PATH but Level Zero / IPEX are usable).
  if command -v sycl-ls >/dev/null 2>&1 && sycl-ls 2>/dev/null | grep -qiE '\[(level_zero|opencl):gpu\]'; then
    GPU_LINE=$(sycl-ls 2>/dev/null | grep -iE '\[(level_zero|opencl):gpu\]' | head -1 | sed 's/^[[:space:]]*//')
    echo "  [ OK ] sycl-ls reports GPU: $GPU_LINE"
  elif python -c 'import torch,sys; sys.exit(0 if torch.xpu.is_available() and torch.xpu.device_count()>0 else 1)' 2>/dev/null; then
    XPU_NAME=$(python -c 'import torch; print(torch.xpu.get_device_name(0))' 2>/dev/null || echo "unknown")
    XPU_CNT=$(python -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null || echo "?")
    echo "  [ OK ] torch.xpu reports $XPU_CNT device(s): $XPU_NAME"
  else
    echo "  [WARN] No XPU detected via sycl-ls or torch.xpu (continuing)"
  fi

  # 4. /dev/dri access. Root has full access regardless of group; otherwise we
  #    require either the render group OR the video group (the BMG XPU image
  #    creates renderD* under group 992 / often `video` inside the container).
  if [[ "$(id -u)" -eq 0 ]]; then
    echo "  [ OK ] Running as root (full /dev/dri access)"
  elif id -nG 2>/dev/null | tr ' ' '\n' | grep -qE '^(render|video)$'; then
    echo "  [ OK ] User in render/video group"
  elif ls /dev/dri/renderD* >/dev/null 2>&1 \
       && [[ -r "$(ls /dev/dri/renderD* | head -1)" ]]; then
    echo "  [ OK ] /dev/dri/renderD* readable by current user"
  else
    echo "  [WARN] User not in render/video group and /dev/dri may be denied"
  fi

  # 5. Free disk
  local free_gb
  free_gb=$(df -PB1G "$RESULT_ROOT" 2>/dev/null | awk 'NR==2 {print $4+0}')
  if [[ -n "$free_gb" && "$free_gb" -ge "$MIN_FREE_DISK_GB" ]]; then
    echo "  [ OK ] Free disk on $RESULT_ROOT: ${free_gb} GB (>= ${MIN_FREE_DISK_GB} GB)"
  else
    echo "  [FAIL] Free disk on $RESULT_ROOT: ${free_gb:-?} GB < ${MIN_FREE_DISK_GB} GB"
    fail=1
  fi

  # 6. vLLM v1 import works (this skill is v1-only; no fallback)
  if python -c 'from vllm.v1.worker.gpu_worker import Worker' 2>/dev/null; then
    echo "  [ OK ] vllm.v1.worker.gpu_worker.Worker importable"
  else
    echo "  [FAIL] vllm.v1.worker.gpu_worker.Worker not importable"
    echo "         This wrapper is vLLM v1-only. Upgrade to a build with the v1"
    echo "         engine (intel/vllm >= 0.14.1-xpu, upstream vLLM >= 0.10)."
    fail=1
  fi

  # 6b. Intel Metrics Discovery library — required by vtune -collect gpu-hotspots.
  # VTune dlopens libigdmd.so and libmd.so; without them collection dies with
  # "Cannot collect GPU hardware metrics". apt package: intel-metrics-discovery.
  if ldconfig -p 2>/dev/null | grep -qE 'libigdmd\.so'; then
    echo "  [ OK ] libigdmd.so resolvable"
  else
    echo "  [FAIL] libigdmd.so not on the loader path"
    echo "         Install with: apt install -y intel-metrics-discovery"
    echo "         (then: ln -sf libigdmd.so.1 /usr/lib/x86_64-linux-gnu/libigdmd.so"
    echo "          if the unversioned symlink is missing)"
    fail=1
  fi

  # 6c. VTune version check — on BMG hardware, VTune 2025.x fails to collect
  # GPU metrics ("Cannot collect GPU hardware metrics because neither
  # libigdmd.so nor libmd.so was found") even with intel-metrics-discovery
  # installed. VTune 2026.0 has been observed to succeed on the same BMG HW.
  # Warn if the resolved vtune binary is older than 2026 AND the GPU appears
  # to be BMG-class (device id 0xe223 or the xe kernel driver).
  if command -v vtune >/dev/null 2>&1; then
    VTV_MAJOR=$(vtune --version 2>&1 | head -1 | grep -oE '[0-9]{4}' | head -1 || echo "0")
    if [[ "$VTV_MAJOR" -lt 2026 ]] 2>/dev/null; then
      GPU_IS_BMG_LIKE=0
      if readlink /sys/class/drm/renderD128/device/driver 2>/dev/null | grep -q '/xe$'; then
        GPU_IS_BMG_LIKE=1
      fi
      if [[ "$GPU_IS_BMG_LIKE" -eq 1 ]]; then
        echo "  [WARN] VTune $VTV_MAJOR + BMG-class GPU (xe driver) is a known"
        echo "         broken combination. Collection dies with 'Cannot collect"
        echo "         GPU hardware metrics'. Prior runs on this hardware only"
        echo "         succeeded with VTune 2026.0. Options:"
        echo "           - install VTune 2026.0 from the standalone Intel installer"
        echo "           - use unitrace instead for BMG kernel attribution"
      fi
    fi
  fi

  # 7. Nothing already on $PORT
  if curl -sf "http://$HOST:$PORT/health" > /dev/null 2>&1; then
    echo "  [FAIL] http://$HOST:$PORT already serving (stale vLLM?)"
    fail=1
  else
    echo "  [ OK ] Port $PORT free"
  fi

  return $fail
}

mkdir -p "$RESULT_ROOT"
echo "===== Pre-flight checks ====="
if ! preflight; then
  echo "Pre-flight checks FAILED. Refusing to start."
  exit 1
fi
echo

mkdir -p "$RESULT_DIR"
echo "Result dir: $RESULT_DIR"
echo "Phase: $VTUNE_PHASE  ROI mode: $VTUNE_ROI_MODE"
echo "Workload: input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS conc=$MAX_CONCURRENCY"

# Stamp metadata for future-you opening this result later.
cat > "$RESULT_DIR/metadata.json" <<META
{
  "model": "$MODEL",
  "dtype": "$DTYPE",
  "max_model_len": $MAX_MODEL_LEN,
  "gpu_memory_utilization": $GPU_MEM_UTIL,
  "enforce_eager": $ENFORCE_EAGER,
  "phase": "$VTUNE_PHASE",
  "roi_mode": "$VTUNE_ROI_MODE",
  "input_len": $INPUT_LEN,
  "output_len": $OUTPUT_LEN,
  "num_prompts": $NUM_PROMPTS,
  "max_concurrency": $MAX_CONCURRENCY,
  "warmup_prompts": $WARMUP_PROMPTS,
  "drain_seconds": $DRAIN_SECONDS,
  "host": "$HOST",
  "port": $PORT,
  "started_at": "$(date -Iseconds)"
}
META

# ---- Step 1: Launch vLLM (via wrapper) under VTune, paused ----
EXTRA_ARGS=()
[[ "$ENFORCE_EAGER" == "1" ]] && EXTRA_ARGS+=(--enforce-eager)

# Multi-GPU hosts: pin the collector to the adapter vLLM uses. Without this
# the trace is empty (Elapsed Time = 0, "no GPU-side trace data" warning).
VTUNE_KNOBS=(-knob collect-programming-api=true)
if [[ -n "$TARGET_GPU" ]]; then
  VTUNE_KNOBS+=(-knob target-gpu="$TARGET_GPU")
  echo "Pinning VTune to GPU adapter: $TARGET_GPU"
else
  echo "WARN: TARGET_GPU not set. On multi-GPU hosts VTune may profile the"
  echo "      wrong adapter and produce an empty trace. If summary.csv shows"
  echo "      'Elapsed Time = 0', set TARGET_GPU=<BDF> and re-run."
fi

vtune -collect gpu-hotspots \
      -start-paused \
      "${VTUNE_KNOBS[@]}" \
      -result-dir "$RESULT_DIR" \
      -- python "$WRAPPER" \
            --model "$MODEL" \
            --host "$HOST" \
            --port "$PORT" \
            --dtype "$DTYPE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            "${EXTRA_ARGS[@]}" &
VTUNE_PID=$!

cleanup() {
  echo "Cleaning up (vtune pid=$VTUNE_PID)..."
  rm -f "$VTUNE_ROI_GATE" 2>/dev/null || true
  vtune -command pause -r "$RESULT_DIR" 2>/dev/null || true
  vtune -command stop  -r "$RESULT_DIR" 2>/dev/null || true
  kill "$VTUNE_PID" 2>/dev/null || true
  wait "$VTUNE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- Step 2: Wait for /health ----
# 900 iterations * 2s = 30 min. vLLM 0.21 cold model-load on this image
# routinely exceeds 10 min; NFS-mounted /hf_cache pushes it higher. Match
# the VLLM_ENGINE_READY_TIMEOUT_S bump we exported above.
echo "Waiting for vLLM server on http://$HOST:$PORT ..."
for _ in $(seq 1 900); do
  if curl -sf "http://$HOST:$PORT/health" > /dev/null; then
    echo "Server ready."
    break
  fi
  sleep 2
done
curl -sf "http://$HOST:$PORT/health" > /dev/null || { echo "Server failed to come up"; exit 1; }

# ---- Step 3: Warmup (VTune still paused, gate still closed) ----
echo "Warmup: $WARMUP_PROMPTS prompts (not profiled)..."
vllm bench serve \
  --backend openai \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$WARMUP_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  > "$RESULT_DIR/warmup.log" 2>&1
sleep "$DRAIN_SECONDS"

# ---- Step 4: Open inner gate -> Resume outer -> Benchmark -> Pause -> Close gate ----
# Order matters: open gate first so the very first benchmark batch is
# already eligible for inner ITT control, then resume the outer collection.
touch "$VTUNE_ROI_GATE"
echo "Resuming VTune collection..."
vtune -command resume -r "$RESULT_DIR"

BENCH_T0=$(date +%s)
echo "Benchmarking $NUM_PROMPTS prompts (profiled)..."
vllm bench serve \
  --backend openai \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --save-result \
  --result-dir "$RESULT_DIR" \
  | tee "$RESULT_DIR/bench.log"
BENCH_T1=$(date +%s)
BENCH_ELAPSED=$((BENCH_T1 - BENCH_T0))

echo "Pausing VTune collection..."
# In per_step / per_step_isolate modes the worker's own itt_pause() after the last
# profiled step usually leaves the collector paused already, so this outer pause can
# fail ("not valid now"). That is not an error there; `vtune -command stop` below
# still finalizes. In window mode a failed pause still aborts.
if ! vtune -command pause -r "$RESULT_DIR"; then
  if [[ "$VTUNE_ROI_MODE" == "window" ]]; then
    echo "ERROR: vtune -command pause failed" >&2; exit 1
  fi
  echo "  (pause not accepted: the per-step ROI already paused collection; continuing)"
fi
rm -f "$VTUNE_ROI_GATE"

# ---- Step 5: Finalize ----
echo "Stopping VTune..."
vtune -command stop -r "$RESULT_DIR"

trap - EXIT INT TERM
kill "$VTUNE_PID" 2>/dev/null || true
wait "$VTUNE_PID" 2>/dev/null || true

# ---- Step 6: Headless reports ----
# We emit only the three canonical CSVs (summary.csv, hotspots.csv, tasks.csv).
# Failures are printed inline; if a CSV comes back empty, we delete it — no
# zero-byte files cluttering the result dir.
echo
echo "===== Headless reports ====="
_write_report() {
  local name="$1" out="$2"; shift 2
  # stderr (VTune's progress lines and errors) goes to a temp file outside the result
  # dir; mixing it into the CSV corrupts the header row.
  local err; err=$(mktemp)
  if vtune "$@" -r "$RESULT_DIR" -format csv -csv-delimiter comma > "$out" 2> "$err"; then
    if [[ -s "$out" ]]; then
      echo "  wrote $name"
    else
      rm -f "$out"
      echo "  $name empty — removed"
    fi
  else
    echo "  $name FAILED:"
    grep -v 'Executing actions' "$err" | sed 's/^/    /' | head -5
    rm -f "$out"
  fi
  rm -f "$err"
}

_write_report "summary.csv"  "$RESULT_DIR/summary.csv"  -report summary
# VTune 2026.6 has no `gpu-hotspots`, `tasks` or `top-tasks` report: kernels come from
# `hotspots -group-by computing-task`, ITT tasks from `-group-by task-domain,task`.
# -limit alone keeps unsorted rows, so sort by time.
_write_report "hotspots.csv" "$RESULT_DIR/hotspots.csv" \
  -report hotspots -group-by computing-task -sort-desc "Computing Task:Total Time" -limit 25
_write_report "tasks.csv"    "$RESULT_DIR/tasks.csv"    -report hotspots -group-by task-domain,task

# ---- Step 7: Post-run verification (references/validation-and-flow.md §E) ----
echo
echo "===== Post-run verification ====="
# Disable errexit + pipefail in this block: we use grep|head|awk pipelines
# whose intermediate stages can SIGPIPE legitimately when extracting fields,
# and we don't want a non-zero in any extractor to short-circuit the rest of
# the verification checks. We restore both at the end of the block.
set +e
set +o pipefail
verify_fail=0

# Helper: the value (last numeric field) of the first summary.csv line whose metric
# name matches the regex. summary.csv rows are "<level>,<metric name>,<value>", so the
# first numeric field is the hierarchy level, not the value. SIGPIPE-safe.
extract_first_num() {
  local pat="$1" file="$2"
  awk -F',' -v pat="$pat" '
    tolower($0) ~ tolower(pat) {
      for (i=NF; i>=1; i--) {
        v = $i; gsub(/[" ]/, "", v)
        if (v ~ /^[0-9.]+$/ && v+0 > 0) { print v; exit 0 }
      }
      exit 0
    }' "$file" 2>/dev/null
}

# Check 1: result is finalized (runss/ or data.0/ exists)
if [[ -d "$RESULT_DIR/runss" || -d "$RESULT_DIR/data.0" ]]; then
  echo "  [ OK ] result directory contents present"
else
  echo "  [FAIL] result directory missing data.0/ and runss/"
  verify_fail=1
fi

# Check 2: non-zero GPU time
if [[ -s "$RESULT_DIR/summary.csv" ]]; then
  GPU_TIME=$(extract_first_num 'GPU.Time' "$RESULT_DIR/summary.csv")
  if [[ -n "$GPU_TIME" ]] && awk -v v="$GPU_TIME" 'BEGIN{exit !(v>0)}'; then
    echo "  [ OK ] GPU Time = ${GPU_TIME}"
  else
    echo "  [FAIL] GPU Time = 0 or unparsable - missing torch.xpu.synchronize() before pause?"
    verify_fail=1
  fi
else
  echo "  [WARN] summary.csv empty or missing; cannot check GPU Time"
fi

# Check 3: top hotspot is a GPU kernel, not Python
if [[ -s "$RESULT_DIR/hotspots.csv" ]]; then
  TOP=$(awk -F',' 'NR==2 {print $1}' "$RESULT_DIR/hotspots.csv" | tr -d '"')
  echo "  [INFO] Top computing task: ${TOP:-<empty>}"
  if [[ -z "$TOP" ]] || echo "$TOP" | grep -qiE 'python|asyncio|uvicorn|<unknown>'; then
    echo "  [FAIL] top hotspot looks like CPU/Python; ROI may be too wide"
    verify_fail=1
  else
    echo "  [ OK ] top hotspot is a GPU kernel"
  fi
else
  echo "  [FAIL] hotspots.csv missing (report failed or empty)"
  verify_fail=1
fi

# Check 3b: the worker patch ran in the process that does the GPU work
if [[ "$VTUNE_PHASE" != "both" || "$VTUNE_ROI_MODE" != "window" ]]; then
  if [[ -s "$RESULT_DIR/tasks.csv" ]] && grep -q 'vllm\.' "$RESULT_DIR/tasks.csv"; then
    echo "  [ OK ] vllm.* ITT tasks present (worker patch active)"
  else
    echo "  [WARN] no vllm.* ITT tasks: the window holds every step of the benchmark;"
    echo "         VTUNE_PHASE / VTUNE_ROI_MODE did not filter anything"
  fi
fi

# Check 4: GPU time in the ROI vs benchmark wall-clock. (The summary's Elapsed Time
# covers the whole collection, paused start-up included, so it is not the ROI length.)
if [[ -s "$RESULT_DIR/summary.csv" && -n "${BENCH_ELAPSED:-}" && "${BENCH_ELAPSED:-0}" -gt 0 && -n "${GPU_TIME:-}" ]]; then
  ratio=$(awk -v e="$GPU_TIME" -v b="$BENCH_ELAPSED" 'BEGIN{printf "%.2f", (e/b)}')
  echo "  [INFO] GPU Time=${GPU_TIME}s, benchmark wall=${BENCH_ELAPSED}s, ratio=${ratio}"
  if awk -v r="$ratio" 'BEGIN{exit !(r>0.5 && r<1.2)}'; then
    echo "  [ OK ] GPU Time matches the benchmark window"
  else
    echo "  [WARN] GPU Time outside [0.5, 1.2] x benchmark wall-clock: ROI too wide or GPU idle"
  fi
fi
echo "  [INFO] Profiled speed is not the real speed: time the same load without VTune"
echo "         before comparing settings (SKILL.md §0)."


if [[ $verify_fail -eq 0 ]]; then
  echo "  Verification: PASS"
else
  echo "  Verification: FAIL - inspect $RESULT_DIR before drawing conclusions"
fi

set -e
set -o pipefail

# ---- Step 8: Shrink the result dir ----
# `-discard-raw-data` drops the raw collector byte-streams once the CSVs and
# sqlite-db are generated. The .vtune project still opens in the GUI but
# in-kernel source-view / re-finalization won't work. Skip via VTUNE_KEEP_RAW=1
# if you need those. Typical savings: ~50-70% (a 160 MB result drops to ~60 MB).
if [[ "${VTUNE_KEEP_RAW:-0}" != "1" ]]; then
  BEFORE=$(du -sm "$RESULT_DIR" 2>/dev/null | awk '{print $1}')
  vtune -finalize -r "$RESULT_DIR" -discard-raw-data >/dev/null 2>&1 || true
  AFTER=$(du -sm "$RESULT_DIR" 2>/dev/null | awk '{print $1}')
  echo "Result dir shrunk from ${BEFORE:-?} MB to ${AFTER:-?} MB (set VTUNE_KEEP_RAW=1 to preserve raw data)."
fi

echo
echo "Done. Result: $RESULT_DIR"
echo "Open with: vtune-gui $RESULT_DIR"
echo "CSV reports (only the ones that produced data): summary.csv, hotspots.csv, tasks.csv"
