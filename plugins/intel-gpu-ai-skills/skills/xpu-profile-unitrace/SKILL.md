---
name: xpu-profile-unitrace
description: Profile Intel-XPU workloads at the SYCL / Level Zero kernel level via Intel pti-gpu's unitrace. Captures per-API-call and per-kernel timing, memory transfers, oneCCL / MPI events, and hardware counters PyTorch-level profilers cannot see. Use when a hot op is already known at the torch.profiler layer and the user needs the SYCL kernel beneath, or when profiling oneCCL collectives in multi-GPU runs. Not for PyTorch-level signal (use torch-xpu-profile / vllm-xpu-profile). Requires building unitrace from source.
---

# xpu-profile-unitrace

`unitrace` profiles XPU workloads at the SYCL / Level Zero kernel
level — captures per-kernel timing, memcopy bytes, oneCCL/MPI
events, and hardware counters that PyTorch profilers can't see.

Use when:

- A PyTorch-level profiler identified a hot op and you need to
  know which SYCL kernel inside it is the cost.
- You need Level Zero command-list events, oneCCL collectives,
  exact memcopy bytes, kernel launch geometry, or HW counters.
- Multi-XPU run with per-rank oneCCL visibility needed.
- Workload is raw SYCL / oneAPI (not PyTorch).

Prefer **torch-xpu-profile** or **vllm-xpu-profile** first; their
output usually answers the question without going to SYCL level.

## Install: check, then build if missing

```sh
command -v unitrace && unitrace --version
```

Most public XPU images don't ship unitrace. To build from source
inside the target image:

```sh
source /opt/intel/oneapi/setvars.sh --force >/dev/null
git clone --depth 1 https://github.com/intel/pti-gpu.git /opt/pti-gpu
cd /opt/pti-gpu/tools/unitrace
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j"$(nproc)"
export PATH="/opt/pti-gpu/tools/unitrace/build:$PATH"
unitrace --version          # confirm
unitrace --device-list      # confirm sees XPU
```

Prerequisites: CMake 3.22+, C++17 compiler, oneAPI Base Toolkit
(present in `vllm/vllm-openai-xpu:latest` and any sglang-xpu image).
Add `-DBUILD_WITH_MPI=1` for multi-GPU collective profiling;
`-DCMAKE_INSTALL_PREFIX=/opt/unitrace && make install` for an
installable layout. Verified clean on `vllm/vllm-openai-xpu:latest`
against unitrace 2.3.0.

If you profile often, bake this into a Dockerfile extending the
runtime image so you don't rebuild every session.

## Quickstart capture

```sh
cd /work       # working dir is where the trace lands
unitrace \
    --chrome-call-logging \
    --chrome-kernel-logging \
    python3 my_workload.py
```

Closing log lines name the exact paths:

```
[INFO] Log is stored in /work/python3.<PID>.json
[INFO] Timeline is stored in python3.<PID>.json
```

Drag `python3.<PID>.json` into <https://ui.perfetto.dev>. `-o NAME`
sets a marker / log path but does **not** rename the timeline;
don't rely on it as a "save as X.json" flag.

For hardware metrics:

```sh
unitrace --stall-sampling --chrome-kernel-logging -o /work/stalls.json python3 my_workload.py
```

`unitrace --device-list` shows visible XPUs;
`unitrace --metric-list` shows available HW counters.

## What unitrace adds over `torch.profiler`

| Layer | torch.profiler | unitrace |
|---|---|---|
| PyTorch op (`aten::matmul`) | yes | yes (passes through) |
| SYCL kernel name + duration | no | yes (e.g. `xetla_gemm_universal_4_b_2_d_4`) |
| Level Zero command-list events | no | yes (queue submit / sync / fence) |
| oneCCL collectives | no | yes (`Allreduce`, `Allgather`, per-rank) |
| Memory copy direction + size | partial | yes (H2D/D2H/P2P with byte counts) |
| Hardware counters | no | yes (`--stall-sampling`, `--metric-query`) |

Use it when the PyTorch op is "matmul" and you need to know which
GEMM kernel was dispatched (XeTLA vs oneDNN vs Triton fallback).

## Reading the timeline

Open the Chrome-trace JSON in Perfetto. Same conventions as
torch.profiler, plus:

- **Kernel-name rows** — group by kernel name (right-click) for
  total time per kernel; top 3–5 dominate.
- **Command-list submit gaps** — wide gaps between adjacent
  kernels (> kernel duration) suggest host is the bottleneck.
- **CCL collective rows** — multi-GPU per-rank timing. A long
  rank blocks others — straggler.
- **Memory copies** — directional rows. Unexpected P2P copies
  often mean a missing `device_map` placement.

## Stall sampling for hot kernels

Once a kernel dominates, rerun with stall sampling to see *why*:

```sh
unitrace --stall-sampling -k --chrome-kernel-logging \
    -o /work/stall.json python3 my_workload.py
```

Output groups stalls by category. High "Memory" -> bandwidth-bound;
high "Pipeline" -> compute-bound. Read against the same roofline
used by **model-config-recommend**.

## Common errors

- `unitrace: command not found` -> build dir not on PATH.
  `export PATH=/opt/pti-gpu/tools/unitrace/build:$PATH`.
- `unable to load metric library` -> oneAPI env not set.
  `source /opt/intel/oneapi/setvars.sh --force`.
- Empty trace -> workload didn't run on XPU. Verify with
  `unitrace --device-list` and `xpu-smi dump -d 0 -m 5`.
- Trace size in GB -> long runs accumulate. Capture a short window
  (5–10 iterations).
- Permission denied reading HW metrics -> some metric modes need
  `--privileged` on the container.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Pin to one XPU before profiling. |
| `LD_LIBRARY_PATH` | Must include oneAPI runtime libs (handled by `setvars.sh`). |
| `ZE_ENABLE_TRACING_LAYER=1` | Force L0 to load the tracing layer when unitrace can't inject it automatically. |

## What this skill does NOT cover

- Fixing a hot SYCL kernel — out of scope.
- Hardware-counter analysis on Data Center GPU Max (richer modes
  not covered here).

## References

- pti-gpu repo: <https://github.com/intel/pti-gpu>
- unitrace README: <https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md>
- Intel performance tools: <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/pti-gpu.html>
- Perfetto: <https://ui.perfetto.dev>
