# `cuda_to_xpu` rewrite whitelist

The transform rewrites `torch.cuda.X` → `torch.xpu.X` only for X in
the intersection of `torch.cuda` and `torch.xpu`. Verified against
upstream `torch/xpu/__init__.py` `__all__`.

CUDA-only attrs go to the scanner's `escalate` bucket with a note;
the agent decides whether to remove the call, replace it with a
torch-native equivalent, or stop and ask.

## Allowed (mechanically rewritten)

| Category | Members |
|---|---|
| Availability | `is_available`, `is_initialized`, `is_bf16_supported`, `is_tf32_supported`, `init` |
| Device | `device_count`, `current_device`, `set_device`, `get_device_name`, `get_device_properties`, `get_device_capability`, `can_device_access_peer`, `device`, `device_of` |
| Execution | `synchronize`, `empty_cache` |
| Memory | `memory_allocated`, `memory_reserved`, `max_memory_allocated`, `max_memory_reserved`, `memory_stats`, `memory_stats_as_nested_dict`, `memory_snapshot`, `reset_peak_memory_stats`, `reset_accumulated_memory_stats` |
| RNG | `manual_seed`, `manual_seed_all`, `seed`, `seed_all`, `initial_seed`, `get_rng_state`, `get_rng_state_all`, `set_rng_state`, `set_rng_state_all` |
| Streams | `Stream`, `Event`, `StreamContext`, `stream`, `current_stream`, `set_stream`, `get_stream_from_external` |
| Graphs | `graph`, `graph_pool_handle`, `make_graphed_callables` |
| Other | `get_arch_list`, `get_gencode_flags`, `MemPool` |

## Not rewritten (escalate)

| CUDA-only attr | Why / suggested action |
|---|---|
| `nccl` | Use XCCL via `torch.distributed` (the `dist_backend` transform handles `init_process_group`). |
| `nvtx` | No XPU equivalent; remove or guard. |
| `cudart`, `current_blas_handle`, `default_stream` | Internal CUDA handles; remove. |
| `clock_rate`, `power_draw`, `utilization` | Telemetry not exposed via `torch.xpu`; use `xpu-smi` (skill: `xpu-discover`). |
| `list_gpu_processes`, `set_sync_debug_mode` | No equivalent; remove. |
| `host_memory_stats`, `memory_summary` | No equivalent; use `torch.xpu.memory_stats()` if you need a snapshot. |
| `*Tensor` / `*Storage` (`FloatTensor`, etc.) | Deprecated even on CUDA; switch to `torch.tensor(..., dtype=...)`. |
| `caching_allocator_*` | No public XPU equivalent. |
| `jiterator`, `has_half`, `has_magma` | CUDA-only; remove. |
| `CUDAGraph` | Use `torch.xpu.XPUGraph`. |
| `tunable` | CUDA-only TunableOp; no XPU equivalent. |
