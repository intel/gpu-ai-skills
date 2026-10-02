#!/usr/bin/env python3
"""Sanity-check the TP collective before serving multi-GPU.

`--tp > 1` reduces activations across ranks through torch-XPU -> oneCCL ->
libfabric. When libfabric binds an interface the ranks cannot reach (common
with `--network host` on a multi-homed host), the collective blocks forever:
the server still reports ready, then the first request never returns. This
probe runs one small all-reduce so that failure surfaces in seconds instead
of in a launch that looks healthy.

The skill always launches TP with FI_TCP_IFACE=lo; this is the diagnostic
for when a launch hangs anyway. Run it inside the serving image, with the
same network mode, ZE_AFFINITY_MASK, and env vars you serve with. It bounds
itself, so run it in the foreground and read the verdict — never background
it and poll:

    docker run --rm --ipc=host --shm-size=2g [--network host] \\
        --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \\
        --group-add "$RENDER_GID" -e ZE_AFFINITY_MASK=0,1 \\
        -e FI_TCP_IFACE=lo \\
        -v "<skill>/scripts/tp_collective_probe.py:/probe.py:ro" \\
        <image> python3 /probe.py

Verdicts (exit code in brackets):
  "COLLECTIVE OK"     [0] transport works; a serving hang is elsewhere.
  "COLLECTIVE HUNG"   [3] transport broken: confirm `FI_TCP_IFACE=lo`
                          reached the container (multi-node: pin the NIC
                          the nodes share instead) or try bridge networking.
  "NOT ENOUGH XPUS"   [2] ZE_AFFINITY_MASK exposes fewer XPUs than asked.
  traceback           [1] a different problem entirely.

A healthy all-reduce completes in 4-6 s, so the default 20 s budget is
generous; raise it with --timeout-s only on a much larger world size.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time


def _worker(rank: int, world_size: int, store: str) -> None:
    import torch
    import torch.distributed as dist

    torch.xpu.set_device(rank)
    dist.init_process_group(
        "xccl", init_method=f"file://{store}", rank=rank, world_size=world_size
    )
    t = torch.ones(4096, device=f"xpu:{rank}")
    dist.all_reduce(t)          # blocks here when the transport is unusable
    torch.xpu.synchronize()
    got = int(t[0].item())
    assert got == world_size, f"all_reduce gave {got}, expected {world_size}"
    if rank == 0:
        print("COLLECTIVE OK", flush=True)
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=2,
                    help="ranks to probe; match your --tp (default 2)")
    ap.add_argument("--store", default=os.path.join(
                        tempfile.gettempdir(), f"tp_probe_store_{os.getpid()}"),
                    help="file-based rendezvous path (avoids port conflicts)")
    ap.add_argument("--timeout-s", type=float, default=20.0,
                    help="verdict deadline; a healthy all-reduce needs 4-6 s")
    args = ap.parse_args()

    import torch
    import torch.multiprocessing as mp

    visible = torch.xpu.device_count()
    if visible < args.world_size:
        print(f"NOT ENOUGH XPUS: {visible} visible, {args.world_size} needed "
              f"(check ZE_AFFINITY_MASK)", file=sys.stderr)
        return 2

    if os.path.exists(args.store):
        os.unlink(args.store)
    mp.set_start_method("spawn", force=True)
    ctx = mp.spawn(_worker, args=(args.world_size, args.store),
                   nprocs=args.world_size, join=False)

    # join(timeout) returns True only once EVERY rank has exited; it returns
    # False as soon as the first one does, so it has to be polled to a
    # deadline rather than called once (a single call misreads a healthy run
    # as hung). It re-raises a child exception, which we let propagate.
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        if ctx.join(timeout=0.5):
            return 0

    # Still alive at the deadline: the all-reduce never completed, which is
    # the transport failure this probe exists to catch. Reaping needs care —
    # a rank wedged in the oneCCL busy-wait ignores SIGTERM, so SIGTERM with a
    # grace period, then SIGKILL.
    for proc in ctx.processes:
        if proc.is_alive():
            proc.terminate()
    grace = time.monotonic() + 5.0
    for proc in ctx.processes:
        proc.join(timeout=max(0.1, grace - time.monotonic()))
    for proc in ctx.processes:
        if proc.is_alive():
            proc.kill()

    print(f"COLLECTIVE HUNG (>{args.timeout_s:.0f}s): the all-reduce did not "
          f"complete. Confirm `-e FI_TCP_IFACE=lo` reached the container "
          f"(ranks on one host) or try bridge networking.", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit, not return: multiprocessing's atexit hook joins surviving
    # children, and a wedged rank would block it forever — which leaves the
    # probe container "Up" indefinitely holding the GPUs.
    os._exit(3)


if __name__ == "__main__":
    sys.exit(main())
