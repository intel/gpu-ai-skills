# Multi-GPU (TP) on XPU: the all-reduce hang and its fix

Everything here is measured on `v0.5.23-xpu` / 2× Arc Pro B70.

## Symptom

`--tp 2` starts normally: weights load, KV cache allocates on both cards,
`Application startup complete`, `/v1/models` answers as fast as on a healthy
server. The **first
inference request then never returns** — no HTTP error, no crash, no
watchdog. The server stays up indefinitely.

It looks exactly like a healthy server, which is why readiness must never
be reported as success on its own.

## Is it safe to always set `FI_TCP_IFACE=lo`?

Yes, for single-host TP — which is why the skill sets it unconditionally
rather than making the agent decide. Risk review, all checked on the
verified host:

| Concern | Finding |
|---|---|
| Throughput cost vs. a working real NIC | none: TP=2, 8 concurrent × 64 tokens — `eno3np0` took ~15% longer than `lo` (loopback skips the NIC) |
| Single-GPU launches | inert — no collectives run |
| **`--nnodes > 1`** | **breaks it** — ranks on other hosts are unreachable over loopback. Pin the NIC the nodes share instead. |

So the only hazard is multi-node, which this skill does not cover; it is
called out explicitly in the Multi-GPU section.

## Diagnosing: probe the collective directly

If a TP launch hangs even with `FI_TCP_IFACE=lo` set, take the model out of
the picture. `scripts/tp_collective_probe.py` does one 2-rank XPU all-reduce
and bounds itself, so run it in the foreground inside the
serving image with the same network mode, `ZE_AFFINITY_MASK`, and env vars
as the launch:

```sh
RENDER_GID=$(getent group render | cut -d: -f3)
# Same image as the hanging server. If IMG is not already set in this shell:
#   IMG=lmsysorg/sglang:<tag from SKILL.md Step 1>
: "${IMG:?resolve the image first — see SKILL.md Step 1}"
docker run --rm --ipc=host --shm-size=2g --network host \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --group-add "$RENDER_GID" -e ZE_AFFINITY_MASK=0,1 -e FI_TCP_IFACE=lo \
    -v "<skill-dir>/scripts/tp_collective_probe.py:/probe.py:ro" \
    "$IMG" python3 /probe.py
```

`<skill-dir>` is the installed `sglang-xpu-run` directory. Pass
`--world-size N` to match `--tp N`.

| Verdict (exit code) | Meaning |
|---|---|
| `COLLECTIVE OK` (0) | Transport works — the hang is elsewhere (py-spy recipe below; engine-side Level Zero hangs in Notes). |
| `COLLECTIVE HUNG` (3) | Transport still broken. Check `FI_TCP_IFACE` reached the container (`docker exec <c> env`); try bridge networking. |
| `NOT ENOUGH XPUS` (2) | `ZE_AFFINITY_MASK` exposes fewer XPUs than `--world-size`. |
| traceback (1) | A different problem entirely. |

Measured probe verdicts on the verified host:

| Probe configuration | Verdict |
|---|---|
| bridge networking, no env var | `COLLECTIVE OK` |
| `--network host`, no env var | `COLLECTIVE HUNG` (log ends at the `CCL_WARN` lines) |
| `--network host`, `FI_TCP_IFACE=lo` | `COLLECTIVE OK` |

Serving results with and without the pin, measured on 2× Arc Pro B70,
Qwen3-0.6B:

| | ready | first request |
|---|---|---|
| `--tp 2`, no `FI_TCP_IFACE` | same as with the pin | **never returns** (still hung when the client timed out) |
| `--tp 2`, `FI_TCP_IFACE=lo` | same as `--tp 1` | returns; several times slower than warm requests (one-time compilation) |

**The hang is still present on the pinned tag** (`fi_info -p shm` is still
empty in this image), so do not treat a newer tag as a reason to drop the
variable. The no-pin run reaches `/v1/models` and
then sits on the first request until the client's own timeout — exactly the
documented signature.

After the fix both cards hold comparable memory (~26–27 GB each for this
model+KV, also with 32 concurrent requests) and `Prefill batch` / `Decode batch` lines appear in the log.

## When a TP smoke request times out

The verify block's `--max-time` fired on a server that reported ready. In
order:

1. Confirm the pin reached the container:
   `docker inspect -f '{{.Config.Env}}' sglang-xpu-tp` must list
   `FI_TCP_IFACE=lo`. A missing variable means the launch line was edited.
2. Run the probe (above) with the same network mode and mask. `COLLECTIVE
   HUNG` is this transport problem; `COLLECTIVE OK` means the hang is
   elsewhere — take the py-spy dump below, then retry at a lower `--tp`.
3. Healthy signs to compare against (Qwen3-0.6B, 2× B70): `--tp 2` becomes
   ready about as fast as `--tp 1`, the first request is several times slower
   than warm ones but returns, and both cards show comparable memory in
   `xpu-smi dump -m 18`. Garbage `content` with a fast response is a
   `--tp` / `ZE_AFFINITY_MASK` mismatch rather than quantization.

## Root cause: host topology + a missing shared-memory provider

Two independent facts combine. Neither is a GPU, driver, or sglang defect.

**1. The stack falls back to TCP for ranks on the same machine.** sglang
spawns its TP ranks itself (no `mpirun`), so oneCCL finds no MPI launcher
and switches to its OFI transport — that is the `|CCL_WARN| … switch to
ATL/OFI` line. libfabric's shared-memory provider would be the natural
intra-node choice, but in this image it yields nothing:

```
$ fi_info -p shm
fi_getinfo: -61 (No data available)
```

and oneCCL duly reports `atl attrs: in: { shm: 0, … } out: { shm: 0, … }`.
So two ranks on one host talk over TCP sockets.

**2. libfabric then picks the wrong interface.** It enumerates non-loopback
domains first (`fi_info -p tcp` lists `eth0` before `lo`), and on a
multi-homed host that means a bridge or a link-local NIC. With
`CCL_LOG_LEVEL=info` the choice is explicit:

```
|CCL_INFO| provider: tcp
|CCL_INFO|   nic: { name tcp:br-61d8f070a03c, state unknown, speed 1.25 GB/s }
```

`br-61d8f070a03c` is a Docker bridge (172.18.0.1). Rank-to-rank traffic
never completes over it, so the collective blocks forever.

### Which interfaces work

Only `FI_TCP_IFACE` changed, on a host offering `lo`, `enx…` (no IPv4),
`eno3np0` (192.168.11.2), `br-…` (172.18.0.1) and `docker0` (172.17.0.1).
Probe = `scripts/tp_collective_probe.py` under `--network host`:

| `FI_TCP_IFACE` | Result |
|---|---|
| unset → libfabric picks `br-…` | probe **HUNG**; server's first request hangs |
| `br-61d8f070a03c` | probe **HUNG** |
| `docker0` | probe fails fast: `onecclCommInitRank failed with code 11` |
| `eno3np0` (routable NIC) | probe OK; server serves (8 × 64 tokens ~15% slower than `lo`) ✅ |
| `lo` | probe OK; server serves ✅ |

So loopback is not magic — a real routable NIC works too. The failures are
the bridge-type interfaces libfabric happens to rank first. `lo` is the
recommended pin because it is always present, always correct for ranks on
one host, and needs no knowledge of the host's NIC names.

**Attribution:** the *host* supplies the confusing interface list, while the
*image* supplies the missing `shm` provider that forces TCP in the first place
and pins no `FI_*` defaults. sglang contributes only by spawning ranks without
MPI.

## Bridge networking also avoids it

| `--tp 2`, Qwen3-0.6B | Network mode | `FI_TCP_IFACE` | Result |
|---|---|---|---|
| | `--network host` | unset | **hangs** at first request |
| | `--network host` | `lo` | serves |
| | bridge `-p 30099:30000` (verify with `PORT=30099`) | unset | serves; readiness and first request match the row above |

Two remedies, either is sufficient: pin `FI_TCP_IFACE=lo`, or use bridge
networking so the container sees only its own `eth0` and `lo`.

## Confirming it is this hang, not something else

```sh
docker exec <container> pip install -q py-spy
docker exec <container> sh -c 'py-spy dump --pid $(pgrep -f scheduler_TP0 | head -1)'
```

A hung TP rank shows this stack — the first collective of the forward
pass, in the vocab-parallel embedding:

```
all_reduce (torch/distributed/distributed_c10d.py)
_all_reduce_in_place (distributed/parallel_state.py)
all_reduce (distributed/parallel_state.py)
forward (vocab_parallel_embedding.py)
forward (qwen3.py) -> _execute_extend (runner/eager_runner.py)
  -> forward_batch_generation (tp_worker.py) -> event_loop_normal (scheduler.py)
```

Both ranks show the same frame: they arrive at the collective and neither
completes it. If instead you see a stack inside `_inductor` or a kernel
launch, this is not the same problem — the dozens of idle
`torch/_inductor/compile_worker` processes are a normal pre-forked pool and
are not evidence of compilation.

## What does *not* fix it

All under `--network host` with no `FI_TCP_IFACE`. Server rows ran
Qwen3-0.6B `--tp 2`; env-var rows ran the collective probe. None works:

| Tried | Result |
|---|---|
| `--attention-backend triton` instead of `intel_xpu` (server) | first request hangs |
| default flags: no `--disable-overlap-schedule`, default page size (server) | first request hangs |
| `--privileged` + `-v /dev/shm:/dev/shm` + `--user root` + `--group-add video` (a common XPU container shape) | probe HUNG |
| `CCL_ZE_IPC_EXCHANGE=sockets` / `=drmfd` | probe HUNG |
| `FI_PROVIDER=tcp` *without* `FI_TCP_IFACE` | probe HUNG |
| `CCL_ATL_TRANSPORT=ofi` + `FI_PROVIDER=tcp` (no iface pin) | probe HUNG |
| `CCL_ZE_ENABLE=0` | probe fails fast: `onecclCommInitRank failed with code 11` |
| `FI_PROVIDER=shm` | probe fails fast: same `code 11` (the image has no `shm` provider) |

`FI_TCP_IFACE=lo` alone is sufficient. The two cards being on separate
PCIe root complexes (no P2P) is *not* the cause — a host-staged transport
would have worked; the transport simply never connected.

## Decode context parallelism (`--dcp-size`)

DCP shards the KV cache across the ranks of a TP group by token position and
merges partial attention by log-sum-exp. Verified on 2× Arc Pro B70 with
`intel_xpu` attention:

```sh
python3 -m sglang.launch_server --model Qwen/Qwen3.5-4B --device xpu \
    --attention-backend intel_xpu --tp 2 --dcp-size 2 \
    --disable-radix-cache --mem-fraction-static 0.60 \
    --host 0.0.0.0 --port 30000      # docker run ... -e FI_TCP_IFACE=lo
```

| `--tp 2`, Qwen3.5-4B | max total tokens | 256-token decode time, warm (relative) |
|---|---|---|
| without `--dcp-size` | 507,008 | 1.00× |
| `--dcp-size 2` (`a2a` comm, the XPU default) | 504,832 | 1.12× |
| `--dcp-size 2` + `--linear-attn-backend intel_xpu` | 504,832 | 1.04× |

Output is correct in all three, but on two cards **DCP bought no capacity
and cost ~5–12% decode time**: it only grows the servable context when `tp_size` exceeds the
model's KV-head count (otherwise TP already splits the KV heads), which on a
2-card host means 1-KV-head models. The slow first request is not DCP (same
without it): it is the Triton JIT for Qwen3.5's linear-attention layers.
`--linear-attn-backend intel_xpu` makes it ~7× faster with DCP too (third row).
If startup fails with
`UR_RESULT_ERROR_OUT_OF_RESOURCES`, lower `--mem-fraction-static`.
