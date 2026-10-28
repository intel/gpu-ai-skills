# Multi-GPU (TP) on XPU: the all-reduce hang and its fix

## Symptom

`--tp 2` starts normally: weights load, KV cache allocates on both cards,
`Application startup complete`, `/v1/models` answers in ~28 s. The **first
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
| Throughput cost vs. a working real NIC | none: TP=2, 8x64 tokens — `lo` 1524 ms/req, `eno3np0` 1655 ms/req (loopback skips the NIC) |
| Single-GPU launches | inert — no collectives run |
| An image whose `shm` provider works | inert — oneCCL prefers shm; the variable only steers libfabric's `tcp` provider |
| RDMA / `verbs` provider | inert — different provider, variable ignored |
| **`--nnodes > 1`** | **breaks it** — ranks on other hosts are unreachable over loopback. Pin the NIC the nodes share instead. |

So the only hazard is multi-node, which this skill does not cover; it is
called out explicitly in the Multi-GPU section.

## Diagnosing: probe the collective directly

If a TP launch hangs even with `FI_TCP_IFACE=lo` set, take the model out of
the picture. `scripts/tp_collective_probe.py` does one 2-rank XPU all-reduce
and bounds itself (20 s default), so run it in the foreground inside the
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

Measured probe verdicts on the verified host (re-checked at v0.5.22):

| Probe configuration | Verdict |
|---|---|
| bridge networking, no env var | `COLLECTIVE OK` in 5 s |
| `--network host`, no env var | `COLLECTIVE HUNG` (log ends at the `CCL_WARN` lines) |
| `--network host`, `FI_TCP_IFACE=lo` | `COLLECTIVE OK` in 6 s |

Serving results with and without the pin, measured on 2× Arc Pro B70,
Qwen3-0.6B:

| | ready | first request | warm request |
|---|---|---|---|
| `--tp 2`, no `FI_TCP_IFACE` | ~27 s | **never returns** (hung >120 s) | — |
| `--tp 2`, `FI_TCP_IFACE=lo` | ~27 s | ~5 s | under 1 s |

**The hang has survived every release bump so far** (re-tested at v0.5.21 and
v0.5.22; `fi_info -p shm` is still empty in the v0.5.22 image), so do not
treat a newer tag as a reason to drop the variable. The no-pin run reaches `/v1/models` and
then sits on the first request until the client's own timeout — exactly the
documented signature.

After the fix both cards hold comparable memory (~27 GB each for this
model+KV) and `Prefill batch` / `Decode batch` lines appear in the log.

## When a TP smoke request times out

The verify block's `--max-time` fired on a server that reported ready. In
order:

1. Confirm the pin reached the container:
   `docker inspect -f '{{.Config.Env}}' sglang-xpu-tp` must list
   `FI_TCP_IFACE=lo`. A missing variable means the launch line was edited.
2. Run the probe (above) with the same network mode and mask. `COLLECTIVE
   HUNG` is this transport problem; `COLLECTIVE OK` means the hang is
   elsewhere — take the py-spy dump below, then retry at a lower `--tp`
   (upstream records engine-side Level Zero hangs on some Arc/B-series hosts;
   see Notes).
3. Healthy numbers to compare against (Qwen3-0.6B, 2× B70): ready ~27 s,
   first request ~5–6 s, under 1 s warm, and comparable memory on both cards
   in `xpu-smi dump -m 18`. Garbage `content` with a fast response is a
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

Same launch, only `FI_TCP_IFACE` changed, on a host offering `lo`,
`enx…`(169.254/16 link-local), `eno3np0`(192.168.11.2), `br-…`(172.18.0.1)
and `docker0`(172.17.0.1):

| `FI_TCP_IFACE` | First request |
|---|---|
| unset → libfabric picks `br-…` | **hangs** |
| `docker0` | **hangs** |
| `eno3np0` (routable NIC) | 6 s ✅ |
| `lo` | 6 s ✅ |

So loopback is not magic — a real routable NIC works too. The failures are
the bridge-type interfaces libfabric happens to rank first. `lo` is the
recommended pin because it is always present, always correct for ranks on
one host, and needs no knowledge of the host's NIC names.

**Attribution:** the *host* supplies the confusing interface list (a
single-NIC machine would likely have worked by default), while the *image*
supplies the missing `shm` provider that forces TCP in the first place and
pins no `FI_*` defaults. sglang contributes only by spawning ranks without
MPI. Expect the same class of failure on any multi-bridge container host.

## Not sglang-specific: vLLM-XPU behaves identically

Both stacks reach the same oneCCL OFI transport, so both break under
`--network host` on this host. vLLM only *appears* immune because its
published recipe uses bridge networking (`-p 8000:8000`).

| Stack | Network mode | `FI_TCP_IFACE` | Result |
|---|---|---|---|
| sglang `--tp 2` | `--network host` | unset | **hangs** at first request |
| sglang `--tp 2` | `--network host` | `lo` | ready 28 s, request 6 s |
| sglang `--tp 2` | bridge `-p 30099:30000` | unset | ready 28 s, request 6 s |
| vLLM `--tensor-parallel-size 2` | bridge `-p 8000:8000` | unset | ready 142 s, request 1 s |
| vLLM `--tensor-parallel-size 2` | `--network host` | unset | **hangs before ready** |
| vLLM `--tensor-parallel-size 2` | `--network host` | `lo` | ready 137 s, request 0 s |

vLLM hangs *earlier* than sglang — it runs its collectives during engine
init, so the server never reaches `Application startup complete`, whereas
sglang becomes ready and then stalls on the first forward pass. Same cause,
different symptom. vLLM's log stops at `|CCL_WARN| value of
CCL_ATL_TRANSPORT changed to be ofi`.

Two remedies, either is sufficient: pin `FI_TCP_IFACE=lo`, or use bridge
networking so the container sees only its own `eth0` and `lo`.

## Confirming it is this hang, not something else

```sh
docker exec <container> pip install -q py-spy
docker exec <container> sh -c 'py-spy dump --pid $(pgrep -f scheduler_TP0)'
```

A hung TP rank shows this stack — the first collective of the forward
pass, in the vocab-parallel embedding:

```
all_reduce (torch/distributed/distributed_c10d.py)
_all_reduce_in_place (distributed/parallel_state.py)
tensor_model_parallel_all_reduce (distributed/communication_op.py)
forward (vocab_parallel_embedding.py)
forward (qwen3.py) -> _execute_extend -> run_batch -> event_loop_normal
```

Both ranks show the same frame: they arrive at the collective and neither
completes it. If instead you see a stack inside `_inductor` or a kernel
launch, this is not the same problem — the dozens of idle
`torch/_inductor/compile_worker` processes are a normal pre-forked pool and
are not evidence of compilation.

## What does *not* fix it

All tested on the verified host; each still hung with no first response:

| Tried | Result |
|---|---|
| `--attention-backend triton` instead of `intel_xpu` | hangs |
| default flags (no `--disable-overlap-schedule`, default page size) | hangs |
| `--privileged` + `-v /dev/shm:/dev/shm` + `--user root` + `--group-add video` (the shape in upstream's XPU docs) | hangs |
| `CCL_ZE_ENABLE=0` | hangs |
| `CCL_ZE_IPC_EXCHANGE=sockets` | hangs |
| `CCL_ZE_IPC_EXCHANGE=drmfd` | hangs |
| `FI_PROVIDER=shm` | hangs |
| `FI_PROVIDER=tcp` *without* `FI_TCP_IFACE` | hangs |
| `CCL_ATL_TRANSPORT=ofi` + `FI_PROVIDER=tcp` (no iface pin) | hangs |

`FI_TCP_IFACE=lo` alone is sufficient. The two cards being on separate
PCIe root complexes (no P2P) is *not* the cause — a host-staged transport
would have worked; the transport simply never connected.

## Decode context parallelism (`--dcp-size`, new in v0.5.22)

DCP shards the KV cache across the ranks of a TP group by token position and
merges partial attention by log-sum-exp. Upstream marks it experimental on
XPU; it needs `intel_xpu` attention. Verified on 2× Arc Pro B70 with
upstream's own example:

```sh
python3 -m sglang.launch_server --model Qwen/Qwen3.5-4B --device xpu \
    --attention-backend intel_xpu --tp 2 --dcp-size 2 \
    --disable-radix-cache --mem-fraction-static 0.60 \
    --host 0.0.0.0 --port 30000      # docker run ... -e FI_TCP_IFACE=lo
```

| `--tp 2`, Qwen3.5-4B | max total tokens | 256-token decode, warm |
|---|---|---|
| without `--dcp-size` | 507,008 | 10.0 s |
| `--dcp-size 2` (`a2a` comm, the XPU default) | 504,832 | 11.0 s |

Output is correct, but on two cards **DCP bought no capacity and cost ~10%
decode time**: it only grows the servable context when `tp_size` exceeds the
model's KV-head count (otherwise TP already splits the KV heads), which on a
2-card host means 1-KV-head models. The ~70 s first request is not DCP (same
without it): it is the Triton JIT for Qwen3.5's linear-attention layers.
`--linear-attn-backend intel_xpu` cut it to 6–20 s at TP=1
(`model-recipes.md`); that flag was not re-tested with DCP. If startup fails with
`UR_RESULT_ERROR_OUT_OF_RESOURCES`, lower `--mem-fraction-static`.

## Notes

- Upstream's XPU CI does cover TP, nightly only: Llama-3.1-8B at TP=2 and
  four MoE models (Gemma-4-26B-A4B, Nemotron-3-Nano-30B-A3B, Qwen3-30B-A3B,
  Qwen3.5-35B-A3B) at TP=4. Expect to re-verify after an image bump.
- Upstream's test code records two Level Zero hangs on some Arc/B-series
  hosts that are *not* this transport problem (the probe says
  `COLLECTIVE OK`):
  - Llama-3.1-8B at TP=4 "wedges the Level Zero driver during the first
    prefill batch", which is why that test runs TP=2
    (`test/registered/xpu/llm_models/test_xpu_llama_3_1_8b.py`).
  - "intel_xpu attention at TP>=2 wedges the Level Zero driver on
    concurrent prefill", which is why the nightly accuracy tests send one
    request at a time by default
    (`python/sglang/test/xpu/simple_eval_gsm8k_xpu_mixin.py`).

  Neither reproduced on the verified host (2× Arc Pro B70, TP=2, 32
  concurrent requests). If a TP launch hangs with `FI_TCP_IFACE=lo` set and
  the probe passes, retry at a lower `--tp`.
- On a single-NIC host the default interface choice may happen to work.
  Setting `FI_TCP_IFACE=lo` anyway costs nothing and removes the variable.
- The same class of failure is why upstream's P/D disaggregation docs set
  `UCX_POSIX_USE_PROC_LINK=n`: intra-node transports picking the wrong
  mechanism inside containers.
