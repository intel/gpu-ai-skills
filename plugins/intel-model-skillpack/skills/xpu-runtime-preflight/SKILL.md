---
name: xpu-runtime-preflight
description: Run a read-only go/no-go preflight before any Intel GPU/XPU skillpack work. Checks driver health, /dev/dri permissions, render/video groups, Docker, /dev/shm, disk, proxy, and optional container-level XPU visibility. Use when the user asks whether a machine is ready for XPU model work or needs a reusable lab readiness report. Not for launching workloads, pulling images, editing system config, or verifying model output.
---

# xpu-runtime-preflight

Use this as the shared readiness gate before Intel GPU/XPU skills in
this pack. It answers "is this host/container ready for the requested
skill path?", reports a `READY` / `READY WITH WARNINGS` / `BLOCKED`
verdict, and routes failures to focused skills. It does not start model
servers, run benchmarks, capture profiles, restart Docker, pull large
images, edit system configuration, or validate generated content.

## When To Use

Run this before a model run, benchmark, profile, or container workflow
when readiness depends on more than GPU discovery. Use `xpu-discover`
for device inventory and driver health; use this skill for the broader
go/no-go path: `/dev/dri`, groups, Docker, shared memory, disk, proxy
and network checks, plus optional container XPU visibility.

If preflight finds a blocker, follow the generated `SUMMARY.md` handoff.
Do not duplicate routing guidance in this skill body.

## Quick start

```sh
plugins/intel-model-skillpack/skills/xpu-runtime-preflight/scripts/check_runtime_preflight.sh \
    --target-gpu 0 \
    --out-dir .out/skills/xpu-runtime-preflight
```

Add an already-local image when the next skill path runs in a container:

```sh
plugins/intel-model-skillpack/skills/xpu-runtime-preflight/scripts/check_runtime_preflight.sh \
    --target-gpu 0 \
    --image <already-local-image>
```

When downloads or image builds are part of the next step, add a network
check:

```sh
plugins/intel-model-skillpack/skills/xpu-runtime-preflight/scripts/check_runtime_preflight.sh \
    --env-file .env \
    --target-gpu 0 \
    --network-check
```

The script writes:

```text
.out/skills/xpu-runtime-preflight/SUMMARY.md
.out/skills/xpu-runtime-preflight/status.tsv
.out/skills/xpu-runtime-preflight/preflight.log
```

It also writes per-check evidence files such as
`xpu-smi-discovery.txt`, `xpu-smi-precheck.txt`,
`xpu-smi-diag-target.txt`, `xpu-smi-stats-target.txt`,
`target-driver.txt`, `dev-dri.txt`, `docker-info.txt`, and, when
`--image` is supplied, `image-preflight.txt`.

## What Counts As Ready

Hard pass before GPU/XPU skill work:

1. `xpu-smi discovery` sees the target GPU.
2. The target GPU has an inspectable kernel driver binding. `xe` and
   `i915` are accepted; `vfio*` blocks because the host driver did not
   bind the target GPU for XPU runtime use.
3. Targeted `xpu-smi diag -d <id> -l 1` and bounded
   `xpu-smi stats -d <id>` complete or leave inspectable warning
   artifacts.
4. `/dev/dri/renderD*` exists and permissions are understandable.
5. Docker daemon is reachable by the current user.
6. `/dev/shm` and disk space are not obviously too small.
7. If `--image` is supplied, the selected container sees XPU through a
   Level Zero or Python XPU visibility probe. The default Python probe
   requires `torch.xpu.is_available()` and a nonzero XPU device count.

Warnings do not block every workflow. A missing proxy is fine on direct
networks; missing BuildKit is fine unless the next step builds an image;
a workstation `xpu-smi diag` permission warning may still allow GPU work
if the computation sub-test passes.

## Result Routing

Use the generated `SUMMARY.md` as the source of truth for failure-to-next
skill routing. The script builds that section from the checks it actually
ran, so do not maintain a separate static routing table in this skill.

For final answers, read `status.tsv` or the `SUMMARY.md` status table,
then report hard blockers first and follow the `SUMMARY.md` "Next Skill
Routing" section for the recommended handoff.

`SUMMARY.md` includes a `Verdict` field:

- `READY` when no checks fail or warn.
- `READY WITH WARNINGS` when no checks fail but at least one warning is
  present.
- `BLOCKED` when any check fails.

When blocked, `SUMMARY.md` also includes `First blocker:` with the
first failing check from `status.tsv`.

## Image Preflight Notes

The script refuses to implicitly pull an image. If `--image` is not
already local, verify host/Docker/container proxy settings or run the
skill that owns the image choice.

The generic image preflight sources oneAPI (if present) and then runs
the XPU visibility probe against whatever `python3` is on the image's
`PATH`. It does not activate conda, miniforge, or any other Python
environment — that is the image's responsibility:

```sh
source /opt/intel/oneapi/setvars.sh --force
```

If the image needs a conda/venv activation, a non-default oneAPI path,
or any other setup before `python3 -c "import torch"` works, pass
`--image-command '<shell command>'` to supply the full probe.

Image preflight uses Docker `bridge` networking by default. This avoids
masking container-network issues with host networking. If the target
runtime will intentionally use another Docker network mode, pass
`--image-network <mode>` so the preflight matches that run.

## Reporting

In final answers, report:

- target host and GPU ID
- verdict
- PASS/WARN/FAIL counts
- first blocker and hard blockers
- the next skill to use for each blocker
- the output artifact paths

Do not claim container readiness from host checks alone when the next
step uses a container image; run the optional `--image` preflight for
that image.
