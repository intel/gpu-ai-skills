---
name: xpu-port
description: Execute a single-target CUDA-to-XPU port of a PyTorch repo with libcst-based scan, mechanical rewrite, and CPU FP64 vs target-dtype correctness verify on one forward pass. Use when the request says "port" — "port my repo to XPU", "port my repo at <path> to XPU", "rewrite the CUDA calls to XPU", "apply the mechanical transforms", "run the scan and rewrite", "make the port changes now". Not for the "migrate" verb ("migrate my repo", "migrate this repo to XPU") or a bare whole-repo workflow request where scope is not yet set — those start with cuda-to-xpu-migration, whose plan routes here. Not for assessment-only, throughput (torch-xpu-bench), op-level slowness (torch-xpu-profile), custom CUDA C++ extensions, or dual-target CUDA+XPU codebases.
---

# xpu-port

Three deterministic scripts; the agent drives the loop.

```
scripts/
├── xpu_port_scan.py     # libcst, classifies sites mechanical/semantic/escalate
├── xpu_port_rewrite.py  # libcst, named transforms, --check previews diff
└── xpu_port_verify.py   # CPU FP64 reference vs target dtype, one forward
```

```sh
pip install libcst
```

## Targets

**Single-target XPU only.** After the port, the codebase runs on
Intel XPU. CUDA call sites are rewritten, not dual-gated. If the
user wants both CUDA and XPU at runtime (dual-target), stop and
say so — that's a different problem (per-call-site guards,
device-aware factories) and this skill produces broken
half-ported code if used for it.

## Backstop — wrong entry point

This skill executes; it does not assess or route. One redirect
exists, pointing one way. If the request is really an assessment
or a whole-repo migration ("migrate this repo", "what would it
take to run on XPU" — scope not yet set), hand it to
**cuda-to-xpu-migration** and stop; its report's Next steps lead
back here for the Python surface.

The same backstop covers mis-scoped ports. This skill reads
Python source only — its gates say nothing about Dockerfiles,
launch scripts, or dependency pins. When
**no migration report is in hand** and the step-1 scan comes back
empty while its advisory flags NVIDIA/CUDA infrastructure
surfaces, the port was mis-scoped: hand it to
**cuda-to-xpu-migration** and stop. When a migration report is in
hand, execute the Python surface it scoped and list its
non-Python surfaces as documented-not-rewritten — an advisory on
the final scan does not reopen this gate (see step 5).

## Procedure

The port's minimum gate is **final scan empty** + **verifier
green**. Both are necessary; neither alone is sufficient (for
service repos with no single forward, the serving skill's smoke
test stands in for the verifier — see step 6).

**Important:** the verifier confirms one forward pass is
numerically correct — it does **not** confirm the training loop
runs end-to-end (e.g. `torch.compile`, custom LR schedulers, or
data-loader workers may still fail on XPU). After the gates pass,
run at least 2 real training iterations on the target device
before declaring the port complete.

### 1. Scan

```sh
python3 path/to/xpu_port_scan.py . > findings.json
```

Findings tagged `mechanical` (safe to rewrite), `semantic` (agent
edits per `references/semantic-patterns.md`), `escalate` (custom
`.cu`, `CUDAExtension`, CUDA-only attrs — agent decides per
finding). Each finding also carries a `route` field: a skill name
(e.g. `torch-xpu-profile`) names the skill an orchestrating agent
should chain to for that site; `null` means there is no single
downstream skill — either xpu-port resolves the site in-loop (a
mechanical transform or a semantic hand-edit) or the site is
`escalate` (redesign, no mechanical port). Read `bucket` first:
`null` does not by itself mean "already handled".

**Empty scan on an NVIDIA-stacked repo:** if the scan found no real
CUDA sites (synthetic `unreadable`/`unparseable` bookkeeping findings
don't count) but the JSON contains an `advisory` block (also printed
to stderr; emitted on directory scans only — a single-file scan never
carries one), the repo has NVIDIA/CUDA surface xpu-port cannot rewrite —
container files, dependency pins, launch scripts, or distinctive
cloud NIM/API endpoints (e.g. `integrate.api.nvidia.com`) in
source. That is an API-first migration, not a `torch.cuda` port.
Stop and route to **cuda-to-xpu-migration** (assessment) then
**xpu-deploy-plan** (serving plan). That is the no-plan case: with
a migration report already in hand, don't bounce back — follow the
report's routes for the non-Python surfaces instead (see Backstop).
A bare 0 with no advisory is a
hint, not a proof of "nothing to migrate": the check is bounded
(distinctive markers, capped file read), so for anything beyond a
pure PyTorch repo, confirm with cuda-to-xpu-migration's full
inventory scan.

### 2. Mechanical pass

```sh
for t in device_string cuda_to_xpu dot_cuda imports dist_backend amp_autocast amp_gradscaler; do
    python3 path/to/xpu_port_rewrite.py --transform "$t" --path .
done
```

`--check` previews the diff without writing. Read
`references/cuda_to_xpu_whitelist.md` before running the mechanical
pass to understand what `cuda_to_xpu` rewrites and why CUDA-only
attrs are flagged instead.

### 3. Re-scan (gate)

```sh
python3 path/to/xpu_port_scan.py . > findings.after_mechanical.json
```

Mechanical bucket must be 0. If not, a transform missed something
— don't proceed until resolved.

### 4. Semantic + escalate edits

Read `references/semantic-patterns.md` before editing any semantic
finding. Hand-edit each `semantic` finding per that reference. For
each `escalate`: replace, document, or stop. Don't invent fixes.

Common semantics on real repos:

- TF32 toggles → **delete**
- `device_type == 'cuda'` runtime gates → **flip to `'xpu'`**
- `flash-attn` / `bitsandbytes` hard imports → **replace with
  XPU-aware fallback**

Full table: `references/semantic-patterns.md`.

### 5. Final scan (gate)

```sh
python3 path/to/xpu_port_scan.py . > findings.final.json
```

Empty = done with edits. Anything left = a missed semantic edit
or an escalation that needs a recorded decision.

An `advisory` block on this final scan is not a failed gate: it
lists the infra surfaces (Dockerfiles, dependency pins, launch
scripts) the Python port intentionally left untouched. It does not
reopen the Python gate. If no migration report is in hand, route to
**cuda-to-xpu-migration**; if a report is already in hand, record
those surfaces as outside the Python port and continue with the
report's routes.

### 6. Verify

```sh
# Write a builder file (avoids shell-quoting issues):
cat > verify_builder.py << 'EOF'
import sys, torch
sys.path.insert(0, ".")
from model import GPT, GPTConfig
m = GPT(GPTConfig(n_layer=2, n_head=2, n_embd=64,
                  block_size=64, vocab_size=128))
x = {"idx": torch.randint(0, 128, (1, 16))}
out = (m, x)
EOF

python3 path/to/xpu_port_verify.py \
    --builder-file verify_builder.py \
    --target-dtype bfloat16 \
    > verify.json
```

`--builder` (inline string) also works but is fragile with shell
quoting for multi-line builders. Prefer `--builder-file` for
anything beyond a one-liner.

Reference is CPU FP64; target is XPU at the chosen dtype.
`PYTORCH_ENABLE_XPU_FALLBACK=0` is set unconditionally at import
time so silent CPU fallback always raises. Without `--no-xpu`, XPU
must be present or the script exits with an error. `--no-xpu` runs
CPU-vs-FP64 — useful on hosts without an Intel GPU but not a
substitute for the target box.

JSON on stdout; builder prints redirected to stderr.

| Target dtype | rtol | atol |
|---|---|---|
| `float32` | 1e-5 | 1e-6 |
| `bfloat16` | 1e-2 | 1e-3 |
| `float16` | 5e-3 | 1e-3 |

`--rtol` / `--atol` to override. A failing verify means either the
port is genuinely off **or** the model isn't deterministic at this
scale (dropout left on, RNG not seeded). Re-run with `--seed 0`.

**No single forward to verify?** Service / microservice repos (an
HTTP API, a Celery worker, a multi-process app) often have no one
`nn.Module` forward pass to build. The FP64 verifier is N/A there —
skip it and confirm correctness with the runtime smoke test from
the serving skill (**vllm-xpu-run** / **sglang-xpu-run** /
**torch-xpu-run**) instead. The final-scan gate still applies.

### 7. Profile (optional)

After correctness passes, if the port is slow, invoke
**torch-xpu-profile** to find the hot op. Common substitutions
in `references/semantic-patterns.md` "After-profile" section.

## Distributed backend

`xccl`, not `ccl` (the legacy `torch_ccl` plugin). The
`dist_backend` transform rewrites `init_process_group(backend="nccl")`
→ `"xccl"` and bare `backend = "nccl"` assignments.

## What this skill does NOT cover

- Assessment-only / migration plan → **cuda-to-xpu-migration**
- CUDA → XPU rule reference → **torch-xpu-run**
- Throughput / TTFT / TPOT → **torch-xpu-bench**
- Op-level slowness → **torch-xpu-profile**, **xpu-profile-unitrace**
- Container / driver setup → **xpu-container-run**, **xpu-discover**
- Custom CUDA C++ extensions, dual-target codebases (out of scope; see "Targets")

## References

- `references/semantic-patterns.md` — what the agent does for each semantic finding
- `references/cuda_to_xpu_whitelist.md` — what `cuda_to_xpu` rewrites and why
- PyTorch XPU: <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>
- PyTorch XCCL: <https://docs.pytorch.org/docs/stable/distributed.html>
- `torch.xpu`: <https://docs.pytorch.org/docs/stable/xpu.html>
- `torch.amp`: <https://docs.pytorch.org/docs/stable/amp.html>
