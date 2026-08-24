#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Offline checks for the xpu-port skill: scanner classification,
# rewriter behaviour (positive + negative), and the end-to-end loop
# (scan -> rewrite -> re-scan empty -> verify pass=true).
#
# Wired into tests/static.sh. Pass -v for verbose output (each section
# prints the synthetic before/after diff).

set -u

cd "$(dirname "$0")/.."

VERBOSE=0
[ "${1:-}" = "-v" ] && VERBOSE=1

PY="${PYTHON:-python3}"
SKILL="plugins/intel-gpu-ai-skills/skills/xpu-port/scripts"

if ! "$PY" -c 'import libcst' 2>/dev/null; then
    echo "SKIP: libcst not importable from $PY (install with: pip install libcst)" >&2
    exit 0
fi
# torch gates ONLY the verifier subtests (Section 4 tail). The scanner and
# rewriter sections (1, 2, 3, 5) need only libcst, so they run regardless —
# otherwise a torch-less host (CI) would skip the whole file and leave the
# route/advisory contract unverified.
HAS_TORCH=1
if ! "$PY" -c 'import torch' 2>/dev/null; then
    HAS_TORCH=0
    echo "NOTE: torch not importable from $PY — verifier subtests will be skipped; scanner/rewriter sections still run." >&2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

pass=0
fail=0
skip=0
note() {
    [ "$VERBOSE" = 1 ] && printf '  %s\n' "$1"
    return 0
}
ok()      { printf '  [OK]   %s\n' "$1"; pass=$((pass + 1)); }
err()     { printf '  [FAIL] %s\n' "$1"; fail=$((fail + 1)); }
skipped() { printf '  [SKIP] %s\n' "$1"; skip=$((skip + 1)); }

# ---------------------------------------------------------------------------
# Section 1 — rewrite checks (positive: input != output)
# ---------------------------------------------------------------------------
printf '\n== Section 1: rewrite checks ==\n'

run_rewrite() {
    local name="$1" transform="$2" input="$3" expected="$4"
    local f="$tmp/$name.py"
    printf '%s' "$input" > "$f"
    "$PY" "$SKILL/xpu_port_rewrite.py" --transform "$transform" --path "$f" \
        >/dev/null 2>&1
    if [ "$(cat "$f")" = "$expected" ]; then
        ok "$name"
        if [ "$VERBOSE" = 1 ]; then
            diff -u <(printf '%s' "$input") "$f" | sed 's/^/      /'
        fi
    else
        err "$name"
        diff -u <(printf '%s' "$expected") "$f" | sed 's/^/      /' | head -20
    fi
}

run_rewrite device_string device_string \
"x = x.to('cuda')
y = x.to(f'cuda:{i}')" \
"x = x.to('xpu')
y = x.to(f'xpu:{i}')"

run_rewrite cuda_to_xpu cuda_to_xpu \
"import torch
torch.cuda.set_device(0)
torch.cuda.synchronize()" \
"import torch
torch.xpu.set_device(0)
torch.xpu.synchronize()"

run_rewrite dist_backend_kwarg dist_backend \
'dist.init_process_group(backend="nccl", rank=0)' \
'dist.init_process_group(backend="xccl", rank=0)'

run_rewrite dist_backend_assign dist_backend \
"backend = 'nccl'
init_process_group(backend=backend)" \
"backend = 'xccl'
init_process_group(backend=backend)"

run_rewrite amp_autocast amp_autocast \
'with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    pass' \
'with torch.amp.autocast("xpu", dtype=torch.bfloat16):
    pass'

run_rewrite amp_autocast_device_type_kwarg amp_autocast \
'with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    pass' \
'with torch.amp.autocast("xpu", dtype=torch.bfloat16):
    pass'

run_rewrite amp_gradscaler amp_gradscaler \
'scaler = torch.cuda.amp.GradScaler()' \
'scaler = torch.amp.GradScaler("xpu")'

run_rewrite imports_safe imports \
"import torch.cuda
from torch.cuda import is_available" \
"import torch.xpu
from torch.xpu import is_available"

run_rewrite dot_cuda_method dot_cuda \
"x = t.cuda()
y = t.cuda(0)" \
"x = t.xpu()
y = t.xpu(0)"

# ---------------------------------------------------------------------------
# Section 2 — rewrite negatives (input MUST equal output)
# ---------------------------------------------------------------------------
printf '\n== Section 2: rewrite negatives (no change expected) ==\n'

run_negative() {
    local name="$1" transform="$2" input="$3"
    local f="$tmp/neg_$name.py"
    printf '%s' "$input" > "$f"
    "$PY" "$SKILL/xpu_port_rewrite.py" --transform "$transform" --path "$f" \
        >/dev/null 2>&1
    if [ "$(cat "$f")" = "$input" ]; then
        ok "$name"
    else
        err "$name (file changed but should not have)"
        diff -u <(printf '%s' "$input") "$f" | sed 's/^/      /' | head -10
    fi
}

# bug-1 lock-in: device_type == 'cuda' must NOT be flipped
run_negative device_type_compare device_string \
"if device_type == 'cuda':
    pass"

# bug-1 lock-in: 'cuda' if 'cuda' in device else 'cpu' must NOT be flipped
run_negative device_label_ternary device_string \
"device_type = 'cuda' if 'cuda' in device else 'cpu'"

# 'cuda' inside a docstring stays
run_negative cuda_in_docstring device_string \
'def f():
    """move tensor to cuda"""
    return 1'

# torch.cuda.utilization is CUDA-only and must NOT be auto-renamed
run_negative cuda_only_attr cuda_to_xpu \
"import torch
print(torch.cuda.utilization())"

# from torch.cuda.amp import ... must NOT be auto-flipped (no torch.xpu.amp)
run_negative cuda_amp_import imports \
"from torch.cuda.amp import autocast"

# torch.cuda(0) is the device-shorthand call — must NOT be flipped
run_negative torch_cuda_shorthand dot_cuda \
"x = torch.cuda(0)"

# ---------------------------------------------------------------------------
# Section 3 — scanner classification
# ---------------------------------------------------------------------------
printf '\n== Section 3: scanner classification ==\n'

scan_file="$tmp/scan_sample.py"
cat > "$scan_file" <<'PY'
import torch
import torch.cuda                       # mechanical: cuda_import
from torch.cuda import is_available     # mechanical: cuda_import_from
from torch.cuda.amp import autocast     # semantic: cuda_import_from_no_xpu

def setup():
    torch.cuda.set_device(0)            # mechanical: cuda_attr
    print(torch.cuda.utilization())     # escalate: cuda_attr_no_xpu
    return torch.cuda.is_available()

def step(x, device):
    if device_type == 'cuda':           # semantic: device_label_compare
        x = x.to('cuda')                # mechanical: device_string
        x = x.cuda()                    # mechanical: dot_cuda
    return x

dist.init_process_group(backend='nccl') # mechanical: dist_backend
torch.backends.cuda.matmul.allow_tf32 = True   # semantic: tf32_toggle
torch.compile(model)                            # semantic: torch_compile
activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]  # semantic: profiler_activity_cuda
PY

scan_out="$tmp/scan.json"
"$PY" "$SKILL/xpu_port_scan.py" "$scan_file" > "$scan_out"

assert_bucket() {
    local bucket="$1" expected="$2"
    local actual
    actual=$("$PY" -c "
import json
d = json.load(open('$scan_out'))
print(d['by_bucket'].get('$bucket', 0))
")
    if [ "$actual" = "$expected" ]; then
        ok "scanner $bucket=$expected"
    else
        err "scanner $bucket: expected $expected got $actual"
    fi
}

assert_category() {
    local category="$1"
    local count
    count=$("$PY" -c "
import json
d = json.load(open('$scan_out'))
print(sum(1 for f in d['findings'] if f['category'] == '$category'))
")
    if [ "$count" -ge 1 ]; then
        ok "scanner category $category present"
    else
        err "scanner category $category missing"
    fi
}

assert_bucket mechanical 7
assert_bucket semantic 5
assert_bucket escalate 1
assert_category device_label_compare        # bug-1 lock-in
assert_category cuda_attr_no_xpu            # escalate path
assert_category cuda_import_from_no_xpu     # cuda.amp gate
assert_category torch_compile               # non-namespace CUDA construct
assert_category profiler_activity_cuda      # non-namespace CUDA construct

# ---------------------------------------------------------------------------
# Section 4 — end-to-end loop (the contract from SKILL.md)
# ---------------------------------------------------------------------------
printf '\n== Section 4: end-to-end loop ==\n'

e2e="$tmp/e2e.py"
cat > "$e2e" <<'PY'
import torch
import torch.cuda
from torch.cuda import is_available
import torch.distributed as dist


def setup():
    torch.cuda.set_device(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.init_process_group(backend="nccl")


def go(x):
    x = x.to("cuda")
    x = x.cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        x = x * 2
    scaler = torch.cuda.amp.GradScaler()
    return x, scaler
PY

e2e_target="$tmp/e2e_target.py"
cat > "$e2e_target" <<'PY'
import torch
import torch.xpu
from torch.xpu import is_available
import torch.distributed as dist


def setup():
    torch.xpu.set_device(0)
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    dist.init_process_group(backend="xccl")


def go(x):
    x = x.to("xpu")
    x = x.xpu()
    with torch.amp.autocast("xpu", dtype=torch.bfloat16):
        x = x * 2
    scaler = torch.amp.GradScaler("xpu")
    return x, scaler
PY

# initial scan: there should be mechanical findings to handle
initial=$("$PY" "$SKILL/xpu_port_scan.py" "$e2e" \
    | "$PY" -c "import json,sys; print(json.load(sys.stdin)['by_bucket'].get('mechanical', 0))")
if [ "$initial" -gt 0 ]; then
    note "initial scan: $initial mechanical findings"
    ok "initial scan has mechanical work"
else
    err "initial scan should report mechanical findings"
fi

# run all transforms in sequence
for t in device_string cuda_to_xpu dot_cuda imports dist_backend amp_autocast amp_gradscaler; do
    "$PY" "$SKILL/xpu_port_rewrite.py" --transform "$t" --path "$e2e" >/dev/null 2>&1
done

# rewritten file must match the expected port byte-for-byte
if diff -q "$e2e_target" "$e2e" >/dev/null 2>&1; then
    ok "rewritten file matches expected port"
    if [ "$VERBOSE" = 1 ]; then
        diff -u "$tmp/e2e_target.py" "$e2e" | sed 's/^/      /' | head -40 || true
    fi
else
    err "rewritten file does not match expected port"
    diff -u "$e2e_target" "$e2e" | sed 's/^/      /' | head -40
fi

# re-scan: mechanical bucket must be empty (the SKILL.md gate)
remaining=$("$PY" "$SKILL/xpu_port_scan.py" "$e2e" \
    | "$PY" -c "import json,sys; print(json.load(sys.stdin)['by_bucket'].get('mechanical', 0))")
if [ "$remaining" = "0" ]; then
    ok "final scan mechanical bucket is empty"
else
    err "final scan still has $remaining mechanical findings"
fi

# idempotency: running all transforms a second time must produce no further changes
cp "$e2e" "$tmp/e2e_before_second_pass.py"
for t in device_string cuda_to_xpu dot_cuda imports dist_backend amp_autocast amp_gradscaler; do
    "$PY" "$SKILL/xpu_port_rewrite.py" --transform "$t" --path "$e2e" >/dev/null 2>&1
done
if diff -q "$tmp/e2e_before_second_pass.py" "$e2e" >/dev/null 2>&1; then
    ok "transforms are idempotent (second pass produces no changes)"
else
    err "transforms are not idempotent (second pass changed the file)"
    diff -u "$tmp/e2e_before_second_pass.py" "$e2e" | sed 's/^/      /' | head -20
fi

if [ "$HAS_TORCH" = 0 ]; then
    skipped "verifier subtests (torch not available)"
else
# verifier smoke: tiny model, --no-xpu, must report pass=true
verify_out="$tmp/verify.json"
"$PY" "$SKILL/xpu_port_verify.py" \
    --target-dtype float32 --no-xpu \
    --builder 'import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x).relu()
m = M()
out = (m, {"x": torch.randn(2, 8)})' \
    > "$verify_out" 2>/dev/null

if "$PY" -c "
import json, sys
d = json.load(open('$verify_out'))
sys.exit(0 if d.get('pass') is True else 1)
" 2>/dev/null; then
    ok "verifier reports pass=true on the tiny model"
else
    err "verifier did not report pass=true"
    cat "$verify_out" | head -20
fi

# verifier stdout capture: builder with a print must still produce parseable JSON
verify_capture_out="$tmp/verify_capture.json"
"$PY" "$SKILL/xpu_port_verify.py" \
    --target-dtype float32 --no-xpu \
    --builder 'import torch
import torch.nn as nn
print("loading model...")
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x).relu()
m = M()
out = (m, {"x": torch.randn(2, 8)})' \
    > "$verify_capture_out" 2>/dev/null

if "$PY" -c "
import json, sys
d = json.load(open('$verify_capture_out'))
sys.exit(0 if d.get('pass') is True else 1)
" 2>/dev/null; then
    ok "verifier stdout stays clean JSON when builder prints to stdout"
else
    err "verifier stdout corrupted when builder prints (stdout capture broken)"
    cat "$verify_capture_out" | head -5
fi

# verifier: missing XPU without --no-xpu must exit non-zero (not silently pass)
verify_noxpu_out="$tmp/verify_noxpu.json"
"$PY" "$SKILL/xpu_port_verify.py" \
    --target-dtype float32 \
    --builder 'import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self): super().__init__(); self.fc = nn.Linear(4, 2)
    def forward(self, x): return self.fc(x)
m = M()
out = (m, {"x": torch.randn(1, 4)})' \
    > "$verify_noxpu_out" 2>/dev/null
noxpu_exit=$?
# On a machine without XPU, exit must be non-zero; on a machine WITH XPU this
# path runs normally (exit 0 or 1 depending on correctness) so skip the check.
if "$PY" -c "import torch; import sys; sys.exit(0 if hasattr(torch,'xpu') and torch.xpu.is_available() else 1)" 2>/dev/null; then
    ok "verifier no-xpu check skipped (XPU is present)"
else
    if [ "$noxpu_exit" -ne 0 ]; then
        ok "verifier exits non-zero when XPU unavailable and --no-xpu not set"
    else
        err "verifier silently passed without XPU and without --no-xpu"
    fi
fi

# verifier --builder-file: reading the builder from a file must work like --builder
builder_file="$tmp/builder_from_file.py"
cat > "$builder_file" <<'BUILDER'
import torch
import torch.nn as nn
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x).relu()
m = M()
out = (m, {"x": torch.randn(2, 8)})
BUILDER
verify_file_out="$tmp/verify_builder_file.json"
"$PY" "$SKILL/xpu_port_verify.py" \
    --target-dtype float32 --no-xpu \
    --builder-file "$builder_file" \
    > "$verify_file_out" 2>/dev/null

if "$PY" -c "
import json, sys
d = json.load(open('$verify_file_out'))
sys.exit(0 if d.get('pass') is True else 1)
" 2>/dev/null; then
    ok "verifier reads builder from --builder-file and reports pass=true"
else
    err "verifier --builder-file path did not report pass=true"
    cat "$verify_file_out" | head -20
fi

# verifier: neither --builder nor --builder-file given must exit non-zero (argparse guard)
"$PY" "$SKILL/xpu_port_verify.py" --target-dtype float32 --no-xpu >/dev/null 2>&1
noargs_exit=$?
if [ "$noargs_exit" -ne 0 ]; then
    ok "verifier exits non-zero when neither --builder nor --builder-file is given"
else
    err "verifier did not error when both --builder and --builder-file are missing"
fi
fi  # end HAS_TORCH guard for verifier subtests

# ---------------------------------------------------------------------------
# Section 5 — empty-scan advisory + route field
# ---------------------------------------------------------------------------
printf '\n== Section 5: empty-scan advisory + route field ==\n'

# 5a — route field: profiler_activity_cuda must carry route=torch-xpu-profile;
# a mechanical device_string finding must carry route=null (xpu-port handles it).
route_out="$tmp/route.json"
"$PY" "$SKILL/xpu_port_scan.py" "$scan_file" > "$route_out"
if "$PY" -c "
import json, sys
d = json.load(open('$route_out'))
prof = [f for f in d['findings'] if f['category'] == 'profiler_activity_cuda']
dev  = [f for f in d['findings'] if f['category'] == 'device_string']
ok = (d['schema_version'] == '1.1'
      and all('route' in f for f in d['findings'])
      and prof and prof[0]['route'] == 'torch-xpu-profile'
      and dev and dev[0]['route'] is None)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    ok "route field: profiler routes to torch-xpu-profile, mechanical is null"
else
    err "route field wrong or missing"
fi

# 5b — empty-scan advisory: a repo with 0 Python CUDA sites but NVIDIA infra
# markers must emit an advisory block naming cuda-to-xpu-migration.
infra_repo="$tmp/infra_repo"
mkdir -p "$infra_repo"
cat > "$infra_repo/app.py" <<'PY'
import requests  # pure-python API client, no torch.cuda
def call(): return requests.post("https://integrate.api.nvidia.com/v1")
PY
cat > "$infra_repo/Dockerfile" <<'DOCKER'
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
RUN pip install flash-attn
DOCKER
advisory_out="$tmp/advisory.json"
"$PY" "$SKILL/xpu_port_scan.py" "$infra_repo" > "$advisory_out" 2> "$advisory_out.err"
if "$PY" -c "
import json, sys
d = json.load(open('$advisory_out'))
a = d.get('advisory')
ok = (d['total_findings'] == 0
      and a is not None
      and a['kind'] == 'empty_scan_infra_markers'
      and a['routes'] == ['cuda-to-xpu-migration', 'xpu-deploy-plan']
      and any('Dockerfile' in f for f in a['files_with_markers']))
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    ok "empty-scan advisory fires on NVIDIA-stacked repo with 0 Python CUDA sites"
else
    err "empty-scan advisory missing or malformed"
    cat "$advisory_out" | head -30
    sed 's/^/      /' "$advisory_out.err"
fi

# stderr contract: the advisory is also printed to stderr with the
# 'xpu-port:' prefix and must name the assessment skill to route to.
if grep -q '^xpu-port: ' "$advisory_out.err" \
        && grep -q 'cuda-to-xpu-migration' "$advisory_out.err"; then
    ok "advisory stderr carries the xpu-port: prefix and names cuda-to-xpu-migration"
else
    err "advisory stderr contract broken (xpu-port: prefix or skill name missing)"
    sed 's/^/      /' "$advisory_out.err"
fi

# 5c — no false advisory: a repo WITH Python CUDA findings must NOT emit an advisory.
noadv_repo="$tmp/noadv_repo"
mkdir -p "$noadv_repo"
cat > "$noadv_repo/train.py" <<'PY'
import torch
torch.cuda.set_device(0)
DOCKER_NONE = "FROM nvidia/cuda"  # marker present but findings exist
PY
noadv_out="$tmp/noadv.json"
"$PY" "$SKILL/xpu_port_scan.py" "$noadv_repo" > "$noadv_out" 2>/dev/null
if "$PY" -c "
import json, sys
d = json.load(open('$noadv_out'))
sys.exit(0 if (d['total_findings'] > 0 and 'advisory' not in d) else 1)
" 2>/dev/null; then
    ok "no advisory when Python CUDA findings exist"
else
    err "advisory wrongly emitted despite non-empty findings"
fi

# 5c2 — custom CUDA kernels must be classified as CUDA source, not parser
# bookkeeping. Because the .cu file is now a real CUDA finding, the
# empty-scan advisory must not fire.
cu_repo="$tmp/cu_repo"
mkdir -p "$cu_repo"
printf '__global__ void kernel() {}\n' > "$cu_repo/kernel.cu"
printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$cu_repo/Dockerfile"
cu_out="$tmp/cu.json"
"$PY" "$SKILL/xpu_port_scan.py" "$cu_repo" > "$cu_out" 2>/dev/null
if "$PY" -c "
import json, sys
d = json.load(open('$cu_out'))
cats = [f['category'] for f in d['findings']]
ok = ('cuda_source_file' in cats
      and 'unparseable' not in cats
      and 'advisory' not in d)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    ok "custom .cu file is explicit cuda_source_file escalation"
else
    err "custom .cu file was not classified as cuda_source_file"
    head -30 "$cu_out" | sed 's/^/      /'
fi

# helper: assert whether the advisory fires for a repo directory.
# Extra arguments after `want` are passed through to the scanner
# (e.g. --exclude <dir>). Scanner stderr is kept in "$out.err" so a
# failure is diagnosable instead of discarded.
assert_advisory() {
    local name="$1" repo="$2" want="$3"  # want = yes|no
    shift 3
    local out="$tmp/adv_$name.json"
    "$PY" "$SKILL/xpu_port_scan.py" "$repo" "$@" > "$out" 2> "$out.err"
    local got
    got=$("$PY" -c "import json,sys; d=json.load(open('$out')); print('yes' if 'advisory' in d else 'no')" 2>/dev/null)
    if [ "$got" = "$want" ]; then
        ok "$name (advisory=$got)"
    else
        err "$name: expected advisory=$want got=$got"
        head -30 "$out" | sed 's/^/      /'
        sed 's/^/      /' "$out.err"
    fi
}

# 5d — word-boundary: a benign 'barracuda' dependency must NOT fire (no 'cuda' substring match).
wb_repo="$tmp/wb_repo"; mkdir -p "$wb_repo"
printf 'print(1)\n' > "$wb_repo/app.py"
printf 'barracuda-agent==1.2.0\nrequests==2.31.0\n' > "$wb_repo/requirements.txt"
assert_advisory word_boundary_barracuda "$wb_repo" no

# 5e — extra infra surfaces: Containerfile / Makefile / setup.cfg must fire.
surf_repo="$tmp/surf_repo"; mkdir -p "$surf_repo"
printf 'x=1\n' > "$surf_repo/app.py"
printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$surf_repo/Containerfile"
printf 'run:\n\tdocker run --gpus all img\n' > "$surf_repo/Makefile"
printf '[options]\ninstall_requires =\n    nvidia-ml-py\n' > "$surf_repo/setup.cfg"
assert_advisory extra_infra_surfaces "$surf_repo" yes
# ... and each of the three surfaces must individually carry a hit.
if "$PY" -c "
import json, sys
a = json.load(open('$tmp/adv_extra_infra_surfaces.json')).get('advisory') or {}
files = a.get('files_with_markers', [])
sys.exit(0 if all(f in files for f in ('Containerfile', 'Makefile', 'setup.cfg')) else 1)
" 2>/dev/null; then
    ok "Containerfile, Makefile and setup.cfg each carry a marker hit"
else
    err "a 5e surface is missing from files_with_markers"
    head -30 "$tmp/adv_extra_infra_surfaces.json" | sed 's/^/      /'
fi

# 5f — inline-.py NIM client (distinctive marker in source, openai dep, no Dockerfile) must fire.
nim_repo="$tmp/nim_repo"; mkdir -p "$nim_repo"
printf 'openai==1.30.0\n' > "$nim_repo/requirements.txt"
printf 'from openai import OpenAI\nc = OpenAI(base_url="https://integrate.api.nvidia.com/v1")\n' > "$nim_repo/main.py"
assert_advisory inline_py_nim_endpoint "$nim_repo" yes

# 5g — .py with generic 'cuda'/'nvidia' ONLY in a comment must NOT fire (distinctive-only in .py).
comment_repo="$tmp/comment_repo"; mkdir -p "$comment_repo"
printf 'requests==2.0\n' > "$comment_repo/requirements.txt"
printf '# once ran on cuda / nvidia hardware, now cloud\nx = 1\n' > "$comment_repo/app.py"
assert_advisory py_comment_generic_marker "$comment_repo" no

# 5h — vendored dir: markers only inside node_modules must NOT fire.
vend_repo="$tmp/vend_repo"; mkdir -p "$vend_repo/node_modules/pkg"
printf 'x=1\n' > "$vend_repo/app.py"
printf 'FROM nvidia/cuda:12.1\n' > "$vend_repo/node_modules/pkg/Dockerfile"
assert_advisory vendored_dir_excluded "$vend_repo" no

# 5i — version-glued 'cuda': the canonical PyTorch CUDA base-image tag and a
# conda cudatoolkit pin, as the ONLY markers, must fire. Regression for the
# trailing-boundary miss ('cuda12.4' / 'cudatoolkit' did not match).
baseimg_repo="$tmp/baseimg_repo"; mkdir -p "$baseimg_repo"
printf 'x=1\n' > "$baseimg_repo/app.py"
printf 'FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel\n' > "$baseimg_repo/Dockerfile"
printf 'dependencies:\n  - cudatoolkit=11.8\n' > "$baseimg_repo/environment.yml"
assert_advisory cuda_version_glued_baseimage "$baseimg_repo" yes

# 5j — 'cupy' substring must not fire on the common word 'occupy' in .py source,
# but MUST fire on a real cupy dependency. Regression for the boundary-less
# distinctive-substring match.
occupy_repo="$tmp/occupy_repo"; mkdir -p "$occupy_repo"
printf 'requests==2.0\n' > "$occupy_repo/requirements.txt"
printf '# worker threads occupy the pool until the job completes\nx = 1\n' > "$occupy_repo/worker.py"
assert_advisory cupy_no_occupy_false_fire "$occupy_repo" no

# Bare 'cupy' pin (no 'cuda' substring) — isolates the cupy boundary match
# from the generic word markers.
cupy_repo="$tmp/cupy_repo"; mkdir -p "$cupy_repo"
printf 'x=1\n' > "$cupy_repo/app.py"
printf 'cupy==13.0.0\nrequests==2.0\n' > "$cupy_repo/requirements.txt"
assert_advisory cupy_real_dep_fires "$cupy_repo" yes

# 5k — '--gpus' as a plain argparse flag in .py source must NOT fire: on a
# device-agnostic repo it is a training knob, not a docker launch signal.
gpus_py_repo="$tmp/gpus_py_repo"; mkdir -p "$gpus_py_repo"
printf 'import argparse\np = argparse.ArgumentParser()\np.add_argument("--gpus", type=int, default=1)\n' \
    > "$gpus_py_repo/train.py"
assert_advisory argparse_gpus_in_py "$gpus_py_repo" no

# 5l — the same token in a shell launcher IS a 'docker run --gpus' signal.
gpus_sh_repo="$tmp/gpus_sh_repo"; mkdir -p "$gpus_sh_repo"
printf 'x=1\n' > "$gpus_sh_repo/app.py"
printf '#!/bin/sh\ndocker run --gpus all img\n' > "$gpus_sh_repo/run.sh"
assert_advisory docker_gpus_in_sh "$gpus_sh_repo" yes

# 5m — an unreadable candidate file must leave a trace (probe_skipped_unreadable
# in JSON + stderr note), not silently look like a clean repo. Root ignores
# file modes, so the fixture is meaningless there.
if [ "$(id -u)" -ne 0 ]; then
    unread_repo="$tmp/unread_repo"; mkdir -p "$unread_repo"
    printf 'x=1\n' > "$unread_repo/app.py"
    printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$unread_repo/Dockerfile"
    chmod 000 "$unread_repo/Dockerfile"
    unread_out="$tmp/unread.json"
    "$PY" "$SKILL/xpu_port_scan.py" "$unread_repo" > "$unread_out" 2> "$unread_out.err"
    chmod 644 "$unread_repo/Dockerfile"  # restore before the EXIT-trap rm -rf
    if "$PY" -c "
import json, sys
d = json.load(open('$unread_out'))
sys.exit(0 if d.get('probe_skipped_unreadable', 0) >= 1 else 1)
" 2>/dev/null && grep -q 'could not read' "$unread_out.err"; then
        ok "unreadable probe file leaves JSON + stderr trace"
    else
        err "unreadable probe file left no trace (silent swallow)"
        head -10 "$unread_out" | sed 's/^/      /'
        sed 's/^/      /' "$unread_out.err"
    fi
else
    skipped "unreadable-file trace (running as root, file modes ignored)"
fi

# 5n — read cap regression: a marker at the TOP of a >64KB file must still fire.
cap_top_repo="$tmp/cap_top_repo"; mkdir -p "$cap_top_repo"
printf 'x=1\n' > "$cap_top_repo/app.py"
"$PY" -c "
lines = ['nvidia-cublas-cu12==12.1.3\n'] + ['filler-pkg-%d==1.0\n' % i for i in range(4000)]
open('$cap_top_repo/requirements.txt', 'w').write(''.join(lines))
"
assert_advisory cap_marker_at_top "$cap_top_repo" yes

# 5o — a marker entirely PAST the 64KB cap is not seen. Accepted miss; this
# fixture documents the bound rather than asserting completeness.
cap_past_repo="$tmp/cap_past_repo"; mkdir -p "$cap_past_repo"
printf 'x=1\n' > "$cap_past_repo/app.py"
"$PY" -c "
lines = ['filler-pkg-%d==1.0\n' % i for i in range(4000)] + ['nvidia-cublas-cu12==12.1.3\n']
open('$cap_past_repo/requirements.txt', 'w').write(''.join(lines))
"
assert_advisory cap_marker_past_cap "$cap_past_repo" no

# 5p — exclusion must match path parts relative to the scan root: a repo
# checked out under a parent named 'build' (or venv, dist, ...) must still
# be probed, while excluded dirs INSIDE the repo stay excluded.
f5_repo="$tmp/build/nested_repo"; mkdir -p "$f5_repo/node_modules/pkg"
printf 'x=1\n' > "$f5_repo/app.py"
printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$f5_repo/Dockerfile"
printf 'FROM nvcr.io/other/image:1.0\n' > "$f5_repo/node_modules/pkg/Dockerfile"
f5_out="$tmp/f5.json"
"$PY" "$SKILL/xpu_port_scan.py" "$f5_repo" > "$f5_out" 2>/dev/null
if "$PY" -c "
import json, sys
d = json.load(open('$f5_out'))
a = d.get('advisory')
ok = (a is not None
      and 'Dockerfile' in a['files_with_markers']
      and all('node_modules' not in f for f in a['files_with_markers']))
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    ok "repo under a 'build' parent is probed; node_modules inside stays excluded"
else
    err "exclusion matched absolute path parts (repo under build/ went dark)"
    head -30 "$f5_out" | sed 's/^/      /'
fi

# 5q — walk() shares the exclusion bug: .py files in a repo under a 'build'
# parent must still be scanned for CUDA sites.
f5_scan_repo="$tmp/build/nested_scan_repo"; mkdir -p "$f5_scan_repo"
printf 'import torch\ntorch.cuda.set_device(0)\n' > "$f5_scan_repo/train.py"
f5_scan_count=$("$PY" "$SKILL/xpu_port_scan.py" "$f5_scan_repo" 2>/dev/null \
    | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['files_scanned'], d['total_findings'])")
if [ "$f5_scan_count" = "1 1" ]; then
    ok "walk() scans a repo under a 'build' parent (files_scanned=1)"
else
    err "walk() excluded the repo by its parent dir (got: $f5_scan_count)"
fi

# 5r — generic words in a prose .txt must NOT fire: only files whose *name*
# marks a dependency pin / build config earn cuda/nvidia word matching.
prose_repo="$tmp/prose_repo"; mkdir -p "$prose_repo"
printf 'x=1\n' > "$prose_repo/app.py"
printf 'Trained on NVIDIA A100 GPUs with CUDA 12.\n' > "$prose_repo/README.txt"
assert_advisory prose_txt_generic_word "$prose_repo" no

# 5s — a requirements-variant name keeps generic-word matching.
reqdev_repo="$tmp/reqdev_repo"; mkdir -p "$reqdev_repo"
printf 'x=1\n' > "$reqdev_repo/app.py"
printf 'nvidia-cublas-cu12==12.1.3\n' > "$reqdev_repo/requirements-dev.txt"
assert_advisory requirements_dev_pin "$reqdev_repo" yes

# 5t — lockfiles are dependency pins: *.lock earns generic-word matching.
lock_repo="$tmp/lock_repo"; mkdir -p "$lock_repo"
printf 'x=1\n' > "$lock_repo/app.py"
printf '[[package]]\nname = "nvidia-cublas-cu12"\nversion = "12.1.3"\n' > "$lock_repo/poetry.lock"
assert_advisory lockfile_pin "$lock_repo" yes

# 5u — setup.py carve-out: it is .py source but matched as a dependency
# surface by name, so even a generic-word comment fires. Pins the
# behavior the find_infra_markers docstring documents.
setup_py_repo="$tmp/setup_py_repo"; mkdir -p "$setup_py_repo"
printf 'x=1\n' > "$setup_py_repo/app.py"
printf '# cuda builds removed\nfrom setuptools import setup\nsetup(name="pkg")\n' > "$setup_py_repo/setup.py"
assert_advisory setup_py_generic_word "$setup_py_repo" yes

# 5v — UTF-16 dependency pins (pip freeze under Windows PowerShell 5.x)
# interleave NULs that survive decode(errors='ignore'); the probe must
# strip them or 'cuda' reads as 'c\0u\0d\0a' and never matches.
utf16_repo="$tmp/utf16_repo"; mkdir -p "$utf16_repo"
printf 'x=1\n' > "$utf16_repo/app.py"
"$PY" -c "
open('$utf16_repo/requirements.txt', 'wb').write(
    'nvidia-cublas-cu12==12.1.3\ncudatoolkit==11.8\n'.encode('utf-16'))
"
assert_advisory utf16_requirements_pins "$utf16_repo" yes

# 5w — synthetic unreadable/unparseable findings are scan bookkeeping, not
# CUDA sites: they must not suppress the advisory.
unparse_repo="$tmp/unparse_repo"; mkdir -p "$unparse_repo"
printf 'print "py2 syntax"\n' > "$unparse_repo/legacy.py"
printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\nRUN pip install tensorrt\n' > "$unparse_repo/Dockerfile"
unparse_out="$tmp/unparse.json"
"$PY" "$SKILL/xpu_port_scan.py" "$unparse_repo" > "$unparse_out" 2>/dev/null
if "$PY" -c "
import json, sys
d = json.load(open('$unparse_out'))
cats = [f['category'] for f in d['findings']]
sys.exit(0 if (cats == ['unparseable'] and 'advisory' in d) else 1)
" 2>/dev/null; then
    ok "advisory fires alongside a synthetic unparseable finding"
else
    err "synthetic unparseable finding suppressed the advisory"
    head -30 "$unparse_out" | sed 's/^/      /'
fi

# 5x — notebooks: a distinctive marker in an .ipynb cell must fire (the
# probe reads notebooks as text, distinctive-only).
nb_marker_repo="$tmp/nb_marker_repo"; mkdir -p "$nb_marker_repo"
cat > "$nb_marker_repo/infer.ipynb" <<'NB'
{"cells":[{"cell_type":"code","source":["import tensorrt\n","engine = tensorrt.Runtime(logger)\n"]}]}
NB
assert_advisory notebook_distinctive_marker "$nb_marker_repo" yes

# 5y — notebooks whose only CUDA content is prose in a markdown cell must
# NOT fire (no generic-word matching in .ipynb).
nb_prose_repo="$tmp/nb_prose_repo"; mkdir -p "$nb_prose_repo"
cat > "$nb_prose_repo/README.ipynb" <<'NB'
{"cells":[{"cell_type":"markdown","source":["This demo runs on NVIDIA GPUs with CUDA 12.\n"]}]}
NB
assert_advisory notebook_prose_only "$nb_prose_repo" no

# 5z — files_with_markers is capped at 10; a truncated list must say so.
capcount_repo="$tmp/capcount_repo"; mkdir -p "$capcount_repo"
printf 'x=1\n' > "$capcount_repo/app.py"
for i in $(seq 1 12); do
    printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$capcount_repo/Dockerfile.$i"
done
capcount_out="$tmp/capcount.json"
"$PY" "$SKILL/xpu_port_scan.py" "$capcount_repo" > "$capcount_out" 2>/dev/null
if "$PY" -c "
import json, sys
a = json.load(open('$capcount_out')).get('advisory') or {}
sys.exit(0 if (len(a.get('files_with_markers', [])) == 10
               and a.get('list_capped') is True) else 1)
" 2>/dev/null; then
    ok "files_with_markers capped at 10 with list_capped indicator"
else
    err "cap indicator missing or wrong on a 12-marker repo"
    head -30 "$capcount_out" | sed 's/^/      /'
fi

# 5z2 — files_with_markers is deterministic when capped: the probe sorts
# the collected hits, so the truncated list is stable regardless of
# filesystem rglob order. Reuses the 12-marker repo from 5z: run twice
# and assert the list is sorted and identical across runs.
capcount_out2="$tmp/capcount2.json"
"$PY" "$SKILL/xpu_port_scan.py" "$capcount_repo" > "$capcount_out2" 2>/dev/null
if "$PY" -c "
import json, sys
a1 = json.load(open('$capcount_out')).get('advisory') or {}
a2 = json.load(open('$capcount_out2')).get('advisory') or {}
f1 = a1.get('files_with_markers', [])
f2 = a2.get('files_with_markers', [])
sys.exit(0 if (f1 == sorted(f1) and f1 == f2 and len(f1) == 10) else 1)
" 2>/dev/null; then
    ok "capped files_with_markers is sorted and stable across runs"
else
    err "capped files_with_markers is unsorted or unstable across runs"
    { echo "run1:"; cat "$capcount_out"; echo "run2:"; cat "$capcount_out2"; } \
        | head -40 | sed 's/^/      /'
fi

# 5aa — 'Dockerfile.gpu' exercises the name prefix-match branch.
dfgpu_repo="$tmp/dfgpu_repo"; mkdir -p "$dfgpu_repo"
printf 'x=1\n' > "$dfgpu_repo/app.py"
printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$dfgpu_repo/Dockerfile.gpu"
assert_advisory dockerfile_gpu_prefix "$dfgpu_repo" yes

# 5ab — minimal Pipfile (exact-name branch, dependency-pin tier).
pipfile_repo="$tmp/pipfile_repo"; mkdir -p "$pipfile_repo"
printf 'x=1\n' > "$pipfile_repo/app.py"
printf '[packages]\nnvidia-cublas-cu12 = "*"\n' > "$pipfile_repo/Pipfile"
assert_advisory pipfile_dep_pin "$pipfile_repo" yes

# 5ac — uppercase marker text: matching is case-insensitive (pins .lower()).
upper_repo="$tmp/upper_repo"; mkdir -p "$upper_repo"
printf 'x=1\n' > "$upper_repo/app.py"
printf 'FROM NVCR.IO/NVIDIA/PYTORCH:24.01-PY3\n' > "$upper_repo/Dockerfile"
assert_advisory uppercase_marker "$upper_repo" yes

# 5ad — a caller-supplied --exclude reaches the probe: markers that live
# only inside the excluded dir must not fire.
callerex_repo="$tmp/callerex_repo"; mkdir -p "$callerex_repo/legacy"
printf 'x=1\n' > "$callerex_repo/app.py"
printf 'FROM nvcr.io/nvidia/pytorch:24.01-py3\n' > "$callerex_repo/legacy/Dockerfile"
assert_advisory caller_exclude_reaches_probe "$callerex_repo" no --exclude legacy

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
printf '\n%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ]
