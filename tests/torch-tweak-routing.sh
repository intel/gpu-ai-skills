#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Routing pins for torch-tweak versus the eager-run, bench, profile,
# and serving skills. Greps the shipped files so a description edit
# cannot hand "tune this model" back to torch-xpu-run or vLLM.
#
# Run from the repo root:
#     bash tests/torch-tweak-routing.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TWEAK="plugins/intel-gpu-ai-skills/skills/torch-tweak/SKILL.md"
RUN="plugins/intel-gpu-ai-skills/skills/torch-xpu-run/SKILL.md"
BENCH="plugins/intel-gpu-ai-skills/skills/torch-xpu-bench/SKILL.md"
PROFILE="plugins/intel-gpu-ai-skills/skills/torch-xpu-profile/SKILL.md"
PLAN="plugins/intel-gpu-ai-skills/skills/xpu-deploy-plan/SKILL.md"
SNIPPETS="plugins/intel-gpu-ai-skills/skills/torch-xpu-bench/references/non-llm-snippets.md"

pass=0
fail=0
ok()  { printf '  [OK]   %s\n' "$1"; pass=$((pass+1)); }
err() { printf '  [FAIL] %s\n' "$1"; fail=$((fail+1)); }

desc_line() {
    awk '/^description:/{print; exit}' "$1"
}

# 1. torch-tweak owns the tune verb and names the backends it searches.
if desc_line "$TWEAK" | grep -q "Tune a PyTorch"; then
    ok "torch-tweak description claims the tune verb"
else
    err "torch-tweak description missing 'Tune a PyTorch'"
fi
if desc_line "$TWEAK" | grep -q "TorchAO, Torch Inductor, and eager"; then
    ok "torch-tweak description names the GPU backends"
else
    err "torch-tweak description missing the GPU backends"
fi
if desc_line "$TWEAK" | grep -q "OpenVINO"; then
    err "torch-tweak description still names OpenVINO"
else
    ok "torch-tweak description does not name OpenVINO"
fi
if desc_line "$TWEAK" | grep -q "Not for vLLM or SGLang serving"; then
    ok "torch-tweak description disclaims vLLM and SGLang serving"
else
    err "torch-tweak description missing the serving disclaimer"
fi
if desc_line "$TWEAK" | grep -q "use torch-xpu-run"; then
    ok "torch-tweak description forwards eager load to torch-xpu-run"
else
    err "torch-tweak description missing the torch-xpu-run forward"
fi
if desc_line "$TWEAK" | grep -q "use torch-xpu-bench"; then
    ok "torch-tweak description forwards measurement to torch-xpu-bench"
else
    err "torch-tweak description missing the torch-xpu-bench forward"
fi

# 2. The eager-run skill must not claim the tune verb. It only points here.
if desc_line "$RUN" | grep -q "Tune a PyTorch"; then
    err "torch-xpu-run description claims the tune verb"
else
    ok "torch-xpu-run description does not claim the tune verb"
fi
if desc_line "$RUN" | grep -q "use torch-tweak"; then
    ok "torch-xpu-run description forwards tuning to torch-tweak"
else
    err "torch-xpu-run description missing 'use torch-tweak'"
fi
if desc_line "$BENCH" | grep -q "use torch-tweak"; then
    ok "torch-xpu-bench description forwards tuning to torch-tweak"
else
    err "torch-xpu-bench description missing 'use torch-tweak'"
fi
if desc_line "$PROFILE" | grep -q "use torch-tweak"; then
    ok "torch-xpu-profile description forwards tuning to torch-tweak"
else
    err "torch-xpu-profile description missing 'use torch-tweak'"
fi

# 3. Bodies that an already-loaded skill will follow.
if awk '/^## What this skill does NOT cover/,0' "$RUN" | grep -q "torch-tweak"; then
    ok "torch-xpu-run body hands tuning off"
else
    err "torch-xpu-run NOT-cover section missing torch-tweak"
fi
if awk '/^## What this skill does NOT cover/,0' "$BENCH" | grep -q "torch-tweak"; then
    ok "torch-xpu-bench body hands tuning off"
else
    err "torch-xpu-bench NOT-cover section missing torch-tweak"
fi
if awk '/^## What this skill does NOT cover/,0' "$PROFILE" | grep -q "torch-tweak"; then
    ok "torch-xpu-profile body hands tuning off"
else
    err "torch-xpu-profile NOT-cover section missing torch-tweak"
fi
if awk '
    /^## Runtime decision table/ { in_table = 1; next }
    in_table && /^## / { exit }
    in_table && /torch-tweak/ { found = 1 }
    END { exit found ? 0 : 1 }
' "$PLAN"; then
    ok "xpu-deploy-plan routes module tuning to torch-tweak"
else
    err "xpu-deploy-plan runtime table missing torch-tweak"
fi
if grep -q "torch-tweak" "$SNIPPETS"; then
    ok "non-llm bench snippets point at torch-tweak"
else
    err "non-llm-snippets.md missing torch-tweak"
fi

# 4. The skill states the load-bearing default: an empty strategy
#    never runs TorchAO. Install commands stay in the upstream guide.
if grep -q "Pass TorchAO, Torch Inductor, and" "$TWEAK"; then
    ok "torch-tweak body requires the explicit backend list"
else
    err "torch-tweak body no longer requires the explicit backend list"
fi
if grep -q "docs/learn/install.md" "$TWEAK"; then
    ok "torch-tweak body points at the install guide"
else
    err "torch-tweak body missing the install guide link"
fi
if grep -q "download.pytorch.org/whl/xpu" "$TWEAK"; then
    err "torch-tweak body still pastes the XPU install command"
else
    ok "torch-tweak body does not paste the XPU install command"
fi
if grep -q "which sizes are the batch" "$TWEAK"; then
    ok "torch-tweak body explains input-shape adaptation"
else
    err "torch-tweak body missing input-shape adaptation"
fi
if grep -q "min_depth" "$TWEAK" && grep -q "tunes those" "$TWEAK"; then
    ok "torch-tweak body tunes each inspected submodule with one strategy"
else
    err "torch-tweak body missing inspect depth and per-submodule tuning"
fi
if grep -q "pre-release" "$TWEAK"; then
    ok "torch-tweak body states the pre-release status"
else
    err "torch-tweak body missing the pre-release status"
fi
if grep -q "https://github.com/intel/torch-tweak" "$TWEAK"; then
    ok "torch-tweak body cites the public repo"
else
    err "torch-tweak body missing the public repo URL"
fi
if grep -q "the user explicitly confirms" "$TWEAK"; then
    ok "torch-tweak body stops before install"
else
    err "torch-tweak body missing the install confirmation stop"
fi
if grep -q 'bare `pip install torch-tweak`' "$TWEAK"; then
    ok "torch-tweak body rejects a bare PyPI install"
else
    err "torch-tweak body missing the bare PyPI install rejection"
fi

# 5. catalog bundle and README section.
if awk '/^tune:/{f=1} f && /^[^ ]/{if(!/^tune:/) exit} f && /torch-tweak/{found=1} END{exit found?0:1}' catalog/bundles.yaml; then
    ok "catalog tune bundle lists torch-tweak"
else
    err "catalog/bundles.yaml tune bundle missing torch-tweak"
fi
if awk '/^### Tune /{f=1} f && /^### /{if(!/^### Tune /) exit} f && /`torch-tweak`/{found=1} END{exit found?0:1}' README.md; then
    ok "README Tune section lists torch-tweak"
else
    err "README.md Tune section missing torch-tweak"
fi

# 6. marketplace.json descriptions match SKILL.md frontmatter for the
#    skills whose routing text changed. Stdlib only.
if python3 - <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(".")
market = json.loads((root / ".claude-plugin/marketplace.json").read_text())
by_name = {p["name"]: p["description"] for p in market["plugins"]}
names = ("torch-tweak", "torch-xpu-run", "torch-xpu-bench", "torch-xpu-profile")
failed = False
for name in names:
    text = (root / "plugins/intel-gpu-ai-skills/skills" / name / "SKILL.md").read_text()
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    block = match.group(1) if match else ""
    desc = re.search(r"^description:\s*(.+)$", block, re.MULTILINE)
    if not desc:
        print(f"missing description in {name}", file=sys.stderr)
        failed = True
        continue
    value = desc.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    value = value.replace(r"\"", '"')
    if by_name.get(name) != value:
        print(f"marketplace description drifted for {name}", file=sys.stderr)
        failed = True
if "torch-tweak" not in by_name:
    print("marketplace.json has no torch-tweak plugin", file=sys.stderr)
    failed = True
sys.exit(1 if failed else 0)
PY
then
    ok "marketplace.json descriptions match the four SKILL.md files"
else
    err "marketplace.json descriptions drifted from SKILL.md"
fi

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
