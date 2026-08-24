#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate agents/AGENTS.md bundle index from skill frontmatter.

Mirrors the huggingface/skills convention: a single AGENTS.md that
lists every skill with its description, for agents that consume
AGENTS.md instead of the per-skill SKILL.md (Codex without the
skills extension, generic AGENTS.md-style agents, etc.).

Run from the repo root: `python3 scripts/generate_agents.py`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO / "plugins" / "intel-gpu-ai-skills" / "skills"
OUT = REPO / "agents" / "AGENTS.md"


def clean_frontmatter_value(value: str) -> str:
    """Normalize a one-line YAML-ish frontmatter value for AGENTS.md."""
    value = " ".join(value.split())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.replace(r"\"", '"')


def parse_frontmatter(text: str) -> dict[str, str]:
    """Return name + description from a SKILL.md's YAML frontmatter."""
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    fm = {}
    block = m.group(1)
    name_m = re.search(r"^name:\s*(.+)$", block, re.MULTILINE)
    desc_m = re.search(r"^description:\s*(.+?)(?=\n[a-z][a-z-]*:|\Z)",
                       block, re.MULTILINE | re.DOTALL)
    if name_m:
        fm["name"] = clean_frontmatter_value(name_m.group(1))
    if desc_m:
        fm["description"] = clean_frontmatter_value(desc_m.group(1))
    return fm


def main() -> int:
    skills = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    entries: list[tuple[str, str, Path]] = []
    for d in skills:
        sk = d / "SKILL.md"
        if not sk.exists():
            continue
        fm = parse_frontmatter(sk.read_text())
        if "name" not in fm or "description" not in fm:
            print(f"warn: {sk} missing name or description", file=sys.stderr)
            continue
        entries.append((fm["name"], fm["description"], sk.relative_to(REPO)))

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        f.write("<skills>\n\n")
        f.write("You have additional SKILLs documented in directories ")
        f.write('containing a "SKILL.md" file.\n\n')
        f.write("These skills are:\n")
        for name, _, path in entries:
            f.write(f' - {name} -> "{path}"\n')
        f.write("\n")
        f.write("IMPORTANT: You MUST read the SKILL.md file whenever the ")
        f.write("description of the skills matches the user intent, or may ")
        f.write("help accomplish their task.\n\n")
        f.write("IMPORTANT: When a SKILL.md points at files under its ")
        f.write("`references/` directory, you MUST read the referenced file ")
        f.write("before acting on the topic it covers — those files hold the ")
        f.write("detailed flag tables, command examples, and gotchas the ")
        f.write("SKILL.md trims out.\n\n")
        f.write("<available_skills>\n\n")
        for name, desc, _ in entries:
            f.write(f"- {name}: {desc}\n\n")
        f.write("</available_skills>\n\n")
        f.write("</skills>\n\n")
        f.write("Paths referenced within SKILL folders are relative to that ")
        f.write("SKILL. For example the `xpu-discover/scripts/x.py` would be ")
        f.write("referenced as `xpu-discover/scripts/x.py`.\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(entries)} skills)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
