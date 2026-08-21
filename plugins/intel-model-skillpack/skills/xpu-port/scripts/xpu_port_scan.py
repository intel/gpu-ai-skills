#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scan a CUDA-targeted PyTorch repo for sites that affect an XPU port.

Read-only. Classifies each site as:
  - mechanical : safe for one of the rewrite transforms
  - semantic   : real translation needed; agent handles via Edit
  - escalate   : cannot be ported by this skill (custom kernels, etc.)

Output is JSON on stdout.

Verified against upstream torch/xpu/__init__.py and torch/cuda/__init__.py
(see references/cuda_to_xpu_whitelist.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import ParentNodeProvider, PositionProvider


# torch.xpu attrs that exist 1:1 with torch.cuda. Verified against the
# upstream torch/xpu/__init__.py __all__ list. Anything outside this
# set under torch.cuda escalates.
XPU_TOP_LEVEL = frozenset(
    {
        "Event",
        "Stream",
        "StreamContext",
        "XPUGraph",
        "can_device_access_peer",
        "current_device",
        "current_stream",
        "device",
        "device_count",
        "device_of",
        "empty_cache",
        "get_arch_list",
        "get_device_capability",
        "get_device_name",
        "get_device_properties",
        "get_gencode_flags",
        "get_rng_state",
        "get_rng_state_all",
        "get_stream_from_external",
        "graph",
        "graph_pool_handle",
        "init",
        "initial_seed",
        "is_available",
        "is_bf16_supported",
        "is_current_stream_capturing",
        "is_initialized",
        "is_tf32_supported",
        "make_graphed_callables",
        "manual_seed",
        "manual_seed_all",
        "max_memory_allocated",
        "max_memory_reserved",
        "memory_allocated",
        "memory_reserved",
        "memory_stats",
        "memory_stats_as_nested_dict",
        "MemPool",
        "reset_accumulated_memory_stats",
        "reset_peak_memory_stats",
        "seed",
        "seed_all",
        "set_device",
        "set_rng_state",
        "set_rng_state_all",
        "set_stream",
        "stream",
        "synchronize",
    }
)


@dataclasses.dataclass
class Finding:
    file: str
    line: int
    col: int
    snippet: str
    bucket: str  # mechanical | semantic | escalate
    category: str
    transform: str | None
    note: str
    # Machine-readable downstream skill for findings this skill does not
    # itself resolve. A skill name means an orchestrating agent should
    # chain to that skill for this site. None means there is no single
    # downstream skill to route to — either xpu-port handles it in-loop
    # (a `mechanical` transform or a `semantic` patterns.md hand-edit) or
    # the site is `escalate` (redesign, no mechanical port). Read `bucket`
    # first: `route is None` does not by itself mean "already handled".
    route: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Scanner(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.lines = source.splitlines()
        self.findings: list[Finding] = []

    def _pos(self, node: cst.CSTNode) -> tuple[int, int]:
        meta = self.get_metadata(PositionProvider, node)
        return meta.start.line, meta.start.column

    def _ancestor_chain(self, node: cst.CSTNode, depth: int = 6) -> list[cst.CSTNode]:
        """Walk up to `depth` parents."""
        out: list[cst.CSTNode] = []
        cur: Any = node
        for _ in range(depth):
            try:
                cur = self.get_metadata(ParentNodeProvider, cur)
            except KeyError:
                return out
            if cur is None:
                return out
            out.append(cur)
        return out

    def _enclosing_widened(self, node: cst.CSTNode) -> bool:
        """Return True if a containing Comparison / Tuple / List / Set
        already contains an 'xpu' string literal. Used to suppress
        noise: after the agent widens `device_type == 'cuda'` to
        `device_type in ('cuda', 'xpu')`, the inner `'cuda'` is not a
        site that still needs work — it's the deliberate left half of
        the widened gate.
        """
        for ancestor in self._ancestor_chain(node, depth=8):
            if isinstance(
                ancestor,
                (cst.Comparison, cst.Tuple, cst.List, cst.Set, cst.IfExp),
            ):
                if _contains_xpu_string(ancestor):
                    return True
        return False

    def _is_in_runtime_gate(self, node: cst.CSTNode) -> bool:
        """A 'cuda' literal is a runtime gate, not a tensor-move target,
        when it appears inside any of:
          - Comparison (device_type == 'cuda')
          - IfExp (`'cuda' if cond else 'cpu'`)
          - the RHS of `device_type = 'cuda'` style assignment
          - ``'cuda' in device`` membership test (Comparison with In)

        These need agent judgment, not a blind device_string rewrite.
        """
        for ancestor in self._ancestor_chain(node):
            if isinstance(ancestor, (cst.Comparison, cst.IfExp)):
                return True
            if isinstance(ancestor, cst.Assign):
                # `device_type = 'cuda'` — RHS scalar is a label, not a device
                for tgt in ancestor.targets:
                    if isinstance(tgt.target, cst.Name) and tgt.target.value in {
                        "device_type",
                        "backend",
                    }:
                        return True
                return False
        return False

    def _snippet(self, line: int) -> str:
        if 1 <= line <= len(self.lines):
            return self.lines[line - 1].strip()
        return ""

    def _add(
        self,
        node: cst.CSTNode,
        bucket: str,
        category: str,
        transform: str | None,
        note: str,
        route: str | None = None,
    ) -> None:
        line, col = self._pos(node)
        self.findings.append(
            Finding(
                file=str(self.path),
                line=line,
                col=col,
                snippet=self._snippet(line),
                bucket=bucket,
                category=category,
                transform=transform,
                note=note,
                route=route,
            )
        )

    # torch.cuda.X reference
    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            chain = _attr_chain(alias.name)
            if not (len(chain) >= 2 and chain[0] == "torch" and chain[1] == "cuda"):
                continue
            if _import_chain_is_safe(chain):
                self._add(
                    node,
                    bucket="mechanical",
                    category="cuda_import",
                    transform="imports",
                    note="import torch.cuda... -> import torch.xpu...",
                )
            else:
                self._add(
                    node,
                    bucket="semantic",
                    category="cuda_import_no_xpu",
                    transform=None,
                    note=(
                        f"import torch.{'.'.join(chain[1:])} has no direct "
                        "torch.xpu equivalent (torch.cuda.amp lives at "
                        "torch.amp on XPU)"
                    ),
                )

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        chain = _attr_chain(node.module)
        if not (len(chain) >= 2 and chain[0] == "torch" and chain[1] == "cuda"):
            return
        if _import_chain_is_safe(chain):
            self._add(
                node,
                bucket="mechanical",
                category="cuda_import_from",
                transform="imports",
                note="from torch.cuda... import ... -> from torch.xpu...",
            )
        else:
            self._add(
                node,
                bucket="semantic",
                category="cuda_import_from_no_xpu",
                transform=None,
                note=(
                    "from torch.cuda.amp import ... -> from torch.amp "
                    "import ... (XPU has no torch.xpu.amp; AMP is unified "
                    "under torch.amp)"
                ),
            )

    def visit_Attribute(self, node: cst.Attribute) -> None:
        chain = _attr_chain(node)
        if not chain:
            return

        # torch.cuda.X
        if len(chain) >= 3 and chain[0] == "torch" and chain[1] == "cuda":
            attr = chain[2]
            if attr == "amp":
                # The call visitor below catches torch.cuda.amp.autocast(...)
                # and GradScaler(...) as mechanical. Reference-style uses
                # (e.g. cls = torch.cuda.amp.GradScaler) are not flagged here
                # and are not flagged by the call visitor either, so they
                # remain in the codebase unflagged; they are rare enough that
                # the agent is expected to notice them during semantic review.
                return
            if attr in XPU_TOP_LEVEL:
                self._add(
                    node,
                    bucket="mechanical",
                    category="cuda_attr",
                    transform="cuda_to_xpu",
                    note=f"torch.cuda.{attr} -> torch.xpu.{attr}",
                )
            else:
                self._add(
                    node,
                    bucket="escalate",
                    category="cuda_attr_no_xpu",
                    transform=None,
                    note=(
                        f"torch.cuda.{attr} has no torch.xpu equivalent; "
                        "agent must replace by hand"
                    ),
                )

        # ProfilerActivity.CUDA — must be changed to ProfilerActivity.XPU
        if len(chain) >= 2 and chain[-1] == "CUDA" and "ProfilerActivity" in chain:
            self._add(
                node,
                bucket="semantic",
                category="profiler_activity_cuda",
                transform=None,
                note=(
                    "ProfilerActivity.CUDA -> ProfilerActivity.XPU on Intel GPU. "
                    "See references/semantic-patterns.md."
                ),
                route="torch-xpu-profile",
            )

        # torch.backends.cuda.matmul.allow_tf32 / torch.backends.cudnn.allow_tf32
        if (
            len(chain) >= 4
            and chain[0] == "torch"
            and chain[1] == "backends"
            and chain[2] in ("cuda", "cudnn")
            and chain[-1] == "allow_tf32"
        ):
            self._add(
                node,
                bucket="semantic",
                category="tf32_toggle",
                transform=None,
                note=(
                    "TF32 toggle is CUDA-only; no-op on XPU. Keep gated "
                    "on torch.cuda.is_available() or delete."
                ),
            )

    def visit_Call(self, node: cst.Call) -> None:
        chain = _attr_chain(node.func)
        if not chain:
            return

        # torch.cuda.amp.autocast(...)
        if (
            chain[:3] == ["torch", "cuda", "amp"]
            and len(chain) == 4
            and chain[3] == "autocast"
        ):
            self._add(
                node,
                bucket="mechanical",
                category="amp_autocast",
                transform="amp_autocast",
                note='torch.cuda.amp.autocast(...) -> torch.amp.autocast("xpu", ...)',
            )
            return

        # torch.cuda.amp.GradScaler(...)
        if (
            chain[:3] == ["torch", "cuda", "amp"]
            and len(chain) == 4
            and chain[3] == "GradScaler"
        ):
            self._add(
                node,
                bucket="mechanical",
                category="amp_gradscaler",
                transform="amp_gradscaler",
                note='torch.cuda.amp.GradScaler(...) -> torch.amp.GradScaler("xpu", ...)',
            )
            return

        # torch.autocast(device_type="cuda", ...)
        # torch.amp.autocast("cuda", ...)
        if chain == ["torch", "autocast"] or chain == ["torch", "amp", "autocast"]:
            if _has_cuda_device_arg(node):
                self._add(
                    node,
                    bucket="semantic",
                    category="autocast_cuda",
                    transform=None,
                    note=(
                        "autocast pinned to 'cuda'; agent should drive the "
                        "device from the runtime device variable, not a "
                        "literal"
                    ),
                )

        # init_process_group / dist.init_process_group with backend=nccl
        if (
            chain == ["init_process_group"]
            or (len(chain) >= 2 and chain[-1] == "init_process_group")
        ):
            for kw in node.args:
                if (
                    kw.keyword
                    and isinstance(kw.keyword, cst.Name)
                    and kw.keyword.value == "backend"
                ):
                    val = _string_value(kw.value)
                    if val == "nccl":
                        self._add(
                            node,
                            bucket="mechanical",
                            category="dist_backend",
                            transform="dist_backend",
                            note='init_process_group(backend="nccl") -> backend="xccl"',
                        )
                    elif val == "ccl":
                        self._add(
                            node,
                            bucket="semantic",
                            category="dist_backend_legacy_ccl",
                            transform=None,
                            note=(
                                "backend='ccl' targets the deprecated "
                                "torch_ccl plugin; upstream PyTorch uses "
                                "'xccl' for XPU"
                            ),
                        )

        # x.cuda() / model.cuda() — mechanical via dot_cuda transform
        if len(chain) >= 2 and chain[-1] == "cuda" and chain[0] != "torch":
            self._add(
                node,
                bucket="mechanical",
                category="dot_cuda",
                transform="dot_cuda",
                note=".cuda(...) -> .xpu(...) (method on a tensor / module)",
            )

        # CUDAExtension(...) inside setup.py — escalate
        if chain == ["CUDAExtension"] or (
            len(chain) >= 2 and chain[-1] == "CUDAExtension"
        ):
            self._add(
                node,
                bucket="escalate",
                category="cuda_extension",
                transform=None,
                note="custom CUDA C++ extension; cannot port mechanically",
            )
            return

        # torch.compile(...) — Inductor backend is build- and graph-dependent on XPU
        if chain == ["torch", "compile"]:
            self._add(
                node,
                bucket="semantic",
                category="torch_compile",
                transform=None,
                note=(
                    "torch.compile (Inductor) support on XPU is "
                    "build- and graph-dependent. Test it on the target: "
                    "keep it if it compiles and a training step runs; "
                    "add an `if device_type != 'xpu':` guard only as a "
                    "fallback when Inductor fails to lower on that build. "
                    "See references/semantic-patterns.md."
                ),
            )
            return

    # 'cuda' / 'cuda:N' device strings inside .to(...), torch.device(...), or kwargs
    def visit_SimpleString(self, node: cst.SimpleString) -> None:
        raw = node.value.strip()
        # raw includes quotes
        inner = _string_value(node)
        if inner is None:
            return

        # Bare `backend = 'nccl'` — distinct finding, separate transform.
        if inner in {"nccl", "ccl"} and self._is_backend_assign_rhs(node):
            self._add(
                node,
                bucket="mechanical" if inner == "nccl" else "semantic",
                category="dist_backend_assign",
                transform="dist_backend" if inner == "nccl" else None,
                note=(
                    f"backend variable assigned {raw}; rewrite to "
                    '"xccl" (the upstream XPU backend)'
                    if inner == "nccl"
                    else "backend='ccl' targets the deprecated torch_ccl plugin"
                ),
            )
            return

        if inner == "cuda" or inner.startswith("cuda:"):
            if self._enclosing_widened(node):
                # Already widened — agent has already paired this 'cuda'
                # with an 'xpu' sibling. Don't re-flag.
                return
            if self._is_in_runtime_gate(node):
                self._add(
                    node,
                    bucket="semantic",
                    category="device_label_compare",
                    transform=None,
                    note=(
                        f"{raw} appears in a comparison / device-label "
                        "context. Hand-edit per skill 'Device-type "
                        "gates'; scanner cannot decide flip vs widen."
                    ),
                )
            else:
                self._add(
                    node,
                    bucket="mechanical",
                    category="device_string",
                    transform="device_string",
                    note=f"{raw} -> '{inner.replace('cuda', 'xpu')}'",
                )

    def visit_FormattedString(self, node: cst.FormattedString) -> None:
        # f"cuda:{i}" — flag via device_string transform unless it's a
        # device-label assignment, in which case it's semantic.
        text = ""
        for part in node.parts:
            if isinstance(part, cst.FormattedStringText):
                text += part.value
            else:
                text += "{?}"
        if text == "cuda" or text.startswith("cuda:"):
            if self._enclosing_widened(node):
                return
            if self._is_in_runtime_gate(node):
                self._add(
                    node,
                    bucket="semantic",
                    category="device_fstring_label",
                    transform=None,
                    note=(
                        f'f-string "{text}" appears in a comparison / '
                        "device-label context, not a tensor-move target"
                    ),
                )
            else:
                self._add(
                    node,
                    bucket="mechanical",
                    category="device_fstring",
                    transform="device_string",
                    note=f'f-string device "{text}" -> xpu form',
                )

    def _is_backend_assign_rhs(self, node: cst.CSTNode) -> bool:
        """True if `node` is the RHS string of an assignment like
        `backend = "nccl"`."""
        for ancestor in self._ancestor_chain(node):
            if isinstance(ancestor, cst.Assign):
                for tgt in ancestor.targets:
                    if (
                        isinstance(tgt.target, cst.Name)
                        and tgt.target.value == "backend"
                    ):
                        return True
                return False
        return False


def _attr_chain(node: cst.CSTNode) -> list[str]:
    """Return ['torch', 'cuda', 'empty_cache'] for torch.cuda.empty_cache.

    Returns [] if the node isn't a pure attribute chain.
    """
    out: list[str] = []
    cur = node
    while isinstance(cur, cst.Attribute):
        out.append(cur.attr.value)
        cur = cur.value
    if isinstance(cur, cst.Name):
        out.append(cur.value)
        return list(reversed(out))
    return []


def _import_chain_is_safe(chain: list[str]) -> bool:
    """True if `import torch.<...>` / `from torch.<...> import` can be
    flipped to torch.xpu mechanically. Conservatively excludes
    torch.cuda.amp (lives at torch.amp on XPU) and unknown sub-modules.
    """
    if len(chain) < 2 or chain[0] != "torch" or chain[1] != "cuda":
        return False
    if len(chain) == 2:
        return True
    return chain[2] in XPU_TOP_LEVEL


def _contains_xpu_string(node: cst.CSTNode) -> bool:
    """True if `node` (typically a Comparison / Tuple / List) contains
    a SimpleString whose inner value is 'xpu' or starts with 'xpu:'.
    Used to detect that a runtime gate has already been widened."""
    found = False

    class _Find(cst.CSTVisitor):
        def visit_SimpleString(self, n: cst.SimpleString) -> None:
            nonlocal found
            v = _string_value(n)
            if v is not None and (v == "xpu" or v.startswith("xpu:")):
                found = True

        def visit_FormattedStringText(
            self, n: cst.FormattedStringText
        ) -> None:
            nonlocal found
            if n.value == "xpu" or n.value.startswith("xpu:"):
                found = True

    node.visit(_Find())
    return found


def _string_value(node: cst.CSTNode | None) -> str | None:
    if isinstance(node, cst.SimpleString):
        # strip quotes; libcst preserves raw
        s = node.value
        if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
            return s[1:-1]
    if isinstance(node, cst.ConcatenatedString):
        # ignore: rare
        return None
    return None


def _has_cuda_device_arg(call: cst.Call) -> bool:
    for arg in call.args:
        if arg.keyword is None:
            # positional first arg
            v = _string_value(arg.value)
            if v == "cuda":
                return True
        else:
            if (
                isinstance(arg.keyword, cst.Name)
                and arg.keyword.value == "device_type"
            ):
                v = _string_value(arg.value)
                if v == "cuda":
                    return True
    return False


# Every error type libcst exports. They share no common base class, so all
# three have to be named: ParserSyntaxError for source that does not tokenize,
# CSTValidationError for source that tokenizes but cannot form a valid tree
# (implicit str+bytes concatenation), and CSTLogicError for internal parser
# invariants (a control character inside a string literal). Any of them means
# "this file is not parseable", which is a skip, not a crash.
PARSE_ERRORS = (
    cst.ParserSyntaxError,
    cst.CSTValidationError,
    cst.CSTLogicError,
    # Nesting that does not use brackets -- a long run of unary operators, for
    # instance -- recurses in the pure-Python layer and raises RecursionError
    # before reaching the native parser. The depth guard below cannot see it
    # (there are no brackets to count), so it is handled here instead: the file
    # is unparseable for our purposes, which is a skip, not a traceback.
    RecursionError,
)

MAX_NESTING_DEPTH = 1000


def nesting_too_deep(source: str, limit: int = MAX_NESTING_DEPTH) -> bool:
    """True if bracket nesting in `source` exceeds what libcst can parse.

    libcst's native parser overflows the C stack on deeply nested input and
    terminates the interpreter with SIGSEGV. That happens below the Python
    frame, so no except clause can intercept it -- the depth has to be
    rejected before parse_module is called. The observed crash threshold is
    between 1200 and 1400 delimiters; the limit here sits under it. Real
    source does not approach this depth.

    Only brackets that are real delimiters count. Counting every bracket
    character would reject a file whose sole "nesting" is a long string
    constant, discarding any genuine finding elsewhere in it, so the source is
    tokenized first and string/comment content skipped. Source that does not
    tokenize is left to parse_module, which reports it through PARSE_ERRORS.

    Kept in sync with the identical guard in xpu_port_rewrite.py; these scripts
    are standalone by design and do not share a module.
    """
    depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.OP:
                continue
            if tok.string in ("(", "[", "{"):
                depth += 1
                if depth > limit:
                    return True
            elif tok.string in (")", "]", "}"):
                if depth > 0:
                    depth -= 1
    except tokenize.TokenError as exc:
        # CPython's tokenizer has its own nesting ceiling (~200), well below
        # `limit`, and reports it as "too many nested parentheses". That is the
        # same input class the guard exists to catch, so it must be treated as
        # too deep -- returning False here would hand the file to libcst and
        # segfault. Other TokenErrors (an unterminated string, unbalanced
        # brackets) say nothing about depth; those go to parse_module, which
        # reports them through PARSE_ERRORS.
        return "nested" in str(exc)
    except (IndentationError, SyntaxError, ValueError, SystemError):
        # Does not tokenize, so it will not parse either; parse_module raises a
        # handled error. SystemError is here because CPython's C tokenizer
        # reports a null byte in the source by setting an exception while still
        # returning a result, which surfaces as SystemError rather than
        # SyntaxError. libcst rejects the same input with ParserSyntaxError,
        # which PARSE_ERRORS covers.
        return False
    return False


def scan_file(path: Path) -> list[Finding]:
    if path.suffix in {".cu", ".cuh"}:
        return [
            Finding(
                file=str(path),
                line=1,
                col=0,
                snippet=path.name,
                bucket="escalate",
                category="cuda_source_file",
                transform=None,
                note="raw CUDA source; replace with Triton XPU / SYCL / native",
            )
        ]

    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        print(f"scan: skip {path}: read error: {exc}", file=sys.stderr)
        return [Finding(file=str(path), line=1, col=0, snippet="",
                        bucket="escalate", category="unreadable",
                        transform=None, note=f"file read failed: {exc}")]
    if nesting_too_deep(source):
        note = (
            f"bracket nesting exceeds {MAX_NESTING_DEPTH}; parsing this file "
            f"would crash the parser"
        )
        print(f"scan: skip {path}: {note}", file=sys.stderr)
        return [Finding(file=str(path), line=1, col=0, snippet="",
                        bucket="escalate", category="unparseable",
                        transform=None, note=note)]
    # The traversal is inside the same try as the parse: MetadataWrapper
    # deep-clones the tree and the visitor recurses over it, so RecursionError
    # surfaces here rather than from parse_module. Source that recurses without
    # brackets (a long run of unary operators) parses fine and only blows up on
    # the walk, which nesting_too_deep cannot predict.
    try:
        tree = cst.parse_module(source)
        wrapper = cst.MetadataWrapper(tree)
        scanner = Scanner(path, source)
        wrapper.visit(scanner)
    except PARSE_ERRORS as exc:
        print(f"scan: skip {path}: parse error: {exc}", file=sys.stderr)
        return [Finding(file=str(path), line=1, col=0, snippet="",
                        bucket="escalate", category="unparseable",
                        transform=None, note=f"libcst parse failed: {exc}")]
    return scanner.findings


# NVIDIA/CUDA infrastructure markers used only for the empty-scan
# advisory (see main()) — the authoritative inventory lives in
# cuda-to-xpu-migration. Four groups by matching strategy:
#
# _INFRA_MARKERS_SUBSTR : distinctive tokens safe as plain substrings.
#   They carry punctuation or are long/specific enough not to collide
#   with an ordinary word, so they are also scanned inside .py source (a
#   NIM/OpenAI cloud endpoint or a hard CUDA-lib import expressed only in
#   Python — the AST scan finds no torch.cuda site, but the repo is still
#   NVIDIA-bound).
# _INFRA_MARKERS_SUBSTR_INFRA : distinctive tokens that are only a signal
#   on infra surfaces (shell launchers, Makefiles, compose). "--gpus" in
#   a launch script means `docker run --gpus`; the same token in .py or
#   .ipynb source is an ordinary argparse flag on a device-agnostic
#   repo, so it is never matched there.
# _INFRA_MARKERS_SUBSTR_WORD : distinctive lib tokens that still need a
#   leading boundary because the token is a substring of a common word
#   ("cupy" sits inside "occupy"). Specific enough to scan in .py too,
#   unlike the generic word markers below.
# _INFRA_MARKERS_WORD : short/generic tokens ("cuda", "nvidia"). A
#   *leading* boundary keeps them from firing inside a longer word
#   ("barracuda"); no trailing boundary, so the version-glued forms a
#   trailing boundary used to miss ("cuda12.4", "cudatoolkit") still
#   match. Scanned only in files whose *name* marks a dependency pin or
#   build config (see _name_is_dep_or_build) — never in .py/.ipynb
#   source, prose .txt, or data .json, where the words are too generic.
_INFRA_MARKERS_SUBSTR = (
    "nvcr.io",  # NGC container registry
    "flash-attn",
    "flash_attn",
    "bitsandbytes",
    "tensorrt",
    "integrate.api.nvidia.com",  # NIM cloud endpoint
    "build.nvidia.com",  # NIM catalog / docs endpoint
    "langchain-nvidia",  # langchain-nvidia-ai-endpoints (NIM client)
    "langchain_nvidia",
)
_INFRA_MARKERS_SUBSTR_INFRA = ("--gpus",)  # docker run --gpus all
_INFRA_MARKERS_SUBSTR_WORD = ("cupy",)  # matches "cupy"/"cupyx", not "occupy"
_INFRA_MARKERS_WORD = ("cuda", "nvidia")
# Leading boundary only (no trailing exclusion): rejects a marker glued
# *after* a letter/digit ("barracuda", "occupy") while still firing on
# "cuda12.4" / "cudatoolkit" / "cupyx" that a trailing boundary dropped.
_INFRA_SUBSTR_WORD_RE = re.compile(
    r"(?<![a-z0-9_])(?:" + "|".join(_INFRA_MARKERS_SUBSTR_WORD) + r")"
)
_INFRA_WORD_RE = re.compile(
    r"(?<![a-z0-9_])(?:" + "|".join(_INFRA_MARKERS_WORD) + r")"
)

# Filenames / suffixes worth reading for infra markers when the Python
# scan is empty. Bounded on purpose — this is a hint, not an inventory.
_INFRA_FILES = (
    "dockerfile",
    "containerfile",
    "makefile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "environment.yml",
)
_INFRA_SUFFIXES = (
    ".sh", ".dockerfile", ".txt", ".toml", ".cfg", ".yml", ".yaml", ".json",
    ".lock",  # dependency lockfiles: poetry.lock, Pipfile.lock, uv.lock
)
# Name patterns (lowercased, fnmatch) extending _INFRA_FILES to common
# variants: requirements-dev.txt, constraints.txt, environment.dev.yaml,
# docker-compose.prod.yml, poetry.lock, ... Together with the exact
# _INFRA_FILES names and Dockerfile/Containerfile prefixes, these are
# the only *names* that earn generic-word matching.
_INFRA_NAME_PATTERNS = (
    "requirements*.txt",
    "constraints*.txt",
    "environment*.yml",
    "environment*.yaml",
    "docker-compose*.yml",
    "docker-compose*.yaml",
    "*.lock",
)
# Vendored / third-party trees whose markers are not the repo's own
# surface. Merged with the caller's exclude set so the infra probe does
# not report, e.g., a CUDA pin inside node_modules or site-packages.
_INFRA_EXCLUDE_EXTRA = frozenset(
    {"node_modules", "site-packages", ".tox", "vendor", "third_party", ".mypy_cache"}
)
# Cap per-file read for the marker probe, applied at the read itself
# (`fh.read(_INFRA_READ_CAP)` bytes), not as a post-read slice. Markers
# of interest sit near the top of Dockerfiles / requirements / compose
# files; the bounded read keeps a single large JSON or lockfile from
# spiking memory.
_INFRA_READ_CAP = 65536


def _text_has_infra_marker(text: str, *, infra: bool, generic_words: bool) -> bool:
    """True if `text` (already lowercased) carries an infra marker.
    Distinctive markers always count (plain substrings, plus the
    boundary-anchored ones like `cupy` that sit inside a common word).
    `infra` adds the infra-surface substrings ("--gpus"), which are launch
    signals in shell/Make/compose files but ordinary argparse flags in .py
    source. `generic_words` adds the short generic words ("cuda",
    "nvidia") — infra files only, never .py source."""
    if any(marker in text for marker in _INFRA_MARKERS_SUBSTR):
        return True
    if _INFRA_SUBSTR_WORD_RE.search(text):
        return True
    if infra and any(marker in text for marker in _INFRA_MARKERS_SUBSTR_INFRA):
        return True
    if generic_words and _INFRA_WORD_RE.search(text):
        return True
    return False


def _name_is_dep_or_build(name: str, is_dockerfile: bool) -> bool:
    """True if the (lowercased) file name marks a dependency-pin or
    build-config surface — the only files where the generic words
    ("cuda", "nvidia") are dependable signals rather than prose. Covers
    Dockerfile/Containerfile prefixes, the exact _INFRA_FILES names
    (including setup.py — a .py file matched as a dependency surface),
    and the _INFRA_NAME_PATTERNS variants."""
    if is_dockerfile or name in _INFRA_FILES:
        return True
    return any(fnmatch.fnmatchcase(name, pat) for pat in _INFRA_NAME_PATTERNS)


def find_infra_markers(
    root: Path, exclude: set[str], limit: int = 10
) -> tuple[list[str], int, bool]:
    """Return (hits, skipped_unreadable, capped): up to `limit`
    repo-relative paths of files that carry NVIDIA/CUDA infrastructure
    markers, the number of candidate files that could not be read, and
    whether more than `limit` files matched (hits truncated).

    Matching is tiered by surface:
      - Dependency-pin / build-config *names* (Dockerfile*, Makefile,
        requirements*.txt, setup.py, pyproject.toml, compose files,
        *.lock, ...) get the full marker set including the generic
        words "cuda"/"nvidia". setup.py is .py source but is matched
        as a dependency surface by name — a generic word in it (even
        a comment) fires.
      - Other infra-suffixed files (.sh launchers, prose .txt, data
        .json) get distinctive markers plus the infra-surface flags
        ("--gpus"), but not the generic words.
      - .py and .ipynb source gets distinctive markers only (cloud
        NIM/API endpoints, hard CUDA-lib imports) — the AST scan
        already cleared the torch.cuda surface of .py files, so a hit
        here means an NVIDIA dependency the port cannot rewrite, not a
        device call it missed.

    Read-only and best-effort: unreadable *files* are skipped but
    counted, so the caller can flag that a bare 0 may be incomplete.
    Unreadable *directories* are swallowed inside Path.rglob before this
    loop sees them and are not counted — a known limit of the probe.
    This is a routing hint for the empty-scan case, not a substitute for
    cuda-to-xpu-migration's inventory scan.
    """
    exclude = exclude | _INFRA_EXCLUDE_EXTRA
    hits: list[str] = []
    skipped_unreadable = 0
    # Iterate rglob directly (no whole-tree materialization) so the
    # len(hits) > limit early-exit can fire after limit+1 matches on a
    # large repo. The returned list is sorted at the end instead, so
    # `files_with_markers` stays deterministic across filesystems even
    # when capped — filesystem iteration order is not stable.
    for p in root.rglob("*"):
        if len(hits) > limit:
            break  # one hit past the cap is enough to know it truncated
        if not p.is_file():
            continue
        # Exclusions match path parts relative to the scan root, so a
        # repo living under e.g. .../build/<repo> is not excluded
        # wholesale by its parent directory's name.
        if any(part in exclude for part in p.relative_to(root).parts):
            continue
        name = p.name.lower()
        is_dockerfile = name == "dockerfile" or name.startswith(
            ("dockerfile.", "containerfile.")
        )
        is_infra_file = (
            is_dockerfile
            or name in _INFRA_FILES
            or p.suffix.lower() in _INFRA_SUFFIXES
        )
        # Notebooks embed Python source (plus prose cells) in JSON, so
        # .ipynb follows the .py tier: distinctive markers only.
        is_source = p.suffix.lower() in (".py", ".ipynb")
        if not (is_infra_file or is_source):
            continue
        try:
            with p.open("rb") as fh:
                data = fh.read(_INFRA_READ_CAP)
        except OSError:
            skipped_unreadable += 1
            continue
        # Strip NULs before matching: UTF-16 dependency pins (pip freeze
        # under Windows PowerShell 5.x) survive decode(errors="ignore")
        # as 'c\0u\0d\0a' and would dodge every substring; NULs carry no
        # signal in legitimate infra files.
        text = data.decode("utf-8", "ignore").replace("\x00", "").lower()
        # .py/.ipynb source: distinctive markers only (no generic
        # cuda/nvidia, which would fire on comments/docstrings the AST
        # already cleared).
        if _text_has_infra_marker(
            text,
            infra=is_infra_file,
            generic_words=_name_is_dep_or_build(name, is_dockerfile),
        ):
            try:
                hits.append(str(p.relative_to(root)))
            except ValueError:
                hits.append(str(p))
    # Sort the (at most limit+1) collected hits so the capped output is
    # deterministic regardless of rglob iteration order.
    hits.sort()
    capped = len(hits) > limit
    return hits[:limit], skipped_unreadable, capped


def walk(root: Path, exclude: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".py", ".cu", ".cuh"}:
            continue
        # Relative to the scan root: a repo checked out under a parent
        # named "build" / "dist" / "venv" must still be scanned.
        if any(part in exclude for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Scan a PyTorch repo for CUDA call sites relevant to an XPU "
            "port. JSON to stdout."
        )
    )
    ap.add_argument("path", type=Path, help="repo root or single .py file")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[".git", ".venv", "venv", "__pycache__", "build", "dist"],
        help="path part to skip (repeatable)",
    )
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"path does not exist: {args.path}", file=sys.stderr)
        return 2

    if args.path.is_file():
        files = [args.path]
    else:
        files = walk(args.path, set(args.exclude))

    findings: list[Finding] = []
    for f in files:
        findings.extend(scan_file(f))

    by_bucket: dict[str, int] = {}
    for f in findings:
        by_bucket[f.bucket] = by_bucket.get(f.bucket, 0) + 1

    out: dict[str, Any] = {
        "schema_version": "1.1",
        "root": str(args.path),
        "files_scanned": len(files),
        "total_findings": len(findings),
        "by_bucket": by_bucket,
        "findings": [f.as_dict() for f in findings],
    }

    # Empty-scan escape hatch: a PyTorch-CUDA repo with 0 findings is
    # done, but an *NVIDIA-stacked* repo with 0 findings (containers,
    # dependency pins, launch scripts, cloud NIM/API clients) is not a
    # port job at all — this skill just can't see its surface. Point the
    # agent at the assessment skill instead of letting "0 findings" read
    # as "nothing to migrate". Only runs on a directory scan. Synthetic
    # unreadable/unparseable findings are scan bookkeeping, not CUDA
    # sites, so they do not count against emptiness.
    real_findings = [
        f for f in findings if f.category not in ("unreadable", "unparseable")
    ]
    if len(real_findings) == 0 and args.path.is_dir():
        try:
            infra, probe_skipped, infra_capped = find_infra_markers(
                args.path, set(args.exclude)
            )
        except Exception as exc:
            # The probe is a routing hint; it must never forfeit the
            # primary report (json.dump below) by escaping.
            print(f"xpu-port: infra probe failed: {exc}", file=sys.stderr)
            infra, probe_skipped, infra_capped = [], 0, False
        if probe_skipped > 0:
            out["probe_skipped_unreadable"] = probe_skipped
            print(
                f"xpu-port: infra probe could not read {probe_skipped} "
                "file(s); a bare 0 may be incomplete",
                file=sys.stderr,
            )
        if infra:
            advisory = (
                "0 Python CUDA sites found, but NVIDIA/CUDA infrastructure "
                "markers are present (container files, dependency pins, "
                "launch scripts, or cloud NIM/API endpoints in source). This "
                "is likely an API-first / container-bound migration, which "
                "xpu-port does not cover. Route to cuda-to-xpu-migration "
                "(assessment + infra inventory), then xpu-deploy-plan "
                "(end-to-end serving plan)."
            )
            out["advisory"] = {
                "kind": "empty_scan_infra_markers",
                "message": advisory,
                # Plural key and list value on purpose: findings carry a
                # scalar `route`; the advisory names an ordered chain.
                "routes": ["cuda-to-xpu-migration", "xpu-deploy-plan"],
                "files_with_markers": infra,
            }
            if infra_capped:
                out["advisory"]["list_capped"] = True
            print(
                "xpu-port: " + advisory + "\n  Files with markers:\n    "
                + "\n    ".join(infra),
                file=sys.stderr,
            )

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
