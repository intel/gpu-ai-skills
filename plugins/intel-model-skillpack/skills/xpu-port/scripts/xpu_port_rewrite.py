#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Apply a small whitelist of mechanical CUDA-to-XPU rewrites with libcst.

One transform per invocation. Idempotent. `--check` previews diffs
without writing.

Transforms:
  device_string   'cuda' / 'cuda:N' / f"cuda:{i}" -> 'xpu' / 'xpu:N' / f"xpu:{i}"
  cuda_to_xpu     torch.cuda.X -> torch.xpu.X for X in the safe whitelist
  dot_cuda        x.cuda() / x.cuda(0) -> x.xpu() / x.xpu(0)  (method, not torch.cuda)
  imports         `import torch.cuda` / `from torch.cuda import ...` -> torch.xpu
  dist_backend    init_process_group(backend="nccl") -> backend="xccl"
  amp_autocast    torch.cuda.amp.autocast(...) -> torch.amp.autocast("xpu", ...)
  amp_gradscaler  torch.cuda.amp.GradScaler(...) -> torch.amp.GradScaler("xpu", ...)

The whitelist used by `cuda_to_xpu` is the verified intersection of
torch.cuda and torch.xpu (see SKILL.md). Anything outside it is left
alone — the scanner flags it for the agent to handle.
"""
from __future__ import annotations

import argparse
import difflib
import io
import sys
import tokenize
from pathlib import Path
from typing import Any

import libcst as cst
from libcst.metadata import ParentNodeProvider


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
    constant, discarding any genuine rewrite in it, so the source is tokenized
    first and string/comment content skipped. Source that does not tokenize is
    left to parse_module, which reports it through PARSE_ERRORS.

    Kept in sync with the identical guard in xpu_port_scan.py; these scripts
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


# ---------------------------------------------------------------------------
# device_string
# ---------------------------------------------------------------------------
class DeviceStringTransform(cst.CSTTransformer):
    """Rewrite 'cuda' / 'cuda:N' / f"cuda:{i}" → xpu form, BUT skip
    runtime gates: comparisons (`device_type == 'cuda'`), conditional
    expressions (`'cuda' if cond else 'cpu'`), and assignments to
    label-style variables (`device_type = 'cuda'`, `backend = 'nccl'`).
    Those are the scanner's `semantic` bucket; rewriting them blindly
    produces wrong code (e.g. flips a CUDA gate into an XPU-only gate
    instead of widening it).
    """

    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def _in_runtime_gate(self, node: cst.CSTNode) -> bool:
        cur: Any = node
        for _ in range(6):
            try:
                cur = self.get_metadata(ParentNodeProvider, cur)
            except KeyError:
                return False
            if cur is None:
                return False
            if isinstance(cur, (cst.Comparison, cst.IfExp)):
                return True
            if isinstance(cur, cst.Assign):
                for tgt in cur.targets:
                    if isinstance(tgt.target, cst.Name) and tgt.target.value in {
                        "device_type",
                        "backend",
                    }:
                        return True
                return False
        return False

    def leave_SimpleString(
        self, original: cst.SimpleString, updated: cst.SimpleString
    ) -> cst.SimpleString:
        s = updated.value
        if len(s) < 2 or s[0] not in ("'", '"') or s[-1] != s[0]:
            return updated
        inner = s[1:-1]
        q = s[0]
        if inner != "cuda" and not inner.startswith("cuda:"):
            return updated
        if self._in_runtime_gate(original):
            return updated
        if inner == "cuda":
            return updated.with_changes(value=f"{q}xpu{q}")
        return updated.with_changes(value=f"{q}xpu{inner[len('cuda'):]}{q}")

    def leave_FormattedStringText(
        self,
        original: cst.FormattedStringText,
        updated: cst.FormattedStringText,
    ) -> cst.FormattedStringText:
        v = updated.value
        if v != "cuda" and not v.startswith("cuda:"):
            return updated
        if self._in_runtime_gate(original):
            return updated
        if v == "cuda":
            return updated.with_changes(value="xpu")
        return updated.with_changes(value="xpu" + v[len("cuda"):])


# ---------------------------------------------------------------------------
# cuda_to_xpu
# ---------------------------------------------------------------------------
class CudaToXpuTransform(cst.CSTTransformer):
    def leave_Attribute(
        self, _: cst.Attribute, updated: cst.Attribute
    ) -> cst.Attribute:
        # Match `torch.cuda.X` exactly; skip `torch.cuda.amp.X` (handled by
        # the amp_* transforms) and anything else.
        if not (
            isinstance(updated.value, cst.Attribute)
            and isinstance(updated.value.value, cst.Name)
            and updated.value.value.value == "torch"
            and updated.value.attr.value == "cuda"
        ):
            return updated
        attr = updated.attr.value
        if attr == "amp":
            return updated
        if attr not in XPU_TOP_LEVEL:
            return updated
        # rewrite torch.cuda -> torch.xpu
        return updated.with_changes(
            value=updated.value.with_changes(
                attr=cst.Name("xpu"),
            )
        )


# ---------------------------------------------------------------------------
# dist_backend
# ---------------------------------------------------------------------------
class DistBackendTransform(cst.CSTTransformer):
    """nccl -> xccl in two shapes:
      - init_process_group(backend="nccl")
      - backend = "nccl"  (top-level config variable later passed in)
    """

    def leave_Call(self, _: cst.Call, updated: cst.Call) -> cst.Call:
        if not _is_init_process_group(updated.func):
            return updated
        new_args = []
        changed = False
        for arg in updated.args:
            if (
                arg.keyword is not None
                and isinstance(arg.keyword, cst.Name)
                and arg.keyword.value == "backend"
                and isinstance(arg.value, cst.SimpleString)
            ):
                s = arg.value.value
                if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
                    inner = s[1:-1]
                    if inner == "nccl":
                        q = s[0]
                        new_args.append(
                            arg.with_changes(
                                value=cst.SimpleString(f"{q}xccl{q}")
                            )
                        )
                        changed = True
                        continue
            new_args.append(arg)
        if not changed:
            return updated
        return updated.with_changes(args=new_args)

    def leave_Assign(self, _: cst.Assign, updated: cst.Assign) -> cst.Assign:
        # Match `backend = "nccl"` (single target, Name == "backend",
        # value is SimpleString "nccl").
        if len(updated.targets) != 1:
            return updated
        tgt = updated.targets[0].target
        if not (isinstance(tgt, cst.Name) and tgt.value == "backend"):
            return updated
        if not isinstance(updated.value, cst.SimpleString):
            return updated
        s = updated.value.value
        if len(s) < 2 or s[0] not in ("'", '"') or s[-1] != s[0]:
            return updated
        inner = s[1:-1]
        if inner != "nccl":
            return updated
        q = s[0]
        return updated.with_changes(value=cst.SimpleString(f"{q}xccl{q}"))


def _is_init_process_group(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.Name) and node.value == "init_process_group":
        return True
    if isinstance(node, cst.Attribute) and node.attr.value == "init_process_group":
        return True
    return False


# ---------------------------------------------------------------------------
# amp_autocast
# ---------------------------------------------------------------------------
class AmpAutocastTransform(cst.CSTTransformer):
    def leave_Call(self, _: cst.Call, updated: cst.Call) -> cst.Call:
        if not _is_cuda_amp(updated.func, "autocast"):
            return updated
        # Build torch.amp.autocast("xpu", *original_args)
        new_func = cst.Attribute(
            value=cst.Attribute(
                value=cst.Name("torch"),
                attr=cst.Name("amp"),
            ),
            attr=cst.Name("autocast"),
        )
        # Drop a redundant `device_type=...` kwarg if any caller passed it.
        original = [
            a
            for a in updated.args
            if not (
                a.keyword is not None
                and isinstance(a.keyword, cst.Name)
                and a.keyword.value == "device_type"
            )
        ]
        device_arg = cst.Arg(value=cst.SimpleString('"xpu"'))
        # The first existing arg, if positional, needs a comma after our
        # new positional. libcst handles that via the Arg's
        # comma=cst.MaybeSentinel.DEFAULT so explicit reassembly works.
        return updated.with_changes(func=new_func, args=[device_arg, *original])


# ---------------------------------------------------------------------------
# amp_gradscaler
# ---------------------------------------------------------------------------
class AmpGradScalerTransform(cst.CSTTransformer):
    def leave_Call(self, _: cst.Call, updated: cst.Call) -> cst.Call:
        if not _is_cuda_amp(updated.func, "GradScaler"):
            return updated
        new_func = cst.Attribute(
            value=cst.Attribute(
                value=cst.Name("torch"),
                attr=cst.Name("amp"),
            ),
            attr=cst.Name("GradScaler"),
        )
        device_arg = cst.Arg(value=cst.SimpleString('"xpu"'))
        return updated.with_changes(
            func=new_func, args=[device_arg, *updated.args]
        )


# ---------------------------------------------------------------------------
# dot_cuda  (method call, e.g. x.cuda(), x.cuda(0))
# ---------------------------------------------------------------------------
class DotCudaTransform(cst.CSTTransformer):
    """Rewrite `obj.cuda(...)` -> `obj.xpu(...)` for any obj that is
    not the literal `torch` (which would be `torch.cuda(...)`, a
    `torch.cuda.device` shorthand the rewriter must not touch).
    """

    def leave_Call(self, _: cst.Call, updated: cst.Call) -> cst.Call:
        func = updated.func
        if not isinstance(func, cst.Attribute):
            return updated
        if func.attr.value != "cuda":
            return updated
        # Skip `torch.cuda(...)` (the device-shorthand form).
        if isinstance(func.value, cst.Name) and func.value.value == "torch":
            return updated
        new_func = func.with_changes(attr=cst.Name("xpu"))
        return updated.with_changes(func=new_func)


# ---------------------------------------------------------------------------
# imports  (`import torch.cuda`, `from torch.cuda import ...`)
# ---------------------------------------------------------------------------
class ImportsTransform(cst.CSTTransformer):
    """Rewrite torch.cuda imports to torch.xpu in:
      - `import torch.cuda` / `import torch.cuda as foo`
      - `from torch.cuda import X` / `from torch.cuda.foo import X`
    """

    def leave_Import(self, _: cst.Import, updated: cst.Import) -> cst.Import:
        new_names = []
        changed = False
        for alias in updated.names:
            chain = _dotted_chain(alias.name)
            if not _is_safe_torch_cuda_chain(chain):
                new_names.append(alias)
                continue
            new_alias = alias.with_changes(name=_swap_cuda_in_dotted(alias.name))
            new_names.append(new_alias)
            changed = True
        if not changed:
            return updated
        return updated.with_changes(names=new_names)

    def leave_ImportFrom(
        self, _: cst.ImportFrom, updated: cst.ImportFrom
    ) -> cst.ImportFrom:
        if updated.module is None:
            return updated
        chain = _dotted_chain(updated.module)
        if not _is_safe_torch_cuda_chain(chain):
            return updated
        new_module = _swap_cuda_in_dotted(updated.module)
        if new_module is updated.module:
            return updated
        return updated.with_changes(module=new_module)


def _is_safe_torch_cuda_chain(chain: list[str]) -> bool:
    """True if `chain` starts with ['torch', 'cuda'] and the rest of the
    path lives under torch.xpu too. Conservatively excludes `torch.cuda.amp`
    (lives at torch.amp on XPU, not torch.xpu.amp) and unknown sub-modules.
    """
    if len(chain) < 2 or chain[0] != "torch" or chain[1] != "cuda":
        return False
    if len(chain) == 2:
        return True
    # torch.cuda.X — only safe when X is in the verified XPU surface.
    if chain[2] in XPU_TOP_LEVEL:
        return True
    return False


def _dotted_chain(node: cst.BaseExpression) -> list[str]:
    out: list[str] = []
    cur = node
    while isinstance(cur, cst.Attribute):
        out.append(cur.attr.value)
        cur = cur.value
    if isinstance(cur, cst.Name):
        out.append(cur.value)
        return list(reversed(out))
    return []


def _swap_cuda_in_dotted(node: cst.BaseExpression) -> cst.BaseExpression:
    """If `node` is `torch.cuda` or `torch.cuda.X...`, swap `cuda` -> `xpu`.
    Returns the original node unchanged otherwise.
    """
    # Walk down the attribute chain; rewrite the node whose .value is
    # `torch` and whose .attr.value is `cuda`.
    if isinstance(node, cst.Attribute):
        if (
            isinstance(node.value, cst.Name)
            and node.value.value == "torch"
            and node.attr.value == "cuda"
        ):
            return node.with_changes(attr=cst.Name("xpu"))
        # Recurse into the prefix; rebuild if changed.
        new_value = _swap_cuda_in_dotted(node.value)
        if new_value is node.value:
            return node
        return node.with_changes(value=new_value)
    return node


def _is_cuda_amp(node: cst.BaseExpression, leaf: str) -> bool:
    """Match torch.cuda.amp.<leaf>."""
    if not isinstance(node, cst.Attribute):
        return False
    if node.attr.value != leaf:
        return False
    p1 = node.value
    if not (isinstance(p1, cst.Attribute) and p1.attr.value == "amp"):
        return False
    p2 = p1.value
    if not (isinstance(p2, cst.Attribute) and p2.attr.value == "cuda"):
        return False
    p3 = p2.value
    return isinstance(p3, cst.Name) and p3.value == "torch"


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
TRANSFORMS = {
    "device_string": DeviceStringTransform,
    "cuda_to_xpu": CudaToXpuTransform,
    "dot_cuda": DotCudaTransform,
    "imports": ImportsTransform,
    "dist_backend": DistBackendTransform,
    "amp_autocast": AmpAutocastTransform,
    "amp_gradscaler": AmpGradScalerTransform,
}


def rewrite_file(
    path: Path, transform_cls: type
) -> tuple[bool, str, str] | None:
    """Return (changed, before, after), or None if the file cannot be read
    or parsed."""
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        print(f"rewrite: skip {path}: read error: {exc}", file=sys.stderr)
        return None
    if nesting_too_deep(source):
        print(
            f"rewrite: skip {path}: bracket nesting exceeds "
            f"{MAX_NESTING_DEPTH}; parsing this file would crash the parser",
            file=sys.stderr,
        )
        return None
    # The transform runs inside the same try as the parse: the visitor recurses
    # over the tree (and MetadataWrapper deep-clones it), so RecursionError
    # surfaces here rather than from parse_module. Source that recurses without
    # brackets (a long run of unary operators) parses fine and only blows up on
    # the walk, which nesting_too_deep cannot predict.
    try:
        tree = cst.parse_module(source)
        # Compare the transform's output against the *regenerated* original,
        # not against the bytes on disk. parse-then-regenerate is not always
        # identity (a form feed, for instance, does not survive the round
        # trip), so comparing to `source` reports a change the transform did
        # not make and writes the file back over an edit nobody asked for.
        baseline = tree.code
        needs_metadata = bool(
            getattr(transform_cls, "METADATA_DEPENDENCIES", ())
        )
        if needs_metadata:
            wrapper = cst.MetadataWrapper(tree)
            new_tree = wrapper.visit(transform_cls())
        else:
            new_tree = tree.visit(transform_cls())
        new = new_tree.code
    except PARSE_ERRORS as exc:
        print(f"rewrite: skip {path}: parse error: {exc}", file=sys.stderr)
        return None
    return new != baseline, source, new


def walk(root: Path, exclude: set[str]) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in exclude for part in p.parts):
            continue
        out.append(p)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Apply one mechanical CUDA-to-XPU transform with libcst."
    )
    ap.add_argument(
        "--transform",
        required=True,
        choices=sorted(TRANSFORMS.keys()),
        help="which transform to apply",
    )
    ap.add_argument(
        "--path", type=Path, required=True, help="repo root or .py file"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="print unified diff to stdout, do not write",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[".git", ".venv", "venv", "__pycache__", "build", "dist"],
    )
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"path does not exist: {args.path}", file=sys.stderr)
        return 2

    transform_cls = TRANSFORMS[args.transform]
    if args.path.is_file():
        files = [args.path]
    else:
        files = walk(args.path, set(args.exclude))

    changed_files = 0
    skipped_files = 0
    for f in files:
        result = rewrite_file(f, transform_cls)
        if result is None:
            skipped_files += 1
            continue
        changed, before, after = result
        if not changed:
            continue
        changed_files += 1
        if args.check:
            sys.stdout.writelines(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=str(f),
                    tofile=str(f),
                )
            )
        else:
            f.write_text(after, encoding="utf-8")
            print(f"rewrote {f}", file=sys.stderr)

    summary = (
        f"{args.transform}: {changed_files} file(s) "
        f"{'would be ' if args.check else ''}changed"
    )
    if skipped_files:
        summary += f", {skipped_files} skipped (parse error)"
    print(summary, file=sys.stderr)
    return 1 if skipped_files else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
