#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bench a Hugging Face causal LM on XPU through pure torch + transformers.

Measures TTFT, decode-step latency, end-to-end throughput, and peak
XPU memory. Synchronises correctly on XPU and warms up before the
timed runs so kernel compile / graph capture do not pollute numbers.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--compile", action="store_true", help="Wrap the model in torch.compile()")
    p.add_argument("--device", default="xpu", help="Override device (default: xpu)")
    p.add_argument("--revision", default="main",
                   help="Pin the HF repo revision (branch, tag, or commit sha)")
    args = p.parse_args(argv)

    import torch
    if args.device.startswith("xpu") and not torch.xpu.is_available():
        sys.exit("torch.xpu.is_available() is False. See xpu-container-run / xpu-discover.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, DTYPES[args.dtype])

    print(f"Model:             {args.model}  (revision: {args.revision})")
    print(f"dtype:             {args.dtype}")
    name = torch.xpu.get_device_name(0) if args.device.startswith("xpu") else "(non-xpu)"
    print(f"torch:             {torch.__version__}  ({args.device} device 0: {name})")
    print(f"prompt_tokens={args.prompt_tokens}, gen_tokens={args.gen_tokens}, "
          f"warmup={args.warmup}, runs={args.runs}, compile={args.compile}")
    print()

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, revision=args.revision,
    )
    model.eval()

    placed = str(next(model.parameters()).device)
    if not placed.startswith(args.device):
        sys.exit(
            f"Model parameters landed on '{placed}', not '{args.device}'. "
            f"On a host with both Intel and NVIDIA GPUs, accelerate's "
            f"device_map can resolve to cuda. Pass --device xpu and "
            f"set CUDA_VISIBLE_DEVICES= (empty) to force XPU."
        )
    if args.compile:
        model = torch.compile(model)

    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Build a deterministic prompt of approximately --prompt-tokens length.
    seed_text = "The quick brown fox jumps over the lazy dog. " * 200
    enc = tok(seed_text, return_tensors="pt", truncation=True,
              max_length=args.prompt_tokens).to(args.device)
    real_prompt_tokens = int(enc.input_ids.shape[1])
    if real_prompt_tokens != args.prompt_tokens:
        print(f"note: tokenized to {real_prompt_tokens} tokens "
              f"(requested {args.prompt_tokens}).")

    def one_run() -> tuple[float, float, float]:
        torch.xpu.reset_peak_memory_stats(0) if args.device.startswith("xpu") else None
        sync = (lambda: torch.xpu.synchronize()) if args.device.startswith("xpu") else (lambda: None)

        # Time TTFT by capping max_new_tokens=1.
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**enc, max_new_tokens=1, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        sync()
        ttft_s = time.perf_counter() - t0

        # Full generate to measure end-to-end and derive decode rate.
        sync()
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.gen_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        sync()
        total_s = time.perf_counter() - t1

        gen = int(out.shape[1] - enc.input_ids.shape[1])
        decode_s_per_tok = (total_s - ttft_s) / max(gen - 1, 1)
        return ttft_s, decode_s_per_tok, total_s

    # Warmup.
    for _ in range(args.warmup):
        one_run()

    print(f"{'Run':>3}  {'TTFT (ms)':>9}  {'Decode (ms/tok)':>15}  "
          f"{'Total (s)':>10}  {'Throughput (tok/s)':>18}")
    ttfts, decodes, throughputs = [], [], []
    for i in range(1, args.runs + 1):
        ttft, dec, total = one_run()
        thr = args.gen_tokens / total
        ttfts.append(ttft); decodes.append(dec); throughputs.append(thr)
        print(f"{i:>3}  {ttft*1000:>9.2f}  {dec*1000:>15.2f}  "
              f"{total:>10.2f}  {thr:>18.2f}")

    med_ttft = statistics.median(ttfts) * 1000
    med_decode = statistics.median(decodes) * 1000
    med_thr = statistics.median(throughputs)
    print()
    print(f"Median: TTFT {med_ttft:.2f} ms, decode {med_decode:.2f} ms/tok, "
          f"throughput {med_thr:.2f} tok/s")

    if args.device.startswith("xpu"):
        peak = torch.xpu.max_memory_allocated(0) / (1024 ** 3)
        print(f"Peak XPU memory: {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
