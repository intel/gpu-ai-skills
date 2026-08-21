# Concurrency sweeps, quant benches, run comparisons

## Concurrency sweep (find the knee)

```sh
for c in 1 2 4 8 16 32 64; do
    vllm bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --host 127.0.0.1 --port 8000 \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len 512 --random-output-len 128 \
        --num-prompts $((200 * (c < 8 ? 1 : c / 4))) \
        --max-concurrency "$c" \
        --metric-percentiles 50,99 \
        --save-result --result-dir "/root/bench-out/c$c"
done
```

The "knee" is the concurrency where throughput plateaus while
p99 TPOT starts climbing. On Arc Pro B70 with a 1–3B model that's
usually around 8–16. Beyond it you trade latency for nothing.

## Benchmarking quantized serving

Bench numbers are meaningful only when the quant kernel is
actually engaged. Most common silent failure: W4A8 falling
through to W4A16 (vLLM #38064) — int4 weights load, requests
succeed, activations are still FP16.

1. Confirm the kernel:
   ```sh
   docker logs <name> 2>&1 | grep -E "Selected.*Kernel|XPUFP8|gemm"
   docker logs <name> 2>&1 | grep -i "Unknown vLLM environment"
   ```
   If you asked for AWQ/GPTQ but see `int4_gemm_w4a16`, you're
   hitting the fall-through.
2. **Read 2–3 sample completions** from the saved JSON before
   trusting throughput. Some quant kernels return non-language
   output without raising; HTTP 200 doesn't prove correctness.

For apples-to-apples between quant kinds, keep
`--no-enable-prefix-caching` on the server — random prompts can
get artificial cache hits otherwise.

## Order-of-magnitude reference (Arc Pro B70, BF16, single GPU, prompt 512 / gen 128, concurrency 1)

| Model size | TTFT (ms) | TPOT (ms/tok) | Single-stream tok/s |
|---|---|---|---|
| 0.5 B | tens | low single digits | low hundreds |
| 1.5 B | tens | ~10 | ~100 |
| 7–8 B | hundreds | tens | ~30 |
| 14 B BF16 | OOM at default | — | needs FP8 or `-tp` |

Order-of-magnitude only; real numbers move with each vLLM release.
Save a baseline JSON, diff against it on upgrades.

## Comparing two runs

Save baseline + candidate to separate `--result-dir`s, then:

```sh
python3 - <<'PY'
import glob, json
def load(d):
    f, = glob.glob(f"{d}/*.json")
    return json.load(open(f))
a = load("bench-out/baseline")
b = load("bench-out/candidate")
for k in ("mean_ttft_ms","p99_ttft_ms","mean_tpot_ms","p99_tpot_ms",
         "request_throughput","output_throughput"):
    print(f"{k:24s}  base={a[k]:.2f}  cand={b[k]:.2f}  delta={b[k]-a[k]:+.2f}")
PY
```

>5% regression on `output_throughput` or >10% on `p99_tpot_ms`
between same-flag runs of the same (model, GPU) is a real
regression. Smaller is usually noise.
