# Speculative decoding and multimodal serving

## Speculative decoding (EAGLE, MTP)

```sh
python -m sglang.launch_server \
    --model <target> \
    --device xpu --tp 1 --attention-backend intel_xpu \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4

# DeepSeek-MTP-style:
    --speculative-algorithm MTP \
    --speculative-num-steps 1
```

Bench with and without on the same prompt set + concurrency. Keep
spec-decode only if TPOT improves without TTFT regression.

## Multimodal

- Verified: `google/gemma-3-4b-it` on Arc Pro B70 with the same
  flags as the text-gen quickstart. OpenAI vision API works
  (`messages[].content[]` with `type: image_url`).
- Unverified on this stack: `Qwen2.5-VL`, `Llama-3.2-Vision`,
  `LLaVA`. Try with the smoke-test pattern; validate output.

## Not on this stack today (route elsewhere)

- DeepSeek MLA path
- MoE fused kernels
- LoRA hot-swap
- `torch.compile` graph mode

For these on Intel, use **vllm-xpu-run**.
