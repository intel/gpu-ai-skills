# Non-LLM bench snippets

`scripts/bench.py` is for causal LMs. For diffusion and encoder-only
models, paste these and adapt. Same procedure as the main bench:
pin device, reset peak memory, warm up, synchronise around the
timer, report median.

## Diffusion (SDXL / Flux / SD)

```python
import time, statistics, torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
).to("xpu")
assert next(pipe.unet.parameters()).device.type == "xpu"

for _ in range(2):
    pipe("a cat", num_inference_steps=10).images

times = []
for _ in range(5):
    torch.xpu.reset_peak_memory_stats(0)
    torch.xpu.synchronize()
    t = time.perf_counter()
    pipe("a cat", num_inference_steps=30).images
    torch.xpu.synchronize()
    times.append(time.perf_counter() - t)
print(f"median {statistics.median(times):.2f} s, "
      f"peak {torch.xpu.max_memory_allocated(0)/1024**3:.2f} GiB")
```

Headline: seconds per generation at N steps. Sweep
`num_inference_steps`, `height`, `width`.

## Encoder-only (BERT, ViT, sentence-transformers)

```python
import time, statistics, torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5",
                                   dtype=torch.bfloat16).to("xpu").eval()
assert next(model.parameters()).device.type == "xpu"
tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
batch = tok(["hello"] * 32, padding="max_length", max_length=256,
            truncation=True, return_tensors="pt").to("xpu")

with torch.no_grad():
    for _ in range(3):
        model(**batch)
    times = []
    for _ in range(10):
        torch.xpu.synchronize()
        t = time.perf_counter()
        model(**batch)
        torch.xpu.synchronize()
        times.append(time.perf_counter() - t)
print(f"forward median {statistics.median(times)*1000:.2f} ms, "
      f"{32/statistics.median(times):.1f} samples/s")
```

Headline: forward-pass latency at (batch, seq), plus samples/s.
