---
name: torch-xpu-run
description: Run an arbitrary Hugging Face safetensors model on an Intel GPU using **upstream PyTorch** (>= 2.8) with the built-in `torch.xpu` device. Covers loading from the Hub, picking the right dtype, autocast, multi-GPU with accelerate's `device_map`, and the CUDA -> XPU code translation a user has to do once. Use for the Transformers / Accelerate / Diffusers path. Not for OpenAI-compatible serving (use vllm-xpu-run); explicitly not via intel-extension-for-pytorch (ipex) or ipex-llm — those paths are end-of-life and upstream PyTorch supersedes them.
---

# torch-xpu-run

Upstream PyTorch since 2.8 has a first-class `torch.xpu` namespace
mirroring `torch.cuda`. Don't use `intel-extension-for-pytorch`
(`ipex`) or `ipex-llm` — both are end-of-life (March 2026), upstream
PyTorch supersedes them.

## CUDA -> XPU code translation

| CUDA | XPU |
|---|---|
| `torch.cuda.is_available()` | `torch.xpu.is_available()` |
| `torch.cuda.device_count()` | `torch.xpu.device_count()` |
| `torch.cuda.empty_cache()` | `torch.xpu.empty_cache()` |
| `torch.cuda.synchronize()` | `torch.xpu.synchronize()` |
| `torch.cuda.memory_allocated(0)` | `torch.xpu.memory_allocated(0)` |
| `model.to("cuda")` | `model.to("xpu")` |
| `tensor.to("cuda:1")` | `tensor.to("xpu:1")` |
| `with torch.autocast("cuda", torch.bfloat16)` | `with torch.autocast("xpu", torch.bfloat16)` |
| `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler("xpu")` |
| `device_map="auto"` (Accelerate) | same; Accelerate detects XPU directly |
| `dist.init_process_group(backend="nccl")` | `dist.init_process_group(backend="xccl")` <- only non-mechanical change |

## Where to get PyTorch with XPU

XPU wheels are **not** on the default PyPI index. A plain
`pip install torch` gets the CUDA/CPU build, where `torch.xpu` exists
as a namespace but reports no devices. Install from the XPU index:

```sh
# stable
pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

# nightly — only when you need an unreleased fix
pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/xpu
```

Pinning works the same way, e.g. `pip install torch==2.11.0
torchvision==0.26.0 torchaudio==2.11.0 --index-url
https://download.pytorch.org/whl/xpu`. Take the three versions from one
release row — mixing rows breaks the ABI.

The Intel GPU driver must already be installed on the host
(`xpu-discover` / `xpu-runtime-preflight` verify this). Binary wheels do
**not** need Intel Deep Learning Essentials; only source builds do.

Verify:

```sh
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
```

`torch.xpu.is_available() is True` is the authoritative signal. Linux
wheels from the XPU index also carry a `+xpu` local version suffix
(`2.8.0+xpu`), but source-built images — including some vendor serving
images — drop it, so a missing suffix is not proof XPU is absent.
`is_available()` returning `False` on XPU hardware almost always means a
missing host driver or a container that can't see `/dev/dri`.

Three options:

1. **Local venv** — same install command, no container.
2. **Build a thin Dockerfile** on `ubuntu:24.04` or
   `python:3.12-slim` and run the XPU-index install above. Canonical
   docs are the PyTorch XPU notes:
   <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>
   (the <https://pytorch.org/get-started/locally/> selector emits the
   same command once you pick Linux + Pip + Python + Intel GPU, but it
   is JS-rendered and shows nothing XPU-related when fetched as text).
3. **Reuse a serving image** — `intel/vllm:*-xpu` ships a working
   torch-xpu inside; start with `--entrypoint /bin/bash`.

Launch with the GPU visible (see **xpu-container-run** for full
flags):

```sh
docker run --rm -it \
    --device /dev/dri \
    --ipc=host \
    -e ZE_AFFINITY_MASK=0 \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --entrypoint /bin/bash \
    <torch-xpu-image>
```

## Quickstart

**STOP — confirm before proceeding.** Before installing packages or
downloading weights, ask the user to confirm:
1. The model ID (and dtype if not bf16)
2. That installing packages (`pip install torch transformers accelerate`)
   and downloading multi-GB weights is acceptable

Do not run `pip install`, `uv pip install`, or model download commands
until the user explicitly confirms. This is a hard requirement.

**Check before installing.** Always verify packages are already present
before running `pip install`:

```sh
python3 -c "import transformers; print(transformers.__version__)" 2>&1 && \
python3 -c "import accelerate; print(accelerate.__version__)" 2>&1
```

Only install if the import check fails:

```sh
pip install --quiet --break-system-packages 'transformers>=4.46' accelerate
```

(`--break-system-packages` is needed under PEP 668 in Ubuntu
24.04+; omit in older images, or use a venv.)

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# Fetch config first to apply pre-load patches.
cfg = AutoConfig.from_pretrained(model_id)

# Rope-scaling: older models omit the 'type' field required by
# Transformers 5.x; inject it to prevent a KeyError on load.
rope = getattr(cfg, "rope_scaling", None)
if isinstance(rope, dict) and "type" not in rope:
    rope["type"] = rope.get("rope_type", "linear")

tok = AutoTokenizer.from_pretrained(model_id)
# Many models ship without a pad token; set it to avoid
# 'does not have a padding token' on batched calls.
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=cfg,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="xpu",
)
inputs = tok("Tell me a joke.", return_tensors="pt").to("xpu")
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0], skip_special_tokens=True))
```

> **`trust_remote_code`:** If the load raises `unrecognized configuration
> class` or the model card shows `config.auto_map`, add
> `trust_remote_code=True` to the `from_pretrained` calls — but warn the
> user first, since this runs the repo's custom Python modules.

## Preflight checklist — settings to apply when a load fails or is new

Apply these in response to specific load errors, or as a starting point for
a model you haven't run on XPU before. Each setting addresses a named signal.

1. **`low_cpu_mem_usage=True`** — avoids loading all weights to CPU RAM
   before copying to XPU (peaks at 2× model size). Apply when you see a
   CPU OOM before the XPU load completes.
2. **`dtype=torch.bfloat16`** — default; halves memory vs fp32. See "Pick the right dtype" below.
3. **`tokenizer.pad_token = tokenizer.eos_token`** — apply when a
   batched call raises `does not have a padding token`. Many models
   (GPT-2, Llama, Qwen) ship without one.
4. **Normalise `config.rope_scaling`** — apply when load raises a
   `KeyError` on `rope_scaling['type']`. Inject `type = rope_scaling.get(
   'rope_type', 'linear')` before calling `from_pretrained`. See
   `xpu-transformers-compat` for the full set of Transformers 5.x shims.
5. **Warn on model size** — before a long checkpoint download, check
   whether the model fits the available VRAM; see `model-can-it-fit`.
6. **Pick the correct loader class** — `AutoModelForCausalLM` for
   decoder LLMs; vision / audio / seq2seq / reward / time-series need
   different classes. See `xpu-model-type-detect`.

## Pick the right dtype

- **`bfloat16`** — default. Battlemage / Arc Pro have full hardware support; `float16` works for most ops but a small set degrades
  or falls back to slow paths.
- **`float32`** — diagnostic fallback when bf16 fails to load
  (rare). Halves memory; not for production.

For quantized models on XPU:

- **Intel AutoRound** (Int4 / Int3 / Int2) is the recommended
  algorithm. Exports as AutoAWQ-style or AutoGPTQ-style packing;
  runtimes auto-detect via `quantization_config.quant_method=auto-round`.
- **AutoAWQ** — loads through Transformers; verify output content,
  not just successful load.
- **GPTQ** — regressed in vLLM v0.19.0 (vLLM #39474); pin v0.18.x
  or use AutoRound's GPTQ-format export.
- **bitsandbytes** — limited XPU support; prefer AutoRound.

For full per-quant CLI / env vars when serving, see
**vllm-xpu-run** Quantization section.

## Multi-GPU on one host

### Single-process, multiple XPUs (`device_map`)

Make both XPUs visible (`-e ZE_AFFINITY_MASK=0,1`), then:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)
```

Accelerate prints layer placement; verify both `xpu:0` and `xpu:1`.

### One process per XPU (DDP / multiple servers)

```python
import torch.distributed as dist
dist.init_process_group(backend="xccl")    # upstream XPU collective backend
```

Backend is `"xccl"`, not `"nccl"`. The older `"ccl"` value targets the
deprecated `torch_ccl` plugin and will fail with current upstream
PyTorch. Launch via `torchrun --nproc_per_node=N` inside a container
that sees all XPUs, or one container per XPU with
`ZE_AFFINITY_MASK=N`.

## Verifying it ran on XPU

```python
print(model.device)                        # xpu:0
print(next(model.parameters()).device)     # xpu:0
print(torch.xpu.memory_allocated(0))       # > 0 after load
```

From the host while generating:

```sh
xpu-smi dump -d 0 -m 5,18 -i 1
```

Memory should climb when the model loads. If it doesn't, the model
is on CPU.

## Common errors

- `Cannot find any XPU devices` -> container missing GPU access;
  see **xpu-container-run**.
- `Torch not compiled with XPU enabled` -> wrong PyTorch build. The
  image must have `torch.__version__` ending in `+xpu`.
- `OSError: Tokenizer ... requires Hub access` -> set `HF_TOKEN` or
  `huggingface-cli login`.
- `CUDA error: ...` literal substring inside an XPU workload -> a
  third-party library (older `bitsandbytes`, `flash-attn`,
  `xformers`) is hard-coded to CUDA. Use an XPU-aware fork or fall
  back to pure PyTorch (Transformers' `attn_implementation="sdpa"`
  covers the common attention case).
- `Expected one of cpu, cuda, ... device type at start of device
  string: xpu` -> very old `transformers` (<4.46) or `accelerate`
  (<0.34). Upgrade.
- `model type '<X>' Transformers does not recognize` -> installed
  transformers is older than the model architecture. Upgrade or
  pin per the model card; install from source if needed
  (`pip install git+https://github.com/huggingface/transformers.git`).
- `float16 not supported on this device` -> switch to
  `dtype=torch.bfloat16`.
- `Unknown scheme for proxy URL ... 'socks://...'` -> `httpx` (used
  by `huggingface_hub`) doesn't support SOCKS proxies without the
  optional transport. Fix: `pip install httpx[socks]`, or unset the
  proxy for that session: `unset ALL_PROXY all_proxy`. If the model
  is already cached, `HF_HUB_OFFLINE=1` also bypasses the issue.
- Hang at "Loading checkpoint shards" -> usually cold cache + slow
  network. Pre-`hf download` into `~/.cache/huggingface`.
- Hang on first `generate()` -> Triton kernel compile is cold. Set
  `TRITON_CACHE_DIR` to a persistent volume.

## Env vars

| Variable | Purpose |
|---|---|
| `ZE_AFFINITY_MASK` | Which XPU(s) the process sees. |
| `HF_TOKEN` | HF auth. |
| `TRITON_CACHE_DIR` | Persist XPU Triton kernels. |
| `TORCH_LOGS=+dynamo` | See what `torch.compile` rewrote. |
| `PYTORCH_ENABLE_XPU_FALLBACK` | `1` allows silent CPU fallback (debugging only); `0` for benches. |
| `OMP_NUM_THREADS` | Cap CPU threads; default oversubscribes. |
| `IGC_EnableAluBinding=1` | Battlemage matmul-codegen hint; bench both. |
| `CCL_ZE_IPC_EXCHANGE=pidfd` | Multi-XPU oneCCL IPC mechanism (default since 2021.14). |

## What this skill does NOT cover

- vLLM serving -> **vllm-xpu-run**.
- SGLang serving -> **sglang-xpu-run**.
- Per-model quant gotchas (AWQ quirks, GPTQ kernel coverage) — go
  in fix-level skills.
- Profiling and kernel-level fixes — out of scope.

## References

- Upstream PyTorch XPU notes: <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>
- Intel GPU/CPU enabling status, H1 2026: <https://dev-discuss.pytorch.org/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320>
- HF Accelerate XPU: <https://huggingface.co/docs/accelerate/en/usage_guides/xpu>
- Intel AutoRound: <https://github.com/intel/auto-round>
