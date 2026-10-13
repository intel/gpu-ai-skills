---
name: torch-tweak
description: "Tune a PyTorch nn.Module or pipeline for Intel GPU inference with Torch Tweak (https://github.com/intel/torch-tweak). Compare OpenVINO, TorchAO, Torch Inductor, and eager, keep the fastest correct backend, and save a .tt checkpoint. Use when the user asks to tune, optimize, compile, or convert a general PyTorch model such as ResNet, BERT, diffusion, speech, or a custom module, and is not asking for an OpenAI server. Trigger phrases include \"tune this model\", \"optimize inference\", \"compile with OpenVINO\", and \"torch-tweak\". Not for vLLM or SGLang serving (use vllm-xpu-run or sglang-xpu-run), not for loading a Hugging Face model in eager PyTorch (use torch-xpu-run), and not for measuring an already chosen generate() path (use torch-xpu-bench)."
---

# torch-tweak

[Torch Tweak](https://github.com/intel/torch-tweak) tunes a PyTorch
`nn.Module` or pipeline in process. It tries compile and conversion
backends, checks the tuned module, and can save the result. It is
pre-release (pre-1.0). Tell the user that before installing. The API
can change.

Use it for a general PyTorch model or pipeline that has no serving
stack: vision, encoders, speech, diffusion, or a custom module.
Decoder-only LLMs that need continuous batching, prefix caching, or an
OpenAI-compatible server stay on **vllm-xpu-run** or **sglang-xpu-run**.

## When to use

- "Tune this ResNet / BERT / diffusion pipeline on my Intel GPU."
- "Optimize inference" or "compile this module" without naming a server.
- "Try OpenVINO, TorchAO, and torch.compile, and keep the fastest."
- The user names Torch Tweak or `torch_tweak`.

## When not to use

| User need | Skill |
|---|---|
| Load or run a Hugging Face model in eager PyTorch | **torch-xpu-run** |
| Measure `generate()` or a diffusion call you already chose | **torch-xpu-bench** |
| Explain which op is slow | **torch-xpu-profile** |
| OpenAI-compatible serving | **vllm-xpu-run** or **sglang-xpu-run** |
| GGUF | **llamacpp-xpu-run** |
| Coordinated server deploy plan | **xpu-deploy-plan** |

If `torch.xpu.is_available()` is false, stop. Do not tune on CPU unless
the user explicitly asks for a CPU run. Point at **xpu-discover** or
**xpu-system-setup**.

## Confirm before installing

**STOP.** Before `git clone`, `pip install`, or a weight download, ask
the user to confirm:

1. The module or pipeline to tune, and a small sample that matches its
   forward inputs.
2. That cloning `intel/torch-tweak` and installing it (this pulls
   OpenVINO and TorchAO) plus a PyTorch XPU wheel is acceptable.

Do not run `pip install`, `uv pip install`, or a model download until
the user explicitly confirms.

## Install

PyTorch XPU wheels are not on the default PyPI index. Torch Tweak is
installed from the GitHub repo, not with a bare `pip install torch-tweak`.
Source: the install guide in that repo
(`docs/learn/install.md`).

Check what is already present:

```sh
python3 -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
python3 -c "import torch_tweak; print('torch_tweak ok')"
```

If torch is missing or `torch.xpu.is_available()` is false on Intel GPU
hardware, install the XPU wheel first (see **torch-xpu-run**, section
"Where to get PyTorch with XPU"). Then:

```sh
git clone https://github.com/intel/torch-tweak
cd torch-tweak
pip install \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple \
    -e ".[xpu]"
```

`--break-system-packages` is needed under PEP 668 on Ubuntu 24.04+;
omit it in older images, or use a venv. Do not add an unpinned
`pip install torch` from PyPI next to this command. That replaces the
XPU build.

Verify:

```sh
python3 -c "import torch, torch_tweak; print(torch.__version__, torch.xpu.is_available())"
```

## Quickstart (one module)

`HighestThroughputStrategy()` with no backends list compares only
Torch Inductor and eager. Pass OpenVINO, TorchAO, Torch Inductor, and
eager explicitly or the conversion backends never run.

Pass at least two batch sizes when tensor shapes can vary. One batch
size forces static shapes. `.enable_find_max_batch_size(False)` skips
the max-batch search on the first tune.

A backend that fails to build is logged and skipped. The strategy
raises only when every backend fails.

```python
import logging

import torch
import torch.nn as nn

import torch_tweak.torch as tt
from torch_tweak.torch.backend import (
    OpenVINOBackend,
    TorchAOBackend,
    TorchEagerBackend,
    TorchInductorBackend,
)

logging.basicConfig(level=logging.INFO, force=True)

if not torch.xpu.is_available():
    raise SystemExit("No XPU. Fix the device before tuning.")

device = "xpu"
model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval().to(device)
sample = torch.randn(32, device=device)

strategy = tt.HighestThroughputStrategy(
    backends=[
        OpenVINOBackend(),
        TorchAOBackend(),
        TorchInductorBackend(),
        TorchEagerBackend(),
    ]
).enable_find_max_batch_size(False)

module = tt.Module(model, name="mlp", strategy=strategy)
tt.tune(module, sample, batch_sizes=[1, 4], device=device)
tt.save(module, "tuned_mlp.tt")
```

Replace the MLP and `sample` with the user's module. The sample's last
dimension has to match the module. After `tune`, call the module for
inference in this process, then `save`. Load later with `tt.load` on a
fresh copy of the same module:

```python
module = tt.Module(model, name="mlp", strategy=strategy)
tt.load(module, "tuned_mlp.tt")
```

Deploy the `.tt` file on the same hardware used for tuning.

## Pipeline

For a pipeline (Stable Diffusion and similar), inspect, wrap the
modules, then tune the pipeline. Full walkthrough:
<https://github.com/intel/torch-tweak/blob/main/docs/learn/quick_start.md>

```python
modules_info = tt.inspect(pipe, input_data)
modules_info.describe()
pipe = tt.wrap(pipe, modules_info.get_modules(), strategy=strategy)
tt.tune(pipe, input_data, batch_sizes=[1, 2], device="xpu")
tt.save(pipe, "tuned_pipe.tt")
```

Use the same explicit backend list as the quickstart. Conditional
logic inside a module is a graph break and blocks tuning of that
module. Pick a nearby module without the branch, or leave it eager.

To force one backend instead of searching:

```python
from torch_tweak.torch.backend import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.tune_strategy import OneBackendStrategy

strategy = OneBackendStrategy(
    backend=OpenVINOBackend(OpenVINOBackendConfig(performance_hint="THROUGHPUT"))
)
```

`OneBackendStrategy` surfaces the original exception.
`FirstWinsStrategy` tries backends in order and keeps the first one
that builds. Use `HighestThroughputStrategy` when the user wants the
fastest measured path.

## What this skill does NOT cover

- Serving, continuous batching, speculative decoding.
- Host driver setup (**xpu-system-setup**, **xpu-discover**).
- A benchmark of an already chosen eager path (**torch-xpu-bench**).
- Training.

## References

- Repo and guides: <https://github.com/intel/torch-tweak>
- Install: <https://github.com/intel/torch-tweak/blob/main/docs/learn/install.md>
- Quick start: <https://github.com/intel/torch-tweak/blob/main/docs/learn/quick_start.md>
- Backends: <https://github.com/intel/torch-tweak/blob/main/docs/learn/backends.md>
- Core API: <https://github.com/intel/torch-tweak/blob/main/docs/learn/core_functionalities.md>
