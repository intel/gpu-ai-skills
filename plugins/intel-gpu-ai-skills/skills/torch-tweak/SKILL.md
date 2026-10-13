---
name: torch-tweak
description: "Tune a PyTorch nn.Module or pipeline for Intel GPU inference with Torch Tweak (https://github.com/intel/torch-tweak). Compare OpenVINO, TorchAO, Torch Inductor, and eager, keep the fastest correct backend, and save a .tt checkpoint. It looks at the input samples you pass and finds which sizes are the batch and which sizes can change. inspect walks into the submodules. You set min_depth and max_depth. One HighestThroughputStrategy then tunes each submodule on its own and keeps the fastest backend for that submodule and for each input shape. Use when the user asks to tune, optimize, compile, or convert a general PyTorch model such as ResNet, BERT, diffusion, speech, or a custom module, and is not asking for an OpenAI server. Trigger phrases include \"tune this model\", \"optimize inference\", \"compile with OpenVINO\", and \"torch-tweak\". Not for vLLM or SGLang serving (use vllm-xpu-run or sglang-xpu-run), not for loading a Hugging Face model in eager PyTorch (use torch-xpu-run), and not for measuring an already chosen generate() path (use torch-xpu-bench)."
---

# torch-tweak

[Torch Tweak](https://github.com/intel/torch-tweak) tunes a PyTorch
`nn.Module` or pipeline in process. It tries compile and conversion
backends, checks the tuned module, and can save the result. It is
pre-release (pre-1.0). Tell the user that before installing. The API
can change.

It looks at the samples you pass and marks which sizes are the batch
and which sizes can change. Use one strategy, usually
`HighestThroughputStrategy`. `inspect` walks into the submodules.
`tune` then measures each one on its own and keeps the fastest backend
for that submodule. See **Input shapes** and **Submodules**.

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

Replace the MLP and `sample` with the user's module. The sample must
be a real input to that module's `forward`. A ResNet takes an image
such as `torch.rand(3, 224, 224)`, not a 1-D vector. After `tune`, call the module for
inference in this process, then `save`. Load later with `tt.load` on a
fresh copy of the same module:

```python
module = tt.Module(model, name="mlp", strategy=strategy)
tt.load(module, "tuned_mlp.tt")
```

Deploy the `.tt` file on the same hardware used for tuning.

## Input shapes

Pass at least two batch sizes, for example `batch_sizes=[1, 4]`.
Torch Tweak runs the module on those samples and compares the shapes.

- A size that grows when the batch size grows is a batch axis.
- A size that changes for another reason is a dynamic axis. Sequence length is a common one.
- A size that never changes is static.

Same number of dimensions, different sizes: one graph. Torch Tweak keeps the dynamic sizes.
A different number of dimensions, or different non-tensor arguments: a new graph. That graph is tuned on its own.

Use samples that look like the real inputs. Later inputs must follow the same pattern of batch, dynamic, and static sizes.

## Submodules

Use one strategy for the whole run. The normal choice is
`HighestThroughputStrategy` from the quickstart. It measures the
backends and keeps the fastest one.

`tt.inspect` runs the model and walks into the submodules. Depth 0 is
the object you passed. `min_depth` is the shallowest module to keep.
`max_depth` is the deepest module to keep. The default `max_depth` is
5. Raise `min_depth` when the object you passed is only a wrapper and
the real work is in its children. `examples/E5Large` in the Torch Tweak
repo uses `min_depth=2`. `examples/ParakeetCTC` uses `min_depth=1`.

Call `modules_info.describe()`, then `get_modules()`. `tt.wrap` puts
the same strategy on each module in that list. `tt.tune` tunes those
modules one by one. The same strategy can keep OpenVINO for one
submodule and Torch Inductor for another. A different input shape on
the same submodule is measured on its own too.

```python
modules_info = tt.inspect(pipe, input_data, min_depth=0, max_depth=5)
modules_info.describe()
modules = modules_info.get_modules()
pipe = tt.wrap(pipe, modules, strategy=strategy)
tt.tune(pipe, input_data, batch_sizes=[1, 4], device="xpu")
tt.save(pipe, "tuned_pipe.tt")
```

`strategy` is the one `HighestThroughputStrategy` from the quickstart.
A submodule you leave out of `modules` stays eager.

Conditional logic inside a module is a graph break and blocks tuning
of that module. Raise `min_depth` so inspect keeps its children, or
leave that module out of `modules`.

Full walkthrough:
<https://github.com/intel/torch-tweak/blob/main/docs/learn/quick_start.md>

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
