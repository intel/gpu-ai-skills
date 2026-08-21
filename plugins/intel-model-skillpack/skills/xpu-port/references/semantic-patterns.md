# Semantic patterns

The scanner flags these but does not rewrite them. The agent decides
the right edit. Single-target XPU is assumed (see SKILL.md
"Targets"); for dual-target, this skill is the wrong tool.

## TF32 toggles

`torch.backends.cuda.matmul.allow_tf32 = True` and
`torch.backends.cudnn.allow_tf32 = True` have no effect on XPU.
`torch.set_float32_matmul_precision()` is documented CUDA-only.

**Action: delete the lines.** They do nothing on the target device
and there is no XPU equivalent that preserves the original intent.

## Device-type gates

Scanner categories: `device_label_compare`, `device_fstring_label`.
Code like `device_type == 'cuda'`, `'cuda' if 'cuda' in device else
'cpu'`, or `if device_type == 'cuda':` is a runtime branch keyed on
a device *label*, not a tensor-move target. The scanner does not
rewrite these because blindly flipping the literal would change
behaviour rather than correct it (the gate could be intended to
widen or flip — only the agent knows).

**Action: hand-edit to use `'xpu'` as the gate value.**

```python
# before
device_type = 'cuda' if 'cuda' in device else 'cpu'
if device_type == 'cuda':
    ...
use_fused = fused_available and device_type == 'cuda'

# after (single-target XPU)
device_type = 'xpu' if 'xpu' in device else 'cpu'
if device_type == 'xpu':
    ...
use_fused = fused_available and device_type == 'xpu'
```

For fused AdamW specifically, XPU is in
`torch.utils._foreach_utils._get_fused_kernels_supported_devices()`,
so the flipped gate enables the fused path correctly.

## `torch.cuda.is_available()` checks

Single-target XPU port: these are rewritten by `cuda_to_xpu` to
`torch.xpu.is_available()`. Do not hand-edit them back. If
dual-target is needed, this skill is the wrong tool.

## Hard-coded CUDA-only library imports

`flash-attn`, `bitsandbytes`, etc. Escalate. Pick the XPU-aware
fallback (`attn_implementation="sdpa"`, AutoRound for quant) and
remove the hard import.

## After-profile op replacements

Only after `torch-xpu-profile` shows the hot op:

- Hand-rolled `(Q @ K.T / sqrt(d)).softmax() @ V` →
  `F.scaled_dot_product_attention`
- Hand-rolled RMSNorm → `F.rms_norm`
- Hand-rolled LayerNorm → `F.layer_norm`
- Conv-heavy NCHW model → set
  `memory_format=torch.channels_last`

## `torch.compile` (Inductor backend)

`torch.compile(model)` invokes the Inductor backend, which needs
XPU-specific compiler support (`pytorch-triton-xpu`). Whether a given
graph lowers is build- and graph-dependent: trivial pointwise graphs
compile on current Battlemage Arc builds, but a full transformer
forward+backward graph was observed to abort on a mid-2026 B60
PyTorch-XPU build (the nanoGPT port). So this is not "on" or "off" —
it has to be tested on the actual target.

**Action: test on the target, then keep or guard.**

1. Run `torch.compile` on the target device and execute at least one
   real training step (the one-forward verify does not exercise it).
2. If it compiles and the step runs, **keep it** — Inductor is a real
   speedup, and disabling it regresses performance.
3. Only if Inductor fails to lower on that build, add a fallback guard:

```python
# fallback only — when torch.compile fails to lower on the target build
if compile and device_type != 'xpu':
    model = torch.compile(model)
```

Do not disable `torch.compile` on XPU by default, and do not hardcode
"never supported" — Inductor-XPU coverage is a moving target that
improves across PyTorch-XPU builds.

## `torch.profiler.ProfilerActivity.CUDA`

When profiling code uses `ProfilerActivity.CUDA`, it must be changed
to `ProfilerActivity.XPU` for Intel GPUs.

**Action: one-line swap.**

```python
# before
from torch.profiler import ProfilerActivity
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

# after
from torch.profiler import ProfilerActivity
activities = [ProfilerActivity.CPU, ProfilerActivity.XPU]
```

Seven entries, not a kernel encyclopedia. Anything else, profile
and reason from data.
