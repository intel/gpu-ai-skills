#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Read-only CUDA/NVIDIA surface inventory. Run from the target repo root.

set -eu

if command -v rg >/dev/null 2>&1; then
    rg -n --hidden \
        --glob '!*.ipynb_checkpoints*' \
        --glob '!**/SKILL.md' \
        --glob '!**/agents/AGENTS.md' \
        --glob '!*_xpu.py' \
        'cuda|CUDA|cudnn|cuDNN|nvidia|NVIDIA|nvidia-smi|--gpus|CUDA_VISIBLE_DEVICES|torch\.cuda|\.cuda\(|cupy|triton|Triton|bitsandbytes|flash_attn|flash-attn|nccl|NCCL|tensorrt|TensorRT|cublas|cuBLAS|cutlass|CUTLASS|cutensor|cuTENSOR|cusparse|cuSPARSE|cusolver|cuSOLVER|cufft|cuFFT|curand|cuRAND|thrust|nvcc|\.cu\b|\.cuh\b|cpp_extension|torch\.compile|set_float32_matmul_precision|\.pin_memory\(' \
        . || true
    rg -n --hidden \
        --glob '!*.ipynb_checkpoints*' \
        --glob '!**/SKILL.md' \
        --glob '!**/agents/AGENTS.md' \
        --glob '!*_xpu.py' \
        'docker run|compose|Dockerfile|requirements|pyproject|environment\.yml|vllm serve|sglang|bench|profile|device_map|attn_implementation' \
        . || true
else
    grep -RInE \
        'cuda|CUDA|cudnn|cuDNN|nvidia|NVIDIA|nvidia-smi|--gpus|CUDA_VISIBLE_DEVICES|torch\.cuda|\.cuda\(|cupy|triton|Triton|bitsandbytes|flash_attn|flash-attn|nccl|NCCL|tensorrt|TensorRT|cublas|cuBLAS|cutlass|CUTLASS|cutensor|cuTENSOR|cusparse|cuSPARSE|cusolver|cuSOLVER|cufft|cuFFT|curand|cuRAND|thrust|nvcc|\.cu\b|\.cuh\b|cpp_extension|torch\.compile|set_float32_matmul_precision|\.pin_memory\(|docker run|compose|Dockerfile|requirements|pyproject|environment\.yml|vllm serve|sglang|bench|profile|device_map|attn_implementation' \
        . || true
fi
