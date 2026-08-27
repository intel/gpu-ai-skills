# Bill of Materials — intel-gpu-ai-skills

**Version:** 0.1.0  
**Release artifact:** branch `pre-release-v0.1`  
**Outbound licence:** Apache-2.0 (`LICENSE`, `NOTICE`)  
**Copyright:** Intel Corporation  
**Date:** 2026-08-19  

**Scope:** The release branch `pre-release-v0.1` only.

---

## 1. First-party content

| Property | Statement |
|---|---|
| Origin | Created from scratch by Intel |
| Outbound Licence | Apache-2.0 |
| Third-party code vendored | **None** |
| Third-party data vendored | **None** |
| Binaries, archives, submodules | **None** (100% text/source) |


---

## 2. Third-party components

**Distributed by you:** **NO** (0 components shipped).  
Every item below is an external dependency installed independently by the end user at runtime.

| Component / Layer | Version / Scope | Licence | Origin / Registry |
|---|---|---|---|
| Linux Kernel (`xe` driver) | ≥ 6.8 | GPL-2.0 | Distro |
| Intel Compute / Level Zero Runtime | Current | MIT | repositories.intel.com / GitHub |
| `intel-opencl-icd`, `intel-ocloc`, `xpu-smi` | Current | MIT | repositories.intel.com / GitHub |
| `intel-gsc` | Current | Apache-2.0 | repositories.intel.com |
| Docker CE / Compose | Current | Apache-2.0 | Docker Official |
| CPython | ≥ 3.11 | PSF-2.0 | python.org |
| `torch` + `torchvision` + `torchaudio` | ≥ 2.8+xpu | BSD-3-Clause / BSD-2-Clause | PyTorch XPU Index |
| `transformers`, `huggingface_hub`, `safetensors`, `accelerate` | PyPI current | Apache-2.0 | PyPI |
| `libcst` | PyPI current | MIT / PSF-2.0 / Apache-2.0 | PyPI |
| `httpx[socks]` | PyPI current | BSD-3-Clause / MIT | PyPI |
| `vllm/vllm-openai-xpu`, `intel/sglang-dev` | Container images | Apache-2.0 | Docker Hub |
| `llama.cpp` | SYCL backend | MIT | ggml-org/llama.cpp |
| `unitrace` | Build from source | MIT | intel/pti-gpu |

---

## 3. Models & Datasets

* **Shipped:** **None** (No weights, configs, or tokenizers are packaged or distributed).
* **Referenced at Runtime:** Public models only (`Qwen`, `openai`, `google`, `meta-llama`, `mistralai`, `deepseek-ai`). End users fetch configs and weights directly under upstream terms.

---

## 4. Attestation & Status

* Outbound Apache-2.0 licensing is applied consistently across all release files (`LICENSE`, `NOTICE`, and `README.md`).
* No internal URLs, private endpoints, or unreleased packages are referenced.
* Repository history contains zero vendored third-party data or cached configurations.

* **Open Items:** **0 (Closed)**