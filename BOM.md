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
| Third-party code vendored | `vtune-vllm` skill: files from vllm-xpu-profiling-toolkit (Apache-2.0, Copyright 2026 Vasudha, commit `4d93a4d`); see section 2a and `plugins/intel-gpu-ai-skills/skills/vtune-vllm/references/PROVENANCE.md` |
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
| `vllm/vllm-openai-xpu`, `lmsysorg/sglang:<version>-xpu` | Container images | Apache-2.0 | Docker Hub |
| `llama.cpp` | SYCL backend | MIT | ggml-org/llama.cpp |
| `unitrace` | Build from source | MIT | intel/pti-gpu |
| Intel VTune Profiler | 2026.x | Intel End User License Agreement for Developer Tools | Intel oneAPI (apt `intel-oneapi-vtune` or standalone installer) |
| `ittapi` (Python ITT bindings, optional) | PyPI current | BSD-3-Clause / GPL-2.0 | PyPI (intel/ittapi) |

### 2a. Vendored third-party content

| Component | Version | Licence | Origin | Files | Modified |
|---|---|---|---|---|---|
| vllm-xpu-profiling-toolkit (`vtune/`) | commit `4d93a4d0ba5a` (2026-07-09) | Apache-2.0 (Copyright 2026 Vasudha; no upstream NOTICE) | https://github.com/Vasud-ha/vllm-xpu-profiling-toolkit | `vtune-vllm/SKILL.md`, `references/{itt-api,troubleshooting,vllm-integration,validation-and-flow,commands-and-workflow}.md`, `scripts/{run_vtune_vllm.sh,serve_with_vtune.py,vtune_itt.py}` | Yes, recorded in `vtune-vllm/references/PROVENANCE.md` (Apache-2.0 §4(b)) |

**Open item:** licensing / BOM review of this vendored component (added with the VTune skills).

---

## 3. Models & Datasets

* **Shipped:** **None** (No weights, configs, or tokenizers are packaged or distributed).
* **Referenced at Runtime:** Public models only (`Qwen`, `openai`, `google`, `meta-llama`, `mistralai`, `deepseek-ai`). End users fetch configs and weights directly under upstream terms.

---

## 4. Attestation & Status

* Outbound Apache-2.0 licensing is applied consistently across all release files (`LICENSE`, `NOTICE`, and `README.md`).
* No internal URLs, private endpoints, or unreleased packages are referenced.
* Repository history contains zero vendored third-party data or cached configurations.

* **Open Items:** **1** — licensing / BOM review of the vendored vllm-xpu-profiling-toolkit content in `vtune-vllm` (section 2a).
