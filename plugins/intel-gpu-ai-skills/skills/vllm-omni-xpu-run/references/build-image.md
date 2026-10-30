# Building the vLLM-Omni XPU image

Upstream ships **no** prebuilt vLLM-Omni image or wheel for Intel XPU. Every
XPU deployment layers vLLM-Omni on the published vLLM XPU image using
`docker/Dockerfile.xpu` from the vllm-omni repository. The "Pre-built wheels",
"Build wheel from source" and "Pre-built images" Intel XPU tabs of the
upstream install page are empty; only "Build your own docker image" has XPU
content.

Source: <https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/gpu/>
(Intel XPU tab). Re-read it live — don't trust a cached copy of this page.

## What the Dockerfile does

Read from `docker/Dockerfile.xpu`:

| Step | Effect |
|---|---|
| `ARG VLLM_BASE_IMAGE=vllm/vllm-openai-xpu`, `ARG VLLM_VERSION=...` | Base = `vllm/vllm-openai-xpu:${VLLM_VERSION}`; `VLLM_BASE` overrides the whole reference |
| `FROM ${VLLM_BASE} AS vllm-omni` | Inherits torch-xpu, oneCCL, the pinned compute-runtime/UMD stack, and vLLM |
| `apt-get install espeak-ng ffmpeg git jq` | Audio/video I/O for TTS and video models |
| `COPY . .` to `/workspace/vllm-omni` | Source tree, including `vllm_omni/deploy/*.yaml`, stays in the image |
| `ENV VLLM_OMNI_TARGET_DEVICE=xpu`; `uv pip install --no-build-isolation ".[dev]"` | Installs vLLM-Omni for XPU |
| `uv pip check` | Fails the build if installing vLLM-Omni broke the base's dependency pins. The base's own Triton (`triton==3.8.0+xpu` on a v0.32.0 base) is kept; older checkouts instead replaced it with `triton-xpu` |
| Final stage `vllm-omni-openai` | `ENV VLLM_WORKER_MULTIPROC_METHOD=spawn`, `ENTRYPOINT ["vllm", "serve", "--omni"]` |

`docker build` without `--target` produces the last stage, so the image
starts the API server. The intermediate `vllm-omni` stage (`--target
vllm-omni`) has an empty entrypoint and `CMD ["/bin/bash"]`. Use it for tests
or offline scripts.

## Two version counters: read the image name, not the number

The install page carries two different `vX.Y.Z` series at once, and the
**image name** tells you which project's counter you are reading:

| String on the install page | Which project's version | Role |
|---|---|---|
| `vllm/vllm-omni:v0.30.0` (CUDA prebuilt example) | **vLLM-Omni's own** release | the omni layer |
| `vllm/vllm-openai:v0.32.0` (CUDA build base) | **vLLM core engine** release | the base image |
| `vllm/vllm-openai-xpu:v0.32.0` (XPU build base) | **vLLM core engine** release | the base image |

So the `v0.32.0` in both the CUDA and the XPU build instructions is the
**vLLM core engine** version of the base image being layered onto — it is
the same number in both because `main` is rebased onto vLLM core 0.32.0.
It is *not* a vLLM-Omni release number. The one place the page names
vLLM-Omni's own version, it says `v0.30.0`.

The release tags confirm the split: vLLM-Omni's newest stable release is
`v0.30.0`, and its `v0.31.0rc1` pre-release still pins a `v0.30.0` core
base, while `vllm/vllm-openai` publishes a stable `v0.32.0`. These move
with every rebase, so verify both before citing either:

```sh
# vLLM-Omni's own releases
curl -s "https://api.github.com/repos/vllm-project/vllm-omni/releases?per_page=5" |
    grep -E '"(tag_name|prerelease)"'
# vLLM core base images available for XPU
curl -s "https://hub.docker.com/v2/repositories/vllm/vllm-openai-xpu/tags?page_size=50" |
    grep -o '"name":"v[^"]*"'
```

## Which vLLM base version to use

**Take it from the checkout's own `ARG VLLM_VERSION=` in
`docker/Dockerfile.xpu` — don't hardcode a number and don't infer one
from a ref name.** That is exactly what upstream's guide does: it clones
vllm-omni unqualified (so, the default branch) and sets `VLLM_VERSION` to
that Dockerfile's own default.

```sh
sed -n 's/^ARG VLLM_VERSION=//p' docker/Dockerfile.xpu | head -1
```

Two reasons not to write the number down anywhere:

- The default branch's pin changes whenever upstream rebases onto a newer
  vLLM core release, with no notice.
- A ref's own name doesn't predict it. On the immutable release tags,
  `v0.28.0` pins `v0.28.0`, but `v0.29.0rc1` pins `v0.28.0` — that
  pre-release was cut before its own base bump landed.

To build a frozen image instead of the default branch, clone with
`--branch <tag>` — the `VLLM_VERSION` read above still applies, and still
comes from that checkout rather than from the tag's name.

Refs older than the layered-Dockerfile cutover build vLLM from source on
`intel/deep-learning-essentials` and ignore the base-image mechanism
entirely; this skill does not cover them. Check what the Dockerfile style
is before building an old ref:

```sh
curl -sL "https://raw.githubusercontent.com/vllm-project/vllm-omni/<ref>/docker/Dockerfile.xpu" | head -5
# Layered style starts: ARG VLLM_BASE_IMAGE=vllm/vllm-openai-xpu
```

Never point an XPU build at `vllm/vllm-openai` (that's the CUDA image).
Never pull `vllm/vllm-omni` either: it's a CUDA-only prebuilt image.

## Commands

Upstream's Intel XPU build, which is what `SKILL.md` step 2 runs:

```sh
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
export VLLM_VERSION=$(sed -n 's/^ARG VLLM_VERSION=//p' docker/Dockerfile.xpu | head -1)
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g \
    --build-arg VLLM_VERSION="$VLLM_VERSION" .
```

The image is named `vllm-omni-xpu`, as in the upstream guide. A bare name
is a moving target: rebuilding replaces it, and `docker run` reuses
whatever is tagged locally without checking upstream. So rebuild when you
want current code, and confirm what you actually have with the version
check under "Verify the build" below. To keep several builds side by side,
add your own suffix (`--image vllm-omni-xpu:<something>`).

### Base tag not published yet

When targeting an unreleased vLLM, build the base from vLLM's own
`docker/Dockerfile.xpu` first:

```sh
git clone --depth 1 --branch "$VLLM_VERSION" https://github.com/vllm-project/vllm ../vllm-src
DOCKER_BUILDKIT=1 docker build -t vllm-openai-xpu:local --target vllm-openai \
    -f ../vllm-src/docker/Dockerfile.xpu ../vllm-src
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.xpu -t vllm-omni-xpu --shm-size=4g \
    --build-arg VLLM_BASE=vllm-openai-xpu:local .
```

### Default branch does not build

The default branch can be broken for XPU between releases, and upstream's
XPU CI then fails the same way. One shape seen: the `uv pip check` at the
end of the install step reports incompatibilities in packages the base
image already ships (`nixl`, `grpcio`, `opentelemetry-sdk`, `decord`), so it fails
before vLLM-Omni is involved. Confirm that it is the base, not this host,
by listing the base's own conflicts:

```sh
docker run --rm --entrypoint /opt/venv/bin/python vllm/vllm-openai-xpu:"$VLLM_VERSION" -m pip check
```

Every base reports some conflicts, including ones whose builds work, so a
non-empty list proves nothing by itself. What matters is overlap: if the
packages named in the failed build appear in the base's list too, the
failure is inherited upstream breakage. Tell the user rather than editing
upstream's Dockerfile. Search the vllm-omni
issues for the failing step, then offer to build the newest release tag
instead (`git clone --branch <tag> …`), which also re-pins
`VLLM_VERSION`:

```sh
curl -s "https://api.github.com/repos/vllm-project/vllm-omni/releases?per_page=5" |
    grep -E '"(tag_name|prerelease)"'
```

### Behind a proxy

`--build-arg http_proxy --build-arg https_proxy --build-arg no_proxy` (no
`=value`) forwards the host's values into `apt-get` and `uv`. Unset variables
are skipped. Include them whenever the host is behind a proxy.

## Size and time

Order-of-magnitude numbers from one build, to size disk — not a guaranteed
figure:

| Build | Base (compressed / on disk) | Omni image on disk |
|---|---|---|
| Base image already pulled | ~4 GB / ~17 GB | ~21 GB |
| Base image pulled during the build | ~5 GB / ~20 GB | ~27 GB |

The omni layers share the base's layers, so the extra disk is roughly
image minus base. Check free space first: `df -h /var/lib/docker`.

## Verify the build

```sh
docker run --rm --entrypoint python3 vllm-omni-xpu -c \
    "import importlib.metadata as m; print('vllm', m.version('vllm')); print('vllm-omni', m.version('vllm-omni'))"
# both report an +xpu build; vllm should match the VLLM_VERSION that was built

docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --group-add "$(getent group render | cut -d: -f3)" \
    --entrypoint python3 vllm-omni-xpu -c \
    "import torch; print(torch.xpu.device_count(), torch.xpu.get_device_name(0))"
# device count should match the number of cards exposed via --device /dev/dri
```

The base image also ships `xpu-smi` (`/usr/bin/xpu-smi`), so you can run
`xpu-smi discovery` inside the container as well. A device count of 0 means
the container lacks GPU access, not that the build failed. Recheck
`--device /dev/dri` and the render group before rebuilding.
