# Troubleshooting vLLM-Omni on Intel XPU

Each stage logs with a `(StageEngineCoreProc_stage<N>_replica0 pid=…)`
prefix. Find the **first** stage-level error. The APIServer's
`Engine core initialization failed … Failed core proc(s): {}` is only the
echo of it.

```sh
docker logs <name> 2>&1 | grep -E "StageEngineCoreProc_stage[0-9].*(ERROR|Error|Available KV)" | head -20
```

## Known issue: Qwen2.5-Omni speech is garbled without its system prompt

Observed on 2× Arc Pro B70 with upstream builds of vllm-omni `v0.30.0`
(vLLM 0.30.0, torch 2.13 XPU) and of `main` (vLLM 0.32.0, torch 2.14 XPU),
in the 2-card default and the single-card layout, for a chat request with
**no system message**:

- Text output is correct ("The capital of France is Paris.").
- The returned WAV is valid (16-bit PCM, 24 kHz) with plausible length.
- Transcribing it back gives unrelated words or Chinese ("我喝。").
  `smoke_test.py --mode audio --roundtrip --system ''` fails with word
  recall 0.00.

Cause: the talker is conditioned on the thinker's hidden states for the
whole prompt, and it was trained with the system prompt from the model
card ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba
Group, capable of perceiving auditory and visual inputs, as well as
generating text and speech."). The chat template's default system prompt
puts that conditioning out of distribution, so the codec tokens come out
wrong while the text is unaffected. Bisected against the Hugging Face
implementation: the thinker's hidden states, the talker's assembled inputs
and the code2wav stage on XPU all match the reference, the pure Hugging
Face pipeline on CPU garbles the same prompt-less request, and the same
XPU server answers intelligibly as soon as the system message is sent. Not XPU-specific:
upstream's e2e tests all send that prompt.

Fix: put that system message in every speech request. `smoke_test.py`
does by default (`--system` overrides it); verified on `main`, the round
trip hears every word (word recall 1.00). Controls: espeak-ng speech and
Qwen3-TTS-12Hz-1.7B-CustomVoice output transcribe correctly on the same
server. On `v0.30.0`, long answers were also cut short (a 75-word answer
gave 5.6 s of audio, 281 talker tokens); upstream PR #9371 fixed that on
`main`.

What to tell the user:

- Send the model card's system prompt with every Qwen2.5-Omni speech
  request; a client that sets its own system prompt will get garbled
  speech. For speech under an arbitrary system prompt, use a dedicated
  TTS model (Qwen3-TTS verified intelligible on XPU) on `/v1/audio/speech`.
- Re-check with `--roundtrip` after every image upgrade.

Qwen2.5-Omni-3B and Qwen3-Omni were not tested.

## Startup failures

| Symptom (first stage error) | Cause | Fix |
|---|---|---|
| `Available KV cache memory: -3.87 GiB` then `ValueError: No available memory for the cache blocks` | Stage budget (`gpu_memory_utilization × card`) < weights + profiling peak. Seen on the single-card layout at the default 32768 `max_num_batched_tokens` | Raise that stage's fraction, or cap `max_num_batched_tokens` / `max_model_len` (8192) for it in `--stage-overrides` |
| `To serve at least one request with the model's max seq len (8192), (0.38 GiB KV cache is needed …` | Stage has KV, just not enough for one full sequence | Move ~0.02 of a card to that stage from another, or lower its `max_model_len` |
| Stage fails at init with free memory < requested | Stages on one card sum > 1.0, or another process holds memory | `xpu-smi ps` / `xpu-smi dump -d <id> -m 18 -n 1`; keep the sum ≤ 0.93 on a 32 GB card with ~1 GB baseline use |
| Device-count mismatch error naming a stage | `devices` count ≠ TP × DP × PP, or a top-level `--tensor-parallel-size` was broadcast to every stage (upstream #5003) | Set TP and `devices` together per stage in `--stage-overrides` |
| Stage asks for logical device `1` on a 1-card host | Default Qwen2.5-Omni XPU layout expects 2 cards | Single-card overrides (`emit_launch.sh --devices 0` applies them) |
| `RuntimeError: Cannot find any XPU devices` / `torch.xpu.device_count() == 0` | Container lacks GPU access | `--device /dev/dri`, `--group-add render`; check host `xpu-smi discovery` |
| `oneCCL … ze_fd_manager.cpp … opendir failed` | `/dev/dri/by-path` not mounted | `-v /dev/dri/by-path:/dev/dri/by-path:ro` |
| Stage init timeout on first launch | Weights still downloading inside the 300 s per-stage default | Pre-download, or `emit_launch.sh --first-launch` (`--stage-init-timeout 1200 --init-timeout 2400`) |
| Exit code 2: `vllm: error: unrecognized arguments: vllm serve` | Extra `vllm serve` after the image name (entrypoint is already `vllm serve --omni`) | Pass only flags after the image |
| `Model architectures … not supported` | Model not wired into this vllm-omni version | Check the supported-models page for that version; rebuild from a newer tag |
| `torch._dynamo` duplicate-handler assertion | `enforce_eager: false` on Qwen2.5 AR stages with torch 2.13 XPU | Keep the YAML's `enforce_eager: true` |

## Runtime problems

| Symptom | Fix |
|---|---|
| HTTP 200 but no `message.audio` | Request `"modalities": ["text","audio"]`; check stage 2 errors in the logs |
| Audio valid but nonsense | The request has no Qwen-Omni system prompt; see the known issue above. Confirm with `--roundtrip` |
| Black or tiny PNG from a diffusion model | Retry at 512×512 with a fixed seed. If still blank, relaunch with `--enforce-eager` and read the stage warnings in `docker logs` |
| First request very slow | Triton/JIT warmup (`JIT kernel warmup` in logs); the second request is the real latency |
| `model not found` (400) | `model` must match `GET /v1/models` exactly |
| Only one card busy on a multi-card layout | `ZE_AFFINITY_MASK` lists fewer cards than the stage `devices` reference |

## Env vars

| Variable | Notes |
|---|---|
| `ZE_AFFINITY_MASK` | Physical cards the container sees; stage `devices` index into it |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Already set in the image; keep it if you override the entrypoint |
| `VLLM_XPU_USE_SAMPLER_KERNEL` | vLLM-Omni's XPU platform defaults it to `0` because the XPU sampler kernel is broken for omni; do not set `1` |
| `VLLM_LOGGING_LEVEL=DEBUG` | Verbose engine logs |
| `HF_TOKEN`, proxy vars | Pass through with `-e NAME` |

Upstream CI also passes `VLLM_OMNI_LOGGING_LEVEL`, but vllm-omni `v0.30.0`
does not read it. Use `VLLM_LOGGING_LEVEL`.

## Remote code (`trust_remote_code`)

Three facts, all read from vllm-omni `v0.30.0` source and confirmed on the
engine's own log lines:

1. 79 of the 141 bundled deploy YAMLs on `main` (71 of 110 in `v0.30.0`) set `trust_remote_code: true` per
   stage: `qwen2_5_omni` and `qwen3_omni_moe` on all 3 stages,
   `qwen3_tts` and `voxtral_tts` on both. Diffusion models such as Z-Image
   have no bundled YAML and get vLLM's default, `False`.
2. The serve CLI maps an absent `--trust-remote-code` to `None`, meaning
   "not specified", and the resolver then keeps the YAML value
   (`config_factory.with_trust_remote_code_override`). A Qwen2.5-Omni-7B
   launch with no flag logs `trust_remote_code=True` for every stage.
3. Per-stage values beat pipeline-wide ones in the merge, so a top-level
   `trust_remote_code: false` in an overlay does **not** turn it off. Only
   per-stage overrides do: `emit_launch.sh --no-remote-code` emits
   `--stage-overrides` with `"trust_remote_code": false` for each stage of
   the models it knows, and refuses for the ones it does not until you name
   the stages yourself, since a flag it cannot apply would disable nothing.
   Qwen2.5-Omni-7B (which ships no remote code) came up healthy with all
   three stages logging `False`.

Read the resolved value, not the command line:

```sh
docker logs <name> 2>&1 | grep -o 'trust_remote_code=[A-Za-z]*' | sort -u
```

For a model `emit_launch.sh` does not know, check its YAML in the image
before deciding:

```sh
docker run --rm --entrypoint grep vllm-omni-xpu -c 'trust_remote_code: true' \
    /workspace/vllm-omni/vllm_omni/deploy/<model_type>.yaml
```

## Privileges and networking

The upstream docs example uses `--privileged --net=host --shm-size 10g`. The
skill's command omits all three:

- `--device /dev/dri` plus the render group was enough for every test
  above, matching upstream's XPU CI script (no `--privileged`).
- `-p <port>:<port>` publishes only the API. Stages talk over
  in-container localhost.
- `--shm-size` has no effect with `--ipc=host`.

Add `--privileged` (`emit_launch.sh --privileged`) only after the user
accepts it, and only when Level Zero device discovery still fails with the
flags above. Use `--host-network` for the multi-process stage-based CLI
across containers or hosts.
