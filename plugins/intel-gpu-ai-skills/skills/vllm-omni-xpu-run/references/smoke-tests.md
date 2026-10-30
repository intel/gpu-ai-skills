# Endpoints and output validation

Each vLLM-Omni server hosts one model. Endpoints are OpenAI-compatible; which
ones work depends on the model's output modality. `GET /health` returns
200 once every stage is initialized. `GET /v1/models` returns the served id
to put in the `model` field.

Upstream serving docs: <https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/>

## Pick the check by model type

| Model family (examples) | Endpoint | `smoke_test.py --mode` |
|---|---|---|
| Omni chat: Qwen2.5-Omni, Qwen3-Omni, Ming-flash-omni | `POST /v1/chat/completions` | `text`, `audio` |
| Text-to-image: Qwen-Image, Z-Image, FLUX, SD3/SDXL | `POST /v1/images/generations` | `image` |
| Image edit: Qwen-Image-Edit, FLUX.1-Kontext | `POST /v1/images/edits` | — (see upstream docs) |
| TTS: Qwen3-TTS, Voxtral TTS, CosyVoice3, Fish S2 Pro | `POST /v1/audio/speech` | `speech` (`--voice` is model-specific) |
| Video: Wan2.x, LTX-2, HunyuanVideo-1.5 | `POST /v1/videos` | — (see upstream docs) |

## Running the bundled checker

```sh
python3 scripts/smoke_test.py --base-url http://127.0.0.1:8091 \
    --mode text --mode audio --roundtrip --expect paris --out ./omni-smoke --wait 900
```

- `--wait N` polls `/health` for up to N seconds first.
- `text` and `audio` send the Qwen-Omni system prompt the talker was
  trained with; `--system` overrides it (`''` sends none).
- `--roundtrip` (audio mode) sends the generated WAV back to the same omni
  server for transcription. It fails if fewer than `--min-recall` (0.5) of
  the answer's words are heard. Always use it with omni chat models: a valid
  WAV of the right length can still be noise.
- Proxies are bypassed for the local URL.
- Decoded artifacts go to `--out` (`smoke_audio.wav`, `smoke_image.png`,
  `smoke_speech.wav`) so the user can listen or look.
- Prints `SMOKE TEST PASS` and exits 0 only when every mode passes. Report
  the printed text, durations, dimensions and `heard:` transcript.

Observed on 2× Arc Pro B70, clean vllm-omni builds:

```text
# Qwen/Qwen2.5-Omni-7B, --mode text --mode audio --roundtrip   (main)
[PASS] text (…s) text: 'The capital of France is Paris. If you want to know more about Paris or France, feel free to ask me.'
[PASS] audio (…s) audio: 6.22s @ 24000 Hz ...; heard: 'The capital of France is Paris. If you want to know more about Paris or France, feel free to ask me.' (word recall 1.00)
# same server, --system '' (no system prompt)
[FAIL] audio (…s) speech is not intelligible: audio: 4.08s @ 24000 Hz ...; heard: '我要。' (word recall 0.00)
# Tongyi-MAI/Z-Image-Turbo, --mode image   (image inspected: correct red apple on a table)
[PASS] image (…s) image: 512x512, 787271 bytes
# Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, --mode speech --voice vivian --language English
[PASS] speech (…s) speech: 4.80s @ 24000 Hz   (transcribed back correctly)
```

The `--system ''` failure is the known-issue shape: without the model's
system prompt the talker babbles (`references/troubleshooting.md`). Without
`--roundtrip`, the same run reports `[PASS] audio`.

For `--mode speech` (TTS servers cannot transcribe), check intelligibility
by sending the saved WAV to any running omni chat server's
`/v1/chat/completions` as an `audio_url` input, or ask the user to listen.

## Manual curl equivalents

Text only. This skips audio *generation*, but every stage stays loaded:

```sh
curl -s http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Describe vLLM in one sentence."}],
         "modalities":["text"],"max_tokens":64}' | jq -r '.choices[0].message.content'
```

Text + audio. On v0.30.0 one choice carries both `content` and
`message.audio.data` (base64 WAV, 16-bit PCM mono 24 kHz for Qwen2.5-Omni):

```sh
curl -s http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"system","content":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."},
                      {"role":"user","content":"Say hello."}],"modalities":["text","audio"]}' \
  | jq -r '.choices[] | select(.message.audio) | .message.audio.data' | base64 -d > out.wav
file out.wav   # RIFF (little-endian) data, WAVE audio
```

Image generation:

```sh
curl -s http://127.0.0.1:8091/v1/images/generations -H 'Content-Type: application/json' \
    -d '{"prompt":"a red apple on a wooden table","size":"512x512","seed":42}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

Speech (Qwen3-TTS CustomVoice example; voices differ per model):

```sh
curl -s http://127.0.0.1:8091/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input":"Hello from an Intel GPU.","voice":"vivian","language":"English"}' -o out.wav
```

## What counts as validated

- Text: non-empty `content` that answers the prompt (use `--expect`).
- Audio and speech: valid RIFF/WAVE with plausible duration (≥ 0.3 s
  for a short sentence). A tiny file or a missing `message.audio` means the
  talker or code2wav stage failed even though text came back.
- Image: valid PNG at the requested size. A few-KB PNG at 512×512 is
  almost always a blank or NaN frame (check dtype and attention backend in
  `references/troubleshooting.md`).
- Placement: every card that hosts a stage shows multi-GB memory in
  `xpu-smi dump -d <id> -m 18 -n 1`.
