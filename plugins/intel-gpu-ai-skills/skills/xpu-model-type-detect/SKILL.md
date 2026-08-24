---
name: xpu-model-type-detect
description: Before loading a Hugging Face model on Intel XPU, detect its actual type (text generation, text encoder, seq2seq, vision classification, vision-language, audio encoder, audio seq2seq, multimodal VL, diffusion, time-series, reward model, masked LM) so the agent picks the right `AutoModel` class and input kwargs. Prevents "got unexpected keyword argument 'pixel_values'" and "empty logits" errors from mis-routing. Use before `torch-xpu-run` or `vllm-xpu-run` when the user gives a model id the agent hasn't seen before, or when a smoke test fails with a wrong-input signature.
---

# xpu-model-type-detect

Pick the right loader class before you load. A wrong loader class
produces an opaque failure *after* the model is already on XPU — 20+
seconds into a load you didn't need to make.

## Reporting the verdict

Run `scripts/detect.py` against the model id rather than answering
its type from memory — even for a familiar model, the detector's
verdict is the citable result. Then paste its output block verbatim
in a fenced code block, keeping the `detected:`, `loader:`, `inputs:`,
and `confidence:` lines as-is rather than reformatting them into a
table — those literal labels are what the user and downstream tools
key on. Lead with the confirmed type, e.g. "detection confirmed:
`multimodal_vl`", to show the verdict came from the run.

Detection is read-only, so there's nothing to health-check
afterward — confirming the verdict block is the result.

## Quickstart

```sh
python3 scripts/detect.py --model openai/clip-vit-base-patch32
```

```
model_id:    openai/clip-vit-base-patch32
detected:    vision_language
loader:      transformers.CLIPModel
processor:   transformers.AutoProcessor
inputs:      pixel_values, input_ids
rationale:   architectures[0]=CLIPModel; text + vision towers detected
```

Stdlib only. `HF_TOKEN` needed for gated repos.

With `--json` the output is a single JSON object suitable for piping
into another tool.

## 3-stage detection

The script tries three signals in order; first hit wins.

1. **Name pattern on `model_id`** — cheapest. Currently catches reward
   models (`*-rm`, `*-reward`), embedding repos (`*-embed*`, `bge-`,
   `gte-`, `e5-`, `jina-embed*`), and incompatible checkpoints
   (`*-mlx*`, `*-gguf*`).
2. **`config.architectures[0]`** — authoritative when present. Pulled
   directly from `config.json` on the Hub (no weight download). This
   is where most non-name-matched models classify.
3. **HF Hub `pipeline_tag`** — last resort; one HTTP call to
   `/api/models/<id>`.

If all three miss, the script prints `unknown` and suggests reading the
model card's first usage snippet.

## Types and their loaders

| Detected type | Loader class | Input kwargs | Typical hit signal |
|---|---|---|---|
| `text_generation` | `AutoModelForCausalLM` | `input_ids`, `attention_mask` | `*CausalLM`, `*LMHeadModel` (decoder-only LLM architectures) |
| `seq2seq` | `AutoModelForSeq2SeqLM` | `input_ids`, `decoder_input_ids` | `*Seq2SeqLM`, `*ForConditionalGeneration` (T5, BART, MBart) |
| `masked_lm` | `AutoModelForMaskedLM` | `input_ids` | `*ForMaskedLM` |
| `text_encoder` | `AutoModel` + mean-pool last_hidden_state | `input_ids`, `attention_mask` | `BertModel`, `RobertaModel`, `DebertaModel`, `XLMRobertaModel`, name `*-embed*`, `*-bge*` |
| `vision_language` (CLIP family) | `CLIPModel` / `SigLIPModel` + `AutoProcessor` | `pixel_values`, `input_ids` | `CLIPModel`, `SiglipModel`, both vision and text towers |
| `vision_classification` | `AutoModelForImageClassification` + `AutoImageProcessor` | `pixel_values` | `*ForImageClassification`, `ViTModel`, `Swin*`, `ConvNext*` |
| `audio_encoder` | `AutoModel` + `AutoFeatureExtractor` | `input_values` | `Wav2Vec2Model`, `HubertModel`, `WavLMModel` |
| `audio_seq2seq` | `AutoModelForSpeechSeq2Seq` | `input_features`, `decoder_input_ids` | `WhisperForConditionalGeneration` |
| `multimodal_vl` | `AutoModelForVision2Seq` + `AutoProcessor` | `pixel_values`, `input_ids` | `LlavaForConditionalGeneration`, `Qwen2VLForConditionalGeneration`, `Gemma3ForConditionalGeneration` |
| `reward_model` | fast-fail; no smoke test | n/a | `*ForScore`, name `*-rm`, `*-reward` |
| `diffusion` | `DiffusionPipeline` (diffusers) | `prompt` | `model_index.json` at repo root |
| `time_series` | `AutoModel` | `past_values`, `past_time_features` | Chronos, PatchTST, `*ForTimeSeriesForecasting` |

## Gotchas

- **Sentence-Transformers** repos often have `architectures[0] =
  BertModel` but the intended usage is `AutoModel` + mean pooling — not
  `AutoModelForSequenceClassification`. The detector catches this via
  the canonical `modules.json` marker file (every ST repo has it;
  `config_sentence_transformers.json` is checked as a secondary marker)
  and returns `text_encoder`.
- **CLIP variants with only `vision_model`** (e.g. pure ViT CLIP
  vision-encoder releases) classify as `vision_classification`, not
  `vision_language`.
- **Reward models** look like classifiers but the output is a scalar
  score, not class logits. Agents should skip smoke tests and tell the
  user reward models aren't meant for conversational use.
- **Time-series** models (Chronos, PatchTST) don't take `input_ids` at
  all — don't try to tokenise text for them.
- **Diffusion repos** have no top-level `config.json`; the detector
  looks for `model_index.json`. Route these to `torch-xpu-bench` for
  sizing (VRAM depends on resolution × steps × scheduler).

## When auto-detect is wrong

The detector is conservative. If it returns `unknown`, or the user
confirms the suggested loader is wrong:

1. Open the model card (README.md in the HF repo); read the first
   usage snippet.
2. If the snippet imports a class not in the table above, search
   Transformers docs for that class's `AutoModel*` equivalent.

## End-to-end flow (beginner-friendly)

Given `Qwen/Qwen2.5-7B-Instruct`:

```sh
python3 scripts/detect.py --model Qwen/Qwen2.5-7B-Instruct --json
# {"detected": "text_generation", "loader": "transformers.AutoModelForCausalLM",
#  "inputs": ["input_ids", "attention_mask"], ...}
```

The agent then uses `torch-xpu-run`'s preflight helper with the
correct loader class and the correct first `generate()` call.

Given `openai/clip-vit-base-patch32`:

```sh
python3 scripts/detect.py --model openai/clip-vit-base-patch32
# detected: vision_language  ->  CLIPModel + AutoProcessor + pixel_values + input_ids
```

The agent supplies an image alongside the prompt instead of calling
`model.generate(input_ids=...)`.

## Output schema (with `--json`)

```json
{
  "model_id": "openai/clip-vit-base-patch32",
  "detected": "vision_language",
  "loader": "transformers.CLIPModel",
  "processor": "transformers.AutoProcessor",
  "inputs": ["pixel_values", "input_ids"],
  "confidence": "architectures[0]",
  "rationale": "architectures[0]=CLIPModel; text + vision towers detected",
  "signals": {
    "name_pattern": null,
    "config_architectures": ["CLIPModel"],
    "pipeline_tag": "zero-shot-image-classification"
  }
}
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Detection succeeded; `detected` is a usable type |
| `2` | `unknown` or `incompatible` — detection could not resolve a loader; read `detected` + `rationale` to distinguish |
| `3` | `reward_model` — detection succeeded but the model is not suitable for conversational smoke tests |

Codes `2` and `3` are expected outcomes, not crashes. Agents should read the `detected` field rather than treating any non-zero exit as a failure.

## What this skill does NOT cover

- Loading the model (see `torch-xpu-run` / `vllm-xpu-run`).
- VRAM sizing (see `model-can-it-fit`).

## References

- HF Transformers `AutoModel*` family: <https://huggingface.co/docs/transformers/model_doc/auto>
- HF Hub `model_info` API: <https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info>
