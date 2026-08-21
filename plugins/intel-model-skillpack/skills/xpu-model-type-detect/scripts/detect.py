#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detect the Hugging Face model type for a given model id.

Picks the right AutoModel class and input kwargs before the agent
loads the model on XPU, preventing misrouting failures from wasting time on 
unnecessary download + load time. Stdlib only; no transformers import.

Usage:
    python3 detect.py --model <hf-id> [--json] [--hf-token TOKEN]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any


HF_HUB = "https://huggingface.co"


# Order matters: the first match wins, so specific architecture
# patterns must come BEFORE generic suffix patterns. 
_ARCH_RULES: list[tuple[re.Pattern[str], str, str, list[str], str]] = [
    # (pattern on architectures[0], detected type, loader class,
    #  input kwargs, rationale)

    # --- Specific architecture names first ---
    (re.compile(r"WhisperForConditionalGeneration"), "audio_seq2seq",
     "transformers.AutoModelForSpeechSeq2Seq",
     ["input_features", "decoder_input_ids"],
     "Whisper family"),
    (re.compile(
        r"LlavaForConditionalGeneration|Qwen2VLForConditionalGeneration|"
        r"Qwen2_5_VLForConditionalGeneration|"
        r"Gemma3ForConditionalGeneration"),
     "multimodal_vl",
     "transformers.AutoModelForVision2Seq",
     ["pixel_values", "input_ids"],
     "multimodal vision-to-sequence family"),
    (re.compile(r"CLIPModel$"), "vision_language",
     "transformers.CLIPModel",
     ["pixel_values", "input_ids"],
     "CLIP vision-language"),
    (re.compile(r"SiglipModel$"), "vision_language",
     "transformers.SiglipModel",
     ["pixel_values", "input_ids"],
     "SigLIP vision-language"),
    (re.compile(r"Wav2Vec2Model$|HubertModel$|WavLMModel$"),
     "audio_encoder",
     "transformers.AutoModel",
     ["input_values"],
     "audio encoder"),
    (re.compile(r"ForScore$|RewardModel"), "reward_model",
     "transformers.AutoModelForSequenceClassification",
     [],
     "reward model — outputs a scalar score, no conversational smoke "
     "test"),
    (re.compile(r"ForTimeSeriesForecasting$|PatchTST|Chronos"),
     "time_series",
     "transformers.AutoModel",
     ["past_values", "past_time_features"],
     "time-series forecasting"),
    (re.compile(
        r"BertModel$|RobertaModel$|DebertaModel$|XLMRobertaModel$|"
        r"ModernBertModel$"),
     "text_encoder",
     "transformers.AutoModel",
     ["input_ids", "attention_mask"],
     "encoder-only; pool last_hidden_state"),
    (re.compile(r"ViTModel$|Swin|ConvNext"), "vision_classification",
     "transformers.AutoModelForImageClassification",
     ["pixel_values"],
     "vision backbone"),

    # --- Generic suffix patterns last ---
    (re.compile(r"ForCausalLM$|LMHeadModel$"), "text_generation",
     "transformers.AutoModelForCausalLM",
     ["input_ids", "attention_mask"],
     "architecture ends in ForCausalLM / LMHeadModel"),
    (re.compile(r"ForConditionalGeneration$"), "seq2seq",
     "transformers.AutoModelForSeq2SeqLM",
     ["input_ids", "decoder_input_ids"],
     "architecture ends in ForConditionalGeneration"),
    (re.compile(r"Seq2SeqLM$"), "seq2seq",
     "transformers.AutoModelForSeq2SeqLM",
     ["input_ids", "decoder_input_ids"],
     "architecture ends in Seq2SeqLM"),
    (re.compile(r"ForMaskedLM$"), "masked_lm",
     "transformers.AutoModelForMaskedLM",
     ["input_ids"],
     "architecture ends in ForMaskedLM"),
    (re.compile(r"ForImageClassification$"), "vision_classification",
     "transformers.AutoModelForImageClassification",
     ["pixel_values"],
     "architecture ends in ForImageClassification"),
]


_NAME_RULES: list[tuple[re.Pattern[str], str, str, list[str], str]] = [
    (re.compile(r"(?i)-reranker"), "unknown",
     None, [],
     "cross-encoder reranker; use AutoModelForSequenceClassification — "
     "read the model card"),
    (re.compile(r"(?i)(-rm|-reward|/reward-)"), "reward_model",
     "transformers.AutoModelForSequenceClassification",
     [],
     "model_id pattern '-rm' or '-reward'"),
    (re.compile(r"(?i)(-embed|/bge-|/gte-|/e5-|/jina-embed)"),
     "text_encoder",
     "transformers.AutoModel",
     ["input_ids", "attention_mask"],
     "model_id matches known embedding-model naming"),
    (re.compile(r"(?i)-mlx($|-)"), "incompatible",
     None, [],
     "-mlx checkpoint targets Apple Silicon; not loadable on XPU"),
    (re.compile(r"(?i)-gguf($|-)"), "incompatible",
     None, [],
     "-gguf checkpoint is for llama.cpp, not HF safetensors path"),
]


# ----- HF Hub fetches (stdlib) -----

def _http_get(url: str, token: str | None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "xpu-detect"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    # Bandit B310 suppression justification: callers build url from the HF_HUB https literal (line 23);
    # scheme and host are not reachable from any parameter.
    with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
        return resp.read()


def _fetch_config(model_id: str, token: str | None) -> dict[str, Any] | None:
    url = f"{HF_HUB}/{model_id}/resolve/main/config.json"
    try:
        return json.loads(_http_get(url, token))
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None


def _fetch_model_index(model_id: str, token: str | None) -> dict | None:
    url = f"{HF_HUB}/{model_id}/resolve/main/model_index.json"
    try:
        return json.loads(_http_get(url, token))
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None


def _fetch_pipeline_tag(model_id: str, token: str | None) -> str | None:
    url = f"{HF_HUB}/api/models/{model_id}"
    try:
        data = json.loads(_http_get(url, token))
        return data.get("pipeline_tag")
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
        return None


def _has_sentence_transformers_marker(
    model_id: str, token: str | None,
) -> bool:
    """Detect a sentence-transformers repo via its canonical marker file.
    """
    for fname in ("modules.json", "config_sentence_transformers.json"):
        url = f"{HF_HUB}/{model_id}/resolve/main/{fname}"
        try:
            req = urllib.request.Request(url, method="HEAD")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            # Bandit B310 suppression justification: url built from the HF_HUB https literal (line 23).
            with urllib.request.urlopen(req, timeout=10):  # nosec B310
                return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
    return False


# ----- Core detection -----

_DIFFUSION_NAME_HINT = re.compile(
    r"(?i)(stable[-_]?diffusion|/sd-|/sdxl|-flux($|-)|/flux-|"
    r"-diffusion($|-)|/wan-|hunyuan-video|cogvideo)"
)


def detect(model_id: str, token: str | None = None) -> dict[str, Any]:
    signals: dict[str, Any] = {
        "name_pattern": None,
        "config_architectures": None,
        "pipeline_tag": None,
    }

    # Stage 1: name pattern.
    for pat, kind, loader, inputs, rationale in _NAME_RULES:
        if pat.search(model_id):
            signals["name_pattern"] = pat.pattern
            return _result(model_id, kind, loader, inputs, rationale,
                           "name_pattern", signals)

    # Stage 2: config.architectures[0]. Try config.json first (the
    # common path), then fall back to model_index.json only when
    # config is missing or the model id looks diffusion-y.
    cfg = _fetch_config(model_id, token)
    if cfg is None:
        # No config.json -> could be diffusion (model_index.json at
        # repo root) or a missing/gated repo.
        mindex = _fetch_model_index(model_id, token)
        if mindex is not None:
            return _result(
                model_id, "diffusion", "diffusers.DiffusionPipeline",
                ["prompt"],
                "model_index.json at repo root (diffusion pipeline)",
                "model_index", signals,
            )
        return _result(
            model_id, "unknown", None, [],
            "could not fetch config.json or model_index.json; repo "
            "missing, private, or gated without HF_TOKEN",
            "none", signals,
        )

    # config.json present but name looks diffusion-y -> still check for
    # a top-level model_index.json (some pipelines ship both).
    if _DIFFUSION_NAME_HINT.search(model_id):
        mindex = _fetch_model_index(model_id, token)
        if mindex is not None:
            return _result(
                model_id, "diffusion", "diffusers.DiffusionPipeline",
                ["prompt"],
                "model_index.json at repo root (diffusion pipeline)",
                "model_index", signals,
            )

    archs = cfg.get("architectures") or []
    signals["config_architectures"] = archs

    # Sentence-Transformers override: encoder arch + ST marker file.
    if archs and _has_sentence_transformers_marker(model_id, token):
        return _result(
            model_id, "text_encoder", "transformers.AutoModel",
            ["input_ids", "attention_mask"],
            "sentence-transformers marker present (modules.json); "
            "use AutoModel + mean pool",
            "sentence_transformers_marker", signals,
        )

    for pat, kind, loader, inputs, rationale in _ARCH_RULES:
        for arch in archs:
            if pat.search(arch):
                return _result(
                    model_id, kind, loader, inputs,
                    f"architectures[0]={arch}; {rationale}",
                    "architectures[0]", signals,
                )

    # Stage 3: pipeline_tag.
    tag = _fetch_pipeline_tag(model_id, token)
    signals["pipeline_tag"] = tag
    mapping = {
        "text-generation": ("text_generation",
                            "transformers.AutoModelForCausalLM",
                            ["input_ids", "attention_mask"]),
        "text2text-generation": ("seq2seq",
                                 "transformers.AutoModelForSeq2SeqLM",
                                 ["input_ids", "decoder_input_ids"]),
        "feature-extraction": ("text_encoder",
                               "transformers.AutoModel",
                               ["input_ids", "attention_mask"]),
        "sentence-similarity": ("text_encoder",
                                "transformers.AutoModel",
                                ["input_ids", "attention_mask"]),
        "image-classification": (
            "vision_classification",
            "transformers.AutoModelForImageClassification",
            ["pixel_values"]),
        "zero-shot-image-classification": (
            "vision_language", "transformers.CLIPModel",
            ["pixel_values", "input_ids"]),
        "automatic-speech-recognition": (
            "audio_seq2seq",
            "transformers.AutoModelForSpeechSeq2Seq",
            ["input_features", "decoder_input_ids"]),
        "fill-mask": ("masked_lm",
                      "transformers.AutoModelForMaskedLM",
                      ["input_ids"]),
        "image-to-text": (
            "multimodal_vl",
            "transformers.AutoModelForVision2Seq",
            ["pixel_values", "input_ids"]),
        "visual-question-answering": (
            "multimodal_vl",
            "transformers.AutoModelForVision2Seq",
            ["pixel_values", "input_ids"]),
        "image-text-to-text": (
            "multimodal_vl",
            "transformers.AutoModelForVision2Seq",
            ["pixel_values", "input_ids"]),
        "object-detection": (
            "vision_classification",
            "transformers.AutoModelForObjectDetection",
            ["pixel_values"]),
        "text-to-image": (
            "diffusion", "diffusers.DiffusionPipeline",
            ["prompt"]),
    }
    hit = mapping.get(tag or "")
    if hit:
        kind, loader, inputs = hit
        return _result(
            model_id, kind, loader, inputs,
            f"pipeline_tag={tag}",
            "pipeline_tag", signals,
        )

    return _result(
        model_id, "unknown", None, [],
        "no match from name pattern, architectures, or pipeline_tag; "
        "read the model card's first usage snippet",
        "none", signals,
    )


def _result(model_id, kind, loader, inputs, rationale, confidence,
            signals):
    out = {
        "model_id": model_id,
        "detected": kind,
        "loader": loader,
        "inputs": list(inputs),
        "confidence": confidence,
        "rationale": rationale,
        "signals": signals,
    }
    if kind == "vision_language":
        out["processor"] = "transformers.AutoProcessor"
    elif kind in ("vision_classification",):
        out["processor"] = "transformers.AutoImageProcessor"
    elif kind in ("audio_encoder", "audio_seq2seq"):
        out["processor"] = "transformers.AutoFeatureExtractor"
    elif kind == "multimodal_vl":
        out["processor"] = "transformers.AutoProcessor"
    return out


def _format_text(r: dict[str, Any]) -> str:
    lines = [
        f"model_id:    {r['model_id']}",
        f"detected:    {r['detected']}",
    ]
    if r.get("loader"):
        lines.append(f"loader:      {r['loader']}")
    if r.get("processor"):
        lines.append(f"processor:   {r['processor']}")
    if r["inputs"]:
        lines.append(f"inputs:      {', '.join(r['inputs'])}")
    lines.append(f"confidence:  {r['confidence']}")
    lines.append(f"rationale:   {r['rationale']}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Detect a Hugging Face model's type for XPU loading."
    )
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--json", action="store_true",
                   help="Emit one JSON object")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Bearer token for gated repos "
                        "(or set HF_TOKEN)")
    args = p.parse_args()

    r = detect(args.model, token=args.hf_token)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(_format_text(r))
    # Reward model is technically supported but not useful for smoke
    # tests; incompatible is fatal. Non-zero exit on both lets agents
    # branch on it.
    if r["detected"] in ("incompatible", "unknown"):
        return 2
    if r["detected"] == "reward_model":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
