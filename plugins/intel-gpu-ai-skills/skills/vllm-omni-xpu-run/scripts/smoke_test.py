#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Output-validated smoke test for a running vLLM-Omni server.

HTTP 200 is not proof of a working omni pipeline: a broken talker or code2wav
stage can still return a text choice, and a diffusion stage can return an
all-black image. Each mode below decodes the payload and checks it.

Modes:
  text    POST /v1/chat/completions, modalities=["text"]; non-empty content
  audio   POST /v1/chat/completions, modalities=["text","audio"]; decodable WAV.
          Add --roundtrip to also transcribe the speech with the same server:
          a valid WAV can still be unintelligible noise.
          text and audio send the Qwen-Omni system prompt (--system); the
          talker is conditioned on it, and without it the speech is garbled.
  image   POST /v1/images/generations; decodable PNG with real dimensions
  speech  POST /v1/audio/speech; decodable WAV

Prints "SMOKE TEST PASS" and exits 0 only when every requested mode passes.
Standard library only, so it runs on the host without extra installs.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
import zlib
from pathlib import Path

# Local server: never route through a corporate proxy picked up from env.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(base: str, path: str, payload: dict | None, timeout: float) -> bytes:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"{path} -> HTTP {exc.code}: {body}") from exc


def wav_seconds(raw: bytes) -> tuple[float, int]:
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError(f"audio is not RIFF/WAVE (first bytes {raw[:12]!r})")
    try:
        with wave.open(io.BytesIO(raw)) as w:
            frames, rate = w.getnframes(), w.getframerate()
            # getnframes() reports what the header DECLARES. A stage that died
            # after writing the header leaves a file whose declared duration is
            # intact while the samples are gone, so the duration check would
            # pass on a header alone. Require the frame data to be present.
            expected = frames * w.getnchannels() * w.getsampwidth()
            actual = len(w.readframes(frames))
            if actual < expected:
                raise RuntimeError(
                    f"WAV truncated: header declares {expected} data bytes, "
                    f"found {actual}"
                )
            return frames / float(rate), rate
    except (wave.Error, EOFError):
        # Float WAV (format 3) is not readable by `wave`, and a file cut off
        # mid-chunk raises EOFError from its chunk reader. Size the data chunk
        # by hand, checking lengths first so a truncated payload reports a
        # clean failure instead of a struct.error traceback.
        pass
    if len(raw) < 36:
        raise RuntimeError(f"WAV truncated: {len(raw)} bytes, no complete fmt chunk")
    rate = struct.unpack("<I", raw[24:28])[0]
    block = struct.unpack("<H", raw[32:34])[0] or 1
    idx = raw.find(b"data")
    if idx < 0 or rate == 0:
        raise RuntimeError("WAV has no data chunk")
    if idx + 8 > len(raw):
        raise RuntimeError(f"WAV truncated: data chunk header cut off at byte {len(raw)}")
    size = struct.unpack("<I", raw[idx + 4:idx + 8])[0]
    available = len(raw) - idx - 8
    if size in (0, 0xFFFFFFFF):  # streaming header: size unknown
        size = available
    elif size > available:
        # Same trap as above, on the hand-parsed path.
        raise RuntimeError(
            f"WAV truncated: data chunk declares {size} bytes, found {available}"
        )
    return size / float(block * rate), rate


def png_size(raw: bytes) -> tuple[int, int]:
    """IHDR dimensions of a PNG whose image data actually decodes.

    IHDR is near the front and IEND at the end, so neither proves the pixels
    in between are readable: a stage that died mid-write, or wrote corrupt
    data under intact framing, would still report a plausible size. Walk every
    chunk, check its CRC, inflate the image data and check it holds exactly
    the rows the header promises.
    """
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"image is not PNG (first bytes {raw[:8]!r})")
    pos, ihdr, idat = 8, None, []
    while True:
        if pos + 12 > len(raw):
            raise RuntimeError(f"PNG truncated: {len(raw)} bytes, no IEND end-of-image chunk")
        length, tag = struct.unpack(">I4s", raw[pos:pos + 8])
        end = pos + 12 + length
        if end > len(raw):
            raise RuntimeError(f"PNG truncated: {len(raw)} bytes, inside a {tag!r} chunk")
        body = raw[pos + 8:end - 4]
        if zlib.crc32(tag + body) != struct.unpack(">I", raw[end - 4:end])[0]:
            raise RuntimeError(f"PNG {tag!r} chunk fails its CRC check")
        if tag == b"IHDR":
            ihdr = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos = end
    if ihdr is None or len(ihdr) != 13:
        raise RuntimeError("PNG has no valid IHDR chunk")
    w, h, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", ihdr)
    try:
        pixels = zlib.decompress(b"".join(idat))
    except zlib.error as exc:
        raise RuntimeError(f"PNG image data does not decompress: {exc}") from exc
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    # Adam7-interlaced rows are laid out per pass; servers emit plain PNGs.
    if channels and not interlace:
        expected = h * (1 + (w * channels * depth + 7) // 8)
        if len(pixels) != expected:
            raise RuntimeError(
                f"PNG image data is {len(pixels)} bytes, expected {expected} for {w}x{h}"
            )
    return w, h


def served_model(base: str, timeout: float) -> str:
    models = json.loads(request(base, "/v1/models", None, timeout))
    ids = [m.get("id") for m in models.get("data", [])]
    if not ids:
        raise RuntimeError(f"/v1/models returned no models: {models}")
    return ids[0]


# The system prompt Qwen2.5-Omni / Qwen3-Omni were trained with (their model
# cards require it for audio output, and upstream's own e2e tests send it).
# The chat template's default system prompt leaves the talker's text
# conditioning out of distribution: text stays correct, speech turns to
# babble.
OMNI_SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                      "capable of perceiving auditory and visual inputs, as well as generating text and speech.")


def chat_messages(args) -> list[dict]:
    system = [{"role": "system", "content": args.system}] if args.system else []
    return system + [{"role": "user", "content": args.prompt}]


def check_text(base, model, args, out: Path) -> str:
    body = json.loads(request(base, "/v1/chat/completions", {
        "model": model,
        "messages": chat_messages(args),
        "modalities": ["text"],
        "max_tokens": args.max_tokens,
    }, args.timeout))
    texts = [(c.get("message") or {}).get("content") or "" for c in body.get("choices", [])]
    text = next((t for t in texts if t.strip()), "")
    if not text.strip():
        raise RuntimeError(f"no text content in response: {json.dumps(body)[:500]}")
    if args.expect and args.expect.lower() not in text.lower():
        raise RuntimeError(f"text does not contain {args.expect!r}: {text[:200]!r}")
    return f"text: {text.strip()[:200]!r}"


def check_audio(base, model, args, out: Path) -> str:
    body = json.loads(request(base, "/v1/chat/completions", {
        "model": model,
        "messages": chat_messages(args),
        # ["text", "audio"] returns one merged choice carrying both content
        # and message.audio on v0.30.0; older builds split them across two
        # choices, which the scan below also handles.
        "modalities": ["text", "audio"],
        "max_tokens": args.max_tokens,
    }, args.timeout))
    audio = next((c["message"]["audio"]["data"] for c in body.get("choices", [])
                  if (c.get("message") or {}).get("audio")), None)
    if not audio:
        raise RuntimeError(f"no message.audio in any choice: {json.dumps(body)[:500]}")
    raw = base64.b64decode(audio)
    secs, rate = wav_seconds(raw)
    if secs < args.min_audio_s:
        raise RuntimeError(f"audio too short: {secs:.2f}s (min {args.min_audio_s}s)")
    path = out / "smoke_audio.wav"
    path.write_bytes(raw)
    texts = [(c.get("message") or {}).get("content") or "" for c in body.get("choices", [])]
    text = next((t for t in texts if t.strip()), "")
    detail = f"audio: {secs:.2f}s @ {rate} Hz -> {path}; text: {text.strip()[:120]!r}"
    if args.roundtrip:
        if not text.strip():
            raise RuntimeError("--roundtrip needs a text answer to compare against; none returned")
        heard = transcribe(base, model, audio, args.timeout)
        recall = word_recall(text, heard)
        detail += f"; heard: {heard.strip()[:120]!r} (word recall {recall:.2f})"
        if recall < args.min_recall:
            raise RuntimeError(f"speech is not intelligible: {detail}")
    return detail


def transcribe(base, model, audio_b64: str, timeout: float) -> str:
    """Ask the same omni server to transcribe audio it generated."""
    body = json.loads(request(base, "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + audio_b64}},
            {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript."},
        ]}],
        "modalities": ["text"],
        "max_tokens": 256,
        "temperature": 0.0,
    }, timeout))
    return ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


def word_recall(reference: str, heard: str) -> float:
    ref = set(re.findall(r"[a-z0-9']+", reference.lower()))
    hyp = set(re.findall(r"[a-z0-9']+", heard.lower()))
    return len(ref & hyp) / len(ref) if ref else 0.0


def check_image(base, model, args, out: Path) -> str:
    body = json.loads(request(base, "/v1/images/generations", {
        "model": model,
        "prompt": args.image_prompt,
        "size": args.size,
        "n": 1,
        "seed": 42,
    }, args.timeout))
    data = (body.get("data") or [{}])[0].get("b64_json")
    if not data:
        raise RuntimeError(f"no data[0].b64_json: {json.dumps(body)[:500]}")
    raw = base64.b64decode(data)
    w, h = png_size(raw)
    requested = re.fullmatch(r"(\d+)x(\d+)", args.size)
    if requested and (w, h) != tuple(map(int, requested.groups())):
        raise RuntimeError(f"image dimensions {w}x{h} do not match requested {args.size}")
    if w < 64 or h < 64:
        raise RuntimeError(f"image too small: {w}x{h}")
    # A collapsed diffusion pass tends to emit a flat image that compresses
    # to almost nothing; real content at >=256px is rarely under 5 KB.
    if len(raw) < 5000 and w * h >= 256 * 256:
        raise RuntimeError(f"PNG is only {len(raw)} bytes at {w}x{h}; likely a blank image")
    path = out / "smoke_image.png"
    path.write_bytes(raw)
    return f"image: {w}x{h}, {len(raw)} bytes -> {path}"


def check_speech(base, model, args, out: Path) -> str:
    payload = {"model": model, "input": args.speech_text, "response_format": "wav"}
    if args.voice:
        payload["voice"] = args.voice
    if args.language:
        payload["language"] = args.language
    raw = request(base, "/v1/audio/speech", payload, args.timeout)
    secs, rate = wav_seconds(raw)
    if secs < args.min_audio_s:
        raise RuntimeError(f"speech too short: {secs:.2f}s (min {args.min_audio_s}s)")
    path = out / "smoke_speech.wav"
    path.write_bytes(raw)
    return f"speech: {secs:.2f}s @ {rate} Hz -> {path}"


CHECKS = {"text": check_text, "audio": check_audio, "image": check_image, "speech": check_speech}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8091")
    p.add_argument("--mode", action="append", choices=sorted(CHECKS), required=True,
                   help="repeat to run several checks, e.g. --mode text --mode audio")
    p.add_argument("--model", help="model id; defaults to the first id in /v1/models")
    p.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    p.add_argument("--system", default=OMNI_SYSTEM_PROMPT,
                   help="system prompt for text/audio modes; pass '' to send none (speech then garbles)")
    p.add_argument("--expect", help="case-insensitive substring the text answer must contain")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--image-prompt", default="A red apple on a wooden table, studio photo")
    p.add_argument("--size", default="512x512")
    p.add_argument("--speech-text", default="Hello from an Intel GPU.")
    p.add_argument("--voice", help="TTS voice; model-specific (e.g. vivian for Qwen3-TTS CustomVoice)")
    p.add_argument("--language", help="TTS language, e.g. English")
    p.add_argument("--min-audio-s", type=float, default=0.3)
    p.add_argument("--roundtrip", action="store_true",
                   help="audio mode: transcribe the generated speech with the same omni "
                        "server and require word recall >= --min-recall vs the text answer")
    p.add_argument("--min-recall", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    p.add_argument("--wait", type=float, default=0.0,
                   help="seconds to poll /health before testing (0 = no wait)")
    p.add_argument("--out", default=".", help="directory for decoded audio/image artifacts")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.base_url

    deadline = time.monotonic() + args.wait
    while True:
        try:
            request(base, "/health", None, 10)
            break
        except (OSError, RuntimeError) as exc:
            if time.monotonic() >= deadline:
                print(f"SMOKE TEST FAIL: /health not ready: {exc}")
                return 1
            time.sleep(5)

    failed = False
    try:
        model = args.model or served_model(base, 30)
        print(f"model: {model}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"SMOKE TEST FAIL: /v1/models: {exc}")
        return 1
    for mode in args.mode:
        t0 = time.monotonic()
        try:
            detail = CHECKS[mode](base, model, args, out)
            print(f"[PASS] {mode} ({time.monotonic() - t0:.1f}s) {detail}")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError,
                AttributeError, IndexError, EOFError, struct.error) as exc:
            # A malformed or truncated payload must read as a clean [FAIL]
            # line, never a traceback: the whole point of this script is to
            # judge the server's output, and a crash judges nothing.
            failed = True
            print(f"[FAIL] {mode} ({time.monotonic() - t0:.1f}s) {type(exc).__name__}: {exc}")
    print("SMOKE TEST FAIL" if failed else "SMOKE TEST PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
