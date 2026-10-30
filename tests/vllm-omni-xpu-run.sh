#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# CPU-only regression checks for the vllm-omni-xpu-run helper scripts.
# No Docker, GPU, or network: `docker` is a PATH shim that records its argv,
# and smoke_test.py talks to a stub server on 127.0.0.1.

set -euo pipefail

cd "$(dirname "$0")/.."

skill=plugins/intel-gpu-ai-skills/skills/vllm-omni-xpu-run
emit_launch="$skill/scripts/emit_launch.sh"
smoke="$skill/scripts/smoke_test.py"

tmpdir=$(mktemp -d)
stub_pid=""
cleanup() {
    if [ -n "$stub_pid" ]; then kill "$stub_pid" 2>/dev/null || true; fi
    rm -rf "$tmpdir"
}
trap cleanup EXIT

fail() {
    printf 'FAIL %s\n' "$1" >&2
    if [ -n "${2:-}" ] && [ -f "$2" ]; then sed 's/^/  | /' "$2" >&2; fi
    exit 1
}
has()   { grep -qF -- "$2" "$1" || fail "$3: missing '$2'" "$1"; }
hasnt() { if grep -qF -- "$2" "$1"; then fail "$3: unexpected '$2'" "$1"; fi; }
# expect_exit WANT OUTFILE CMD...  (stdout+stderr -> OUTFILE)
expect_exit() {
    local want=$1 out=$2 got=0
    shift 2
    "$@" >"$out" 2>&1 || got=$?
    [ "$got" = "$want" ] || fail "exit $got, want $want: $*" "$out"
}

# docker shim: one shell-quoted line per call, so assertions can grep for
# exact argv sequences.
shims="$tmpdir/bin"
calls="$tmpdir/calls.log"
mkdir -p "$shims"
cat >"$shims/docker" <<EOF
#!/usr/bin/env bash
{ printf '%q' docker; printf ' %q' "\$@"; printf '\n'; } >>"$calls"
EOF
chmod +x "$shims/docker"
with_shims() { PATH="$shims:$PATH" "$@"; }

# ---------------------------------------------------------------- emit_launch
out="$tmpdir/launch.sh"
err="$tmpdir/launch.err"

# Single card + Qwen2.5-Omni: the hardware-tested colocation preset. Every
# Qwen2.5-Omni launch below passes --no-remote-code: its bundled deploy YAML
# enables trust_remote_code, so the launcher demands a decision (tested in its
# own block further down).
bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0 --no-remote-code >"$out" 2>"$err"
bash -n "$out" || fail "emitted single-card launch does not parse" "$out"
for want in "--device /dev/dri" "-v /dev/dri/by-path:/dev/dri/by-path:ro" \
            "--ipc=host" "-p 8091:8091" "ZE_AFFINITY_MASK=0 " \
            "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
            "--model Qwen/Qwen2.5-Omni-7B --port 8091" "--stage-overrides '"; do
    has "$out" "$want" "single-card launch"
done
# Image line only: 4-space indent distinguishes it from `--name vllm-omni-xpu`,
# which would otherwise satisfy a bare substring match no matter what image
# was selected.
has "$out" "
    vllm-omni-xpu \\" "single-card launch image is the plain upstream name"
for bad in "--gpus" "--privileged" "--net=host" "--shm-size" "vllm serve" "# note" \
           "vllm-omni-xpu:"; do
    hasnt "$out" "$bad" "single-card launch"
done
has "$err" "colocating all 3 stages" "single-card stderr note"
python3 - "$out" <<'PY' || fail "single-card preset drifted from the tested split" "$out"
import json, re, sys
ov = json.loads(re.search(r"--stage-overrides '([^']+)'", open(sys.argv[1]).read()).group(1))
assert set(ov) == {"0", "1", "2"}, ov
assert all(stage["devices"] == "0" for stage in ov.values()), ov
# Found on an Arc Pro B70 (references/deploy-configs.md); retune on hardware only.
assert [ov[k]["gpu_memory_utilization"] for k in "012"] == [0.62, 0.19, 0.12], ov
for k in "01":
    assert ov[k]["max_model_len"] == 8192, ov[k]
    assert ov[k]["max_num_batched_tokens"] == 8192, ov[k]
# --no-remote-code merged into the preset without disturbing it.
assert all(stage["trust_remote_code"] is False for stage in ov.values()), ov
PY

# Two cards: upstream default layout, no memory preset.
bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0,1 --no-remote-code >"$out" 2>"$err"
has "$out" "ZE_AFFINITY_MASK=0,1 " "two-card launch"
hasnt "$out" "0.62" "two-card launch"

# Caller overrides merge over the single-card preset: their values win key by
# key, while a stage they did not name keeps the preset's placement. Taking a
# partial override as a complete layout would leave the unnamed stages on
# logical device 1, which one exposed card does not have.
bash "$emit_launch" --model Qwen/Qwen2.5-Omni-3B --devices 0 --no-remote-code \
    --stage-overrides '{"1":{"gpu_memory_utilization":0.25}}' >"$out" 2>"$err"
has "$err" "merged over that preset" "caller overrides merge"
python3 - "$out" <<'PY' || fail "caller overrides merge over the preset" "$out"
import json, re, sys
ov = json.loads(re.search(r"--stage-overrides '([^']+)'", open(sys.argv[1]).read()).group(1))
assert set(ov) == {"0", "1", "2"}, ov
assert all(stage["devices"] == "0" for stage in ov.values()), ov   # every stage placed
assert ov["1"]["gpu_memory_utilization"] == 0.25, ov               # caller value wins
assert ov["0"]["gpu_memory_utilization"] == 0.62, ov               # preset fills the rest
assert ov["2"]["gpu_memory_utilization"] == 0.12, ov
PY
# A placement the caller states explicitly is still theirs to get right, but a
# logical index the exposed cards do not have cannot start, so it is refused
# before the command is printed.
expect_exit 2 "$out" bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0 \
    --no-remote-code --stage-overrides '{"2":{"devices":"1"}}'
has "$out" "stage 2 is placed on logical device" "off-card placement refused"
expect_exit 2 "$out" bash "$emit_launch" --model some/unknown-model --devices 0,1 \
    --no-remote-code --stage-overrides '{"0":{"devices":"2"}}'
has "$out" "indices 0-1" "off-card placement names the valid range"
deploy="$tmpdir/my-omni.yaml"
printf 'stages: []\n' >"$deploy"
bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0 --no-remote-code --deploy-config "$deploy" >"$out" 2>"$err"
has "$out" "$deploy:/workspace/deploy/my-omni.yaml:ro" "deploy-config mount"
has "$out" "--deploy-config /workspace/deploy/my-omni.yaml" "deploy-config flag"
hasnt "$out" "0.62" "deploy-config launch"

# Host cache path. The default is emitted unexpanded so $HOME resolves on the
# host at run time. A caller-supplied path is emitted verbatim inside the
# double-quoted -v, so anything the shell could interpret there must be
# refused, not escaped (the emitted command gets executed, as above).
has "$out" '-v "$HOME/.cache/huggingface:/root/.cache/huggingface"' "default hf cache mount"
bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0,1 --no-remote-code --hf-cache /srv/hf-cache_v2 >"$out" 2>"$err"
has "$out" '-v "/srv/hf-cache_v2:/root/.cache/huggingface"' "custom hf cache mount"
hasnt "$out" '$HOME' "custom hf cache mount"
expect_exit 2 "$out" bash "$emit_launch" --model a/b --hf-cache relative/path
has "$out" "--hf-cache must be an absolute path" "relative hf-cache rejected"
# Each of these would break out of, or expand inside, the -v "..." argument.
for bad in '/srv/x"; echo pwned; echo "' '/srv/$(id)' '/srv/`id`' '/srv/a:b' \
           '/srv/with space' '/srv/back\slash'; do
    expect_exit 2 "$out" bash "$emit_launch" --model a/b --hf-cache "$bad"
    has "$out" "--hf-cache contains unsupported characters" "unsafe hf-cache rejected: $bad"
done

# Qwen3-Omni on too few cards: its default layout addresses logical devices
# 0-4, so with fewer exposed the talker lands on a device that does not exist
# and the container cannot start. There is no validated preset to substitute
# (unlike Qwen2.5-Omni above), so refuse instead of emitting a launch that is
# known to fail.
expect_exit 2 "$out" bash "$emit_launch" --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --devices 0,1 --no-remote-code
has "$out" "needs 5 devices" "Qwen3-Omni refused on too few cards"
has "$out" "within 0-1" "Qwen3-Omni refusal names the exposed range"
# --stage-overrides does not lift the guard. Fitting this pipeline on fewer
# cards also means shrinking the thinker's tensor_parallel_size, which the
# bundled YAML sets to 4; moving the stages while leaving TP at 4 fails just
# as surely, and the YAML cannot be read here to tell the two apart. An
# overlay states the whole layout in one file, which is what
# references/deploy-configs.md prescribes for this model.
expect_exit 2 "$out" bash "$emit_launch" --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --devices 0,1 --no-remote-code \
    --stage-overrides '{"0":{"devices":"0,1"},"1":{"devices":"0"},"2":{"devices":"0"}}'
has "$out" "tensor_parallel_size" "Qwen3-Omni placement alone is not accepted"
# A deploy overlay is, and the remote-code disable still covers all 3 stages.
bash "$emit_launch" --model Qwen/Qwen3-Omni-30B-A3B-Instruct --devices 0,1 \
    --no-remote-code --deploy-config "$deploy" >"$out" 2>"$err"
has "$out" "--deploy-config /workspace/deploy/my-omni.yaml" "Qwen3-Omni overlay accepted"
python3 - "$out" <<'PY' || fail "Qwen3-Omni overlay keeps the remote-code disable" "$out"
import json, re, sys
ov = json.loads(re.search(r"--stage-overrides '([^']+)'", open(sys.argv[1]).read()).group(1))
assert set(ov) == {"0", "1", "2"}, ov
assert all(stage["trust_remote_code"] is False for stage in ov.values()), ov
PY
# With all 5 exposed the upstream layout fits, so no placement is required.
bash "$emit_launch" --model Qwen/Qwen3-Omni-30B-A3B-Instruct --devices 0,1,2,3,4 --no-remote-code >"$out" 2>"$err"
has "$out" "ZE_AFFINITY_MASK=0,1,2,3,4 " "Qwen3-Omni on 5 cards"

# Remote code. Omitting the CLI flag leaves a bundled YAML's per-stage
# trust_remote_code in force, so the launcher must refuse to emit a launch
# until the caller decides -- for EVERY model, not just the families whose
# stage layout it knows: 79 of 141 bundled YAMLs enable it, and an allowlist
# would emit silently for the rest. --no-remote-code must disable it on every
# known stage, merged into whatever overrides are already there, and must
# refuse models whose stage layout it does not know. (The acknowledge path and
# the mutual exclusion are checked in tests/secure-config.sh.)
for m in Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice mistralai/Voxtral-4B-TTS-2603 \
         Tongyi-MAI/Z-Image-Turbo lance/some-model; do
    expect_exit 2 "$out" bash "$emit_launch" --model "$m" --devices 0
    has "$out" "does not turn it off" "remote-code decision required: $m"
done
# --deploy-config is not a decision. A live run copied the bundled
# qwen3_tts.yaml (trust_remote_code: true on both stages) out of the image and
# passed only that, and the server ran the model repository's code with
# nobody having opted in.
for m in Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice lance/some-model; do
    expect_exit 2 "$out" bash "$emit_launch" --model "$m" --deploy-config "$deploy"
    has "$out" "does not turn it off" "deploy-config alone is not a remote-code decision: $m"
    hasnt "$out" "docker run" "deploy-config alone is not a remote-code decision: $m"
done

bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0,1 --no-remote-code >"$out" 2>"$err"
has "$err" "trust_remote_code disabled on stages 0,1,2" "no-remote-code note"
has "$err" "grep -o 'trust_remote_code=[A-Za-z]*'" "resolved-value verification hint"
python3 - "$out" <<'PY' || fail "two-card --no-remote-code overrides" "$out"
import json, re, sys
ov = json.loads(re.search(r"--stage-overrides '([^']+)'", open(sys.argv[1]).read()).group(1))
assert set(ov) == {"0", "1", "2"}, ov
assert all(stage == {"trust_remote_code": False} for stage in ov.values()), ov
PY
bash "$emit_launch" --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --devices 0 --no-remote-code \
    --stage-overrides '{"1":{"devices":"0"}}' >"$out" 2>"$err"
python3 - "$out" <<'PY' || fail "--no-remote-code merges into caller overrides" "$out"
import json, re, sys
ov = json.loads(re.search(r"--stage-overrides '([^']+)'", open(sys.argv[1]).read()).group(1))
assert ov == {"0": {"trust_remote_code": False}, "1": {"devices": "0", "trust_remote_code": False}}, ov
PY
# A model whose stage layout the script does not know. Only per-stage values
# beat the YAML, so there is nothing for --no-remote-code to set, and emitting
# the launch with a warning alone would make the flag a silent no-op on exactly
# the models whose YAML cannot be read here. It must refuse until the caller
# names the stages.
expect_exit 2 "$out" bash "$emit_launch" --model some/unknown-model --no-remote-code
has "$out" "stage layout is unknown" "no-remote-code refuses an unknown stage layout"
bash "$emit_launch" --model some/unknown-model --no-remote-code \
    --stage-overrides '{"0":{},"1":{}}' >"$out" 2>"$err"
has "$err" "trust_remote_code disabled on stages 0,1" "caller-named stages"
python3 - "$out" <<'PY' || fail "caller-named stages are each disabled" "$out"
import json, re, sys
ov = json.loads(re.search(r"--stage-overrides '([^']+)'", open(sys.argv[1]).read()).group(1))
assert ov == {"0": {"trust_remote_code": False}, "1": {"trust_remote_code": False}}, ov
PY
# Stage ids become override keys and are joined with commas, so a key that is
# not an integer cannot be one of this pipeline's stages.
expect_exit 2 "$out" bash "$emit_launch" --model some/unknown-model --no-remote-code \
    --stage-overrides '{"thinker":{}}'
has "$out" "keys must be stage ids" "non-integer stage id rejected"
# A --deploy-config does not substitute for naming the stages: an overlay can
# inherit stages from a base_config inside the image, so reading the host file
# could never prove every stage ends up disabled. Declining has to be enforced
# or refused, never merely asserted.
expect_exit 2 "$out" bash "$emit_launch" --model some/unknown-model --no-remote-code \
    --deploy-config "$deploy"
has "$out" "does not substitute" "deploy-config is not an enforceable decline"
# Named stages are enforced, and the deploy config can still ride along. Every
# launch also gets the resolved-value verification hint.
bash "$emit_launch" --model some/unknown-model --no-remote-code --deploy-config "$deploy" \
    --stage-overrides '{"0":{},"1":{}}' >"$out" 2>"$err"
has "$out" "--deploy-config /workspace/deploy/my-omni.yaml" "stages named beside a deploy config"
has "$err" "trust_remote_code disabled on stages 0,1" "stages named beside a deploy config"
has "$err" "confirm the resolved value" "verification hint on unknown model"

# Every option, then execute the emitted command against the docker shim to
# prove quoting survives the round trip (JSON and passthrough args intact).
# --no-remote-code appears here only because the decision gate requires a
# choice for every model, and it carries the stage id because Z-Image's layout
# is not one the script knows; this file must never carry the live
# acknowledgement flag (tests/secure-config.sh scans for it and owns the
# opt-in semantics).
VLLM_OMNI_XPU_IMAGE=custom/omni:t1 bash "$emit_launch" --model Tongyi-MAI/Z-Image-Turbo \
    --devices 1 --port 9000 --name img --privileged --host-network --first-launch \
    --no-remote-code --stage-overrides '{"0":{}}' \
    -- --limit-mm-per-prompt '{"video":0}' --max-num-seqs 1 >"$out" 2>"$err"
bash -n "$out" || fail "emitted option-heavy launch does not parse" "$out"
: >"$calls"
with_shims bash "$out"
for want in "docker run -d --name img" "--privileged" "--net=host" \
            "-e ZE_AFFINITY_MASK=1" "custom/omni:t1 --model Tongyi-MAI/Z-Image-Turbo --port 9000" \
            "--stage-init-timeout 1200 --init-timeout 2400" \
            "--limit-mm-per-prompt \\{\\\"video\\\":0\\} --max-num-seqs 1"; do
    has "$calls" "$want" "executed option-heavy launch"
done
hasnt "$calls" "-p 9000:9000" "host-network launch"

# Inputs that must be rejected before anything is printed.
expect_exit 2 "$out" bash "$emit_launch"
expect_exit 2 "$out" bash "$emit_launch" --model 'x;y'
expect_exit 2 "$out" bash "$emit_launch" --model a/b --devices 0,,1
expect_exit 2 "$out" bash "$emit_launch" --model a/b --devices gpu0
expect_exit 2 "$out" bash "$emit_launch" --model a/b --port 80a
expect_exit 2 "$out" bash "$emit_launch" --model a/b --stage-overrides "{'0':{}}"
expect_exit 2 "$out" bash "$emit_launch" --model a/b --no-remote-code --stage-overrides '{"0":{}}' \
    --deploy-config "$tmpdir/missing.yaml"
has "$out" "--deploy-config file not found" "missing deploy config"
expect_exit 2 "$out" bash "$emit_launch" --model a/b --bogus

# A JSON-escaped apostrophe passes the raw-input check, and merging the
# single-card preset decodes it into a literal ' that would close the emitted
# single-quoted argument. The check has to run on the final JSON.
esc=$(printf '%su0027' '\')
expect_exit 2 "$out" bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0 \
    --no-remote-code --stage-overrides "{\"0\":{\"x\":\"${esc}; id; ${esc}\"}}"
has "$out" "must not contain single quotes" "decoded apostrophe refused"
hasnt "$out" "docker run" "decoded apostrophe refused"

# Upstream keeps the last of a repeated option, so a launcher-managed option
# passed again after -- would replace the one emitted here, dropping the
# remote-code disable. It also accepts underscores and unambiguous prefixes.
for again in "--stage-overrides {}" "--stage_overrides={}" "--stage-over {}"; do
    # shellcheck disable=SC2086
    expect_exit 2 "$out" bash "$emit_launch" --model Qwen/Qwen2.5-Omni-7B --devices 0,1 \
        --no-remote-code -- $again
    has "$out" "would replace the launcher's own --stage-overrides" "passthrough refused: $again"
done

# A value-taking option with no value must die with a focused message. Without
# the guard, the failed `shift 2` leaves $# unchanged and the parser spins on
# the same flag forever, so run these under a timeout.
expect_exit 2 "$out" timeout 5 bash "$emit_launch" --model
has "$out" "--model requires a value" "trailing --model"
hasnt "$out" "shift count" "trailing --model"
expect_exit 2 "$out" timeout 5 bash "$emit_launch" --model a/b --hf-cache
has "$out" "--hf-cache requires a value" "trailing --hf-cache"
hasnt "$out" "shift count" "trailing --hf-cache"

# ----------------------------------------------------------------- smoke_test
# Stub server: the first URL path segment selects behaviour, so one process
# serves every scenario (smoke_test.py appends /health, /v1/... to --base-url).
stub="$tmpdir/stub_server.py"
cat >"$stub" <<'PY'
import base64, io, json, random, struct, sys, wave, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = "The capital of France is Paris."

def wav(seconds, rate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()

def png(width, height, noisy):
    rng = random.Random(0)
    row = lambda: b"\x00" + (rng.randbytes(width * 3) if noisy else bytes(width * 3))
    raw = b"".join(row() for _ in range(height))
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def route(self):
        mode, _, rest = self.path.lstrip("/").partition("/")
        return mode, "/" + rest

    speak_without_system = False

    def do_GET(self):
        mode, path = self.route()
        if path == "/health":
            return self.reply(200, b"", "text/plain")
        if path == "/v1/models":
            return self.reply(200, {"data": [{"id": "stub/omni"}]})
        self.reply(404, {"error": path})

    def do_POST(self):
        mode, path = self.route()
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if path == "/v1/chat/completions":
            if mode == "error":
                return self.reply(500, {"error": "stage 2 crashed"})
            msgs = req["messages"]
            content = msgs[-1]["content"]
            if isinstance(content, list):  # --roundtrip transcription request
                # Like the real talker: speech is garbled unless the speak
                # request carried a system prompt.
                garbled = mode == "garbled" or type(self).speak_without_system
                heard = "我喝。" if garbled else "the capital of france is paris"
                return self.reply(200, {"choices": [{"message": {"content": heard}}]})
            type(self).speak_without_system = not any(
                m.get("role") == "system"
                and m.get("content") == (
                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                )
                for m in msgs
            )
            msg = {"role": "assistant", "content": ANSWER}
            if "audio" in req.get("modalities", []):
                blob = wav(0.1 if mode == "short" else 1.9)
                if mode == "truncwav":
                    blob = blob[:30]  # cut inside the fmt chunk
                elif mode == "headeronlywav":
                    # Valid, complete header; samples gone. The declared
                    # duration still reads as 1.9 s.
                    blob = blob[:44]
                data = base64.b64encode(blob).decode()
                msg["audio"] = {"id": "a", "data": data, "expires_at": 0, "transcript": ""}
            return self.reply(200, {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})
        if path == "/v1/images/generations":
            width, height = map(int, req["size"].split("x"))
            if mode == "wrongsize":  # a server that ignores the requested size
                width, height = width // 2, height // 2
            blob = png(width, height, noisy=(mode != "blank"))
            if mode == "corruptpng":
                # Intact IHDR, IEND and CRCs around image data that does not
                # inflate: the framing alone must not pass as a valid image.
                garbage = random.Random(1).randbytes(20000)
                blob = (blob[:33] + struct.pack(">I", len(garbage)) + b"IDAT" + garbage
                        + struct.pack(">I", zlib.crc32(b"IDAT" + garbage)) + blob[-12:])
            if mode == "truncpng":
                blob = blob[:16]  # magic plus a partial IHDR, no dimensions
            elif mode == "headeronlypng":
                # Full IHDR (so the dimensions read as 512x512) but the image
                # data and IEND are missing.
                blob = blob[:40]
            data = base64.b64encode(blob).decode()
            return self.reply(200, {"data": [{"b64_json": data}]})
        if path == "/v1/audio/speech":
            blob = wav(1.2)
            if mode == "truncwav":
                blob = blob[:30]
            elif mode == "headeronlywav":
                blob = blob[:44]
            return self.reply(200, blob, "audio/wav")
        self.reply(404, {"error": path})

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(server.server_address[1]))
import os; os.replace(sys.argv[1] + ".tmp", sys.argv[1])
server.serve_forever()
PY
portfile="$tmpdir/stub.port"
python3 "$stub" "$portfile" &
stub_pid=$!
for _ in $(seq 1 100); do [ -s "$portfile" ] && break; sleep 0.1; done
[ -s "$portfile" ] || fail "stub server did not start"
base="http://127.0.0.1:$(cat "$portfile")"
art="$tmpdir/artifacts"
sm="$tmpdir/smoke.out"
run_smoke() { python3 "$smoke" --out "$art" --timeout 30 "$@"; }

# Healthy omni server: text + intelligible audio.
expect_exit 0 "$sm" run_smoke --base-url "$base/good" --mode text --mode audio --roundtrip --expect paris
has "$sm" "SMOKE TEST PASS" "good omni"
has "$sm" "word recall 1.00" "good omni"
[ -s "$art/smoke_audio.wav" ] || fail "good omni: smoke_audio.wav not written" "$sm"

# The Qwen2.5-Omni failure shape: valid WAV, unrelated speech.
expect_exit 1 "$sm" run_smoke --base-url "$base/garbled" --mode text --mode audio --roundtrip
has "$sm" "[PASS] text" "garbled speech"
has "$sm" "speech is not intelligible" "garbled speech"
has "$sm" "word recall 0.00" "garbled speech"
has "$sm" "SMOKE TEST FAIL" "garbled speech"
# Without --roundtrip the same response passes: format checks cannot catch it,
# which is why the skill mandates --roundtrip for omni chat models.
expect_exit 0 "$sm" run_smoke --base-url "$base/garbled" --mode audio
has "$sm" "SMOKE TEST PASS" "garbled speech without --roundtrip"
# The talker needs the Qwen-Omni system prompt: the checker sends it by
# default (the good stub stays intelligible above), and dropping it garbles.
expect_exit 1 "$sm" run_smoke --base-url "$base/good" --mode audio --roundtrip --system ""
has "$sm" "word recall 0.00" "no system prompt"

expect_exit 1 "$sm" run_smoke --base-url "$base/short" --mode audio
has "$sm" "audio too short" "short audio"
expect_exit 1 "$sm" run_smoke --base-url "$base/good" --mode text --expect berlin
has "$sm" "does not contain" "--expect mismatch"
expect_exit 1 "$sm" run_smoke --base-url "$base/error" --mode text
has "$sm" "HTTP 500" "server error"

# Diffusion: real-looking PNG passes, a flat frame is flagged as blank.
expect_exit 0 "$sm" run_smoke --base-url "$base/good" --mode image
has "$sm" "image: 512x512" "good image"
[ -s "$art/smoke_image.png" ] || fail "good image: smoke_image.png not written" "$sm"
expect_exit 1 "$sm" run_smoke --base-url "$base/blank" --mode image
has "$sm" "likely a blank image" "blank image"

# TTS endpoint.
expect_exit 0 "$sm" run_smoke --base-url "$base/good" --mode speech --voice vivian --language English
has "$sm" "speech: 1.20s @ 24000 Hz" "speech"
[ -s "$art/smoke_speech.wav" ] || fail "speech: smoke_speech.wav not written" "$sm"

# Truncated payloads (a stage that died mid-stream) must read as a clean
# [FAIL] line with a reason, never a struct.error or EOFError traceback.
expect_exit 1 "$sm" run_smoke --base-url "$base/truncwav" --mode audio
has "$sm" "[FAIL] audio" "truncated chat audio"
has "$sm" "WAV truncated" "truncated chat audio"
hasnt "$sm" "Traceback" "truncated chat audio"
expect_exit 1 "$sm" run_smoke --base-url "$base/truncwav" --mode speech --voice vivian
has "$sm" "[FAIL] speech" "truncated speech"
has "$sm" "WAV truncated" "truncated speech"
hasnt "$sm" "Traceback" "truncated speech"
expect_exit 1 "$sm" run_smoke --base-url "$base/truncpng" --mode image
has "$sm" "[FAIL] image" "truncated image"
has "$sm" "PNG truncated" "truncated image"
hasnt "$sm" "Traceback" "truncated image"

# A payload whose header is intact but whose data is gone must NOT pass on the
# declared duration/dimensions. A stage that dies after writing the header is
# the realistic source of this, and the format checks are the only thing
# standing between it and a "validated" report.
expect_exit 1 "$sm" run_smoke --base-url "$base/headeronlywav" --mode audio
has "$sm" "[FAIL] audio" "header-only chat audio"
has "$sm" "WAV truncated" "header-only chat audio"
hasnt "$sm" "Traceback" "header-only chat audio"
expect_exit 1 "$sm" run_smoke --base-url "$base/headeronlywav" --mode speech --voice vivian
has "$sm" "WAV truncated" "header-only speech"
expect_exit 1 "$sm" run_smoke --base-url "$base/headeronlypng" --mode image
has "$sm" "[FAIL] image" "header-only image"
has "$sm" "PNG truncated" "header-only image"
hasnt "$sm" "Traceback" "header-only image"
expect_exit 1 "$sm" run_smoke --base-url "$base/corruptpng" --mode image
has "$sm" "[FAIL] image" "corrupt image data"
has "$sm" "does not decompress" "corrupt image data"
hasnt "$sm" "Traceback" "corrupt image data"

# A server that ignores the requested resolution has not served the request.
expect_exit 1 "$sm" run_smoke --base-url "$base/wrongsize" --mode image
has "$sm" "do not match requested 512x512" "wrong image size"

# Nothing listening: fail fast with a readiness message, not a traceback.
expect_exit 1 "$sm" run_smoke --base-url "http://127.0.0.1:1" --wait 0 --mode text
has "$sm" "SMOKE TEST FAIL: /health not ready" "unreachable server"
hasnt "$sm" "Traceback" "unreachable server"

echo "OK vllm-omni-xpu-run helper script checks passed"
