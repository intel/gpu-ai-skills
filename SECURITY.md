# Security Policy
Intel is committed to rapidly addressing security vulnerabilities affecting our customers and providing clear guidance on the solution, impact, severity and mitigation. 

## Reporting a Vulnerability
Please report any security vulnerabilities in this project utilizing the guidelines [here](https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html).

## What this pack is, for threat-modelling purposes

These skills emit commands for you to run; they are documentation and helper
scripts, not a service. Two properties of the emitted commands are deliberate
and are the operator's responsibility, not defects:

- **The model servers launched by these skills are unauthenticated and reachable
  on every host interface.** None of the emitted launch lines set an API key.
  `sglang-xpu-run` and `llamacpp-xpu-run` pass `--host 0.0.0.0` explicitly;
  `vllm-xpu-run` publishes the container port with `docker run -p 8000:8000`,
  which binds all host interfaces by default. This is deliberate — the documented
  purpose is single-host bring-up and benchmarking on a developer machine or a
  lab node — but it means anyone who can reach the port can submit inference
  requests, read the loaded model's identity, and consume the GPU. Publish to
  `127.0.0.1` instead (`-p 127.0.0.1:8000:8000`, or `--host 127.0.0.1`), put the
  port behind an authenticating reverse proxy, or confine it to a trusted network
  before exposing one of these servers beyond the host that runs it.
  `.env.example` ships `LLM_API_KEY="not-needed-for-local"` for the same reason —
  it is a placeholder for a local server, not a claim that authentication is
  unnecessary.

- **`--trust-remote-code` is never added for you, and never without a pinned
  revision.** It permits arbitrary Python from a model repository to execute
  inside the engine. The launcher defaults it off, and `model-config-recommend`
  will tell you when a model's `config.json` declares `auto_map` but still will
  not add the flag — you have to pass it yourself, having reviewed the repository
  and decided you trust the publisher at that revision. Naming that revision is
  required, not advisory: `recommend.py` refuses `--trust-remote-code` for a Hub
  model unless you also pass `--revision`, because an unpinned repository
  resolves its default branch when the server starts, so the code that executes
  is the publisher's latest push rather than the code you reviewed. The emitted
  launch line then pins both `--revision` and `--code-revision` — vLLM resolves
  repo-local modeling code separately from the weights.

- **`xpu-system-setup` installs system packages and requires `sudo`.** It fetches
  Docker's official convenience script over TLS and records its SHA-256 in the
  run log; the digest is recorded, not verified against a pin, because upstream
  republishes it frequently. Review the script yourself, or install Docker from
  Docker's apt repository with their signing key pinned, if unverified execution
  under `sudo` is not acceptable in your environment.

`tests/secure-config.sh` is the gate that keeps these properties from drifting;
it runs as part of `tests/static.sh`.
