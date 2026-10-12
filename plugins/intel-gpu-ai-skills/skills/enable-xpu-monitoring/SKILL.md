---
name: enable-xpu-monitoring
description: Deploy a Prometheus + Grafana + XPU Manager monitoring stack on an Intel GPU host using the intel/xe-full-stack-monitoring repo (plain Docker Compose). Brings up containers for real-time GPU, CPU, and vLLM metrics collection and visualization. Use when the user wants to monitor XPU utilization, temperature, power, memory, or vLLM inference performance.
---

# enable-xpu-monitoring

Deploy a single-node monitoring stack on an Intel GPU host from
[intel/xe-full-stack-monitoring](https://github.com/intel/xe-full-stack-monitoring).
The stack is **pure Docker Compose** — no CLI, no Python package, no tarball.
That repo's [`AGENT.md`](https://github.com/intel/xe-full-stack-monitoring/blob/main/AGENT.md)
is the source of truth; this skill gives the short path and defers detail to it.

## What gets deployed

| Container | Port | Purpose |
|-----------|------|---------|
| `xpumd` (XPU Manager v2.0) | 9471 | GPU hardware metrics via Level Zero |
| `node-exporter` | 9100 | CPU, RAM, disk, network |
| `intel-xpu-prometheus` | 9090 | Scrapes xpumd, node, and vLLM |
| `intel-xpu-grafana` | 3000 | 4 dashboards: Quickview, GPU, vLLM, System |

All listeners bind to `127.0.0.1`. Grafana has no login (anonymous, read-only Viewer).

## Step 1 — Preflight

```sh
ls /dev/dri/renderD* >/dev/null 2>&1 && echo "[OK] GPU render nodes" || echo "[FAIL] no render nodes"
docker info >/dev/null 2>&1 && echo "[OK] docker" || echo "[FAIL] docker not reachable"
docker compose version >/dev/null 2>&1 && echo "[OK] compose" || echo "[FAIL] compose plugin missing"
for p in 9471 9100 9090 3000; do
  (ss -ltn 2>/dev/null || netstat -ltn) | grep -q ":$p " && echo "port $p IN USE" || echo "port $p free"
done
```

Fix any `[FAIL]`. Ports held by a previous run of this same stack are fine.
If 9471, 9100 or 9090 is held by anything else, it must be freed; those ports cannot be changed.

**If port 3000 is held by another app, do not stop that app.** Move Grafana to a free port instead:

```sh
GRAFANA_PORT=3000
if (ss -ltn 2>/dev/null || netstat -ltn) | grep -q ":3000 " \
   && ! docker ps --format '{{.Names}}' | grep -qx intel-xpu-grafana; then
  for p in $(seq 3001 3099); do
    (ss -ltn 2>/dev/null || netstat -ltn) | grep -q ":$p " || { GRAFANA_PORT=$p; break; }
  done
fi
echo "Grafana port: $GRAFANA_PORT"
```

If `GRAFANA_PORT` is not 3000, tell the user clearly:

> **Port 3000 is already in use on this host, so Grafana will run on port `<GRAFANA_PORT>`.**
> Use this port for the dashboard URL and the SSH tunnel.

## Step 2 — Get the repo

If the stack is already running, reuse the repo it was started from, so
Compose manages the same containers instead of a second copy:

```sh
REPO=$(docker inspect intel-xpu-grafana \
  --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' 2>/dev/null)
[ -n "$REPO" ] && [ -d "$REPO" ] && cd "$REPO" && echo "Existing stack in $REPO"
```

If that prints `Existing stack in ...`, skip the clone and go to Step 3.
Otherwise:

```sh
git clone https://github.com/intel/xe-full-stack-monitoring.git
cd xe-full-stack-monitoring
```

If `git clone` fails or hangs (no internet or proxy on the host), copy the repo
from the user's laptop instead. Run on the laptop:

```sh
git clone https://github.com/intel/xe-full-stack-monitoring.git
scp -r xe-full-stack-monitoring <ssh-host-alias>:~/
```

Then on the host: `cd ~/xe-full-stack-monitoring`.

Run every `docker compose` command below from this repo root (volume paths are relative).

If Step 1 moved Grafana, create an override file in the repo root. Compose reads it
automatically, and the repo's tracked files stay unchanged:

```sh
cat > docker-compose.override.yml <<EOF
services:
  grafana:
    environment:
      - GF_SERVER_HTTP_PORT=$GRAFANA_PORT
      - GF_SERVER_ROOT_URL=http://localhost:$GRAFANA_PORT/
EOF
docker compose config | grep GF_SERVER_HTTP_PORT   # shows the new port
```

## Step 3 — Tell the user the default vLLM ports, then ask

Before deploying, confirm the defaults from the cloned repo and state them clearly:

```sh
sed -n '/job_name: "vllm"/,$p' prometheus/prometheus.yml | grep localhost
```

> **Heads-up: vLLM metrics are scraped only on ports 8000–8008 by default.**
> A vLLM server on any other port will NOT show up in Grafana.
> Do you run vLLM on a different port that should be added? (default: no)

If the answer is no, skip Step 6.

## Step 4 — Deploy

This pulls ~1.5 GB of images and creates four containers. **Get the user's
confirmation first**, then:

```sh
docker compose up -d
```

Run this even when the stack is already up. It leaves running containers
alone, starts any that are stopped, and applies config changes. Do not skip
it, and do not stop or remove containers to "start fresh".

## Step 5 — Verify

Wait 10–15 s, then:

```sh
docker compose ps                                       # four containers Up
curl -s http://localhost:9090/-/healthy                 # Prometheus healthy
curl -s http://localhost:$GRAFANA_PORT/api/health       # Grafana healthy
curl -s http://localhost:9471/metrics | grep -c hw_gpu  # >0 = GPU metrics flowing
```

`vllm` targets show **down** until a vLLM server runs on a scraped port — that is
expected, not a failure. For deeper checks and troubleshooting, see the repo's `AGENT.md`.

## Step 6 — Add a non-default vLLM port (only if the user asked)

1. Edit `prometheus/prometheus.yml`, and add one line per port under the `vllm` job's `targets:`:

   ```yaml
         - targets:
             - "localhost:8000"
             # ... through 8008 ...
             - "localhost:8101"     # ← added
   ```

2. Restart **only** Prometheus so it picks up the new config:

   ```sh
   docker compose restart prometheus
   ```

   Do not use `curl -X POST localhost:9090/-/reload`. The lifecycle API is off.
   Also, the config is a single-file bind mount, so after an editor saves it,
   a reload can still read the old file. Do not run `docker compose down` for this change.
   Metrics history is kept in a volume, so it survives the restart.

3. Confirm the container sees it and the target is registered:

   ```sh
   docker exec intel-xpu-prometheus grep 8101 /etc/prometheus/prometheus.yml
   curl -s 'http://localhost:9090/api/v1/targets?state=active' | grep 8101
   ```

To stop monitoring a port, delete its line and restart Prometheus again.

## Step 7 — Report back and give remote access

Tell the user: Grafana is at **http://localhost:<GRAFANA_PORT>** (no login), how many GPUs
xpumd found, and which targets are up. Repeat which vLLM ports are monitored
(8000–8008, plus any added in Step 6). If Grafana was moved off 3000, say so again here.
Name the source: the stack is
[intel/xe-full-stack-monitoring](https://github.com/intel/xe-full-stack-monitoring),
deployed from `<repo path>`. Say this even when Step 2 reused a running stack.

Services listen on localhost only, so a remote user needs an SSH tunnel. **Always
give this generic form, and never fill in an IP address or hostname.** Replace
`<GRAFANA_PORT>` with the port from Step 1 (3000 by default). Run it on the user's laptop:

```sh
ssh -N -L <GRAFANA_PORT>:localhost:<GRAFANA_PORT> -L 9090:localhost:9090 <ssh-host-alias>
```

`<ssh-host-alias>` is a `Host` entry in the user's `~/.ssh/config`. They add it
themselves, for example:

```
Host gpu-box
    HostName <gpu-host>
    User <user>
    # ProxyJump <jump-host>   # if the GPU host is behind a bastion
```

Then open http://localhost:<GRAFANA_PORT> in the laptop browser. If that port is
already in use on the laptop, change only the first (local) number, for example
`-L 13000:localhost:<GRAFANA_PORT>`, and open http://localhost:13000.

## Limitations and common failures

| Symptom | Cause / fix |
|---------|-------------|
| `git clone` hangs | Host is behind a corporate proxy. Pass it for this command only, for example `git -c http.proxy=http://<proxy>:<port> clone ...`, or use the `scp` copy in Step 2 |

The skill does not install GPU drivers, change kernel parameters, or put Grafana on a non-localhost address.

## Teardown

```sh
docker compose down        # remove containers, keep metrics history
docker compose down -v     # also deletes volumes, which DESTROYS history. Confirm with the user first
```
