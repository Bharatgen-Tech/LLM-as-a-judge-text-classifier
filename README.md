# Epic Data Classifier

This repository contains inference and utility scripts used for model evaluation and deployment. Key files added recently include:

- `async_3.py` — main asynchronous inference script (entrypoint for batch/async inference).
- `chk_dup.py` — duplicate-checking helper.
- `deploy.sh` — helper script to start the inference service (vLLM-based).
- `stop.sh` — helper script to stop the inference service.
- `haproxy_class.cfg` — HAProxy configuration for load balancing / routing.
- `ider.py`, `ider_dup.py`, `parser.py`, `segg_data.py`, `qual_class_prompts.yaml` — utility and data preparation scripts.

**This README** documents prerequisites, how to run HAProxy in Docker using the provided config, and how to set up and run vLLM-backed inference using `deploy.sh` and `stop.sh`.

## Prerequisites

- Linux host (instructions assume Bash). GPU recommended for vLLM inference.
- Docker (for HAProxy or containerized services).
- Python 3.9+ and `virtualenv` or `venv` for local Python environment.
- NVIDIA drivers + CUDA toolkit (if using GPU). Ensure `nvidia-smi` works.

Optional but recommended:
- `git` to manage code and updates.

## Files Overview

- `async_3.py`: Async inference worker / server. Use `deploy.sh` to run in production mode.
- `deploy.sh`: Starts the inference process (expects Python env and dependencies installed).
- `stop.sh`: Stops the running inference process started by `deploy.sh`.
- `haproxy_class.cfg`: HAProxy config — used to route traffic to inference workers or upstreams.

## HAProxy with Docker

Edit the HAProxy configuration files  with the node IPs and ports:
```bash 
  server <name> <IP>:<port> check
```


Run HAProxy using the official image while mounting the provided configuration file. From the repository root run:

```bash
docker run -d \
  --name haproxy \
  -p 80:80 \
  -p 1936:1936 \
  -v $(pwd)/haproxy_class.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro \
  haproxy:2.6
```

- `-p 80:80` exposes the HTTP frontend defined in `haproxy_class.cfg`.
- `-p 1936:1936` exposes the HAProxy stats/metrics socket (if enabled in the config).
- Adjust ports as needed (for example if your inference service listens on a different port).

To view HAProxy logs:

```bash
docker logs -f haproxy
```

To stop and remove the HAProxy container:

```bash
docker stop haproxy && docker rm haproxy
```

## Setting up vLLM for inference

vLLM is used to serve LLM models efficiently on GPU. The repository includes `deploy.sh` which automates starting `async_3.py` with the recommended environment. The `stop.sh` script will stop the running process.

Below are recommended installation and deployment steps. These are intentionally generic so they work across different systems — adjust model paths, ports, and environment variables to your setup.

1. Create and activate a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
```

2. Install dependencies (example)

```bash
pip install vllm transformers accelerate sentencepiece
# If you have a requirements.txt, run:
# pip install -r requirements.txt
```

Notes:
- vLLM requires a CUDA-enabled environment for GPU inference. If you are using CPU-only or a different accelerator, adjust expectations accordingly.
- Installing `vllm` may require additional system packages (C++ build tools, CUDA headers). Refer to the vLLM docs for platform-specific instructions.

## Using `deploy.sh` and `stop.sh`

Make the scripts executable (if needed) and run `deploy.sh` to start the inference service.

Concurrently deploys a [vLLM](https://github.com/vllm-project/vllm) server across multiple remote hosts over SSH. Each host gets its own `tmux` session and an auto-incremented port, with all output captured to a timestamped remote log file.

---

## Requirements

- `tmux` installed on every remote host
- `uv` available in the remote environment
- SSH key-based (passwordless) access to all hosts
- A venv containing vLLM at a known path on the remote hosts

---

## Usage
Edit the deploy.sh file to add the IPs of the hosts on the hardcoded list
```bash
./deploy.sh [OPTIONS]
```

### Required flags

| Flag | Description |
|---|---|
| `-m, --model MODEL` | Model name or path to serve (e.g. `openai/gpt-oss-120b`) |
| `-e, --venv PATH` | Absolute path to the venv `activate` script on the remote hosts |

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `-t, --tp N` | `4` | Tensor parallel size |
| `-d, --dp N` | `2` | Data parallel size |
| `-p, --port N` | `30600` | Base port — each host gets `base + index` |
| `-g, --gpus LIST` | `0,1,2,3,4,5,6,7` | `CUDA_VISIBLE_DEVICES` value |
| `-H, --hosts H1,H2,...` | hardcoded list | Comma-separated list of remote hostnames |
| `-u, --gpu-mem F` | `0.90` | GPU memory utilisation fraction |
| `--extra "FLAGS"` | prefix-caching + trust-remote-code + async-scheduling | Extra vLLM flags passed through verbatim |
| `-h, --help` | | Print usage and exit |

---

## Examples

**Standard two-host deployment with 8 GPUs each:**
```bash
./deploy_vllm.sh \
  --model openai/gpt-oss-120b \
  --tp 4 --dp 2 \
  --port 30600 \
  --venv /fsxnew/user/vllmenv/bin/activate
```
Host `ip-10-0-249-61` serves on port `30600`, host `ip-10-0-252-170` on `30601`.

**Custom host list, 4 GPUs, with extra vLLM flags:**
```bash
./deploy_vllm.sh \
  --model meta-llama/Llama-3-70b \
  --tp 4 --dp 1 \
  --port 8000 \
  --gpus 0,1,2,3 \
  --hosts ip-10-0-1-10,ip-10-0-1-11,ip-10-0-1-12 \
  --venv /home/ubuntu/venv/bin/activate \
  --extra "--max-num-seqs 64 --max-model-len 32768"
```

---

## Behaviour

- Hosts are deployed **concurrently** (each in a background subshell).
- An existing `tmux` session with the same name is killed before re-deploying.
- SSH connectivity is verified before any deployment step.
- Errors are printed to stdout **and** appended to `deployment_errors.log` in the script directory.
- Remote stdout/stderr is captured to `~/vllm_<model>_<timestamp>.log` on each host.

### Checking on a running deployment

```bash
# Attach to the tmux session on a host
ssh ip-10-0-249-61 "tmux attach -t vllm_ip-10-0-249-61"

# Tail the remote log (without attaching)
ssh ip-10-0-249-61 "tail -f ~/vllm_*.log"
```

---

## Port allocation

Ports are assigned by host order, starting from `--port`:

| Host index | Port |
|---|---|
| 0 (first host) | `BASE_PORT` |
| 1 | `BASE_PORT + 1` |
| N | `BASE_PORT + N` |

---

## Error log

All deployment failures are appended to `deployment_errors.log` in the same directory as the script, with the format:

```
[YYYY-MM-DD HH:MM:SS] DEPLOYMENT ERROR | host=<host> | <reason>
```
To stop the service:

```bash
./stop.sh
# or, if the script uses a PID file:
# kill $(cat service.pid)
```

If you'd rather run `async_3.py` directly for debugging, a typical invocation is:

```bash
# example, replace args with actual script options
python async_3.py --model /path/to/model --port 8000 --other-flags
```

Then send requests to the HTTP/gRPC endpoint or test locally with `curl`:

```bash
curl -X POST http://localhost:8000/generate -d '{"prompt":"Hello"}'
```

Adjust the endpoint path to match what `async_3.py` exposes.

## Detailed CLI reference

Below are the exact required inputs and common options for the main scripts so you can run them reliably.

- `async_3.py` (core async inference runner)

  Required arguments:
  - `--input-path` : path to a JSON/JSONL/Parquet file containing input records.
  - `--output-file`: path to the output JSONL file to append generated results to.
  - `--instruction-path`: path to a YAML file containing instruction templates.
  - `--task`      : Instruction in the YAML path.

  Common optional arguments:
  - `--template-fields` : list of JSON keys in your input data used to fill template placeholders in task prompt.
  - `--backend`         : backend type (choices include `vllm`, `vllm-chat`, `sglang`, `trt`, ...). Default: `vllm-chat`.
  - `--model`           : model id (if omitted the script will query the backend for a default model).
  - `--base-url`        : full base URL to the model server (overrides `--host`/`--port`).
  - `--host` / `--port` : host and port of the model server (defaults chosen by backend if omitted).
  - `--extra-request-body` : JSON string of extra body fields to include in each request.
  - `--max-concurrency`, `--request-rate`, `--enable-stream`, `--disable-tqdm`, `--parse-quality`.

  Input expectations and outputs:
  - Input records must contain the fields referenced by the instruction template (commonly `id` and any fields used by `--template-fields`).
  - Supported input formats: `.json`, `.jsonl`, `.parquet`.
  - Output: newline-delimited JSON written to `--output-file` (appends). A companion errors file is created as `*.errors.jsonl` for parse/request errors.

  Example:

  ```bash
  python async_3.py \
    --input-path data/tasks.jsonl \
    --output-file out/results.jsonl \
    --instruction-path instructions.yaml \
    --task my_task.path \
    --template-fields text_field id \
    --backend vllm-chat \
    --base-url http://localhost:8000 \
    --max-concurrency 16
  ```

- `chk_dup.py` (detect duplicate IDs)

  Usage:
  ```bash
  python chk_dup.py /path/to/file.jsonl --field id --show-dupes 20
  ```

- `ider.py` (add UUIDs to records missing `id`)

  Usage (stdin/stdout or files):
  ```bash
  python ider.py input.jsonl output.jsonl
  # or
  cat input.jsonl | python ider.py > output.jsonl
  ```

- `ider_dup.py` (append UUID suffix to duplicate IDs and add ID if no ID exists)

  Usage:
  ```bash
  python ider_dup.py input.jsonl output.jsonl
  ```

- `parser.py` (re-parse `generated_text` into structured judge fields)

  Usage:
  ```bash
  python parser.py --input scored.jsonl --output reparsed.jsonl [--text-key generated_text]
  ```

- `segg_data.py` (join eval results with data and segregate per `label`)

  Usage:
  ```bash
  python segg_data.py --input eval.jsonl --data source.jsonl --output-dir ./out --label-field label
  ```

## Example environment variables

- `MODEL_PATH` — path to the pre-downloaded model or model repo.
- `PORT` — port for the inference server to listen on (default examples: `8000`, `8080`).
- `NUM_WORKERS` — number of worker processes or threads.
- `CUDA_VISIBLE_DEVICES` — GPU selection if multiple GPUs are available.

Export before running `deploy.sh` or include them in the script:

```bash
export MODEL_PATH=/path/to/model
export PORT=8000
export NUM_WORKERS=1
export CUDA_VISIBLE_DEVICES=0
./deploy.sh
```

## Troubleshooting

- If `deploy.sh` fails: run the script commands step-by-step to find missing dependencies.
- If vLLM fails to import: ensure you have matching CUDA toolkit and driver versions, and that `pip` installed the GPU-enabled wheels.
- If HAProxy routing fails: check `haproxy_class.cfg` for correct backend server definitions and ports, and confirm the inference service is listening on the expected port.
- If you get permission errors with Docker bind-mounts, ensure file permissions are readable by the Docker daemon.

## Next steps and customization

- Review `deploy.sh` to confirm how it launches `async_3.py` and what environment variables it expects.
- If you want HAProxy to balance multiple worker instances, start multiple inference workers on different ports and add them to the backend pool in `haproxy_class.cfg`.
- Consider using `docker-compose` or Kubernetes for production orchestration when scaling multiple workers.

