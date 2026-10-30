# ImagineIt

ImagineIt is split into two independent processes:

- **App backend** (`app/`) owns the public API, MariaDB, MinIO, job state, and the worker gateway.
- **GPU worker** (`gpu_worker.py`) runs on the GPU machine, loads the model, and performs inference. It opens one **outbound** WebSocket and never listens on a port.

The frontend (`webapp/`) only talks to the app backend.

## Architecture

```
Browser ──HTTP──> App backend ──WS (outbound connection from GPU)──> GPU worker
                     │                                                  │
                     ├── MariaDB: job state, progress, errors            └── model + GPU
                     └── MinIO: finished JPEGs
```

The app is the only server. GPU workers dial `APP_WS_URL`, authenticate with a per-worker pre-shared key using an HMAC challenge-response, and then exchange AES-256-GCM encrypted messages. Prompts, parameters, progress, errors, and image bytes are all inside the ciphertext, so content security does not depend on TLS. Running over `wss://` is still recommended as an extra layer.

A worker can reconnect after a network drop and continue its current job. The app keeps the job on a lease, and results are retried until acknowledged.

## Requirements

App host:

- Python 3.11+
- MariaDB, MinIO, and optionally Valkey
- `pip install -e .`

GPU host:

- Python 3.11+
- `pip install -e '.[gpu]'`
- Apple Silicon: MLX, macOS
- Linux/Windows: torch, diffusers, transformers, and the appropriate GPU driver
- Network access to the app backend, plus Hugging Face on first model download

## Setup

```bash
cp .env.example .env
docker compose up -d
alembic upgrade head
```

Generate a key pair for one GPU. The script writes the app's worker list and a private worker file:

```bash
python scripts/generate_worker_key.py \
  --worker-id gpu-01 \
  --app-file gpu_workers.json \
  --worker-file gpu-01.worker.json
```

Both generated files are ignored by git. Keep them mode `0600`.

Start the app:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 4600
```

On the GPU machine, copy the worker key file over a secure channel and run:

```bash
git clone <repo> imagineit3
cd imagineit3
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu]'

python gpu_worker.py \
  --app-url ws://app-host:4600/internal/v1/workers/ws \
  --key-file /path/to/gpu-01.worker.json
```

For a production or remote link, use `wss://`:

```bash
python gpu_worker.py \
  --app-url wss://app.example.com/internal/v1/workers/ws \
  --key-file /path/to/gpu-01.worker.json
```

If the app uses a private CA, pass `--ca-file /path/to/ca.pem`. Certificate verification cannot be disabled.

## Worker options

```bash
python gpu_worker.py --help
```

Useful flags and environment variables:

| Flag | Environment | Default |
| --- | --- | --- |
| `--app-url` | `APP_WS_URL` | `ws://localhost:4600/internal/v1/workers/ws` |
| `--key-file` | `GPU_WORKER_KEY_FILE` | `~/.imagineit/worker.json` |
| `--worker-id` | `GPU_WORKER_ID` | from key file |
| `--key-id` | `GPU_WORKER_KEY_ID` | from key file |
| `--model` | `GPU_MODEL` | `wssdaas/nova-anime-xl-ilv-19.0` |
| `--ca-file` | `GPU_CA_FILE` | system trust store |
| `--max-concurrency` | | `1` |
| `--gpus` | `GPU_WORKER_GPUS` | `1` |

Run the worker under `systemd` or `launchd` so it restarts automatically after a crash or reboot.

### Multiple GPUs

On a machine with multiple CUDA/XPU devices, start one worker process per GPU:

```bash
python gpu_worker.py --gpus auto --key-file /path/to/gpu-01.worker.json
```

Use `--gpus 2` to choose an explicit count.  Each child process gets its own `CUDA_VISIBLE_DEVICES` and loads its own copy of the model, and the app sees them as separate workers such as `gpu-01` and `gpu-01-1`.

The generated app key entry includes `allow_instances: true`, which lets the base worker key authenticate those numbered instance IDs.  Existing key files need that flag added or must be regenerated before using multiple GPUs.  Apple Silicon MLX exposes one GPU device, so `--gpus auto` runs one worker there.

Each GPU loads a full model copy, so memory use scales with the worker count.

For UI and protocol testing without a GPU, use the fake worker:

```bash
python scripts/fake_gpu_worker.py \
  --app-url ws://localhost:4600/internal/v1/workers/ws \
  --key-file gpu-01.worker.json
```

## App configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `GPU_WORKERS_FILE` | JSON worker key list | `gpu_workers.json` |
| `WORKER_HEARTBEAT_INTERVAL` | Worker heartbeat seconds | `15` |
| `WORKER_LEASE_SECONDS` | Job lease granted to a worker | `300` |
| `WORKER_DISCONNECT_GRACE_SECONDS` | Time a disconnected worker has to reconnect | `60` |
| `MAX_RESULT_BYTES` | Largest accepted JPEG | `8388608` |

If you raise `MAX_RESULT_BYTES`, also raise Uvicorn's WebSocket frame limit with `--ws-max-size`; the default result size is chosen to fit under Uvicorn's 16 MiB default after base64 encoding.
| `MINIO_HOST` / `MINIO_PORT` | Object storage endpoint | `localhost:9000` |

The key file format is:

```json
{
  "workers": [
    {"worker_id": "gpu-01", "key_id": "k1", "key": "<base64 32-byte key>", "allow_instances": true}
  ]
}
```

Multiple entries for the same `worker_id` with different `key_id` values allow key rotation with an overlap window. The app reads this file at startup, so restart it after changing keys.

## Public API

- `GET /api/v3/status`
- `GET /api/v3/workers` returns connected GPU worker metadata for the status panel
- `POST /api/v3/images/inferences` returns `{"promise_id": "..."}`
- `GET /api/v3/images/inferences/{promise_id}` returns `status`, `steps_status`, `steps_completed`, `steps_total`, and `error`
- `GET /api/v3/images/{promise_id}` returns the JPEG once it exists

The inference status values are `queued`, `dispatched`, `running`, `succeeded`, `failed`, and `cancelled`.

## Security notes

- Every application message after the handshake is AES-256-GCM encrypted with keys derived from the worker's pre-shared key.
- The handshake uses HMAC-SHA256 challenge-response, so the key itself is never transmitted.
- Sequence numbers and associated data prevent replay, reordering, and tampering.
- A static pre-shared key does not provide forward secrecy; a leaked key exposes captured sessions, so rotate keys and use `wss://` for defence in depth.
- Plain `ws://` still hides message contents, but leaks connection metadata and timing; use `wss://` when possible.
- The app backend can decrypt prompts and images. This is transport/content security, not zero-knowledge.
- Prompts and generation parameters are stored in MariaDB as JSON so queued jobs survive an app restart; protect database access accordingly.
- Keys are stored outside the database. Do not commit `gpu_workers.json` or `*.worker.json`.
- The app backend currently runs as one instance because a worker connection lives in one process. Multi-instance routing would require a shared broker such as Valkey.

## Tests

```bash
pip install -e '.[dev]'
pytest
```
