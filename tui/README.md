# imagineit-tui

A small, modern terminal client for ImagineIt. It talks to the public v3 API,
decrypts generated images locally, and renders them inline with
[`ratatui-image`](https://crates.io/crates/ratatui-image).

```
✦ imagineit  terminal                    ● account 68a1d1d…
127.0.0.1:4600  ·  kitty                 ● 1 gpu ready
╭ CREATE ──────────────────────────╮  ╭ OUTPUT ────────────────────────────╮
│ ╭ Prompt required──────────────╮ │  │                                    │
│ │ a fox in a paper garden      │ │  │          your image here           │
│ ╰──────────────────────────────╯ │  │                                    │
│ ...                              │  │                                    │
╰──────────────────────────────────╯  ╰────────────────────────────────────╯
 ✔ Image ready · 1024×1024 · seed 459280952 · 1.7s
```

## Features

- Create an account or sign in with an existing ImagineIt token.
- Multi-line prompt and negative-prompt editors (`ratatui-textarea`).
- Width, height, steps, CFG, seed, CLIP skip, and scheduler inputs.
- Live queued/running progress, then inline image rendering.
- Connected-GPU indicator and an `F3` GPU setup panel with per-account worker
  key retrieval and the exact `gpu_worker.py` command.
- `F4` Kaggle GPU panel: paste a `KGAT_…` API token, create a Kaggle profile,
  start/stop the ImagineIt notebook, and delete profiles.
- Kitty graphics protocol via `ratatui-image` (Sixel, iTerm2, and halfblocks
  fallbacks are available too).
- Saves the decrypted JPEG with `F2`.
- Keeps the account token and P-256 private key in a local, owner-only file.
- Uses the same key escrow as the web client: the token can restore the
  private key on another machine, and the server cannot decrypt stored prompts
  or images.

## Requirements

- Rust 1.88 or newer.
- An ImagineIt v3 server (defaults to `http://localhost:4600`).
- A GPU worker connected to the server for the same account. Jobs stay queued
  until one is available; the TUI can fetch the worker key for you (see
  [GPU worker setup](#gpu-worker-setup)). An account-bound worker only receives
  jobs belonging to that account.
- Kitty 0.28+ for Kitty graphics. Other terminals can use Sixel or iTerm2, or
  fall back to Unicode halfblocks.

## Build and run

```bash
cd tui
cargo run --release
```

Or install the binary:

```bash
cargo install --path . --locked
imagineit-tui --url http://localhost:4600
```

Point it at a different server with a flag or environment variable:

```bash
cargo run --release -- --url https://imagineit.example.com
# or
IMAGINEIT_URL=https://imagineit.example.com cargo run --release
```

The hosted deployment at `https://imagineit.kitsui.co` exposes the API through
Caddy's `/ui-api` bridge. Pass either the bare hostname or the bridge path —
the client probes `/api/v3/status` at startup and adds `/ui-api` automatically
when the bare API is gated:

```bash
imagineit-tui --url https://imagineit.kitsui.co
imagineit-tui --url https://imagineit.kitsui.co/ui-api
```

If graphics detection misbehaves (for example through a terminal multiplexer),
force a protocol:

```bash
cargo run --release -- --protocol kitty
```

`--protocol` accepts `auto`, `kitty`, `sixel`, `iterm2`, and `halfblocks`.
The detected protocol is shown in the header.

## Account and key storage

The first launch shows a small welcome screen:

- **Create a new account** generates a P-256 key pair, creates the account, and
  wraps the private key with the returned token before escrowing it on the
  server. The token is shown once (and saved locally).
- **Sign in with an existing token** restores the wrapped private key from the
  server and saves it locally.

The account file defaults to
`$XDG_CONFIG_HOME/imagineit-tui/account.json` (usually
`~/.config/imagineit-tui/account.json`) and is created with mode `0600`.
Override it with `--config PATH` or `IMAGINEIT_CONFIG`. Losing the file means
losing the local key; the same token can restore it from escrow as long as the
account was created by a client that escrowed the key.

Generated images are decrypted in memory and rendered locally. `F2` writes the
original JPEG to `~/Pictures/imagineit/`.

## GPU worker setup

A generation job is only dispatched to a GPU worker authenticated with a key
belonging to the same account. If the header shows `no gpu · f3 setup` or a job
stays queued, set up a worker:

1. Press `F3` to open the GPU Workers panel.
2. Press `k`. The TUI calls `GET /api/v3/workers/keys/current` (creating a key if
   necessary) and saves it as
   `~/.config/imagineit-tui/workers/<worker-id>.worker.json` with mode `0600`.
3. Copy that file to the GPU machine. If the TUI is running on the GPU machine,
   use the path shown directly.
4. On the GPU host, from an ImagineIt checkout:

   ```bash
   pip install -e '.[gpu]'
   python gpu_worker.py \
     --app-url wss://imagineit.kitsui.co/internal/v1/workers/ws \
     --key-file /path/to/<worker-id>.worker.json \
     --gpus auto
   ```

   The panel shows the exact URL and key path for the server you are connected
   to. The worker opens an outbound WebSocket, so it does not need an inbound
   port.

The header polls `/api/v3/workers` every few seconds. Once the worker appears as
`ready`, new jobs are dispatched to it. Press `r` in the panel to refresh
immediately.

### Kaggle

ImagineIt can also start its training notebook on a Kaggle GPU using your own
Kaggle API token:

1. Press `F4` to open the Kaggle GPU panel.
2. Paste a Kaggle API token (`KGAT_…`). Create one at **kaggle.com → Settings
   → API → Create New Token**.
3. Press `Enter` to save it as a Kaggle GPU profile. The token is sent to the
   server and stored encrypted; only the resulting profile is kept locally in
   the TUI.
4. Press `F5` to start the ImagineIt notebook. This can take up to two minutes.
   Once the Kaggle worker connects, the header changes to `● 1 gpu ready`.
5. Press `F6` to ask the Kaggle worker to stop. Press `F7` twice to delete the
   active profile.

The server injects that account's worker key into the private notebook, so
only your jobs are sent to the Kaggle worker. Profiles can also be managed from
the web UI; the TUI lists the ones returned by the server. These controls use
the server's `/api/v3/gpu/profiles` endpoints.

## Keys

| Key | Action |
| --- | --- |
| `Tab` / `Shift+Tab` | Move between fields |
| `↑` / `↓` | Nudge a number field; move lines inside a prompt |
| `Enter` | Newline in prompts, next field in single-line fields |
| `Ctrl+G` / `F5` / `Ctrl+Enter` | Generate |
| `F2` | Save the current image |
| `F3` | GPU workers and worker-key setup |
| `F4` | Kaggle GPU profiles and API key |
| `F1` / `Esc` | Open/close help |
| `Ctrl+Q` / `Ctrl+C` | Quit |

In the GPU panel, `k` fetches/saves a worker key, `r` refreshes worker status,
and `Esc` closes the panel.

In the Kaggle panel, `Enter` saves a new profile, `F5` starts it, `F6` stops
it, `F7` deletes it (press twice), and `Esc` closes the panel.

In the scheduler field, use `←`/`→` (or `↑`/`↓`) to cycle algorithms.

## Development

```bash
cargo test
cargo clippy --all-targets
cargo fmt
```

The crypto tests include fixtures produced by the server's Python `e2e.py`
implementation and by a web-client-compatible key escrow, so the envelope and
wrapped-key formats are checked against the real clients rather than only
round-tripping in Rust.
