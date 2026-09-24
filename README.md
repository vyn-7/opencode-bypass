# opencode-bypass

An OpenAI-shaped SSE proxy that tunnels OpenCode free-tier models
(`big-pickle`, `mimo-v2.6-flash-free`, `nemotron-3-ultra-free`, …) through the
local OpenCode CLI — so Hermes Agent and any OpenAI-compatible client can
use them with full conversation memory and agentic tool calling.

Works on **Linux, macOS, and Windows**.

## Why it exists

OpenCode locks its free tier behind local session identity — bare HTTP
requests are rejected (403). The only authenticated transport is the
OpenCode runtime. This proxy translates OpenAI's SSE wire format into
native backend sessions:

```
Hermes  -->  proxy (:18788)  -->  opencode serve  -->  zen relay
              OpenAI SSE            native sessions (delta prompts)
                    \-> opencode run (fallback when serve is down)
```

Two transports, one contract: the client's `messages` array is always
authoritative — the proxy never truncates, summarizes, or reorders it.

## Requirements

- Python 3.10+
- OpenCode CLI installed and authenticated — verify with:
  `opencode run --model opencode/big-pickle "hi"`
- Linux/macOS: `./install.sh` · Windows: `.\install.ps1`
  (both create the venv and install `aiohttp` themselves)

## Quickstart

### Linux / macOS

```bash
git clone <this-repo> && cd opencode-bypass
./install.sh
```

### Windows (PowerShell)

```powershell
git clone <this-repo>; cd opencode-bypass
Set-ExecutionPolicy -Scope Process Bypass -Force   # if scripts are blocked
.\install.ps1
```

Both commands: check prerequisites, build `.venv`, install the autostart
hook, start the proxy, and verify `/health`. Re-run only to change ports or
uninstall.

Check it works:

```bash
./test_proxy.sh          # Linux/macOS
```
```powershell
.\test_proxy.ps1         # Windows
```

```bash
curl -N -X POST http://127.0.0.1:18788/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

## Autostart

| Platform | Method | Manage |
|---|---|---|
| Linux | systemd user service (preferred), cron `@reboot` fallback | `systemctl --user status opencode-proxy.service` |
| Windows | launcher `.cmd` in the user Startup folder (no admin) | delete `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\opencode-bypass-autostart.cmd` or `.\install.ps1 -Uninstall` |

Useful commands:

```bash
# Linux
systemctl --user restart opencode-proxy.service
journalctl --user -u opencode-proxy.service -f
./install.sh --uninstall
./install.sh --port 18789
./run.sh start|stop|status|restart|logs
```

```powershell
# Windows
.\run.ps1 start -Port 18789
.\run.ps1 stop
.\run.ps1 status
.\run.ps1 logs
.\install.ps1 -Uninstall
```

## Using it with Hermes Agent

Register the proxy as a custom provider in Hermes `config.yaml`:

```yaml
model:
  default: big-pickle
  provider: custom
  base_url: http://127.0.0.1:18788/v1
  api_key: none   # the proxy ignores auth; OpenCode identity comes from the CLI
```

Make sure that provider entry has streaming enabled (`stream: true`) so
Hermes uses the SSE path instead of downgrading to blocking requests.

### Agentic tool calling

Standard OpenAI `tools` / `tool_choice` / `tool_calls` work end-to-end:

```json
POST /v1/chat/completions
{
  "model": "big-pickle",
  "messages": [{"role": "user", "content": "list files"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "list_files",
      "description": "List files in a directory",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
    }
  }],
  "tool_choice": "auto"
}
```

- Blocking responses return `finish_reason: "tool_calls"` with a standard
  `message.tool_calls[]` array.
- Streaming sends `delta.tool_calls[]` chunks before the final
  `finish_reason: "tool_calls"` + `[DONE]`.
- `tool_choice: "none"` strips tools (plain completion).
- The backend never receives a prompt-body `tools` map (that would trip the
  free-tier 403). Instead the proxy injects a `[client tools]` prompt
  section and parses the model's `[tool_call]…[/tool_call]` blocks into
  OpenAI `tool_calls`.
- A request-level hash of the effective tools + `tool_choice` rides on the
  digest registry entry: changing the tool contract mid-conversation forces
  a full resync instead of silently reusing a session staged for the old
  tools.

### Reasoning (thinking) translation

OpenCode streams reasoning as a `reasoning` part; the proxy maps it to
OpenAI's reasoning channel on both paths:

- **Streaming** → `delta.reasoning_content` chunks (never `delta.content`).
- **Blocking** → `message.reasoning_content` on the assistant message.
- Reasoning is never persisted into the digest/memory layer (`last_reply`
  stores the visible answer only) and never appears inside `content`.
- Per-part state survives the documented OpenCode race where
  `message.part.delta` can arrive *before* `message.part.updated`
  (opencode#26924), and snapshot/delta overlap is deduplicated.
- Reasoning never terminates a stream by itself: the final
  `finish_reason` still reflects content or tool_calls semantics.
- The `run` fallback always passes `--thinking` so the CLI emits reasoning
  parts (without the flag they are dropped). Note: `run --format json`
  reports complete parts only (no token-level deltas) — opencode#38638 —
  so the fallback path batches reasoning/text per part while `serve`
  streams live deltas.

### Error semantics

- Failures **before** the HTTP 200 is sent surface as a real error
  response (HTTP 502 JSON `{"error": ...}`) so Hermes' retry/fallback
  logic sees them.
- Failures **mid-stream** (headers already sent) emit one structured
  `data: {"error": {"message": ..., "type": "server_error"}}` event and
  then `[DONE]` — the proxy never fabricates assistant content or a
  `finish_reason` after a failure (a fake `stop` would be treated as a
  successful provider answer).

## How memory works (no goldfish, no second database)

The proxy is a **protocol translator, not a memory system**. Hermes sends
the full `messages` array every turn; the proxy keeps only an in-memory
**digest-prefix registry** (sha256 per message, LRU 64) that maps a
conversation to a backend session id — cursors, never a transcript copy:

1. **First turn** → new `opencode serve` session, full authoritative
   transcript flattened into one labeled prompt (plus `[client tools]`
   contract when tools are present).
2. **Append-only turns** → *deduped delta* prompt: only genuinely new tail
   state is sent; the backend already holds the prefix. Assistant
   `reasoning_content` / visible-content echoes that equal the stored
   `last_reasoning` / `last_reply` cursors are stripped (the backend holds
   that reasoning + tool-call text natively) — only the minimal
   `[assistant tool calls]` replay with verbatim OpenAI call IDs plus the
   new `[tool result: …] (call …)` block is sent. The `[client tools]`
   catalog is NOT resent when its `tools_signature` is unchanged (it lives
   in the backend prefix); a changed signature forces a full resync with
   the new contract. Provider prompt caching keeps input tokens flat.
3. **History compression / edit / rollback** (digest divergence), proxy
   restart (registry loss), missing/tainted backend session, or tools
   change → fresh session + full authoritative replay preserving
   `reasoning_content`, `tool_calls`, `tool_call_id`s, tool results, and
   ordering. Correctness always wins over tokens.
4. **Serve unavailable** → automatic fallback to `opencode run` with
   session continuation (`run -s SID`), same digest logic, prompts split at
   argv space boundaries (byte-exact under `run`'s single-space join).

Agentic details (pinned by `tests/test_context_fidelity.py`):

1. **Tool calls are serialized, not dropped — and actually consumed.**
   An assistant turn flattens to `[assistant tool calls]` with each
   `- name(args) (call call_…)` line (IDs preserved verbatim, never
   regenerated; ordering preserved). Tool results use an unambiguous
   wrapper so the model can distinguish result from instruction:
   `[tool result: read_file] (call call_123)` + `<result>` … `</result>`
   (raw output byte-preserved: JSON/JS/HTML/CSS/terminal/stack/logs/
   Markdown; a literal `</result>` inside payload is escaped as
   `<\/result>`). Deterministic mock integration tests prove the final
   answer is *derived from* the tool value (`UNIQUE_TOOL_VALUE_94721`,
   `ORBIT-731`, `ALPHA/BETA/GAMMA` chain) — not merely present in logs.
2. **Nothing is ever truncated.** There is no char budget, no trimming, no
   adapter-generated summary. 1 KB / 50 KB / 300 KB tool results reach the
   model in full on the continuation path (pinned by tests). Inbound body
   limit is 64 MB.
3. **Both API paths are served** (`/v1/chat/completions` and
   `/chat/completions`), and every SSE stream opens with a role chunk and
   closes with `finish_reason` (`tool_calls` vs `stop`) + `[DONE]` —
   `reasoning_content` / `content` / `tool_calls` never mixed; provider
   errors never fabricated as `stop`.
4. **Images are NOT forwarded.** Multimodal `content` parts of type
   `image_url` / `input_image` are replaced with the explicit placeholder
   `[attached image omitted]` (text parts preserved verbatim). This is
   intentional compatibility behavior — the proxy never invents fake image
   descriptions. Text-only behavior is pinned by regression tests.
5. **Debug context trace (opt-in).** `OPENCODE_PROXY_DEBUG=1` logs a safe
   per-turn trace: request (counts/roles/tools/reasoning-presence/sizes/
   hashes), session (sid-short/digest-match/delta-vs-resync/tools-sig),
   outbound (replay-vs-delta/result+call counts/reasoning+tools inclusion/
   prompt hash), inbound (reasoning/text/tool-call counts/finish). Only
   lengths + short hashes by default — never API keys, auth, cookies, or
   full bodies. `OPENCODE_PROXY_VERBOSE_PAYLOADS=1` adds a bounded preview
   for controlled local debugging.

Memory + tool-bridge + reasoning behavior is pinned by
`tests/test_memory.py` (unit), `tests/test_stream.py` (wire-level SSE
with a mocked EventHub — reasoning channels, race buffering, structured
mid-stream errors), `tests/test_tool_loop.py` (tool protocol), and
`tests/test_context_fidelity.py` (end-to-end consumption, multi-tool,
IDs, dedup, resync, restart, large results, trace, optional live model) —
stdlib `unittest`, no network or CLI needed:
`.venv/bin/python -m unittest discover -s tests`, plus live turn-2 recall
in `./test_proxy.sh` / `.\test_proxy.ps1`. Optional live-model proof:
`OPENCODE_LIVE_TEST=1 OPENCODE_PROXY_URL=… .venv/bin/python -m unittest
tests.test_context_fidelity.TestLiveBackend -v`.

### Free-tier constraints (measured — do not "optimize" into these)

The zen relay returns **403 `FreeTierError`** for anything that looks like
a restricted pipeline. All of these were verified and must stay enabled:

| Attempt | Result |
|---|---|
| Bare HTTP to the zen relay | 403 "only from within OpenCode" |
| Prompt-body `tools` map (any form) | 403 |
| Session permission deny on any tool | 403 |
| Tools-disabled agent | 403 |
| OpenAI `tools` array passed through to serve prompt API | 400 `Expected object \| null` |

So the backend tool pipeline **stays on** (and tools are bridged at the
protocol layer, see above), and the proxy runs `opencode serve --pure` to
drop plugin/MCP schemas (~6k tokens/call).

## Free models

`GET /v1/models` queries the live serve catalog
(`/config/providers` → `opencode` provider, `cost.input == 0` and
`cost.output == 0`, non-deprecated) with a 5-minute cache. Static fallback
when serve is down:

- `big-pickle`
- `ling-3.0-flash-fin-free`
- `mimo-v2.6-flash-free`
- `muse-spark-1.2-contributor-free`
- `muse-spark-1.3-contributor-free`
- `nemotron-3-ultra-free`
- `nemotron-3.5-lightning-free`

Any model name still passes through (`big-pickle` → `opencode/big-pickle`;
already-qualified names untouched), so paid/other-provider models work
without updating the proxy.

## Configuration

`opencode_proxy.py` flags (all also settable for manual runs via
`run.sh` / `run.ps1`):

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--port` | `OPENCODE_PROXY_PORT` | `18788` | listen port (localhost) |
| `--host` | `OPENCODE_PROXY_HOST` | `127.0.0.1` | bind address — keep loopback |
| `--cli` | `OPENCODE_CLI` | auto-detect | `opencode` on PATH, else well-known per-OS locations |
| `--work-dir` | — | cwd | directory `serve`/`run` execute in |
| `--serve-port` | `OPENCODE_SERVE_PORT` | `18790` | port for the managed `opencode serve` instance |
| `--serve-url` | `OPENCODE_SERVE_URL` | — | use an existing serve at this URL (never spawned) |
| `--serve-keep-plugins` | — | off | load MCP/plugin schemas too (default: `--pure`, saves ~6k tokens/call) |
| `--agent` | `OPENCODE_AGENT` | `bypass-lite` | opencode agent for prompts; `''` = built-in default |
| `--default-model` | — | `big-pickle` | model when the request names none |

If `OPENCODE_SERVER_PASSWORD` is set, the proxy authenticates to serve with
HTTP Basic (`OPENCODE_SERVER_USERNAME`, default `opencode`).

`GET /health` reports status, resolved CLI path, discovered free models,
active backend (`serve:…` or `run`), agent, and registry size.

## API

- `GET /`, `GET /health` → `{"status":"ok", "backend":"serve:…|run", ...}`
- `GET /v1/models` → OpenAI model list (live free-tier catalog)
- `POST /v1/chat/completions`, `POST /chat/completions` → OpenAI chat
  completion (blocking) or SSE stream when `"stream":true`;
  accepts `tools` / `tool_choice` and returns OpenAI `tool_calls`;
  reasoning arrives as `delta.reasoning_content` / `message.reasoning_content`
- Errors from the backend surface as HTTP 502 (pre-stream) or a structured
  SSE error event (mid-stream); malformed bodies get HTTP 400.
  Inbound request bodies up to 64 MB are accepted (full histories welcome).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FATAL: opencode CLI not found` | install from https://opencode.ai, ensure `opencode` is on PATH or pass `--cli` |
| `cannot bind 127.0.0.1:18788` | another copy is running — `./run.sh status` / `.\run.ps1 status`, or stop the autostart service |
| HTTP 502 `prompt failed` / `opencode exit N` | run the CLI directly (`opencode run --model opencode/big-pickle "hi"`) — usually expired auth; re-authenticate OpenCode |
| `/health` shows `"backend":"run"` | managed serve is down — check `opencode-serve.log`; proxy respawns it automatically on the next request |
| HTTP 503/403 from zen | free-tier gate — never send prompt-body `tools`, tool permission denies, or tools-disabled agents (see constraints table) |
| HTTP 400 `Expected object \| null` on serve | proxy should never forward OpenAI tools arrays — update the proxy |
| `(empty response)` | model returned no text deltas; check proxy logs / journal |
| Hermes 404 | point `base_url` at `http://127.0.0.1:18788/v1` (or use the `/chat/completions` path, also served) |
| Slow first turn | normal: first request spawns `opencode serve --pure` (~8s); later turns reuse it |
| `install.sh` / `install.ps1` warns about auth | log into OpenCode first, then re-run the installer |

## Project structure

```
opencode_proxy.py                the proxy (aiohttp, digest registry, serve/run backends, tools bridge, reasoning translator)
.opencode/agents/bypass-lite.md  completion-backend agent ([tool_call] protocol, tools kept for 403 gate)
tests/test_memory.py             memory/delta/wire/tools-bridge/reasoning unit tests (stdlib only)
tests/test_stream.py             wire-level SSE tests (mocked EventHub, race + error semantics)
install.sh / install.ps1         one-command install + autostart + verify (Linux / Windows)
run.sh / run.ps1                 manual run/start/stop/restart/status/logs
test_proxy.sh / test_proxy.ps1   live smoke test (health, models, stream, turn-2 recall, tools)
systemd/opencode-proxy.service.template   Linux systemd user unit template
requirements.txt / pyproject.toml         deps + `opencode-bypass` console script
```

## License

MIT — see [LICENSE](LICENSE).
