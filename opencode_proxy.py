#!/usr/bin/env python3
"""OpenAI-shaped SSE-streaming proxy tunneling OpenCode free-tier models.

OpenCode's free tier rejects bare HTTP (403 "only from within OpenCode") and
also rejects *restricted* agent configurations (prompt-body ``tools``, session
permission denies, tools-disabled agents) with the same gate — the free-tier
attestation rides on the normal tool-enabled agent pipeline. The working
transports are therefore:

  primary:  ``opencode serve`` HTTP API (long-lived, native sessions)
  fallback: ``opencode run`` subprocess (per-call, session-continued via -s)

Architecture (protocol translation only — not a second memory system):

    Hermes  -->  proxy (:18788)  -->  opencode serve  -->  zen relay
              OpenAI SSE            native sessions

The incoming OpenAI ``messages`` array stays fully authoritative: every role,
order, assistant tool call and tool result is preserved; nothing is ever
truncated or summarized by the adapter. Efficiency comes from using the
backend's own session state instead of replaying history every call:

  * a digest-prefix registry (sha256 per message — cursors, not a transcript)
    maps an OpenAI conversation to an ``opencode serve`` session id;
  * append-only turns send only the *delta* (new messages) to the existing
    session — the backend already holds the prefix;
  * if Hermes compresses/edits history (digest divergence), the adapter
    resyncs: fresh session + full authoritative replay;
  * the previous assistant echo at the head of the delta is skipped when it
    textually equals our last reply (the backend already has that turn).

Agentic tool calling (Hermes / any OpenAI client):

  The serve prompt API rejects OpenAI ``tools`` arrays (400). The proxy
  therefore bridges tools at the protocol level: client ``tools`` /
  ``tool_choice`` are never forwarded as prompt-body ``tools`` (that would
  also trip the free-tier 403 gate). Instead they are rendered into a
  ``[client tools]`` prompt section, the agent instructs the model to emit
  ``[tool_call]\\n{json}\\n[/tool_call]`` blocks, and the proxy converts
  those blocks into standard OpenAI ``tool_calls`` (blocking + streaming).

Free-tier constraints discovered empirically (do not "optimize" into these):
  * prompt body ``tools`` map            -> 403 FreeTierError
  * session permission deny on tools     -> 403 FreeTierError
  * tools-disabled agent (tools: "*": false) -> 403 FreeTierError
So backend tools stay enabled; a minimal ``bypass-lite`` agent prompt
(.opencode/agents/bypass-lite.md) reframes the backend as a completion
channel and documents the ``[tool_call]`` protocol. ``--pure`` serve drops
plugin/MCP schemas (~6k tokens/call).

Run-fallback prompts travel on argv (MAX_ARG_STRLEN 128 KiB); oversized
transcripts are split at space boundaries across multiple argv entries —
``opencode run`` joins positionals with a single space, so the reassembled
prompt is byte-identical.

Cross-platform (Linux / macOS / Windows): CLI discovery covers PATH,
``.exe``/``.cmd``, and per-OS well-known locations; ``.cmd``/``.bat``
wrappers spawn via ``cmd.exe /c``; detached serve uses
``DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP`` on Windows and
``start_new_session`` elsewhere; shutdown kills the process tree
(``taskkill /T /F`` on Windows).
"""

import argparse
import asyncio
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

IS_WINDOWS = os.name == "nt"

# Static fallback for GET /v1/models when serve is unreachable. The live
# list is discovered from serve's /config/providers (opencode provider,
# cost.input == 0 and cost.output == 0) and cached for MODELS_CACHE_TTL.
KNOWN_MODELS = [
    "big-pickle",
    "ling-3.0-flash-fin-free",
    "mimo-v2.6-flash-free",
    "muse-spark-1.2-contributor-free",
    "muse-spark-1.3-contributor-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
]

PER_READ_TIMEOUT = 180       # seconds without a serve event / CLI output line
SERVE_PROMPT_TIMEOUT = 600   # seconds for one completion (blocking call)
SERVE_HEALTH_CACHE = 1.0     # seconds a health probe result stays valid
MODELS_CACHE_TTL = 300.0     # seconds a discovered free-model list stays valid
MAX_BODY_BYTES = 64 * 1024 * 1024   # inbound Hermes history: no char limits
MAX_ARG_CHARS = 120_000      # per-argv entry for `run` (kernel limit 131072)
REGISTRY_MAX = 64            # conversation cursors kept (LRU)

# Windows process creation flags (numeric so import works on POSIX).
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200

TOOL_OPEN = "[tool_call]"
TOOL_CLOSE = "[/tool_call]"

_models_cache: dict = {"at": 0.0, "ids": None}


def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[proxy {ts}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Cross-platform process helpers
# --------------------------------------------------------------------------

def find_cli(explicit: str | None = None) -> str:
    """Resolve the opencode binary: --cli flag > PATH > well-known locations."""
    candidates: list[str] = []
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    which = shutil.which("opencode")
    if which:
        candidates.append(which)
    if IS_WINDOWS:
        candidates += [
            r"~\.opencode\bin\opencode.exe",
            r"~\.opencode\bin\opencode.cmd",
            r"~\.local\bin\opencode.exe",
            r"~\scoop\shims\opencode.exe",
            r"~\scoop\shims\opencode.cmd",
            r"~\AppData\Local\Microsoft\WinGet\Links\opencode.exe",
            r"~\AppData\Roaming\npm\opencode.cmd",
            r"~\AppData\Local\Programs\opencode\opencode.exe",
        ]
    else:
        candidates += [
            "~/.opencode/bin/opencode",
            "~/.local/bin/opencode",
            "/usr/local/bin/opencode",
            "/usr/bin/opencode",
        ]
    for cand in candidates:
        expanded = os.path.expanduser(cand)
        if not os.path.isfile(expanded):
            continue
        if IS_WINDOWS or os.access(expanded, os.X_OK):
            return expanded
    raise FileNotFoundError(
        "opencode CLI not found. Install it (https://opencode.ai) or pass --cli. "
        f"Tried: {candidates}"
    )


def cli_argv(cli: str, *args: str) -> list[str]:
    """Build argv for the opencode CLI; wrap ``.cmd``/``.bat`` on Windows."""
    if IS_WINDOWS and cli.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", cli, *args]
    return [cli, *args]


def spawn_detached_kwargs() -> dict:
    """Kwargs that put a child in its own session/process group."""
    if IS_WINDOWS:
        return {"creationflags": _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def kill_process_tree(proc: asyncio.subprocess.Process | None) -> None:
    """Terminate a child and its descendants (Windows: taskkill /T /F)."""
    if proc is None or proc.returncode is not None:
        return
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True, timeout=10, check=False,
            )
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            pass
        return
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)
    except Exception:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def normalize_model(model: str) -> str:
    """Accept ``big-pickle`` or ``opencode/big-pickle``; never double the prefix."""
    model = (model or "").strip() or "big-pickle"
    if "/" in model:
        return model
    return f"opencode/{model}"


def split_model(model: str) -> tuple[str, str]:
    """``opencode/big-pickle`` -> (``opencode``, ``big-pickle``)."""
    qualified = normalize_model(model)
    provider, _, model_id = qualified.partition("/")
    return provider, model_id or "big-pickle"


# --------------------------------------------------------------------------
# Transcript flattening (authoritative, never truncating)
# --------------------------------------------------------------------------

def _text_of(content) -> str:
    """Coerce message content to plain text (None, str, or multipart list)."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if isinstance(p.get("text"), str) and p["text"]:
                parts.append(p["text"])
            elif "image_url" in p or p.get("type") in ("image_url", "input_image"):
                parts.append("[attached image omitted]")
        return " ".join(parts)
    return str(content)


def _render_tool_calls(tool_calls) -> str:
    """Render assistant tool_calls so the model remembers what it did.

    Without this, an agentic turn looks like ``user -> (empty assistant) ->
    tool result`` and the model loses which action produced the result.
    """
    rendered = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name", tc.get("name", "unknown-tool"))
        args = fn.get("arguments", tc.get("arguments", ""))
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        rendered.append(f"- {name}({args})")
    if not rendered:
        return ""
    return "[assistant tool calls]\n" + "\n".join(rendered)


def flatten_history(msgs) -> str:
    """Flatten the conversation into one labeled transcript. No size budget.

    Every turn is preserved in order: system (+developer), user, assistant
    (content AND tool calls), tool results (with tool name / call id when
    provided). Empty turns carry no information and are skipped. Nothing is
    ever truncated — Hermes' ``messages`` array is authoritative.
    """
    blocks: list[tuple[bool, str]] = []  # (is_system, text)
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        text = _text_of(m.get("content", ""))
        if role in ("system", "developer"):
            if text:
                blocks.append((True, f"[system]\n{text}"))
        elif role == "assistant":
            calls = _render_tool_calls(m.get("tool_calls"))
            if text and calls:
                blocks.append((False, f"[assistant]\n{text}\n{calls}"))
            elif text:
                blocks.append((False, f"[assistant]\n{text}"))
            elif calls:
                blocks.append((False, f"[assistant]\n{calls}"))
        elif role == "tool":
            name = m.get("name", "")
            tcid = m.get("tool_call_id", "")
            label = f"[tool result: {name}]" if name else "[tool result]"
            if tcid:
                label += f" (call {tcid})"
            if text:
                blocks.append((False, f"{label}\n{text}"))
        else:  # user and anything unknown
            if text:
                blocks.append((False, f"[user]\n{text}"))
    system = [b for s, b in blocks if s]
    rest = [b for s, b in blocks if not s]
    return "\n\n".join(system + rest).strip()


def flatten_tail(msgs) -> str:
    """Flatten a delta slice; later slices must not repeat ``[system]`` labels.

    After the first (full) prompt the system message lives in the backend's
    stored prefix, so only the *first* block of a full replay carries the
    system label. Delta slices normally contain no system message at all —
    this helper only guards the rare case where a divergence replays a slice
    that still starts with system.
    """
    return flatten_history(msgs)


# --------------------------------------------------------------------------
# Digest registry (cursors, not a transcript database)
# --------------------------------------------------------------------------

def msg_digest(m: dict) -> str:
    blob = json.dumps(m, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def digests_of(msgs) -> list[str]:
    return [msg_digest(m) for m in (msgs or []) if isinstance(m, dict)]


def drop_echoed_assistant(tail_msgs, last_reply: str):
    """Drop a leading assistant echo of our previous reply (backend has it)."""
    if not tail_msgs or not last_reply:
        return tail_msgs
    head = tail_msgs[0]
    if not isinstance(head, dict) or head.get("role") != "assistant":
        return tail_msgs
    if head.get("tool_calls"):
        return tail_msgs  # never drop turns that carry tool calls
    if _text_of(head.get("content", "")).strip() == last_reply.strip():
        return tail_msgs[1:]
    return tail_msgs


class Registry:
    """Maps OpenAI conversation digests to backend sessions (LRU, in-memory).

    An entry matches only when its stored digest list is an exact prefix of
    the incoming one (pure append). Any mid-history edit/compression/rollback
    is a divergence -> full authoritative resync on a fresh session. Lost
    state (proxy restart) only costs one full replay.
    """

    def __init__(self, max_entries: int = REGISTRY_MAX):
        self.max_entries = max_entries
        self._entries: list[dict] = []  # ordered oldest -> newest by last_used

    def __len__(self) -> int:
        return len(self._entries)

    def match(self, digests: list[str], backend_kind: str) -> dict | None:
        best: dict | None = None
        best_len = -1
        for e in self._entries:
            if e["backend"] != backend_kind:
                continue
            stored = e["digests"]
            if len(stored) <= best_len:
                continue
            if digests[: len(stored)] == stored:
                best, best_len = e, len(stored)
        if best is not None:
            self._entries.remove(best)
            self._entries.append(best)  # touch LRU
        return best

    def record(self, digests: list[str], backend_kind: str, sid: str,
               last_reply: str) -> dict:
        entry = {
            "digests": list(digests),
            "backend": backend_kind,
            "sid": sid,
            "last_reply": last_reply,
            "used": time.time(),
        }
        self._entries = [e for e in self._entries
                         if not (e["backend"] == backend_kind and e["sid"] == sid)]
        self._entries.append(entry)
        while len(self._entries) > self.max_entries:
            self._entries.pop(0)
        return entry

    def peek(self, digests: list[str], backend_kind: str):
        """Match without LRU touch — used for logging/tests."""
        for e in self._entries:
            if e["backend"] == backend_kind and \
               digests[: len(e["digests"])] == e["digests"]:
                return e
        return None


def plan_prompt(msgs, entry: dict | None) -> tuple[str, str, dict]:
    """Decide delta-vs-resync for one request.

    Returns (prompt_text, mode, record_payload). ``record_payload`` always
    carries the full incoming digest list — the caller records it only
    after a successful completion (retries then replay the same tail).
    """
    digs = digests_of(msgs)
    if entry is None:
        return flatten_history(msgs), "resync", {"digests": digs}
    stored = entry["digests"]
    if digs[: len(stored)] != stored:
        return flatten_history(msgs), "resync", {"digests": digs}
    tail = list(msgs[len(stored):])
    tail = drop_echoed_assistant(tail, entry.get("last_reply", ""))
    if not tail:
        # Degenerate repeat (client re-sent an identical conversation):
        # regenerate from the last message rather than emitting nothing.
        tail = list(msgs[-1:])
    return flatten_tail(tail), "delta", {"digests": digs}


# --------------------------------------------------------------------------
# OpenAI tools <-> [tool_call] protocol bridge
# --------------------------------------------------------------------------

def effective_tools(tools, tool_choice=None) -> list[dict]:
    """Filter a client ``tools`` payload down to usable function specs.

    ``tool_choice == "none"`` drops tools entirely. Malformed entries are
    ignored — the request must never 400 just because of tools shape.
    """
    if tool_choice == "none":
        return []
    if not isinstance(tools, list):
        return []
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type", "function") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append(t)
    return out


def format_client_tools(tools: list[dict], tool_choice=None) -> str:
    """Render OpenAI function tools as a prompt section for the model."""
    if not tools:
        return ""
    lines = ["[client tools]"]
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name", "")
        desc = (fn.get("description") or "").strip()
        params = fn.get("parameters") or fn.get("input_schema") or {}
        try:
            params_s = json.dumps(params, ensure_ascii=False)
        except (TypeError, ValueError):
            params_s = "{}"
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
        lines.append(f"  parameters: {params_s}")
    lines.append("")
    lines.append(
        "To invoke a tool, emit one fenced block per call and no other markup "
        "around it. arguments must be a JSON object matching that tool's "
        "parameters schema:"
    )
    lines.append(TOOL_OPEN)
    lines.append('{"name": "tool_name", "arguments": { /* ... */ }}')
    lines.append(TOOL_CLOSE)
    lines.append(
        "Rules: one block per tool call; you may include normal text before "
        "or after blocks; never invent tools that are not listed above; "
        "never call the backend's built-in tool runner — the proxy translates "
        "these blocks into the client's tool_calls format."
    )
    if tool_choice == "required":
        lines.append("You MUST emit at least one tool_call block this turn.")
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        if fn.get("name"):
            lines.append(f"You MUST call the tool {fn['name']!r} this turn.")
    return "\n".join(lines)


def _parse_tool_call_body(body: str) -> dict | None:
    """Parse the JSON object inside a [tool_call] block into an OpenAI call."""
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        try:
            obj = json.loads(body.strip().rstrip(","))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or (obj.get("function") or {}).get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        arg_str = args
    else:
        try:
            arg_str = json.dumps(args if args is not None else {},
                                 ensure_ascii=False)
        except (TypeError, ValueError):
            arg_str = "{}"
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arg_str},
    }


def extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Strip complete [tool_call] blocks (parsed) from ``text``.

    Unclosed markers are flushed into content as-is (final malformed output
    stays visible). Malformed but complete markers are kept verbatim.
    """
    if not text or TOOL_OPEN not in text:
        return text or "", []
    calls: list[dict] = []
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find(TOOL_OPEN, i)
        if j < 0:
            parts.append(text[i:])
            break
        parts.append(text[i:j])
        k = text.find(TOOL_CLOSE, j + len(TOOL_OPEN))
        if k < 0:
            parts.append(text[j:])  # unclosed: keep visible
            break
        body = text[j + len(TOOL_OPEN):k]
        call = _parse_tool_call_body(body)
        if call is not None:
            calls.append(call)
        else:
            parts.append(text[j:k + len(TOOL_CLOSE)])
        i = k + len(TOOL_CLOSE)
    return "".join(parts), calls


def streaming_view(text: str) -> tuple[str, list[dict]]:
    """Content safe to emit now + complete calls so far.

    Unlike :func:`extract_tool_calls`, an unclosed marker (or a partial
    ``[tool_call]`` prefix at the end of the buffer) is *held back* — those
    bytes may still turn into a complete block and must never leak into the
    client's content stream.
    """
    if not text:
        return "", []
    calls: list[dict] = []
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = text.find(TOOL_OPEN, i)
        if j < 0:
            rest = text[i:]
            max_h = min(len(TOOL_OPEN) - 1, len(rest))
            hold = 0
            for k in range(max_h, 0, -1):
                if TOOL_OPEN.startswith(rest[len(rest) - k:]):
                    hold = k
                    break
            parts.append(rest[: len(rest) - hold] if hold else rest)
            break
        parts.append(text[i:j])
        k = text.find(TOOL_CLOSE, j + len(TOOL_OPEN))
        if k < 0:
            break  # hold from OPEN through end
        body = text[j + len(TOOL_OPEN):k]
        call = _parse_tool_call_body(body)
        if call is not None:
            calls.append(call)
        else:
            parts.append(text[j:k + len(TOOL_CLOSE)])
        i = k + len(TOOL_CLOSE)
    return "".join(parts), calls


class ToolCallStreamParser:
    """Feed raw deltas; emit only client-safe content; collect finished calls."""

    def __init__(self) -> None:
        self.raw = ""
        self.emitted = 0  # length of clean content already sent

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self.raw += delta
        clean, _ = streaming_view(self.raw)
        if len(clean) <= self.emitted:
            return ""
        out = clean[self.emitted:]
        self.emitted = len(clean)
        return out

    def finish(self, final_text: str | None = None) -> tuple[str, list[dict]]:
        """Flush held content (if final) and return authoritative calls."""
        if final_text is not None:
            self.raw = final_text
        clean, calls = extract_tool_calls(self.raw)
        out = ""
        if len(clean) > self.emitted:
            out = clean[self.emitted:]
            self.emitted = len(clean)
        return out, calls


# --------------------------------------------------------------------------
# argv chunking for the `run` fallback (byte-exact space-boundary splits)
# --------------------------------------------------------------------------

def chunk_argv(text: str, limit: int = MAX_ARG_CHARS) -> list[str]:
    """Split text into argv-safe pieces; `run` rejoins them with one space.

    Splits *at* a space and omits it from both sides so the joiner's single
    space reproduces the original byte-for-byte. Windows without any space
    (pathological blobs) fall back to a hard split (one boundary char becomes
    a space — logged by the caller if it happens).
    """
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + limit, n)
        if j < n:
            k = text.rfind(" ", i, j)
            if k > i:
                parts.append(text[i:k])
                i = k + 1
                continue
            parts.append(text[i:j])  # no space in window: hard split
            i = j
        else:
            parts.append(text[i:j])
            i = j
    return [p for p in parts if p]


def chunk(completion_id, model, delta=None, finish=None, tool_calls=None):
    d: dict = {}
    if delta is not None:
        d["delta"] = delta if isinstance(delta, dict) else {"content": delta}
    if tool_calls is not None:
        d["delta"] = {**(d.get("delta") or {}), "tool_calls": tool_calls}
    if finish:
        d["finish_reason"] = finish
    return json.dumps({
        "id": completion_id, "object": "chat.completion.chunk",
        "created": 0, "model": model,
        "choices": [{"index": 0, **d}],
    })


# --------------------------------------------------------------------------
# Event hub: one shared SSE reader on serve's global /event
# --------------------------------------------------------------------------

class EventHub:
    def __init__(self, base_url: str, headers: dict):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self._subs: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._session: ClientSession | None = None

    def set_session(self, session: ClientSession) -> None:
        self._session = session

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="event-hub")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def _run(self) -> None:
        backoff = 0.5
        while self._subs:
            try:
                assert self._session is not None
                async with self._session.get(
                    f"{self.base_url}/event",
                    headers=self.headers,
                    timeout=ClientTimeout(total=None, sock_connect=10),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"/event HTTP {resp.status}")
                    backoff = 0.5
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        try:
                            evt = json.loads(line[5:])
                        except json.JSONDecodeError:
                            continue
                        for q in list(self._subs):
                            q.put_nowait(evt)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # reconnect until subscribers go away
                if not self._subs:
                    break
                log(f"event hub: {e!r}; reconnect in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)


# --------------------------------------------------------------------------
# Free-model discovery (serve /config/providers, opencode, cost == 0)
# --------------------------------------------------------------------------

def _free_ids_from_providers(data) -> list[str]:
    """Extract zero-cost, non-deprecated model ids from a providers payload."""
    providers = data.get("providers") if isinstance(data, dict) else None
    oc = None
    if isinstance(providers, dict):
        oc = providers.get("opencode")
    elif isinstance(providers, list):
        for p in providers:
            if isinstance(p, dict) and p.get("id") == "opencode":
                oc = p
                break
    if not isinstance(oc, dict):
        return []
    models = oc.get("models") or {}
    ids: list[str] = []
    if isinstance(models, dict):
        items = list(models.items())
    elif isinstance(models, list):
        items = []
        for m in models:
            if isinstance(m, dict) and m.get("id"):
                items.append((m["id"], m))
    else:
        return []
    for mid, m in items:
        if not isinstance(m, dict) or not mid:
            continue
        cost = m.get("cost") or {}
        if cost.get("input") == 0 and cost.get("output") == 0 and \
           m.get("status", "active") != "deprecated":
            ids.append(str(mid))
    return sorted(set(ids))


async def discover_free_models(http: ClientSession, base_url: str,
                               headers: dict) -> list[str]:
    """Live free-model list from serve, cached; static fallback on failure."""
    now = time.monotonic()
    cached = _models_cache.get("ids")
    if cached and now - float(_models_cache.get("at") or 0) < MODELS_CACHE_TTL:
        return list(cached)
    try:
        async with http.get(
            f"{base_url.rstrip('/')}/config/providers",
            headers=headers,
            timeout=ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            data = await resp.json(content_type=None)
    except Exception as e:
        log(f"model discovery failed ({e}); using static list")
        return list(KNOWN_MODELS)
    ids = _free_ids_from_providers(data)
    if not ids:
        log("model discovery returned no free opencode models; using static list")
        return list(KNOWN_MODELS)
    _models_cache["ids"] = ids
    _models_cache["at"] = now
    log(f"discovered {len(ids)} free models: {', '.join(ids)}")
    return list(ids)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class ServeBackend:
    kind = "serve"

    def __init__(self, cli: str, work_dir: str, port: int, pure: bool,
                 external_url: str | None, log_path: Path):
        self.cli = cli
        self.work_dir = work_dir
        self.port = port
        self.pure = pure
        self.external_url = external_url.rstrip("/") if external_url else None
        self.log_path = log_path
        self.proc: asyncio.subprocess.Process | None = None
        self.adopted = False
        self._url = self.external_url or f"http://127.0.0.1:{port}"
        self._health_at = 0.0
        self._health_ok = False
        self.http: ClientSession | None = None
        self._headers: dict = {}
        password = os.environ.get("OPENCODE_SERVER_PASSWORD")
        if password:
            import base64
            user = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            self._headers["Authorization"] = f"Basic {token}"

    @property
    def url(self) -> str:
        return self._url

    @property
    def headers(self) -> dict:
        return self._headers

    async def healthy(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._health_at < SERVE_HEALTH_CACHE:
            return self._health_ok
        self._health_at = now
        if self.proc is not None and self.proc.returncode is not None:
            log(f"serve process exited ({self.proc.returncode})")
            self.proc = None
        if self.http is None or self.http.closed:
            self._health_ok = False
            return False
        try:
            async with self.http.get(
                f"{self._url}/global/health",
                headers=self._headers,
                timeout=ClientTimeout(total=5),
            ) as resp:
                self._health_ok = resp.status == 200
        except Exception:
            self._health_ok = False
        return self._health_ok

    async def ensure(self) -> bool:
        """Adopt or spawn a healthy serve; False -> caller falls back to run."""
        if await self.healthy():
            return True
        if self.external_url:
            log(f"external serve unreachable at {self._url}")
            return False
        # Port already serves (someone else's instance)? Adopt it.
        if self.http is not None and not self.http.closed:
            try:
                async with self.http.get(
                    f"{self._url}/global/health",
                    timeout=ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        self.adopted = True
                        self._health_ok = True
                        self._health_at = time.monotonic()
                        log(f"adopted existing serve on :{self.port}")
                        return True
            except Exception:
                pass
        return await self._spawn()

    async def _spawn(self) -> bool:
        if self.port == 0:
            log("serve port is 0; cannot spawn deterministically")
            return False
        args = ["serve", "--port", str(self.port), "--hostname", "127.0.0.1"]
        if self.pure:
            args.append("--pure")
        cmd = cli_argv(self.cli, *args)
        log(f"spawning {' '.join(cmd)} (cwd={self.work_dir})")
        logf = open(self.log_path, "ab")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.work_dir,
                stdout=logf,
                stderr=logf,
                **spawn_detached_kwargs(),
            )
        finally:
            logf.close()
        for _ in range(40):
            await asyncio.sleep(0.5)
            if self.proc.returncode is not None:
                log(f"serve died on startup (rc={self.proc.returncode}); "
                    f"see {self.log_path}")
                self.proc = None
                return False
            if await self.healthy(force=True):
                log(f"serve ready on :{self.port} (pure={self.pure})")
                return True
        log("serve health timeout")
        return False

    async def shutdown(self) -> None:
        await kill_process_tree(self.proc)

    def _payload(self, prompt: str, model: str) -> dict:
        provider, model_id = split_model(model)
        body: dict = {
            "parts": [{"type": "text", "text": prompt}],
            "model": {"providerID": provider, "modelID": model_id},
        }
        if AGENT:
            body["agent"] = AGENT
        return body

    async def create_session(self, model: str) -> str:
        provider, model_id = split_model(model)
        body = {
            "title": f"bypass {_dt.datetime.now().strftime('%H:%M:%S')}",
            "model": {"providerID": provider, "id": model_id},
        }
        async with self.http.post(
            f"{self._url}/session", json=body, headers=self._headers,
            timeout=ClientTimeout(total=30),
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200 or not isinstance(data, dict) or "id" not in data:
                raise RuntimeError(f"session create failed: HTTP {resp.status} "
                                   f"{json.dumps(data)[:300]}")
            return data["id"]

    async def prompt(self, sid: str, prompt: str, model: str) -> dict:
        """Blocking prompt; returns the assistant message envelope."""
        async with self.http.post(
            f"{self._url}/session/{sid}/message",
            json=self._payload(prompt, model),
            headers=self._headers,
            timeout=ClientTimeout(total=SERVE_PROMPT_TIMEOUT),
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(f"prompt failed: HTTP {resp.status} "
                                   f"{json.dumps(data)[:300]}")
            return data


class RunBackend:
    kind = "run"

    def __init__(self, cli: str, work_dir: str):
        self.cli = cli
        self.work_dir = work_dir

    async def run(self, model: str, prompt: str, session_id: str | None,
                  on_delta=None) -> tuple[str, str | None]:
        """One `run` invocation. Returns (text, discovered_session_id).

        ``on_delta`` (optional async) receives incremental text as it arrives.
        """
        args = chunk_argv(prompt)
        hard_split = any(" " not in a and len(a) >= MAX_ARG_CHARS for a in args)
        if hard_split:
            log(f"warning: argv hard-split (no space boundary in "
                f"{MAX_ARG_CHARS} chars) — one boundary char may differ")
        cmd_args = [
            "run",
            "--format", "json",
            "--model", normalize_model(model),
            "--dir", self.work_dir,
        ]
        if AGENT:
            cmd_args += ["--agent", AGENT]
        if session_id:
            cmd_args += ["--session", session_id]
        cmd_args += args
        cmd = cli_argv(self.cli, *cmd_args)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.work_dir,
        )
        parts: list[str] = []
        sent = ""
        found_sid = session_id

        async def pump():
            nonlocal sent, found_sid
            assert proc.stdout is not None
            while True:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=PER_READ_TIMEOUT)
                if not line:
                    break
                line = line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not found_sid and obj.get("sessionID"):
                    found_sid = obj["sessionID"]
                if obj.get("type") == "text":
                    part = obj.get("part", {})
                    new = part if isinstance(part, str) else \
                        (part.get("text", obj.get("text", "")) if isinstance(part, dict)
                         else obj.get("text", ""))
                    if new and len(new) > len(sent):
                        delta = new[len(sent):]
                        parts.append(delta)
                        sent = new
                        if on_delta is not None:
                            await on_delta(delta)

        try:
            await pump()
        except Exception:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            raise
        rc = await proc.wait()
        if rc != 0:
            assert proc.stderr is not None
            err = (await proc.stderr.read()).decode("utf-8", "replace")[-500:]
            raise RuntimeError(f"opencode exit {rc}: {err}")
        return "".join(parts), found_sid

    async def shutdown(self) -> None:
        pass


# Agent presence check — set by make_app at startup.
AGENT = ""


def agent_available(agent: str, work_dir: str) -> bool:
    if not agent:
        return False
    name = agent if agent.endswith(".md") else f"{agent}.md"
    candidates = [
        Path(work_dir) / ".opencode" / "agents" / name,
        Path.home() / ".config" / "opencode" / "agents" / name,
        Path.home() / ".opencode" / "agents" / name,
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "opencode" / "agents" / name)
    return any(p.is_file() for p in candidates)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

def make_app(cli: str, work_dir: str, *,
             serve_port: int = 18790,
             serve_pure: bool = True,
             serve_url: str | None = None,
             agent: str = "bypass-lite",
             default_model: str = "big-pickle") -> web.Application:
    global AGENT
    if agent and not agent_available(agent, work_dir):
        log(f"warning: agent {agent!r} not found under {work_dir}/.opencode/agents "
            f"— falling back to the default agent")
        agent = ""
    AGENT = agent

    registry = Registry()
    serve = ServeBackend(
        cli, work_dir, serve_port, serve_pure, serve_url,
        Path(work_dir) / "opencode-serve.log",
    )
    run_backend = RunBackend(cli, work_dir)
    state: dict = {"http": None, "hub": None, "backend_kind": None}

    def ensure_http() -> ClientSession:
        if state["http"] is None or state["http"].closed:
            state["http"] = ClientSession()
            serve.http = state["http"]
            if state["hub"] is None:
                state["hub"] = EventHub(serve.url, serve.headers)
            state["hub"].set_session(state["http"])
        return state["http"]

    async def get_backend():
        """Serve when healthy, else run. Returns (backend, kind)."""
        ensure_http()
        if await serve.ensure():
            state["backend_kind"] = "serve"
            return serve, "serve"
        state["backend_kind"] = "run"
        return run_backend, "run"

    def build_prompt(msgs, entry, tools_list, tool_choice) -> tuple[str, str, dict]:
        prompt, mode, rec = plan_prompt(msgs, entry)
        if tools_list:
            block = format_client_tools(tools_list, tool_choice)
            if block:
                prompt = f"{prompt}\n\n{block}"
        return prompt, mode, rec

    async def complete(msgs, model, tools_list, tool_choice):
        """One completion. Returns (text, tokens, sid, mode, tool_calls)."""
        backend, kind = await get_backend()
        digs = digests_of(msgs)
        entry = registry.match(digs, kind)
        prompt, mode, rec = build_prompt(msgs, entry, tools_list, tool_choice)
        bridge = bool(tools_list)

        if kind == "serve":
            sid = entry["sid"] if (entry and mode == "delta") else None
            if sid is None:
                sid = await backend.create_session(model)
            full_chars = sum(len(_text_of(m.get("content", "")))
                             for m in msgs if isinstance(m, dict))
            log(f"{mode:6} {len(msgs)} msgs ({len(prompt)} chars "
                f"of {full_chars} full) -> serve {sid[-8:]}")
            envelope = await backend.prompt(sid, prompt, model)
            info = envelope.get("info", {}) or {}
            err = info.get("error")
            if err:
                msg = (err.get("data") or {}).get("message") or err.get("name") \
                    or "backend error"
                raise RuntimeError(msg)
            raw = "".join(p.get("text", "") for p in envelope.get("parts", [])
                          if p.get("type") == "text")
            if bridge:
                text, calls = extract_tool_calls(raw)
            else:
                text, calls = raw, []
            registry.record(rec["digests"], kind, sid, text)
            return text, info.get("tokens") or {}, sid, mode, calls

        # run backend
        sid = entry["sid"] if (entry and mode == "delta") else None
        log(f"{mode:6} {len(msgs)} msgs ({len(prompt)} chars) -> run "
            f"{'-s ' + sid[-8:] if sid else '(new session)'}")
        raw, new_sid = await backend.run(model, prompt, sid)
        if not raw:
            raise RuntimeError("empty response from opencode run")
        if bridge:
            text, calls = extract_tool_calls(raw)
        else:
            text, calls = raw, []
        registry.record(rec["digests"], kind, new_sid or sid or "", text)
        return text, {}, new_sid or sid or "", mode, calls

    async def handle_chat(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": {"message": "invalid JSON body"}},
                                     status=400)
        if not isinstance(body, dict):
            return web.json_response(
                {"error": {"message": "JSON body must be an object"}}, status=400)
        model = body.get("model", default_model) or default_model
        stream = bool(body.get("stream", False))
        msgs = body.get("messages", [])
        if not isinstance(msgs, list) or not msgs:
            return web.json_response(
                {"error": {"message": "'messages' must be a non-empty array"}},
                status=400)
        tools_list = effective_tools(body.get("tools"), body.get("tool_choice"))
        tool_choice = body.get("tool_choice")

        digs = digests_of(msgs)
        preview = _text_of(msgs[-1].get("content", ""))[:60] \
            if isinstance(msgs[-1], dict) else ""
        log(f"{'stream' if stream else 'block '} {model}: {len(msgs)} msgs, "
            f"tools={len(tools_list)}, last={preview!r}")
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if not stream:
            try:
                text, tokens, _sid, mode, calls = await complete(
                    msgs, model, tools_list, tool_choice)
            except Exception as e:
                log(f"ERROR: {e}")
                return web.json_response({"error": {"message": str(e)}},
                                         status=502)
            return web.json_response(
                _blocking_response(cid, model, text, tokens, calls))

        # ---- SSE: blocking completion + live deltas from the event hub ----
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)

        async def send(payload: str) -> None:
            await resp.write(f"data: {payload}\n\n".encode())

        q: asyncio.Queue | None = None
        post: asyncio.Task | None = None
        bridge = bool(tools_list)
        parser = ToolCallStreamParser() if bridge else None
        try:
            await send(chunk(cid, model, delta={"role": "assistant", "content": ""}))
            backend, kind = await get_backend()
            entry = registry.match(digs, kind)
            prompt, mode, rec = build_prompt(msgs, entry, tools_list, tool_choice)
            log(f"{mode:6} {len(msgs)} msgs ({len(prompt)} chars) -> {kind}")

            q = state["hub"].subscribe()
            sent = 0

            async def emit_content(piece: str) -> None:
                nonlocal sent
                if not piece:
                    return
                sent += len(piece)
                await send(chunk(cid, model, delta=piece))

            if kind == "serve":
                sid = entry["sid"] if (entry and mode == "delta") else None
                if sid is None:
                    sid = await backend.create_session(model)
                post = asyncio.create_task(backend.prompt(sid, prompt, model))

                part_types: dict[str, str] = {}
                assistant_ids: set[str] = set()
                buffered: dict[str, list[str]] = {}

                async def pump_evt(evt) -> None:
                    if not isinstance(evt, dict):
                        return
                    etype = evt.get("type")
                    props = evt.get("properties") or {}
                    if props.get("sessionID") != sid:
                        return
                    if etype == "message.updated":
                        info = props.get("info") or {}
                        if info.get("role") == "assistant" and info.get("id"):
                            assistant_ids.add(info["id"])
                        return
                    if etype == "message.part.updated":
                        part = props.get("part") or {}
                        pid = part.get("id")
                        if pid and part.get("type"):
                            part_types[pid] = part["type"]
                            if pid in buffered:
                                queued = buffered.pop(pid)
                                if part["type"] == "text":
                                    for d in queued:
                                        if parser is not None:
                                            await emit_content(parser.feed(d))
                                        else:
                                            await emit_content(d)
                        return
                    if etype == "message.part.delta":
                        pid = props.get("partID")
                        mid = props.get("messageID")
                        delta = props.get("delta") or ""
                        if not delta or mid not in assistant_ids:
                            return
                        ptype = part_types.get(pid)
                        if ptype == "text":
                            if parser is not None:
                                await emit_content(parser.feed(delta))
                            else:
                                await emit_content(delta)
                        elif ptype is None:
                            # part type not known yet — buffer until updated
                            buffered.setdefault(pid, []).append(delta)

                # pump queue while the blocking prompt runs
                while not post.done():
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    await pump_evt(evt)
                # quiet drain for straggler deltas after completion
                while True:
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=0.3)
                    except asyncio.TimeoutError:
                        break
                    await pump_evt(evt)

                envelope = await post
                info = envelope.get("info", {}) or {}
                err = info.get("error")
                if err:
                    msg = (err.get("data") or {}).get("message") or \
                        err.get("name") or "backend error"
                    raise RuntimeError(msg)
                final_raw = "".join(
                    p.get("text", "") for p in envelope.get("parts", [])
                    if p.get("type") == "text")
                calls: list[dict] = []
                if parser is not None:
                    rest, calls = parser.finish(final_raw)
                    await emit_content(rest)
                    text, _ = extract_tool_calls(final_raw)
                else:
                    if len(final_raw) > sent:
                        await emit_content(final_raw[sent:])
                    text = final_raw
                registry.record(rec["digests"], kind, sid, text)
                tokens = info.get("tokens") or {}
            else:
                async def on_delta(delta: str) -> None:
                    if parser is not None:
                        await emit_content(parser.feed(delta))
                    else:
                        await emit_content(delta)

                raw, new_sid = await backend.run(
                    model, prompt,
                    entry["sid"] if (entry and mode == "delta") else None,
                    on_delta=on_delta,
                )
                calls = []
                if parser is not None:
                    rest, calls = parser.finish(raw)
                    await emit_content(rest)
                    text, _ = extract_tool_calls(raw)
                else:
                    text = raw
                    if raw and sent < len(raw):
                        await emit_content(raw[sent:])
                registry.record(rec["digests"], kind, new_sid or "", text)
                tokens = {}

            # tool_calls deltas must arrive before the final finish_reason
            for i, tc in enumerate(calls or []):
                await send(chunk(cid, model, tool_calls=[{
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }]))
            finish = "tool_calls" if calls else "stop"
            await send(chunk(cid, model, finish=finish))
            await send("[DONE]")
            await resp.write_eof()
            log(f"-> done (stream, {sent} chars, {len(calls or [])} tool_calls)")
        except Exception as e:
            log(f"STREAM ERROR: {e}")
            try:
                await send(chunk(cid, model, delta=f"[proxy error] {e}"))
                await send(chunk(cid, model, finish="stop"))
                await send("[DONE]")
                await resp.write_eof()
            except Exception:
                pass
        finally:
            if post is not None:
                if not post.done():
                    post.cancel()
                elif not post.cancelled():
                    post.exception()  # mark retrieved (avoid GC warning)
            if q is not None:
                state["hub"].unsubscribe(q)
        return resp

    async def handle_models(_request: web.Request) -> web.Response:
        http = ensure_http()
        ids = await discover_free_models(http, serve.url, serve.headers)
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model"} for m in ids
        ]})

    async def health(_request: web.Request) -> web.Response:
        # get_backend lazily spawns/adopts serve so /health reports reality
        _backend, kind = await get_backend()
        serve_ok = kind == "serve"
        models = list(_models_cache.get("ids") or KNOWN_MODELS)
        if serve_ok and (models is KNOWN_MODELS or
                         time.monotonic() - float(_models_cache.get("at") or 0)
                         >= MODELS_CACHE_TTL):
            models = await discover_free_models(
                state["http"], serve.url, serve.headers)
        return web.json_response({
            "status": "ok",
            "cli": cli,
            "models": models,
            "backend": f"serve:{serve.url}" if serve_ok else "run",
            "serve_pure": serve.pure,
            "agent": agent or None,
            "sessions": len(registry),
        })

    async def on_startup(app: web.Application) -> None:
        log(f"backend policy: serve:{serve.url} (pure={serve.pure}) -> run; "
            f"agent={agent or '(default)'}")

    async def on_cleanup(app: web.Application) -> None:
        if state["hub"] is not None and state["hub"]._task is not None:
            state["hub"]._subs.clear()
            state["hub"]._task.cancel()
        await serve.shutdown()
        await run_backend.shutdown()
        if state["http"] is not None and not state["http"].closed:
            await state["http"].close()

    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat)
    app.router.add_post("/chat/completions", handle_chat)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def _blocking_response(cid: str, model: str, text: str, tokens: dict,
                       tool_calls: list[dict] | None = None) -> dict:
    prompt_tokens = int(tokens.get("input", 0)) + \
        int((tokens.get("cache") or {}).get("read", 0)) + \
        int((tokens.get("cache") or {}).get("write", 0))
    completion_tokens = int(tokens.get("output", 0))
    message: dict = {"role": "assistant", "content": text}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": cid, "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0,
                     "message": message,
                     "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="OpenCode free-tier OpenAI SSE proxy (Linux/macOS/Windows)")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("OPENCODE_PROXY_PORT", 18788)))
    ap.add_argument("--host",
                    default=os.environ.get("OPENCODE_PROXY_HOST", "127.0.0.1"))
    ap.add_argument("--cli",
                    default=os.environ.get("OPENCODE_CLI"),
                    help="path to opencode binary (auto-detected if omitted)")
    ap.add_argument("--work-dir", default=os.getcwd(),
                    help="cwd for opencode serve/run (default: current dir)")
    ap.add_argument("--serve-port", type=int,
                    default=int(os.environ.get("OPENCODE_SERVE_PORT", 18790)),
                    help="port for the managed `opencode serve` instance")
    ap.add_argument("--serve-url",
                    default=os.environ.get("OPENCODE_SERVE_URL"),
                    help="use an existing serve at this URL (no spawn)")
    ap.add_argument("--serve-keep-plugins",
                    action="store_true",
                    help="load MCP/plugin schemas too (default: --pure serve, "
                         "saves ~6k tokens/call)")
    ap.add_argument("--agent", default=os.environ.get("OPENCODE_AGENT",
                                                       "bypass-lite"),
                    help="opencode agent for prompts; '' = built-in default")
    ap.add_argument("--default-model", default="big-pickle",
                    help="model used when the request names none")
    args = ap.parse_args()

    try:
        cli = find_cli(args.cli)
    except FileNotFoundError as e:
        print(f"[proxy] FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    log(f"streaming {args.host}:{args.port} (cli={cli} dir={work_dir} "
        f"serve=:{args.serve_port}{'/'+args.serve_url if args.serve_url else ''} "
        f"agent={args.agent or '(default)'} default_model={args.default_model})")
    try:
        web.run_app(
            make_app(cli, work_dir,
                     serve_port=args.serve_port,
                     serve_pure=not args.serve_keep_plugins,
                     serve_url=args.serve_url,
                     agent=args.agent,
                     default_model=args.default_model),
            host=args.host, port=args.port, print=None,
        )
    except OSError as e:
        print(f"[proxy] FATAL: cannot bind {args.host}:{args.port}: {e}",
              file=sys.stderr)
        hint = "./run.sh status" if not IS_WINDOWS else ".\\run.ps1 status"
        print(f"[proxy] Is another copy already running? Check `{hint}`.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
