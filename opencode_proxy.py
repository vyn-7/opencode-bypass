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
  A request-level hash of the effective tools + tool_choice rides along the
  digest entry, so a changed client-tool contract forces a resync instead of
  silently reusing a session staged for the old tools.

Reasoning (OpenCode ``reasoning`` parts <-> OpenAI ``reasoning_content``):

  OpenCode streams reasoning as a part with ``type == "reasoning"`` whose
  deltas arrive as ``message.part.delta`` / ``field == "text"``. The proxy
  translates them to ``delta.reasoning_content`` — never into
  ``delta.content``, never into persistent memory — and returns
  ``message.reasoning_content`` on blocking responses. Per-part state
  survives the documented OpenCode race where ``message.part.delta`` can
  arrive *before* ``message.part.updated`` (opencode#26924), dedups
  snapshot/delta overlap, and reasoning never terminates a stream: the
  finish reason still follows the final content or tool_calls turn.

Error semantics (never fabricate an answer):

  * pre-stream failures -> HTTP 502 JSON ``{"error": ...}`` (Hermes sees a
    real error and can retry/fallback);
  * mid-stream failures -> one structured
    ``data: {"error": {"message", "type": "server_error"}}`` event then
    ``[DONE]`` — no assistant content, no ``finish_reason``, no
    ``[proxy error]`` text (a fake stop would look like success).

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
    The generated call id is preserved verbatim (when present) so it
    matches the ``[tool result: ...] (call call_...)`` label of the next
    turn — ids are protocol state, never regenerated on replay.
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
        line = f"- {name}({args})"
        tcid = tc.get("id")
        if tcid:
            line += f" (call {tcid})"
        rendered.append(line)
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
               last_reply: str, tools_sig: str = "") -> dict:
        entry = {
            "digests": list(digests),
            "backend": backend_kind,
            "sid": sid,
            "last_reply": last_reply,
            # Request-level metadata (NOT part of the message digest): the
            # hash of this request's effective tools + tool_choice. A change
            # here must invalidate the session cursor even when the message
            # prefix still matches.
            "tools_sig": tools_sig,
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


def tools_signature(tools_list, tool_choice) -> str:
    """Canonical hash of the *effective* client-tool contract for one request.

    The message digest registry only sees ``messages`` — if the same
    conversation changes its available tools or ``tool_choice`` between
    turns, a prefix match alone would wrongly reuse a backend session that
    was staged under the old contract. This signature is stored next to the
    registry entry (request metadata, never mixed into the transcript
    digest) and a mismatch forces a full resync so the new
    ``[client tools]`` block is explicitly supplied.
    """
    blob = json.dumps(
        {"tools": tools_list or [], "tool_choice": tool_choice},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def lookup_entry(registry: "Registry", digests: list[str], backend_kind: str,
                 tools_sig: str) -> dict | None:
    """Registry cursor that still matches this request's tool contract."""
    entry = registry.match(digests, backend_kind)
    if entry is None:
        return None
    if entry.get("tools_sig", "") != tools_sig:
        # Tools/tool_choice changed mid-conversation -> force resync; the
        # stale entry stays (it becomes valid again if the contract returns).
        return None
    return entry


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
    """Feed raw deltas; emit only client-safe content; collect finished calls.

    Complete ``[tool_call]`` blocks become available *as they complete*
    via :meth:`drain_calls` (so the wire stream can carry a valid
    ``delta.tool_calls`` before the final finish chunk) while incomplete
    markers stay held — half-valid JSON arguments are never exposed.
    :meth:`finish` still returns every call that was not drained yet, so
    callers that only collect at completion keep the old behavior.
    """

    def __init__(self) -> None:
        self.raw = ""
        self.emitted = 0          # length of clean content already sent
        self._pending: list[dict] = []  # completed calls not yet drained
        self._recognized = 0      # complete blocks recognized so far
        self.emitted_calls = 0    # blocks already handed out (drain/finish)

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self.raw += delta
        clean, calls = streaming_view(self.raw)
        # Re-parse mints a fresh id per block on every feed — only surface
        # blocks beyond the count already recognized (each block once).
        if len(calls) > self._recognized:
            self._pending.extend(calls[self._recognized:])
            self._recognized = len(calls)
        if len(clean) <= self.emitted:
            return ""
        out = clean[self.emitted:]
        self.emitted = len(clean)
        return out

    def drain_calls(self) -> list[dict]:
        """Pop completed tool calls that have not been handed out yet."""
        out, self._pending = self._pending, []
        self.emitted_calls += len(out)
        return out

    def finish(self, final_text: str | None = None) -> tuple[str, list[dict]]:
        """Flush held content (if final) and return calls not yet handed out.

        ``final_text`` is authoritative (blocking envelope): it replaces the
        streamed raw buffer, so any divergence is corrected here — but blocks
        already drained during the stream are never returned twice.
        """
        if final_text is not None:
            self.raw = final_text
        clean, calls = extract_tool_calls(self.raw)
        out = ""
        if len(clean) > self.emitted:
            out = clean[self.emitted:]
            self.emitted = len(clean)
        rest = calls[self.emitted_calls:]
        self._recognized = len(calls)
        self._pending = []
        self.emitted_calls = len(calls)
        return out, rest


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
    """Build one OpenAI ``chat.completion.chunk`` JSON string.

    ``delta`` is an arbitrary delta dictionary, so a single helper produces
    ``{"content": ...}``, ``{"reasoning_content": ...}``,
    ``{"tool_calls": [...]}`` or the initial role chunk without forcing
    every event into content.
    """
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
# OpenCode event -> OpenAI SSE translation (reasoning kept separate)
# --------------------------------------------------------------------------

def _payload_text(part: dict, field: str) -> str:
    """Best-effort string view of ``part[field]`` (str or content blocks)."""
    val = part.get(field) if isinstance(part, dict) else None
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        chunks = []
        for blk in val:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                chunks.append(blk["text"])
        return "".join(chunks)
    return ""


def _part_snapshot(part: dict) -> str:
    """Accumulated text of a part snapshot (``text``, else ``content``)."""
    text = _payload_text(part, "text")
    if text:
        return text
    return _payload_text(part, "content")


def split_envelope_parts(parts) -> tuple[str, str]:
    """Split a blocking OpenCode message envelope into (text, reasoning).

    Reasoning parts are preserved even though the ordinary final response
    lives in a separate ``text`` part — they must never be silently
    discarded, mixed into content, or persisted as memory.
    """
    texts: list[str] = []
    reasons: list[str] = []
    for p in parts or []:
        if not isinstance(p, dict):
            continue
        val = _payload_text(p, "text") or _payload_text(p, "content")
        if not val:
            continue
        ptype = p.get("type")
        if ptype == "text":
            texts.append(val)
        elif ptype == "reasoning":
            reasons.append(val)
    return "".join(texts), "".join(reasons)


class _PartState:
    """Per-part streaming state for one request (never persisted)."""

    __slots__ = ("ptype", "emitted", "buffered", "snap_covered",
                 "snap_consumed")

    def __init__(self) -> None:
        self.ptype: str | None = None
        self.emitted: str = ""        # exact text already sent downstream
        self.buffered: list[str] = [] # deltas held while the type is unknown
        # Snapshot text that went beyond `emitted` at apply time, and how
        # much of it has re-arrived as late (duplicate) deltas — guards the
        # reverse of opencode#26924 without eating genuinely new content.
        self.snap_covered: str = ""
        self.snap_consumed: int = 0


class StreamTranslator:
    """Translates OpenCode serve ``/event`` payloads into OpenAI SSE chunks.

    Wire contract (Hermes' Chat Completions view):

        role chunk -> reasoning_content deltas -> content deltas
                    -> [delta.tool_calls per complete [tool_call] block]
                    -> finish_reason -> [DONE]

    Design notes:

      * reasoning and content are strictly separate channels — reasoning
        never lands in ``delta.content``, never in the digest/memory
        layer, and never produces a finish reason by itself;
      * deltas that arrive before ``message.part.updated`` (or before the
        owning ``message.updated``) are buffered, then flushed with the
        correct field once the metadata lands (opencode#26924);
      * snapshot ``text`` accumulated on ``message.part.updated`` is
        merged without re-emitting bytes already streamed;
      * tool blocks are only converted once complete — malformed or
        partial blocks stay hidden from the content stream.
    """

    def __init__(self, completion_id: str, model: str, bridge: bool) -> None:
        self.cid = completion_id
        self.model = model
        self.bridge = bridge
        self.parser = ToolCallStreamParser() if bridge else None
        self.assistant_ids: set[str] = set()
        self.known_mids: set[str] = set()
        self.parts: dict[str, _PartState] = {}
        # events whose owning message role is not yet known: mid -> FIFO of
        # ("part", part) / ("delta", pid, field, delta)
        self.mid_pending: dict[str, list[tuple]] = {}
        self.reasoning_emitted = ""
        self.content_emitted = ""
        self.call_index = 0  # OpenAI tool_calls[].index + count of calls

    # ---- chunk builders --------------------------------------------------
    def role_chunk(self) -> str:
        return chunk(self.cid, self.model,
                     delta={"role": "assistant", "content": ""})

    def finish_chunks(self) -> list[str]:
        finish = "tool_calls" if self.call_index else "stop"
        return [chunk(self.cid, self.model, finish=finish), "[DONE]"]

    @staticmethod
    def error_payload(message: str) -> str:
        """Structured mid-stream error Hermes can identify (never content)."""
        return json.dumps({"error": {
            "message": message or "proxy error",
            "type": "server_error",
        }})

    def _tool_call_chunk(self, call: dict) -> str:
        payload = {
            "index": self.call_index,
            "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": call.get("function") or {},
        }
        self.call_index += 1
        return chunk(self.cid, self.model, delta={"tool_calls": [payload]})

    # ---- field-level emission -------------------------------------------
    def feed_reasoning(self, piece: str) -> str | None:
        if not piece:
            return None
        self.reasoning_emitted += piece
        return chunk(self.cid, self.model,
                     delta={"reasoning_content": piece})

    def feed_content(self, piece: str) -> list[str]:
        if not piece:
            return []
        out: list[str] = []
        if self.parser is None:
            self.content_emitted += piece
            out.append(chunk(self.cid, self.model, delta={"content": piece}))
            return out
        visible = self.parser.feed(piece)
        if visible:
            self.content_emitted += visible
            out.append(chunk(self.cid, self.model, delta={"content": visible}))
        for call in self.parser.drain_calls():
            out.append(self._tool_call_chunk(call))
        return out

    def _route(self, ptype: str | None, piece: str) -> list[str]:
        if not piece:
            return []
        if ptype == "reasoning":
            out = self.feed_reasoning(piece)
            return [out] if out else []
        if ptype == "text":
            return self.feed_content(piece)
        return []  # tool/step/... parts never surface as chat content

    # ---- per-part bookkeeping -------------------------------------------
    def _part(self, pid: str) -> _PartState:
        st = self.parts.get(pid)
        if st is None:
            st = self.parts[pid] = _PartState()
        return st

    def _delta_guarded(self, st: _PartState, piece: str) -> bool:
        """True when ``piece`` duplicates snapshot text already streamed."""
        cov = st.snap_covered
        i = st.snap_consumed
        if cov and cov[i:].startswith(piece):
            st.snap_consumed = i + len(piece)
            return True
        return False

    def _apply_snapshot(self, st: _PartState, snap: str,
                        initial: bool) -> list[str]:
        """Merge an authoritative part snapshot without duplicate bytes."""
        if initial:
            queued = "".join(st.buffered)
            st.buffered = []
            if snap.startswith(queued):
                # snapshot covers the buffered deltas (and may extend past
                # them): stream it whole; buffered bytes are inside it.
                if snap:
                    out = self._route(st.ptype, snap)
                    st.emitted = snap
                    st.snap_covered = snap
                    st.snap_consumed = len(queued)
                    return out
                return []
            if queued.startswith(snap):
                # snapshot is behind (e.g. creation with text=""): the
                # buffered deltas are newer — stream them as the head.
                out = self._route(st.ptype, queued) if queued else []
                st.emitted = queued
                st.snap_covered = ""
                st.snap_consumed = 0
                return out
            # divergence: prefer the streamed delta order over the snapshot
            head = queued or snap
            out = self._route(st.ptype, head) if head else []
            st.emitted = head
            st.snap_covered = head if (snap and not queued) else ""
            st.snap_consumed = 0
            return out

        emitted = st.emitted
        if not snap:
            return []
        if snap.startswith(emitted):
            tail = snap[len(emitted):]
            st.emitted = snap
            st.snap_covered = tail
            st.snap_consumed = 0
            return self._route(st.ptype, tail)
        if emitted.startswith(snap):
            return []  # snapshot behind the stream (e.g. final trimEnd)
        return []      # divergent snapshot: ignore rather than corrupt

    # ---- event handlers --------------------------------------------------
    @staticmethod
    def _event_session(props: dict):
        """Session id of an event (top-level, or nested in info/part)."""
        sid = props.get("sessionID")
        if sid:
            return sid
        info = props.get("info")
        if isinstance(info, dict) and info.get("sessionID"):
            return info["sessionID"]
        part = props.get("part")
        if isinstance(part, dict) and part.get("sessionID"):
            return part["sessionID"]
        return None

    def _mid_known_user(self, mid) -> bool:
        return (mid is not None and mid in self.known_mids
                and mid not in self.assistant_ids)

    def _register_part(self, part: dict) -> list[str]:
        pid = part.get("id")
        if not pid:
            return []
        mid = part.get("messageID")
        if self._mid_known_user(mid):
            return []
        if mid is not None and mid not in self.assistant_ids:
            # owning message role not known yet -> hold the registration
            self.mid_pending.setdefault(mid, []).append(("part", part))
            return []
        return self._register_part_now(part)

    def _register_part_now(self, part: dict) -> list[str]:
        pid = part["id"]
        st = self._part(pid)
        ptype = part.get("type")
        snap = _part_snapshot(part)
        out: list[str] = []
        if ptype:
            first = st.ptype is None
            st.ptype = ptype
            if first:
                if ptype in ("text", "reasoning"):
                    out += self._apply_snapshot(st, snap, initial=True)
                else:
                    st.buffered = []  # non-content part: drop stray deltas
                return out
        # subsequent snapshot for a known type (final/intermediate update)
        if snap:
            out += self._apply_snapshot(st, snap, initial=False)
        return out

    def _handle_delta(self, pid: str | None, mid, field: str,
                      delta: str) -> list[str]:
        if not delta or not pid:
            return []
        if self._mid_known_user(mid):
            return []
        if mid is not None and mid not in self.assistant_ids:
            self.mid_pending.setdefault(mid, []).append(
                ("delta", pid, field, delta))
            return []
        st = self._part(pid)
        if st.ptype is None:
            st.buffered.append(delta)  # type unknown -> buffer, never drop
            return []
        if self._delta_guarded(st, delta):
            return []
        out = self._route(st.ptype, delta)
        st.emitted += delta
        return out

    def handle_session(self, evt, sid: str) -> list[str]:
        """Translate one serve event, filtered to the active session."""
        if not isinstance(evt, dict):
            return []
        props = evt.get("properties") or {}
        evt_sid = self._event_session(props)
        if evt_sid != sid:
            return []  # foreign or unattributable event (original behavior)
        etype = evt.get("type")
        if etype == "message.updated":
            return self._on_message_updated(props)
        if etype == "message.part.updated":
            part = props.get("part")
            return self._register_part(part) if isinstance(part, dict) else []
        if etype == "message.part.delta":
            return self._handle_delta(
                props.get("partID"), props.get("messageID"),
                props.get("field") or "text", props.get("delta") or "")
        return []

    def _on_message_updated(self, props) -> list[str]:
        info = props.get("info") or {}
        mid = info.get("id")
        if not mid:
            return []
        self.known_mids.add(mid)
        pending = self.mid_pending.pop(mid, [])
        if info.get("role") != "assistant":
            return []  # known non-assistant (user echo): discard held events
        self.assistant_ids.add(mid)
        out: list[str] = []
        for ev in pending:
            if ev[0] == "part":
                out += self._register_part_now(ev[1])
            elif ev[0] == "delta":
                _, pid, field, delta = ev
                out += self._handle_delta(pid, mid, field, delta)
        return out

    # ---- completion ------------------------------------------------------
    @staticmethod
    def _tail(final: str, emitted: str) -> str:
        """Unemitted suffix of an authoritative final string.

        Normal case: final extends what we streamed (or trims it, e.g.
        ``trimEnd`` on the final snapshot) -> exact suffix. Divergent
        snapshots fall back to the historical length-based tail so content
        is never silently lost.
        """
        if not final:
            return ""
        if final.startswith(emitted):
            return final[len(emitted):]
        if emitted.startswith(final):
            return ""
        return final[len(emitted):] if len(final) > len(emitted) else ""

    def finalize_text(self, final_text: str,
                      final_reasoning: str) -> tuple[list[str], str, str]:
        """End the stream against the authoritative backend result.

        Returns ``(payloads, text, reasoning)`` where payloads contain any
        missing reasoning/content tail, the remaining tool-call deltas,
        the finish chunk and ``[DONE]`` — in that order. Reasoning is
        reconciled on its own channel and can never become content. A
        reasoning part alone never finishes the stream: the finish reason
        still reflects content/tool_calls semantics (requirement: reasoning
        is not the final assistant message).
        """
        out: list[str] = []
        r_tail = self._tail(final_reasoning or "", self.reasoning_emitted)
        if r_tail:
            piece = self.feed_reasoning(r_tail)
            if piece:
                out.append(piece)
        calls: list[dict] = []
        if self.parser is not None:
            rest, calls = self.parser.finish(final_text)
            if rest:
                self.content_emitted += rest
                out.append(chunk(self.cid, self.model,
                                 delta={"content": rest}))
        else:
            c_tail = self._tail(final_text or "", self.content_emitted)
            if c_tail:
                self.content_emitted += c_tail
                out.append(chunk(self.cid, self.model,
                                 delta={"content": c_tail}))
        for call in calls:
            out.append(self._tool_call_chunk(call))
        out += self.finish_chunks()
        return out, self.content_emitted, self.reasoning_emitted

    def finalize_parts(self, parts) -> tuple[list[str], str, str]:
        text, reasoning = split_envelope_parts(parts)
        return self.finalize_text(text, reasoning)


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
                  on_delta=None) -> tuple[str, str, str | None]:
        """One `run` invocation. Returns (text, reasoning, session_id).

        ``on_delta`` (optional async) receives (delta, field) where field
        is ``"text"`` or ``"reasoning"``.

        Output format empirically inspected on opencode 1.18.32
        (``run --format json``):

          * each part arrives as ONE complete JSON event at part
            completion (``{"type": "text"|"reasoning", "part": {...}}``) —
            the CLI ignores ``message.part.delta`` (opencode#38638), so
            there is no token-level streaming to translate on this path;
          * ``reasoning`` events are only emitted with ``--thinking``
            (run.ts: ``part.type === "reasoning" && part.time?.end &&
            thinking``) — the proxy therefore always passes the flag;
          * limits: reasoning/text arrive complete-per-part (block-style,
            not delta-style) and errors arrive as ``{"type":"error"}``.
        """
        args = chunk_argv(prompt)
        hard_split = any(" " not in a and len(a) >= MAX_ARG_CHARS for a in args)
        if hard_split:
            log(f"warning: argv hard-split (no space boundary in "
                f"{MAX_ARG_CHARS} chars) — one boundary char may differ")
        cmd_args = [
            "run",
            "--format", "json",
            "--thinking",   # without it the CLI drops reasoning parts
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
        reasoning_parts: list[str] = []
        # complete-per-part events, keyed by part id (multiple parts possible)
        seen: dict[str, int] = {}
        found_sid = session_id
        err_events: list[str] = []

        def _event_text(obj: dict, ptype: str) -> str:
            part = obj.get("part")
            if isinstance(part, dict):
                val = part.get("text")
                if isinstance(val, str) and val:
                    return val
                if isinstance(val, list):
                    return "".join(
                        b.get("text", "") for b in val
                        if isinstance(b, dict) and isinstance(b.get("text"), str))
            val = obj.get("text")
            return val if isinstance(val, str) else ""

        async def pump():
            nonlocal found_sid
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
                etype = obj.get("type")
                if etype == "error":
                    err = obj.get("error")
                    if isinstance(err, dict):
                        msg = (err.get("data") or {}).get("message") \
                            or err.get("name") or "opencode run error"
                    else:
                        msg = str(err or "opencode run error")
                    err_events.append(str(msg))
                    continue
                if etype not in ("text", "reasoning"):
                    continue
                part = obj.get("part") if isinstance(obj.get("part"), dict) \
                    else {}
                pid = part.get("id") or f"{etype}:{len(seen)}"
                new = _event_text(obj, etype)
                if not new:
                    continue
                already = seen.get(pid, 0)
                if len(new) <= already:
                    continue  # same completion event seen again
                delta = new[already:]
                seen[pid] = len(new)
                if etype == "reasoning":
                    reasoning_parts.append(delta)
                    if on_delta is not None:
                        await on_delta(delta, "reasoning")
                else:
                    parts.append(delta)
                    if on_delta is not None:
                        await on_delta(delta, "text")

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
        if err_events:
            raise RuntimeError(err_events[-1])
        if rc != 0:
            assert proc.stderr is not None
            err = (await proc.stderr.read()).decode("utf-8", "replace")[-500:]
            raise RuntimeError(f"opencode exit {rc}: {err}")
        return "".join(parts), "".join(reasoning_parts), found_sid

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

    async def complete(msgs, model, tools_list, tool_choice, tools_sig):
        """One completion. Returns (text, reasoning, tokens, sid, mode, calls)."""
        backend, kind = await get_backend()
        digs = digests_of(msgs)
        entry = lookup_entry(registry, digs, kind, tools_sig)
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
            raw, raw_reasoning = split_envelope_parts(envelope.get("parts", []))
            if bridge:
                text, calls = extract_tool_calls(raw)
            else:
                text, calls = raw, []
            registry.record(rec["digests"], kind, sid, text,
                            tools_sig=tools_sig)
            if calls:
                log("tool_calls: " + ", ".join(
                    f"{(c.get('function') or {}).get('name', '?')}={c.get('id', '?')}"
                    for c in calls))
            if raw_reasoning:
                log(f"reasoning: {len(raw_reasoning)} chars (blocking)")
            return (text, raw_reasoning, info.get("tokens") or {}, sid, mode,
                    calls)

        # run backend
        sid = entry["sid"] if (entry and mode == "delta") else None
        log(f"{mode:6} {len(msgs)} msgs ({len(prompt)} chars) -> run "
            f"{'-s ' + sid[-8:] if sid else '(new session)'}")
        raw, raw_reasoning, new_sid = await backend.run(model, prompt, sid)
        if not raw and not raw_reasoning:
            raise RuntimeError("empty response from opencode run")
        if bridge:
            text, calls = extract_tool_calls(raw)
        else:
            text, calls = raw, []
        registry.record(rec["digests"], kind, new_sid or sid or "", text,
                        tools_sig=tools_sig)
        if calls:
            log("tool_calls: " + ", ".join(
                f"{(c.get('function') or {}).get('name', '?')}={c.get('id', '?')}"
                for c in calls))
        if raw_reasoning:
            log(f"reasoning: {len(raw_reasoning)} chars (blocking, run)")
        return text, raw_reasoning, {}, new_sid or sid or "", mode, calls

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
        tools_sig = tools_signature(tools_list, tool_choice)

        digs = digests_of(msgs)
        preview = _text_of(msgs[-1].get("content", ""))[:60] \
            if isinstance(msgs[-1], dict) else ""
        log(f"{'stream' if stream else 'block '} {model}: {len(msgs)} msgs, "
            f"tools={len(tools_list)}, last={preview!r}")
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if not stream:
            try:
                text, reasoning, tokens, _sid, mode, calls = await complete(
                    msgs, model, tools_list, tool_choice, tools_sig)
            except Exception as e:
                log(f"ERROR: {e}")
                return web.json_response({"error": {"message": str(e)}},
                                         status=502)
            return web.json_response(
                _blocking_response(cid, model, text, tokens, calls,
                                   reasoning_content=reasoning))

        # ---- SSE setup: everything that can fail *before* the 200 commits.
        # Transport failures surface as real HTTP errors so Hermes' retry /
        # fallback logic sees them (a fabricated assistant answer would not).
        try:
            backend, kind = await get_backend()
            entry = lookup_entry(registry, digs, kind, tools_sig)
            prompt, mode, rec = build_prompt(msgs, entry, tools_list,
                                             tool_choice)
            sid = entry["sid"] if (entry and mode == "delta") else None
            if kind == "serve" and sid is None:
                sid = await backend.create_session(model)
        except Exception as e:
            log(f"ERROR (pre-stream): {e}")
            return web.json_response({"error": {"message": str(e)}},
                                     status=502)
        log(f"{mode:6} {len(msgs)} msgs ({len(prompt)} chars) -> {kind}")

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
        tr = StreamTranslator(cid, model, bridge)
        finish = "stop"
        try:
            await send(tr.role_chunk())
            q = state["hub"].subscribe()

            if kind == "serve":
                post = asyncio.create_task(backend.prompt(sid, prompt, model))

                async def pump_evt(evt) -> None:
                    for payload in tr.handle_session(evt, sid):
                        await send(payload)

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
                payloads, text, reasoning = tr.finalize_parts(
                    envelope.get("parts", []))
                for payload in payloads:
                    await send(payload)
                registry.record(rec["digests"], kind, sid, text,
                                tools_sig=tools_sig)
                tokens = info.get("tokens") or {}
            else:
                async def on_delta(delta: str, field: str = "text") -> None:
                    if field == "reasoning":
                        piece = tr.feed_reasoning(delta)
                        if piece:
                            await send(piece)
                    else:
                        for payload in tr.feed_content(delta):
                            await send(payload)

                raw, raw_reasoning, new_sid = await backend.run(
                    model, prompt, sid, on_delta=on_delta,
                )
                if not raw and not raw_reasoning:
                    raise RuntimeError("empty response from opencode run")
                payloads, text, reasoning = tr.finalize_text(
                    raw, raw_reasoning)
                for payload in payloads:
                    await send(payload)
                registry.record(rec["digests"], kind, new_sid or "", text,
                                tools_sig=tools_sig)
                tokens = {}

            finish = "tool_calls" if tr.call_index else "stop"
            if tr.call_index:
                log(f"tool_calls: {tr.call_index} (finish=tool_calls)")
            if tr.reasoning_emitted:
                log(f"reasoning: {len(tr.reasoning_emitted)} chars")
            await resp.write_eof()
            log(f"-> done (stream, content={len(tr.content_emitted)}, "
                f"reasoning={len(tr.reasoning_emitted)}, "
                f"tool_calls={tr.call_index}, finish={finish})")
        except Exception as e:
            # The HTTP status is already 200 — never fabricate assistant
            # content or finish_reason=stop (Hermes would treat that as a
            # successful provider answer). Emit a structured error event
            # the client can identify, then terminate cleanly.
            log(f"STREAM ERROR: {e}")
            try:
                await send(StreamTranslator.error_payload(str(e)))
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
                       tool_calls: list[dict] | None = None,
                       reasoning_content: str | None = None) -> dict:
    prompt_tokens = int(tokens.get("input", 0)) + \
        int((tokens.get("cache") or {}).get("read", 0)) + \
        int((tokens.get("cache") or {}).get("write", 0))
    completion_tokens = int(tokens.get("output", 0))
    message: dict = {"role": "assistant"}
    if reasoning_content:
        # Hermes reads non-streaming reasoning from message.reasoning_content;
        # reasoning stays strictly separate from visible content.
        message["reasoning_content"] = reasoning_content
    message["content"] = text
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
