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
  ``<tool_call>\\n{json}\\n</tool_call>`` blocks (preferred, closer to MiMo's
  native trained format; legacy ``[tool_call]...[/tool_call]`` still
  accepted), and the proxy converts those blocks into standard OpenAI
  ``tool_calls`` (blocking + streaming). Both serializations normalize to
  the same internal representation: exactly one JSON object per block
  (``{"name": ..., "arguments": {...}}``), no markdown fences, no wrapper
  text inside the block. Incomplete blocks are buffered until the matching
  close tag arrives and never leak into visible content; malformed JSON is
  kept verbatim as content, never exposed as a tool call. A request-level
  hash of the effective tools + tool_choice rides along the digest entry,
  so a changed client-tool contract forces a resync instead of silently
  reusing a session staged for the old tools.

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
  Multi-turn tool use preserves prior ``reasoning_content`` in the backend
  context as a separate ``[assistant reasoning]...[/assistant reasoning]``
  section — never merged into visible ``[assistant]`` text, never written
  to persistent memory, never stored as a separate digest-registry memory
  object. The backend session natively retains the previous reasoning +
  text-with-tool-call output, so delta prompts carry only the new tail
  (assistant replay with IDs + tool result); the model thus receives prior
  reasoning exactly once per continuation without redundant second
  assistant events, while generated OpenAI call IDs survive unchanged for
  tool-result association.

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

# Streaming transport/recovery (task: treat /event as an incremental transport
# that may disconnect, stall, reorder metadata/deltas, or lose events).
# All timeouts are based on actual event/byte activity, never total duration —
# a request may legitimately run much longer than the idle timeout (long
# thinking is normal for reasoning models).
EVENT_STREAM_IDLE_TIMEOUT = float(
    os.environ.get("OPENCODE_EVENT_IDLE_TIMEOUT", "150"))
EVENT_RECONNECT_INITIAL = float(
    os.environ.get("OPENCODE_EVENT_RECONNECT_INITIAL", "1.0"))
EVENT_RECONNECT_MAX = float(
    os.environ.get("OPENCODE_EVENT_RECONNECT_MAX", "16.0"))
BUFFERED_DELTA_TTL = float(
    os.environ.get("OPENCODE_BUFFERED_DELTA_TTL", "120"))
SESSION_FETCH_TIMEOUT = float(
    os.environ.get("OPENCODE_SESSION_FETCH_TIMEOUT", "10"))

# Session health: a backend session that began mutating and then failed must
# not be silently reused merely because the Hermes digest is unchanged.
SESSION_HEALTHY = "healthy"
SESSION_TAINTED = "tainted"
SESSION_INVALID = "invalid"

# Windows process creation flags (numeric so import works on POSIX).
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200

TOOL_OPEN = "[tool_call]"
TOOL_CLOSE = "[/tool_call]"
# Preferred XML-style serialization (closer to MiMo's native trained format).
# The parser accepts BOTH formats and normalizes them into the same internal
# OpenAI tool-call representation. Do NOT remove the bracket syntax yet.
TOOL_OPEN_BRACKET = "[tool_call]"
TOOL_CLOSE_BRACKET = "[/tool_call]"
TOOL_OPEN_XML = "<tool_call>"
TOOL_CLOSE_XML = "</tool_call>"
# Ordered preferred-first: XML is tried first when scanning for the next block.
_TOOL_MARKERS: tuple[tuple[str, str], ...] = (
    (TOOL_OPEN_XML, TOOL_CLOSE_XML),
    (TOOL_OPEN_BRACKET, TOOL_CLOSE_BRACKET),
)

_models_cache: dict = {"at": 0.0, "ids": None}

# Detailed tool-call diagnostics are gated behind this flag so normal runs
# stay quiet. Enabled via --debug or OPENCODE_PROXY_DEBUG=1. Never logs API
# keys, credentials, or arbitrary environment variables — only protocol
# metadata (counts, names, ids, booleans, finish reasons, transport/mode).
DEBUG = os.environ.get("OPENCODE_PROXY_DEBUG", "").lower() in (
    "1", "true", "yes", "on")


def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[proxy {ts}] {msg}", flush=True)


def debug_log(msg: str) -> None:
    if DEBUG:
        log(f"[debug] {msg}")


# Verbose payload logging is strictly opt-in (full event/part bodies can
# contain private tool results or source). Normal runs log only protocol
# metadata (counts, ids, names, booleans, finish reasons, transport/mode).
VERBOSE_PAYLOADS = os.environ.get("OPENCODE_PROXY_VERBOSE_PAYLOADS", "").lower() in (
    "1", "true", "yes", "on")


def verbose_log(msg: str) -> None:
    if VERBOSE_PAYLOADS:
        log(f"[payload] {msg}")


def slog(request_id: str, event: str, **fields) -> None:
    """Structured streaming state-machine log (never secrets).

    Always logs: request id + event name + supplied metadata fields.
    Callers must only pass protocol metadata (sids truncated to 8 chars,
    counts, ids, booleans, finish reasons) — never API keys, auth headers,
    cookies, env secrets, full tool results, source code, or full bodies
    (those stay behind VERBOSE_PAYLOADS).
    """
    try:
        parts = [f"req={request_id}", f"ev={event}"]
        for k, v in fields.items():
            # Defensive redaction: drop anything that looks like a secret.
            lk = str(k).lower()
            if any(s in lk for s in ("key", "token", "auth", "cookie",
                                     "secret", "password", "bearer")):
                continue
            parts.append(f"{k}={v}")
        log("[stream] " + " ".join(parts))
    except Exception:
        pass


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


def _reasoning_of_msg(m: dict) -> str:
    """Extract assistant reasoning_content without merging into visible text.

    Hermes sends prior thinking as ``reasoning_content`` (alias ``reasoning``
    accepted). Returns "" when absent. Never returns visible content.
    """
    if not isinstance(m, dict):
        return ""
    val = m.get("reasoning_content")
    if val is None:
        val = m.get("reasoning")
    if isinstance(val, str):
        return val if val.strip() else ""
    if isinstance(val, list):
        # Multipart-style reasoning: join text parts, ignore non-text.
        chunks: list[str] = []
        for p in val:
            if isinstance(p, dict) and isinstance(p.get("text"), str) \
                    and p["text"]:
                chunks.append(p["text"])
            elif isinstance(p, str) and p:
                chunks.append(p)
        joined = "".join(chunks)
        return joined if joined.strip() else ""
    return ""


def _render_tool_calls(tool_calls) -> str:
    """Render assistant tool_calls so the model remembers what it did.

    Without this, an agentic turn looks like ``user -> (empty assistant) ->
    tool result`` and the model loses which action produced the result.
    The generated call id is preserved verbatim (when present) so it
    matches the ``[tool result: ...] (call call_...)`` label of the next
    turn — ids are protocol state, never regenerated on replay.
    Ordering of tool_calls is preserved as received.
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
    (reasoning + content + tool calls), tool results (with tool name / call
    id when provided). Reasoning is kept strictly separate from visible
    content as ``[assistant reasoning]...[/assistant reasoning]`` — never
    merged into ``[assistant]`` text, never written to persistent memory,
    never stored as a separate digest-registry memory object. Its purpose
    is strictly to preserve model context for the next reasoning/tool turn
    (MiMo API: previous reasoning_content is retained during multi-turn
    tool use). Empty turns carry no information and are skipped — but an
    assistant tool-call message with empty ``content`` is NOT empty when it
    carries tool_calls or reasoning. Nothing is ever truncated — Hermes'
    ``messages`` array is authoritative.
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
            reasoning = _reasoning_of_msg(m)
            calls = _render_tool_calls(m.get("tool_calls"))
            if reasoning:
                blocks.append(
                    (False,
                     f"[assistant reasoning]\n{reasoning}\n[/assistant reasoning]"))
            if text and calls:
                blocks.append((False, f"[assistant]\n{text}\n{calls}"))
            elif text:
                blocks.append((False, f"[assistant]\n{text}"))
            elif calls:
                blocks.append((False, f"[assistant]\n{calls}"))
            elif not reasoning:
                continue  # truly empty assistant turn: skip
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
    """Drop a leading assistant echo of our previous reply (backend has it).

    The backend session natively retains the previous turn's reasoning part
    + text part containing ``<tool_call>`` (or legacy ``[tool_call]``), so
    the delta must not recreate a redundant second assistant event for the
    same turn. However:

    * a turn carrying ``tool_calls`` is NEVER dropped — its OpenAI call IDs
      are protocol state that does not exist in the backend's original
      textual output and must survive verbatim for tool-result association;
    * a turn carrying ``reasoning_content`` is NEVER dropped on content
      equality alone — ``last_reply`` is content-only (reasoning never
      stored) so dropping would discard prior thinking that the next
      reasoning/tool turn needs;
    * an assistant tool-call message with empty ``content`` is NOT empty
      when it carries tool_calls/reasoning and must not be discarded.
    """
    if not tail_msgs or not last_reply:
        return tail_msgs
    head = tail_msgs[0]
    if not isinstance(head, dict) or head.get("role") != "assistant":
        return tail_msgs
    if head.get("tool_calls"):
        return tail_msgs  # never drop turns that carry tool calls
    if _reasoning_of_msg(head):
        return tail_msgs  # reasoning preserved; backend echo-skip is content-only
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
            # Session health: a successful completion always restores HEALTHY.
            # Failures mark TAINTED/INVALID via mark_tainted/invalidate so the
            # next turn forces a clean resync instead of reusing a partially
            # mutated backend session.
            "health": SESSION_HEALTHY,
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

    def mark_tainted(self, sid: str, backend_kind: str) -> bool:
        """Mark a backend session TAINTED (do not silently reuse).

        Returns True when an entry was found. A TAINTED session forces the
        next safe request into a full resynchronization on a fresh backend
        session (see lookup_entry). Backwards compatible: entries created
        before health existed default to HEALTHY.
        """
        found = False
        for e in self._entries:
            if e.get("sid") == sid and e.get("backend") == backend_kind:
                e["health"] = SESSION_TAINTED
                e["used"] = time.time()
                found = True
        return found

    def invalidate(self, sid: str, backend_kind: str) -> bool:
        """Drop a registry entry entirely (INVALID -> fresh session next)."""
        before = len(self._entries)
        self._entries = [e for e in self._entries
                         if not (e.get("sid") == sid
                                 and e.get("backend") == backend_kind)]
        return len(self._entries) != before

    def health_of(self, sid: str, backend_kind: str) -> str:
        for e in self._entries:
            if e.get("sid") == sid and e.get("backend") == backend_kind:
                return e.get("health", SESSION_HEALTHY)
        return SESSION_INVALID


def plan_prompt(msgs, entry: dict | None) -> tuple[str, str, dict]:
    """Decide delta-vs-resync for one request.

    Returns (prompt_text, mode, record_payload). ``record_payload`` always
    carries the full incoming digest list — the caller records it only
    after a successful completion (retries then replay the same tail).

    Backend-session accounting (explicit): after the previous model
    response the ``opencode serve`` session natively stores the reasoning
    part + the text part containing the textual ``<tool_call>`` (or legacy
    ``[tool_call]``) block. The next delta therefore primarily provides
    the tool result (+ new information) with the OpenAI call ID preserved
    in the textual replay (``(call call_...)``) so the relationship stays
    unambiguous — it does NOT create a redundant second assistant event
    in the backend session (prompts travel as user text, not as new
    assistant messages). The prior reasoning thus reaches the model
    exactly once per continuation: once natively in the session prefix,
    once quoted in the delta tail when it is genuinely new (never
    duplicated from the stored prefix, never merged into visible text).
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
    if entry.get("health", SESSION_HEALTHY) != SESSION_HEALTHY:
        # TAINTED/INVALID sessions must never be silently reused: a streamed
        # backend response began and then failed (or could not be reconciled),
        # so the backend session may be partially mutated even though the
        # Hermes digest is unchanged. Force a fresh session + full replay.
        return None
    return entry


def _incoming_has_reasoning(msgs) -> bool:
    """Whether the incoming Hermes request replays prior reasoning_content."""
    for m in msgs or []:
        if isinstance(m, dict) and m.get("role") == "assistant" \
                and _reasoning_of_msg(m):
            return True
    return False


def _debug_tool_turn(*, model: str, transport: str, mode: str,
                     tools_list, tool_choice, raw: str,
                     calls: list[dict], reasoning: str,
                     incoming_msgs) -> None:
    """Detailed per-turn diagnostics behind the debug flag (no secrets).

    Logs: client-tools present/count, tool_choice, backend model/transport,
    whether the backend emitted <tool_call> / [tool_call], parser result
    (count), generated IDs + function names + arg-parse success, finish
    reason, reasoning present, next-request reasoning present, resync/delta.
    Never logs API keys, credentials, env vars, or full argument values
    (only names/ids/booleans/counts).
    """
    if not DEBUG:
        return
    try:
        has_tools = bool(tools_list)
        n_tools = len(tools_list) if isinstance(tools_list, list) else 0
        emitted_xml = TOOL_OPEN_XML in (raw or "")
        emitted_bracket = TOOL_OPEN_BRACKET in (raw or "")
        finish = "tool_calls" if calls else "stop"
        has_reasoning = bool((reasoning or "").strip())
        incoming_reasoning = _incoming_has_reasoning(incoming_msgs)
        names: list[str] = []
        ids: list[str] = []
        for c in calls or []:
            fn = (c.get("function") or {}) if isinstance(c, dict) else {}
            names.append(str(fn.get("name", "?")))
            ids.append(str(c.get("id", "?") if isinstance(c, dict) else "?"))
        # Arg-parse success: every emitted call parsed exactly one JSON
        # object (guaranteed by _parse_tool_call_body); malformed blocks
        # never become calls.
        arg_ok = bool(calls)  # True when at least one call parsed
        debug_log(
            f"tool-turn model={model} transport={transport} mode={mode} "
            f"client_tools={has_tools} n_tools={n_tools} "
            f"tool_choice={json.dumps(tool_choice, default=str)} "
            f"emitted_xml={emitted_xml} emitted_bracket={emitted_bracket} "
            f"parser_calls={len(calls or [])} ids={ids} names={names} "
            f"args_parse_ok={arg_ok} finish={finish} "
            f"reasoning_present={has_reasoning} "
            f"incoming_reasoning={incoming_reasoning}"
        )
    except Exception:
        pass


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
    """Render OpenAI function tools as a prompt section for the model.

    Preferred serialization is XML-style ``<tool_call>`` (closer to MiMo's
    native trained format); legacy ``[tool_call]`` remains accepted for
    backward compatibility but the model is taught to emit XML first.
    """
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
        "To invoke a tool, emit one XML block per call and no other markup "
        "around it. arguments must be a JSON object matching that tool's "
        "parameters schema:"
    )
    lines.append(TOOL_OPEN_XML)
    lines.append('{"name": "tool_name", "arguments": {"key": "value"}}')
    lines.append(TOOL_CLOSE_XML)
    lines.append(
        "Rules: exactly one JSON object per tool call; one block per tool "
        "call; you may include normal text before or after blocks; do not "
        "use markdown fences around the tool call; do not emit additional "
        "wrapper text inside the tool-call block; never invent tools that "
        "are not listed above; never call the backend's built-in tool "
        "runner — the proxy translates these blocks into the client's "
        "OpenAI tool_calls format. "
        f"Legacy {TOOL_OPEN_BRACKET}...{TOOL_CLOSE_BRACKET} syntax is still "
        "accepted for backward compatibility, but prefer "
        f"{TOOL_OPEN_XML}...{TOOL_CLOSE_XML}."
    )
    if tool_choice == "required":
        lines.append("You MUST emit at least one tool_call block this turn.")
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        if fn.get("name"):
            lines.append(f"You MUST call the tool {fn['name']!r} this turn.")
    return "\n".join(lines)


def _parse_tool_call_body(body: str) -> dict | None:
    """Parse the JSON object inside a tool-call block into an OpenAI call.

    Accepts both ``<tool_call>`` (preferred) and legacy ``[tool_call]``
    bodies — both normalize to the same representation. Exactly one JSON
    object per block; no markdown fences; no wrapper text inside. Malformed
    JSON returns None (caller keeps the block verbatim as visible content,
    never exposes it as a tool call).
    """
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
        # arguments as string must itself be valid JSON object text when
        # non-empty; malformed strings are rejected (not a tool call).
        s = args.strip()
        if s and not (s.startswith("{") or s.startswith('"')):
            # Plain non-JSON strings are not valid tool arguments per the
            # "exactly one JSON object" contract — but be lenient: if it
            # fails to parse as JSON, keep raw string only when it looks
            # like JSON; otherwise reject to avoid malformed calls.
            try:
                json.loads(s)
            except json.JSONDecodeError:
                return None
        arg_str = args
    else:
        try:
            arg_str = json.dumps(args if args is not None else {},
                                 ensure_ascii=False)
        except (TypeError, ValueError):
            arg_str = "{}"
    # arguments must be a JSON object when parsed (not a list/scalar).
    try:
        parsed_args = json.loads(arg_str) if isinstance(arg_str, str) else args
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed_args, dict):
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arg_str},
    }


def _find_next_open(text: str, start: int) -> tuple[int, str, str] | None:
    """Earliest tool-call open at/after ``start``: (idx, open, close)."""
    best: tuple[int, str, str] | None = None
    for o, c in _TOOL_MARKERS:
        j = text.find(o, start)
        if j < 0:
            continue
        if best is None or j < best[0]:
            best = (j, o, c)
    return best


def _partial_open_hold(rest: str) -> int:
    """Longest suffix of ``rest`` that could become a tool open prefix."""
    hold = 0
    for o, _ in _TOOL_MARKERS:
        max_h = min(len(o) - 1, len(rest))
        for k in range(max_h, 0, -1):
            if o.startswith(rest[len(rest) - k:]):
                hold = max(hold, k)
                break
    return hold


def extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Strip complete tool-call blocks (parsed) from ``text``.

    Accepts both ``<tool_call>...</tool_call>`` (preferred) and legacy
    ``[tool_call]...[/tool_call]``; both normalize identically. Unclosed
    markers are flushed into content as-is (final malformed output stays
    visible). Malformed but complete markers are kept verbatim (never a
    tool call).
    """
    if not text:
        return "", []
    if TOOL_OPEN_XML not in text and TOOL_OPEN_BRACKET not in text:
        return text, []
    calls: list[dict] = []
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        nxt = _find_next_open(text, i)
        if nxt is None:
            parts.append(text[i:])
            break
        j, o, c = nxt
        parts.append(text[i:j])
        k = text.find(c, j + len(o))
        if k < 0:
            parts.append(text[j:])  # unclosed: keep visible
            break
        body = text[j + len(o):k]
        call = _parse_tool_call_body(body)
        if call is not None:
            calls.append(call)
        else:
            parts.append(text[j:k + len(c)])
        i = k + len(c)
    return "".join(parts), calls


def streaming_view(text: str) -> tuple[str, list[dict]]:
    """Content safe to emit now + complete calls so far.

    Unlike :func:`extract_tool_calls`, an unclosed marker (or a partial
    ``<tool_call>`` / ``[tool_call]`` prefix at the end of the buffer) is
    *held back* — those bytes may still turn into a complete block and must
    never leak into the client's content stream as ordinary assistant
    content while still being generated.
    """
    if not text:
        return "", []
    calls: list[dict] = []
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        nxt = _find_next_open(text, i)
        if nxt is None:
            rest = text[i:]
            hold = _partial_open_hold(rest)
            parts.append(rest[: len(rest) - hold] if hold else rest)
            break
        j, o, c = nxt
        parts.append(text[i:j])
        k = text.find(c, j + len(o))
        if k < 0:
            break  # hold from OPEN through end
        body = text[j + len(o):k]
        call = _parse_tool_call_body(body)
        if call is not None:
            calls.append(call)
        else:
            parts.append(text[j:k + len(c)])
        i = k + len(c)
    return "".join(parts), calls


class ToolCallStreamParser:
    """Feed raw deltas; emit only client-safe content; collect finished calls.

    Complete ``<tool_call>`` (preferred) or legacy ``[tool_call]`` blocks
    become available *as they complete* via :meth:`drain_calls` (so the
    wire stream can carry a valid ``delta.tool_calls`` before the final
    finish chunk) while incomplete markers stay held — half-valid JSON
    arguments are never exposed and incomplete blocks never leak as
    ordinary content. :meth:`finish` still returns every call that was not
    drained yet, so callers that only collect at completion keep the old
    behavior.
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
                 "snap_consumed", "created_at", "buffered_at", "field",
                 "seq")

    def __init__(self) -> None:
        self.ptype: str | None = None
        self.emitted: str = ""        # exact text already sent downstream
        self.buffered: list[str] = [] # deltas held while the type is unknown
        # Snapshot text that went beyond `emitted` at apply time, and how
        # much of it has re-arrived as late (duplicate) deltas — guards the
        # reverse of opencode#26924 without eating genuinely new content.
        self.snap_covered: str = ""
        self.snap_consumed: int = 0
        # Bounded-lifetime buffering for the dangerous case where
        # message.part.delta arrives but message.part.updated never does.
        # Never guess field=="text" means visible content — resolution must
        # consult authoritative session state (reasoning vs text separate).
        self.created_at: float = time.monotonic()
        self.buffered_at: float | None = None  # first buffered delta time
        self.field: str | None = None  # last seen delta field (metadata only)
        self.seq: int = 0  # order counter for deltas on this part


class StreamTranslator:
    """Translates OpenCode serve ``/event`` payloads into OpenAI SSE chunks.

    Wire contract (Hermes' Chat Completions view):

        role chunk -> reasoning_content deltas -> content deltas
                    -> [delta.tool_calls per complete <tool_call> block]
                    -> finish_reason -> [DONE]

    (Legacy ``[tool_call]`` blocks follow the same contract.)

    Design notes:

      * reasoning and content are strictly separate channels — reasoning
        never lands in ``delta.content``, never in the digest/memory
        layer, and never produces a finish reason by itself;
      * deltas that arrive before ``message.part.updated`` (or before the
        owning ``message.updated``) are buffered, then flushed with the
        correct field once the metadata lands (opencode#26924);
      * snapshot ``text`` accumulated on ``message.part.updated`` is
        merged without re-emitting bytes already streamed;
      * tool blocks (``<tool_call>`` preferred, ``[tool_call]`` legacy)
        are only converted once complete — malformed, partial, or
        unclosed blocks stay held (never leak as content) until the
        matching close tag arrives.
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
        # retained for debug logging (ids/names only, never full secrets)
        self.emitted_calls: list[dict] = []

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
        self.emitted_calls.append(payload)
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
            # Type unknown -> buffer with bounded lifetime, never drop
            # immediately and never guess field=="text" means visible content.
            st.buffered.append(delta)
            now = time.monotonic()
            if st.buffered_at is None:
                st.buffered_at = now
            st.field = field
            st.seq += 1
            return []
        if self._delta_guarded(st, delta):
            return []
        out = self._route(st.ptype, delta)
        st.emitted += delta
        st.seq += 1
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

    # ---- recovery: stale buffers + session-state reconciliation -----------
    def unknown_pending(self) -> list[str]:
        """Part ids with buffered deltas but still-unknown type."""
        return [pid for pid, st in self.parts.items()
                if st.ptype is None and st.buffered]

    def stale_unknown(self, now: float | None = None,
                      ttl: float | None = None) -> list[str]:
        """Unknown-type buffers whose bounded lifetime has expired."""
        if now is None:
            now = time.monotonic()
        if ttl is None:
            ttl = BUFFERED_DELTA_TTL
        out = []
        for pid, st in self.parts.items():
            if st.ptype is not None or not st.buffered:
                continue
            born = st.buffered_at if st.buffered_at is not None \
                else st.created_at
            if now - born >= ttl:
                out.append(pid)
        return out

    def has_stale_unknown(self, now: float | None = None,
                          ttl: float | None = None) -> bool:
        return bool(self.stale_unknown(now, ttl))

    def streamed_offsets(self) -> dict:
        """Per-part streamed offsets for logging/tests (no bodies)."""
        return {pid: {"type": st.ptype, "emitted_len": len(st.emitted),
                      "buffered_len": sum(len(b) for b in st.buffered),
                      "seq": st.seq}
                for pid, st in self.parts.items()}

    def resolve_unknown_via_session(self, session_parts) -> list[str]:
        """Flush stale unknown buffers using authoritative session state.

        ``session_parts`` is a list of OpenCode part dicts (id/type/text).
        Never guesses field=="text" means content: the part type decides the
        channel (reasoning vs text vs drop). Unknown ids stay buffered for a
        later retry — they are never fabricated as content.
        """
        out: list[str] = []
        if not session_parts:
            return out
        by_id = {p.get("id"): p for p in session_parts
                 if isinstance(p, dict) and p.get("id")}
        for pid in self.unknown_pending():
            part = by_id.get(pid)
            if not isinstance(part, dict):
                continue  # still unknown: keep buffered, retry later
            # Reuse the normal registration path so snapshot/buffer merge
            # stays single-sourced (no duplicate bytes).
            out += self._register_part_now(part)
        return out

    def reconcile_with_session_parts(self, session_parts) -> list[str]:
        """Emit only missing authoritative info after a stall/disconnect.

        Handles both unknown-type resolution and per-part tails without
        replaying already-delivered bytes, duplicating tool calls, or moving
        reasoning into content. Returns OpenAI SSE payloads (no finish).
        """
        out: list[str] = []
        # 1. Resolve unknown buffers first (correct channel matters).
        out += self.resolve_unknown_via_session(session_parts)
        if not session_parts:
            return out
        # 2. Per-part tails for known text/reasoning parts.
        by_id = {p.get("id"): p for p in session_parts
                 if isinstance(p, dict) and p.get("id")}
        for pid, part in by_id.items():
            st = self.parts.get(pid)
            if st is None or st.ptype is None:
                continue  # handled above or non-streamed part
            if st.ptype not in ("text", "reasoning"):
                continue
            snap = _part_snapshot(part)
            if not snap:
                continue
            # _apply_snapshot with initial=False emits only the missing tail
            # and suppresses duplicates/divergence safely.
            out += self._apply_snapshot(st, snap, initial=False)
        return out

    def reconcile_with_messages(self, messages) -> list[str]:
        """Reconcile against GET /session/{sid}/message output.

        ``messages`` is the authoritative list [{info, parts}]. Only the
        parts of assistant messages are considered; user parts are ignored.
        Emits only missing tails (no finish chunk, no [DONE]).
        """
        parts: list[dict] = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            info = m.get("info") or {}
            # Envelope shape from prompt() is {info, parts}; fetch shape is
            # [{info, parts}]. Only reconcile assistant content.
            role = info.get("role")
            # Some payloads nest parts at top level; be lenient.
            plist = m.get("parts")
            if role is not None and role != "assistant":
                continue
            if isinstance(plist, list):
                parts.extend(p for p in plist if isinstance(p, dict))
        # Bridge case: authoritative full text may extend parser.raw. The
        # per-part path above already handles tails, but tool-call text needs
        # the parser-level tail too when parts lack granular snapshots.
        out = self.reconcile_with_session_parts(parts)
        if self.parser is not None and parts:
            auth_text, auth_reason = split_envelope_parts(parts)
            # Reasoning tail via global channel (per-part already covered;
            # _tail is idempotent so double-call emits nothing new).
            if auth_reason:
                r_tail = self._tail(auth_reason, self.reasoning_emitted)
                if r_tail:
                    piece = self.feed_reasoning(r_tail)
                    if piece:
                        out.append(piece)
            # Text tail for bridge: only when authoritative raw extends the
            # parser buffer (avoids duplicating stripped content).
            if auth_text and auth_text.startswith(self.parser.raw):
                tail_raw = auth_text[len(self.parser.raw):]
                if tail_raw:
                    out.extend(self.feed_content(tail_raw))
        elif parts:
            # Non-bridge global tails are already covered per-part, but keep
            # the envelope-level fallback for sessions whose part ids rotated
            # (new ids after reconnect) — emit only truly missing bytes.
            auth_text, auth_reason = split_envelope_parts(parts)
            r_tail = self._tail(auth_reason or "", self.reasoning_emitted)
            if r_tail:
                piece = self.feed_reasoning(r_tail)
                if piece:
                    out.append(piece)
            c_tail = self._tail(auth_text or "", self.content_emitted)
            # Only emit if no per-part emission already covered it: check
            # that the tail is not already represented in per-part offsets.
            if c_tail and not out:
                self.content_emitted += c_tail
                out.append(chunk(self.cid, self.model,
                                 delta={"content": c_tail}))
        return out

    def is_divergent(self, final_text: str, final_reasoning: str) -> bool:
        """Whether the final envelope contradicts already-streamed state."""
        # Tool-bridge finals are authoritative via parser.finish (markers
        # stripped) — divergence there is expected, not a taint signal.
        if self.parser is not None:
            return False
        ft = final_text or ""
        fr = final_reasoning or ""
        ce = self.content_emitted or ""
        re_ = self.reasoning_emitted or ""
        def _div(final: str, emitted: str) -> bool:
            if not final and not emitted:
                return False
            if not final or not emitted:
                return False  # empty side is not contradiction
            return not (final.startswith(emitted)
                        or emitted.startswith(final))
        return _div(ft, ce) or _div(fr, re_)

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
# Session-state helpers (authoritative recovery source)
# --------------------------------------------------------------------------

def latest_assistant_parts(messages) -> list[dict]:
    """Parts of the latest assistant message in a fetch payload."""
    latest: list[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        info = m.get("info") or {}
        if info.get("role") != "assistant":
            continue
        plist = m.get("parts")
        if isinstance(plist, list):
            latest = [p for p in plist if isinstance(p, dict)]
    return latest


def all_session_parts(messages) -> list[dict]:
    """All assistant parts across a fetch payload (for reconciliation)."""
    out: list[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        info = m.get("info") or {}
        if info.get("role") is not None and info.get("role") != "assistant":
            continue
        plist = m.get("parts")
        if isinstance(plist, list):
            out.extend(p for p in plist if isinstance(p, dict))
    return out


# --------------------------------------------------------------------------
# Event hub: one shared SSE reader on serve's global /event
# --------------------------------------------------------------------------

class EventHub:
    """Single shared SSE reader with inactivity watchdog + safe reconnect.

    State machine (logged): CONNECTED -> ACTIVE <-> IDLE -> STALE ->
    RECONNECTING -> CONNECTED ... -> FAILED (only when subscribers remain
    and reconnects keep failing; FAILED still retries with capped backoff).

    Guarantees:

    * exactly one active ``/event`` reader at a time (no duplicate
      consumers); reconnects never create a second competing loop;
    * reconnects never emit duplicate output themselves — deduplication
      lives in :class:`StreamTranslator` (snapshot/delta guards) plus the
      session-state reconciliation layer (authoritative fetch);
    * bounded exponential backoff ``1s, 2s, 4s, 8s, 16s`` (configurable max),
      reset after any successfully parsed event;
    * cancellation-safe: ``CancelledError`` always propagates; ``unsubscribe``
      never cancels a loop still serving other subscribers;
    * malformed SSE lines never kill the hub (skipped + counted).
    """

    def __init__(self, base_url: str, headers: dict,
                 idle_timeout: float | None = None,
                 reconnect_initial: float | None = None,
                 reconnect_max: float | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self._subs: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._session: ClientSession | None = None
        self._explicit_idle = idle_timeout is not None
        self._explicit_reconnect_initial = reconnect_initial is not None
        self._explicit_reconnect_max = reconnect_max is not None
        self._idle_timeout = float(idle_timeout) if idle_timeout is not None \
            else float(EVENT_STREAM_IDLE_TIMEOUT)
        self._reconnect_initial = float(reconnect_initial) \
            if reconnect_initial is not None \
            else float(EVENT_RECONNECT_INITIAL)
        self._reconnect_max = float(reconnect_max) \
            if reconnect_max is not None else float(EVENT_RECONNECT_MAX)
        # Observability (no bodies, only metadata).
        self._conn_id = 0
        self._state = "IDLE"
        self._connect_time = 0.0
        self._last_activity = 0.0
        self._last_event_type = ""
        self._bytes = 0
        self._events = 0
        self._malformed = 0
        self._reconnects = 0

    def set_session(self, session: ClientSession) -> None:
        self._session = session

    @property
    def idle_timeout(self) -> float:
        # Explicit constructor values win (unit tests); otherwise read the
        # module global dynamically so CLI/env patches apply without
        # rebuilding the hub.
        if getattr(self, "_explicit_idle", False):
            return self._idle_timeout
        try:
            return float(EVENT_STREAM_IDLE_TIMEOUT)
        except Exception:
            return self._idle_timeout

    @property
    def reconnect_max(self) -> float:
        if getattr(self, "_explicit_reconnect_max", False):
            return self._reconnect_max
        try:
            return float(EVENT_RECONNECT_MAX)
        except Exception:
            return self._reconnect_max

    @property
    def reconnect_initial(self) -> float:
        if getattr(self, "_explicit_reconnect_initial", False):
            return self._reconnect_initial
        try:
            return float(EVENT_RECONNECT_INITIAL)
        except Exception:
            return self._reconnect_initial

    def stats(self) -> dict:
        return {
            "conn": self._conn_id,
            "state": self._state,
            "bytes": self._bytes,
            "events": self._events,
            "malformed": self._malformed,
            "reconnects": self._reconnects,
            "last_event": self._last_event_type,
        }

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="event-hub")
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    async def _run(self) -> None:
        backoff = self.reconnect_initial
        while self._subs:
            self._conn_id += 1
            conn = self._conn_id
            self._state = "CONNECTED"
            self._connect_time = time.monotonic()
            self._last_activity = self._connect_time
            connect_wall = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log(f"event hub: conn={conn} CONNECTED at {connect_wall} "
                f"(subs={len(self._subs)})")
            try:
                assert self._session is not None
                async with self._session.get(
                    f"{self.base_url}/event",
                    headers=self.headers,
                    timeout=ClientTimeout(total=None, sock_connect=10),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"/event HTTP {resp.status}")
                    self._state = "ACTIVE"
                    idle = self.idle_timeout
                    buf = ""
                    while self._subs:
                        try:
                            raw = await asyncio.wait_for(
                                resp.content.readany(), timeout=idle)
                        except asyncio.TimeoutError:
                            # STALE: no bytes/events for idle_timeout. Close
                            # and reconnect; per-request recovery reconciles
                            # via authoritative session state.
                            self._state = "STALE"
                            idle_for = time.monotonic() - self._last_activity
                            log(f"event hub: conn={conn} STALE "
                                f"(idle={idle_for:.1f}s>={idle:.0f}s "
                                f"events={self._events} "
                                f"last={self._last_event_type or '-'}) "
                                f"-> reconnect")
                            break
                        if not raw:
                            # Clean EOF (server closed): reconnect.
                            log(f"event hub: conn={conn} EOF "
                                f"(events={self._events}) -> reconnect")
                            break
                        self._bytes += len(raw)
                        self._last_activity = time.monotonic()
                        if self._state != "ACTIVE":
                            self._state = "ACTIVE"
                        try:
                            text = raw.decode("utf-8", "replace")
                        except Exception:
                            continue
                        buf += text
                        # SSE framing: one or more lines per chunk, partial
                        # lines span chunks. Split complete lines, keep tail.
                        lines = buf.split("\n")
                        buf = lines.pop()
                        for line in lines:
                            s = line.strip()
                            if not s:
                                continue
                            if s.startswith(":"):
                                # keep-alive comment: byte activity only.
                                verbose_log(f"hub conn={conn} keep-alive")
                                continue
                            if not s.startswith("data:"):
                                continue
                            payload = s[5:].strip()
                            if not payload:
                                continue
                            try:
                                evt = json.loads(payload)
                            except json.JSONDecodeError:
                                self._malformed += 1
                                debug_log(
                                    f"hub conn={conn} malformed SSE skipped "
                                    f"(total_malformed={self._malformed})")
                                continue
                            self._events += 1
                            try:
                                self._last_event_type = str(evt.get("type", ""))
                            except Exception:
                                self._last_event_type = "?"
                            self._last_activity = time.monotonic()
                            # Successful event resets backoff (spec).
                            backoff = self.reconnect_initial
                            verbose_log(f"hub conn={conn} evt="
                                        f"{self._last_event_type}")
                            for q in list(self._subs):
                                try:
                                    q.put_nowait(evt)
                                except asyncio.QueueFull:
                                    pass
                    # Inner loop broke (STALE/EOF): fall through to reconnect.
                    raise RuntimeError(f"/event stale/eof conn={conn}")
            except asyncio.CancelledError:
                raise
            except Exception as e:  # reconnect until subscribers go away
                if not self._subs:
                    break
                self._state = "RECONNECTING"
                self._reconnects += 1
                # Cap backoff; reset happens on next successful event.
                capped = min(max(backoff, 0.1), self.reconnect_max)
                log(f"event hub: conn={conn} RECONNECTING "
                    f"({e!r}; retry in {capped:.1f}s, "
                    f"reconnects={self._reconnects})")
                try:
                    await asyncio.sleep(capped)
                except asyncio.CancelledError:
                    raise
                backoff = min(capped * 2, self.reconnect_max)
                self._state = "IDLE"


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

    async def fetch_messages(self, sid: str) -> list[dict]:
        """Authoritative session state for reconciliation.

        ``GET /session/{sid}/message`` returns ``[{info, parts}]`` covering
        everything the backend has stored — including partial assistant
        output emitted while ``/event`` was stalled/disconnected. Used to
        recover missed reasoning/text/tool-call bytes without replaying the
        whole session. Raises on HTTP/transport failure (caller decides
        whether to taint the session).
        """
        timeout = SESSION_FETCH_TIMEOUT
        try:
            timeout = float(SESSION_FETCH_TIMEOUT)
        except Exception:
            timeout = 10.0
        async with self.http.get(
            f"{self._url}/session/{sid}/message",
            headers=self._headers,
            timeout=ClientTimeout(total=timeout),
        ) as resp:
            data = await resp.json(content_type=None)
            if resp.status != 200:
                raise RuntimeError(
                    f"session fetch failed: HTTP {resp.status} "
                    f"{json.dumps(data)[:300] if isinstance(data, dict) else str(data)[:300]}")
            if not isinstance(data, list):
                raise RuntimeError("session fetch: unexpected envelope shape")
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
            _debug_tool_turn(model=model, transport="serve", mode=mode,
                             tools_list=tools_list, tool_choice=tool_choice,
                             raw=raw, calls=calls,
                             reasoning=raw_reasoning, incoming_msgs=msgs)
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
        _debug_tool_turn(model=model, transport="run", mode=mode,
                         tools_list=tools_list, tool_choice=tool_choice,
                         raw=raw, calls=calls,
                         reasoning=raw_reasoning, incoming_msgs=msgs)
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
        # Structured per-request state for observability (no bodies).
        _hermes_prefix = (digs[0][:8] if digs else "empty")
        _sid_short = (sid[-8:] if isinstance(sid, str) and sid else "new")
        try:
            await send(tr.role_chunk())
            q = state["hub"].subscribe()
            try:
                _hub_stats = state["hub"].stats()
            except Exception:
                _hub_stats = {}
            slog(cid, "connect",
                 hermes=_hermes_prefix, sid=_sid_short, mode=mode,
                 hub_conn=_hub_stats.get("conn", "?"),
                 hub_state=_hub_stats.get("state", "?"))

            if kind == "serve":
                post = asyncio.create_task(backend.prompt(sid, prompt, model))
                last_activity = time.monotonic()
                connect_time = last_activity
                last_event_type = "-"
                events_seen = 0
                stalls = 0
                reconciles = 0
                last_fetch = 0.0
                last_hub_conn = _hub_stats.get("conn", 0)

                async def pump_evt(evt) -> None:
                    nonlocal last_activity, last_event_type, events_seen
                    nonlocal last_hub_conn
                    last_activity = time.monotonic()
                    try:
                        last_event_type = str(evt.get("type", "?")) \
                            if isinstance(evt, dict) else "?"
                    except Exception:
                        last_event_type = "?"
                    events_seen += 1
                    try:
                        cur_conn = state["hub"].stats().get("conn", "?")
                    except Exception:
                        cur_conn = "?"
                    if cur_conn != last_hub_conn:
                        slog(cid, "reconnect-success",
                             sid=_sid_short, hub_conn=cur_conn,
                             events=events_seen, last_event=last_event_type)
                        last_hub_conn = cur_conn
                    verbose_log(f"req={cid} evt={last_event_type} "
                                f"sid={_sid_short}")
                    for payload in tr.handle_session(evt, sid):
                        await send(payload)

                async def _try_reconcile(reason: str) -> int:
                    """Fetch authoritative state + emit only missing bytes.

                    Returns number of SSE payloads emitted (0 = no progress).
                    Never duplicates tool calls (parser tracks emitted_calls),
                    never moves reasoning into content, never fabricates.
                    """
                    nonlocal last_fetch, reconciles, last_activity
                    now = time.monotonic()
                    last_fetch = now
                    slog(cid, "reconciliation-start",
                         sid=_sid_short, reason=reason,
                         unknown=len(tr.unknown_pending()),
                         events=events_seen, last_event=last_event_type)
                    try:
                        fetch_timeout = float(SESSION_FETCH_TIMEOUT)
                    except Exception:
                        fetch_timeout = 10.0
                    try:
                        msgs = await asyncio.wait_for(
                            backend.fetch_messages(sid),
                            timeout=fetch_timeout + 2.0)
                    except Exception as e:
                        slog(cid, "reconciliation-result",
                             sid=_sid_short, reason=reason,
                             result=f"fetch-failed: {e!r}")
                        return 0
                    try:
                        payloads = tr.reconcile_with_messages(msgs)
                    except Exception as e:
                        slog(cid, "reconciliation-result",
                             sid=_sid_short, reason=reason,
                             result=f"reconcile-error: {e!r}")
                        return 0
                    for payload in payloads:
                        await send(payload)
                    if payloads:
                        reconciles += 1
                        last_activity = time.monotonic()
                        slog(cid, "reconciliation-result",
                             sid=_sid_short, reason=reason,
                             result=f"recovered={len(payloads)} "
                                    f"content={len(tr.content_emitted)} "
                                    f"reasoning={len(tr.reasoning_emitted)} "
                                    f"calls={tr.call_index}")
                    else:
                        slog(cid, "reconciliation-result",
                             sid=_sid_short, reason=reason,
                             result="no-missing-bytes")
                    # Duplicate suppression is inherent: reconcile emits only
                    # tails (_apply_snapshot guards), so already-delivered
                    # bytes are never re-sent.
                    return len(payloads)

                # pump queue while the blocking prompt runs; idle watchdog is
                # based on actual event/byte activity, not total duration.
                while not post.done():
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        now = time.monotonic()
                        idle_for = now - last_activity
                        # Bounded-lifetime buffered deltas: part metadata
                        # never arrived -> consult authoritative state.
                        if tr.has_stale_unknown(now) and \
                                now - last_fetch >= 5.0:
                            slog(cid, "stale-buffer",
                                 sid=_sid_short, idle_for=f"{idle_for:.1f}s",
                                 unknown=tr.unknown_pending(),
                                 offsets=tr.streamed_offsets())
                            await _try_reconcile("stale-buffer")
                            now = time.monotonic()
                            idle_for = now - last_activity
                        try:
                            idle_timeout = float(EVENT_STREAM_IDLE_TIMEOUT)
                        except Exception:
                            idle_timeout = 150.0
                        if idle_for >= idle_timeout:
                            stalls += 1
                            try:
                                hub_st = state["hub"].stats()
                            except Exception:
                                hub_st = {}
                            slog(cid, "stall-detected",
                                 sid=_sid_short, idle_for=f"{idle_for:.1f}s",
                                 idle_timeout=f"{idle_timeout:.0f}s",
                                 events=events_seen,
                                 last_event=last_event_type,
                                 hub_conn=hub_st.get("conn", "?"),
                                 hub_state=hub_st.get("state", "?"),
                                 hub_events=hub_st.get("events", "?"))
                            # Distinguish long thinking from dead stream via
                            # backend session state: if the fetch shows new
                            # bytes, recovery continues the stream; if not,
                            # keep waiting (thinking) but throttle retries to
                            # one fetch per idle_timeout window.
                            n = await _try_reconcile("stall")
                            if n == 0:
                                slog(cid, "stall-no-progress",
                                     sid=_sid_short,
                                     note="backend fetch ok but no new bytes; "
                                          "continuing (long thinking?)")
                                last_activity = time.monotonic()
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
                raw_parts = envelope.get("parts", [])
                # Final authoritative reconciliation: emit only not-yet-sent
                # tails (per-part offsets), never duplicate reasoning/text/
                # tool calls. Exactly one [DONE] comes from finalize.
                n_before_c = len(tr.content_emitted)
                n_before_r = len(tr.reasoning_emitted)
                n_before_calls = tr.call_index
                payloads, text, reasoning = tr.finalize_parts(raw_parts)
                for payload in payloads:
                    await send(payload)
                slog(cid, "final-reconciliation",
                     sid=_sid_short, streamed_c=n_before_c,
                     streamed_r=n_before_r, streamed_calls=n_before_calls,
                     final_c=len(text), final_r=len(reasoning),
                     emitted_c=len(tr.content_emitted),
                     emitted_r=len(tr.reasoning_emitted),
                     emitted_calls=tr.call_index,
                     finish=("tool_calls" if tr.call_index else "stop"))
                # If the final envelope contradicts streamed state (non-bridge
                # divergence) or tool boundaries are ambiguous, the backend
                # session may be partially mutated -> taint so the next turn
                # resyncs on a fresh session instead of reusing it.
                _ambiguous_tools = False
                try:
                    _auth_text_dbg, _auth_reas_dbg = \
                        split_envelope_parts(raw_parts)
                    # Ambiguous when streamed calls exist but final has no
                    # tool markers, or vice versa (excluding bridge parsing
                    # which is authoritative via parser.finish).
                    if not bridge:
                        _has_markers = (
                            TOOL_OPEN_XML in (_auth_text_dbg or "")
                            or TOOL_OPEN_BRACKET in (_auth_text_dbg or ""))
                        if bool(tr.call_index) != bool(_has_markers) and \
                                (text or reasoning or tr.content_emitted):
                            # Only taint when there was real streamed output
                            # to contradict (avoid tainting empty turns).
                            if tr.content_emitted or tr.reasoning_emitted:
                                _ambiguous_tools = True
                except Exception:
                    pass
                _divergent = False
                try:
                    _divergent = tr.is_divergent(text, reasoning)
                except Exception:
                    pass
                if _divergent or _ambiguous_tools:
                    try:
                        registry.mark_tainted(sid, kind)
                    except Exception:
                        pass
                    slog(cid, "session-tainted",
                         sid=_sid_short,
                         reason=("divergent-final"
                                 if _divergent else "ambiguous-tools"))
                else:
                    registry.record(rec["digests"], kind, sid, text,
                                    tools_sig=tools_sig)
                tokens = info.get("tokens") or {}
                slog(cid, "active",
                     sid=_sid_short, events=events_seen,
                     stalls=stalls, reconciles=reconciles,
                     last_event=last_event_type,
                     bytes_content=len(tr.content_emitted),
                     bytes_reasoning=len(tr.reasoning_emitted))
                if DEBUG:
                    _raw_dbg, _reas_dbg = split_envelope_parts(raw_parts)
                    _debug_tool_turn(
                        model=model, transport="serve", mode=mode,
                        tools_list=tools_list, tool_choice=tool_choice,
                        raw=_raw_dbg, calls=[
                            {"id": c.get("id"),
                             "function": c.get("function") or {}}
                            for c in tr.emitted_calls],
                        reasoning=tr.reasoning_emitted or _reas_dbg,
                        incoming_msgs=msgs)
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
                _debug_tool_turn(
                    model=model, transport="run", mode=mode,
                    tools_list=tools_list, tool_choice=tool_choice,
                    raw=raw, calls=[
                        {"id": c.get("id"),
                         "function": c.get("function") or {}}
                        for c in tr.emitted_calls],
                    reasoning=tr.reasoning_emitted,
                    incoming_msgs=msgs)

            finish = "tool_calls" if tr.call_index else "stop"
            if tr.call_index:
                log(f"tool_calls: {tr.call_index} (finish=tool_calls)")
            if tr.reasoning_emitted:
                log(f"reasoning: {len(tr.reasoning_emitted)} chars")
            await resp.write_eof()
            slog(cid, "done",
                 sid=_sid_short, content=len(tr.content_emitted),
                 reasoning=len(tr.reasoning_emitted),
                 tool_calls=tr.call_index, finish=finish)
            log(f"-> done (stream, content={len(tr.content_emitted)}, "
                f"reasoning={len(tr.reasoning_emitted)}, "
                f"tool_calls={tr.call_index}, finish={finish})")
        except Exception as e:
            # The HTTP status is already 200 — never fabricate assistant
            # content or finish_reason=stop (Hermes would treat that as a
            # successful provider answer). Emit a structured error event
            # the client can identify, then terminate cleanly. [DONE] here
            # terminates the failed turn; it is NOT a successful finish
            # (no finish_reason was ever sent). Successful turns emit [DONE]
            # exactly once via finalize_*; failed turns emit error + [DONE]
            # exactly once here — never both.
            slog(cid, "stream-error",
                 sid=_sid_short, error=str(e)[:200],
                 streamed_c=len(tr.content_emitted),
                 streamed_r=len(tr.reasoning_emitted),
                 streamed_calls=tr.call_index)
            log(f"STREAM ERROR: {e}")
            # TAINTED: backend output began but the turn failed -> the
            # persistent session may be partially mutated. Next request must
            # resync on a fresh session even if the Hermes digest matches.
            try:
                _had_output = bool(tr.content_emitted or
                                   tr.reasoning_emitted or tr.call_index)
                # events_seen only exists on the serve path; be lenient.
                try:
                    _had_output = _had_output or (events_seen > 0)
                except NameError:
                    pass
                if _had_output and isinstance(sid, str) and sid:
                    try:
                        registry.mark_tainted(sid, kind)
                    except Exception:
                        pass
                    slog(cid, "session-tainted",
                         sid=_sid_short, reason="partial-turn-failed")
            except Exception:
                pass
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
    ap.add_argument("--debug", action="store_true",
                    default=os.environ.get("OPENCODE_PROXY_DEBUG", "")
                    .lower() in ("1", "true", "yes", "on"),
                    help="verbose per-turn tool-call diagnostics (no secrets)")
    ap.add_argument("--event-idle-timeout", type=float,
                    default=float(os.environ.get(
                        "OPENCODE_EVENT_IDLE_TIMEOUT", "150")),
                    help="seconds without /event bytes/events before the "
                         "stream is declared stale (default 150; "
                         "conservative for long thinking)")
    ap.add_argument("--event-reconnect-initial", type=float,
                    default=float(os.environ.get(
                        "OPENCODE_EVENT_RECONNECT_INITIAL", "1.0")),
                    help="initial /event reconnect backoff in seconds")
    ap.add_argument("--event-reconnect-max", type=float,
                    default=float(os.environ.get(
                        "OPENCODE_EVENT_RECONNECT_MAX", "16.0")),
                    help="maximum /event reconnect backoff in seconds")
    ap.add_argument("--buffered-delta-ttl", type=float,
                    default=float(os.environ.get(
                        "OPENCODE_BUFFERED_DELTA_TTL", "120")),
                    help="bounded lifetime for deltas buffered while part "
                         "type is unknown (default 120s)")
    ap.add_argument("--verbose-payloads", action="store_true",
                    default=os.environ.get(
                        "OPENCODE_PROXY_VERBOSE_PAYLOADS", "")
                    .lower() in ("1", "true", "yes", "on"),
                    help="opt-in full SSE payload logging (may contain "
                         "private data; off by default)")
    args = ap.parse_args()
    global DEBUG, VERBOSE_PAYLOADS
    global EVENT_STREAM_IDLE_TIMEOUT, EVENT_RECONNECT_INITIAL
    global EVENT_RECONNECT_MAX, BUFFERED_DELTA_TTL
    if args.debug:
        DEBUG = True
    if args.verbose_payloads:
        VERBOSE_PAYLOADS = True
    EVENT_STREAM_IDLE_TIMEOUT = float(args.event_idle_timeout)
    EVENT_RECONNECT_INITIAL = float(args.event_reconnect_initial)
    EVENT_RECONNECT_MAX = float(args.event_reconnect_max)
    BUFFERED_DELTA_TTL = float(args.buffered_delta_ttl)

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
