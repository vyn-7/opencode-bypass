#!/usr/bin/env python3
"""Streaming transport/recovery tests: stall, disconnect, reconcile, taint.

Covers the 26 required failure modes at the SSE/event boundary:

  1. normal reasoning stream
  2. normal content stream
  3. reasoning followed by content
  4. reasoning followed by tool call
  5. tool call followed by tool result
  6. multiple tool calls
  7. message.part.delta before message.part.updated
  8. message.part.updated missing (bounded buffer + session resolve)
  9. /event disconnect during reasoning
 10. /event disconnect during visible content
 11. /event disconnect during tool-call generation
 12. /event remains connected but emits no bytes/events
 13. reconnect after inactivity
 14. reconnect does not duplicate already delivered content
 15. reconciliation recovers missing reasoning
 16. reconciliation recovers missing text
 17. reconciliation recovers missing tool call
 18. reconciliation does not duplicate tool call
 19. final message reconciliation does not duplicate streamed output
 20. tainted backend session forces safe resynchronization
 21. Hermes tool result continues correctly after resynchronization
 22. proxy restart does not corrupt registry state
 23. malformed SSE event does not kill the entire EventHub
 24. malformed event does not produce fabricated assistant content
 25. real HTTP provider failure becomes a provider failure
 26. successful response ends with exactly one valid [DONE]

Plus the full agent-loop integration test (with injected interruption):

  USER "Read package.json..." -> reasoning -> <tool_call> -> Hermes exec ->
  tool result -> reasoning -> final answer (and the same with DISCONNECT).

Pipeline under test (realistic, not helper-only):

  EventHub -> translator -> queue -> chat handler -> OpenAI SSE output

Run: .venv/bin/python -m unittest tests.test_recovery -v
"""
import asyncio
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import opencode_proxy as proxy  # noqa: E402
from opencode_proxy import (  # noqa: E402
    TOOL_CLOSE_XML,
    TOOL_OPEN_XML,
    EventHub,
    Registry,
    StreamTranslator,
    digests_of,
    lookup_entry,
)

SID = "sess_rec_1"
MID = "msg_rec_1"


def _msg_updated(role="assistant", mid=MID, sid=SID):
    return {"type": "message.updated", "properties": {
        "sessionID": sid, "info": {"id": mid, "role": role}}}


def _part_updated(pid, ptype, text="", mid=MID, sid=SID):
    return {"type": "message.part.updated", "properties": {
        "sessionID": sid,
        "part": {"id": pid, "messageID": mid, "type": ptype, "text": text}}}


def _part_delta(pid, delta, field="text", mid=MID, sid=SID):
    return {"type": "message.part.delta", "properties": {
        "sessionID": sid, "messageID": mid, "partID": pid,
        "delta": delta, "field": field}}


def _envelope(text="", reasoning="", tokens=None, error=None):
    parts = []
    if reasoning:
        parts.append({"id": "p_reason", "type": "reasoning", "text": reasoning})
    if text:
        parts.append({"id": "p_text", "type": "text", "text": text})
    info = {"tokens": tokens or {}}
    if error is not None:
        info["error"] = error
    return {"info": info, "parts": parts}


def _fetch_messages(text="", reasoning=""):
    """Authoritative GET /session/{sid}/message payload."""
    parts = []
    if reasoning:
        parts.append({"id": "p_reason", "type": "reasoning",
                      "text": reasoning, "sessionID": SID,
                      "messageID": MID})
    if text:
        parts.append({"id": "p_text", "type": "text",
                      "text": text, "sessionID": SID, "messageID": MID})
    return [{"info": {"id": MID, "role": "assistant", "sessionID": SID},
             "parts": parts}]


def _content_of(payloads):
    out = ""
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        try:
            o = json.loads(p)
        except Exception:
            continue
        if "error" in o:
            continue
        d = (o.get("choices") or [{}])[0].get("delta") or {}
        out += d.get("content") or ""
    return out


def _reasoning_of(payloads):
    out = ""
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        try:
            o = json.loads(p)
        except Exception:
            continue
        if "error" in o:
            continue
        d = (o.get("choices") or [{}])[0].get("delta") or {}
        out += d.get("reasoning_content") or ""
    return out


def _calls_of(payloads):
    out = []
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        try:
            o = json.loads(p)
        except Exception:
            continue
        d = (o.get("choices") or [{}])[0].get("delta") or {}
        out.extend(d.get("tool_calls") or [])
    return out


def _finishes_of(payloads):
    out = []
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        try:
            o = json.loads(p)
        except Exception:
            continue
        if "error" in o:
            continue
        fr = (o.get("choices") or [{}])[0].get("finish_reason")
        if fr:
            out.append(fr)
    return out


def _xml_block(name="read_file", args=None):
    a = args if args is not None else {"path": "package.json"}
    return (f"{TOOL_OPEN_XML}\n"
            f"{json.dumps({'name': name, 'arguments': a})}\n"
            f"{TOOL_CLOSE_XML}")


# --------------------------------------------------------------------------
# Translator-level: normal streams (1-6)
# --------------------------------------------------------------------------

class TestNormalStreams(unittest.TestCase):
    def test_01_normal_reasoning_stream(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("r1", "reasoning", ""), SID)
        out = tr.handle_session(_part_delta("r1", "thinking..."), SID)
        self.assertEqual(_reasoning_of(out), "thinking...")
        self.assertEqual(_content_of(out), "")
        payloads, text, reasoning = tr.finalize_text("", "thinking...")
        self.assertEqual(reasoning, "thinking...")
        self.assertEqual(_finishes_of(payloads), ["stop"])

    def test_02_normal_content_stream(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        out = tr.handle_session(_part_delta("t1", "hello"), SID)
        self.assertEqual(_content_of(out), "hello")
        self.assertEqual(_reasoning_of(out), "")
        payloads, text, _ = tr.finalize_text("hello", "")
        self.assertEqual(text, "hello")
        self.assertEqual(_finishes_of(payloads), ["stop"])

    def test_03_reasoning_followed_by_content(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("r1", "reasoning", ""), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        tr.handle_session(_part_delta("r1", "think"), SID)
        out = tr.handle_session(_part_delta("t1", "answer"), SID)
        self.assertEqual(_content_of(out), "answer")
        payloads, text, reasoning = tr.finalize_text("answer", "think")
        self.assertEqual(text, "answer")
        self.assertEqual(reasoning, "think")
        # wire order: reasoning never inside content
        self.assertNotIn("think", text)

    def test_04_reasoning_followed_by_tool_call(self):
        tr = StreamTranslator("cid", "m", True)
        block = _xml_block("ls", {"path": "."})
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("r1", "reasoning", ""), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        tr.handle_session(_part_delta("r1", "need ls"), SID)
        out = tr.handle_session(_part_delta("t1", "go " + block), SID)
        self.assertEqual(_reasoning_of(out), "")
        calls = _calls_of(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "ls")
        payloads, _, _ = tr.finalize_text("go " + block, "need ls")
        self.assertEqual(_finishes_of(payloads), ["tool_calls"])

    def test_05_tool_call_followed_by_tool_result_flatten(self):
        # tool_calls survive into the next-turn transcript with exact ids
        msgs = [
            {"role": "assistant", "content": "checking",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": '{"path":"package.json"}'}}]},
            {"role": "tool", "name": "read_file", "tool_call_id": "call_1",
             "content": '{"version":"1.0"}'},
        ]
        out = proxy.flatten_history(msgs)
        self.assertIn("(call call_1)", out)
        self.assertIn("[tool result: read_file] (call call_1)", out)

    def test_06_multiple_tool_calls(self):
        tr = StreamTranslator("cid", "m", True)
        b1 = _xml_block("a", {})
        b2 = _xml_block("b", {})
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        out = tr.handle_session(_part_delta("t1", b1 + "mid" + b2), SID)
        calls = _calls_of(out)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])
        self.assertEqual(_content_of(out), "mid")


# --------------------------------------------------------------------------
# Part ordering (7-8)
# --------------------------------------------------------------------------

class TestPartOrdering(unittest.TestCase):
    def test_07_delta_before_updated(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        out = tr.handle_session(_part_delta("p1", "Buffered "), SID)
        self.assertEqual(out, [])
        out = tr.handle_session(_part_updated("p1", "text", ""), SID)
        self.assertEqual(_content_of(out), "Buffered ")
        out = tr.handle_session(_part_delta("p1", "content"), SID)
        self.assertEqual(_content_of(out), "content")

    def test_08_updated_missing_bounded_buffer_and_resolve(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        # delta arrives, metadata never arrives -> buffered, not emitted
        out = tr.handle_session(_part_delta("pX", "held-bytes"), SID)
        self.assertEqual(out, [])
        self.assertIn("pX", tr.unknown_pending())
        # bounded lifetime: fresh buffer is not stale with default TTL
        self.assertFalse(tr.has_stale_unknown())
        # force staleness with ttl=0
        self.assertTrue(tr.has_stale_unknown(now=time.monotonic(), ttl=0))
        # recovery consults authoritative state: reasoning vs text separate
        parts = [{"id": "pX", "type": "reasoning", "text": "held-bytes",
                  "sessionID": SID, "messageID": MID}]
        out = tr.resolve_unknown_via_session(parts)
        self.assertEqual(_reasoning_of(out), "held-bytes")
        self.assertEqual(_content_of(out), "")
        # text case
        tr2 = StreamTranslator("cid2", "m", False)
        tr2.handle_session(_msg_updated(), SID)
        tr2.handle_session(_part_delta("pY", "visible"), SID)
        out2 = tr2.resolve_unknown_via_session(
            [{"id": "pY", "type": "text", "text": "visible",
              "sessionID": SID, "messageID": MID}])
        self.assertEqual(_content_of(out2), "visible")
        self.assertEqual(_reasoning_of(out2), "")
        # unknown id stays buffered (never fabricated as content)
        tr3 = StreamTranslator("cid3", "m", False)
        tr3.handle_session(_msg_updated(), SID)
        tr3.handle_session(_part_delta("pZ", "mystery"), SID)
        out3 = tr3.resolve_unknown_via_session(
            [{"id": "other", "type": "text", "text": "x"}])
        self.assertEqual(out3, [])
        self.assertIn("pZ", tr3.unknown_pending())


# --------------------------------------------------------------------------
# Disconnect / stall / reconnect via translator + session fetch (9-18)
# --------------------------------------------------------------------------

class TestDisconnectReconcile(unittest.TestCase):
    def test_09_disconnect_during_reasoning_recovered(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("r1", "reasoning", ""), SID)
        tr.handle_session(_part_delta("r1", "part-one "), SID)
        # DISCONNECT: no more events; authoritative fetch has full reasoning
        out = tr.reconcile_with_messages(
            _fetch_messages(reasoning="part-one part-two"))
        self.assertEqual(_reasoning_of(out), "part-two")
        self.assertEqual(_content_of(out), "")
        payloads, _, reasoning = tr.finalize_text("", "part-one part-two")
        self.assertEqual(_reasoning_of(payloads), "")
        self.assertEqual(reasoning, "part-one part-two")

    def test_10_disconnect_during_content_recovered(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        tr.handle_session(_part_delta("t1", "Hello "), SID)
        out = tr.reconcile_with_messages(_fetch_messages(text="Hello world"))
        self.assertEqual(_content_of(out), "world")
        payloads, text, _ = tr.finalize_text("Hello world", "")
        self.assertEqual(_content_of(payloads), "")
        self.assertEqual(text, "Hello world")

    def test_11_disconnect_during_tool_call_recovered(self):
        tr = StreamTranslator("cid", "m", True)
        block = _xml_block("read_file", {"path": "package.json"})
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        # only a prefix of the tool block arrived before disconnect
        tr.handle_session(_part_delta("t1", "Read " + block[:15]), SID)
        # authoritative fetch has the complete raw text with full block
        full = "Read " + block
        out = tr.reconcile_with_messages(_fetch_messages(text=full))
        calls = _calls_of(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        # finalize must not duplicate the tool call
        payloads, _, _ = tr.finalize_text(full, "")
        self.assertEqual(_calls_of(payloads), [])
        self.assertEqual(_finishes_of(payloads), ["tool_calls"])

    def test_14_reconnect_no_duplicate(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        tr.handle_session(_part_delta("t1", "Hello"), SID)
        # reconnect replays the same snapshot (already delivered) + extension
        out = tr.handle_session(_part_updated("t1", "text", "Hello world"),
                                SID)
        self.assertEqual(_content_of(out), " world")
        # replaying the identical snapshot again emits nothing
        out2 = tr.handle_session(_part_updated("t1", "text", "Hello world"),
                                 SID)
        self.assertEqual(_content_of(out2), "")

    def test_15_recover_missing_reasoning(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("r1", "reasoning", ""), SID)
        # no deltas arrived at all; fetch recovers everything
        out = tr.reconcile_with_messages(
            _fetch_messages(reasoning="full-thought"))
        self.assertEqual(_reasoning_of(out), "full-thought")

    def test_16_recover_missing_text(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        out = tr.reconcile_with_messages(_fetch_messages(text="full-answer"))
        self.assertEqual(_content_of(out), "full-answer")

    def test_17_recover_missing_tool_call(self):
        tr = StreamTranslator("cid", "m", True)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        block = _xml_block("grep", {"pattern": "x"})
        full = "Run " + block
        out = tr.reconcile_with_messages(_fetch_messages(text=full))
        calls = _calls_of(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "grep")

    def test_18_no_duplicate_tool_call_on_reconcile(self):
        tr = StreamTranslator("cid", "m", True)
        block = _xml_block("ls", {"path": "."})
        full = "Hi " + block
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        out = tr.handle_session(_part_delta("t1", full), SID)
        self.assertEqual(len(_calls_of(out)), 1)
        # reconnect + fetch with identical authoritative text: no re-emit
        out2 = tr.reconcile_with_messages(_fetch_messages(text=full))
        self.assertEqual(_calls_of(out2), [])
        self.assertEqual(_content_of(out2), "")

    def test_19_final_no_duplicate(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("t1", "text", ""), SID)
        tr.handle_session(_part_delta("t1", "Hello world"), SID)
        payloads, text, _ = tr.finalize_text("Hello world", "")
        self.assertEqual(_content_of(payloads), "")
        self.assertEqual(text, "Hello world")
        self.assertEqual(_finishes_of(payloads), ["stop"])
        self.assertEqual(payloads[-1], "[DONE]")


# --------------------------------------------------------------------------
# EventHub unit tests at the byte level (12, 13, 23, 24)
# --------------------------------------------------------------------------

class _FakeContent:
    """Queue of byte-chunks; readany() blocks until a chunk or EOF."""

    def __init__(self, chunks, delay_per_chunk=0.0):
        # chunks: list of bytes or Exception to raise
        self._chunks = list(chunks)
        self._delay = delay_per_chunk

    async def readany(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._chunks:
            await asyncio.sleep(3600)  # stall forever (no bytes)
            return b""
        item = self._chunks.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeResp:
    def __init__(self, content, status=200):
        self.content = content
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Each get() returns the next scripted response."""

    def __init__(self, resps):
        self._resps = list(resps)
        self.get_calls = 0

    def get(self, *a, **k):
        self.get_calls += 1
        idx = min(self.get_calls - 1, len(self._resps) - 1)
        return self._resps[idx]


def _sse_evt(etype, props):
    return "data: " + json.dumps({"type": etype,
                                  "properties": props}) + "\n\n"


class TestEventHub(unittest.IsolatedAsyncioTestCase):
    async def test_23_malformed_does_not_kill_hub(self):
        good = _sse_evt("message.updated",
                        {"sessionID": SID, "info": {"id": MID,
                                                   "role": "assistant"}})
        chunks = [b"data: not-json\n\n",
                  good.encode(),
                  b"data: {bad\n\n",
                  _sse_evt("message.part.delta",
                           {"sessionID": SID, "messageID": MID,
                            "partID": "p1", "delta": "hi",
                            "field": "text"}).encode()]
        sess = _FakeSession([_FakeResp(_FakeContent(chunks))])
        hub = EventHub("http://x", {}, idle_timeout=5.0)
        hub.set_session(sess)
        q = hub.subscribe()
        try:
            got = []
            for _ in range(2):
                got.append(await asyncio.wait_for(q.get(), timeout=2))
            self.assertEqual(len(got), 2)
            self.assertEqual(hub.stats()["malformed"], 2)
            self.assertEqual(hub.stats()["events"], 2)
        finally:
            hub.unsubscribe(q)
            if hub._task is not None:
                hub._task.cancel()
                try:
                    await hub._task
                except asyncio.CancelledError:
                    pass

    async def test_12_13_stall_triggers_reconnect(self):
        # First connection stalls (no bytes); hub must declare STALE and
        # reconnect to the second connection which delivers an event.
        good = _sse_evt("message.updated",
                        {"sessionID": SID, "info": {"id": MID,
                                                   "role": "assistant"}})
        stalled = _FakeContent([])  # readany stalls forever
        live = _FakeContent([good.encode()])
        sess = _FakeSession([_FakeResp(stalled), _FakeResp(live)])
        hub = EventHub("http://x", {}, idle_timeout=0.2,
                       reconnect_initial=0.05, reconnect_max=0.2)
        hub.set_session(sess)
        q = hub.subscribe()
        try:
            evt = await asyncio.wait_for(q.get(), timeout=3)
            self.assertEqual(evt["type"], "message.updated")
            # at least one reconnect happened (stale -> reconnect)
            await asyncio.sleep(0.1)
            self.assertGreaterEqual(hub.stats()["reconnects"], 1)
            self.assertEqual(sess.get_calls, 2)
        finally:
            hub.unsubscribe(q)
            if hub._task is not None:
                hub._task.cancel()
                try:
                    await hub._task
                except asyncio.CancelledError:
                    pass

    async def test_single_reader_no_duplicate_consumers(self):
        good = _sse_evt("message.updated",
                        {"sessionID": SID, "info": {"id": MID,
                                                   "role": "assistant"}})
        sess = _FakeSession([_FakeResp(_FakeContent([good.encode()]))])
        hub = EventHub("http://x", {}, idle_timeout=5.0)
        hub.set_session(sess)
        q1 = hub.subscribe()
        q2 = hub.subscribe()
        try:
            # only one underlying connection for two subscribers
            await asyncio.sleep(0.2)
            self.assertEqual(sess.get_calls, 1)
            e1 = await asyncio.wait_for(q1.get(), timeout=2)
            e2 = await asyncio.wait_for(q2.get(), timeout=2)
            self.assertEqual(e1, e2)  # broadcast, not duplicate reads
        finally:
            hub.unsubscribe(q1)
            hub.unsubscribe(q2)
            if hub._task is not None:
                hub._task.cancel()
                try:
                    await hub._task
                except asyncio.CancelledError:
                    pass

    async def test_24_malformed_never_fabricates_content(self):
        # Malformed bytes must never become translator content.
        tr = StreamTranslator("cid", "m", False)
        # hub would skip malformed; translator sees nothing -> no output
        self.assertEqual(tr.handle_session({"bogus": 1}, SID), [])
        self.assertEqual(tr.handle_session("not-a-dict", SID), [])
        payloads, text, reasoning = tr.finalize_text("", "")
        # finalize of empty stream is an empty (but valid) stop, not error
        self.assertEqual(text, "")
        self.assertEqual(reasoning, "")


# --------------------------------------------------------------------------
# Wire-level: chat handler + hub + translator + fetch (9-11, 25, 26, 20-21)
# --------------------------------------------------------------------------

class FakeEventHub:
    last = None

    def __init__(self, *a, **k):
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task = None
        self._subs = set()
        FakeEventHub.last = self

    def set_session(self, session) -> None:
        pass

    def subscribe(self) -> asyncio.Queue:
        return self.queue

    def unsubscribe(self, q) -> None:
        pass

    def stats(self):
        return {"conn": 1, "state": "ACTIVE", "events": 0}


class WireCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeEventHub.last = None
        self.events: list[dict] = []
        self.envelope = _envelope(text="Hello world")
        self.fetch_payload = _fetch_messages(text="Hello world")
        self.prompt_error: Exception | None = None
        self.fetch_error: Exception | None = None
        self.ensure_ok = True
        self.create_error: Exception | None = None
        self.captured_prompts: list[str] = []
        self.fetch_calls = 0

        patches = [
            mock.patch.object(proxy, "EventHub", FakeEventHub),
            mock.patch.object(proxy.ServeBackend, "ensure",
                              self._fake_ensure),
            mock.patch.object(proxy.ServeBackend, "create_session",
                              self._fake_create_session),
            mock.patch.object(proxy.ServeBackend, "prompt",
                              self._fake_prompt),
            mock.patch.object(proxy.ServeBackend, "fetch_messages",
                              self._fake_fetch),
            mock.patch.object(proxy.RunBackend, "run", self._fake_run),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        # Short timeouts so stall tests run fast; restored per-test.
        self._old_idle = proxy.EVENT_STREAM_IDLE_TIMEOUT
        self._old_ttl = proxy.BUFFERED_DELTA_TTL
        proxy.EVENT_STREAM_IDLE_TIMEOUT = 0.4
        proxy.BUFFERED_DELTA_TTL = 0.3
        self.addCleanup(self._restore_timeouts)

        app = proxy.make_app("/nonexistent/opencode",
                             str(Path(__file__).parent.parent),
                             agent="", serve_port=18790)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        # Keep a handle to the app registry via captured prompts is enough;
        # for taint tests we inspect behavior (delta vs resync).

    def _restore_timeouts(self):
        proxy.EVENT_STREAM_IDLE_TIMEOUT = self._old_idle
        proxy.BUFFERED_DELTA_TTL = self._old_ttl

    async def _fake_ensure(self):
        return self.ensure_ok

    async def _fake_create_session(self, model):
        if self.create_error is not None:
            raise self.create_error
        return SID

    async def _fake_prompt(self, sid, prompt, model):
        self.captured_prompts.append(prompt)
        hub = FakeEventHub.last
        for evt in self.events:
            hub.queue.put_nowait(evt)
        if self.prompt_error is not None:
            # Simulate a stall before the failure: hold the prompt briefly
            # so the handler's watchdog path executes at least once.
            await asyncio.sleep(0.5)
            raise self.prompt_error
        return self.envelope

    async def _fake_fetch(self, sid):
        self.fetch_calls += 1
        if self.fetch_error is not None:
            raise self.fetch_error
        # tiny delay to simulate network without slowing tests
        await asyncio.sleep(0.01)
        return self.fetch_payload

    async def _fake_run(self, model, prompt, session_id, on_delta=None):
        self.captured_prompts.append(prompt)
        return "", "", session_id

    async def post_chat(self, body):
        return await self.client.post("/v1/chat/completions", json=body)

    async def read_sse(self, resp) -> list[str]:
        payloads: list[str] = []
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                payloads.append(line[6:])
        return payloads

    @staticmethod
    def objs(payloads):
        return [json.loads(p) for p in payloads if p != "[DONE]"]


class TestWireDisconnect(WireCase):
    async def test_09_wire_disconnect_during_reasoning(self):
        # Stream delivers reasoning prefix; envelope has the full string.
        # The quiet-drain + finalize path must recover the tail with no dupes.
        self.events = [
            _msg_updated(),
            _part_updated("p_r", "reasoning", ""),
            _part_delta("p_r", "Let me "),
        ]
        self.envelope = _envelope(text="done",
                                  reasoning="Let me think hard.")
        self.fetch_payload = _fetch_messages(
            text="done", reasoning="Let me think hard.")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(_reasoning_of(payloads), "Let me think hard.")
        self.assertEqual(_content_of(payloads), "done")
        self.assertEqual(
            payloads.count("[DONE]"), 1, "exactly one [DONE]")

    async def test_10_wire_disconnect_during_content(self):
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "Hello "),
        ]
        self.envelope = _envelope(text="Hello world")
        self.fetch_payload = _fetch_messages(text="Hello world")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        payloads = await self.read_sse(resp)
        self.assertEqual(_content_of(payloads), "Hello world")
        self.assertEqual(payloads.count("[DONE]"), 1)

    async def test_11_wire_disconnect_during_tool_call(self):
        block = _xml_block("read_file", {"path": "package.json"})
        full = "Reading " + block
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "Reading "),
            # tool block never arrives as deltas (disconnect); fetch has it
        ]
        self.envelope = _envelope(text=full)
        self.fetch_payload = _fetch_messages(text=full)
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "read pkg"}],
            "stream": True,
            "tools": [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object"}}}],
        })
        payloads = await self.read_sse(resp)
        objs = self.objs(payloads)
        calls = _calls_of(payloads)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        finishes = [o["choices"][0].get("finish_reason") for o in objs]
        finishes = [f for f in finishes if f]
        self.assertEqual(finishes, ["tool_calls"])
        self.assertEqual(payloads.count("[DONE]"), 1)

    async def test_25_provider_failure_is_structured_error(self):
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "partial"),
        ]
        self.prompt_error = RuntimeError("upstream 500 exploded")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        self.assertEqual(payloads[-1], "[DONE]")
        err = json.loads(payloads[-2])
        self.assertIn("error", err)
        self.assertIn("upstream 500", err["error"]["message"])
        self.assertEqual(err["error"]["type"], "server_error")
        joined = "\n".join(payloads)
        self.assertNotIn("[proxy error]", joined)
        for p in payloads[:-1]:
            if p == "[DONE]":
                continue
            o = json.loads(p)
            if "error" in o:
                continue
            self.assertIsNone(o["choices"][0].get("finish_reason"))

    async def test_26_exactly_one_done_on_success(self):
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "hi"),
            _part_updated("p_t", "text", "hi"),
        ]
        self.envelope = _envelope(text="hi")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        payloads = await self.read_sse(resp)
        self.assertEqual(payloads.count("[DONE]"), 1)
        self.assertEqual(payloads[-1], "[DONE]")
        objs = self.objs(payloads)
        self.assertEqual(objs[0]["choices"][0]["delta"].get("role"),
                         "assistant")


class TestWireTaint(WireCase):
    async def test_20_tainted_session_forces_resync(self):
        # Turn 1 fails after partial output -> session TAINTED.
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "partial"),
        ]
        self.prompt_error = RuntimeError("backend blew up mid-turn")
        resp1 = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "first question"}],
            "stream": True,
        })
        payloads1 = await self.read_sse(resp1)
        err = json.loads(payloads1[-2])
        self.assertIn("error", err)
        n_prompts_after_fail = len(self.captured_prompts)

        # Turn 2 replays the same history + one new message. Digest alone
        # would look like a safe delta, but the session is tainted so the
        # proxy must resync (fresh session + full replay), not reuse it.
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "recovered"),
            _part_updated("p_t", "text", "recovered"),
        ]
        self.envelope = _envelope(text="recovered")
        self.fetch_payload = _fetch_messages(text="recovered")
        self.prompt_error = None
        resp2 = await self.post_chat({
            "model": "big-pickle",
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "user", "content": "second question"},
            ],
            "stream": True,
        })
        self.assertEqual(resp2.status, 200)
        payloads2 = await self.read_sse(resp2)
        self.assertEqual(_content_of(payloads2), "recovered")
        # Resync proof: the second prompt contains the FULL history
        # (both questions), not just the delta tail.
        self.assertGreater(len(self.captured_prompts), n_prompts_after_fail)
        prompt2 = self.captured_prompts[-1]
        self.assertIn("first question", prompt2)
        self.assertIn("second question", prompt2)

    async def test_21_tool_result_continues_after_resync(self):
        tools = [{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object"}}}] 
        block = _xml_block("read_file", {"path": "package.json"})
        raw1 = "Reading " + block
        # Turn 1: tool call (success, establishes session cursor).
        self.events = [
            _msg_updated(mid="m1"),
            _part_updated("t1", "text", "", mid="m1"),
            _part_delta("t1", raw1, mid="m1"),
            _part_updated("t1", "text", raw1, mid="m1"),
        ]
        self.envelope = _envelope(text=raw1)
        resp1 = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user",
                          "content": "Read package.json"}],
            "stream": True, "tools": tools,
        })
        p1 = await self.read_sse(resp1)
        calls = _calls_of(p1)
        self.assertEqual(len(calls), 1)
        call_id = calls[0]["id"]

        # Simulate a failed continuation that taints the session.
        self.events = [
            _msg_updated(mid="mX"),
            _part_updated("tX", "text", "", mid="mX"),
            _part_delta("tX", "partial", mid="mX"),
        ]
        self.prompt_error = RuntimeError("mid-turn failure")
        resp_bad = await self.post_chat({
            "model": "big-pickle",
            "messages": [
                {"role": "user", "content": "Read package.json"},
                {"role": "assistant", "content": "Reading",
                 "tool_calls": [{"id": call_id, "type": "function",
                                 "function": {
                                     "name": "read_file",
                                     "arguments": '{"path":"package.json"}'}}]},
                {"role": "tool", "name": "read_file",
                 "tool_call_id": call_id, "content": '{"name":"x"}'},
            ],
            "stream": True, "tools": tools,
        })
        await self.read_sse(resp_bad)

        # Turn 3 (retry with same history + tool result): must resync and
        # the backend prompt must still carry the EXACT tool_call_id and the
        # tool result exactly once (no duplication, no orphan).
        self.events = [
            _msg_updated(mid="m2"),
            _part_updated("r2", "reasoning", "", mid="m2"),
            _part_updated("t2", "text", "", mid="m2"),
            _part_delta("r2", "Got it.", mid="m2"),
            _part_delta("t2", "Version 1.0.", mid="m2"),
            _part_updated("r2", "reasoning", "Got it.", mid="m2"),
            _part_updated("t2", "text", "Version 1.0.", mid="m2"),
        ]
        self.envelope = _envelope(text="Version 1.0.", reasoning="Got it.")
        self.fetch_payload = [
            {"info": {"id": "m2", "role": "assistant", "sessionID": SID},
             "parts": [
                 {"id": "r2", "type": "reasoning", "text": "Got it."},
                 {"id": "t2", "type": "text", "text": "Version 1.0."},
             ]}]
        self.prompt_error = None
        resp3 = await self.post_chat({
            "model": "big-pickle",
            "messages": [
                {"role": "user", "content": "Read package.json"},
                {"role": "assistant", "content": "Reading",
                 "tool_calls": [{"id": call_id, "type": "function",
                                 "function": {
                                     "name": "read_file",
                                     "arguments": '{"path":"package.json"}'}}]},
                {"role": "tool", "name": "read_file",
                 "tool_call_id": call_id, "content": '{"name":"x"}'},
            ],
            "stream": True, "tools": tools,
        })
        p3 = await self.read_sse(resp3)
        self.assertEqual(_content_of(p3), "Version 1.0.")
        prompt3 = self.captured_prompts[-1]
        self.assertIn(f"(call {call_id})", prompt3)
        # Exactly two occurrences: one for the assistant tool-call line and
        # one for the tool-result association label (no duplication).
        self.assertEqual(prompt3.count(f"(call {call_id})"), 2,
                         "tool call id must appear for call + result only")
        self.assertIn(f"[tool result: read_file] (call {call_id})", prompt3)
        self.assertEqual(
            prompt3.count(f"[tool result: read_file] (call {call_id})"), 1)

    async def test_22_restart_forces_full_replay(self):
        # A fresh Registry (proxy restart) has no cursors -> resync.
        reg = Registry()
        msgs = [{"role": "user", "content": "hello"}]
        digs = digests_of(msgs)
        self.assertIsNone(lookup_entry(reg, digs, "serve", ""))
        # After recording, the same digest matches (delta path available).
        reg.record(digs, "serve", "s1", "hi", tools_sig="")
        self.assertIsNotNone(lookup_entry(reg, digs + ["extra"], "serve",
                                          ""))
        # A brand-new registry (restart) misses again -> full replay.
        reg2 = Registry()
        self.assertIsNone(lookup_entry(reg2, digs + ["extra"], "serve", ""))


# --------------------------------------------------------------------------
# Full agent loop with interruption (section 16)
# --------------------------------------------------------------------------

class TestAgentLoop(WireCase):
    async def test_full_loop_then_interrupted_loop(self):
        tools = [{"type": "function", "function": {
            "name": "read_file", "description": "Read a file",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}}}}]
        # ---- Clean loop: USER -> reasoning -> tool -> result -> answer ----
        block = _xml_block("read_file", {"path": "package.json"})
        raw1 = "Checking " + block
        self.events = [
            _msg_updated(mid="m1"),
            _part_updated("r1", "reasoning", "", mid="m1"),
            _part_updated("t1", "text", "", mid="m1"),
            _part_delta("r1", "Need to read package.json.", mid="m1"),
            _part_delta("t1", "Checking ", mid="m1"),
            _part_delta("t1", block, mid="m1"),
            _part_updated("r1", "reasoning",
                          "Need to read package.json.", mid="m1"),
            _part_updated("t1", "text", raw1, mid="m1"),
        ]
        self.envelope = _envelope(text=raw1,
                                  reasoning="Need to read package.json.")
        resp1 = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user",
                          "content": "Read package.json and tell me "
                                     "the Node version."}],
            "stream": True, "tools": tools, "tool_choice": "auto",
        })
        p1 = await self.read_sse(resp1)
        reasoning1 = _reasoning_of(p1)
        content1 = _content_of(p1)
        calls1 = _calls_of(p1)
        self.assertEqual(reasoning1, "Need to read package.json.")
        self.assertNotIn("Need to read", content1)
        self.assertEqual(len(calls1), 1)
        self.assertEqual(calls1[0]["function"]["name"], "read_file")
        call_id = calls1[0]["id"]
        call_args = calls1[0]["function"]["arguments"]

        tool_output = '{"name":"demo","engines":{"node":">=20"}}'
        msgs2 = [
            {"role": "user",
             "content": "Read package.json and tell me the Node version."},
            {"role": "assistant", "content": content1,
             "reasoning_content": reasoning1,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": call_args}}]},
            {"role": "tool", "name": "read_file",
             "tool_call_id": call_id, "content": tool_output},
        ]
        raw2 = "Node >=20."
        self.events = [
            _msg_updated(mid="m2"),
            _part_updated("r2", "reasoning", "", mid="m2"),
            _part_updated("t2", "text", "", mid="m2"),
            _part_delta("r2", "Result shows >=20.", mid="m2"),
            _part_delta("t2", raw2, mid="m2"),
            _part_updated("r2", "reasoning", "Result shows >=20.", mid="m2"),
            _part_updated("t2", "text", raw2, mid="m2"),
        ]
        self.envelope = _envelope(text=raw2, reasoning="Result shows >=20.")
        resp2 = await self.post_chat({
            "model": "big-pickle", "messages": msgs2,
            "stream": True, "tools": tools, "tool_choice": "auto",
        })
        p2 = await self.read_sse(resp2)
        self.assertEqual(_reasoning_of(p2), "Result shows >=20.")
        self.assertEqual(_content_of(p2), raw2)
        self.assertEqual(p2.count("[DONE]"), 1)

        # ---- Interrupted loop: DISCONNECT during tool-call generation ----
        # New conversation so the registry path is clean.
        self.events = [
            _msg_updated(mid="n1"),
            _part_updated("rn1", "reasoning", "", mid="n1"),
            _part_updated("tn1", "text", "", mid="n1"),
            _part_delta("rn1", "Need to read. ", mid="n1"),
            _part_delta("tn1", "Checking ", mid="n1"),
            # tool block cut off by DISCONNECT/STALL; fetch reconciles it
        ]
        self.envelope = _envelope(text=raw1,
                                  reasoning="Need to read package.json.")
        self.fetch_payload = [
            {"info": {"id": "n1", "role": "assistant", "sessionID": SID},
             "parts": [
                 {"id": "rn1", "type": "reasoning",
                  "text": "Need to read package.json."},
                 {"id": "tn1", "type": "text", "text": raw1},
             ]}]
        resp3 = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user",
                          "content": "Read package.json again"}],
            "stream": True, "tools": tools, "tool_choice": "auto",
        })
        p3 = await self.read_sse(resp3)
        calls3 = _calls_of(p3)
        # Reconciliation must recover the tool call exactly once.
        self.assertEqual(len(calls3), 1)
        self.assertEqual(calls3[0]["function"]["name"], "read_file")
        self.assertNotIn(TOOL_OPEN_XML, _content_of(p3))
        self.assertEqual(p3.count("[DONE]"), 1)

        # ---- reasoning -> tool result -> DISCONNECT -> reasoning -> answer --
        msgs4 = [
            {"role": "user", "content": "Read package.json again"},
            {"role": "assistant", "content": _content_of(p3),
             "reasoning_content": _reasoning_of(p3) or "Need to read.",
             "tool_calls": [{"id": calls3[0]["id"], "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": calls3[0]["function"][
                                              "arguments"]}}]},
            {"role": "tool", "name": "read_file",
             "tool_call_id": calls3[0]["id"], "content": tool_output},
        ]
        self.events = [
            _msg_updated(mid="n2"),
            _part_updated("rn2", "reasoning", "", mid="n2"),
            _part_delta("rn2", "Result ", mid="n2"),
            # DISCONNECT before visible text; fetch has the rest
        ]
        self.envelope = _envelope(text="Node >=20.",
                                  reasoning="Result shows >=20.")
        self.fetch_payload = [
            {"info": {"id": "n2", "role": "assistant", "sessionID": SID},
             "parts": [
                 {"id": "rn2", "type": "reasoning",
                  "text": "Result shows >=20."},
                 {"id": "tn2", "type": "text", "text": "Node >=20."},
             ]}]
        resp4 = await self.post_chat({
            "model": "big-pickle", "messages": msgs4,
            "stream": True, "tools": tools, "tool_choice": "auto",
        })
        p4 = await self.read_sse(resp4)
        # Must not call the tool twice, must not lose the tool result.
        self.assertEqual(_calls_of(p4), [])
        self.assertEqual(_content_of(p4), "Node >=20.")
        self.assertEqual(_reasoning_of(p4), "Result shows >=20.")
        self.assertEqual(p4.count("[DONE]"), 1)


if __name__ == "__main__":
    unittest.main()
