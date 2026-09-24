#!/usr/bin/env python3
"""Wire-level SSE tests: mocked EventHub + serve/run backends, full HTTP path.

Exercises the proxy through ``make_app`` + aiohttp TestServer with the real
``handle_chat`` code path, but with:

  * ``ServeBackend.ensure/create_session/prompt`` stubbed to canned envelopes;
  * ``EventHub.subscribe`` replaced by a queue the test feeds directly
    (no live ``/event`` connection);
  * ``RunBackend.run`` stubbed for the fallback path.

Pins the OpenAI SSE wire contract:
  * role chunk first, ``[DONE]`` last;
  * reasoning arrives only as ``delta.reasoning_content`` (never content);
  * mid-stream failures emit a structured ``{"error": ...}`` event and NO
    fabricated content / finish_reason (never ``[proxy error]`` text);
  * pre-stream failures surface as HTTP 502 JSON;
  * tool_calls chunks precede ``finish_reason: "tool_calls"``;
  * opencode#26924 race: deltas buffered until part/message metadata lands.

Run:  python3 -m unittest discover -s tests -v
"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

import opencode_proxy as proxy  # noqa: E402
from opencode_proxy import make_app  # noqa: E402

SID = "sess_test_1"
MID = "msg_assistant_1"


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


class FakeEventHub:
    """Drop-in EventHub: subscribe() returns a queue the test controls."""

    last = None  # most recent instance (fake_prompt pushes onto its queue)

    def __init__(self, *args, **kwargs):
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


class WireCase(unittest.IsolatedAsyncioTestCase):
    """Shared harness: app + TestClient + stubbed serve/run/hub."""

    async def asyncSetUp(self):
        FakeEventHub.last = None
        self.events: list[dict] = []      # served onto the hub by fake_prompt
        self.envelope = _envelope(text="Hello world")
        self.prompt_error: Exception | None = None
        self.ensure_ok = True
        self.create_error: Exception | None = None
        self.run_result = None            # (text, reasoning, sid) or exception
        self.run_deltas: list[tuple[str, str]] = []

        patches = [
            mock.patch.object(proxy, "EventHub", FakeEventHub),
            mock.patch.object(proxy.ServeBackend, "ensure",
                              self._fake_ensure),
            mock.patch.object(proxy.ServeBackend, "create_session",
                              self._fake_create_session),
            mock.patch.object(proxy.ServeBackend, "prompt",
                              self._fake_prompt),
            mock.patch.object(proxy.RunBackend, "run", self._fake_run),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        app = make_app("/nonexistent/opencode", str(Path(__file__).parent.parent),
                       agent="", serve_port=18790)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    # NOTE: patching a class attr with a bound method means the proxy's
    # `backend.method(...)` call does NOT pass the backend instance — only
    # the explicit arguments. `self` here is the test case.

    async def _fake_ensure(self):
        return self.ensure_ok

    async def _fake_create_session(self, model):
        if self.create_error is not None:
            raise self.create_error
        return SID

    async def _fake_prompt(self, sid, prompt, model):
        hub = FakeEventHub.last
        for evt in self.events:
            hub.queue.put_nowait(evt)
        if self.prompt_error is not None:
            raise self.prompt_error
        return self.envelope

    async def _fake_run(self, model, prompt, session_id, on_delta=None):
        if isinstance(self.run_result, Exception):
            raise self.run_result
        for delta, field in self.run_deltas:
            if on_delta is not None:
                await on_delta(delta, field)
        if self.run_result is None:
            return "", "", session_id
        return self.run_result

    async def post_chat(self, body):
        return await self.client.post("/v1/chat/completions", json=body)

    async def read_sse(self, resp) -> list[str]:
        """Collect SSE `data:` payloads until the stream ends."""
        payloads: list[str] = []
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                payloads.append(line[6:])
        return payloads

    @staticmethod
    def objs(payloads):
        return [json.loads(p) for p in payloads if p != "[DONE]"]


class TestStreamServeReasoning(WireCase):
    async def test_reasoning_then_content_then_stop(self):
        self.events = [
            _msg_updated(),
            _part_updated("p_r", "reasoning", ""),
            _part_updated("p_t", "text", ""),
            _part_delta("p_r", "Let me think"),
            _part_delta("p_r", " hard."),
            _part_delta("p_t", "Hello "),
            _part_delta("p_t", "world"),
            _part_updated("p_r", "reasoning", "Let me think hard."),
            _part_updated("p_t", "text", "Hello world"),
        ]
        self.envelope = _envelope(text="Hello world",
                                  reasoning="Let me think hard.",
                                  tokens={"input": 10, "output": 5})
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        self.assertEqual(payloads[-1], "[DONE]")
        objs = self.objs(payloads)

        # first chunk: role
        first = objs[0]["choices"][0]["delta"]
        self.assertEqual(first.get("role"), "assistant")

        reasoning = ""
        content = ""
        finishes = []
        for o in objs:
            ch = o["choices"][0]
            d = ch.get("delta") or {}
            if "reasoning_content" in d:
                reasoning += d["reasoning_content"] or ""
            if "content" in d:
                content += d["content"] or ""
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])

        self.assertEqual(reasoning, "Let me think hard.")
        self.assertEqual(content, "Hello world")  # no duplicate snapshot bytes
        self.assertNotIn("think", content)        # never leaked into content
        self.assertEqual(finishes, ["stop"])

    async def test_delta_before_part_updated_race(self):
        """opencode#26924: stream delta arrives before message.part.updated."""
        self.events = [
            _msg_updated(),
            _part_delta("p_t", "Buffered "),     # before part metadata
            _part_updated("p_t", "text", ""),    # metadata lands (empty snap)
            _part_delta("p_t", "content"),
            _part_updated("p_t", "text", "Buffered content"),
        ]
        self.envelope = _envelope(text="Buffered content")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        content = ""
        for o in self.objs(payloads):
            content += (o["choices"][0].get("delta") or {}).get("content") or ""
        self.assertEqual(content, "Buffered content")

    async def test_delta_before_message_updated_race(self):
        self.events = [
            _part_updated("p_t", "text", ""),    # before message.updated
            _part_delta("p_t", "Early"),         # held in mid_pending
            _msg_updated(),                      # flushes pending
            _part_updated("p_t", "text", "Early"),
        ]
        self.envelope = _envelope(text="Early")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        payloads = await self.read_sse(resp)
        content = ""
        for o in self.objs(payloads):
            content += (o["choices"][0].get("delta") or {}).get("content") or ""
        self.assertEqual(content, "Early")

    async def test_foreign_session_events_ignored(self):
        self.events = [
            _msg_updated(sid="other-session"),
            _part_delta("p_t", "WRONG", sid="other-session"),
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "right"),
            _part_updated("p_t", "text", "right"),
        ]
        self.envelope = _envelope(text="right")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        payloads = await self.read_sse(resp)
        content = ""
        for o in self.objs(payloads):
            content += (o["choices"][0].get("delta") or {}).get("content") or ""
        self.assertEqual(content, "right")


class TestStreamServeTools(WireCase):
    async def test_tool_calls_before_finish_tool_calls(self):
        block = ('[tool_call]\n'
                 '{"name":"ls","arguments":{"path":"."}}\n'
                 '[/tool_call]')
        raw = "Checking " + block + " done"
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "Checking "),
            _part_delta("p_t", block[:20]),
            _part_delta("p_t", block[20:]),
            _part_delta("p_t", " done"),
            _part_updated("p_t", "text", raw),
        ]
        self.envelope = _envelope(text=raw)
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "ls"}],
            "stream": True,
            "tools": [{"type": "function", "function": {
                "name": "ls", "parameters": {"type": "object"}}}],
            "tool_choice": "auto",
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        objs = self.objs(payloads)

        content = ""
        tool_calls = []
        finishes = []
        for i, o in enumerate(objs):
            ch = o["choices"][0]
            d = ch.get("delta") or {}
            if "content" in d:
                content += d["content"] or ""
            if d.get("tool_calls"):
                # every tool_calls chunk must precede any finish_reason
                self.assertFalse(finishes)
                tool_calls.extend(d["tool_calls"])
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])

        self.assertNotIn("[tool_call]", content)  # bridge strips the block
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "ls")
        self.assertTrue(tool_calls[0]["id"].startswith("call_"))
        self.assertEqual(finishes, ["tool_calls"])
        self.assertEqual(payloads[-1], "[DONE]")


class TestStreamErrors(WireCase):
    async def test_prestream_failure_is_http_502_json(self):
        self.create_error = RuntimeError("session create failed")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 502)
        data = await resp.json()
        self.assertIn("error", data)
        self.assertIn("session create failed", data["error"]["message"])

    async def test_midstream_failure_structured_error_no_fabrication(self):
        self.events = [
            _msg_updated(),
            _part_updated("p_t", "text", ""),
            _part_delta("p_t", "partial"),
        ]
        self.prompt_error = RuntimeError("backend exploded")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)  # headers already sent
        payloads = await self.read_sse(resp)
        self.assertEqual(payloads[-1], "[DONE]")

        # structured error event, not assistant content
        err = json.loads(payloads[-2])
        self.assertIn("error", err)
        self.assertIn("backend exploded", err["error"]["message"])
        self.assertEqual(err["error"]["type"], "server_error")

        # never fabricate content or a finish_reason after failure
        joined = "\n".join(payloads)
        self.assertNotIn("[proxy error]", joined)
        for p in payloads[:-1]:
            if p == "[DONE]":
                continue
            o = json.loads(p)
            fr = o.get("choices", [{}])[0].get("finish_reason")
            self.assertIsNone(fr, f"unexpected finish_reason {fr}")

    async def test_empty_run_response_is_structured_error(self):
        self.ensure_ok = False
        self.run_result = ("", "", SID)  # empty response
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        err = json.loads(payloads[-2])
        self.assertIn("error", err)
        self.assertNotIn("[proxy error]", "\n".join(payloads))


class TestBlockingPath(WireCase):
    async def test_blocking_returns_reasoning_content(self):
        self.envelope = _envelope(text="The answer",
                                  reasoning="Step by step",
                                  tokens={"input": 7, "output": 3})
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        msg = data["choices"][0]["message"]
        self.assertEqual(msg["content"], "The answer")
        self.assertEqual(msg["reasoning_content"], "Step by step")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["usage"]["prompt_tokens"], 7)
        self.assertEqual(data["usage"]["completion_tokens"], 3)

    async def test_blocking_backend_error_is_502(self):
        self.envelope = _envelope(error={"data": {"message": "upstream 403"}})
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        self.assertEqual(resp.status, 502)


class TestStreamRunFallback(WireCase):
    async def test_run_reasoning_and_content_deltas(self):
        self.ensure_ok = False
        self.run_deltas = [("pondering", "reasoning"),
                           ("Answer ", "text"),
                           ("here", "text")]
        self.run_result = ("Answer here", "pondering", SID)
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        self.assertEqual(payloads[-1], "[DONE]")

        reasoning = ""
        content = ""
        finishes = []
        for o in self.objs(payloads):
            ch = o["choices"][0]
            d = ch.get("delta") or {}
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])

        self.assertEqual(reasoning, "pondering")
        self.assertEqual(content, "Answer here")  # finalize tail, no dupes
        self.assertEqual(finishes, ["stop"])
        self.assertNotIn("pondering", content)

    async def test_run_tool_calls_bridge(self):
        self.ensure_ok = False
        block = ('[tool_call]\n'
                 '{"name":"grep","arguments":{"pattern":"x"}}\n'
                 '[/tool_call]')
        self.run_deltas = [("Running ", "text"), (block, "text")]
        self.run_result = ("Running " + block, "", SID)
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "grep x"}],
            "stream": True,
            "tools": [{"type": "function", "function": {
                "name": "grep", "parameters": {"type": "object"}}}],
        })
        payloads = await self.read_sse(resp)
        objs = self.objs(payloads)
        content = ""
        tool_calls = []
        finishes = []
        for o in objs:
            ch = o["choices"][0]
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            if d.get("tool_calls"):
                self.assertFalse(finishes)
                tool_calls.extend(d["tool_calls"])
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])
        self.assertNotIn("[tool_call]", content)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "grep")
        self.assertEqual(finishes, ["tool_calls"])


if __name__ == "__main__":
    unittest.main()
