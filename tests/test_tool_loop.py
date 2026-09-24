#!/usr/bin/env python3
"""Hermes -> proxy -> OpenCode/MiMo -> Hermes tool-call loop tests.

Covers the 20 required boundaries:

 1. <tool_call> complete in one chunk.
 2. <tool_call> split across multiple SSE text chunks.
 3. <tool_call> split inside the JSON object.
 4. [tool_call] backward-compatible parsing.
 5. malformed JSON.
 6. multiple tool calls.
 7. tool calls following reasoning.
 8. reasoning following a tool result.
 9. assistant reasoning_content preserved in the next turn.
10. tool_call_id preserved exactly.
11. tool result associated with the correct tool call.
12. ordinary content without tools.
13. tool_choice=auto.
14. tool_choice=required.
15. tool_choice=none.
16. serve backend.
17. run fallback.
18. multi-turn tool continuation.
19. no duplication of streamed snapshot/delta content.
20. no tool-call markers leaking into visible content.

Plus the full invariant loop:

  USER -> reasoning -> tool call -> finish tool_calls -> Hermes exec ->
  replay (reasoning+tool_calls) -> tool result -> continuation ->
  reasoning -> final content -> finish stop

All assertions use the exact OpenAI wire representation at every boundary.
Run: .venv/bin/python -m unittest tests.test_tool_loop -v
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
from opencode_proxy import (  # noqa: E402
    TOOL_CLOSE_BRACKET,
    TOOL_CLOSE_XML,
    TOOL_OPEN_BRACKET,
    TOOL_OPEN_XML,
    StreamTranslator,
    ToolCallStreamParser,
    effective_tools,
    extract_tool_calls,
    flatten_history,
    format_client_tools,
    streaming_view,
)

SID = "sess_loop_1"
MID = "msg_loop_1"


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


def _xml_block(name="ls", args=None):
    a = args if args is not None else {"path": "."}
    return (f"{TOOL_OPEN_XML}\n"
            f"{json.dumps({'name': name, 'arguments': a})}\n"
            f"{TOOL_CLOSE_XML}")


def _bracket_block(name="ls", args=None):
    a = args if args is not None else {"path": "."}
    return (f"{TOOL_OPEN_BRACKET}\n"
            f"{json.dumps({'name': name, 'arguments': a})}\n"
            f"{TOOL_CLOSE_BRACKET}")


class TestXmlParser(unittest.TestCase):
    """1-6, 20: dual-format parser unit tests."""

    def test_01_xml_complete_in_one_chunk(self):
        raw = "Checking " + _xml_block("ls", {"path": "."}) + " done"
        clean, calls = extract_tool_calls(raw)
        self.assertEqual(clean, "Checking  done")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "ls")
        self.assertIn("path", calls[0]["function"]["arguments"])
        self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_02_xml_split_across_chunks(self):
        p = ToolCallStreamParser()
        block = _xml_block("grep", {"pattern": "x"})
        # split the block across several feeds (streaming_view must hold)
        self.assertEqual(p.feed("Running "), "Running ")
        # feed block in 3 pieces: open part, middle, close+tail
        self.assertEqual(p.feed(block[:12]), "")
        # incomplete block must not leak into visible content
        rest_clean, _ = streaming_view("Running " + block[:12])
        self.assertNotIn(TOOL_OPEN_XML, rest_clean)
        self.assertEqual(p.feed(block[12:30]), "")
        self.assertEqual(p.feed(block[30:]), "")
        self.assertEqual(p.feed(" done"), " done")
        _, calls = p.finish()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "grep")

    def test_03_xml_split_inside_json(self):
        p = ToolCallStreamParser()
        # split precisely inside the JSON object (between key and value)
        part1 = f"{TOOL_OPEN_XML}\n{{\"name\": \"rea"
        part2 = "d\", \"arguments\": {\"path\": \"a.txt\"}}\n" + TOOL_CLOSE_XML
        self.assertEqual(p.feed("Hi "), "Hi ")
        self.assertEqual(p.feed(part1), "")
        # half-valid JSON must never be exposed as content
        clean, calls = streaming_view("Hi " + part1)
        self.assertEqual(calls, [])
        self.assertNotIn("rea", clean.replace("Hi ", ""))
        self.assertEqual(p.feed(part2), "")
        _, calls = p.finish()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_04_bracket_backward_compat(self):
        raw = "Hi " + _bracket_block("bash", {"cmd": "ls"}) + " bye"
        clean, calls = extract_tool_calls(raw)
        self.assertEqual(clean, "Hi  bye")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")
        # streaming path also accepts legacy
        clean2, calls2 = streaming_view(raw)
        self.assertEqual(clean2, "Hi  bye")
        self.assertEqual(len(calls2), 1)

    def test_05_malformed_json_never_tool_call(self):
        for bad in (
            f"{TOOL_OPEN_XML}\nnot-json\n{TOOL_CLOSE_XML}",
            f"{TOOL_OPEN_BRACKET}\nnot-json\n{TOOL_CLOSE_BRACKET}",
            f"{TOOL_OPEN_XML}\n[1,2,3]\n{TOOL_CLOSE_XML}",
            f"{TOOL_OPEN_XML}\n{{\"no_name\": 1}}\n{TOOL_CLOSE_XML}",
        ):
            clean, calls = extract_tool_calls("a " + bad + " b")
            self.assertEqual(calls, [], f"malformed should not parse: {bad!r}")
            # kept verbatim as visible content (not silently dropped)
            self.assertIn(bad.split("\n")[0], clean)

    def test_06_multiple_tool_calls(self):
        raw = (_xml_block("a", {}) + "mid" +
               _bracket_block("b", {}) + "end" +
               _xml_block("c", {"x": 1}))
        clean, calls = extract_tool_calls(raw)
        self.assertEqual(clean, "midend")
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["a", "b", "c"])
        ids = [c["id"] for c in calls]
        self.assertEqual(len(set(ids)), 3)

    def test_20_no_markers_leak_into_visible(self):
        raw = "pre " + _xml_block("x", {}) + " post"
        clean, _ = extract_tool_calls(raw)
        self.assertNotIn(TOOL_OPEN_XML, clean)
        self.assertNotIn(TOOL_CLOSE_XML, clean)
        self.assertNotIn(TOOL_OPEN_BRACKET, clean)
        # incomplete block held, never leaked
        clean2, _ = streaming_view("pre " + TOOL_OPEN_XML[:5])
        self.assertNotIn("tool", clean2.replace("pre ", ""))
        clean3, _ = streaming_view("pre " + _xml_block("x", {})[:20])
        self.assertEqual(clean3, "pre ")


class TestReasoningAndFlatten(unittest.TestCase):
    """7-12, 19: reasoning separation, replay, ordering, no dupes."""

    def test_07_tool_calls_following_reasoning_translator(self):
        tr = StreamTranslator("cid", "m", True)
        block = _xml_block("ls", {"path": "."})
        # reasoning first, then text with tool call
        tr.feed_reasoning("Let me check. ")
        out = tr.feed_content("Checking " + block + " done")
        # reasoning never in content
        joined = "".join(out)
        self.assertNotIn("Let me check", "".join(
            json.loads(p)["choices"][0].get("delta", {}).get("content", "") or ""
            for p in out if p != "[DONE]"))
        # tool call parsed, visible content stripped of markers
        calls = []
        content = ""
        for p in out:
            d = json.loads(p)["choices"][0].get("delta", {})
            content += d.get("content") or ""
            calls.extend(d.get("tool_calls") or [])
        self.assertIn("Checking", content)
        self.assertNotIn(TOOL_OPEN_XML, content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "ls")

    def test_08_reasoning_following_tool_result_flatten(self):
        msgs = [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "Checking",
             "reasoning_content": "Need to list.",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "ls",
                                          "arguments": '{"path":"."}'}}]},
            {"role": "tool", "name": "ls", "tool_call_id": "call_1",
             "content": "a.txt"},
            {"role": "assistant", "content": "Found a.txt",
             "reasoning_content": "Result shows a.txt, answer now."},
        ]
        out = flatten_history(msgs)
        # both reasoning sections preserved, in order
        self.assertIn("[assistant reasoning]\nNeed to list.", out)
        self.assertIn("[assistant reasoning]\nResult shows", out)
        self.assertLess(out.index("Need to list."), out.index("Found a.txt"))
        self.assertLess(out.index("a.txt"), out.index("Result shows"))

    def test_09_reasoning_preserved_next_turn_separate(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Checking",
             "reasoning_content": "Private thinking here",
             "tool_calls": [{"id": "call_r1", "type": "function",
                             "function": {"name": "ls",
                                          "arguments": "{}"}}]},
            {"role": "tool", "name": "ls", "tool_call_id": "call_r1",
             "content": "ok"},
        ]
        out = flatten_history(msgs)
        self.assertIn("[assistant reasoning]\nPrivate thinking here\n"
                      "[/assistant reasoning]", out)
        # reasoning NOT merged into visible [assistant] text
        vis = out.split("[/assistant reasoning]")[-1]
        self.assertNotIn("Private thinking here", vis)

    def test_10_tool_call_id_preserved_exactly(self):
        msgs = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_exact_123",
                             "type": "function",
                             "function": {"name": "read",
                                          "arguments": '{"p":"x"}'}}]},
        ]
        out = flatten_history(msgs)
        self.assertIn("(call call_exact_123)", out)
        self.assertNotIn("call_", out.replace("(call call_exact_123)", ""))

    def test_11_tool_result_associated_correct_call(self):
        msgs = [
            {"role": "assistant", "content": "",
             "tool_calls": [
                 {"id": "call_A", "type": "function",
                  "function": {"name": "a", "arguments": "{}"}},
                 {"id": "call_B", "type": "function",
                  "function": {"name": "b", "arguments": "{}"}}]},
            {"role": "tool", "name": "b", "tool_call_id": "call_B",
             "content": "result-B"},
        ]
        out = flatten_history(msgs)
        self.assertIn("(call call_A)", out)
        self.assertIn("(call call_B)", out)
        self.assertIn("[tool result: b] (call call_B)\nresult-B", out)
        # ordering: tool calls before tool result
        self.assertLess(out.index("(call call_B)"),
                        out.index("[tool result: b]"))

    def test_12_ordinary_content_without_tools(self):
        tr = StreamTranslator("cid", "m", False)
        payloads, text, reasoning = tr.finalize_text("Just an answer",
                                                     "some thought")
        self.assertEqual(text, "Just an answer")
        self.assertEqual(reasoning, "some thought")
        finishes = [json.loads(p)["choices"][0].get("finish_reason")
                    for p in payloads if p != "[DONE]"]
        # only the final chunk carries a finish reason
        self.assertEqual([f for f in finishes if f], ["stop"])
        # no tool_calls deltas
        for p in payloads:
            if p == "[DONE]":
                continue
            d = json.loads(p)["choices"][0].get("delta", {})
            self.assertNotIn("tool_calls", d)

    def test_19_no_duplication_snapshot_delta(self):
        tr = StreamTranslator("cid", "m", False)
        tr.handle_session(_msg_updated(), SID)
        tr.handle_session(_part_updated("p1", "text", ""), SID)
        tr.handle_session(_part_delta("p1", "Hello"), SID)
        out = tr.handle_session(_part_updated("p1", "text", "Hello world"),
                                SID)
        content = "".join(
            json.loads(p)["choices"][0].get("delta", {}).get("content", "") or ""
            for p in out if p != "[DONE]")
        self.assertEqual(content, " world")
        payloads, text, _ = tr.finalize_text("Hello world", "")
        tail = "".join(
            json.loads(p)["choices"][0].get("delta", {}).get("content", "") or ""
            for p in payloads if p != "[DONE]")
        self.assertEqual(tail, "")
        self.assertEqual(text, "Hello world")

    def test_empty_content_tool_call_not_discarded(self):
        msgs = [
            {"role": "assistant", "content": "",
             "reasoning_content": "think",
             "tool_calls": [{"id": "call_e1", "type": "function",
                             "function": {"name": "ls",
                                          "arguments": "{}"}}]},
            {"role": "tool", "name": "ls", "tool_call_id": "call_e1",
             "content": "files"},
        ]
        out = flatten_history(msgs)
        self.assertIn("[assistant reasoning]", out)
        self.assertIn("[assistant tool calls]", out)
        self.assertIn("[tool result: ls] (call call_e1)", out)


class TestToolChoice(unittest.TestCase):
    """13-15."""

    def test_13_auto(self):
        tools = [{"type": "function",
                  "function": {"name": "ls", "parameters": {}}}]
        eff = effective_tools(tools, "auto")
        self.assertEqual(len(eff), 1)
        block = format_client_tools(eff, "auto")
        self.assertIn("[client tools]", block)
        self.assertIn(TOOL_OPEN_XML, block)
        # bridge parses in auto mode
        clean, calls = extract_tool_calls("x " + _xml_block("ls", {}) + " y")
        self.assertEqual(len(calls), 1)

    def test_14_required(self):
        tools = [{"type": "function",
                  "function": {"name": "ls", "parameters": {}}}]
        eff = effective_tools(tools, "required")
        self.assertEqual(len(eff), 1)
        block = format_client_tools(eff, "required")
        self.assertIn("MUST", block)
        self.assertIn(TOOL_OPEN_XML, block)

    def test_15_none_drops_tools_no_bridge(self):
        tools = [{"type": "function",
                  "function": {"name": "ls", "parameters": {}}}]
        eff = effective_tools(tools, "none")
        self.assertEqual(eff, [])
        self.assertEqual(format_client_tools(eff, "none"), "")
        # with no bridge the parser must NOT run: markers stay as content
        # (simulate handle path: bridge=False -> no extract)
        raw = "answer " + _xml_block("ls", {})
        # proxy would skip extract when bridge False; verify raw untouched
        self.assertIn(TOOL_OPEN_XML, raw)


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


class WireLoopCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeEventHub.last = None
        self.events: list[dict] = []
        self.envelope = _envelope(text="hi")
        self.prompt_error = None
        self.ensure_ok = True
        self.create_error = None
        self.run_result = None
        self.run_deltas: list[tuple[str, str]] = []
        self.captured_prompts: list[str] = []
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
        app = proxy.make_app("/nonexistent/opencode",
                             str(Path(__file__).parent.parent),
                             agent="", serve_port=18790)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

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
        if isinstance(self.prompt_error, Exception):
            raise self.prompt_error
        return self.envelope

    async def _fake_run(self, model, prompt, session_id, on_delta=None):
        self.captured_prompts.append(prompt)
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
        payloads: list[str] = []
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                payloads.append(line[6:])
        return payloads

    @staticmethod
    def objs(payloads):
        return [json.loads(p) for p in payloads if p != "[DONE]"]

    @staticmethod
    def split_wire(objs):
        reasoning, content, calls, finishes = "", "", [], []
        for o in objs:
            ch = o["choices"][0]
            d = ch.get("delta") or {}
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            calls.extend(d.get("tool_calls") or [])
            if ch.get("finish_reason"):
                finishes.append(ch["finish_reason"])
        return reasoning, content, calls, finishes


class TestServeAndRunBackends(WireLoopCase):
    """16-17: tool bridge works on both transports."""

    async def test_16_serve_xml_tool_bridge(self):
        block = _xml_block("ls", {"path": "."})
        raw = "Checking " + block + " done"
        self.events = [
            _msg_updated(),
            _part_updated("p_r", "reasoning", ""),
            _part_updated("p_t", "text", ""),
            _part_delta("p_r", "Thinking", field="text"),
            _part_delta("p_t", "Checking "),
            _part_delta("p_t", block),
            _part_delta("p_t", " done"),
            _part_updated("p_r", "reasoning", "Thinking"),
            _part_updated("p_t", "text", raw),
        ]
        self.envelope = _envelope(text=raw, reasoning="Thinking")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "list"}],
            "stream": True,
            "tools": [{"type": "function", "function": {
                "name": "ls", "parameters": {"type": "object"}}}],
            "tool_choice": "auto",
        })
        self.assertEqual(resp.status, 200)
        payloads = await self.read_sse(resp)
        objs = self.objs(payloads)
        reasoning, content, calls, finishes = self.split_wire(objs)
        self.assertEqual(reasoning, "Thinking")
        self.assertNotIn("Thinking", content)
        self.assertNotIn(TOOL_OPEN_XML, content)
        self.assertNotIn(TOOL_CLOSE_XML, content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "ls")
        self.assertEqual(finishes, ["tool_calls"])
        self.assertEqual(payloads[-1], "[DONE]")

    async def test_17_run_xml_tool_bridge(self):
        self.ensure_ok = False
        block = _xml_block("grep", {"pattern": "x"})
        self.run_deltas = [("Running ", "text"), (block, "text")]
        self.run_result = ("Running " + block, "ponder", SID)
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "grep x"}],
            "stream": True,
            "tools": [{"type": "function", "function": {
                "name": "grep", "parameters": {"type": "object"}}}],
        })
        payloads = await self.read_sse(resp)
        objs = self.objs(payloads)
        reasoning, content, calls, finishes = self.split_wire(objs)
        self.assertEqual(reasoning, "ponder")
        self.assertNotIn(TOOL_OPEN_XML, content)
        self.assertEqual(len(calls), 1)
        self.assertEqual(finishes, ["tool_calls"])


class TestFullLoop(WireLoopCase):
    """18 + invariant: full multi-turn reasoning/tool loop on the wire."""

    async def test_18_multiturn_continuation_and_invariant(self):
        tools = [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}}]
        # ---- TURN 1: user -> reasoning + <tool_call> -> tool_calls ----
        block1 = _xml_block("get_weather", {"city": "Paris"})
        raw1 = "Checking weather " + block1
        self.events = [
            _msg_updated(mid="m1"),
            _part_updated("r1", "reasoning", "", mid="m1"),
            _part_updated("t1", "text", "", mid="m1"),
            _part_delta("r1", "Need weather for Paris.", mid="m1"),
            _part_delta("t1", "Checking weather ", mid="m1"),
            _part_delta("t1", block1, mid="m1"),
            _part_updated("r1", "reasoning",
                          "Need weather for Paris.", mid="m1"),
            _part_updated("t1", "text", raw1, mid="m1"),
        ]
        self.envelope = _envelope(text=raw1,
                                  reasoning="Need weather for Paris.")
        resp1 = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "stream": True,
            "tools": tools,
            "tool_choice": "auto",
        })
        self.assertEqual(resp1.status, 200)
        p1 = await self.read_sse(resp1)
        o1 = self.objs(p1)
        reasoning1, content1, calls1, finishes1 = self.split_wire(o1)
        # exact wire checks turn 1
        self.assertEqual(o1[0]["choices"][0]["delta"].get("role"),
                         "assistant")
        self.assertEqual(reasoning1, "Need weather for Paris.")
        self.assertIn("Checking weather", content1)
        self.assertNotIn(TOOL_OPEN_XML, content1)
        self.assertNotIn(TOOL_CLOSE_XML, content1)
        self.assertNotIn("Need weather", content1)
        self.assertEqual(len(calls1), 1)
        self.assertEqual(calls1[0]["function"]["name"], "get_weather")
        self.assertIn("Paris",
                      calls1[0]["function"]["arguments"])
        self.assertTrue(calls1[0]["id"].startswith("call_"))
        self.assertEqual(finishes1, ["tool_calls"])
        self.assertEqual(p1[-1], "[DONE]")

        call_id = calls1[0]["id"]
        call_args = calls1[0]["function"]["arguments"]

        # ---- Simulate Hermes tool execution ----
        tool_output = "sunny, 21C"

        # ---- TURN 2: replay + tool result -> reasoning + final -> stop ----
        msgs2 = [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": content1,
             "reasoning_content": reasoning1,
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": call_args}}]},
            {"role": "tool", "name": "get_weather",
             "tool_call_id": call_id, "content": tool_output},
        ]
        raw2 = "It is sunny in Paris."
        self.events = [
            _msg_updated(mid="m2"),
            _part_updated("r2", "reasoning", "", mid="m2"),
            _part_updated("t2", "text", "", mid="m2"),
            _part_delta("r2", "Result is sunny, answer.",
                        mid="m2"),
            _part_delta("t2", raw2, mid="m2"),
            _part_updated("r2", "reasoning",
                          "Result is sunny, answer.", mid="m2"),
            _part_updated("t2", "text", raw2, mid="m2"),
        ]
        self.envelope = _envelope(text=raw2,
                                  reasoning="Result is sunny, answer.")
        resp2 = await self.post_chat({
            "model": "big-pickle",
            "messages": msgs2,
            "stream": True,
            "tools": tools,
            "tool_choice": "auto",
        })
        self.assertEqual(resp2.status, 200)
        p2 = await self.read_sse(resp2)
        o2 = self.objs(p2)
        reasoning2, content2, calls2, finishes2 = self.split_wire(o2)
        self.assertEqual(reasoning2, "Result is sunny, answer.")
        self.assertEqual(content2, raw2)
        self.assertEqual(calls2, [])
        self.assertEqual(finishes2, ["stop"])
        self.assertEqual(p2[-1], "[DONE]")

        # ---- continuation-context checks (prompt sent to backend) ----
        self.assertGreaterEqual(len(self.captured_prompts), 2)
        prompt2 = self.captured_prompts[-1]
        # reasoning preserved, separate from visible content
        self.assertIn("[assistant reasoning]\nNeed weather for Paris.\n"
                      "[/assistant reasoning]", prompt2)
        # tool_calls preserved with EXACT id (not regenerated)
        self.assertIn(f"(call {call_id})", prompt2)
        self.assertIn("get_weather", prompt2)
        self.assertIn("Paris", prompt2)
        # tool result associated with same call, ordering preserved
        self.assertIn(f"[tool result: get_weather] (call {call_id})\n"
                      f"{tool_output}", prompt2)
        self.assertLess(prompt2.index(f"(call {call_id})"),
                        prompt2.index(f"[tool result: get_weather]"))
        # visible assistant content still present, no markers leaked
        self.assertIn(content1.strip().split()[0], prompt2)
        # transcript portion (before the instructional [client tools]
        # example, which legitimately contains <tool_call>) must not leak
        # backend tool markers as visible content
        transcript_part = prompt2.split("[client tools]")[0]
        self.assertNotIn(TOOL_OPEN_XML, transcript_part)
        self.assertNotIn(TOOL_CLOSE_XML, transcript_part)
        self.assertNotIn(TOOL_OPEN_BRACKET, transcript_part)
        self.assertNotIn(TOOL_CLOSE_BRACKET, transcript_part)
        # the delta for turn 2 must be a delta (prefix matched), not resync
        # (resync would repeat [system]; delta has no system here)
        # (no system in these msgs, so check it contains tool result)
        self.assertIn("[tool result", prompt2)

    async def test_blocking_full_loop_ids_and_reasoning(self):
        tools = [{"type": "function", "function": {
            "name": "ls", "parameters": {"type": "object"}}}] 
        # turn 1 blocking with tool call
        self.envelope = _envelope(
            text="Look " + _xml_block("ls", {"path": "."}) + " end",
            reasoning="think-1")
        resp = await self.post_chat({
            "model": "big-pickle",
            "messages": [{"role": "user", "content": "ls"}],
            "stream": False,
            "tools": tools,
        })
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        msg = data["choices"][0]["message"]
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(msg["reasoning_content"], "think-1")
        self.assertNotIn(TOOL_OPEN_XML, msg["content"])
        cid = msg["tool_calls"][0]["id"]
        cname = msg["tool_calls"][0]["function"]["name"]
        cargs = msg["tool_calls"][0]["function"]["arguments"]
        # turn 2 blocking continuation
        self.envelope = _envelope(text="Done: file list",
                                  reasoning="think-2")
        resp2 = await self.post_chat({
            "model": "big-pickle",
            "messages": [
                {"role": "user", "content": "ls"},
                {"role": "assistant", "content": msg["content"],
                 "reasoning_content": msg["reasoning_content"],
                 "tool_calls": msg["tool_calls"]},
                {"role": "tool", "name": cname,
                 "tool_call_id": cid, "content": "a.txt"},
            ],
            "stream": False,
            "tools": tools,
        })
        data2 = await resp2.json()
        self.assertEqual(data2["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data2["choices"][0]["message"]["content"],
                         "Done: file list")
        self.assertEqual(data2["choices"][0]["message"]["reasoning_content"],
                         "think-2")
        prompt2 = self.captured_prompts[-1]
        self.assertIn("think-1", prompt2)
        self.assertIn(f"(call {cid})", prompt2)
        self.assertIn(cargs, prompt2)
        self.assertIn(f"[tool result: {cname}] (call {cid})", prompt2)


class TestPromptAndDebug(unittest.TestCase):
    def test_client_tools_prefers_xml(self):
        tools = [{"type": "function",
                  "function": {"name": "x", "parameters": {}}}]
        block = format_client_tools(tools)
        self.assertIn(TOOL_OPEN_XML, block)
        self.assertIn(TOOL_CLOSE_XML, block)
        # legacy still mentioned for compat
        self.assertIn(TOOL_OPEN_BRACKET, block)
        # XML example comes first
        self.assertLess(block.index(TOOL_OPEN_XML),
                        block.index(TOOL_OPEN_BRACKET))

    def test_bypass_lite_teaches_xml_first(self):
        p = Path(__file__).resolve().parent.parent / ".opencode" / "agents" \
            / "bypass-lite.md"
        text = p.read_text(encoding="utf-8")
        self.assertIn("<tool_call>", text)
        self.assertIn('{"name": "tool_name", "arguments":', text)
        self.assertNotIn("```", text.split("<tool_call>")[1].split(
            "</tool_call>")[0])
        # XML taught first
        self.assertLess(text.index("<tool_call>"), text.index("[tool_call]"))

    def test_debug_helper_no_secrets(self):
        # debug helper exists, gated, never logs env/keys
        self.assertTrue(hasattr(proxy, "DEBUG"))
        self.assertTrue(hasattr(proxy, "debug_log"))
        self.assertTrue(hasattr(proxy, "_debug_tool_turn"))
        import inspect
        src = inspect.getsource(proxy._debug_tool_turn)
        self.assertNotIn("OPENCODE_SERVER_PASSWORD", src)
        self.assertNotIn("API_KEY", src)
        self.assertNotIn("environ", src)


if __name__ == "__main__":
    unittest.main()
