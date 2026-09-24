#!/usr/bin/env python3
"""Unit tests for the transcript + digest-delta layer (no network, no CLI).

Pins the architecture contract:
  * Hermes' ``messages`` array is authoritative — flattening preserves every
    role, order, tool call and tool result, with NO size budget / truncation;
  * the digest registry is a cursor map (sha256 prefix match), not a
    transcript database — append-only turns become deltas, any mid-history
    edit/compression is a divergence -> full authoritative resync;
  * argv chunking for the `run` fallback is byte-exact under the joiner's
    single-space reassembly.

Run:  python3 -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opencode_proxy import (  # noqa: E402
    KNOWN_MODELS,
    MAX_ARG_CHARS,
    REGISTRY_MAX,
    _blocking_response,
    _free_ids_from_providers,
    _render_tool_calls,
    _text_of,
    Registry,
    StreamTranslator,
    ToolCallStreamParser,
    agent_available,
    chunk,
    chunk_argv,
    cli_argv,
    digests_of,
    drop_echoed_assistant,
    effective_tools,
    extract_tool_calls,
    flatten_history,
    flatten_tail,
    format_client_tools,
    lookup_entry,
    msg_digest,
    normalize_model,
    plan_prompt,
    spawn_detached_kwargs,
    split_envelope_parts,
    split_model,
    streaming_view,
    tools_signature,
)


class TestTextOf(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(_text_of(None), "")
        self.assertEqual(_text_of(""), "")

    def test_plain_string(self):
        self.assertEqual(_text_of("hello"), "hello")

    def test_multipart_joins_text(self):
        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        text = _text_of(content)
        self.assertIn("look at this", text)
        self.assertIn("image", text)  # omission is marked, not silent


class TestToolCalls(unittest.TestCase):
    def test_render_name_and_args(self):
        out = _render_tool_calls([{
            "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
        }])
        self.assertIn("[assistant tool calls]", out)
        self.assertIn("read", out)
        self.assertIn("a.txt", out)

    def test_render_empty(self):
        self.assertEqual(_render_tool_calls([]), "")
        self.assertEqual(_render_tool_calls(None), "")


class TestFlattenHistory(unittest.TestCase):
    def test_order_and_labels(self):
        msgs = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "my color is teal"},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "what is it?"},
        ]
        out = flatten_history(msgs)
        self.assertLess(out.index("[system]"), out.index("teal"))
        self.assertLess(out.index("teal"), out.index("Noted."))
        self.assertIn("[user]\nwhat is it?", out)

    def test_developer_maps_to_system(self):
        out = flatten_history([{"role": "developer", "content": "Be terse."}])
        self.assertIn("[system]\nBe terse.", out)

    def test_nothing_ever_truncated(self):
        """The goldfish regression test: every turn must survive, at any size."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "uno"},
        ]
        for i in range(50):
            msgs.append({"role": "user", "content": f"question-{i:02d} " + "x" * 2000})
            msgs.append({"role": "assistant", "content": f"answer-{i:02d} " + "y" * 2000})
        out = flatten_history(msgs)
        self.assertGreater(len(out), 100_000)  # no budget kicks in
        for word in ("sys", "one", "uno", "question-00", "answer-49"):
            self.assertIn(word, out)

    def test_huge_tool_result_survives(self):
        blob = "T" * 300_000  # one tool result > any plausible char limit
        out = flatten_history([
            {"role": "user", "content": "read"},
            {"role": "tool", "name": "read", "content": blob},
        ])
        self.assertIn(blob, out)

    def test_assistant_tool_calls_preserved(self):
        msgs = [
            {"role": "user", "content": "read a.txt"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "read", "arguments": '{"path":"a.txt"}'}}]},
            {"role": "tool", "name": "read", "content": "file bytes"},
            {"role": "assistant", "content": "The file says hi."},
        ]
        out = flatten_history(msgs)
        self.assertIn("read", out)
        self.assertIn("a.txt", out)
        self.assertIn("[tool result: read]\nfile bytes", out)
        self.assertIn("The file says hi.", out)

    def test_tool_result_without_name(self):
        out = flatten_history([{"role": "tool", "content": "42"}])
        self.assertIn("[tool result]\n42", out)

    def test_empty_turns_skipped(self):
        out = flatten_history([
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": ""},
            {"role": "user", "content": "hi"},
        ])
        self.assertEqual(out, "[user]\nhi")

    def test_flatten_tail_is_history(self):
        msgs = [{"role": "user", "content": "tail only"}]
        self.assertEqual(flatten_tail(msgs), flatten_history(msgs))


class TestNormalizeModel(unittest.TestCase):
    def test_bare_gets_prefix(self):
        self.assertEqual(normalize_model("big-pickle"), "opencode/big-pickle")

    def test_qualified_untouched(self):
        self.assertEqual(normalize_model("opencode/big-pickle"), "opencode/big-pickle")
        self.assertEqual(normalize_model("google/gemini-3-flash"), "google/gemini-3-flash")

    def test_empty_defaults(self):
        self.assertEqual(normalize_model(""), "opencode/big-pickle")
        self.assertEqual(normalize_model(None), "opencode/big-pickle")

    def test_split_model_roundtrip(self):
        self.assertEqual(split_model("big-pickle"), ("opencode", "big-pickle"))
        self.assertEqual(split_model("opencode/muse"), ("opencode", "muse"))


class TestDigest(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        a = {"role": "user", "content": "hi"}
        b = {"role": "user", "content": "hi"}
        c = {"role": "user", "content": "ho"}
        self.assertEqual(msg_digest(a), msg_digest(b))
        self.assertNotEqual(msg_digest(a), msg_digest(c))

    def test_key_order_irrelevant(self):
        self.assertEqual(
            msg_digest({"role": "user", "content": "x"}),
            msg_digest({"content": "x", "role": "user"}),
        )

    def test_digests_of_skips_non_dicts(self):
        digs = digests_of([{"role": "user", "content": "x"}, "junk", 42])
        self.assertEqual(len(digs), 1)


class TestDropEchoedAssistant(unittest.TestCase):
    def test_matching_echo_dropped(self):
        tail = [
            {"role": "assistant", "content": "My reply."},
            {"role": "user", "content": "next"},
        ]
        out = drop_echoed_assistant(tail, "My reply.")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")

    def test_mismatch_kept(self):
        tail = [{"role": "assistant", "content": "other"}, {"role": "user", "content": "q"}]
        self.assertEqual(len(drop_echoed_assistant(tail, "My reply.")), 2)

    def test_tool_call_turn_never_dropped(self):
        tail = [{
            "role": "assistant", "content": "My reply.",
            "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
        }, {"role": "user", "content": "q"}]
        self.assertEqual(len(drop_echoed_assistant(tail, "My reply.")), 2)

    def test_empty_last_reply_keeps_everything(self):
        tail = [{"role": "assistant", "content": "x"}]
        self.assertEqual(drop_echoed_assistant(tail, ""), tail)


class TestPlanPrompt(unittest.TestCase):
    def setUp(self):
        self.h1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "teal is my color"},
            {"role": "assistant", "content": "Noted."},
        ]

    def test_no_entry_full_resync(self):
        prompt, mode, rec = plan_prompt(self.h1, None)
        self.assertEqual(mode, "resync")
        self.assertIn("[system]", prompt)
        self.assertIn("teal is my color", prompt)
        self.assertEqual(rec["digests"], digests_of(self.h1))

    def test_append_delta_drops_prefix(self):
        entry = {"digests": digests_of(self.h1), "sid": "s1",
                 "last_reply": "Noted.", "backend": "serve"}
        h2 = self.h1 + [
            {"role": "assistant", "content": "Noted."},   # client echo
            {"role": "user", "content": "what color?"},
        ]
        prompt, mode, rec = plan_prompt(h2, entry)
        self.assertEqual(mode, "delta")
        self.assertNotIn("[system]", prompt)
        self.assertNotIn("teal is my color", prompt)  # prefix not resent
        self.assertIn("what color?", prompt)
        # echo of our last reply is skipped (backend already has it)
        self.assertNotIn("[assistant]\nNoted.", prompt)
        self.assertEqual(rec["digests"], digests_of(h2))

    def test_middle_edit_diverges(self):
        entry = {"digests": digests_of(self.h1), "sid": "s1",
                 "last_reply": "Noted.", "backend": "serve"}
        edited = list(self.h1)
        edited[1] = {"role": "user", "content": "REWRITTEN by compression"}
        h2 = edited + [{"role": "user", "content": "more"}]
        prompt, mode, rec = plan_prompt(h2, entry)
        self.assertEqual(mode, "resync")
        self.assertIn("REWRITTEN by compression", prompt)

    def test_shorter_history_diverges(self):
        entry = {"digests": digests_of(self.h1), "sid": "s1",
                 "last_reply": "Noted.", "backend": "serve"}
        prompt, mode, _ = plan_prompt(self.h1[:1], entry)
        self.assertEqual(mode, "resync")
        self.assertIn("sys", prompt)

    def test_identical_resend_uses_last_message(self):
        entry = {"digests": digests_of(self.h1), "sid": "s1",
                 "last_reply": "Noted.", "backend": "serve"}
        prompt, mode, _ = plan_prompt(self.h1, entry)
        self.assertEqual(mode, "delta")
        self.assertTrue(prompt)  # regenerate rather than emit nothing


class TestRegistry(unittest.TestCase):
    def test_prefix_match_and_lru_touch(self):
        reg = Registry(max_entries=3)
        d1 = digests_of([{"role": "user", "content": "a"}])
        reg.record(d1, "serve", "sid-a", "r1")
        d2 = digests_of([{"role": "user", "content": "a"},
                         {"role": "user", "content": "b"}])
        reg.record(d2, "serve", "sid-b", "r2")
        # exact longest prefix wins
        hit = reg.match(d1 + [msg_digest({"role": "user", "content": "z"})], "serve")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["sid"], "sid-a")
        # full extension of d2
        hit = reg.match(d2 + [msg_digest({"role": "user", "content": "z"})], "serve")
        self.assertEqual(hit["sid"], "sid-b")

    def test_non_prefix_is_miss(self):
        reg = Registry()
        d1 = digests_of([{"role": "user", "content": "a"}])
        reg.record(d1, "serve", "s", "r")
        other = digests_of([{"role": "user", "content": "DIFFERENT"}])
        self.assertIsNone(reg.match(other, "serve"))

    def test_backend_kind_isolated(self):
        reg = Registry()
        d = digests_of([{"role": "user", "content": "a"}])
        reg.record(d, "serve", "s", "r")
        self.assertIsNone(reg.match(d, "run"))

    def test_record_same_sid_updates_cursor(self):
        reg = Registry()
        d1 = digests_of([{"role": "user", "content": "a"}])
        reg.record(d1, "serve", "s", "r1")
        d2 = d1 + [msg_digest({"role": "user", "content": "b"})]
        reg.record(d2, "serve", "s", "r2")
        self.assertEqual(len(reg), 1)
        self.assertEqual(reg.peek(d2, "serve")["last_reply"], "r2")

    def test_lru_eviction_beyond_capacity(self):
        reg = Registry(max_entries=2)
        for i in range(4):
            reg.record([f"digest-{i}"], "serve", f"sid-{i}", "r")
        self.assertEqual(len(reg), 2)
        self.assertIsNone(reg.peek(["digest-0"], "serve"))
        self.assertIsNotNone(reg.peek(["digest-3"], "serve"))

    def test_capacity_constant(self):
        self.assertEqual(REGISTRY_MAX, 64)


class TestChunkArgv(unittest.TestCase):
    def test_short_untouched(self):
        self.assertEqual(chunk_argv("hello world"), ["hello world"])

    def test_space_boundary_roundtrip(self):
        """Splits rejoin with one space == original (run's join semantics)."""
        text = ("lorem ipsum dolor sit amet " * 200).strip()
        self.assertGreater(len(text), 1000)
        parts = chunk_argv(text, limit=200)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 200 for p in parts))
        self.assertEqual(" ".join(parts), text)

    def test_hard_split_when_no_space(self):
        blob = "x" * 500
        parts = chunk_argv(blob, limit=100)
        self.assertEqual(len(parts), 5)
        self.assertEqual("".join(parts), blob)  # no spaces introduced

    def test_default_limit_under_kernel(self):
        self.assertLessEqual(MAX_ARG_CHARS, 131072)
        self.assertGreater(MAX_ARG_CHARS, 100_000)


class TestChunk(unittest.TestCase):
    def test_delta_chunk_shape(self):
        obj = json.loads(chunk("id-1", "big-pickle", delta="hi"))
        self.assertEqual(obj["object"], "chat.completion.chunk")
        self.assertEqual(obj["choices"][0]["delta"], {"content": "hi"})

    def test_finish_chunk_has_finish_reason(self):
        """Hermes's 'empty stream with no finish_reason' must never recur."""
        obj = json.loads(chunk("id-1", "big-pickle", finish="stop"))
        self.assertEqual(obj["choices"][0]["finish_reason"], "stop")

    def test_role_delta_chunk(self):
        obj = json.loads(chunk("id-1", "m", delta={"role": "assistant", "content": ""}))
        self.assertEqual(obj["choices"][0]["delta"]["role"], "assistant")


class TestAgentAvailable(unittest.TestCase):
    def test_missing_agent_is_false(self):
        self.assertFalse(agent_available("no-such-agent", "/nonexistent"))

    def test_empty_is_false(self):
        self.assertFalse(agent_available("", "/tmp"))

    def test_bundled_agent_found(self):
        root = str(Path(__file__).resolve().parent.parent)
        self.assertTrue(agent_available("bypass-lite", root))


class TestEffectiveTools(unittest.TestCase):
    def test_filters_non_function_and_junk(self):
        tools = [
            {"type": "function", "function": {"name": "bash"}},
            {"type": "mcp"},
            "junk",
            {"type": "function", "function": {}},  # no name
            None,
        ]
        out = effective_tools(tools)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "bash")

    def test_tool_choice_none_drops_all(self):
        tools = [{"type": "function", "function": {"name": "bash"}}]
        self.assertEqual(effective_tools(tools, "none"), [])

    def test_non_list_ignored(self):
        self.assertEqual(effective_tools(None), [])
        self.assertEqual(effective_tools({"type": "function"}), [])


class TestFormatClientTools(unittest.TestCase):
    def test_contains_protocol_and_tool(self):
        tools = [{"type": "function", "function": {
            "name": "list_files", "description": "List files",
            "parameters": {"type": "object"},
        }}]
        block = format_client_tools(tools)
        self.assertIn("[client tools]", block)
        self.assertIn("list_files", block)
        self.assertIn("[tool_call]", block)
        self.assertIn("[/tool_call]", block)

    def test_required_adds_must(self):
        tools = [{"type": "function", "function": {"name": "x"}}]
        self.assertIn("MUST", format_client_tools(tools, "required"))

    def test_empty_tools_empty_block(self):
        self.assertEqual(format_client_tools([]), "")


class TestExtractToolCalls(unittest.TestCase):
    def test_parses_and_strips_block(self):
        raw = ('Hello [tool_call]\n'
               '{"name":"bash","arguments":{"cmd":"ls"}}\n'
               '[/tool_call] done')
        clean, calls = extract_tool_calls(raw)
        self.assertEqual(clean, "Hello  done")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertEqual(calls[0]["type"], "function")
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertIn("ls", calls[0]["function"]["arguments"])

    def test_no_markers_passthrough(self):
        clean, calls = extract_tool_calls("plain text")
        self.assertEqual(clean, "plain text")
        self.assertEqual(calls, [])

    def test_unclosed_marker_stays_visible(self):
        clean, calls = extract_tool_calls('x [tool_call] {"name":')
        self.assertIn("[tool_call]", clean)
        self.assertEqual(calls, [])

    def test_malformed_json_kept_visible(self):
        raw = "a [tool_call] not-json [/tool_call] b"
        clean, calls = extract_tool_calls(raw)
        self.assertIn("[tool_call]", clean)
        self.assertEqual(calls, [])

    def test_two_calls(self):
        raw = ('[tool_call]{"name":"a","arguments":{}}[/tool_call]'
               'mid'
               '[tool_call]{"name":"b","arguments":{}}[/tool_call]')
        clean, calls = extract_tool_calls(raw)
        self.assertEqual(clean, "mid")
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])


class TestStreamingView(unittest.TestCase):
    def test_complete_marker_removed(self):
        raw = 'hi [tool_call]{"name":"x","arguments":{}}[/tool_call] bye'
        clean, calls = streaming_view(raw)
        self.assertEqual(clean, "hi  bye")
        self.assertEqual(len(calls), 1)

    def test_unclosed_held_from_open(self):
        clean, calls = streaming_view("abc [tool_call] partial")
        self.assertEqual(clean, "abc ")
        self.assertEqual(calls, [])

    def test_partial_open_prefix_held(self):
        clean, calls = streaming_view("abc [tool_c")
        self.assertEqual(clean, "abc ")
        self.assertEqual(calls, [])


class TestToolCallStreamParser(unittest.TestCase):
    def test_split_across_feeds(self):
        p = ToolCallStreamParser()
        self.assertEqual(p.feed("Hello "), "Hello ")
        self.assertEqual(p.feed("[tool_c"), "")
        self.assertEqual(p.feed("all]\n"), "")
        self.assertEqual(p.feed('{"name":"bash","arguments":{}}'), "")
        self.assertEqual(p.feed("[/tool_call]"), "")
        self.assertEqual(p.feed(" bye"), " bye")
        rest, calls = p.finish()
        self.assertEqual(rest, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_text_only_passthrough(self):
        p = ToolCallStreamParser()
        self.assertEqual(p.feed("just text"), "just text")
        rest, calls = p.finish()
        self.assertEqual(rest, "")
        self.assertEqual(calls, [])

    def test_finish_with_final_text_replaces_raw(self):
        p = ToolCallStreamParser()
        already = p.feed('partial [tool_call] {"name":')
        self.assertEqual(already, "partial ")  # safe prefix already streamed
        rest, calls = p.finish(
            'partial [tool_call]{"name":"z","arguments":{}}[/tool_call]!')
        self.assertEqual(rest, "!")  # only the unstreamed tail remains
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "z")


class TestFreeIds(unittest.TestCase):
    def test_filters_cost_and_status(self):
        data = {"providers": {"opencode": {"models": {
            "big-pickle": {"cost": {"input": 0, "output": 0}, "status": "active"},
            "paid": {"cost": {"input": 1, "output": 1}},
            "old": {"cost": {"input": 0, "output": 0}, "status": "deprecated"},
        }}}}
        self.assertEqual(_free_ids_from_providers(data), ["big-pickle"])

    def test_missing_provider_empty(self):
        self.assertEqual(_free_ids_from_providers({"providers": {}}), [])
        self.assertEqual(_free_ids_from_providers({}), [])

    def test_known_models_nonempty(self):
        self.assertIn("big-pickle", KNOWN_MODELS)
        self.assertTrue(all("free" in m or m == "big-pickle"
                            for m in KNOWN_MODELS))


class TestPlatformHelpers(unittest.TestCase):
    def test_cli_argv_plain(self):
        self.assertEqual(cli_argv("/x/opencode", "run"),
                         ["/x/opencode", "run"])

    def test_spawn_kwargs_posix(self):
        import os
        if os.name == "nt":
            self.assertIn("creationflags", spawn_detached_kwargs())
        else:
            self.assertEqual(spawn_detached_kwargs(),
                             {"start_new_session": True})


class TestToolCallWire(unittest.TestCase):
    def test_blocking_response_with_tool_calls(self):
        calls = [{"id": "call_x", "type": "function",
                  "function": {"name": "n", "arguments": "{}"}}]
        resp = _blocking_response("id", "m", "hi", {}, calls)
        self.assertEqual(resp["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(resp["choices"][0]["message"]["tool_calls"][0]["id"],
                         "call_x")

    def test_blocking_response_without_tool_calls(self):
        resp = _blocking_response("id", "m", "hi", {})
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("tool_calls", resp["choices"][0]["message"])

    def test_chunk_tool_calls_delta(self):
        obj = json.loads(chunk("id", "m",
                               tool_calls=[{"index": 0, "id": "c"}],
                               finish="tool_calls"))
        self.assertEqual(obj["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(obj["choices"][0]["delta"]["tool_calls"][0]["id"], "c")

    def test_tool_result_includes_call_id(self):
        out = flatten_history([
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "name": "bash", "tool_call_id": "call_abc",
             "content": "ok"},
        ])
        self.assertIn("[tool result: bash] (call call_abc)", out)


# --------------------------------------------------------------------------
# Req 13: tools-signature / lookup / reasoning translation / translator
# --------------------------------------------------------------------------

def _tr(bridge=False):
    return StreamTranslator("cid", "m", bridge)


def _msg_updated(sid, mid, role="assistant"):
    return {"type": "message.updated",
            "properties": {"sessionID": sid,
                           "info": {"id": mid, "role": role}}}


def _part_updated(sid, mid, pid, ptype, text=""):
    return {"type": "message.part.updated",
            "properties": {"sessionID": sid,
                           "part": {"id": pid, "messageID": mid,
                                    "type": ptype, "text": text}}}


def _part_delta(sid, mid, pid, delta, field="text"):
    return {"type": "message.part.delta",
            "properties": {"sessionID": sid, "messageID": mid,
                           "partID": pid, "delta": delta, "field": field}}


def _content_of(payloads):
    out = ""
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        obj = json.loads(p)
        d = (obj.get("choices") or [{}])[0].get("delta") or {}
        if "content" in d and d["content"]:
            out += d["content"]
    return out


def _reasoning_of(payloads):
    out = ""
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        obj = json.loads(p)
        d = (obj.get("choices") or [{}])[0].get("delta") or {}
        if "reasoning_content" in d and d["reasoning_content"]:
            out += d["reasoning_content"]
    return out


def _tool_call_chunks(payloads):
    out = []
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        obj = json.loads(p)
        d = (obj.get("choices") or [{}])[0].get("delta") or {}
        if d.get("tool_calls"):
            out.extend(d["tool_calls"])
    return out


def _finish_reasons(payloads):
    out = []
    for p in payloads:
        if p == "[DONE]" or not isinstance(p, str):
            continue
        obj = json.loads(p)
        fr = (obj.get("choices") or [{}])[0].get("finish_reason")
        if fr:
            out.append(fr)
    return out


class TestToolsSignature(unittest.TestCase):
    def test_same_tools_same_sig(self):
        tools = [{"type": "function", "function": {"name": "x"}}]
        self.assertEqual(tools_signature(tools, "auto"),
                         tools_signature(list(tools), "auto"))

    def test_different_tools_different_sig(self):
        a = [{"type": "function", "function": {"name": "x"}}]
        b = [{"type": "function", "function": {"name": "y"}}]
        self.assertNotEqual(tools_signature(a, None), tools_signature(b, None))

    def test_tool_choice_changes_sig(self):
        tools = [{"type": "function", "function": {"name": "x"}}]
        self.assertNotEqual(tools_signature(tools, "auto"),
                             tools_signature(tools, "required"))

    def test_sig_not_part_of_message_digests(self):
        """tools_sig rides as request metadata — never inside digests."""
        msgs = [{"role": "user", "content": "hi"}]
        d1 = digests_of(msgs)
        tools = [{"type": "function", "function": {"name": "x"}}]
        self.assertNotEqual(tools_signature(tools, None),
                             tools_signature([], None))
        # digests_of is independent of any tools payload
        self.assertEqual(d1, digests_of(msgs))

    def test_empty_and_none_stable(self):
        self.assertEqual(tools_signature([], None), tools_signature([], None))
        self.assertEqual(tools_signature(None, None),
                         tools_signature([], None))


class TestLookupEntry(unittest.TestCase):
    def setUp(self):
        self.msgs = [{"role": "user", "content": "hi"}]
        self.digs = digests_of(self.msgs)
        self.tools = [{"type": "function", "function": {"name": "x"}}]
        self.sig = tools_signature(self.tools, "auto")
        self.reg = Registry()
        self.reg.record(self.digs, "serve", "s1", "reply", tools_sig=self.sig)

    def test_match_with_same_sig(self):
        entry = lookup_entry(self.reg, self.digs, "serve", self.sig)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["sid"], "s1")

    def test_sig_mismatch_forces_resync(self):
        other = tools_signature(self.tools, "required")
        self.assertIsNone(lookup_entry(self.reg, self.digs, "serve", other))

    def test_no_match_returns_none(self):
        self.assertIsNone(lookup_entry(
            self.reg, digests_of([{"role": "user", "content": "other"}]),
            "serve", self.sig))

    def test_wrong_backend_kind(self):
        self.assertIsNone(lookup_entry(self.reg, self.digs, "run", self.sig))

    def test_legacy_entry_without_sig_mismatch(self):
        reg = Registry()
        reg.record(self.digs, "serve", "s2", "reply")  # tools_sig defaults ""
        self.assertIsNone(lookup_entry(reg, self.digs, "serve", self.sig))
        self.assertIsNotNone(lookup_entry(reg, self.digs, "serve", ""))


class TestRegistryToolsSig(unittest.TestCase):
    def test_record_stores_tools_sig(self):
        reg = Registry()
        sig = tools_signature([{"type": "function",
                                "function": {"name": "x"}}], None)
        entry = reg.record(["d1"], "serve", "s1", "reply", tools_sig=sig)
        self.assertEqual(entry["tools_sig"], sig)
        self.assertEqual(reg.match(["d1"], "serve")["tools_sig"], sig)

    def test_last_reply_is_content_not_reasoning(self):
        """last_reply persistence is content-only (reasoning never stored)."""
        reg = Registry()
        reg.record(["d1"], "serve", "s1", "visible answer")
        self.assertEqual(reg.match(["d1"], "serve")["last_reply"],
                         "visible answer")
        self.assertNotIn("reason", reg.match(["d1"], "serve")["last_reply"])


class TestSplitEnvelope(unittest.TestCase):
    def test_separates_text_and_reasoning(self):
        text, reasoning = split_envelope_parts([
            {"type": "reasoning", "text": "thinking..."},
            {"type": "text", "text": "Hello world"},
        ])
        self.assertEqual(text, "Hello world")
        self.assertEqual(reasoning, "thinking...")

    def test_content_blocks_joined(self):
        text, reasoning = split_envelope_parts([
            {"type": "text", "content": [
                {"type": "text", "text": "part A "},
                {"type": "text", "text": "part B"},
            ]},
        ])
        self.assertEqual(text, "part A part B")
        self.assertEqual(reasoning, "")

    def test_non_content_parts_ignored(self):
        text, reasoning = split_envelope_parts([
            {"type": "step-start"},
            {"type": "tool", "tool": "bash", "state": {}},
            {"type": "text", "text": "ok"},
        ])
        self.assertEqual(text, "ok")
        self.assertEqual(reasoning, "")

    def test_empty_and_none(self):
        self.assertEqual(split_envelope_parts(None), ("", ""))
        self.assertEqual(split_envelope_parts([]), ("", ""))


class TestBlockingReasoning(unittest.TestCase):
    def test_blocking_response_with_reasoning_content(self):
        resp = _blocking_response("id", "m", "visible", {},
                                  reasoning_content="private thoughts")
        msg = resp["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], "private thoughts")
        self.assertEqual(msg["content"], "visible")
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")

    def test_blocking_response_without_reasoning_omits_key(self):
        resp = _blocking_response("id", "m", "visible", {})
        self.assertNotIn("reasoning_content", resp["choices"][0]["message"])

    def test_empty_reasoning_omits_key(self):
        resp = _blocking_response("id", "m", "visible", {},
                                  reasoning_content="")
        self.assertNotIn("reasoning_content", resp["choices"][0]["message"])


class TestRenderToolCallsCallId(unittest.TestCase):
    def test_call_id_appended(self):
        out = _render_tool_calls([{
            "id": "call_abc123",
            "function": {"name": "read", "arguments": "{}"},
        }])
        self.assertIn("(call call_abc123)", out)


class TestStreamTranslatorRoles(unittest.TestCase):
    def test_role_chunk_shape(self):
        obj = json.loads(_tr().role_chunk())
        self.assertEqual(obj["choices"][0]["delta"],
                         {"role": "assistant", "content": ""})

    def test_reasoning_never_in_content(self):
        tr = _tr()
        p1 = tr.feed_reasoning("pondering")
        p2 = tr.feed_content("answer")  # returns a list of chunks
        self.assertEqual(_reasoning_of([p1]), "pondering")
        self.assertEqual(_content_of(p2), "answer")
        self.assertEqual(_content_of([p1]), "")
        self.assertEqual(_reasoning_of(p2), "")
        self.assertEqual(tr.reasoning_emitted, "pondering")
        self.assertEqual(tr.content_emitted, "answer")

    def test_feed_empty_is_noop(self):
        tr = _tr()
        self.assertIsNone(tr.feed_reasoning(""))
        self.assertEqual(tr.feed_content(""), [])


class TestStreamTranslatorRaces(unittest.TestCase):
    """opencode#26924: delta may arrive before part/message metadata."""

    SID, MID, PID = "sess1", "msg1", "part1"

    def test_delta_before_part_updated_buffered_then_flushed(self):
        tr = _tr()
        # message role known first
        out = tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        self.assertEqual(out, [])
        # delta arrives before the part's type is known
        out = tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, "Hello "), self.SID)
        self.assertEqual(out, [])  # held, not dropped
        # metadata lands with a snapshot behind the stream
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", ""),
            self.SID)
        self.assertEqual(_content_of(out), "Hello ")
        # later deltas flow directly
        out = tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, "world"), self.SID)
        self.assertEqual(_content_of(out), "world")

    def test_delta_before_message_updated_pending(self):
        tr = _tr()
        # part + delta arrive before we know who owns the message
        tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", ""),
            self.SID)
        out = tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, "Hi"), self.SID)
        self.assertEqual(out, [])  # held in mid_pending
        # assistant role lands -> pending flushed in order
        out = tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        self.assertEqual(_content_of(out), "Hi")

    def test_user_message_events_discarded(self):
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, "user1", role="user"),
                          self.SID)
        out = tr.handle_session(
            _part_updated(self.SID, "user1", "up1", "text", "echo"), self.SID)
        self.assertEqual(out, [])
        out = tr.handle_session(
            _part_delta(self.SID, "user1", "up1", "echo"), self.SID)
        self.assertEqual(out, [])

    def test_user_part_before_role_known_discarded_on_update(self):
        tr = _tr()
        tr.handle_session(
            _part_updated(self.SID, "user1", "up1", "text", "echo"),
            self.SID)
        out = tr.handle_session(
            _part_delta(self.SID, "user1", "up1", "echo"), self.SID)
        self.assertEqual(out, [])  # still pending
        out = tr.handle_session(_msg_updated(self.SID, "user1", role="user"),
                                self.SID)
        self.assertEqual(out, [])  # pending discarded, not emitted

    def test_foreign_session_ignored(self):
        tr = _tr()
        tr.handle_session(_msg_updated("other", self.MID), self.SID)
        out = tr.handle_session(
            _part_delta("other", self.MID, self.PID, "x"), self.SID)
        self.assertEqual(out, [])
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", "x"),
            "other")
        self.assertEqual(out, [])


class TestStreamTranslatorSnapshots(unittest.TestCase):
    SID, MID, PID = "sess1", "msg1", "part1"

    def test_snapshot_merge_no_duplicate_bytes(self):
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", ""),
            self.SID)
        tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, "Hello"), self.SID)
        # final snapshot repeats what was streamed, then extends
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text",
                          "Hello world"), self.SID)
        self.assertEqual(_content_of(out), " world")

    def test_snapshot_behind_stream_emits_nothing(self):
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", "Hello w"),
            self.SID)
        tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, "orld"), self.SID)
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", "Hello"),
            self.SID)  # trailing-trim style snapshot
        self.assertEqual(_content_of(out), "")

    def test_late_duplicate_delta_after_snapshot_guarded(self):
        """Reverse of #26924: snapshot first, same bytes re-arrive late."""
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", "Hello"),
            self.SID)
        self.assertEqual(
            _content_of(tr.handle_session(
                _part_delta(self.SID, self.MID, self.PID, "Hello"), self.SID)),
            "")
        # genuinely new delta still passes
        self.assertEqual(
            _content_of(tr.handle_session(
                _part_delta(self.SID, self.MID, self.PID, "!"), self.SID)),
            "!")

    def test_reasoning_part_snapshot_routed_to_reasoning(self):
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, "r1", "reasoning", "think"),
            self.SID)
        self.assertEqual(_reasoning_of(out), "think")
        self.assertEqual(_content_of(out), "")

    def test_step_and_tool_parts_do_not_surface(self):
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, "s1", "step-start"), self.SID)
        self.assertEqual(out, [])
        out = tr.handle_session(
            _part_updated(self.SID, self.MID, "t1", "tool", "bash"),
            self.SID)
        self.assertEqual(out, [])
        out = tr.handle_session(
            _part_delta(self.SID, self.MID, "t1", "noise"), self.SID)
        self.assertEqual(out, [])


class TestStreamTranslatorTools(unittest.TestCase):
    SID, MID, PID = "sess1", "msg1", "part1"
    BLOCK = ('[tool_call]\n'
             '{"name":"bash","arguments":{"cmd":"ls"}}\n'
             '[/tool_call]')

    def _feed_text(self, tr, piece):
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", ""),
            self.SID)
        return tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, piece), self.SID)

    def test_tool_block_hidden_until_complete(self):
        tr = _tr(bridge=True)
        out = self._feed_text(tr, "Before " + self.BLOCK[:15])
        self.assertEqual(_content_of(out), "Before ")
        self.assertEqual(_tool_call_chunks(out), [])

    def test_tool_calls_drained_incrementally_then_finish(self):
        tr = _tr(bridge=True)
        out = self._feed_text(tr, "Hi " + self.BLOCK)
        # visible content only holds the prefix; call arrives as tool_calls
        self.assertEqual(_content_of(out), "Hi ")
        calls = _tool_call_chunks(out)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertEqual(calls[0]["index"], 0)
        self.assertTrue(calls[0]["id"].startswith("call_"))
        # final envelope: no duplicate call, finish=tool_calls
        final = ('Hi ' + self.BLOCK)
        payloads, text, _ = tr.finalize_text(final, "")
        self.assertEqual(_content_of(payloads), "")
        self.assertEqual(_tool_call_chunks(payloads), [])  # not re-emitted
        self.assertEqual(_finish_reasons(payloads), ["tool_calls"])
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(text, "Hi ")

    def test_finish_stop_without_calls(self):
        tr = _tr()
        payloads, text, reasoning = tr.finalize_text("done", "thought")
        self.assertEqual(_finish_reasons(payloads), ["stop"])
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(text, "done")
        self.assertEqual(reasoning, "thought")

    def test_reasoning_alone_still_finishes_stop_with_empty_content(self):
        """Reasoning never terminates a stream as the final message itself."""
        tr = _tr()
        tr.feed_reasoning("only thinking")
        payloads, text, reasoning = tr.finalize_text("", "only thinking")
        self.assertEqual(_finish_reasons(payloads), ["stop"])
        self.assertEqual(_content_of(payloads), "")
        self.assertEqual(text, "")
        self.assertEqual(reasoning, "only thinking")

    def test_finalize_emits_missing_content_tail_only(self):
        tr = _tr()
        tr.handle_session(_msg_updated(self.SID, self.MID), self.SID)
        tr.handle_session(
            _part_updated(self.SID, self.MID, self.PID, "text", ""),
            self.SID)
        tr.handle_session(
            _part_delta(self.SID, self.MID, self.PID, "Hel"), self.SID)
        payloads, text, _ = tr.finalize_text("Hello", "")
        self.assertEqual(_content_of(payloads), "lo")
        self.assertEqual(text, "Hello")

    def test_finalize_emits_missing_reasoning_tail_only(self):
        tr = _tr()
        tr.feed_reasoning("par")
        payloads, _, reasoning = tr.finalize_text("ans", "partial thought")
        self.assertEqual(_reasoning_of(payloads), "tial thought")
        self.assertEqual(_content_of(payloads), "ans")
        self.assertEqual(reasoning, "partial thought")


class TestStreamTranslatorError(unittest.TestCase):
    def test_error_payload_structured(self):
        obj = json.loads(StreamTranslator.error_payload("boom"))
        self.assertEqual(obj["error"]["message"], "boom")
        self.assertEqual(obj["error"]["type"], "server_error")

    def test_error_payload_default_message(self):
        obj = json.loads(StreamTranslator.error_payload(""))
        self.assertEqual(obj["error"]["message"], "proxy error")


class TestDrainCallsNoDoubleEmit(unittest.TestCase):
    def test_finish_does_not_rereturn_drained_calls(self):
        p = ToolCallStreamParser()
        block = ('[tool_call]{"name":"x","arguments":{}}[/tool_call]')
        p.feed("a " + block + " b")
        drained = p.drain_calls()
        self.assertEqual(len(drained), 1)
        rest, calls = p.finish("a " + block + " b")
        self.assertEqual(calls, [])  # already handed out
        self.assertEqual(rest, "")

    def test_finish_returns_undrained_calls(self):
        p = ToolCallStreamParser()
        p.feed('x [tool_call]{"name":"y","arguments":{}}[/tool_call]')
        rest, calls = p.finish()  # never drained
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "y")


class TestFinishChunkOrdering(unittest.TestCase):
    def test_tool_calls_before_finish(self):
        """Wire contract: all delta.tool_calls chunks precede finish_reason."""
        tr = _tr(bridge=True)
        block = ('[tool_call]{"name":"z","arguments":{}}[/tool_call]')
        out = []
        out.extend(tr.feed_content(block))
        out.extend(tr.finish_chunks())
        saw_finish = False
        for p in out:
            if p == "[DONE]":
                continue
            obj = json.loads(p)
            d = (obj.get("choices") or [{}])[0]
            if d.get("delta", {}).get("tool_calls"):
                self.assertFalse(saw_finish)
            if d.get("finish_reason"):
                saw_finish = True
        self.assertTrue(saw_finish)


if __name__ == "__main__":
    unittest.main()
