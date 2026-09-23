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
    msg_digest,
    normalize_model,
    plan_prompt,
    spawn_detached_kwargs,
    split_model,
    streaming_view,
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


if __name__ == "__main__":
    unittest.main()
