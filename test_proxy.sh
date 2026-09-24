#!/usr/bin/env bash
# Smoke-test the proxy. Expects it running (./install.sh or ./run.sh start).
# Fails loudly (nonzero exit) on the first broken endpoint.
set -euo pipefail
BASE="${1:-http://127.0.0.1:18788}"
CURL="curl -sS -f"

echo "== /health =="
$CURL "$BASE/health"; echo

echo "== /v1/models =="
$CURL "$BASE/v1/models"; echo

echo "== non-streaming chat =="
NONSTREAM=$($CURL -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"say hi in 3 words"}]}')
echo "$NONSTREAM"
python3 -c '
import json,sys
d=json.load(sys.stdin)
msg=d["choices"][0]["message"]
# reasoning_content is optional (model-dependent) but must be a string
# separate from content when present
rc=msg.get("reasoning_content")
if rc is not None:
    assert isinstance(rc,str), "reasoning_content must be string"
    print("blocking reasoning_content OK:", len(rc), "chars")
print("blocking content OK:", repr((msg.get("content") or "")[:60]))
' <<<"$NONSTREAM"

echo "== streaming chat =="
STREAM_OUT=$($CURL -N -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"say hi in 3 words"}],"stream":true}')
echo "$STREAM_OUT"
python3 -c '
import json,sys
raw=sys.stdin.read()
assert "[proxy error]" not in raw, "proxy fabricated [proxy error] content"
lines=[l[6:] for l in raw.splitlines() if l.startswith("data: ")]
assert lines, "no SSE data lines"
assert lines[-1]=="[DONE]", "stream must end with [DONE], got %r" % lines[-1]
first=json.loads(lines[0])
assert first["choices"][0]["delta"].get("role")=="assistant", "first chunk must be role"
# reasoning may be absent (model-dependent); when present it must never
# appear inside content deltas
reasoning=""; content=""
for p in lines[1:-1]:
    o=json.loads(p)
    if "error" in o:
        continue  # structured mid-stream error event (allowed, not content)
    d=o["choices"][0].get("delta") or {}
    reasoning+=d.get("reasoning_content") or ""
    content+=d.get("content") or ""
finishes=[json.loads(p)["choices"][0].get("finish_reason") for p in lines[1:-1]]
finishes=[f for f in finishes if f]
assert finishes, "missing finish_reason"
if reasoning:
    assert reasoning not in content or reasoning==content, "reasoning leaked into content"
    assert not any(f=="stop" and not content for f in finishes) or content, "reasoning-only finish"
    print("reasoning channel OK:", len(reasoning), "chars,", len(content), "content chars")
else:
    print("no reasoning this run (model-dependent), content chars:", len(content))
print("stream structure OK: role first, finish=%r, [DONE] last" % finishes)
' <<<"$STREAM_OUT"

echo "== memory recall (turn 2, delta session) =="
TURN1=$($CURL -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"My favorite color is teal. Reply with just OK."}]}')
echo "$TURN1"
T1_CONTENT=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' <<<"$TURN1")
TURN2=$($CURL -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys; t1=sys.argv[1]; print(json.dumps({"model":"big-pickle","messages":[{"role":"user","content":"My favorite color is teal. Reply with just OK."},{"role":"assistant","content":t1},{"role":"user","content":"What is my favorite color? One word only."}]}))' "$T1_CONTENT")")
echo "$TURN2"
T2_CONTENT=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' <<<"$TURN2")
echo "turn2 said: $T2_CONTENT"
if ! grep -qi "teal" <<<"$T2_CONTENT"; then
  echo "FAIL: turn 2 does not recall 'teal'" >&2
  exit 1
fi

echo "== tools bridge (blocking) =="
TOOL_RESP=$($CURL -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"Use the list_files tool to list files. Reply with one word after if needed."}],"tools":[{"type":"function","function":{"name":"list_files","description":"List files in the current directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}],"tool_choice":"auto"}')
echo "$TOOL_RESP"
python3 -c '
import json,sys
d=json.load(sys.stdin)
ch=d["choices"][0]
fin=ch.get("finish_reason")
msg=ch.get("message") or {}
if fin=="tool_calls":
    tcs=msg.get("tool_calls") or []
    assert tcs and tcs[0].get("function",{}).get("name"), "missing tool_call name"
    print("tool_calls OK:", tcs[0]["function"]["name"], tcs[0]["function"].get("arguments"))
else:
    print("finish=", fin, "content=", (msg.get("content") or "")[:80])
' <<<"$TOOL_RESP"

echo "== legacy path /chat/completions =="
$CURL -X POST "$BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"say hi in 3 words"}]}'; echo

echo ALL SMOKE TESTS DONE
