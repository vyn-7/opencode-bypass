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
$CURL -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"say hi in 3 words"}]}'; echo

echo "== streaming chat =="
$CURL -N -X POST "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"say hi in 3 words"}],"stream":true}'; echo

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
