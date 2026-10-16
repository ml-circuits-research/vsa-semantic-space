#!/usr/bin/env bash
# Test 4 (build order step 4) + test 6 (the demo page): the worlds server.
# Creates a world, uploads a legend (+ its lexicon) and two examples, checks them, downloads them back byte-identical; formalize answers 503
# without a model; path traversal is rejected; writes to model/ and runs/ are refused; DELETE needs ?confirm; requests and versions routes;
# GET / serves the page with the formalize textarea; with a model copied in (from tool/worlds/reference when it exists), formalize answers 200.
set -uo pipefail
TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sop_server_XXXX")"; PORT=$(( 20000 + RANDOM % 20000 )); B="http://127.0.0.1:$PORT"
FAILS=0; ok() { if [ "$1" = "$2" ]; then echo "PASS $3"; else echo "FAIL $3 (got $1, want $2)"; FAILS=$((FAILS+1)); fi; }
node "$TOOL/server.js" --root "$ROOT" --port "$PORT" > "$ROOT/server.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$ROOT"' EXIT
for i in $(seq 50); do curl -s "$B/worlds" >/dev/null 2>&1 && break; sleep 0.1; done
code() { curl -s -o "$ROOT/body" -w '%{http_code}' "$@"; }

ok "$(code -X POST "$B/worlds/shop")" 201 "POST /worlds/shop creates the folder layout"
ok "$(code -X POST "$B/worlds/shop")" 409 "POST again: 409 exists"
for f in world.json status.json examples raw audit model runs versions requests; do [ -e "$ROOT/shop/$f" ] || { echo "FAIL layout: $f missing"; FAILS=$((FAILS+1)); }; done
python3 -c "import json,sys; w=json.load(open('$ROOT/shop/world.json')); assert set(w['models'])=={'pipeline','annotate','second_look','evolve'} and w['second_look_escalate']==0.05 and 'min_examples' in w and 'max_versions_per_run' in w" \
  && echo "PASS world.json carries models per step, min_examples, max_versions_per_run, second_look_escalate" || { echo "FAIL world.json fields"; FAILS=$((FAILS+1)); }

cat > "$ROOT/legend.json" <<'EOF'
{"version": 1, "roles": [{"id": "buyer", "question": "who buys?"}, {"id": "item", "question": "what?"}],
 "events": [{"id": "purchase", "verbs": ["buy"]}], "operators": ["not", "says"], "moods": ["statement"],
 "kinds": {"source": "lexicon", "lexicon": "kinds.tsv"}, "reader": {"predicate_role": true, "occupancy_marker": true, "definite_dets": ["the"]}}
EOF
printf 'Anna\tP\nPaul\tP\nbuys\tbuys\tbuy\nbook\tN\npen\tN\na\ta\n.\t.\n' > "$ROOT/kinds.tsv"
SHA="sha256:$(sha256sum "$ROOT/legend.json" | cut -d' ' -f1)"
printf '!world shop\n!legend %s\n\n@w1 purchase\n_text=Anna buys a book .\n_source=human\nbuyer=Anna\nitem=book\n' "$SHA" > "$ROOT/a.sop"
printf '!world shop\n!legend %s\n\n@w1 purchase\n_text=Paul buys a pen .\n_source=human\nbuyer=Paul\nitem="pen"\n' "$SHA" > "$ROOT/b.sop"
ok "$(code -X PUT --data-binary @"$ROOT/legend.json" "$B/worlds/shop/files/legend.json")" 201 "PUT legend.json"
ok "$(code -X PUT --data-binary @"$ROOT/kinds.tsv" "$B/worlds/shop/files/kinds.tsv")" 201 "PUT kinds.tsv"
ok "$(code -X PUT --data-binary @"$ROOT/a.sop" "$B/worlds/shop/files/examples/a.sop")" 201 "PUT examples/a.sop"
ok "$(code -X PUT --data-binary @"$ROOT/b.sop" "$B/worlds/shop/files/examples/sub/b.sop")" 201 "PUT examples/sub/b.sop (subfolder created)"
ok "$(code -X POST "$B/worlds/shop/check")" 200 "POST check"
python3 -c "import json; r=json.load(open('$ROOT/body')); assert r['ok'] and len(r['report']['files'])==2, r" && echo "PASS check report: 2 files OK" || { echo "FAIL check report: $(cat "$ROOT/body")"; FAILS=$((FAILS+1)); }
curl -s "$B/worlds/shop/files/examples/a.sop" -o "$ROOT/a.back"; curl -s "$B/worlds/shop/files/examples/sub/b.sop" -o "$ROOT/b.back"
cmp -s "$ROOT/a.sop" "$ROOT/a.back" && cmp -s "$ROOT/b.sop" "$ROOT/b.back" && echo "PASS download byte-identical (2 examples)" || { echo "FAIL download differs"; FAILS=$((FAILS+1)); }
printf '!world shop\n!legend %s\n\n@w1 purchase\n_text=Anna buys .\nbuyer=Anna\nprice=4\n' "$SHA" > "$ROOT/bad.sop"
code -X PUT --data-binary @"$ROOT/bad.sop" "$B/worlds/shop/files/examples/bad.sop" >/dev/null
code -X POST "$B/worlds/shop/check" >/dev/null
python3 -c "import json; r=json.load(open('$ROOT/body')); assert not r['ok'] and r['exit_code']==1" && echo "PASS check reports the bad example (exit 1, unknown key)" || { echo "FAIL bad example not reported"; FAILS=$((FAILS+1)); }
ok "$(code -X DELETE "$B/worlds/shop/files/examples/bad.sop")" 200 "DELETE examples/bad.sop"
ok "$(code -X POST --data-binary 'Anna buys a book .' "$B/worlds/shop/formalize")" 503 "formalize without a model: 503"
ok "$(code --path-as-is "$B/worlds/shop/files/../../etc/passwd")" 400 "path traversal (../) rejected"
ok "$(code "$B/worlds/shop/files/examples/%2e%2e/%2e%2e/world.json")" 400 "path traversal (%2e%2e) rejected"
ok "$(code -X PUT --data-binary x "$B/worlds/shop/files/examples/%2e%2e/model/x")" 400 "encoded traversal on PUT rejected"
ok "$(code -X PUT --data-binary x "$B/worlds/shop/files/model/config.json")" 403 "PUT into model/ refused"
ok "$(code -X PUT --data-binary x "$B/worlds/shop/files/runs/x.log")" 403 "PUT into runs/ refused"
ok "$(code -X PUT --data-binary '{}' "$B/worlds/shop/files/split_hints.json")" 201 "PUT split_hints.json (top-level writable)"
ok "$(code -X DELETE "$B/worlds/shop/files/split_hints.json")" 200 "DELETE of a top-level writable file allowed"
ok "$(code -X DELETE "$B/worlds/shop/files/world.json")" 403 "DELETE world.json refused (the manifest)"
ok "$(code -X PUT --data-binary x "$B/worlds/shop/files/status.json")" 403 "PUT status.json refused (written by training)"
ok "$(code "$B/worlds/shop/files?dir=examples")" 200 "GET files?dir=examples"
python3 -c "import json; r=json.load(open('$ROOT/body')); assert sorted(f['name'] for f in r['files'])==['a.sop','sub/b.sop'], r" && echo "PASS listing shows a.sop, sub/b.sop" || { echo "FAIL listing"; FAILS=$((FAILS+1)); }
ok "$(code -X PUT --data-binary 'please add a role for the price' "$B/worlds/shop/requests/price.md")" 201 "PUT requests/price.md"
ok "$(code "$B/worlds/shop/versions")" 200 "GET versions"
mkdir -p "$ROOT/shop/versions/v1"; cp "$ROOT/legend.json" "$ROOT/shop/versions/v1/legend.json"
ok "$(code -X POST "$B/worlds/shop/versions/1/pin")" 200 "POST versions/1/pin"
ok "$(code -X POST "$B/worlds/shop/versions/7/reject")" 404 "reject of a missing version: 404"
ok "$(code "$B/worlds")" 200 "GET /worlds"
ok "$(code "$B/worlds/shop")" 200 "GET /worlds/shop (world, legend, inventory, status)"
ok "$(code -X DELETE "$B/worlds/shop")" 400 "DELETE world without ?confirm: 400"
ok "$(code "$B/")" 200 "GET / (test 6: the page)"
grep -q "<textarea" "$ROOT/body" && grep -q "/formalize" "$ROOT/body" && grep -q "version history" "$ROOT/body" && echo "PASS page has the worlds list, upload form, version history, formalize textarea" || { echo "FAIL page content"; FAILS=$((FAILS+1)); }
if [ -f "$TOOL/worlds/reference/model/config.json" ]; then
  ok "$(code -X POST "$B/worlds/ref")" 201 "POST /worlds/ref"
  cp -r "$TOOL/worlds/reference/model/." "$ROOT/ref/model/"
  ok "$(code -X POST --data-binary $'p3 says that p0 gives o1 to p2 .\np5 receives o2 from p1 .' "$B/worlds/ref/formalize")" 200 "formalize with a model (test 6: the textarea's call)"
  grep -q "^@w1 says" "$ROOT/body" && grep -q "_world=reference" "$ROOT/body" && echo "PASS formalize returns SOP Lang with _world" || { echo "FAIL formalize body: $(head -5 "$ROOT/body")"; FAILS=$((FAILS+1)); }
else echo "SKIP formalize-with-model (no tool/worlds/reference/model; run tests/test2_formalize.sh first)"; fi
ok "$(code -X DELETE "$B/worlds/shop?confirm=shop")" 200 "DELETE world with ?confirm"
[ "$(wc -l < "$ROOT/server.log")" -gt 20 ] && echo "PASS one log line per request ($(wc -l < "$ROOT/server.log") lines)" || { echo "FAIL server log"; FAILS=$((FAILS+1)); }
echo; [ $FAILS -eq 0 ] && echo "test4 (server) + test6 (page): PASS" || echo "test4/6: $FAILS FAILED"
exit $FAILS
