#!/usr/bin/env bash
# The six tests of the build order (indications section 2), in one command:
#   1 SOP Lang parser/writer + sop check   2 sop formalize on the three worlds   3 sop diff / stats (injected errors)
#   4 the server                            5 the skill loop + missing-role recovery   6 the demo page (inside test 4)
# CPU only. Test 2 trains the smoke readers that tests 3, 4 (formalize with a model) and test_sop.sh use.
set -uo pipefail
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TOOL="$(dirname "$T")"
PY="${SOP_PYTHON:-}"; for c in "$TOOL/../.venv/bin/python3" "$TOOL/../../../../.venv/bin/python3"; do [ -z "$PY" ] && [ -x "$c" ] && PY="$c"; done; PY="${PY:-python3}"
declare -A R
"$PY" "$T/test1_soplang.py"; R[1]=$?
"$T/test2_formalize.sh"; R[2]=$?
"$PY" "$T/test3_diff.py"; R[3]=$?
"$T/test4_server.sh"; R[4]=$?
"$T/test5_skill.sh"; R[5]=$?
echo; echo "== summary"
for k in 1 2 3 4 5; do echo "test $k: $([ "${R[$k]}" -eq 0 ] && echo PASS || echo "FAIL (${R[$k]})")"; done
echo "test 6 (page): inside test 4"
for k in 1 2 3 4 5; do [ "${R[$k]}" -eq 0 ] || exit 1; done
