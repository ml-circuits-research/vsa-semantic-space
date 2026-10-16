#!/usr/bin/env bash
# The server test of spec section 2 (create a world, upload a legend and two examples, check, download byte-identical; plus 503, traversal,
# refused writes, the page): see tests/test4_server.sh.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tests/test4_server.sh" "$@"
