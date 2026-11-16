#!/bin/bash
# Run tests/test_ui.py on a machine without a screen (Xvfb). usage: run_ui_test.sh <blender> <perm.py> [SciPy folder] [screenshot folder]
# Exit code 0 = the script reported PASS and Blender printed no Python traceback while drawing.
BL="$1"; ADDON="$2"; SITE="$3"; SHOTS="$4"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="${LOG:-ui_test.log}"
export BLENDER_USER_RESOURCES="${BLENDER_USER_RESOURCES:-$(mktemp -d)}"      # a clean user folder: nothing of yours is touched
DISP="${DISP:-:97}"
Xvfb "$DISP" -screen 0 1920x1200x24 >/dev/null 2>&1 &
XPID=$!; sleep 2
DISPLAY="$DISP" timeout "${TIMEOUT:-600}" "$BL" --enable-event-simulate --python-exit-code 1 --window-geometry 0 0 1920 1200 \
    --python "$HERE/test_ui.py" -- --addon "$ADDON" ${SITE:+--site "$SITE"} ${SHOTS:+--shots "$SHOTS"} > "$LOG" 2>&1
RC=$?
kill $XPID 2>/dev/null
grep -E "^UI (RESULT|FAIL)" "$LOG"
if grep -q "Traceback (most recent call last)" "$LOG"; then echo "UI: Python traceback(s) in the log:"; grep -n -A6 "Traceback (most recent call last)" "$LOG" | head -60; RC=1; fi
grep -q "^UI RESULT: PASS" "$LOG" || RC=1
exit $RC
