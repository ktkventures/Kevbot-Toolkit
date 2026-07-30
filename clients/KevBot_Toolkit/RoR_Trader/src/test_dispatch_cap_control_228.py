"""Acceptance test — board #228, step 4: the daily-cap control on /admin/dispatch.

Step 1-2 made the cap runtime-adjustable (`effective_daily_cap()` re-reads
`system_settings.dispatcher_daily_cap` every poll). That is only half a feature
until there is somewhere to change it; Kevin ruled that somewhere is
/admin/dispatch (07-30: "yea put it on dispatch when you get to that point").

Two rails, and each is asserted with the behaviour that FAILS without it:

  RAIL 1 — THE PANEL READS THE LIVE ROW, NOT THE MIRROR. `DISPATCHER.DAILY_CAP`
    in taskBoardShared.tsx is a BUILD-TIME copy of dispatcher.py's fallback
    constant. It went stale under #219 and needed #226 to fix. Rendering it
    beside an editable control would put two different caps on one panel, so
    the panel renders the settings row and falls back to the constant only when
    the row is missing or unreadable — and SAYS which, because a cap that has
    quietly reverted to 50 looks identical to one that is working (#157).

  RAIL 2 — OUT OF RANGE IS REFUSED, WITH THE REASON. The loop CLAMPS to
    [1, DAILY_CAP_MAX] rather than honouring nonsense. If the page wrote an
    out-of-range value, the loop would enforce a cap nobody chose. So the page
    refuses it, using the SAME bounds, before any request is sent.

The heart of this file is not a source grep: `resolveDailyCap` /
`validateCapInput` are LIFTED OUT of the .tsx and EXECUTED in node, then
cross-checked against `effective_daily_cap()` — the Python SSOT — on the same
inputs. A divergence between the page's policy and the loop's policy fails here
rather than in production. (The frontend has no JS test runner; node itself is
enough for two pure functions.)

`effective_daily_cap()` is NOT modified by this step and is not re-tested here —
that is test_dispatcher_runtime_cap_228.py's job. This file consumes it as truth.

Run:  .venv/bin/python src/test_dispatch_cap_control_228.py   (from RoR_Trader root)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
FE = os.path.join(ROOT, "frontend", "src")
sys.path.insert(0, os.path.join(ROOT, "tools", "team_dispatcher"))
import dispatcher as D  # noqa: E402

FAILED = []
PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}" + (f" — {detail}" if detail else ""))
        FAILED.append(label)


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


PAGE = read(FE, "views", "AdminDispatchPage.tsx")
SHARED = read(FE, "views", "taskBoardShared.tsx")


# ── lifting the pure TS out of the page so it can actually be RUN ───────────

def js_body(src, name):
    """Return `function name(args) { ... }` as runnable JS.

    Only the SIGNATURE is rewritten (params de-typed, return type dropped) —
    the body is copied verbatim, so what runs here is what ships. The body's
    end is found by brace matching from the signature's final `{`.
    """
    m = re.search(rf"^export function {name}\(", src, re.M)
    assert m, f"{name} not found in the page — the control is missing"
    line_end = src.index("\n", m.start())
    sig = src[m.start():line_end]
    params = sig[sig.index("(") + 1:sig.index(")")]
    names = [p.split(":")[0].strip() for p in params.split(",") if p.strip()]
    i, depth = line_end, 1                      # the signature's own `{`
    while depth:
        i += 1
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
    return f"function {name}({', '.join(names)}) {{\n" + src[line_end + 1:i + 1]


mirror = re.search(r"export const DISPATCHER = \{[^}]*\}", SHARED)
ok("taskBoardShared still exports the DISPATCHER mirror", bool(mirror))
PRELUDE = "const " + mirror.group(0).split("export const ", 1)[1] + ";\n"

NODE = shutil.which("node")
ok("node is available to execute the lifted functions", bool(NODE))


def run_js(cases):
    """Execute the page's own resolveDailyCap/validateCapInput on `cases`."""
    harness = (
        PRELUDE
        + js_body(PAGE, "resolveDailyCap") + "\n"
        + js_body(PAGE, "validateCapInput") + "\n"
        + "const cases = JSON.parse(process.argv[2]);\n"
        + "console.log(JSON.stringify(cases.map(c => c.fn === 'resolve'\n"
        + "  ? resolveDailyCap(c.arg) : validateCapInput(c.arg))));\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        out = subprocess.run([NODE, path, json.dumps(cases)],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:400]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


class FakeApi:
    """Stands in for dispatcher.api() so the Python SSOT can be driven offline."""

    def __init__(self, rows):
        self.rows = rows

    def __call__(self, method, path, *a, **k):
        return self.rows


def py_cap(db_value, missing=False):
    real = D.api
    D.api = FakeApi([] if missing else [{"value": db_value}])
    try:
        return D.effective_daily_cap()
    finally:
        D.api = real


# ── RAIL 1a — the mirrored bounds/key are dispatcher.py's, not invented ─────
print("― rail 1a: the UI's bounds and key mirror dispatcher.py, the SSOT ―")
ui = dict(re.findall(r"(\w+):\s*(\d+)", mirror.group(0)))
ok(f"DAILY_CAP_MAX mirrors dispatcher.py ({D.DAILY_CAP_MAX})",
   int(ui.get("DAILY_CAP_MAX", -1)) == D.DAILY_CAP_MAX, f"UI has {ui.get('DAILY_CAP_MAX')!r}")
ok("DAILY_CAP_MIN is the 1 effective_daily_cap() clamps to",
   int(ui.get("DAILY_CAP_MIN", -1)) == 1, f"UI has {ui.get('DAILY_CAP_MIN')!r}")
key = re.search(r"DAILY_CAP_SETTING:\s*'([^']+)'", mirror.group(0))
ok(f"DAILY_CAP_SETTING mirrors dispatcher.py ('{D.DAILY_CAP_SETTING}')",
   bool(key) and key.group(1) == D.DAILY_CAP_SETTING, key.group(1) if key else "absent")
ok("the endpoint is BUILT from that key, so read and write cannot drift",
   re.search(r"const CAP_ENDPOINT = `/api/admin/system-settings/\$\{DISPATCHER\.DAILY_CAP_SETTING\}`",
             PAGE) is not None)

# ── RAIL 1b — the panel renders the live value, never the mirror ────────────
print("― rail 1b: the lane panel prints the LIVE cap, not DISPATCHER.DAILY_CAP ―")
ok("the runs-today line is denominated by the live capState",
   "{startedToday}/{capState.cap}" in PAGE)
ok("it is NOT denominated by the build-time mirror",
   "{startedToday}/{DISPATCHER.DAILY_CAP}" not in PAGE)
# The ONLY places the mirror may still appear are the fallback legs and the
# prose that explains it is a fallback — never as the rendered live number.
live_render = re.findall(r"capState\.cap\b", PAGE)
ok("capState.cap is what the control and the panel both render", len(live_render) >= 2,
   f"found {len(live_render)}")
ok("the lifted body carries no TS annotations node cannot run (it is real code, not a copy)",
   ": number;" not in js_body(PAGE, "resolveDailyCap"))
ok("the page reads the settings row on its own cadence",
   "const loadCap = useCallback" in PAGE and "apiFetch<CapSetting>(CAP_ENDPOINT)" in PAGE)
ok("an unreadable settings endpoint is NOT collapsed into 'no row'",
   "capRead === 'error'" in PAGE and "constant (settings unreadable)" in PAGE)
ok("the source the loop reports is surfaced on the panel",
   "source: <code" in PAGE and "capState.source" in PAGE)
ok("a fallback to the constant is called out loudly, not left to look normal",
   "fallback constant" in PAGE and "!capState.fromSettings" in PAGE)
ok("the panel says WHEN a change takes effect (next poll), so a lag is not a bug",
   re.search(r"next poll \(~\{DISPATCHER\.POLL_S\}s\)", PAGE) is not None)
ok("and says it costs no deploy and no restart",
   "no deploy, no\n              restart" in PAGE or "no deploy, no restart" in PAGE)

# ── RAIL 1c — the page's resolve policy IS the loop's, executed side by side ─
print("― rail 1c: resolveDailyCap == effective_daily_cap(), run on the same inputs ―")
CASES = [75, 50, 1, 500, 0, -5, 501, 9999, "60", " 60 ", "60.5", "abc", 12.5, -0.5]
js = run_js([{"fn": "resolve", "arg": c} for c in CASES])
for c, got in zip(CASES, js):
    want_cap, want_src = py_cap(c)
    ok(f"row={c!r} -> cap {want_cap} (python) == {got['cap']} (page)",
       got["cap"] == want_cap, f"page said {got['cap']}, source {got['source']!r}")
    # Sources are compared by CLASSIFICATION, not byte-for-byte: Python and JS
    # quote a bad value differently. What must agree is which leg was taken.
    def leg(s):
        if "no row" in s or "not an int" in s or "unreadable" in s:
            return "fallback"
        return "clamped" if "CLAMPED" in s else "settings"
    ok(f"row={c!r} -> same leg ({leg(want_src)})", leg(got["source"]) == leg(want_src),
       f"python {want_src!r} vs page {got['source']!r}")

missing = run_js([{"fn": "resolve", "arg": None}])[0]
py_missing_cap, py_missing_src = py_cap(None, missing=True)
ok("a missing row falls back to the same constant on both sides",
   missing["cap"] == py_missing_cap == D.DAILY_CAP, f"page {missing['cap']}, python {py_missing_cap}")
ok("and both say 'no row'", "no row" in missing["source"] and "no row" in py_missing_src,
   missing["source"])
ok("the fallback leg is flagged as NOT from settings", missing["fromSettings"] is False)

# ── RAIL 2 — out of range is refused by the page, with a reason ─────────────
print("― rail 2: the edit box REFUSES what the loop would silently clamp ―")
BAD = ["0", "-1", "-500", "501", "1000", "", "  ", "abc", "12.5", "1e3", "50 runs"]
GOOD = ["1", "50", "500", " 75 "]
res_bad = run_js([{"fn": "validate", "arg": b} for b in BAD])
for raw, got in zip(BAD, res_bad):
    ok(f"refused: {raw!r}", got["ok"] is False, str(got))
    ok(f"…and the refusal names a reason: {raw!r}", bool(got.get("reason")), str(got))
res_good = run_js([{"fn": "validate", "arg": g} for g in GOOD])
for raw, got in zip(GOOD, res_good):
    ok(f"accepted: {raw!r} -> {got.get('value')}", got["ok"] is True, str(got))
    py_c, _ = py_cap(got["value"])
    ok(f"…and the loop would enforce exactly that ({raw!r})", py_c == got["value"],
       f"python {py_c} vs accepted {got['value']}")
# The bound must come from the mirror, not a literal typed twice.
ok("the validator's bounds come from DISPATCHER, not a hard-coded literal",
   "DAILY_CAP_MIN: lo, DAILY_CAP_MAX: hi } = DISPATCHER" in PAGE)
ok("0/negatives are refused for the loop's own reason (no disabling the breaker)",
   "no setting may disable the breaker" in PAGE)
save = PAGE.split("const saveCap = useCallback")[1].split("}, [capDraft")[0]
ok("the gate runs BEFORE the request — a refused value never reaches the row",
   save.index("validateCapInput") < save.index("apiFetch"))
ok("a refusal returns without saving", "if (!v.ok)" in save and "return; }" in save)

# ── no schema, no SSOT edit ────────────────────────────────────────────────
print("― rails: no schema change, and effective_daily_cap() is untouched ―")
ok("the page adds no migration and no new table",
   "CREATE TABLE" not in PAGE.upper() and "migrations/" not in PAGE)
ok("effective_daily_cap() is still the one place the clamp is defined",
   "return clamped, f\"settings CLAMPED" in read(ROOT, "tools", "team_dispatcher", "dispatcher.py"))
ok("the page never reaches the table directly — it goes through the admin API",
   not re.search(r"supabase|\.from\(['\"]system_settings", PAGE))

print(f"\n{'ALL PASS' if not FAILED else 'FAILURES'} ({PASS} checks"
      + (f", {len(FAILED)} FAILED" if FAILED else "") + ")")
if FAILED:
    for f_ in FAILED:
        print(f"  - {f_}")
    sys.exit(1)
