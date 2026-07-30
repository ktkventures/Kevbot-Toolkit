"""Acceptance test — board #240: the M-session activity log is BOUNDED to render.

Kevin, 07-30, first real use of the dashboard shipped hours earlier: *"some of
these modules are really long, like the activity log module. I think we should
probably have it have a scroll bar inside of the module and then have it
lazy-load stuff… it's kind of stalling my computer out trying to look at it."*

Measured cause: `/api/dev-tasks/comments/recent` returns up to 300 comments WITH
bodies, `run_history` adds ~135 more, and `AdminMSessionPage.tsx` mapped every
one of them into the DOM on first paint — **~435 rich rows**, each with a body
blob, an expand handler and three mark controls.

The constraint that shapes the fix: the activity log is an ACCOUNTABILITY
RECORD (event → M saw it → M did X, or judged none needed). A window that only
ever HELD the last N events would quietly stop being one. So the rail is
**render less at a time, never keep less**.

Four rails, each asserted by the behaviour that FAILS without it:

  RAIL 1 — THE MODULE SCROLLS INSIDE ITSELF. Section 2's day list lives in a
    container with a bounded `maxHeight` and `overflowY: auto`, so the PAGE
    stops growing with the board and sections 3/4 and the known-gaps footer
    stay reachable. Without it the page is as tall as the board is old.

  RAIL 2 — THE FIRST PAINT IS CAPPED WELL BELOW THE ROW COUNT. `feedWindow` is
    LIFTED OUT of mSessionFeed.ts and EXECUTED in node against a synthetic
    435-row feed — the measured first paint. One page must be a small fraction
    of it, and the page must start at page 1.

  RAIL 3 — SECTION 5 (ASK-AI) IS *NOT* BOUNDED. It is Kevin's fast lane and the
    top-priority queue; bounding it would hide the one thing on this page that
    gets worse with time. Asserted as an ABSENCE inside section 5's own JSX
    region, so a future copy-paste of the scroller into it fails here.

  RAIL 4 — NO EVENT IS DROPPED. Executed, not asserted in prose: raising the
    page count until the window is exhausted must reproduce the input list
    byte-for-byte and in order, the input must not be mutated, and the page must
    not slice `comments` / `feed` / `feedRows` anywhere. The fetch limits stay
    at their pre-#240 values — the bound is on the DOM, not on the record.

Plus the payload trim (`max_chars` 1400 → 600) — a secondary, not a rail: it
must go DOWN without falling below what the feed excerpt renders.

The frontend has no JS test runner; node itself is enough for two pure
functions (the board-#228 lifting pattern, reused deliberately). Nothing here
touches `api/routers/m_session.py` — see the #220 path-computation incident.

Run:  .venv/bin/python src/test_m_session_activity_log_bound_240.py
      (from RoR_Trader root)
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
    p = os.path.join(*parts)
    if not os.path.exists(p):
        print(f"  FAIL: missing {p}")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return f.read()


PAGE = read(FE, "views", "AdminMSessionPage.tsx")
FEEDTS = read(FE, "views", "mSessionFeed.ts")


# ── Lifting the pure TS out of the module so it can actually be RUN ─────────

def js_fn(src, name):
    """Return `function name(args) { ... }` as runnable JS.

    Only the SIGNATURE is rewritten (param types dropped, defaults kept, return
    type dropped) — the body is copied VERBATIM, so what runs here is what
    ships. Handles a signature wrapped over several lines, which the #228
    lifter did not need to.
    """
    m = re.search(rf"^export function {name}\(", src, re.M)
    if not m:
        ok(f"mSessionFeed.ts exports {name}() — the render bound itself", False,
           "the activity log is unbounded: there is no window function to lift")
        return None
    i = src.index("(", m.start())
    depth, j = 1, i
    while depth:                                   # match the param list
        j += 1
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
    params_src = src[i + 1:j]
    brace = src.index("{", j)
    depth, k = 1, brace
    while depth:                                   # match the body
        k += 1
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
    names = []
    for raw in params_src.split(","):
        p = raw.strip().rstrip(",")
        if not p:
            continue
        default = f" = {p.split('=', 1)[1].strip()}" if "=" in p else ""
        names.append(p.split(":")[0].strip() + default)
    return f"function {name}({', '.join(names)}) {{\n" + src[brace + 1:k + 1]


def ts_const(name, fallback):
    """The named constant, or `fallback` with a FAIL recorded. A missing
    constant is not a broken test — it is the bound being absent, which is
    exactly what this file exists to catch."""
    m = re.search(rf"^export const {name} = (-?\d+);", FEEDTS, re.M)
    if not m:
        ok(f"mSessionFeed.ts declares {name}", False,
           "the render bound is not declared — the log renders every row")
        return fallback
    return int(m.group(1))


NODE = shutil.which("node")
ok("node is available to execute the lifted functions", bool(NODE))
if not NODE:
    sys.exit(1)

PAGE_ROWS = ts_const("FEED_PAGE_ROWS", 50)
MAX_H = ts_const("FEED_MAX_HEIGHT", -1)
SLACK = ts_const("FEED_SCROLL_SLACK", 240)

PRELUDE = (
    f"const FEED_PAGE_ROWS = {PAGE_ROWS};\n"
    f"const FEED_MAX_HEIGHT = {MAX_H};\n"
    f"const FEED_SCROLL_SLACK = {SLACK};\n"
)


WINDOW_FN = js_fn(FEEDTS, "feedWindow")
NEAR_FN = js_fn(FEEDTS, "nearBottom")


def run_js(script):
    if WINDOW_FN is None or NEAR_FN is None:
        return None
    harness = PRELUDE + WINDOW_FN + "\n" + NEAR_FN + "\n" + script
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        out = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise AssertionError(f"node failed: {out.stderr.strip()[:600]}")
        return json.loads(out.stdout)
    finally:
        os.unlink(path)


ok("the lifted bodies carry no TS annotations node cannot run (real code, not a copy)",
   WINDOW_FN is not None and NEAR_FN is not None
   and ": number" not in WINDOW_FN and ": number" not in NEAR_FN)


def region(src, start_marker, end_marker, label):
    """The JSX between two markers — used to assert PRESENCE in one section and
    ABSENCE in another, which a whole-file grep cannot distinguish."""
    a = src.find(start_marker)
    b = src.find(end_marker, a) if a >= 0 else -1
    ok(f"located the {label} region", b > a >= 0)
    return src[a:b] if b > a >= 0 else ""


SEC5 = region(PAGE, 'n="5"', "{/* ── 1 · M's QUEUE", "ask-ai inbox (section 5)")
SEC2 = region(PAGE, 'n="2"', "{/* ── 3 · THE RULES", "activity log (section 2)")


# ── RAIL 1 — the module scrolls inside itself ───────────────────────────────
print("― rail 1: the activity log is a bounded, internally-scrolling module ―")
ok("mSessionFeed declares the bound as a named constant, not a magic number",
   "export const FEED_MAX_HEIGHT" in FEEDTS)
ok(f"the bound is a real one ({MAX_H}px — shorter than a viewport, not 100000)",
   200 <= MAX_H <= 1000, f"FEED_MAX_HEIGHT={MAX_H}")
ok("section 2 has a container with that maxHeight",
   "maxHeight: FEED_MAX_HEIGHT" in SEC2)
ok("…and it actually overflows internally",
   re.search(r"overflowY:\s*'auto'", SEC2) is not None)
ok("the scroll is contained — a bottomed-out log does not chain to the page",
   "overscrollBehavior: 'contain'" in SEC2)

scroller = SEC2.find("maxHeight: FEED_MAX_HEIGHT")
ok("the DAY LIST is INSIDE the scroller (a bound above an unbounded list is decoration)",
   scroller >= 0 and SEC2.find("feedDays.map") > scroller,
   "feedDays.map renders before the bounded container opens")
ok("the filter controls stay OUTSIDE it — you cannot filter a list you scrolled past",
   scroller >= 0 and 0 <= SEC2.find("setShowRoutine") < scroller)
ok("the scroller is wired to a ref and an onScroll handler",
   "ref={feedScroll}" in SEC2 and "onScroll={onFeedScroll}" in SEC2)
ok("the page declares that ref",
   "const feedScroll = useRef<HTMLDivElement | null>(null)" in PAGE)


# ── RAIL 2 — the first paint is capped well below the row count ─────────────
print("― rail 2: one page, not 435 rows — executed, not eyeballed ―")
ok("mSessionFeed exports the page size as a constant", "export const FEED_PAGE_ROWS" in FEEDTS)
res = run_js(
    "const N = 435;\n"                          # the measured first paint
    "const rows = Array.from({length: N}, (_, i) => ({key: 'e' + i, at: String(i)}));\n"
    "const frozen = rows.slice();\n"
    "const first = feedWindow(rows, 1);\n"
    "let pages = 1, w = first, seen = [];\n"
    "while (w.more && pages < 1000) { pages += 1; w = feedWindow(rows, pages); }\n"
    "seen = w.rows.map(r => r.key);\n"
    "console.log(JSON.stringify({\n"
    "  firstShown: first.shown, firstHidden: first.hidden, firstTotal: first.total,\n"
    "  firstMore: first.more, firstKeys: first.rows.map(r => r.key),\n"
    "  pages, finalShown: w.shown, finalHidden: w.hidden, finalMore: w.more,\n"
    "  seen, inputUnchanged: JSON.stringify(rows) === JSON.stringify(frozen),\n"
    "  empty: feedWindow([], 1), zeroPages: feedWindow(rows, 0).shown,\n"
    "  negPages: feedWindow(rows, -5).shown, nanPages: feedWindow(rows, NaN).shown,\n"
    "  nullRows: feedWindow(null, 1).total,\n"
    "  near: [nearBottom(0, 620, 4000), nearBottom(3200, 620, 4000),\n"
    "         nearBottom(3140, 620, 4000), nearBottom(0, 300, 300)],\n"
    "}));\n")


class _Absent(dict):
    """With no window function to run, every executed assertion below must go
    RED — not raise. `feedWindow` missing IS the defect."""

    def __missing__(self, k):
        return None


res = res if res is not None else _Absent()
if isinstance(res, _Absent):
    res["empty"] = _Absent()
    res["near"] = [None, None, None, None]

ok(f"one page renders exactly FEED_PAGE_ROWS ({PAGE_ROWS}) of 435",
   res["firstShown"] == PAGE_ROWS, f"got {res['firstShown']}")
ok("that is well below the full row count (≤15% of the measured first paint)",
   isinstance(res["firstShown"], int) and res["firstShown"] <= 435 * 0.15,
   f"{res['firstShown']}/435")
ok("the rest is HIDDEN, not gone — the count is carried on the window",
   res["firstHidden"] == 435 - PAGE_ROWS and res["firstTotal"] == 435,
   f"hidden={res['firstHidden']} total={res['firstTotal']}")
ok("the window says there is more to paint", res["firstMore"] is True)
ok("the first page is the NEWEST rows, in feed order",
   res["firstKeys"] == [f"e{i}" for i in range(PAGE_ROWS)])
ok("the page starts at page 1", "useState(1)" in PAGE and "const [feedPages, setFeedPages]" in PAGE)
ok("the page renders the WINDOW, not the full filtered list",
   "feedWindow(feedRows, feedPages)" in PAGE
   and "groupByDay(win.rows, utcDay)" in PAGE)
ok("changing the filter re-bounds instead of inheriting the other list's page count",
   "setFeedPages(1); }, [showRoutine])" in PAGE)

print("― rail 2b: degenerate inputs cannot un-bound the render ―")
ok("pages=0 still paints one page, never zero", res["zeroPages"] == PAGE_ROWS)
ok("a negative page count is clamped, not treated as a slice from the end",
   res["negPages"] == PAGE_ROWS)
ok("NaN pages is clamped to one page", res["nanPages"] == PAGE_ROWS)
ok("an empty feed is an empty window, not a crash",
   res["empty"]["shown"] == 0 and res["empty"]["more"] is False)
ok("a null rows list is survived (total 0)", res["nullRows"] == 0)


# ── RAIL 3 — section 5 is NOT bounded ───────────────────────────────────────
print("― rail 3: the Ask-AI inbox is deliberately UNBOUNDED ―")
ok("section 5 has no maxHeight", "maxHeight" not in SEC5,
   "a bound crept into Kevin's fast lane")
ok("section 5 has no internal overflow", "overflowY" not in SEC5)
ok("section 5 does not window its rows",
   "feedWindow" not in SEC5 and "FEED_PAGE_ROWS" not in SEC5)
ok("every open Ask-AI still renders", "inbox.open.map((it) => askRow(it, 'ask-ai'))" in PAGE)
ok("the page SAYS section 5 is unbounded on purpose, so a future reader does not 'fix' it",
   "UNBOUNDED" in SEC5)


# ── RAIL 4 — nothing is dropped from the record ─────────────────────────────
print("― rail 4: render less at a time, never keep less ―")
ok("exhausting the pages reaches every event, in the feed's own order",
   res["seen"] == [f"e{i}" for i in range(435)],
   f"recovered {len(res['seen'] or [])} of 435")
ok("the exhausted window reports nothing hidden and no more",
   res["finalHidden"] == 0 and res["finalMore"] is False and res["finalShown"] == 435)
ok("feedWindow does not mutate its input", res["inputUnchanged"] is True)
ok("the fetch window is NOT shrunk — the record is unchanged (COMMENT_LIMIT 400)",
   "const COMMENT_LIMIT = 400;" in PAGE)
ok("…nor the run window (RUN_LIMIT 400)", "const RUN_LIMIT = 400;" in PAGE)
ok("the full feed is still built from every comment and run",
   "buildFeed(comments, runs)" in PAGE)
for name in ("comments.slice(", "feed.slice(", "feedRows.slice("):
    ok(f"the page never truncates the record itself ({name}…)", name not in PAGE)
ok("the header prints kept vs rendered, so the bound is never mistaken for an empty board",
   "${win.shown} rendered of ${feedRows.length} shown, ${feed.length} kept" in PAGE)
ok("the unpainted remainder is printed at the end of the list",
   "win.hidden} older event" in PAGE and "win.more &&" in PAGE)
ok("…with an explicit control, so lazy-load does not depend on a scroll event firing",
   "onClick={showMore}" in PAGE and "const showMore = useCallback" in PAGE)
ok("the known-gaps footer records the render bound as a KNOWN blind spot",
   "ctrl-F finds only what is painted" in PAGE)


# ── The lazy extend, executed ───────────────────────────────────────────────
print("― lazy extend: nearBottom is a tested predicate, not a browser guess ―")
near = res["near"]
ok("top of a long list is not near the bottom", near[0] is False)
ok("bottomed out IS near the bottom", near[1] is True)
ok(f"within the {SLACK}px slack counts as near the bottom", near[2] is True)
ok("a container that does not overflow is already at its bottom", near[3] is True)
ok("the handler only extends while rows remain unpainted",
   "n * FEED_PAGE_ROWS < feedRows.length ? n + 1 : n" in PAGE)


# ── The payload trim (secondary) ────────────────────────────────────────────
print("― secondary: the wire payload came down without eliding what is rendered ―")
chars = int(re.search(r"const COMMENT_CHARS = (\d+);", PAGE).group(1))
excerpt = int(re.search(r"const FEED_EXCERPT = (\d+);", PAGE).group(1))
ok(f"max_chars is DOWN from the shipped 1400 (now {chars})", chars < 1400)
ok("…but still clears what a feed row actually renders", chars > excerpt,
   f"COMMENT_CHARS={chars} FEED_EXCERPT={excerpt}")
ok("the request still sends it", f"max_chars=${{COMMENT_CHARS}}" in PAGE)
ok("a truncated body still says so rather than eliding silently",
   "excerpt of {e.bodyChars} chars" in PAGE)


# ── Scope: the #220 path-computation incident stays untouched ───────────────
print("― scope: no backend, no schema, no path computation (#220) ―")
endpoints = sorted(set(re.findall(r"apiFetch<[^>]*>\(\s*[`'\"]([^`'\"]+)", PAGE)
                       + re.findall(r"apiFetch\(\s*[`'\"]([^`'\"]+)", PAGE)))
KNOWN = {
    "/api/dev-tasks?include_done=true", "/api/run-history?limit=", "/api/m-session/state",
    "/api/m-session/rules", "/api/m-session/order", "/api/m-session/canvas",
    "/api/m-session/marks", "/api/dev-tasks/comments/recent?limit=",
}
unknown = [e for e in endpoints if not any(e.startswith(k) for k in KNOWN)]
ok("the fix adds no new endpoint — it is entirely render-side", not unknown, str(unknown))
ok("no new migration is introduced by this step",
   not os.path.exists(os.path.join(SRC, "migrations", "m_session_render.sql")))

print()
if FAILED:
    print(f"FAILED {len(FAILED)} of {PASS + len(FAILED)}:")
    for f in FAILED:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASS — {PASS} assertions (board #240: the activity log is bounded to render)")
