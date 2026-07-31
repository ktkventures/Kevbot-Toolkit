"""Acceptance test — board #278: one long token must not squeeze the task detail view.

THE BUG, AND WHY A TEST HAD TO BE A MEASUREMENT
-----------------------------------------------
Kevin hit this on #163 (and 2-3 times before): the detail modal renders
unreadably — the ACTIVITY column widens and squeezes the left
`CONTEXT · WHAT TO DO` panel to a word or two per line, exactly when he needs to
READ a task in order to authorise it.

The cause is `min-width: auto` on a flex item. A flex child never shrinks below
its MIN-CONTENT width, so one unbreakable token in a comment sets a floor under
the Activity column and the sibling panel absorbs 100% of the shrinking.

Every part of that is invisible to a source read, which is why this suite MEASURES
it in a real Chromium instead of asserting on strings alone. Rails 1-6 build a
fixture out of THE ACTUAL VALUES PARSED FROM THE SOURCE (`MD_CSS` verbatim, the
two columns' `minWidth`) and read `getBoundingClientRect().width` back out of the
browser. Revert any one of the fixes in the .tsx and the corresponding rail goes
red, because the fixture is downstream of the file.

MEASURED ON THIS MACHINE, 2026-07-31 (1168px row — the modal's real content
width at `maxWidth: 1200` minus its 16px padding; undisturbed left column = 771px):

    activity column               left panel   verdict
    ---------------------------   ----------   -------------------------------
    60-char token, dev's CSS         551px     starved (this is the bug)
    ...with `min-width: 0`           771px     fixed
    ...with `overflow-wrap:anywhere` 771px     fixed
    fenced code block (<pre>)        212px     UNREADABLE — the worst case
    markdown table                     0px     total collapse
    <pre>/table + both fixes         771px     fixed

Two things that measurement corrected about the report's own diagnosis, and they
matter for whoever reads this next:

  * A long PATH is not actually the worst offender. Chromium takes break
    opportunities after `/` and `-`, so `/home/kevin/projects/Kevbot-wt-…-163`
    wraps on its own. What has NO break opportunity is an underscore-joined
    constant — `ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED` — and a fenced code block,
    whose `white-space: pre` makes the whole line one unbreakable run. So the
    fixture uses a 60-char underscore token, per the step SOP, plus the <pre> and
    <table> cases that measured worse.
  * `overflow-x: auto` on the `<pre>` DOES NOT save the layout on its own
    (212px, row 4 above, unchanged from no overflow at all). It is necessary —
    it is what makes the block scroll in its own box — but only `min-width: 0`
    on the flex child stops the starvation. Shipping the `<pre>` rule alone
    would have looked like a fix and fixed nothing.

WHAT IS DELIBERATELY NOT THE FIX
--------------------------------
Shortening what agents write. M hand-shortened #163's tokens twice (59 → 36 → 27
chars) and it was still unusable — that IS the experiment, and it failed. Agents
write full paths, branch names and flag constants because those are the checkable
references that make a report verifiable rather than plausible;
`ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED` is 36 characters and has no shorter correct
spelling. Rail 8 pins that the fix lives in the CSS.

RAILS
-----
  1. the ACTIVITY column can shrink — a 60-char unbreakable token in a comment
     leaves the left panel EXACTLY as wide as an empty thread does.
  2. a fenced code block does not starve it either, and scrolls in its own box.
  3. a wide markdown table does not collapse it (measured 0px before).
  4. the left panel keeps a readable floor EVEN IF a sibling regains a floor —
     the belt to rail 1's braces.
  5. `overflow-wrap: anywhere`, not `break-word`: only `anywhere` may reduce
     min-content, and the two are otherwise indistinguishable on screen.
  6. system thread lines — which are NOT markdown, and are the ones carrying
     dispatch paths — wrap too.
  7. the same pattern is fixed on the M-session, R-session and Kevin dashboards.
  8. the fix is CSS, not an authoring rule.

Run:  .venv/bin/python src/test_task_detail_layout_278.py
      (from the RoR_Trader root)

Needs a Chromium binary for rails 1-6. It looks for $CHROME_BIN, then the usual
PATH names, then the Playwright browser cache. It REFUSES rather than skips if it
cannot find one: a layout rail that silently does not run is how this regresses.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE = os.path.join(ROOT, "frontend", "src", "views")

PASS = 0
FAILED = []


def ok(label, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok: {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL: {label}" + (f"  [{detail}]" if detail else ""))


def read(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


MODAL = read(os.path.join(FE, "TaskDetailModal.tsx"))
MDSRC = read(os.path.join(FE, "taskMarkdown.tsx"))
MSESS = read(os.path.join(FE, "AdminMSessionPage.tsx"))
RSESS = read(os.path.join(FE, "AdminRSessionPage.tsx"))
KEVIN = read(os.path.join(FE, "AdminKevinPage.tsx"))


# ── source extraction ────────────────────────────────────────────────────────
# The fixture is built from these, so a revert in the .tsx shows up as a red
# MEASUREMENT rather than as a missing string.

def md_css():
    """MD_CSS's contents, verbatim."""
    m = re.search(r"export const MD_CSS = `(.*?)`;", MDSRC, re.S)
    if not m:
        raise SystemExit("could not find MD_CSS in taskMarkdown.tsx")
    return m.group(1)


def context_min_width():
    """The left column's declared min-width, resolved through the const."""
    body = re.search(
        r"\{/\* ── Body:.*?<div style=\{\{\s*\n?\s*flex: 2,(.*?)\}\}>", MODAL, re.S)
    if not body:
        raise SystemExit("could not find the modal's left column in TaskDetailModal.tsx")
    decl = re.search(r"minWidth:\s*([A-Za-z_][A-Za-z_0-9]*|0|'[^']*')", body.group(1))
    if not decl:
        return "auto"
    val = decl.group(1)
    if val.startswith("'"):
        return val.strip("'")
    if val == "0":
        return "0"
    const = re.search(r"const %s = '?([^';]*)'?;" % re.escape(val), MODAL)
    if not const:
        raise SystemExit(f"left column min-width refers to {val}, which is not a const")
    return const.group(1).strip()


def activity_min_width():
    """The ACTIVITY column's effective min-width (its own, else `panel`'s)."""
    jsx = re.search(r"\{/\* Activity panel.*?<div style=\{\{(.*?)\}\}>", MODAL, re.S)
    if not jsx:
        raise SystemExit("could not find the Activity panel in TaskDetailModal.tsx")
    own = re.search(r"minWidth:\s*(0|'[^']*')", jsx.group(1))
    if own:
        return own.group(1).strip("'")
    if "...panel" in jsx.group(1):
        pan = re.search(r"const panel: React\.CSSProperties = \{(.*?)\};", MODAL, re.S)
        if pan:
            inherited = re.search(r"minWidth:\s*(0|'[^']*')", pan.group(1))
            if inherited:
                return inherited.group(1).strip("'")
    return "auto"


def system_line_style():
    """The style on the system-comment body span (plain text, not markdown)."""
    m = re.search(r"<RoleChip role=\"system\".*?<span style=\{\{(.*?)\}\}>\{cm\.body\}",
                  MODAL, re.S)
    return m.group(1) if m else ""


MD = md_css()
CTX_MIN = context_min_width()
ACT_MIN = activity_min_width()

# 60 characters, no break opportunity anywhere in it: underscores and digits are
# all class AL/NU to the line breaker, unlike `/` and `-`.
TOKEN = "RORT_ALGO_HISTORY_LIVE_DUAL_WRITE_ENABLED_SHADOW_CANARY_267X"
assert len(TOKEN) == 60, len(TOKEN)

ROW_W = 1168   # modal content width: maxWidth 1200 − 2×16 padding
GAP = 12       # the body row's `gap: 12`


# ── the browser ──────────────────────────────────────────────────────────────

def find_chromium():
    if os.environ.get("CHROME_BIN") and os.path.exists(os.environ["CHROME_BIN"]):
        return os.environ["CHROME_BIN"]
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome",
                 "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    cache = os.path.expanduser("~/.cache/ms-playwright")
    pats = ("*/chrome-linux/chrome", "*/chrome-linux64/chrome",
            "*/chrome-headless-shell-linux64/chrome-headless-shell")
    hits = sorted(h for p in pats for h in glob.glob(os.path.join(cache, p)))
    return hits[-1] if hits else None


def measure(rows):
    """Lay `rows` out in a real browser; return each row's left-panel width in px.

    Each row is (left_css, right_css, right_inner_html). Chromium renders the
    file, the page writes its own measurements into the DOM, and `--dump-dom`
    hands the serialised result back — no driver library, no network.
    """
    chrome = find_chromium()
    if not chrome:
        raise SystemExit(
            "REFUSING to report a layout rail without measuring it: no Chromium found.\n"
            "  Set $CHROME_BIN, or install one:  npx --yes playwright install chromium")
    body = []
    for left_css, right_css, inner in rows:
        body.append(
            f'<div class="row" style="display:flex;gap:{GAP}px;width:{ROW_W}px">'
            f'<div class="L" style="flex:2;{left_css}">context</div>'
            f'<div class="R" style="flex:1;{right_css}">{inner}</div></div>')
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "body{margin:0;font:13px/1.5 sans-serif}" + MD + "</style></head><body>"
        + "".join(body) +
        '<pre id="out">PENDING</pre><script>'
        "var o=[];document.querySelectorAll('.row').forEach(function(r){"
        "o.push(Math.round(r.querySelector('.L').getBoundingClientRect().width));});"
        "document.getElementById('out').textContent='MEASURED:'+o.join(',');"
        "</script></body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "fixture.html")
        with open(page, "w", encoding="utf-8") as fh:
            fh.write(html)
        proc = subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             f"--user-data-dir={os.path.join(tmp, 'profile')}",
             "--virtual-time-budget=2000", "--dump-dom", "file://" + page],
            capture_output=True, text=True, timeout=120)
    m = re.search(r"MEASURED:([0-9,]+)", proc.stdout)
    if not m:
        raise SystemExit("browser produced no measurement:\n" + proc.stderr[-2000:])
    return [int(x) for x in m.group(1).split(",")]


LEFT = f"min-width:{CTX_MIN}"
RIGHT = f"min-width:{ACT_MIN}"
MDWRAP = '<div class="task-md">%s</div>'
LONG_TABLE = ("<table><thead><tr><th>token</th><th>chars</th></tr></thead>"
              f"<tbody><tr><td>{TOKEN}</td><td>{TOKEN}</td></tr></tbody></table>")

print("── board #278 · task detail view: long tokens must not squeeze the panel ──")
print(f"context column min-width  : {CTX_MIN}")
print(f"activity column min-width : {ACT_MIN}")
print(f"probe token               : {len(TOKEN)} chars, no break opportunity")

widths = measure([
    # 0 · control — an empty thread. This is what "not squeezed" looks like.
    (LEFT, RIGHT, MDWRAP % "<p>No activity yet.</p>"),
    # 1 · the reported bug: one 60-char unbreakable token in a comment.
    (LEFT, RIGHT, MDWRAP % f"<p>armed <code>{TOKEN}</code> on the data-worker</p>"),
    # 2 · a fenced code block — measured worse than any inline token.
    (LEFT, RIGHT, MDWRAP % f"<pre><code>{TOKEN} {TOKEN}</code></pre>"),
    # 3 · a markdown table, like the one in this very task's description.
    (LEFT, RIGHT, MDWRAP % LONG_TABLE),
    # 4 · belt-and-braces: a sibling that has REGAINED a min-content floor
    #     (i.e. someone reverted rail 1). The left panel must still be readable.
    (LEFT, "min-width:auto", MDWRAP % f"<pre><code>{TOKEN} {TOKEN}</code></pre>"),
    # 5 · dev's configuration, reproduced verbatim: no min-width on the
    #     Activity column, `overflow-wrap: break-word` prose, and a <pre> with
    #     only `overflow-x: auto`. This is what Kevin was looking at. It is here
    #     so every "identical to control" above is anchored to a control that IS
    #     the bug, rather than to a fixture too weak to fail.
    ("min-width:0", "min-width:auto",
     '<div style="overflow-wrap:break-word">'
     f'<p>armed <code>{TOKEN}</code></p>'
     f'<pre style="overflow-x:auto"><code style="white-space:pre">{TOKEN} {TOKEN}</code></pre>'
     '</div>'),
    # 6 · a system thread line: plain text, never touched by MD_CSS.
    (LEFT, RIGHT,
     '<div style="display:flex;gap:6px;align-items:baseline">'
     '<span>sys</span>'
     f'<span style="min-width:0;overflow-wrap:anywhere">dispatch skipped: {TOKEN}</span></div>'),
    # 7 · isolates the PROSE rule. The sibling has no min-width here, so
    #     `overflow-wrap` is the only thing holding the layout up — which is
    #     what makes rail 5 a measurement and not just a grep. Bare prose, NOT
    #     inside <code>: the `.task-md code` rule would otherwise cover for the
    #     `.task-md` one and the isolation would be fictional.
    (LEFT, "min-width:auto", MDWRAP % f"<p>armed {TOKEN} on the data-worker</p>"),
])
control = widths[0]
print(f"measured left-panel widths: {widths}  (control = {control}px)")

print("\n── RAIL 1 · a 60-char unbreakable token changes nothing ─────────────")
ok("the control row gives the left panel its full 2/3 of the row",
   control >= ROW_W * 0.6, f"{control}px of {ROW_W}px")
ok("a comment containing a 60-char unbreakable token leaves it IDENTICAL",
   widths[1] == control, f"{widths[1]}px vs control {control}px")
ok("...and that is not vacuous: dev's exact CSS, same content, starves it",
   widths[5] < control - 200, f"unfixed {widths[5]}px vs control {control}px")

print("\n── RAIL 2 · a fenced code block does not starve it either ───────────")
ok("a <pre> code block leaves the left panel at its control width",
   widths[2] == control, f"{widths[2]}px vs control {control}px")
ok("MD_CSS makes <pre> scroll INSIDE ITS OWN BOX rather than push layout",
   "overflow-x: auto" in MD and "max-width: 100%" in MD.split(".task-md pre {")[1].split("}")[0],
   "`.task-md pre` needs both overflow-x:auto and max-width:100%")

print("\n── RAIL 3 · a wide markdown table does not collapse it ──────────────")
ok("a two-column table of long tokens leaves the left panel at control width",
   widths[3] == control, f"{widths[3]}px vs control {control}px")
ok("...because the table is its own scroll container, not a width demand",
   all(t in MD.split(".task-md table {")[1].split("}")[0]
       for t in ("overflow-x: auto", "max-width: 100%")))

print("\n── RAIL 4 · the left panel keeps a readable floor regardless ────────")
ok("the context column declares a floor, not just `min-width: 0`",
   CTX_MIN not in ("0", "auto"), CTX_MIN)
ok("...and it HOLDS even against a sibling that regained a min-content floor",
   widths[4] >= 340, f"{widths[4]}px, floor should be 340px at a {ROW_W}px row")
ok("...which is a real difference: the identical row without the floor collapses",
   widths[5] < widths[4], f"floored {widths[4]}px vs unfloored {widths[5]}px")
ok("the floor is expressed so it cannot become its own overflow bug on a phone",
   "min(" in CTX_MIN, CTX_MIN)

print("\n── RAIL 5 · `anywhere`, not `break-word` ───────────────────────────")
ok("prose wraps with overflow-wrap: anywhere", "overflow-wrap: anywhere" in MD)
ok("measured: with the prose rule doing the work ALONE — a sibling with no "
   "min-width — the panel still holds its full width",
   widths[7] == control, f"{widths[7]}px vs control {control}px")
ok("the old `break-word` spelling is gone from MD_CSS — it does NOT reduce "
   "min-content, so it looks fixed and is not",
   "break-word" not in MD, MD)
ok("inline <code> — where flag constants land — wraps too",
   "overflow-wrap: anywhere" in MD.split(".task-md code {")[1].split("}")[0])
ok("...but code INSIDE a <pre> does not wrap: it scrolls, so it stays copyable",
   "white-space: pre" in MD.split(".task-md pre code {")[1].split("}")[0])
ok("the M-session feed no longer uses the deprecated word-break spelling",
   "wordBreak" not in MSESS, "AdminMSessionPage.tsx")

print("\n── RAIL 6 · system lines wrap (they are not markdown) ───────────────")
sysline = system_line_style()
ok("the system-comment body carries its own wrap rule",
   "overflowWrap: 'anywhere'" in sysline, sysline)
ok("...and its own minWidth: 0, since it is a flex child", "minWidth: 0" in sysline, sysline)
ok("measured: a 60-char token in a system line leaves the panel at control width",
   widths[6] == control, f"{widths[6]}px vs control {control}px")

print("\n── RAIL 7 · the same pattern on the sibling dashboards ──────────────")
for name, src in (("M-session", MSESS), ("R-session", RSESS), ("Kevin", KEVIN)):
    head = re.search(r"const Panel = .*?<div style=\{\{ display: 'flex'.*?\n\s*"
                     r"(?:\{/\*.*?\*/\}\s*\n\s*)?<div style=\{\{([^}]*)\}\}>", src, re.S)
    ok(f"{name}: the Panel header's prose child carries minWidth: 0",
       bool(head) and "minWidth: 0" in head.group(1),
       head.group(1) if head else "header not found")
ok("Kevin's plate rows already scoped their prose child (no regression here)",
   "flex: 1, minWidth: 0" in KEVIN)
ok("the M-session ask rows likewise", "flex: 1, minWidth: 0" in MSESS)
ok("the Kevin dashboard reuses TaskDetailModal, so it inherits the fix (#257 rail 20)",
   "TaskDetailModal" in KEVIN)
ok("M-session renders agent prose through the shared Md/MD_CSS, so it inherits it too",
   "MD_CSS" in MSESS and "from './taskMarkdown'" in MSESS)

print("\n── RAIL 8 · the fix is CSS, not an authoring rule ───────────────────")
ok("the modal still renders comment bodies verbatim through <Md>, unshortened",
   "<Md text={cm.body} />" in MODAL)
ok("nothing truncates or elides a comment body on its way to the panel",
   not re.search(r"cm\.body\.slice\(", MODAL))
ok("the reason is written down where the next person will change it",
   "board #278" in MDSRC.lower() or "#278" in MDSRC)

print("\n" + "=" * 72)
print(f"{PASS} passed, {len(FAILED)} failed")
if FAILED:
    for f in FAILED:
        print(f"  x {f}")
    sys.exit(1)
print("ALL RAILS GREEN")
