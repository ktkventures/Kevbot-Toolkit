"""Acceptance test — board #252: the top-level `Team` sidebar section.

Kevin, 07-31: *"I want to create a new kind of divider page in the side nav in
between admin and settings, and call this new one Team. I want to move the
agents, dispatch, and our session dashboards to that tab. Also, I don't
currently see navigation to the R session."*

The last sentence is the bug this file exists for. Board #251 shipped
`AdminRSessionPage`, its route and its sign-of-life, and never added a nav
entry — so the page was live and reachable ONLY by typing the URL. Nothing in
the suite noticed, because nothing asserted that a shipped page is reachable.
That is the regression these rails pin.

What is asserted, and why each one can rot silently:

  • REACHABILITY (rail 1) — `R Session` → `/admin/r-session` is in the nav.
    The #251 miss, verbatim. Generalised: every `/admin/<x>-session` page that
    has a route must have a nav entry, so the NEXT session dashboard cannot
    ship orphaned the same way.
  • EXACTLY ONCE (rail 2) — each moved entry appears in exactly one section.
    #252 is a MOVE, not a copy: two nav paths to one page is how a sidebar
    starts lying about where a thing lives. A half-applied move (added to
    `Team`, left in `Admin`) is the likely failure and looks fine by eye.
  • ORDER (rail 3) — `Team` sits strictly between `Admin` and `Settings`; the
    four children #252 moved are all still there in Kevin's stated order, and
    `Tasks` (added 08-01) leads the group.
  • ROUTES DID NOT MOVE (rail 4) — the pages stay at `/admin/*`.
    `docs/_active/R_Session_Opener.md` hands a LIVE R session the literal
    string `/admin/r-session`, and Kevin has these bookmarked. A later
    "tidy-up" that relocates the routes to `/team/*` breaks instructions
    already in flight; this rail makes that a deliberate, loud change (update
    the opener + add redirects) instead of a silent one.
  • WHERE THE BOARD LIVES (rail 5) — `Tasks` is under `Team`; `Roadmap` is
    under `Admin`. This one CHANGED on 08-01. Kevin, 08-01: *"Can we move over
    the tasks page into the team that is nested under team instead of admin?"*
    — so `Tasks` moved, and `Roadmap` did not, because he named `Tasks` only.
    The rail is BIDIRECTIONAL on purpose: present in `Team` AND absent from
    `Admin`. A presence-only check passes on a duplicate, which is exactly the
    half-applied move rail 2 exists to catch.

The sidebar is parsed structurally (sections and their children), not
string-matched, so these hold under reformatting and comment edits.

Static-source assertions: offline, hermetic, runs anywhere.

Run:  .venv/bin/python src/test_team_nav_section_252.py   (from RoR_Trader root)
"""
import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
FE = os.path.join(ROOT, "frontend", "src")

PASS = 0


def ok(label, cond, detail=""):
    global PASS
    if not cond:
        print(f"FAIL: {label} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  ok: {label}")


def read(*parts):
    p = os.path.join(*parts)
    ok(f"exists: {os.path.relpath(p, ROOT)}", os.path.exists(p))
    with open(p, encoding="utf-8") as f:
        return f.read()


SIDEBAR = read(FE, "components", "Sidebar.tsx")
OPENER = read(ROOT, "docs", "_active", "R_Session_Opener.md")


# ── parse navItems into [(label, parent_href, [(child_href, child_label), …])] ──
def parse_nav(src):
    m = re.search(r"const navItems: NavItem\[\] = \[(.*?)\n\];", src, re.S)
    ok("Sidebar.tsx declares navItems", bool(m))
    body = m.group(1)
    # Comments are prose; strip them so they cannot be mistaken for entries.
    body = re.sub(r"^\s*//.*$", "", body, flags=re.M)

    sections = []
    # A section header is the object line carrying `label:` + `icon:`.
    heads = list(re.finditer(
        r"href:\s*'([^']+)',\s*label:\s*'([^']+)',\s*icon:\s*'([^']*)'", body))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        chunk = body[h.end():end]
        kids = []
        cm = re.search(r"children:\s*\[(.*?)\n\s*\],", chunk, re.S)
        if cm:
            kids = re.findall(r"\{\s*href:\s*'([^']+)',\s*label:\s*'([^']+)'\s*\}",
                              cm.group(1))
        sections.append((h.group(2), h.group(1), kids))
    return sections


NAV = parse_nav(SIDEBAR)
LABELS = [s[0] for s in NAV]
ok("parsed a plausible nav (>= 8 sections)", len(NAV) >= 8, LABELS)

BY_LABEL = {s[0]: s for s in NAV}
MOVED = [
    ("Agents", "/admin/agents"),
    ("Dispatch", "/admin/dispatch"),
    ("M Session", "/admin/m-session"),
    ("R Session", "/admin/r-session"),
]
# Moved into `Team` later, by the 08-01 audible. Kept separate from MOVED
# because rail 3 pins the ORIGINAL four in their original relative order;
# lumping this in would silently rewrite that assertion.
TASKS = ("Tasks", "/admin/tasks")

print("― rail 1: the #251 regression — R Session is REACHABLE from the nav ―")

all_children = [(lbl, href) for _, _, kids in NAV for href, lbl in kids]
child_hrefs = [h for _, h in all_children]
child_labels = [l for l, _ in all_children]

ok("nav links to /admin/r-session", "/admin/r-session" in child_hrefs,
   "the #251 miss: page shipped, nav entry did not")
ok("it is labelled 'R Session'",
   ("R Session", "/admin/r-session") in all_children)

# Generalised so the NEXT session dashboard cannot ship orphaned the same way.
routed = sorted(
    d for d in os.listdir(os.path.join(FE, "app", "admin"))
    if d.endswith("-session")
    and os.path.exists(os.path.join(FE, "app", "admin", d, "page.tsx")))
ok("found the session-dashboard routes on disk", len(routed) >= 2, routed)
for d in routed:
    ok(f"routed page /admin/{d} has a nav entry", f"/admin/{d}" in child_hrefs,
       "a page with a route and no nav entry is reachable only by typing the URL")

print("― rail 2: this was a MOVE — each entry appears exactly ONCE ―")

for label, href in MOVED + [TASKS]:
    ok(f"'{label}' appears exactly once in the whole sidebar",
       child_labels.count(label) == 1, f"count={child_labels.count(label)}")
    ok(f"{href} appears exactly once in the whole sidebar",
       child_hrefs.count(href) == 1, f"count={child_hrefs.count(href)}")
    owner = [sec for sec, _, kids in NAV if any(h == href for h, _ in kids)]
    ok(f"{href} lives under 'Team' (not 'Admin')", owner == ["Team"], owner)

ok("no duplicate child href ANYWHERE in the sidebar",
   len(child_hrefs) == len(set(child_hrefs)),
   sorted({h for h in child_hrefs if child_hrefs.count(h) > 1}))

print("― rail 3: Team sits between Admin and Settings, children in order ―")

for needed in ("Admin", "Team", "Settings"):
    ok(f"section '{needed}' exists", needed in BY_LABEL, LABELS)
ok("order is Admin → Team → Settings",
   LABELS.index("Admin") < LABELS.index("Team") < LABELS.index("Settings"),
   LABELS)
ok("Team is IMMEDIATELY between them (no section wedged in)",
   LABELS.index("Team") == LABELS.index("Admin") + 1
   and LABELS.index("Settings") == LABELS.index("Team") + 1, LABELS)

team_kids = [(l, h) for h, l in BY_LABEL["Team"][2]]
# Board #257 added `Your Dashboard` to this section, so the rail is no longer
# EQUALITY — it is CONTAINMENT AND ORDER: the four pages #252 moved must all
# still be here, still in this order, and none may be dropped or re-parented by
# a later addition. That is what #252 was actually protecting; an equality check
# would simply forbid the section from ever growing, which is not the rule.
ok("the four pages #252 moved are all still under Team, in their original order",
   [k for k in team_kids if k in MOVED] == MOVED, team_kids)
ok("...and none of them was dropped when the section grew",
   all(k in team_kids for k in MOVED), team_kids)
# 08-01: the board leads the group — it is the surface Kevin opens most, and
# this section is intent-ordered rather than alphabetical, so "first" is a
# decision worth pinning, not an accident of where the line was pasted.
ok("'Tasks' is FIRST under Team", team_kids[0] == TASKS, team_kids)
ok("Team is a real top-level section with an icon", bool(BY_LABEL["Team"][1]) and
   re.search(r"label: 'Team', icon: '\S+'", SIDEBAR) is not None)

print("― rail 4: NAVIGATION-only — the routes did NOT move ―")

# The opener is handed to a live R session; it cites the path literally.
ok("R_Session_Opener.md still cites /admin/r-session", "/admin/r-session" in OPENER)
for label, href in MOVED + [TASKS]:
    seg = href.split("/")[-1]
    ok(f"the {label} page still routes at {href}",
       os.path.exists(os.path.join(FE, "app", "admin", seg, "page.tsx")))
    ok(f"no /team/{seg} route shadows it",
       not os.path.exists(os.path.join(FE, "app", "team", seg, "page.tsx")),
       "if the routes ever DO move, update the opener + add redirects in the same PR")
ok("every Team child still points at /admin/*",
   all(h.startswith("/admin/") for h, _ in BY_LABEL["Team"][2]))

print("― rail 5: Tasks moved to Team (08-01); Roadmap stayed under Admin ―")

admin_kids = {l: h for h, l in BY_LABEL["Admin"][2]}
team_labels = {l for _, l in BY_LABEL["Team"][2]}
team_kids_map = {l: h for h, l in BY_LABEL["Team"][2]}

# BOTH directions. Presence alone would pass on a duplicate — leaving `Tasks`
# in `Admin` as well is the exact half-applied move this file was written for.
ok("'Tasks' is under Team", team_kids_map.get("Tasks") == "/admin/tasks",
   team_kids_map)
ok("'Tasks' is NO LONGER under Admin", "Tasks" not in admin_kids,
   "the 08-01 move must be a MOVE — a copy leaves two nav paths to one page")
ok("nothing under Admin points at /admin/tasks anymore",
   "/admin/tasks" not in set(admin_kids.values()), admin_kids)

# Roadmap did NOT move: Kevin named `Tasks` only, and moving both is a guess.
ok("'Roadmap' is still under Admin", admin_kids.get("Roadmap") == "/admin/roadmap",
   admin_kids)
ok("'Roadmap' was not also copied into Team", "Roadmap" not in team_labels)

ok("Admin kept its platform entries (Overview/Users/Strategy Health)",
   {"Platform Overview", "Users", "Strategy Health"} <= set(admin_kids))

print("― rail 6: exactly ONE section highlights per route ―")

# `Team`'s parent href is `/admin/agents` (it was the first child when the
# section was created; the header is a toggle button, not a link, so the href is
# used only for highlighting) — which means `Admin` and `Team` both sit on the
# `/admin` prefix, and
# `isActive()` resolves that overlap by a "is some child a better match?" scan.
# Get it wrong and BOTH sections light up on /admin/r-session, or `Admin` stays
# lit for pages that no longer live under it. Neither is a type error and
# neither shows up in the entry lists above.
#
# MIRROR, not the implementation: this reproduces Sidebar.tsx's isActive(). The
# drift check below pins the two clauses it models, so a rewrite of the real one
# fails here rather than silently leaving this rail testing a fossil.
ok("isActive still resolves overlap by exact match + better-child scan",
   "if (pathname === href) return true;" in SIDEBAR
   and "hasBetterMatch" in SIDEBAR
   and re.search(r"pathname\?\.startsWith\(href \+ '/'\)", SIDEBAR) is not None)

ALL_CHILD_HREFS = child_hrefs


def is_active(href, path):
    if path == href:
        return True
    if href != "/" and path.startswith(href + "/"):
        return not any(h != href and path.startswith(h) and h.startswith(href)
                       for h in ALL_CHILD_HREFS)
    return False


def lit(path):
    return [lbl for lbl, phref, kids in NAV
            if is_active(phref, path) or any(is_active(h, path) for h, _ in kids)]


for path, expect in (
    ("/admin", "Admin"),                    # Platform Overview stays Admin's
    ("/admin/users", "Admin"),
    # 08-01: the board moved sections, so the highlight must move with it —
    # `Admin` must NOT stay lit on /admin/tasks now that it no longer owns it.
    ("/admin/tasks", "Team"),
    ("/admin/roadmap", "Admin"),
    ("/admin/agents", "Team"),
    ("/admin/dispatch", "Team"),
    ("/admin/m-session", "Team"),
    ("/admin/r-session", "Team"),
    ("/settings/themes", "Settings"),
):
    hot = lit(path)
    ok(f"{path} lights exactly one section, and it is '{expect}'",
       hot == [expect], hot)

print(f"\nPASS — {PASS} assertions (board #252 Team nav section)")
