"""R-Session Dashboard API — /admin/r-session (board #251 step 3).

WHY THIS EXISTS
---------------
On 07-30, #246 was approved, built, tested and PR'd inside an hour — and then
sat in `Staged` for four hours because nobody created the train that ships it.
Nothing was broken. Nobody was watching. The R session is the watcher; this
router serves the three facts the app can actually establish for it, and prints
the ones it cannot rather than faking them.

WHAT IS SERVED HERE, AND WHY IT IS *ONLY* THIS
----------------------------------------------
The lead module — **CARS WAITING FOR A TRAIN** — is deliberately NOT here. It is
a pure derivation over `dev_tasks` + `run_history`, both of which the page
already fetches, and it must reuse the shipping-lane resolver (#245) rather than
grow a second definition of "has unmerged code". Two definitions drift; this one
already has a test. So it lives beside that resolver in the frontend
(`views/rSessionSignals.ts` → `carsWaitingForATrain`) and this router serves the
three things the browser genuinely cannot get:

  1. `/heartbeats`  — SIGN OF LIFE. Which live sessions are actually running.
  2. `/deploy-log`  — the deploy log AS THE SERVING CONTAINER SEES IT, which is
                      what makes "is the Wave-N log PR still open?" answerable.
  3. `/service`     — this API's own `/health` answer + the commit it is
                      serving, plus a NAMED list of what it cannot see.

THE MIGRATION QUESTION, ANSWERED: THERE ISN'T ONE
-------------------------------------------------
Sign-of-life needs no schema. `system_settings` is already a key/value table
(`key · value JSONB · description · updated_at · updated_by`), already read and
written by the dispatcher's daily cap. Each live session writes
`session_heartbeat_<LETTER>`. `updated_by` is a `UUID REFERENCES auth.users(id)`
— so the ACTOR is carried inside the JSONB value, never in that column; writing
a letter there would violate the FK and 500 the heartbeat.

FAIL LOUD, NEVER SILENTLY EMPTY (the #157 dead-alarm lesson)
------------------------------------------------------------
`#157` was a "0 healthy" alarm that read a field nothing had written since
07-21: an absent signal rendered as a confident zero. Every read below
distinguishes three answers — **fresh**, **stale**, and *"nothing has ever
written this key"* — and the last one is reported as `never`, with its own
reason string, so a dashboard can say NOT RUNNING because it knows, not because
it found nothing.
"""
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import get_current_user

router = APIRouter(prefix="/api/r-session", tags=["r-session"])

log = logging.getLogger(__name__)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _now() -> str:
    return _now_dt().isoformat()


# ── Sign of life ────────────────────────────────────────────────────────────

HEARTBEAT_PREFIX = "session_heartbeat_"

# The live sessions whose absence is worth rendering. A session with no key at
# all is NOT omitted — it is rendered `never`, which is the entire point: the
# 07-30 stall was an R session that did not exist, and a page that only lists
# sessions it can find would have shown an empty, reassuring panel.
TRACKED_SESSIONS = ("M", "R")

# Thresholds, in seconds. Returned in the payload so the UI renders THESE
# numbers rather than restating them — a second copy in the .tsx is a second
# source of truth for "is R alive", and it would drift silently.
#
# 10m/30m is set against the cadence the opener asks for (a check per cycle,
# each cycle minutes not hours): one skipped beat is `stale` (amber — probably
# mid-task), two-plus is `down` (red — treat as NOT RUNNING).
FRESH_MAX_S = 10 * 60
STALE_MAX_S = 30 * 60

# `session_heartbeat_R`, `session_heartbeat_R-A`. Anchored and bounded: the key
# is built by string concatenation, so an unvalidated letter is a write to an
# arbitrary settings key.
_LETTER_RE = re.compile(r"^[A-Z](?:[0-9]|-[A-Z])?$")


def heartbeat_key(letter: str) -> str:
    return f"{HEARTBEAT_PREFIX}{letter}"


def parse_letter(raw) -> str:
    """Validate a session letter, or 400. Never a silent default."""
    letter = str(raw or "").strip().upper()
    if not _LETTER_RE.match(letter):
        raise HTTPException(400, (
            f"letter {raw!r} is not a session letter — expected e.g. 'R', 'M', "
            "'E2' or 'R-A'. The letter becomes a `system_settings` key, so it "
            "is validated rather than trusted."))
    return letter


def _age_s(iso, now: datetime):
    """Seconds since `iso`, or None if it is unreadable. A future stamp comes
    back NEGATIVE and is preserved: clock skew is a real condition and a
    dashboard that clamps it to 0 reports a healthy session that isn't."""
    if not iso:
        return None
    try:
        s = str(iso).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds()


def heartbeat_state(row, now: datetime) -> dict:
    """One session's sign of life: (state, age, and WHICH stamp answered).

    `row` is the `system_settings` row (or None when the key has never been
    written). Four states, and the fourth is the one that matters:

      fresh   — beat within FRESH_MAX_S
      stale   — beat within STALE_MAX_S (amber: probably mid-task)
      down    — beat older than STALE_MAX_S → the UI says NOT RUNNING
      never   — the key has NEVER been written. NOT the same as `down`, and
                conflating them is how #157 happened: "no session has ever
                run" and "the session stopped" want different reactions.

    The freshness stamp is `value.at` (what the session itself wrote), falling
    back to the row's `updated_at`. WHICH ONE WAS USED IS RETURNED (`at_source`)
    — an unlabelled proxy is how a dashboard starts lying.
    """
    if not row:
        return {"state": "never", "at": None, "at_source": None, "age_s": None,
                "actor": None, "note": None, "updated_at": None,
                "why": "no `system_settings` row for this key — this session "
                       "has never written a heartbeat"}
    val = row.get("value")
    if not isinstance(val, dict):
        val = {}
    at = val.get("at") or None
    at_source = "value.at" if at else None
    if not at:
        at = row.get("updated_at") or None
        at_source = "updated_at" if at else None
    age = _age_s(at, now)
    if age is None:
        state, why = "never", (
            "the row exists but carries no readable timestamp in `value.at` or "
            "`updated_at` — treat as no signal, never as fresh")
    elif age <= FRESH_MAX_S:
        state, why = "fresh", f"beat {int(age)}s ago (fresh ≤ {FRESH_MAX_S}s)"
    elif age <= STALE_MAX_S:
        state, why = "stale", (
            f"last beat {int(age)}s ago — past the {FRESH_MAX_S}s cadence but "
            f"within {STALE_MAX_S}s; probably mid-task")
    else:
        state, why = "down", (
            f"last beat {int(age)}s ago, over the {STALE_MAX_S}s limit — "
            "treat as NOT RUNNING")
    return {"state": state, "at": at, "at_source": at_source,
            "age_s": age, "actor": val.get("actor"), "note": val.get("note"),
            "updated_at": row.get("updated_at"), "why": why}


@router.get("/heartbeats")
def heartbeats(user=Depends(get_current_user)):
    """Sign of life for every tracked session, plus any other beat on record.

    One list read, not one GET per session: three round trips for one strip is
    the poll-efficiency mistake board #148 measured.
    """
    from db import list_system_settings
    now = _now_dt()
    try:
        rows = {r["key"]: r for r in (list_system_settings() or [])
                if str(r.get("key", "")).startswith(HEARTBEAT_PREFIX)}
        readable, read_error = True, None
    except Exception as e:  # noqa: BLE001
        rows, readable, read_error = {}, False, str(e)[:300]
    out = []
    seen = set()
    for letter in TRACKED_SESSIONS:
        key = heartbeat_key(letter)
        seen.add(key)
        st = heartbeat_state(rows.get(key), now)
        out.append({"letter": letter, "key": key, "tracked": True, **st})
    # Beats from sessions nobody declared — shown, never dropped. A heartbeat
    # written by a session the page does not track is information, not noise.
    for key, row in sorted(rows.items()):
        if key in seen:
            continue
        st = heartbeat_state(row, now)
        out.append({"letter": key[len(HEARTBEAT_PREFIX):], "key": key,
                    "tracked": False, **st})
    return {
        "sessions": out,
        "thresholds": {"fresh_max_s": FRESH_MAX_S, "stale_max_s": STALE_MAX_S},
        "tracked": list(TRACKED_SESSIONS),
        # NOT swallowed: if the settings table could not be read, every session
        # would otherwise render `never`, i.e. "nobody is running" — a false
        # red that reads exactly like a real one.
        "readable": readable,
        "read_error": read_error,
        "generated_at": _now(),
    }


@router.post("/heartbeat")
def write_heartbeat(payload: dict = Body(...), user=Depends(get_current_user)):
    """A live session says "I am here". Body: `{letter, actor?, note?}`.

    Whole-value write, deliberately: a heartbeat's whole meaning IS its latest
    value, and a partial JSONB update is the trap memory
    `feedback_jsonb_partial_updates` records.
    """
    letter = parse_letter(payload.get("letter"))
    value = {
        "at": _now(),
        "actor": str(payload.get("actor") or letter)[:64],
        "note": (str(payload.get("note"))[:500] if payload.get("note") else None),
    }
    from db import set_system_setting
    # `updated_by` is a UUID FK to auth.users — the ACTOR goes in the JSONB
    # value, never there. Passing a letter would violate the constraint.
    ok = set_system_setting(
        heartbeat_key(letter), value,
        description=(f"Sign of life for the {letter} session — written by the "
                     "session itself each cycle; the session dashboards render "
                     "NOT RUNNING when it goes stale (board #251)."))
    if not ok:
        raise HTTPException(500, f"failed to persist {heartbeat_key(letter)}")
    return {"key": heartbeat_key(letter), "letter": letter, "value": value}


# ── The deploy log, as the serving container sees it ────────────────────────

# Same two-root reasoning as m_session.py, and the same hard lesson: computing
# a parent that does not exist raised IndexError AT IMPORT and served 502 for
# ~20 minutes after Wave 24. Never eager, always guarded.
_ROR_ROOT = Path(__file__).resolve().parents[3]
_ROR_ROOT = Path(os.getenv("RORT_RULES_ROOT") or _ROR_ROOT)

DEPLOY_LOG_PATH = "docs/Deploy_Log.md"

# Bounded (#240 — the M-session activity log painted ~435 rows and stalled
# Kevin's browser). Render less at a time, never keep less: the file is not
# truncated, only the slice returned is.
DEPLOY_LOG_MAX_ENTRIES = 12
DEPLOY_LOG_EXCERPT = 400

_ENTRY_RE = re.compile(
    r"^- \*\*(?P<time>[0-9:]+ UTC[^*]*)\*\* — `(?P<sha>[0-9a-f]+)`(?P<rest>.*)$")
_WAVE_RE = re.compile(r"\bWave[- ](\d+)\b", re.I)


def parse_deploy_log(text: str, limit: int = DEPLOY_LOG_MAX_ENTRIES) -> list:
    """The most recent deploy-log entries, newest first, BOUNDED.

    The log is append-at-top under a `## YYYY-MM-DD` day heading, one entry per
    `- **HH:MM:SS UTC (…)** — \\`sha\\` …` line. Only the headline is parsed —
    the sub-bullets are prose and belong in the file, not on a dashboard.

    The `wave` number is what makes this module answer a question the container
    otherwise cannot: **if Wave N appears here, its deploy-log PR has merged**,
    because this file is read out of the commit the API is serving. A train
    that ran with no entry here still owes its log.

    KNOWN LIMIT, measured: some entries do not name their wave in the HEADLINE
    (Wave 24's does not; it appears only in the sub-bullets, which are prose and
    are not parsed). Such a train renders `owed / PR open` when its log in fact
    merged. That is the SAFE direction — a visible row R clears in one glance,
    rather than a silently-forgotten log — but it is a limit, not a fact, and
    the page labels the answer as inferred.
    """
    entries, day = [], None
    for line in (text or "").splitlines():
        if line.startswith("## ") and re.match(r"^## \d{4}-\d{2}-\d{2}", line):
            day = line[3:].strip()
            continue
        m = _ENTRY_RE.match(line)
        if not m:
            continue
        rest = m.group("rest").strip()
        w = _WAVE_RE.search(rest)
        entries.append({
            "day": day,
            "time": m.group("time").strip(),
            "sha": m.group("sha"),
            "wave": int(w.group(1)) if w else None,
            "excerpt": re.sub(r"\*\*|`", "", rest)[:DEPLOY_LOG_EXCERPT],
        })
        if len(entries) >= limit:
            break
    return entries


@router.get("/deploy-log")
def deploy_log(limit: int = DEPLOY_LOG_MAX_ENTRIES, user=Depends(get_current_user)):
    """Recent deploy-log entries + the waves they record.

    Always 200, even when the file is missing: the page must be able to SHOW
    the failure, and it reports the ABSOLUTE path it tried. A renderer that
    quietly returns nothing reads as "no deploys have happened".
    """
    limit = max(1, min(int(limit or DEPLOY_LOG_MAX_ENTRIES),
                       DEPLOY_LOG_MAX_ENTRIES))
    p = _ROR_ROOT / DEPLOY_LOG_PATH
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        return {"entries": [], "waves": [], "found": False,
                "resolved_path": str(p), "limit": limit,
                "error": f"could not read {p}: {e}",
                "generated_at": _now()}
    entries = parse_deploy_log(raw, limit)
    return {
        "entries": entries,
        "waves": sorted({e["wave"] for e in entries if e["wave"]}, reverse=True),
        "found": True,
        "resolved_path": str(p),
        "limit": limit,
        "error": None,
        "note": ("read out of the commit THIS CONTAINER is serving — a wave "
                 "missing here either has no deploy-log entry yet (its PR is "
                 "still open) or landed after this container deployed"),
        "generated_at": _now(),
    }


# ── The service itself ──────────────────────────────────────────────────────

# Named, not guessed. What this container cannot see is printed ON the page —
# a dashboard whose blind spots are invisible manufactures false confidence
# (Spec_Dispatch_Dashboard.md §3, and the #157 dead-alarm lesson).
KNOWN_GAPS = [
    "the per-service Railway deploy rollup (7 services) — this container can "
    "report only ITSELF; the full rollup needs `railway status`, which is E's",
    "GitHub PR state (open / merged) — there is no checkout and no credential "
    "here; deploy-log PR state is inferred from whether the entry is present "
    "in the served commit, which lags by one deploy",
    "`git merge-base --is-ancestor` — merged-ness is read from #242's "
    "git-gated ship-step tick, never computed here",
]


@router.get("/service")
def service(user=Depends(get_current_user)):
    """This API's own health answer + the commit it is serving.

    `/health` returning 200 is the standing done-condition for a train (#238):
    on Wave 24 deploy-watch reported 7/7 `success` while the api served 502 for
    twenty minutes. Deploy status is not health. This endpoint answers for the
    api service only — and says so.
    """
    return {
        "status": "ok",
        "commit": os.getenv("RAILWAY_GIT_COMMIT_SHA"),
        "branch": os.getenv("RAILWAY_GIT_BRANCH"),
        "service": os.getenv("RAILWAY_SERVICE_NAME"),
        "deployment_id": os.getenv("RAILWAY_DEPLOYMENT_ID"),
        "scope": "the api service only",
        "known_gaps": KNOWN_GAPS,
        "generated_at": _now(),
    }
