#!/usr/bin/env python3
"""annotate_orphaned_alerts.py — Board #117 (V1.14) dangling alert-history opens.

Offline, idempotent, dry-run-first annotation sweeper for dangling alert-lane
"Open" rows. Layer 1 of the #117 closure design (see
src/migrations/alerts_orphaned_columns.sql for the schema; the UI render is F's
lane; position-state reconciliation is board #114).

THE REFRAME (why this is annotate-only, not a "close position")
  Alert-lane "Open" is NOT stored DB state. The `alerts` table is one row per
  EVENT (type='entry_signal' / 'exit_signal'); there is NO exit_time / status /
  is_open column. "Open" is DERIVED client-side by LIFO-pairing entries to exits
  (frontend/src/views/StrategyDetailPage.tsx:2211-2266): sort by timestamp asc;
  ENTRY -> push; EXIT -> pop most-recent; EXIT-on-empty-stack -> dropped;
  leftover entries render result:'Open' forever. So a "dangling open" is simply
  an entry_signal with no matching exit_signal -- there is no open-flag to flip.

  Kevin's hard rule (#117): alert history must stay TRUE to when alerts actually
  fired. We NEVER synthesize a retroactive exit_signal (firing a sell days later
  would push a live order for a position no longer held). The only correct
  closure is to ANNOTATE the leftover entry row (orphaned_at + orphaned_reason),
  leaving its type / timestamp / price / direction untouched.

WHAT THIS SCRIPT DOES
  1. Reads all alerts (server-side cursor, full-table scan -> run POST-CLOSE).
  2. Replays the frontend LIFO per strategy -> the set of unpaired entry_signal
     rows = "danglers" (byte-faithful to StrategyDetailPage.tsx).
  3. Classifies each dangler (same rule as the #117 inventory):
       eod_churn            fired in the 19:54-20:06Z end-of-day churn window
       redundant_entry      a following entry_signal (<=1h) with no exit between
       redundant_entry_slow same, gap >1h
       lost_exit            no following entry; exit genuinely never fired
  4. Eligibility gate (ALL must hold to annotate):
       - class in --annotate-classes
           default = redundant_entry, redundant_entry_slow, eod_churn
           (the PROVABLE display artifacts: a later entry / EOD churn shows the
            engine round-tripped the position; these are NOT stuck live opens)
       - timestamp < cutoff, cutoff = (--as-of) - (--stale-days) days
           (protects anything recent enough to still be a genuinely-open live
            position; those and all 'lost_exit' route to board #114)
  5. DRY-RUN by default: prints the fleet inventory + the annotate manifest and
     writes NOTHING. --apply performs the UPDATE, guarded WHERE orphaned_at IS
     NULL (idempotent), and writes the exact stamped ids to a manifest CSV.
     It NEVER inserts an exit_signal and NEVER edits type/timestamp/price.

REVERSIBILITY
  --apply writes <out-dir>/orphaned_apply_manifest_<as_of>.csv listing every id
  it stamped. --rollback <manifest.csv> NULLs orphaned_at/orphaned_reason for
  exactly those ids. (The columns are additive-nullable; a full reset is also
  the one-liner in the migration's rollback note.)

TYPICAL USAGE
  # deterministic dry-run report (writes nothing):
  python -m src.annotate_orphaned_alerts --as-of 2026-07-26 --stale-days 3
  # apply (attended / two-touch; NOT for the headless agent):
  python -m src.annotate_orphaned_alerts --as-of 2026-07-26 --stale-days 3 --apply
  # nightly (rolling): omit --as-of (defaults to today UTC 00:00)
  python -m src.annotate_orphaned_alerts --stale-days 3 --apply
  # reverse a run:
  python -m src.annotate_orphaned_alerts --rollback <manifest.csv>
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta, time as dtime

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, "src", ".env"))
import psycopg  # noqa: E402  (after load_dotenv, mirrors the repo's script pattern)

# End-of-day churn window (UTC), matches the #117 inventory + the EOD-reentry arm.
EOD_START, EOD_END = dtime(19, 54), dtime(20, 6)
ANNOTATABLE = ("redundant_entry", "redundant_entry_slow", "eod_churn")
ALL_CLASSES = ANNOTATABLE + ("lost_exit",)


def is_eod(ts):
    return EOD_START <= ts.astimezone(timezone.utc).time() <= EOD_END


def classify(idx, evs, e):
    """Faithful to the #117 inventory classifier. eod window wins first."""
    if is_eod(e["ts"]):
        return "eod_churn"
    nxt = evs[idx + 1] if idx + 1 < len(evs) else None
    if nxt and nxt["is_entry"]:
        gap = (nxt["ts"] - e["ts"]).total_seconds()
        return "redundant_entry" if gap <= 3600 else "redundant_entry_slow"
    return "lost_exit"


def load_alerts(conn):
    """Full-history scan -> {sid: [event, ...]} ordered by (timestamp, id)."""
    rows_by_sid = defaultdict(list)
    with conn.cursor(name="orphan_scan") as cur:
        cur.itersize = 20000
        cur.execute(
            """SELECT strategy_id, type, timestamp, id, symbol, direction,
                      source, live_model, price, user_id
               FROM alerts
               ORDER BY strategy_id, timestamp ASC, id ASC"""
        )
        for sid, typ, ts, aid, sym, direction, source, model, price, uid in cur:
            rows_by_sid[sid].append(
                {
                    "is_entry": "entry" in (typ or "").lower(),
                    "ts": ts,
                    "id": aid,
                    "sym": sym,
                    "direction": direction,
                    "source": source,
                    "model": model,
                    "price": price,
                    "user_id": uid,
                }
            )
    return rows_by_sid


def find_danglers(rows_by_sid):
    """Replicate the frontend LIFO -> leftover unpaired entries, classified."""
    danglers = []
    for sid, evs in rows_by_sid.items():
        evs.sort(key=lambda e: (e["ts"], e["id"]))
        stack = []
        for idx, e in enumerate(evs):
            if e["is_entry"]:
                stack.append((idx, e))
            elif stack:
                stack.pop()
        for idx, e in stack:
            e = {**e, "sid": sid, "cls": classify(idx, evs, e)}
            danglers.append(e)
    return danglers


def columns_present(conn):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'alerts'
                 AND column_name IN ('orphaned_at', 'orphaned_reason')"""
        )
        return {r[0] for r in cur.fetchall()} == {"orphaned_at", "orphaned_reason"}


def already_annotated_ids(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM alerts WHERE orphaned_at IS NOT NULL")
        return {r[0] for r in cur.fetchall()}


# ---------------------------------------------------------------------------


def cmd_rollback(conn, manifest_path):
    ids = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(int(row["alert_id"]))
    if not ids:
        print("rollback: manifest has no ids; nothing to do.")
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE alerts SET orphaned_at = NULL, orphaned_reason = NULL "
            "WHERE id = ANY(%s) AND orphaned_at IS NOT NULL",
            (ids,),
        )
        n = cur.rowcount
    conn.commit()
    print(f"rollback: cleared annotation on {n} of {len(ids)} manifest rows.")


def report(danglers, cutoff, annotate_classes, eligible, protected_recent,
           excluded_class, skipped_annotated):
    by_class = defaultdict(int)
    by_sym = defaultdict(int)
    for d in danglers:
        by_class[d["cls"]] += 1
        by_sym[d["sym"]] += 1
    users = defaultdict(int)
    for d in danglers:
        users[str(d["user_id"])] += 1

    print("=" * 72)
    print(f"DANGLING OPEN INVENTORY (fleet-wide)  as-of cutoff = {cutoff.isoformat()}")
    print("=" * 72)
    print(f"total danglers: {len(danglers)}  across "
          f"{len({d['sid'] for d in danglers})} strategies")
    print("\n-- by class (full history) --")
    for c in sorted(by_class, key=lambda c: -by_class[c]):
        print(f"   {c:22s} {by_class[c]:5d}")
    print("\n-- by symbol --")
    for s in sorted(by_sym, key=lambda s: -by_sym[s]):
        print(f"   {str(s):8s} {by_sym[s]}")
    print("\n-- by user_id (annotation must not touch a foreign user) --")
    for u in sorted(users, key=lambda u: -users[u]):
        print(f"   {u:40s} {users[u]}")

    print("\n" + "-" * 72)
    print("ANNOTATE MANIFEST (what --apply would stamp)")
    print("-" * 72)
    print(f"annotate classes: {', '.join(annotate_classes)}")
    print(f"eligible to annotate now: {len(eligible)}")
    ecls = defaultdict(int)
    for d in eligible:
        ecls[d["cls"]] += 1
    for c in sorted(ecls, key=lambda c: -ecls[c]):
        print(f"   {c:22s} {ecls[c]:5d}")
    if skipped_annotated:
        print(f"   (already annotated, idempotent skip: {skipped_annotated})")

    print("\n-- NOT auto-annotated (route to board #114 / separate review) --")
    print(f"   lost_exit (exit genuinely never fired):        {len(excluded_class)}")
    print(f"   recent (ts >= cutoff, possible live position):  {len(protected_recent)}")
    if protected_recent:
        print("   recent detail (review individually — could be a real carry):")
        for d in sorted(protected_recent, key=lambda d: d["ts"]):
            print(f"     sid {d['sid']:4d}  {d['ts'].isoformat()}  {d['sym']:6s} "
                  f"{str(d['direction']):5s} {d['cls']:20s} id={d['id']}")


def main():
    ap = argparse.ArgumentParser(description="Annotate dangling alert opens (#117).")
    ap.add_argument("--apply", action="store_true",
                    help="perform the UPDATE (default: dry-run, writes nothing)")
    ap.add_argument("--rollback", metavar="MANIFEST_CSV",
                    help="un-annotate the ids listed in a prior apply manifest")
    ap.add_argument("--as-of", metavar="YYYY-MM-DD",
                    help="reference 'now' for the staleness cutoff "
                         "(default: today UTC 00:00)")
    ap.add_argument("--stale-days", type=int, default=3,
                    help="only annotate danglers older than as-of minus N days "
                         "(default: 3)")
    ap.add_argument("--annotate-classes", default=",".join(ANNOTATABLE),
                    help="comma list of classes to annotate "
                         f"(default: {','.join(ANNOTATABLE)})")
    ap.add_argument("--include-lost-exit", action="store_true",
                    help="also annotate lost_exit (NOT recommended — those are "
                         "possible real positions; default routes them to #114)")
    ap.add_argument("--user-id", metavar="UUID",
                    help="restrict to a single user_id (default: all)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "src"),
                    help="where to write manifest CSVs (default: src/)")
    args = ap.parse_args()

    conn_str = os.getenv("SUPABASE_CONNECTION_STRING", "")
    if not conn_str:
        sys.exit("FATAL: SUPABASE_CONNECTION_STRING not set (src/.env).")

    with psycopg.connect(conn_str, connect_timeout=20) as conn:
        if args.rollback:
            cmd_rollback(conn, args.rollback)
            return

        cols_ok = columns_present(conn)
        as_of = (datetime.fromisoformat(args.as_of).replace(tzinfo=timezone.utc)
                 if args.as_of
                 else datetime.now(timezone.utc).replace(
                     hour=0, minute=0, second=0, microsecond=0))
        cutoff = as_of - timedelta(days=args.stale_days)

        annotate_classes = set(c.strip() for c in args.annotate_classes.split(",")
                               if c.strip())
        if args.include_lost_exit:
            annotate_classes.add("lost_exit")
        bad = annotate_classes - set(ALL_CLASSES)
        if bad:
            sys.exit(f"FATAL: unknown class(es) in --annotate-classes: {bad}")

        rows_by_sid = load_alerts(conn)
        danglers = find_danglers(rows_by_sid)
        if args.user_id:
            danglers = [d for d in danglers if str(d["user_id"]) == args.user_id]

        annotated_ids = already_annotated_ids(conn) if cols_ok else set()

        eligible, protected_recent, excluded_class, skipped = [], [], [], 0
        for d in danglers:
            if d["cls"] not in annotate_classes:
                excluded_class.append(d)
                continue
            if d["ts"] >= cutoff:
                protected_recent.append(d)
                continue
            if d["id"] in annotated_ids:
                skipped += 1
                continue
            eligible.append(d)

        report(danglers, cutoff, sorted(annotate_classes), eligible,
               protected_recent, excluded_class, skipped)

        # Always write the full worklist CSV (inventory artifact, deterministic).
        stamp = as_of.date().isoformat()
        os.makedirs(args.out_dir, exist_ok=True)
        worklist = os.path.join(args.out_dir, f"dangler_inventory_{stamp}.csv")
        with open(worklist, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sid", "alert_id", "ts", "symbol", "direction", "price",
                        "source", "live_model", "class", "user_id",
                        "eligible_to_annotate"])
            elig_ids = {d["id"] for d in eligible}
            for d in sorted(danglers, key=lambda d: (d["sid"], d["ts"])):
                w.writerow([d["sid"], d["id"], d["ts"].isoformat(), d["sym"],
                            d["direction"], d["price"], d["source"], d["model"],
                            d["cls"], d["user_id"], d["id"] in elig_ids])
        print(f"\ninventory CSV -> {worklist}")

        if not args.apply:
            print("\n[DRY-RUN] no rows written. Re-run with --apply to annotate "
                  f"the {len(eligible)} eligible rows.")
            if not cols_ok:
                print("NOTE: alerts.orphaned_at/orphaned_reason do not exist yet "
                      "— apply src/migrations/alerts_orphaned_columns.sql first.")
            return

        # ---- APPLY ----
        if not cols_ok:
            sys.exit("FATAL: --apply requires the migration "
                     "(alerts_orphaned_columns.sql) to be applied first.")
        if not eligible:
            print("\n--apply: nothing eligible; no rows written.")
            return

        # Group ids by reason (class) so each row records WHY it was orphaned.
        by_reason = defaultdict(list)
        for d in eligible:
            by_reason[d["cls"]].append(d["id"])
        total = 0
        with conn.cursor() as cur:
            for reason, ids in by_reason.items():
                cur.execute(
                    "UPDATE alerts SET orphaned_at = now(), orphaned_reason = %s "
                    "WHERE id = ANY(%s) AND orphaned_at IS NULL",
                    (reason, ids),
                )
                total += cur.rowcount
        conn.commit()

        manifest = os.path.join(args.out_dir, f"orphaned_apply_manifest_{stamp}.csv")
        with open(manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alert_id", "sid", "ts", "orphaned_reason"])
            for d in eligible:
                w.writerow([d["id"], d["sid"], d["ts"].isoformat(), d["cls"]])
        print(f"\n--apply: stamped {total} rows. manifest -> {manifest}")
        print("NO exit_signal rows were inserted; only orphaned_at/orphaned_reason "
              "were set. Reverse with --rollback <manifest>.")


if __name__ == "__main__":
    main()
