"""Generate deploy_history.json from git log on origin/dev.

Each push/merge to `dev` triggers a Railway redeploy of all services
(including the worker, which restarts Ralph). The "By Deploy" tab on the
Strategy Health page uses this file to bucket pair-rate metrics by deploy
window.

WHY A STATIC FILE (and why it goes stale):
  The running container has no git history — `.dockerignore` excludes
  `.git/`, and the Dockerfiles `COPY src/` at build time. So the API cannot
  read git directly, and the build cannot regenerate this either. The JSON
  must be regenerated *and committed* BEFORE a dev merge so the fresh copy
  is baked into the image. If nobody regenerates it, the tab silently shows
  "no deploys recorded in window" (board #50, 2026-07-26 refresh).

AUTOMATION (so it never goes stale again — see
docs/_active/Plan_Deploy_History_Automation.md):
  Run this as a standard pre-merge/release step. This script now
  `git fetch origin dev` itself, so a bare run is always accurate:

  cd src && ../.venv/bin/python _generate_deploy_history.py

  Then commit src/deploy_history.json onto the branch being merged.

  A pre-push hook that automates this for local dev pushes lives at
  .githooks/pre-push (install once with:
  `git config core.hooksPath .githooks`). It regenerates against the LOCAL
  ref being pushed via `--ref`, so the commits about to deploy are included.

Writes: src/deploy_history.json (committed to dev).

The script pulls commits since 2026-05-28 (start of our analysis window,
matching the /by-deploy endpoint's default `since`). If you need a wider
range, edit the SINCE constant AND the endpoint default together.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

SINCE = "2026-05-28"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "deploy_history.json")


def _fetch(ref: str) -> None:
    """Best-effort `git fetch` so a bare run is accurate even from a stale
    local branch. Non-fatal: offline runs fall back to last-known refs."""
    # Only fetch tracking refs (origin/*); a local ref (e.g. a hook passing
    # the tip being pushed) is already current and needs no fetch.
    if not ref.startswith("origin/"):
        return
    remote_branch = ref.split("/", 1)[1]
    try:
        subprocess.run(
            ["git", "fetch", "origin", remote_branch, "--quiet"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"warning: git fetch origin {remote_branch} failed "
              f"({e.stderr.strip()}); using last-known {ref}",
              file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", default="origin/dev",
        help="git ref to read history from (default: origin/dev). A pre-push "
             "hook can pass the local dev tip so the commits being pushed are "
             "included.")
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="skip the automatic `git fetch` (for offline / hook use).")
    args = parser.parse_args()

    if not args.no_fetch:
        _fetch(args.ref)

    fmt = "%H|%cI|%s"
    try:
        result = subprocess.run(
            ["git", "log", args.ref, f"--since={SINCE}",
             f"--pretty=format:{fmt}"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"git log {args.ref} failed: {e.stderr}", file=sys.stderr)
        return 1

    commits = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        sha, ts, subject = parts
        commits.append({
            "sha": sha[:7],
            "full_sha": sha,
            "timestamp_iso": ts,
            "subject": subject.strip(),
        })

    # Sort descending (newest first); duplicates impossible
    commits.sort(key=lambda c: c["timestamp_iso"], reverse=True)

    payload = {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "since": SINCE,
        "ref": args.ref,
        "count": len(commits),
        "commits": commits,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH} — {len(commits)} commits from {args.ref} "
          f"since {SINCE}")
    if commits:
        print(f"  newest: {commits[0]['timestamp_iso']} "
              f"{commits[0]['sha']} {commits[0]['subject'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
