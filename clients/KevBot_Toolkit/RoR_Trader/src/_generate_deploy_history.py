"""Generate deploy_history.json from git log on origin/dev.

Each push to `dev` triggers a Railway redeploy of all services
(including the worker, which restarts Ralph). The "By Deploy" tab
on the Strategy Health page uses this file to bucket pair-rate
metrics by deploy window.

USAGE: run this whenever you want the By Deploy view to reflect
the latest commits. The API endpoint reads from a static JSON
file (not from git directly) because the running container does
not have a working tree with git history.

  cd src && ../.venv/bin/python _generate_deploy_history.py

Writes: src/deploy_history.json (committed to dev).

The script pulls commits from `git log origin/dev` since 2026-05-28
(start of our analysis window). If you need a wider range, edit the
SINCE constant.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

SINCE = "2026-05-28"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "deploy_history.json")


def main() -> int:
    fmt = "%H|%cI|%s"
    try:
        result = subprocess.run(
            ["git", "log", "origin/dev", f"--since={SINCE}",
             f"--pretty=format:{fmt}"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"git log failed: {e.stderr}", file=sys.stderr)
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
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "since": SINCE,
        "count": len(commits),
        "commits": commits,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_PATH} — {len(commits)} commits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
