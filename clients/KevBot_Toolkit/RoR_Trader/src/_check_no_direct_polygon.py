"""CI guard — the M-RS4 invariant: bar_cache is the ONLY thing that calls Polygon.

Every consumer must load bars from a cache (bar_cache / live_bars), never call
Polygon REST directly. This guard greps production modules for direct Polygon
calls and fails (exit 1) if a NON-allowlisted module makes one — so a new
consumer can't silently re-introduce a Polygon bypass.

Allowlisted callers are the legitimate ones: the bar_cache ingest/backfill (the
single Polygon caller by design), the WS ingest + live_bars REST gap-fillers,
diagnostics/parity tools, reference-data, and the load_from_polygon implementation
itself. Underscore-prefixed (`_*.py`) and `test_*` files are offline/diagnostic
and skipped.

Run:  PYTHONPATH=. python _check_no_direct_polygon.py     (exit 0 = clean)
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Direct-Polygon call signatures (calls, not defs/imports/comments).
PATTERNS = [
    re.compile(r"\bload_from_polygon\s*\("),
    re.compile(r"\b_polygon_fetch_bars\s*\("),
    re.compile(r"\b_polygon_1s\s*\("),
    re.compile(r"\b_polygon_1m\s*\("),
    re.compile(r"https?://api\.polygon\.io"),
    re.compile(r"wss?://socket\.polygon\.io"),
    re.compile(r"files\.polygon\.io"),
]

# Modules allowed to call Polygon directly (by basename / relative path).
ALLOWLIST = {
    "data_loader.py",        # load_from_polygon impl + _polygon_* + fetch_1s + crypto
    "bar_cache.py",          # the ONE Polygon caller: ingest/backfill/maintain
    "data_worker_ingest.py", # the write-through ingest writer
    "gap_healer.py",         # live_bars (WS) gap-healing
    "live_bars_rest_backfill.py",
    "live_bars_rest_backfill_subminute.py",
    "live_bars_writer.py",
    "rest_verifier.py",      # diagnostic REST verifier
    "fidelity_parity_suite.py",  # the cache-vs-Polygon parity gate (must call both)
    "flat_file_ingestion.py",    # bulk historical ingest (Polygon S3)
    "polygon_conditions.py",     # reference data (trade condition codes), not bars
    "ralph_engine.py",           # the live WS feed (socket.polygon.io)
    "worker.py",                 # live_bars 1Min gap fill
    os.path.join("api", "routers", "admin_parity.py"),  # admin parity diagnostic
    os.path.join("api", "routers", "rest_verifier.py"),
}


def _rel(path: str) -> str:
    return os.path.relpath(path, HERE)


def main() -> int:
    files = (glob.glob(os.path.join(HERE, "*.py"))
             + glob.glob(os.path.join(HERE, "api", "routers", "*.py")))
    violations = []
    for path in files:
        base = os.path.basename(path)
        rel = _rel(path)
        if base.startswith("_") or base.startswith("test_") or base.endswith("_test.py"):
            continue
        if base in ALLOWLIST or rel in ALLOWLIST:
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue  # skip comments / docstring lines
            for pat in PATTERNS:
                if pat.search(line):
                    violations.append(f"{rel}:{i}: {line.strip()[:100]}")
                    break

    if violations:
        print("❌ DIRECT-POLYGON INVARIANT VIOLATED — these consumers must read a "
              "cache (bar_cache/live_bars), not call Polygon directly:")
        for v in violations:
            print(f"  {v}")
        print(f"\n{len(violations)} violation(s). If a call is legitimate (an ingest/"
              "diagnostic path), add the module to ALLOWLIST in this file.")
        return 1
    print("✅ no direct-Polygon calls in production consumer modules "
          "(bar_cache is the sole Polygon caller).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
