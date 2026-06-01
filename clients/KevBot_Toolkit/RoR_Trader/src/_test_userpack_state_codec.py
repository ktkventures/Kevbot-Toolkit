"""Per-pack round-trip tests for the user-pack state codec.

For every user-pack incremental engine in `user_packs/<slug>/
indicator_incremental.py`:

  1. Instantiate with default params.
  2. Feed 100 synthetic OHLCV bars.
  3. Snapshot state via `serialize_pack_state`.
  4. Instantiate a fresh engine and `restore_pack_state(eng, state)`.
  5. Feed 50 more identical bars to BOTH engines.
  6. Assert outputs from `update_bar` are bit-identical bar-for-bar.

Catches: missing state attrs, type-coercion bugs, version-skew handling,
and any pack whose state silently fails round-trip (the original bug
class).

Run from `src/`:
    .venv/bin/python _test_userpack_state_codec.py
or:
    .venv/bin/python _test_userpack_state_codec.py --slug ut_bot_v4
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# Pack discovery
# ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
USER_PACKS_DIR = REPO_ROOT / "user_packs"


def discover_packs() -> list[tuple[str, Path]]:
    """Return [(slug, incremental_path)] for every pack with an
    indicator_incremental.py file.
    """
    out = []
    for pack_dir in sorted(USER_PACKS_DIR.iterdir()):
        if not pack_dir.is_dir():
            continue
        inc = pack_dir / "indicator_incremental.py"
        if inc.exists():
            out.append((pack_dir.name, inc))
    return out


def import_incremental_class(slug: str, path: Path):
    """Import the incremental module and return the (single) class
    defined in it whose name ends with `Incremental`.
    """
    spec = importlib.util.spec_from_file_location(
        f"_codec_test_{slug}", str(path))
    mod = importlib.util.module_from_spec(spec)
    # Some packs do `from indicator_incremental import X` at module load —
    # mirror pack_registry's trick of pre-registering under the bare name.
    sys.modules["indicator_incremental"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("indicator_incremental", None)
    for name in dir(mod):
        if name.endswith("Incremental"):
            return getattr(mod, name)
    raise RuntimeError(f"no *Incremental class found in {path}")


# ─────────────────────────────────────────────────────────────────────
# Synthetic bars
# ─────────────────────────────────────────────────────────────────────

def make_bars(n: int, seed: int = 42) -> list[dict]:
    """Deterministic bars that wiggle around a drifting price + spikes."""
    import random
    rng = random.Random(seed)
    base = 100.0
    out = []
    ts0 = pd.Timestamp("2026-01-02 14:30:00", tz="UTC")
    for i in range(n):
        drift = i * 0.03
        noise = rng.uniform(-0.5, 0.5)
        spike = (1.5 if i % 17 == 0 else 0.0) * (1 if rng.random() < 0.5 else -1)
        c = base + drift + noise + spike
        o = c + rng.uniform(-0.2, 0.2)
        h = max(o, c) + abs(rng.uniform(0, 0.4))
        l = min(o, c) - abs(rng.uniform(0, 0.4))
        v = rng.uniform(1000, 5000)
        out.append({
            "open": float(o), "high": float(h), "low": float(l),
            "close": float(c), "volume": float(v),
            "timestamp": ts0 + pd.Timedelta(seconds=i * 60),
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# Output comparison
# ─────────────────────────────────────────────────────────────────────

def outputs_match(a: dict, b: dict, tol: float = 1e-12) -> tuple[bool, str]:
    """Compare two pack `update_bar` return dicts. NaN==NaN counts as equal."""
    if set(a.keys()) != set(b.keys()):
        return False, f"key mismatch: only_a={set(a)-set(b)} only_b={set(b)-set(a)}"
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, float) and isinstance(bv, float):
            if math.isnan(av) and math.isnan(bv):
                continue
            if math.isnan(av) or math.isnan(bv):
                return False, f"key {k!r}: NaN mismatch a={av} b={bv}"
            if abs(av - bv) > tol:
                return False, f"key {k!r}: {av!r} vs {bv!r}"
        elif av != bv:
            return False, f"key {k!r}: {av!r} vs {bv!r}"
    return True, ""


# ─────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if ok:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def run_pack_test(slug: str, inc_path: Path) -> None:
    """Round-trip one pack: 100-bar warmup → codec → 50-bar parallel run."""
    try:
        cls = import_incremental_class(slug, inc_path)
    except Exception as e:
        _check(f"{slug}: import", False, str(e))
        return

    try:
        eng_a = cls()
    except Exception as e:
        _check(f"{slug}: instantiate no-args", False, str(e))
        return

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from user_pack_state_codec import serialize_pack_state, restore_pack_state

    bars = make_bars(150, seed=ord(slug[0]))
    warmup, tail = bars[:100], bars[100:]

    # Warm engine A through 100 bars.
    try:
        for b in warmup:
            eng_a.update_bar(b)
    except Exception as e:
        _check(f"{slug}: warmup 100 bars", False, str(e))
        return

    # Snapshot via codec, restore onto fresh engine B.
    try:
        state = serialize_pack_state(eng_a)
    except Exception as e:
        _check(f"{slug}: serialize", False, str(e))
        return
    try:
        eng_b = cls()
        restore_pack_state(eng_b, state)
    except Exception as e:
        _check(f"{slug}: restore on fresh instance", False, str(e))
        return

    # Verify the restored engine's __dict__ matches the source.
    if not hasattr(eng_b, "serialize_state"):
        attrs_a = {k: v for k, v in vars(eng_a).items()}
        attrs_b = {k: v for k, v in vars(eng_b).items()}
        if set(attrs_a) != set(attrs_b):
            _check(
                f"{slug}: __dict__ keys match",
                False,
                f"only_a={set(attrs_a)-set(attrs_b)} "
                f"only_b={set(attrs_b)-set(attrs_a)}",
            )
            return

    # Run 50 more bars on both engines in lockstep; outputs must match.
    mismatches = []
    for i, b in enumerate(tail):
        out_a = eng_a.update_bar(dict(b))
        out_b = eng_b.update_bar(dict(b))
        ok, why = outputs_match(out_a, out_b)
        if not ok:
            mismatches.append(f"bar +{i}: {why}")
            if len(mismatches) >= 3:
                break

    _check(
        f"{slug}: 50-bar post-restore parity",
        not mismatches,
        "; ".join(mismatches) if mismatches else "",
    )


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default=None,
                   help="run only this pack (default: all)")
    args = p.parse_args()

    packs = discover_packs()
    if args.slug:
        packs = [(s, p) for (s, p) in packs if s == args.slug]
        if not packs:
            print(f"no pack named {args.slug!r}")
            sys.exit(1)

    print(f"running user-pack state codec round-trip on {len(packs)} packs")
    print("=" * 60)
    for slug, path in packs:
        run_pack_test(slug, path)
    print("=" * 60)
    print(f"  {_PASS} passed, {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)


if __name__ == "__main__":
    main()
