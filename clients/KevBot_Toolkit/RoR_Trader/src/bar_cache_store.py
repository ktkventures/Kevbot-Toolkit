"""Bar-cache ("dough") serialization — Tier 2 groundwork (2026-06-18).

Persist the output of ``unified_engine.precompute_bar_cache`` (a
``List[CachedBarState]`` + metadata dict) so a saved strategy's backtest lane
can be re-derived later via ``run_trades_from_cache`` (the 10-50x replay path)
instead of a full recompute — and survive an API redeploy.

This module is SERIALIZATION ONLY. It does not pick a storage backend; a later
step wires it to a persistent blob store (Supabase Storage / Railway volume) +
a small index table. Kept separate so the lossless round-trip can be validated
in isolation.

Format: ``gzip(pickle((cache, metadata)))``. ``CachedBarState`` carries nested
dicts (current_values/prev_values/c_triggers/l_fills) and a set
(confluence_records); pickle handles all of these losslessly, which a flat
columnar format (Parquet) would not without a flatten/unflatten pass. The dough
is fully REGENERABLE (just re-run precompute_bar_cache), so pickle's
class-version fragility is acceptable: a load that fails to deserialize (e.g.
after a CachedBarState shape change) is treated as a cache miss and the caller
falls back to a fresh recompute. A ``schema_version`` in the envelope lets us
reject stale-shape blobs explicitly rather than crash.
"""

import gzip
import pickle
from typing import Any, List, Tuple

# Bump when CachedBarState's shape or precompute_bar_cache's metadata contract
# changes in a way that makes old blobs unsafe to replay. A mismatch on load is
# a cache miss (regenerate), never a crash.
DOUGH_SCHEMA_VERSION = 1

_PROTOCOL = pickle.HIGHEST_PROTOCOL


def serialize_dough(cache: List[Any], metadata: dict) -> bytes:
    """Serialize (cache, metadata) → gzip(pickle) bytes.

    ``cache`` is the List[CachedBarState] from precompute_bar_cache; ``metadata``
    is its companion dict (available_triggers, etc.). Envelope carries a schema
    version so deserialize can reject incompatible blobs.
    """
    envelope = {
        'schema_version': DOUGH_SCHEMA_VERSION,
        'n_bars': len(cache),
        'cache': cache,
        'metadata': metadata,
    }
    return gzip.compress(pickle.dumps(envelope, protocol=_PROTOCOL))


def deserialize_dough(blob: bytes) -> Tuple[List[Any], dict]:
    """Inverse of serialize_dough → (cache, metadata).

    Raises ValueError on a schema-version mismatch so the caller treats it as a
    cache miss and recomputes, rather than replaying an incompatible dough.
    """
    envelope = pickle.loads(gzip.decompress(blob))
    if not isinstance(envelope, dict) or 'cache' not in envelope:
        raise ValueError("dough blob is not a recognized envelope")
    ver = envelope.get('schema_version')
    if ver != DOUGH_SCHEMA_VERSION:
        raise ValueError(
            f"dough schema_version {ver} != current {DOUGH_SCHEMA_VERSION} "
            "— treat as cache miss and recompute")
    return envelope['cache'], envelope['metadata']


def dough_blob_size(cache: List[Any], metadata: dict) -> int:
    """Compressed byte size of the serialized dough (for storage planning)."""
    return len(serialize_dough(cache, metadata))


# ── Persistent store (Supabase Storage) ────────────────────────────────────
# One dough per (user, symbol, tf, session, window) — keyed by config, NOT by
# candidate, since all candidates of a search share the superset dough.
# Fallback-safe everywhere: any failure returns None so the caller recomputes.

DOUGH_BUCKET = 'dough-cache'


def dough_key(strat: dict) -> str:
    """Deterministic blob key for a strategy's dough, derived ONLY from the
    window-defining fields (symbol/tf/session/data_days/backtest_start_date) —
    NOT user_id. The dough is pure market data + superset indicators/triggers
    (no user-private data; the user's confluence is applied at re-bake, and a
    trigger missing from the superset safely falls back to recompute), so it's
    shareable across users by window. Keying without user_id lets the Mass
    Builder producer (which doesn't carry user_id in run_mass_search) and the
    save-side consumer compute the SAME key. Returns '' if underspecified."""
    import hashlib
    cfg = strat.get('config') if isinstance(strat.get('config'), dict) else {}
    symbol = str(strat.get('symbol') or '')
    timeframe = str(strat.get('timeframe') or '')
    parts = [
        symbol, timeframe,
        str(strat.get('trading_session') or strat.get('session') or 'RTH'),
        str(strat.get('data_days') or (cfg or {}).get('data_days') or ''),
        str(strat.get('backtest_start_date')
            or (cfg or {}).get('backtest_start_date') or ''),
    ]
    if not symbol or not timeframe:
        return ''
    h = hashlib.sha1('|'.join(parts).encode()).hexdigest()[:24]
    return f"dough/{symbol}_{timeframe}_{h}.dough.gz"


def put_dough(key: str, cache: List[Any], metadata: dict,
              built_at_iso: str) -> bool:
    """Upload a serialized dough to the bucket (upsert). built_at_iso is
    stamped into the metadata so get_dough can enforce freshness. Returns
    True on success, False on any failure (non-fatal: search still proceeds)."""
    if not key:
        return False
    try:
        from db import get_admin_client
        meta2 = dict(metadata or {})
        meta2['_built_at'] = built_at_iso
        blob = serialize_dough(cache, meta2)
        get_admin_client().storage.from_(DOUGH_BUCKET).upload(
            key, blob,
            {'content-type': 'application/octet-stream', 'upsert': 'true'})
        return True
    except Exception:
        return False


def get_dough(key: str, max_age_hours: float = 36.0):
    """Download + deserialize a dough. Returns (cache, metadata) or None on
    miss / schema mismatch / staleness (built_at older than max_age_hours).
    Never raises — a None return means 'recompute instead'."""
    if not key:
        return None
    try:
        from db import get_admin_client
        blob = get_admin_client().storage.from_(DOUGH_BUCKET).download(key)
        if not blob:
            return None
        cache, metadata = deserialize_dough(blob)
        if max_age_hours is not None:
            built = (metadata or {}).get('_built_at')
            if built:
                from datetime import datetime, timezone
                try:
                    b = datetime.fromisoformat(str(built).replace('Z', '+00:00'))
                    if b.tzinfo is None:
                        b = b.replace(tzinfo=timezone.utc)
                    age_h = (datetime.now(timezone.utc) - b).total_seconds() / 3600.0
                    if age_h > max_age_hours:
                        return None  # stale → recompute
                except Exception:
                    pass
        return cache, metadata
    except Exception:
        return None
