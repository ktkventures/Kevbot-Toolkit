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
