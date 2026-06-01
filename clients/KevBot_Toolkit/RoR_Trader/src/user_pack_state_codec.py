"""Generic state codec for user-pack incremental engines.

The persistent snapshot path (`unified_engine.snapshot_state(persistent=True)`
+ `serialize_backtest_snapshot`) historically dropped every user-pack
engine's state because `pickle.dumps` could not serialize an instance of an
importlib-loaded class with a synthetic module name (each pack lives in a
fabricated module path the unpickler cannot resolve at restore time). The
fix here pickles only the engine's `__dict__` contents — primitives, lists,
pandas.Timestamp etc., all of which pickle in isolation — sidestepping the
synthetic-class problem.

A pack class may opt out of the generic codec by defining its own
`serialize_state(self) -> dict` and `restore_state(self, state: dict)`
methods (the codec dispatches to those when both are present).

Default state schema for the generic path:
    {'_v': <state_version>, 'attrs': dict(eng.__dict__)}

State versioning:
- Each engine class may set a class attribute `state_version: int`
  (default 1).
- The codec stamps the version on serialize; restore reads it back.
- A pack that changes its internal attrs should bump `state_version` and
  either provide an explicit `restore_state` to migrate, or accept that
  old snapshots will lose the touched fields (self-heal on next save).

This module has no dependency on `unified_engine` — it operates on any
object with a `__dict__`.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_STATE_VERSION = 1


class PackStateCodecError(Exception):
    """Raised when serialize/restore cannot proceed safely."""


def _has_override(eng) -> bool:
    """A pack opts in to explicit (de)serialization by defining BOTH
    `serialize_state` and `restore_state`. Defining one without the other
    is a contract bug — `validate_state_protocol()` rejects that at
    registration time; we treat it as missing here so failure is localized.
    """
    return (callable(getattr(eng, 'serialize_state', None))
            and callable(getattr(eng, 'restore_state', None)))


def serialize_pack_state(eng) -> dict:
    """Snapshot a user-pack engine's state to a pickle-safe dict.

    The returned dict is INDEPENDENT of the source engine — deep-copies
    every value so subsequent engine updates don't mutate the snapshot
    (lists are the common gotcha; ring-buffer packs would otherwise share
    refs and corrupt parallel state). Production pickles the envelope
    anyway, but the codec's own contract is "give me state I can hold."
    """
    import copy
    state_version = int(getattr(
        type(eng), 'state_version', _DEFAULT_STATE_VERSION))
    if _has_override(eng):
        try:
            payload = eng.serialize_state()
        except Exception as e:
            raise PackStateCodecError(
                f"{type(eng).__name__}.serialize_state() raised: {e}") from e
        if not isinstance(payload, dict):
            raise PackStateCodecError(
                f"{type(eng).__name__}.serialize_state() must return dict; "
                f"got {type(payload).__name__}")
        payload.setdefault('_v', state_version)
        return payload
    return {'_v': state_version, 'attrs': copy.deepcopy(eng.__dict__)}


def restore_pack_state(eng, state: dict) -> None:
    """Restore a snapshot dict produced by `serialize_pack_state`.

    On version-skew (`state['_v']` != class `state_version`) in the generic
    path, log a warning and attempt the restore anyway — typically the
    new code tolerates missing or extra attrs. The opt-in path owns its
    own migration logic.
    """
    if not isinstance(state, dict):
        raise PackStateCodecError(
            f"restore_pack_state expected dict, got {type(state).__name__}")
    if _has_override(eng):
        try:
            eng.restore_state(state)
        except Exception as e:
            raise PackStateCodecError(
                f"{type(eng).__name__}.restore_state() raised: {e}") from e
        return
    cls_v = int(getattr(
        type(eng), 'state_version', _DEFAULT_STATE_VERSION))
    snap_v = int(state.get('_v', _DEFAULT_STATE_VERSION))
    if snap_v != cls_v:
        logger.warning(
            "user-pack state version skew for %s: snapshot v=%s class v=%s — "
            "attempting restore", type(eng).__name__, snap_v, cls_v)
    attrs = state.get('attrs')
    if not isinstance(attrs, dict):
        raise PackStateCodecError(
            f"generic codec restore for {type(eng).__name__} missing 'attrs' "
            f"dict (state keys: {list(state)})")
    for k, v in attrs.items():
        setattr(eng, k, v)
