"""Lock the snapshot lineage guard (2026-06-11, iter 0611a).

Appends resume from persisted engine snapshots; full UAD never rewrites
them, so past state divergence is fossilized (sid 302: append-created
trades exited via stop while a fresh run + live both exited via
utv4_bear_flip). The guard refuses resume unless the lineage was
anchored fresh at/after SNAPSHOT_LINEAGE_EPOCH; refused runs take the
warmup-windowed path and anchor a fresh lineage on persist.

Run:  cd src && ../.venv/bin/python -m pytest test_snapshot_lineage_guard.py -v
"""
import sys
sys.path.insert(0, '.')

from api.services.forward_test_service import (
    SNAPSHOT_LINEAGE_EPOCH,
    _guarded_snapshot,
    _snapshot_lineage_key,
    _snapshot_lineage_ok,
    _stamp_lineage_on_persist,
)

SNAP = 'gASVfakesnapshot=='


def test_no_anchor_refuses_resume():
    cfg = {'engine_snapshot_b64': SNAP, 'engine_snapshot_at': '2026-06-11T18:40:00+00:00'}
    assert _guarded_snapshot(cfg, 'bt', 'engine_snapshot_b64', 999) is None


def test_pre_epoch_anchor_refuses_resume():
    cfg = {'engine_snapshot_b64': SNAP,
           'snapshot_lineage_anchor_at': '2026-06-10T12:00:00+00:00'}
    assert _guarded_snapshot(cfg, 'bt', 'engine_snapshot_b64', 999) is None


def test_post_epoch_anchor_allows_resume():
    cfg = {'engine_snapshot_b64': SNAP,
           'snapshot_lineage_anchor_at': '2026-06-11T21:00:00+00:00'}
    assert _guarded_snapshot(cfg, 'bt', 'engine_snapshot_b64', 999) == SNAP


def test_recent_save_time_does_not_bypass_guard():
    """The 302 trap: snapshot SAVED today but lineage inherited from a
    poisoned past. Save-time must not count — only the anchor does."""
    cfg = {'engine_snapshot_b64': SNAP,
           'engine_snapshot_at': '2026-06-11T18:40:00+00:00'}  # post-epoch save
    assert _guarded_snapshot(cfg, 'bt', 'engine_snapshot_b64', 999) is None


def test_missing_snapshot_returns_none():
    assert _guarded_snapshot({}, 'bt', 'engine_snapshot_b64', 999) is None


def test_lanes_have_separate_anchor_keys():
    assert _snapshot_lineage_key('bt') != _snapshot_lineage_key('algo')
    cfg = {'engine_snapshot_b64_algo': SNAP,
           'snapshot_lineage_anchor_at': '2026-06-11T21:00:00+00:00'}  # bt anchor only
    assert _guarded_snapshot(cfg, 'algo', 'engine_snapshot_b64_algo', 999) is None


def test_fresh_run_anchors_lineage():
    cfg = {}
    _stamp_lineage_on_persist(cfg, 'bt', resumed_from_b64=None)
    anchor = cfg.get('snapshot_lineage_anchor_at')
    assert anchor and anchor >= SNAPSHOT_LINEAGE_EPOCH
    assert _snapshot_lineage_ok(cfg, 'bt')


def test_resumed_run_keeps_existing_anchor():
    cfg = {'snapshot_lineage_anchor_at': '2026-06-11T21:00:00+00:00'}
    _stamp_lineage_on_persist(cfg, 'bt', resumed_from_b64=SNAP)
    assert cfg['snapshot_lineage_anchor_at'] == '2026-06-11T21:00:00+00:00'


def test_self_heal_cycle():
    """End-to-end shape: poisoned lineage -> refused -> fresh run persists
    -> anchored -> next append resumes normally."""
    cfg = {'engine_snapshot_b64': SNAP}            # legacy, no anchor
    resumed = _guarded_snapshot(cfg, 'bt', 'engine_snapshot_b64', 1)
    assert resumed is None                          # append 1: warmup-windowed
    cfg['engine_snapshot_b64'] = 'gASVnewclean=='   # persist new snapshot
    _stamp_lineage_on_persist(cfg, 'bt', resumed)
    assert _guarded_snapshot(cfg, 'bt', 'engine_snapshot_b64', 1) == 'gASVnewclean=='
