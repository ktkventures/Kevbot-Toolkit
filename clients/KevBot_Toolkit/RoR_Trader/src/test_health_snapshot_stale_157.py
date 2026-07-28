"""Board #157 — the "0 healthy" dead alarm.

`config.engine_snapshot_at` is written only by the legacy data_worker Phase-2
flush; once the fleet moved to the M-RS5a resident/shadow engine (~2026-07-21)
that flush stopped advancing for resident strategies, so `snapshot_stale`
tripped fleet-wide even though every strategy has a FRESH shadow heartbeat
(engine provably current). The fix repoints the freshness check: when the
`RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT` kill switch is ON, a fresh heartbeat
supersedes the deprecated snapshot signal — but a genuine stall (heartbeat
also stale) still fires, and the staleness threshold is NOT widened.

Run:  cd src && ../.venv/bin/pytest test_health_snapshot_stale_157.py -v
"""
from api.routers import strategy_health as sh

_STALE = sh._STALE_SNAPSHOT_SEC   # 1h


# ── _hb_supersedes_snapshot(): env-flag parsing ─────────────────────────
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT", raising=False)
    assert sh._hb_supersedes_snapshot() is False


def test_flag_truthy_values(monkeypatch):
    for v in ("1", "true", "TRUE", "yes", "on", " On "):
        monkeypatch.setenv("RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT", v)
        assert sh._hb_supersedes_snapshot() is True, v


def test_flag_falsy_values(monkeypatch):
    for v in ("", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("RORT_HEALTH_HB_SUPERSEDES_SNAPSHOT", v)
        assert sh._hb_supersedes_snapshot() is False, v


# ── _snapshot_red_flag(): the decision truth table ──────────────────────
def _flag(snapshot_present, snapshot_age, hb_fresh, hb_supersedes):
    snapshot_at = sh.datetime.now(sh.timezone.utc) if snapshot_present else None
    return sh._snapshot_red_flag(snapshot_at, snapshot_age, hb_fresh,
                                 hb_supersedes)


def test_off_stale_snapshot_fresh_hb_still_flags():
    """Kill switch OFF ⇒ pre-fix behaviour: stale snapshot flags even with a
    fresh heartbeat (this is the dead alarm we are fixing)."""
    assert _flag(True, _STALE + 1, hb_fresh=True, hb_supersedes=False) \
        == "snapshot_stale"


def test_on_stale_snapshot_fresh_hb_suppressed():
    """THE FIX: kill switch ON + fresh heartbeat ⇒ no snapshot flag; the
    engine is provably current via shadow_heartbeats."""
    assert _flag(True, _STALE + 1, hb_fresh=True, hb_supersedes=True) is None


def test_on_stale_snapshot_stale_hb_genuine_stall_fires():
    """Genuine stall: kill switch ON but heartbeat ALSO stale ⇒ the flag
    still fires. The fix does not hide real outages."""
    assert _flag(True, _STALE + 1, hb_fresh=False, hb_supersedes=True) \
        == "snapshot_stale"


def test_on_missing_snapshot_fresh_hb_suppressed():
    """A never-snapshotted strategy (sid 339 in prod) with a fresh heartbeat
    is current, not broken ⇒ suppressed when ON."""
    assert _flag(False, None, hb_fresh=True, hb_supersedes=True) is None


def test_on_missing_snapshot_stale_hb_flags_missing():
    """Missing snapshot AND no fresh heartbeat ⇒ 'snapshot_missing'."""
    assert _flag(False, None, hb_fresh=False, hb_supersedes=True) \
        == "snapshot_missing"


def test_off_missing_snapshot_flags_missing():
    assert _flag(False, None, hb_fresh=True, hb_supersedes=False) \
        == "snapshot_missing"


def test_fresh_snapshot_never_flags():
    """A recently-flushed snapshot (age < threshold) never flags, regardless
    of heartbeat / switch state."""
    for supersedes in (True, False):
        for hb in (True, False):
            assert _flag(True, _STALE - 1, hb_fresh=hb,
                         hb_supersedes=supersedes) is None


def test_threshold_not_widened_boundary():
    """Boundary: age exactly at the threshold does NOT flag (strict >), age
    one second past DOES. Proves the fix repoints — it does not widen."""
    assert _flag(True, _STALE, hb_fresh=False, hb_supersedes=True) is None
    assert _flag(True, _STALE + 1, hb_fresh=False, hb_supersedes=True) \
        == "snapshot_stale"


def test_module_imports_os():
    """Regression: the flag read needs `os` imported at module scope
    (the env-flag tests above would NameError otherwise)."""
    assert hasattr(sh, "os")
