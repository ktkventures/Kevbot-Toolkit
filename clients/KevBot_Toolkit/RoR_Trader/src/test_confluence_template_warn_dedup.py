"""Tests for the confluence-group missing-template warning dedup
(board #140 / #9). A group referencing a removed pack (rsi_zones /
rsi_zones_3) is correctly SKIPPED, but the warning must fire once per
missing template per process — not once per config reload. Run from src/:
    python -m pytest test_confluence_template_warn_dedup.py -q
"""
import sys

sys.path.insert(0, '.')

import confluence_groups as cg  # noqa: E402


def _group(gid, base_template):
    return {"id": gid, "base_template": base_template, "version": "Default"}


def test_missing_template_warns_once_across_reloads(capsys):
    cg._WARNED_MISSING_TEMPLATES.clear()
    raw = [_group("g1", "rsi_zones"), _group("g2", "rsi_zones")]

    # Three "reloads" of the same removed-pack group.
    for _ in range(3):
        groups = cg._parse_group_list(raw)

    # The removed-template groups are dropped (only migration defaults remain).
    assert all(g.base_template != "rsi_zones" for g in groups)

    out = capsys.readouterr().out
    warn_lines = [ln for ln in out.splitlines()
                  if "rsi_zones" in ln and "not found" in ln]
    assert len(warn_lines) == 1, f"expected one warning, got: {warn_lines}"
    assert "suppressed" in warn_lines[0]


def test_distinct_missing_templates_each_warn_once(capsys):
    cg._WARNED_MISSING_TEMPLATES.clear()
    raw = [_group("a", "rsi_zones"), _group("b", "rsi_zones_3"),
           _group("c", "rsi_zones"), _group("d", "rsi_zones_3")]
    cg._parse_group_list(raw)
    cg._parse_group_list(raw)  # reload

    out = capsys.readouterr().out
    warned = [ln for ln in out.splitlines() if "not found" in ln]
    assert len(warned) == 2  # one per distinct missing template
    assert any("rsi_zones'" in ln for ln in warned)
    assert any("rsi_zones_3'" in ln for ln in warned)


def test_known_template_produces_no_warning(capsys):
    cg._WARNED_MISSING_TEMPLATES.clear()
    # 'bar_count' is a real registered template — no warning expected.
    assert "bar_count" in cg.TEMPLATES
    cg._parse_group_list([_group("bc", "bar_count")])
    out = capsys.readouterr().out
    assert "not found" not in out
