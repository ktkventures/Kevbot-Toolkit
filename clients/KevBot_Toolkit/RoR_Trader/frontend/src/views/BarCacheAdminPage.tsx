/**
 * Bar-Cache Admin — M-RS2 Phase 1 control surface.
 *
 * Manage which (symbol, timeframe) native layers we capture + keep updated.
 * Per-target: row count, coverage span, last backfill / maintain. Actions:
 * add target, enable/disable, backfill-now, maintain-all, delete.
 * Capture resolutions are native-fetchable (1Sec..1Min); coarse TFs are
 * materialized from 1Min in Phase 2 (serving), not captured here.
 * Talks to /api/bar-cache. All hooks before any early return (Terser rules).
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';

interface Target {
  symbol: string;
  timeframe: string;
  enabled: boolean;
  capture_days: number | null;
  last_backfill_at: string | null;
  last_maintained_at: string | null;
  rows: number | null;
  min_ts: string | null;
  max_ts: string | null;
  backfill_status: string | null;
}
interface ReadHealth {
  read_enabled: boolean;
  direct_pg_available: boolean;
  sample_read_ok: boolean | null;
  detail: string | null;
}
interface SupplyCoverage {
  symbol: string;
  timeframe: string;
  min_ts: string | null;
  max_ts: string | null;
  last_revised: string | null;
  unsettled: number;
}
interface TargetsResp {
  targets: Target[];
  resolutions: string[];
  read_enabled: boolean;
  maintain_cron_enabled: boolean;
  writethrough_enabled?: boolean;
  supply_coverage?: SupplyCoverage[] | null;
  read_health?: ReadHealth;
}

const cell: React.CSSProperties = { padding: '7px 8px', verticalAlign: 'middle', fontSize: 13 };
const input: React.CSSProperties = {
  background: 'var(--bg-input)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 6, padding: '5px 7px', fontSize: 13,
};
const btn = (bg: string): React.CSSProperties => ({
  background: bg, color: '#fff', border: 'none', borderRadius: 6,
  padding: '4px 10px', fontSize: 12, fontWeight: 600, cursor: 'pointer',
});
const pill = (bg: string): React.CSSProperties => ({
  display: 'inline-block', padding: '2px 9px', borderRadius: 11, fontSize: 12,
  fontWeight: 600, background: bg, color: '#fff',
});

function fmtTs(ts: string | null): string {
  if (!ts) return '—';
  return ts.slice(0, 19).replace('T', ' ');
}
function fmtNum(n: number | null): string {
  return n == null ? '—' : n.toLocaleString();
}
// Age (now − ts) in seconds, or null if no ts.
function ageSec(ts: string | null): number | null {
  if (!ts) return null;
  const t = new Date(ts.endsWith('Z') || ts.includes('+') ? ts : ts + 'Z').getTime();
  if (Number.isNaN(t)) return null;
  return Math.max(0, (Date.now() - t) / 1000);
}
function fmtAge(sec: number | null): string {
  if (sec == null) return '—';
  if (sec < 90) return `${Math.round(sec)}s`;
  if (sec < 5400) return `${Math.round(sec / 60)}m`;
  return `${(sec / 3600).toFixed(1)}h`;
}
// Lag color for the live edge: green ≤2min = actively streaming; gray older =
// quiet/settled (normal after-hours), NOT an error. Missing data is flagged
// separately (the real red signal), so we don't false-alarm on overnight quiet.
function lagColor(sec: number | null): string {
  if (sec == null) return 'var(--text-tertiary)';
  return sec <= 120 ? 'var(--green)' : 'var(--text-tertiary)';
}
// Coverage span in days between min_ts and max_ts.
function spanDays(minTs: string | null, maxTs: string | null): number | null {
  if (!minTs || !maxTs) return null;
  const a = new Date(minTs.endsWith('Z') || minTs.includes('+') ? minTs : minTs + 'Z').getTime();
  const b = new Date(maxTs.endsWith('Z') || maxTs.includes('+') ? maxTs : maxTs + 'Z').getTime();
  if (Number.isNaN(a) || Number.isNaN(b)) return null;
  return Math.max(0, (b - a) / 86400000);
}
function fmtDate(ts: string | null): string {
  return ts ? ts.slice(0, 10) : '—';
}

export default function BarCacheAdminPage() {
  const [resp, setResp] = useState<TargetsResp | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [nt, setNt] = useState({ symbol: '', timeframe: '1Sec', capture_days: 365 });

  const load = useCallback(async () => {
    try {
      setErr(null);
      const data = await apiFetch<TargetsResp>('/api/bar-cache/targets');
      setResp(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh every 4s so backfill progress + row counts stay live.
  useEffect(() => {
    load();
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, [load]);

  const addTarget = useCallback(async () => {
    const sym = nt.symbol.trim().toUpperCase();
    if (!sym) return;
    setBusy('add');
    try {
      await apiFetch('/api/bar-cache/targets', {
        method: 'POST',
        body: JSON.stringify({ symbol: sym, timeframe: nt.timeframe, capture_days: nt.capture_days, enabled: true }),
      });
      setNt({ ...nt, symbol: '' });
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [nt, load]);

  const toggleEnabled = useCallback(async (t: Target) => {
    await apiFetch('/api/bar-cache/targets', {
      method: 'POST',
      body: JSON.stringify({ symbol: t.symbol, timeframe: t.timeframe, enabled: !t.enabled, capture_days: t.capture_days }),
    });
    await load();
  }, [load]);

  const backfill = useCallback(async (t: Target) => {
    setBusy(`bf:${t.symbol}/${t.timeframe}`);
    try {
      await apiFetch('/api/bar-cache/backfill', {
        method: 'POST',
        body: JSON.stringify({ symbol: t.symbol, timeframe: t.timeframe, days: t.capture_days || 365 }),
      });
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [load]);

  const removeTarget = useCallback(async (t: Target) => {
    if (!confirm(`Remove capture target ${t.symbol}/${t.timeframe}? (cached bars are NOT deleted)`)) return;
    await apiFetch(`/api/bar-cache/targets/${t.symbol}/${t.timeframe}`, { method: 'DELETE' });
    await load();
  }, [load]);

  const maintainAll = useCallback(async () => {
    setBusy('maintain');
    try {
      await apiFetch('/api/bar-cache/maintain', { method: 'POST', body: JSON.stringify({}) });
      await load();
    } finally {
      setBusy(null);
    }
  }, [load]);

  const resolutions = useMemo(() => resp?.resolutions ?? ['1Sec', '1Min'], [resp]);
  const targets = useMemo(() => resp?.targets ?? [], [resp]);
  const coverage = useMemo(() => resp?.supply_coverage ?? [], [resp]);
  // The stream is "active" if any layer was written within the last 2 minutes —
  // the real proof write-through is running (independent of which service holds
  // the RORT_BARCACHE_WRITETHROUGH flag).
  const streamActive = useMemo(() => {
    const ages = (resp?.supply_coverage ?? [])
      .map((f) => ageSec(f.last_revised))
      .filter((s): s is number => s != null);
    return ages.length > 0 && Math.min(...ages) <= 120;
  }, [resp]);

  if (loading && !resp) {
    return <Card><div style={{ padding: 20 }}>Loading bar-cache config…</div></Card>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: 16 }}>
      <Card>
        <div style={{ padding: 16 }}>
          <h2 style={{ margin: '0 0 4px', fontSize: 18 }}>Bar-Cache Supply</h2>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 13, marginBottom: 12 }}>
            One continuously-fresh source of truth. The data-worker streams native Polygon bars (1Sec + 1Min)
            write-through into <code>bar_cache</code>; all consumers (backtests, charts, recompute) read it.
            The live stream below is the freshness proof; the targets table is the historical-depth registry.
          </div>
          <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
            <span style={pill(streamActive ? 'var(--green)' : 'var(--red)')}>
              write-through stream: {streamActive ? 'ACTIVE (writing)' : 'quiet / stale'}
            </span>
            <span style={pill(resp?.read_enabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              read path (BAR_CACHE_ENABLED): {resp?.read_enabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.maintain_cron_enabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              maintain cron (BAR_CACHE_MAINTAIN_ENABLED): {resp?.maintain_cron_enabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.read_health?.direct_pg_available ? 'var(--green)' : 'var(--red)')}>
              direct-PG (SUPABASE_CONNECTION_STRING): {resp?.read_health?.direct_pg_available ? 'connected' : 'MISSING'}
            </span>
            <span style={pill(resp?.read_health?.sample_read_ok ? 'var(--green)'
              : resp?.read_health?.sample_read_ok === false ? 'var(--red)' : 'var(--text-tertiary)')}>
              live sample read: {resp?.read_health?.sample_read_ok === true ? 'OK'
                : resp?.read_health?.sample_read_ok === false ? 'FAILED' : 'n/a'}
            </span>
          </div>
          {resp?.read_health?.detail && (
            <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: -6, marginBottom: 12 }}>
              read health (api service): {resp.read_health.detail}
              {' · '}worker env confirmed separately via Update-All logs
            </div>
          )}
          {err && <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 10 }}>{err}</div>}

          {/* Add target */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <input
              style={{ ...input, width: 110, textTransform: 'uppercase' }}
              placeholder="SYMBOL"
              value={nt.symbol}
              onChange={(e) => setNt({ ...nt, symbol: e.target.value })}
              onKeyDown={(e) => { if (e.key === 'Enter') addTarget(); }}
            />
            <select style={input} value={nt.timeframe} onChange={(e) => setNt({ ...nt, timeframe: e.target.value })}>
              {resolutions.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            <input
              style={{ ...input, width: 90 }}
              type="number"
              value={nt.capture_days}
              onChange={(e) => setNt({ ...nt, capture_days: Number(e.target.value) })}
            />
            <span style={{ color: 'var(--text-tertiary)', fontSize: 12 }}>days back</span>
            <button style={btn('var(--blue)')} disabled={busy === 'add'} onClick={addTarget}>
              {busy === 'add' ? 'Adding…' : '+ Add target'}
            </button>
            <button style={btn('var(--amber, #d98c00)')} disabled={busy === 'maintain'} onClick={maintainAll}>
              {busy === 'maintain' ? 'Maintaining…' : 'Maintain all now'}
            </button>
          </div>
        </div>
      </Card>

      {/* Supply coverage + freshness — every tracked symbol (all accounts), not
          just whatever is printing this second. */}
      <Card>
        <div style={{ padding: 16 }}>
          <h3 style={{ margin: '0 0 4px', fontSize: 15 }}>Supply Coverage &amp; Freshness</h3>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 12, marginBottom: 10 }}>
            Every symbol any strategy uses (all accounts) × source layer. <b>Coverage</b> = how much history
            we hold; <b>last write</b>/<b>lag</b> = freshness (green ≤2 min = streaming now; gray = quiet, normal
            after-hours); <b>unsettled</b> = provisional tail (settles after ~16 min). <b style={{ color: 'var(--red)' }}>NO
            DATA</b> = a strategy needs this symbol but the cache is empty (backfill gap).
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: 'var(--text-tertiary)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
                <th style={cell}>Symbol</th>
                <th style={cell}>TF</th>
                <th style={cell}>Coverage (UTC dates)</th>
                <th style={cell}>Span</th>
                <th style={cell}>Last write (UTC)</th>
                <th style={cell}>Lag</th>
                <th style={cell}>Unsettled</th>
                <th style={cell}>Status</th>
              </tr>
            </thead>
            <tbody>
              {coverage.length === 0 && (
                <tr><td style={{ ...cell, color: 'var(--text-tertiary)' }} colSpan={8}>No tracked symbols.</td></tr>
              )}
              {coverage.map((f) => {
                const lag = ageSec(f.last_revised);
                const sd = spanDays(f.min_ts, f.max_ts);
                const missing = !f.max_ts;
                const status = missing
                  ? { label: 'NO DATA', bg: 'var(--red)' }
                  : (lag != null && lag <= 120
                    ? { label: 'streaming', bg: 'var(--green)' }
                    : { label: 'quiet', bg: 'var(--text-tertiary)' });
                return (
                  <tr key={`${f.symbol}/${f.timeframe}`} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ ...cell, fontWeight: 600 }}>{f.symbol}</td>
                    <td style={cell}>{f.timeframe}</td>
                    <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>
                      {fmtDate(f.min_ts)} → {fmtDate(f.max_ts)}
                    </td>
                    <td style={{ ...cell, fontSize: 12 }}>{sd == null ? '—' : `${Math.round(sd)}d`}</td>
                    <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>{fmtTs(f.last_revised)}</td>
                    <td style={{ ...cell, fontWeight: 600, color: lagColor(lag) }}>{fmtAge(lag)}</td>
                    <td style={cell}>{fmtNum(f.unsettled)}</td>
                    <td style={cell}><span style={pill(status.bg)}>{status.label}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', color: 'var(--text-tertiary)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
              <th style={cell}>Symbol</th>
              <th style={cell}>TF</th>
              <th style={cell}>On</th>
              <th style={cell}>Days</th>
              <th style={cell}>Rows</th>
              <th style={cell}>Coverage (UTC)</th>
              <th style={cell}>Last backfill</th>
              <th style={cell}>Last maintain</th>
              <th style={cell}>Backfill status</th>
              <th style={cell}></th>
            </tr>
          </thead>
          <tbody>
            {targets.length === 0 && (
              <tr><td style={{ ...cell, color: 'var(--text-tertiary)' }} colSpan={10}>No capture targets yet — add one above.</td></tr>
            )}
            {targets.map((t) => (
              <tr key={`${t.symbol}/${t.timeframe}`} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ ...cell, fontWeight: 600 }}>{t.symbol}</td>
                <td style={cell}>{t.timeframe}</td>
                <td style={cell}>
                  <button style={btn(t.enabled ? 'var(--green)' : 'var(--text-tertiary)')} onClick={() => toggleEnabled(t)}>
                    {t.enabled ? 'on' : 'off'}
                  </button>
                </td>
                <td style={cell}>{t.capture_days ?? '—'}</td>
                <td style={cell}>{fmtNum(t.rows)}</td>
                <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>
                  {fmtTs(t.min_ts)} → {fmtTs(t.max_ts)}
                </td>
                <td style={{ ...cell, fontSize: 12 }}>{fmtTs(t.last_backfill_at)}</td>
                <td style={{ ...cell, fontSize: 12 }}>{fmtTs(t.last_maintained_at)}</td>
                <td style={{ ...cell, fontSize: 12 }}>
                  {t.backfill_status === 'running'
                    ? <span style={pill('var(--amber, #d98c00)')}>running…</span>
                    : (t.backfill_status || '—')}
                </td>
                <td style={cell}>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      style={btn('var(--blue)')}
                      disabled={busy === `bf:${t.symbol}/${t.timeframe}` || t.backfill_status === 'running'}
                      onClick={() => backfill(t)}
                    >
                      Backfill
                    </button>
                    <button style={btn('var(--red)')} onClick={() => removeTarget(t)}>✕</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
