/**
 * Bar-Cache Supply — the one source-of-truth control surface.
 *
 * Catalog-driven: rows are every (symbol, timeframe) any strategy uses (all
 * accounts) — NOT a manually-managed target list. Per row: coverage span vs
 * target depth, live freshness (write-through), unsettled tail, and a
 * backfill-to-a-start-date control with status. Strategies read bar_cache as
 * their primary source, so this page is where we confirm "one fresh source +
 * enough history". Talks to /api/bar-cache. All hooks before any early return
 * (Terser rules).
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';

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
  backfill_status?: string | null;
  target_days?: number | null;
}
interface TargetsResp {
  read_enabled: boolean;
  maintain_cron_enabled: boolean;
  writethrough_enabled?: boolean;
  supply_coverage?: SupplyCoverage[] | null;
  read_health?: ReadHealth;
}

const cell: React.CSSProperties = { padding: '7px 8px', verticalAlign: 'middle', fontSize: 13 };
const input: React.CSSProperties = {
  background: 'var(--bg-input)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 6, padding: '4px 6px', fontSize: 12,
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
  // Per-row backfill start date (YYYY-MM-DD); defaults to 1y ago when unset.
  const [starts, setStarts] = useState<Record<string, string>>({});

  // Default backfill floor = 1 year back, on BOTH 1Sec and 1Min (deep 1Sec is
  // wanted for hi-fi backtesting). Stable across renders.
  const defaultStart = useMemo(
    () => new Date(Date.now() - 365 * 86400000).toISOString().slice(0, 10), []);

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

  // Auto-refresh every 4s so backfill progress + freshness stay live.
  useEffect(() => {
    load();
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, [load]);

  const backfill = useCallback(async (sym: string, tf: string, startDate: string) => {
    setBusy(`bf:${sym}/${tf}`);
    try {
      await apiFetch('/api/bar-cache/backfill', {
        method: 'POST',
        body: JSON.stringify({ symbol: sym, timeframe: tf, start_date: startDate }),
      });
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }, [load]);

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
    return <Card><div style={{ padding: 20 }}>Loading bar-cache supply…</div></Card>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: 16 }}>
      <Card>
        <div style={{ padding: 16 }}>
          <h2 style={{ margin: '0 0 4px', fontSize: 18 }}>Bar-Cache Supply</h2>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 13, marginBottom: 12 }}>
            One continuously-fresh source of truth. The data-worker streams native Polygon bars (1Sec + 1Min)
            write-through into <code>bar_cache</code>; strategies read it as their primary source. Symbols come
            from the catalog (every strategy, all accounts) — set a start date per layer to backfill its history.
          </div>
          <div style={{ display: 'flex', gap: 10, marginBottom: 4, flexWrap: 'wrap' }}>
            <span style={pill(streamActive ? 'var(--green)' : 'var(--red)')}>
              write-through stream: {streamActive ? 'ACTIVE (writing)' : 'quiet / stale'}
            </span>
            <span style={pill(resp?.read_enabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              read path (BAR_CACHE_ENABLED): {resp?.read_enabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.maintain_cron_enabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              maintain cron: {resp?.maintain_cron_enabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.read_health?.direct_pg_available ? 'var(--green)' : 'var(--red)')}>
              direct-PG: {resp?.read_health?.direct_pg_available ? 'connected' : 'MISSING'}
            </span>
            <span style={pill(resp?.read_health?.sample_read_ok ? 'var(--green)'
              : resp?.read_health?.sample_read_ok === false ? 'var(--red)' : 'var(--text-tertiary)')}>
              live sample read: {resp?.read_health?.sample_read_ok === true ? 'OK'
                : resp?.read_health?.sample_read_ok === false ? 'FAILED' : 'n/a'}
            </span>
          </div>
          {resp?.read_health?.detail && (
            <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 6 }}>
              read health (api service): {resp.read_health.detail}
            </div>
          )}
          {err && <div style={{ color: 'var(--red)', fontSize: 13, marginTop: 10 }}>{err}</div>}
        </div>
      </Card>

      {/* Supply coverage + freshness — every catalog symbol × source layer, with a
          per-row backfill-to-start-date control. */}
      <Card>
        <div style={{ padding: 16 }}>
          <h3 style={{ margin: '0 0 4px', fontSize: 15 }}>Supply Coverage &amp; Freshness</h3>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 12, marginBottom: 10 }}>
            <b>Coverage</b> = history held; <b>span/target</b> shows it against the backfill floor (amber = under
            target). <b>Lag</b> = freshness (green ≤2 min = streaming; gray = quiet, normal after-hours).
            <b> Unsettled</b> = provisional tail (settles ~16 min). <b style={{ color: 'var(--red)' }}>NO DATA</b> =
            a strategy needs this symbol but the cache is empty. Set a start date and <b>Backfill</b> to fill
            history (default 1 year, on both 1Sec and 1Min).
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: 'var(--text-tertiary)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
                <th style={cell}>Symbol</th>
                <th style={cell}>TF</th>
                <th style={cell}>Coverage (UTC dates)</th>
                <th style={cell}>Span / target</th>
                <th style={cell}>Last write</th>
                <th style={cell}>Lag</th>
                <th style={cell}>Unsettled</th>
                <th style={cell}>Status</th>
                <th style={cell}>Backfill from</th>
                <th style={cell}></th>
              </tr>
            </thead>
            <tbody>
              {coverage.length === 0 && (
                <tr><td style={{ ...cell, color: 'var(--text-tertiary)' }} colSpan={10}>No tracked symbols.</td></tr>
              )}
              {coverage.map((f) => {
                const key = `${f.symbol}/${f.timeframe}`;
                const lag = ageSec(f.last_revised);
                const sd = spanDays(f.min_ts, f.max_ts);
                const tgt = f.target_days ?? null;
                const under = sd != null && tgt != null && sd < tgt * 0.9;
                const missing = !f.max_ts;
                const status = missing
                  ? { label: 'NO DATA', bg: 'var(--red)' }
                  : (lag != null && lag <= 120
                    ? { label: 'streaming', bg: 'var(--green)' }
                    : { label: 'quiet', bg: 'var(--text-tertiary)' });
                const running = f.backfill_status === 'running';
                const startVal = starts[key] ?? defaultStart;
                return (
                  <tr key={key} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ ...cell, fontWeight: 600 }}>{f.symbol}</td>
                    <td style={cell}>{f.timeframe}</td>
                    <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>
                      {fmtDate(f.min_ts)} → {fmtDate(f.max_ts)}
                    </td>
                    <td style={{ ...cell, fontSize: 12, color: under ? 'var(--amber, #d98c00)' : undefined }}>
                      {sd == null ? '—' : `${Math.round(sd)}d`}{tgt != null ? ` / ${tgt}d` : ''}
                    </td>
                    <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>{fmtTs(f.last_revised)}</td>
                    <td style={{ ...cell, fontWeight: 600, color: lagColor(lag) }}>{fmtAge(lag)}</td>
                    <td style={cell}>{fmtNum(f.unsettled)}</td>
                    <td style={cell}><span style={pill(status.bg)}>{status.label}</span></td>
                    <td style={cell}>
                      <input
                        type="date"
                        style={{ ...input, width: 130 }}
                        value={startVal}
                        max={new Date(Date.now() - 86400000).toISOString().slice(0, 10)}
                        onChange={(e) => setStarts((s) => ({ ...s, [key]: e.target.value }))}
                      />
                    </td>
                    <td style={cell}>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <button
                          style={btn(running ? 'var(--text-tertiary)' : 'var(--blue)')}
                          disabled={running || busy === `bf:${key}`}
                          onClick={() => backfill(f.symbol, f.timeframe, startVal)}
                        >
                          {busy === `bf:${key}` ? '…' : 'Backfill'}
                        </button>
                        {running
                          ? <span style={pill('var(--amber, #d98c00)')}>running…</span>
                          : (f.backfill_status
                            ? <span style={{ fontSize: 12, color: f.backfill_status.startsWith('error') ? 'var(--red)' : 'var(--text-tertiary)' }}>
                              {f.backfill_status}
                            </span>
                            : null)}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
