/**
 * Resampled Store — M-RS2 Phase 2 §8 observability (read-only).
 *
 * The canonical session-keyed coarse-bar store (`resampled_bar_cache`,
 * 2Min..1Day resampled from cached 1Min). Both lanes are meant to read the
 * SAME completed coarse gate bar from here, so this page is where drift /
 * staleness is visible before it bites. Per (symbol, tf, session): coverage
 * span, row count, freshness (last revised_at — badge STALE if >2h old during
 * market hours), settled %. Talks to /api/admin/resampled-store. All hooks
 * before any early return (Terser rules). API snake_case → camelCase mapped
 * at the fetch boundary.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';

// ---- API (snake_case) ----
interface CoverageRowApi {
  symbol: string;
  tf_seconds: number;
  tf_label: string;
  session: string;
  first_ts: string | null;
  last_ts: string | null;
  row_count: number;
  last_revised_at: string | null;
  settled_pct: number | null;
}
interface CoverageRespApi {
  write_enabled: boolean;
  read_enabled: boolean;
  compare_enabled: boolean;
  serve_enabled: boolean;
  direct_pg_available: boolean;
  coverage: CoverageRowApi[];
  detail: string | null;
}

// ---- View (camelCase) ----
interface CoverageRow {
  symbol: string;
  tfSeconds: number;
  tfLabel: string;
  session: string;
  firstTs: string | null;
  lastTs: string | null;
  rowCount: number;
  lastRevisedAt: string | null;
  settledPct: number | null;
}
interface CoverageResp {
  writeEnabled: boolean;
  readEnabled: boolean;
  compareEnabled: boolean;
  serveEnabled: boolean;
  directPgAvailable: boolean;
  coverage: CoverageRow[];
  detail: string | null;
}

function mapResp(r: CoverageRespApi): CoverageResp {
  return {
    writeEnabled: r.write_enabled,
    readEnabled: r.read_enabled,
    compareEnabled: r.compare_enabled,
    serveEnabled: r.serve_enabled,
    directPgAvailable: r.direct_pg_available,
    detail: r.detail,
    coverage: (r.coverage ?? []).map((c) => ({
      symbol: c.symbol,
      tfSeconds: c.tf_seconds,
      tfLabel: c.tf_label,
      session: c.session,
      firstTs: c.first_ts,
      lastTs: c.last_ts,
      rowCount: c.row_count,
      lastRevisedAt: c.last_revised_at,
      settledPct: c.settled_pct,
    })),
  };
}

const cell: React.CSSProperties = { padding: '7px 8px', verticalAlign: 'middle', fontSize: 13 };
const pill = (bg: string): React.CSSProperties => ({
  display: 'inline-block', padding: '2px 9px', borderRadius: 11, fontSize: 12,
  fontWeight: 600, background: bg, color: '#fff',
});

function fmtTs(ts: string | null): string {
  if (!ts) return '—';
  return ts.slice(0, 19).replace('T', ' ');
}
function fmtDate(ts: string | null): string {
  return ts ? ts.slice(0, 10) : '—';
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
// Coverage span in days between firstTs and lastTs.
function spanDays(minTs: string | null, maxTs: string | null): number | null {
  if (!minTs || !maxTs) return null;
  const a = new Date(minTs.endsWith('Z') || minTs.includes('+') ? minTs : minTs + 'Z').getTime();
  const b = new Date(maxTs.endsWith('Z') || maxTs.includes('+') ? maxTs : maxTs + 'Z').getTime();
  if (Number.isNaN(a) || Number.isNaN(b)) return null;
  return Math.max(0, (b - a) / 86400000);
}
// Rough US-equity RTH check (Mon–Fri 09:30–16:00 ET) — enough for the stale
// badge; holidays read as "market hours" which only makes the badge stricter.
function isMarketHoursET(now: Date): boolean {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', weekday: 'short',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(now);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? '';
  const wd = get('weekday');
  if (wd === 'Sat' || wd === 'Sun') return false;
  const mins = parseInt(get('hour'), 10) * 60 + parseInt(get('minute'), 10);
  return mins >= 9 * 60 + 30 && mins < 16 * 60;
}

const STALE_S = 2 * 3600; // maintain runs on the sweeper cadence — >2h old during RTH = stuck

export default function ResampledStoreAdminPage() {
  const [resp, setResp] = useState<CoverageResp | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setErr(null);
      const data = await apiFetch<CoverageRespApi>('/api/admin/resampled-store/coverage');
      setResp(mapResp(data));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh every 30s — the store is maintained on the sweeper cadence
  // (minutes), so a live-ticking page doesn't need the 4s bar-cache poll.
  useEffect(() => {
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, [load]);

  // Group rows by symbol → (tf, session) rows, tf-major order (API is already
  // sorted symbol, tf_seconds, session; grouping keys the section headers).
  const bySymbol = useMemo(() => {
    const groups: { symbol: string; rows: CoverageRow[] }[] = [];
    for (const row of resp?.coverage ?? []) {
      const g = groups[groups.length - 1];
      if (g && g.symbol === row.symbol) g.rows.push(row);
      else groups.push({ symbol: row.symbol, rows: [row] });
    }
    return groups;
  }, [resp]);

  const marketHours = useMemo(() => isMarketHoursET(new Date()), [resp]); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading && !resp) {
    return <Card><div style={{ padding: 20 }}>Loading resampled store…</div></Card>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: 16 }}>
      <Card>
        <div style={{ padding: 16 }}>
          <h2 style={{ margin: '0 0 4px', fontSize: 18 }}>Resampled Store</h2>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 13, marginBottom: 12 }}>
            Canonical session-keyed coarse bars (<code>resampled_bar_cache</code>, 2Min..1Day resampled
            from cached 1Min). Gate-TF bars both lanes are meant to read — per (symbol, TF, session):
            coverage span, freshness (last re-resample), and settled %. Read-only view; writes happen on
            the data-worker&apos;s maintain cadence. Base layers (1Sec/1Min REST + WS) are on the{' '}
            <a href="/admin/bar-cache" style={{ color: 'var(--blue)' }}>Bar Cache page</a>.
            ⚠ <b>Floor = 2Min:</b> sub-minute TFs (10s/30s) have no canonical store yet — each lane
            derives its own from 1Sec/WS, the root of the 339-class drift
            (<code>Plan_SubMinute_Canonical_Source.md</code>).
          </div>
          <div style={{ display: 'flex', gap: 10, marginBottom: 4, flexWrap: 'wrap' }}>
            <span style={pill(resp?.writeEnabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              write (RORT_RESAMPLED_STORE_WRITE): {resp?.writeEnabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.readEnabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              read (RORT_RESAMPLED_STORE_READ): {resp?.readEnabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.compareEnabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              comparator (RORT_RESAMPLED_STORE_COMPARE): {resp?.compareEnabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.serveEnabled ? 'var(--green)' : 'var(--text-tertiary)')}>
              compute-skip (RORT_RESAMPLED_STORE_SERVE): {resp?.serveEnabled ? 'ON' : 'off'}
            </span>
            <span style={pill(resp?.directPgAvailable ? 'var(--green)' : 'var(--red)')}>
              direct-PG: {resp?.directPgAvailable ? 'connected' : 'MISSING'}
            </span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 6 }}>
            NOTE: flag pills reflect the API service&apos;s env; the writer is the data-worker —
            fresh <b>Last revised</b> values are the real proof it&apos;s maintaining.
          </div>
          {resp?.detail && (
            <div style={{ fontSize: 12, color: 'var(--amber, #d98c00)', marginTop: 6 }}>{resp.detail}</div>
          )}
          {err && <div style={{ color: 'var(--red)', fontSize: 13, marginTop: 10 }}>{err}</div>}
        </div>
      </Card>

      <Card>
        <div style={{ padding: 16 }}>
          <h3 style={{ margin: '0 0 4px', fontSize: 15 }}>Coverage &amp; Freshness</h3>
          <div style={{ color: 'var(--text-tertiary)', fontSize: 12, marginBottom: 10 }}>
            <b>Coverage</b> = history held per (TF, session). <b>Last revised</b> = newest
            re-resample write; badge <b style={{ color: 'var(--red)' }}>STALE</b> when it&apos;s
            &gt;2h old during market hours (maintain runs on the sweeper cadence), gray <b>quiet</b> when
            off-hours. <b>Settled %</b> = share of rows past the settle window (permanent truth vs
            provisional tail).
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: 'var(--text-tertiary)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
                <th style={cell}>TF</th>
                <th style={cell}>Session</th>
                <th style={cell}>Coverage (UTC dates)</th>
                <th style={cell}>Span</th>
                <th style={cell}>Rows</th>
                <th style={cell}>Last revised (UTC)</th>
                <th style={cell}>Age</th>
                <th style={cell}>Freshness</th>
                <th style={cell}>Settled %</th>
              </tr>
            </thead>
            <tbody>
              {bySymbol.length === 0 && (
                <tr><td style={{ ...cell, color: 'var(--text-tertiary)' }} colSpan={9}>
                  Store is empty (seed/backfill has not run, or direct-PG is unavailable).
                </td></tr>
              )}
              {bySymbol.map((g) => (
                <React.Fragment key={g.symbol}>
                  <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-input)' }}>
                    <td style={{ ...cell, fontWeight: 700 }} colSpan={9}>
                      {g.symbol}
                      <span style={{ color: 'var(--text-tertiary)', fontWeight: 400, fontSize: 12, marginLeft: 8 }}>
                        {g.rows.length} (TF × session) layers · {fmtNum(g.rows.reduce((a, r) => a + r.rowCount, 0))} rows
                      </span>
                    </td>
                  </tr>
                  {g.rows.map((r) => {
                    const lag = ageSec(r.lastRevisedAt);
                    const sd = spanDays(r.firstTs, r.lastTs);
                    const missing = !r.lastTs;
                    const stale = !missing && lag != null && lag > STALE_S && marketHours;
                    const status = missing
                      ? { label: 'NO DATA', bg: 'var(--red)' }
                      : stale
                        ? { label: 'STALE', bg: 'var(--red)' }
                        : (lag != null && lag <= STALE_S
                          ? { label: 'fresh', bg: 'var(--green)' }
                          : { label: 'quiet', bg: 'var(--text-tertiary)' });
                    return (
                      <tr key={`${r.symbol}/${r.tfSeconds}/${r.session}`} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ ...cell, fontWeight: 600 }}>{r.tfLabel}</td>
                        <td style={cell}>{r.session}</td>
                        <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>
                          {fmtDate(r.firstTs)} → {fmtDate(r.lastTs)}
                        </td>
                        <td style={{ ...cell, fontSize: 12 }}>{sd == null ? '—' : `${Math.round(sd)}d`}</td>
                        <td style={cell}>{fmtNum(r.rowCount)}</td>
                        <td style={{ ...cell, fontSize: 12, color: 'var(--text-tertiary)' }}>{fmtTs(r.lastRevisedAt)}</td>
                        <td style={{ ...cell, fontWeight: 600, color: stale ? 'var(--red)' : 'var(--text-tertiary)' }}>
                          {fmtAge(lag)}
                        </td>
                        <td style={cell}><span style={pill(status.bg)}>{status.label}</span></td>
                        <td style={cell}>
                          {r.settledPct == null ? '—' : `${(r.settledPct * 100).toFixed(1)}%`}
                        </td>
                      </tr>
                    );
                  })}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
