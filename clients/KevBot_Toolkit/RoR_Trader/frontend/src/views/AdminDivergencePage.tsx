'use client';

/**
 * Admin Divergence Summary page (2026-05-12).
 *
 * Surfaces the same kind of cross-strategy divergence analysis that
 * was previously a one-off script. For a chosen date window, runs the
 * 3-way joiner (backtest / algo / live) across every strategy and shows
 * counts + drift KPIs in a sortable table.
 *
 * Use case: after any code change touching the engine, writers, or
 * Hi-Fi, hit this page with "today" as the window and confirm that
 * entry drift stays at ~0s on C-type strategies and that exit drift
 * outliers don't suddenly spike. Faster than the per-strategy
 * Divergence tab when you want a fleet-wide read.
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import {
  useAdminDivergenceSummary,
  type AdminDivergenceRow,
} from '@/hooks/queries/useStrategies';

function todayStartUtcIso(): string {
  const d = new Date();
  d.setUTCHours(0, 0, 0, 0);
  return d.toISOString();
}

function nowUtcIso(): string {
  return new Date().toISOString();
}

// Drift cell color: green ≤2s, yellow ≤30s, red >30s (matches the
// existing Divergence tab convention). null/undefined → muted.
function driftColor(val: number | null | undefined): string {
  if (val == null) return 'var(--text-muted)';
  const abs = Math.abs(val);
  if (abs <= 2) return 'var(--green)';
  if (abs <= 30) return 'var(--warning, #f0a020)';
  return 'var(--red)';
}

function fmtSec(v: number | null | undefined): string {
  if (v == null) return '—';
  if (v === 0) return '0s';
  if (Math.abs(v) < 10) return `${v.toFixed(1)}s`;
  return `${Math.round(v)}s`;
}

type SortKey =
  | 'sid'
  | 'sym'
  | 'tf'
  | 'bt'
  | 'algo'
  | 'live'
  | '3w'
  | 'erc_med'
  | 'erc_max'
  | 'xcl_med'
  | 'xcl_max';

export default function AdminDivergencePage() {
  const [start, setStart] = useState<string>(() => todayStartUtcIso());
  const [end, setEnd] = useState<string>(() => nowUtcIso());
  const [query, setQuery] = useState<string>('');
  const [sortKey, setSortKey] = useState<SortKey>('sid');
  const [sortDesc, setSortDesc] = useState<boolean>(false);

  const { data, isLoading, isFetching, refetch, error } =
    useAdminDivergenceSummary(start, end, true);

  const sortedRows = useMemo(() => {
    if (!data?.rows) return [];
    const filtered = query
      ? data.rows.filter(r =>
          (r.name || '').toLowerCase().includes(query.toLowerCase())
          || (r.symbol || '').toLowerCase().includes(query.toLowerCase())
          || String(r.strategy_id).includes(query))
      : data.rows;
    const keyFn = (r: AdminDivergenceRow): number | string => {
      switch (sortKey) {
        case 'sid': return r.strategy_id;
        case 'sym': return r.symbol || '';
        case 'tf': return r.timeframe || '';
        case 'bt': return r.backtest_count ?? -1;
        case 'algo': return r.algo_count ?? -1;
        case 'live': return r.live_count ?? -1;
        case '3w': return r.matched_3way ?? -1;
        case 'erc_med': return r.entry_rc_median_s ?? -1;
        case 'erc_max': return r.entry_rc_max_s ?? -1;
        case 'xcl_med': return r.exit_cl_median_s ?? -1;
        case 'xcl_max': return r.exit_cl_max_s ?? -1;
      }
    };
    const sorted = [...filtered].sort((a, b) => {
      const ka = keyFn(a);
      const kb = keyFn(b);
      if (typeof ka === 'number' && typeof kb === 'number') return ka - kb;
      return String(ka).localeCompare(String(kb));
    });
    return sortDesc ? sorted.reverse() : sorted;
  }, [data, query, sortKey, sortDesc]);

  const onSort = (k: SortKey) => {
    if (sortKey === k) {
      setSortDesc(v => !v);
    } else {
      setSortKey(k);
      setSortDesc(['bt', 'algo', 'live', '3w', 'erc_max', 'xcl_max'].includes(k));
    }
  };

  const Th = ({ k, label, hint }: { k: SortKey; label: string; hint?: string }) => (
    <th
      onClick={() => onSort(k)}
      title={hint}
      style={{
        cursor: 'pointer',
        padding: '6px 8px',
        textAlign: 'left',
        fontSize: 12,
        whiteSpace: 'nowrap',
        userSelect: 'none',
      }}
    >
      {label}
      {sortKey === k ? (sortDesc ? ' ▼' : ' ▲') : ''}
    </th>
  );

  return (
    <div style={{ padding: 16, maxWidth: 1600, margin: '0 auto' }}>
      <div style={{ marginBottom: 16 }}>
        <h1 style={{ fontSize: 20, fontWeight: 600, marginBottom: 4 }}>
          Divergence Summary (Admin)
        </h1>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          3-way joiner (backtest / algo / live) across every strategy in the
          selected window. Drift values shown in seconds — entry REST↔CACHE
          (eRC) measures backtest-vs-algo divergence; exit CACHE↔LIVE (xCL)
          measures algo-vs-live execution divergence. Green ≤2s, yellow ≤30s,
          red &gt;30s.
        </p>
      </div>

      <Card>
        <div style={{
          display: 'flex',
          gap: 12,
          alignItems: 'flex-end',
          flexWrap: 'wrap',
          marginBottom: 12,
        }}>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>
              Start (UTC)
            </label>
            <input
              type="datetime-local"
              value={start.slice(0, 16)}
              onChange={e => setStart(e.target.value ? new Date(e.target.value + 'Z').toISOString() : start)}
              style={{ padding: '4px 6px', fontSize: 13 }}
            />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>
              End (UTC)
            </label>
            <input
              type="datetime-local"
              value={end.slice(0, 16)}
              onChange={e => setEnd(e.target.value ? new Date(e.target.value + 'Z').toISOString() : end)}
              style={{ padding: '4px 6px', fontSize: 13 }}
            />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)', marginBottom: 2 }}>
              Filter
            </label>
            <input
              type="text"
              placeholder="sid / sym / name"
              value={query}
              onChange={e => setQuery(e.target.value)}
              style={{ padding: '4px 6px', fontSize: 13, width: 160 }}
            />
          </div>
          <button
            onClick={() => refetch()}
            disabled={isLoading || isFetching}
            style={{
              padding: '6px 14px',
              fontSize: 13,
              cursor: (isLoading || isFetching) ? 'wait' : 'pointer',
            }}
          >
            {isFetching ? 'Loading…' : 'Refresh'}
          </button>
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={() => {
                setStart(todayStartUtcIso());
                setEnd(nowUtcIso());
              }}
              style={{ padding: '4px 10px', fontSize: 12 }}
            >
              Today
            </button>
            <button
              onClick={() => {
                const d = new Date();
                d.setUTCDate(d.getUTCDate() - 1);
                d.setUTCHours(0, 0, 0, 0);
                setStart(d.toISOString());
                const e = new Date();
                e.setUTCDate(e.getUTCDate() - 1);
                e.setUTCHours(23, 59, 59, 999);
                setEnd(e.toISOString());
              }}
              style={{ padding: '4px 10px', fontSize: 12 }}
            >
              Yesterday
            </button>
            <button
              onClick={() => {
                const d = new Date();
                d.setUTCDate(d.getUTCDate() - 7);
                setStart(d.toISOString());
                setEnd(nowUtcIso());
              }}
              style={{ padding: '4px 10px', fontSize: 12 }}
            >
              Last 7 days
            </button>
          </div>
        </div>

        {error && (
          <div style={{
            padding: 12,
            background: 'var(--red-bg, #fff0f0)',
            color: 'var(--red)',
            fontSize: 13,
            marginBottom: 12,
            borderRadius: 4,
          }}>
            Error loading divergence summary: {String((error as Error).message || error)}
          </div>
        )}

        {data && (
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
            {sortedRows.length} of {data.strategy_count} strategies · window {data.start} → {data.end}
          </div>
        )}

        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: 12,
          }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-subtle, #f8f8f8)' }}>
                <Th k="sid" label="sid" />
                <th style={{ padding: '6px 8px', textAlign: 'left', fontSize: 12 }}>name</th>
                <Th k="sym" label="sym" />
                <Th k="tf" label="tf" />
                <Th k="bt" label="backtest" hint="Count of backtest_% trades in window" />
                <Th k="algo" label="algo" hint="Count of cache_% trades in window" />
                <Th k="live" label="live" hint="Count of alerts in window" />
                <Th k="3w" label="3-way" hint="Trades matched across all three lanes" />
                <Th k="erc_med" label="eRC med" hint="Entry drift REST↔CACHE median (seconds)" />
                <Th k="erc_max" label="eRC max" hint="Entry drift REST↔CACHE max" />
                <Th k="xcl_med" label="xCL med" hint="Exit drift CACHE↔LIVE median" />
                <Th k="xcl_max" label="xCL max" hint="Exit drift CACHE↔LIVE max" />
                <th style={{ padding: '6px 8px', textAlign: 'left' }}></th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={13} style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>
                  Loading…
                </td></tr>
              ) : sortedRows.length === 0 ? (
                <tr><td colSpan={13} style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>
                  No strategies match the filter.
                </td></tr>
              ) : sortedRows.map(r => (
                <tr key={r.strategy_id} style={{
                  borderBottom: '1px solid var(--border-subtle, #eee)',
                  opacity: r.no_activity || r.error ? 0.6 : 1,
                }}>
                  <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{r.strategy_id}</td>
                  <td style={{ padding: '6px 8px', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.name}>
                    {r.name}
                  </td>
                  <td style={{ padding: '6px 8px' }}>{r.symbol}</td>
                  <td style={{ padding: '6px 8px' }}>{r.timeframe}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{r.backtest_count ?? '—'}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{r.algo_count ?? '—'}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{r.live_count ?? '—'}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{r.matched_3way ?? '—'}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: driftColor(r.entry_rc_median_s), fontFamily: 'monospace' }}>
                    {fmtSec(r.entry_rc_median_s)}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: driftColor(r.entry_rc_max_s), fontFamily: 'monospace' }}>
                    {fmtSec(r.entry_rc_max_s)}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: driftColor(r.exit_cl_median_s), fontFamily: 'monospace' }}>
                    {fmtSec(r.exit_cl_median_s)}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: driftColor(r.exit_cl_max_s), fontFamily: 'monospace' }}>
                    {fmtSec(r.exit_cl_max_s)}
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <Link
                      href={`/strategies/${r.strategy_id}?tab=divergence`}
                      style={{ fontSize: 11, color: 'var(--accent, #2563eb)' }}
                    >
                      open →
                    </Link>
                    {r.error && (
                      <div style={{ fontSize: 10, color: 'var(--red)', marginTop: 2 }} title={r.error}>
                        error
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
