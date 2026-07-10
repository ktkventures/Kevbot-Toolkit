'use client';

/**
 * Strategy Health V5 — Model Parity tab (W2-6c, 2026-07-07).
 *
 * One ROW per strategy, percentages only:
 *
 *   1. Model columns FIRST — backtest_model / algo_model / live_model as
 *      RUNTIME-EFFECTIVE values. Badges: `explicit` (green) when set in
 *      config; `unset→<default>` (amber) showing what the runtime
 *      actually resolves to; red when the live id is invalid or the algo
 *      lane is fully unset (the runtime tripwire raises — PR #26; the
 *      old "echo the admin default" display hid that for 3 weeks,
 *      ledger W2-6). Resolution mirrored server-side from the real
 *      engine sites (ralph_engine / forward_test_service), not the
 *      registry defaults.
 *
 *   2. Parity columns — Bar exists / O / H / L / C, then per-indicator
 *      chips for THAT strategy's pack columns (utv4 trailing stop,
 *      rv2 rvol, …). Computed server-side with the SAME logic + window
 *      (last 300 common bars) + tolerances as the Strategy Detail
 *      "Per-Bar Parity Drift · v2" module, so numbers reconcile 1:1
 *      with the detail page.
 *
 *   3. Live-values lens toggle — first (decision-time WS) vs latest
 *      (REST-corrected) — re-fetches per lens (client + server cache
 *      keyed by (sid, lens)).
 *
 * Perf: parity is HEAVY (two full indicator pipelines per strategy).
 * Rows are paginated; parity loads lazily for the visible page only,
 * in sequential batches of 4 (server hard-caps at 8/request), against
 * a ~10 min server-side cache. Never a fleet-wide compute in one shot.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import Card from '@/components/Card';
import {
  useFleetModels,
  fetchParityBatch,
  type FleetModelRow,
  type ResolvedModel,
  type StrategyParityRow,
  type ParityMetric,
} from '@/hooks/queries/useModelParity';

// ── Constants ─────────────────────────────────────────────────────────

const PAGE_SIZE = 15;
// Sids per /parity request (server hard-caps at 8). Cold compute is
// ~30-90s per sid (two full indicator pipelines), so keep batches at 2
// to stay well inside HTTP timeouts; cached sids return instantly.
const BATCH_SIZE = 2;

type Lens = 'first' | 'latest';

/** Client-side cell state per (sid, lens). */
type ParityCell =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'done'; row: StrategyParityRow };

function cellKey(sid: number, lens: Lens): string {
  return `${sid}:${lens}`;
}

// ── Model badge ───────────────────────────────────────────────────────

const MODEL_SITE_DOC: Record<string, string> = {
  backtest: 'Runtime site: data_worker_engine (state.bt_model) / '
    + "forward_test_service HIFI path — cfg.backtest_model or the literal 'rest_hifi'.",
  algo: 'Runtime site: forward_test_service._require_algo_model (PR #26 tripwire) — '
    + 'algo_model → backtest_model; ALL-unset RAISES (never silently defaults; '
    + "an unset model used to write data_source='cache_None').",
  live: 'Runtime site: ralph_engine StrategyMonitor.__init__ — declared-if-valid '
    + 'else get_default_live_model().',
};

function badgePalette(m: ResolvedModel): { bg: string; fg: string } {
  if (m.source === 'explicit') return { bg: 'rgba(76,175,80,0.15)', fg: '#7fd081' };
  if (m.source === 'error' || m.source === 'invalid_fallback') {
    return { bg: 'rgba(244,67,54,0.18)', fg: '#ef5350' };
  }
  return { bg: 'rgba(255,193,7,0.15)', fg: '#ffc107' };  // default / fallback_backtest
}

function badgeText(m: ResolvedModel): string {
  switch (m.source) {
    case 'explicit':          return m.effective ?? '—';
    case 'default':           return `unset→${m.effective}`;
    case 'fallback_backtest': return `unset→${m.effective}`;
    case 'invalid_fallback':  return `${m.declared}⚠→${m.effective}`;
    case 'error':             return 'unset→ERROR';
  }
}

function badgeTitle(lane: 'backtest' | 'algo' | 'live', m: ResolvedModel): string {
  const doc = MODEL_SITE_DOC[lane];
  switch (m.source) {
    case 'explicit':
      return `Explicit — config.${lane}_model = '${m.declared}'. ${doc}`;
    case 'default':
      return `Unset in config — runtime resolves '${m.effective}'. ${doc}`;
    case 'fallback_backtest':
      return `algo_model unset — runtime falls back to config.backtest_model = '${m.effective}'. ${doc}`;
    case 'invalid_fallback':
      return `Configured '${m.declared}' is not a valid model id — engine falls back to '${m.effective}'. ${doc}`;
    case 'error':
      return `algo_model AND backtest_model both unset — the runtime RAISES `
        + `(cache_None tripwire). Set config.algo_model (fleet standard: cache_locked). ${doc}`;
  }
}

function ModelBadge({ lane, m }: { lane: 'backtest' | 'algo' | 'live'; m: ResolvedModel }) {
  const { bg, fg } = badgePalette(m);
  return (
    <span
      title={badgeTitle(lane, m)}
      style={{
        background: bg, color: fg,
        padding: '2px 6px', borderRadius: 3,
        fontSize: 11, whiteSpace: 'nowrap',
        display: 'inline-block',
      }}
    >
      {badgeText(m)}
    </span>
  );
}

// ── Parity % cell ─────────────────────────────────────────────────────

/** Same thresholds as the detail ribbon's row-% coloring (≥99 green,
 *  ≥90 amber, else red). */
function pctColor(pct: number | null | undefined): string {
  if (pct == null) return 'var(--text-muted)';
  if (pct >= 99) return '#7fd081';
  if (pct >= 90) return '#ffc107';
  return '#ef5350';
}

function fmtPct(pct: number | null | undefined): string {
  if (pct == null) return '—';
  return `${pct.toFixed(pct >= 99.95 ? 0 : 1)}%`;
}

function metricTitle(m: ParityMetric): string {
  return `${m.label}: ${m.match} match · ${m.minor} minor · ${m.major} major`
    + ` (of ${m.comparable} comparable bars)`;
}

// ── Component ─────────────────────────────────────────────────────────

export default function StrategyHealthV5() {
  const { data: modelRows, isLoading, error, refetch } = useFleetModels();

  const [lens, setLens] = useState<Lens>('first');
  const [page, setPage] = useState(0);
  const [filter, setFilter] = useState('');
  const [parityMap, setParityMap] = useState<Record<string, ParityCell>>({});
  const inFlightRef = useRef(false);

  const filteredRows = useMemo(() => {
    const rows = modelRows ?? [];
    const q = filter.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(r =>
      (r.name ?? '').toLowerCase().includes(q)
      || (r.symbol ?? '').toLowerCase().includes(q)
      || String(r.strategyId).includes(q));
  }, [modelRows, filter]);

  const pageCount = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = useMemo(
    () => filteredRows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE),
    [filteredRows, safePage],
  );

  // Sids on the CURRENT page that have no cell yet for the active lens —
  // the progressive loader works through these in small batches.
  const pendingSids = useMemo(
    () => pageRows
      .filter(r => parityMap[cellKey(r.strategyId, lens)] === undefined)
      .map(r => r.strategyId),
    [pageRows, parityMap, lens],
  );

  // Progressive batch loader: one request in flight at a time, BATCH_SIZE
  // sids per request. Marking the batch 'loading' mutates parityMap →
  // pendingSids recomputes → the effect re-runs and picks up the next
  // batch once the previous one settles. Errors are recorded per-sid so
  // a failing batch can't spin the loop.
  useEffect(() => {
    if (inFlightRef.current || pendingSids.length === 0) return;
    const batch = pendingSids.slice(0, BATCH_SIZE);
    inFlightRef.current = true;
    setParityMap(prev => {
      const next = { ...prev };
      for (const sid of batch) next[cellKey(sid, lens)] = { status: 'loading' };
      return next;
    });
    fetchParityBatch(batch, lens)
      .then(rows => {
        setParityMap(prev => {
          const next = { ...prev };
          for (const row of rows) {
            next[cellKey(row.strategyId, lens)] = row.error
              ? { status: 'error', message: row.error }
              : { status: 'done', row };
          }
          // Any sid the server didn't echo back → error, not eternal spinner.
          for (const sid of batch) {
            if (next[cellKey(sid, lens)]?.status === 'loading') {
              next[cellKey(sid, lens)] = { status: 'error', message: 'no row returned' };
            }
          }
          return next;
        });
      })
      .catch((e: unknown) => {
        const message = e instanceof Error ? e.message : String(e);
        setParityMap(prev => {
          const next = { ...prev };
          for (const sid of batch) next[cellKey(sid, lens)] = { status: 'error', message };
          return next;
        });
      })
      .finally(() => { inFlightRef.current = false; });
  }, [pendingSids, lens]);

  const loadedCount = useMemo(
    () => pageRows.filter(r => {
      const c = parityMap[cellKey(r.strategyId, lens)];
      return c !== undefined && c.status !== 'loading';
    }).length,
    [pageRows, parityMap, lens],
  );

  const retryErrors = () => {
    setParityMap(prev => {
      const next = { ...prev };
      for (const r of pageRows) {
        const k = cellKey(r.strategyId, lens);
        if (next[k]?.status === 'error') delete next[k];
      }
      return next;
    });
  };

  const thStyle: React.CSSProperties = { padding: '6px 8px', fontWeight: 500 };
  const thRight: React.CSSProperties = { ...thStyle, textAlign: 'right' };

  return (
    <div className="p-4 space-y-4">
      {/* Toolbar */}
      <Card>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>Model Parity</h3>
          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
            runtime-effective models + per-bar parity vs the live cache
            (same computation as the detail page&apos;s Per-Bar Parity Drift v2 —
            last 300 common bars)
          </span>
          <div style={{ flex: 1 }} />
          {/* Live-values lens toggle — mirrors the v2 ribbon toggle. */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
            <span style={{ color: 'var(--text-muted)' }}>Live values:</span>
            {(['first', 'latest'] as const).map(s => (
              <button
                key={s}
                onClick={() => setLens(s)}
                title={s === 'first'
                  ? 'Decision-time WS values (first_close) — what the live engine SAW when each bar formed'
                  : 'REST-corrected values — what the cache HEALED to afterward'}
                style={{
                  padding: '3px 8px', borderRadius: 4, fontSize: 12,
                  background: lens === s ? 'var(--accent, #4f8cf6)' : 'var(--bg-input)',
                  color: lens === s ? 'white' : 'var(--text-muted)',
                  border: lens === s ? 'none' : '1px solid var(--border)',
                  cursor: 'pointer',
                }}
              >
                {s === 'first' ? 'first (decision-time)' : 'latest (corrected)'}
              </button>
            ))}
          </div>
          <input
            value={filter}
            onChange={e => { setFilter(e.target.value); setPage(0); }}
            placeholder="filter name / symbol / sid"
            style={{
              padding: '4px 8px', borderRadius: 4, fontSize: 12,
              background: 'var(--bg-input)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', width: 190,
            }}
          />
          <button
            onClick={() => refetch()}
            style={{
              padding: '3px 10px', borderRadius: 4, fontSize: 12,
              background: 'var(--bg-input)', border: '1px solid var(--border)',
              color: 'var(--text-muted)', cursor: 'pointer',
            }}
          >
            Refresh models
          </button>
        </div>
        <div style={{ marginTop: 8, color: 'var(--text-muted)', fontSize: 11 }}>
          Parity loads lazily for the visible page ({loadedCount}/{pageRows.length} loaded,
          batches of {BATCH_SIZE}; server caches ~10 min per strategy+lens).
          Bar Parity (data) on the Health Overview tab is the lightweight
          OHLC-only data-layer check — this tab is the full engine-lens view
          including each pack&apos;s indicator columns.
          {pageRows.some(r => parityMap[cellKey(r.strategyId, lens)]?.status === 'error') && (
            <button
              onClick={retryErrors}
              style={{
                marginLeft: 8, padding: '1px 8px', borderRadius: 4, fontSize: 11,
                background: 'rgba(244,67,54,0.18)', color: '#ef5350',
                border: 'none', cursor: 'pointer',
              }}
            >
              retry failed
            </button>
          )}
        </div>
      </Card>

      {/* Table */}
      <Card>
        {isLoading && (
          <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading fleet…</p>
        )}
        {!isLoading && !!error && (
          <p style={{ color: '#ef5350', fontSize: 13 }}>
            Failed to load fleet models: {error instanceof Error ? error.message : String(error)}
          </p>
        )}
        {!isLoading && !error && (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'left' }}>
                    <th style={thStyle}>Strategy</th>
                    <th style={thStyle}>Symbol</th>
                    <th style={thStyle}>TF</th>
                    <th style={thStyle}
                        title={MODEL_SITE_DOC.backtest}>backtest_model</th>
                    <th style={thStyle}
                        title={MODEL_SITE_DOC.algo}>algo_model</th>
                    <th style={thStyle}
                        title={MODEL_SITE_DOC.live}>live_model</th>
                    <th style={thRight}
                        title="Bar exists: % of window bars present on BOTH lenses (backtest REST and live cache).">
                      Bar exists
                    </th>
                    <th style={thRight}>Open</th>
                    <th style={thRight}>High</th>
                    <th style={thRight}>Low</th>
                    <th style={thRight}>Close</th>
                    <th style={thStyle}
                        title="Per-indicator match % for this strategy's pack columns (same tolerance as the detail ribbon: 0.1% match / 1% minor).">
                      Indicators
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {pageRows.map(r => (
                    <ParityRow
                      key={`${r.userId}:${r.strategyId}`}
                      row={r}
                      cell={parityMap[cellKey(r.strategyId, lens)]}
                      lens={lens}
                    />
                  ))}
                  {pageRows.length === 0 && (
                    <tr>
                      <td colSpan={12}
                          style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
                        No strategies match the filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              marginTop: 10, fontSize: 12, color: 'var(--text-muted)',
            }}>
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={safePage === 0}
                style={{
                  padding: '3px 10px', borderRadius: 4,
                  background: 'var(--bg-input)', border: '1px solid var(--border)',
                  color: safePage === 0 ? 'var(--text-muted)' : 'var(--text-primary)',
                  cursor: safePage === 0 ? 'default' : 'pointer',
                }}
              >
                ← prev
              </button>
              <span>
                page {safePage + 1} / {pageCount} · {filteredRows.length} strategies
              </span>
              <button
                onClick={() => setPage(p => Math.min(pageCount - 1, p + 1))}
                disabled={safePage >= pageCount - 1}
                style={{
                  padding: '3px 10px', borderRadius: 4,
                  background: 'var(--bg-input)', border: '1px solid var(--border)',
                  color: safePage >= pageCount - 1 ? 'var(--text-muted)' : 'var(--text-primary)',
                  cursor: safePage >= pageCount - 1 ? 'default' : 'pointer',
                }}
              >
                next →
              </button>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

// ── Row ───────────────────────────────────────────────────────────────

function ParityRow({ row, cell, lens }: {
  row: FleetModelRow;
  cell: ParityCell | undefined;
  lens: Lens;
}) {
  const parity = cell?.status === 'done' ? cell.row : null;

  const byKey = useMemo(() => {
    const m = new Map<string, ParityMetric>();
    for (const metric of parity?.metrics ?? []) m.set(metric.key, metric);
    return m;
  }, [parity]);

  const indicatorMetrics = useMemo(
    () => (parity?.metrics ?? []).filter(m => m.kind === 'num'),
    [parity],
  );

  const barsTitle = parity?.bars
    ? `window: ${parity.bars.windowBars} bars`
      + ` (${parity.bars.common} common · ${parity.bars.backtestOnly} backtest-only`
      + ` · ${parity.bars.liveOnly} live-only)`
      + `${parity.bars.truncated ? ` — last ${parity.bars.windowBars} of ${parity.bars.totalOverlapBars}` : ''}`
      + ` · ${parity.bars.windowStart ?? '—'} → ${parity.bars.windowEnd ?? '—'}`
      + ` · lens=${lens}${parity.cached ? ' · cached' : ''}`
    : undefined;

  const pctCell = (key: string) => {
    if (cell === undefined || cell.status === 'loading') {
      return <span style={{ color: 'var(--text-muted)' }}>…</span>;
    }
    if (cell.status === 'error') {
      return <span title={cell.message} style={{ color: '#ef5350' }}>!</span>;
    }
    const m = byKey.get(key);
    if (!m || m.pct == null) {
      return <span title={parity?.note ?? 'not comparable'}
                   style={{ color: 'var(--text-muted)' }}>—</span>;
    }
    return (
      <span title={`${metricTitle(m)}${barsTitle ? `\n${barsTitle}` : ''}`}
            style={{ color: pctColor(m.pct), fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
        {fmtPct(m.pct)}
      </span>
    );
  };

  return (
    <tr style={{ borderTop: '1px solid var(--border)' }}>
      <td style={{ padding: '6px 8px' }}>
        <div style={{ color: 'var(--text-primary)' }}>
          {row.name || `sid ${row.strategyId}`}
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          sid {row.strategyId}{row.forwardTesting ? ' · fwd' : ''}
        </div>
      </td>
      <td style={{ padding: '6px 8px', fontVariantNumeric: 'tabular-nums' }}>
        {row.symbol || '—'}
      </td>
      <td style={{ padding: '6px 8px' }}>{row.timeframe || '—'}</td>
      <td style={{ padding: '6px 8px' }}>
        <ModelBadge lane="backtest" m={row.backtestModel} />
      </td>
      <td style={{ padding: '6px 8px' }}>
        <ModelBadge lane="algo" m={row.algoModel} />
      </td>
      <td style={{ padding: '6px 8px' }}>
        <ModelBadge lane="live" m={row.liveModel} />
      </td>
      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{pctCell('exists')}</td>
      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{pctCell('open')}</td>
      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{pctCell('high')}</td>
      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{pctCell('low')}</td>
      <td style={{ padding: '6px 8px', textAlign: 'right' }}>{pctCell('close')}</td>
      <td style={{ padding: '6px 8px' }}>
        {cell?.status === 'done' && indicatorMetrics.length === 0 && (
          <span title={parity?.note ?? 'no pack indicator columns on the chart payload'}
                style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>
        )}
        {indicatorMetrics.map(m => (
          <span
            key={m.key}
            title={`${metricTitle(m)}${barsTitle ? `\n${barsTitle}` : ''}`}
            style={{
              display: 'inline-block', marginRight: 6, marginBottom: 2,
              padding: '1px 6px', borderRadius: 3, fontSize: 11,
              background: 'var(--bg-input)', whiteSpace: 'nowrap',
            }}
          >
            <span style={{ color: 'var(--text-secondary)' }}>{m.label} </span>
            <span style={{ color: pctColor(m.pct), fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
              {fmtPct(m.pct)}
            </span>
          </span>
        ))}
        {(cell === undefined || cell.status === 'loading') && (
          <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>…</span>
        )}
        {cell?.status === 'error' && (
          <span title={cell.message} style={{ color: '#ef5350', fontSize: 11 }}>
            compute failed
          </span>
        )}
      </td>
    </tr>
  );
}
