'use client';

/**
 * Strategy Health V3 — "By Hour" tab.
 *
 * Surfaces the cross-hour pair-rate analysis at ±5s window:
 *   - Cross-hour KPI summary table (every hour with KPIs + ranks)
 *   - Click any hour row to drill into per-strategy detail
 *
 * Combined % = paired / (paired + phantom + missed). The primary
 * single-number KPI for live↔backtest pair quality. Rank columns
 * show how each hour compares to other hours by avg/max/min combined %
 * (rank 1 = best). The per-hour detail drill shows every strategy's
 * combined % with a rank within that hour (rank 1 = cleanest strategy
 * in that hour).
 *
 * Pure read-only dashboard, no engine impact.
 */

import { useMemo, useState } from 'react';
import Card from '@/components/Card';
import {
  useStrategyHealthByHour,
  type ByHourKpiRow,
  type ByHourStrategyRow,
} from '@/hooks/queries/useStrategyHealthByHour';
import {
  FilterBar,
  passesFilters,
  useFilterState,
  type StrategyMeta,
} from '@/components/StrategyFilters';

const DATE_PRESETS = [
  { id: '24h', name: 'Last 24 hours', days: 1 },
  { id: '7d', name: 'Last 7 days', days: 7 },
  { id: '14d', name: 'Last 14 days', days: 14 },
] as const;

function fmtPct(v: number | null): string {
  if (v === null || v === undefined) return '—';
  return `${v.toFixed(1)}%`;
}

function pctCellColor(v: number | null): React.CSSProperties {
  if (v === null) return {};
  if (v >= 90) return { color: 'var(--green, #22c55e)', fontWeight: 600 };
  if (v >= 70) return { color: 'var(--text)' };
  if (v >= 40) return { color: 'var(--orange, #f59e0b)' };
  return { color: 'var(--red, #ef4444)' };
}

function rankCell(rank: number | null, total: number): React.ReactNode {
  if (rank === null || rank === undefined) return '—';
  const star = rank <= 5 ? '⭐ ' : '';
  return (
    <span style={rank <= 5 ? { fontWeight: 600 } : undefined}>
      {star}
      {rank}/{total}
    </span>
  );
}

const TOLERANCE_PRESETS = [
  { id: 5, label: '±5s', description: 'Tight — standard for engine fidelity' },
  { id: 10, label: '±10s', description: 'Catches "almost-paired" cases' },
  { id: 60, label: '±60s', description: 'Looser — useful for >1m timeframes' },
] as const;

export default function StrategyHealthByHourV3() {
  const [preset, setPreset] = useState<(typeof DATE_PRESETS)[number]['id']>(
    '7d',
  );
  const [tolerance, setTolerance] = useState<5 | 10 | 60>(5);
  const [stabilityTail, setStabilityTail] = useState<number>(15);
  const [selectedHour, setSelectedHour] = useState<string | null>(null);

  const sinceISO = useMemo(() => {
    const d = DATE_PRESETS.find((p) => p.id === preset)?.days ?? 7;
    const dt = new Date(Date.now() - d * 24 * 3600 * 1000);
    return dt.toISOString();
  }, [preset]);

  const { data, isLoading, error } = useStrategyHealthByHour({
    since: sinceISO,
    pairWindowS: tolerance,
    stabilityTailMinutes: stabilityTail,
  });

  const cohortMeta: StrategyMeta[] = data?.strategy_metadata ?? [];
  const { filters } = useFilterState();

  // Allowed sids = strategies passing the active filters
  const allowedSids = useMemo(() => {
    if (cohortMeta.length === 0) return null;
    const s = new Set<number>();
    for (const m of cohortMeta) {
      if (passesFilters(m, filters)) s.add(m.strategy_id);
    }
    return s;
  }, [cohortMeta, filters]);

  // Re-aggregate hours based on filtered cohort
  const hourRows: ByHourKpiRow[] = useMemo(() => {
    if (!data) return [];
    if (!allowedSids) return data.hours;
    // Recompute KPIs + ranks across filtered set
    const computed = Object.entries(data.strategies_by_hour).map(
      ([hour, rows]) => {
        const inFilter = rows.filter((r) => allowedSids.has(r.strategy_id));
        const ranked = inFilter.filter(
          (r) => r.alerts >= 1 && r.bt_events >= 1,
        );
        const cpcts = ranked.map((r) => r.combined_pct);
        if (cpcts.length === 0) {
          return {
            hour_utc: hour,
            n_active: inFilter.length,
            n_ranked: 0,
            avg_combined_pct: null,
            max_combined_pct: null,
            min_combined_pct: null,
            avg_rank: null,
            max_rank: null,
            min_rank: null,
            ranked_total: 0,
          };
        }
        return {
          hour_utc: hour,
          n_active: inFilter.length,
          n_ranked: cpcts.length,
          avg_combined_pct:
            Math.round(
              (cpcts.reduce((a, b) => a + b, 0) / cpcts.length) * 10,
            ) / 10,
          max_combined_pct: Math.round(Math.max(...cpcts) * 10) / 10,
          min_combined_pct: Math.round(Math.min(...cpcts) * 10) / 10,
          avg_rank: null,
          max_rank: null,
          min_rank: null,
          ranked_total: 0,
        };
      },
    );
    // Sort hours chronologically + assign cross-hour ranks
    const typed: ByHourKpiRow[] = computed.map((c) => ({
      hour_utc: c.hour_utc,
      n_active: c.n_active,
      n_ranked: c.n_ranked,
      avg_combined_pct: c.avg_combined_pct,
      max_combined_pct: c.max_combined_pct,
      min_combined_pct: c.min_combined_pct,
      avg_rank: null as number | null,
      max_rank: null as number | null,
      min_rank: null as number | null,
      ranked_total: 0,
    }));
    typed.sort((a, b) => a.hour_utc.localeCompare(b.hour_utc));
    const withData = typed.filter((h) => h.avg_combined_pct !== null);
    const n = withData.length;
    const byAvg = [...withData].sort(
      (a, b) => (b.avg_combined_pct ?? 0) - (a.avg_combined_pct ?? 0),
    );
    const byMax = [...withData].sort(
      (a, b) => (b.max_combined_pct ?? 0) - (a.max_combined_pct ?? 0),
    );
    const byMin = [...withData].sort(
      (a, b) => (b.min_combined_pct ?? 0) - (a.min_combined_pct ?? 0),
    );
    const avgRankMap = new Map(byAvg.map((h, i) => [h.hour_utc, i + 1]));
    const maxRankMap = new Map(byMax.map((h, i) => [h.hour_utc, i + 1]));
    const minRankMap = new Map(byMin.map((h, i) => [h.hour_utc, i + 1]));
    for (const h of typed) {
      h.avg_rank = avgRankMap.get(h.hour_utc) ?? null;
      h.max_rank = maxRankMap.get(h.hour_utc) ?? null;
      h.min_rank = minRankMap.get(h.hour_utc) ?? null;
      h.ranked_total = n;
    }
    return typed;
  }, [data, allowedSids]);

  // Filter per-strategy rows for the drill-down + recompute rank-in-hour
  const stratsForHour: ByHourStrategyRow[] = useMemo(() => {
    if (!selectedHour || !data) return [];
    const rows = data.strategies_by_hour[selectedHour] ?? [];
    const filtered = allowedSids
      ? rows.filter((r) => allowedSids.has(r.strategy_id))
      : rows;
    // Recompute within-hour rank within the filtered set
    const rankable = filtered
      .filter((r) => r.alerts >= 1 && r.bt_events >= 1)
      .sort((a, b) => {
        const d = b.combined_pct - a.combined_pct;
        return d !== 0 ? d : b.alerts - a.alerts;
      });
    const rankMap = new Map(
      rankable.map((r, i) => [r.strategy_id, i + 1]),
    );
    return filtered.map((r) => ({
      ...r,
      rank: rankMap.get(r.strategy_id) ?? null,
      rank_total: rankable.length,
    }));
  }, [selectedHour, data, allowedSids]);

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <Card>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            flexWrap: 'wrap',
            padding: '12px',
          }}
        >
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Date range</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Combined % = paired / (paired + phantom + missed).
              Stability tail clips the last {stabilityTail} min to exclude
              BT-lane lag.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {DATE_PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  setPreset(p.id);
                  setSelectedHour(null);
                }}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  fontSize: 12,
                  border: '1px solid var(--border)',
                  background:
                    preset === p.id
                      ? 'var(--blue-muted, #1e40af44)'
                      : 'var(--bg-input)',
                  color:
                    preset === p.id
                      ? 'var(--blue, #3b82f6)'
                      : 'var(--text-secondary)',
                  fontWeight: preset === p.id ? 600 : 400,
                  cursor: 'pointer',
                }}
              >
                {p.name}
              </button>
            ))}
          </div>
          {/* Tolerance selector */}
          <div
            style={{
              display: 'flex',
              gap: 6,
              borderLeft: '1px solid var(--border)',
              paddingLeft: 12,
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                alignSelf: 'center',
                marginRight: 4,
              }}
            >
              Tolerance:
            </div>
            {TOLERANCE_PRESETS.map((t) => (
              <button
                key={t.id}
                onClick={() => {
                  setTolerance(t.id);
                  setSelectedHour(null);
                }}
                title={t.description}
                style={{
                  padding: '6px 10px',
                  borderRadius: 6,
                  fontSize: 12,
                  border: '1px solid var(--border)',
                  background:
                    tolerance === t.id
                      ? 'var(--blue-muted, #1e40af44)'
                      : 'var(--bg-input)',
                  color:
                    tolerance === t.id
                      ? 'var(--blue, #3b82f6)'
                      : 'var(--text-secondary)',
                  fontWeight: tolerance === t.id ? 600 : 400,
                  cursor: 'pointer',
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
          {/* Stability tail toggle */}
          <div
            style={{
              display: 'flex',
              gap: 6,
              alignItems: 'center',
              borderLeft: '1px solid var(--border)',
              paddingLeft: 12,
            }}
          >
            <label
              style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                cursor: 'pointer',
              }}
              title="Clip the upper bound to (now - N min) to exclude BT-lane lag"
            >
              <input
                type="checkbox"
                checked={stabilityTail > 0}
                onChange={(e) => setStabilityTail(e.target.checked ? 15 : 0)}
                style={{ cursor: 'pointer' }}
              />
              Stability tail (15 min)
            </label>
          </div>
          {data && (
            <div
              style={{
                marginLeft: 'auto',
                fontSize: 11,
                color: 'var(--text-muted)',
              }}
            >
              cohort: {data.cohort_size} strategies · hours with data:{' '}
              {data.hours_with_data} / {data.hours_count}
            </div>
          )}
        </div>
      </Card>

      {/* Filter bar */}
      {cohortMeta.length > 0 && (
        <Card>
          <div style={{ padding: 12 }}>
            <FilterBar cohort={cohortMeta} />
          </div>
        </Card>
      )}

      {/* Loading / error */}
      {isLoading && (
        <Card>
          <div style={{ padding: 24, color: 'var(--text-muted)' }}>
            Loading hourly pair-rate analysis…
          </div>
        </Card>
      )}
      {error && (
        <Card>
          <div style={{ padding: 12, color: 'var(--red)' }}>
            Failed to load: {(error as any)?.message || 'unknown'}
          </div>
        </Card>
      )}

      {/* Cross-hour summary */}
      {data && !isLoading && (
        <Card>
          <div style={{ padding: '12px 16px' }}>
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>
              Cross-hour summary
            </div>
            <div
              style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                marginBottom: 10,
              }}
            >
              Each row = one hour. Rank columns show this hour&apos;s rank
              across all {data.hours_with_data} hours with data (rank 1 =
              best by that metric, top 5 marked with ⭐). Click any row to
              drill into the per-strategy detail for that hour.
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 12,
                  fontFamily: 'monospace',
                }}
              >
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>
                      Hour (UTC)
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Ranked N
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Avg Combined %
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Max Combined %
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Min Combined %
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Avg Rank
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Max Rank
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Min Rank
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {hourRows.map((r) => {
                    const isSel = selectedHour === r.hour_utc;
                    return (
                      <tr
                        key={r.hour_utc}
                        onClick={() => setSelectedHour(r.hour_utc)}
                        style={{
                          borderBottom: '1px solid var(--border-muted)',
                          cursor: 'pointer',
                          background: isSel
                            ? 'var(--blue-muted, #1e40af22)'
                            : undefined,
                        }}
                      >
                        <td style={{ padding: '4px 8px' }}>{r.hour_utc}</td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.n_ranked}/{r.n_active}
                        </td>
                        <td
                          style={{
                            textAlign: 'right',
                            padding: '4px 8px',
                            ...pctCellColor(r.avg_combined_pct),
                          }}
                        >
                          {fmtPct(r.avg_combined_pct)}
                        </td>
                        <td
                          style={{
                            textAlign: 'right',
                            padding: '4px 8px',
                            ...pctCellColor(r.max_combined_pct),
                          }}
                        >
                          {fmtPct(r.max_combined_pct)}
                        </td>
                        <td
                          style={{
                            textAlign: 'right',
                            padding: '4px 8px',
                            ...pctCellColor(r.min_combined_pct),
                          }}
                        >
                          {fmtPct(r.min_combined_pct)}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {rankCell(r.avg_rank, r.ranked_total)}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {rankCell(r.max_rank, r.ranked_total)}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {rankCell(r.min_rank, r.ranked_total)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {/* Per-hour drill-down */}
      {selectedHour && data && (
        <Card>
          <div style={{ padding: '12px 16px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <div>
                <div style={{ fontSize: 15, fontWeight: 600 }}>
                  Hour {selectedHour} UTC — per-strategy detail
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: 'var(--text-muted)',
                    marginTop: 2,
                  }}
                >
                  Rank = cohort rank within this hour (rank 1 = cleanest
                  strategy this hour). Only strategies with alerts ≥ 1
                  AND BT events ≥ 1 are ranked.
                </div>
              </div>
              <button
                onClick={() => setSelectedHour(null)}
                style={{
                  padding: '6px 10px',
                  borderRadius: 6,
                  fontSize: 12,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-input)',
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
            <div style={{ overflowX: 'auto', marginTop: 10 }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: 12,
                  fontFamily: 'monospace',
                }}
              >
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      sid
                    </th>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>
                      Strategy
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Alerts
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      BT events
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Paired
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Phantom
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Missed
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Combined %
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Alert-pair %
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Rank
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {stratsForHour
                    .slice()
                    .sort((a, b) => {
                      // Rank ascending (best first); unranked last
                      if (a.rank === null && b.rank === null)
                        return a.strategy_id - b.strategy_id;
                      if (a.rank === null) return 1;
                      if (b.rank === null) return -1;
                      return a.rank - b.rank;
                    })
                    .map((r) => (
                      <tr
                        key={r.strategy_id}
                        style={{
                          borderBottom: '1px solid var(--border-muted)',
                        }}
                      >
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.strategy_id}
                        </td>
                        <td style={{ padding: '4px 8px' }}>
                          {r.name || '(unknown)'}{' '}
                          <span
                            style={{
                              color: 'var(--text-muted)',
                              fontSize: 11,
                            }}
                          >
                            {r.symbol} {r.timeframe}
                          </span>
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.alerts}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.bt_events}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.paired}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.phantom}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.missed}
                        </td>
                        <td
                          style={{
                            textAlign: 'right',
                            padding: '4px 8px',
                            ...pctCellColor(r.combined_pct),
                          }}
                        >
                          {fmtPct(r.combined_pct)}
                        </td>
                        <td
                          style={{
                            textAlign: 'right',
                            padding: '4px 8px',
                            ...pctCellColor(r.alert_pair_pct),
                          }}
                        >
                          {fmtPct(r.alert_pair_pct)}
                        </td>
                        <td
                          style={{ textAlign: 'right', padding: '4px 8px' }}
                        >
                          {r.rank
                            ? `${r.rank}/${r.rank_total}`
                            : '—'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
