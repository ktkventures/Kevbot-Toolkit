'use client';

/**
 * Strategy Health V4 — "By Deploy" tab.
 *
 * Each row = one commit's active window (from its timestamp until the
 * next deploy, or now for the latest). Columns include the commit SHA
 * + subject + window duration + pair-rate metrics during that window.
 *
 * The combined % column is the cohort-wide aggregate (sum across all
 * strategies). The avg-strategy column averages each strategy's
 * combined % equally regardless of volume — diagnostic for whether
 * a deploy affected "all strategies a little" vs "a few strategies a lot."
 *
 * Backed by /api/admin/strategy-health/by-deploy which reads the
 * commit list from src/deploy_history.json. Run
 * `_generate_deploy_history.py` after any push to refresh.
 */

import { useMemo, useState } from 'react';
import Card from '@/components/Card';
import {
  useStrategyHealthByDeploy,
  type DeployWindowRow,
} from '@/hooks/queries/useStrategyHealthByDeploy';
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
  { id: '30d', name: 'Last 30 days', days: 30 },
] as const;

function fmtPct(v: number | null): string {
  if (v === null || v === undefined) return '—';
  return `${v.toFixed(1)}%`;
}

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return m === 0 ? `${h}h` : `${h}h ${m}m`;
  }
  const d = Math.floor(seconds / 86400);
  const h = Math.round((seconds % 86400) / 3600);
  return h === 0 ? `${d}d` : `${d}d ${h}h`;
}

function fmtTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    return (
      d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC'
    );
  } catch {
    return iso;
  }
}

function pctCellColor(v: number | null): React.CSSProperties {
  if (v === null) return {};
  if (v >= 90) return { color: 'var(--green, #22c55e)', fontWeight: 600 };
  if (v >= 70) return { color: 'var(--text)' };
  if (v >= 40) return { color: 'var(--orange, #f59e0b)' };
  return { color: 'var(--red, #ef4444)' };
}

export default function StrategyHealthByDeployV4() {
  const [preset, setPreset] = useState<(typeof DATE_PRESETS)[number]['id']>(
    '14d',
  );

  const sinceISO = useMemo(() => {
    const d = DATE_PRESETS.find((p) => p.id === preset)?.days ?? 14;
    const dt = new Date(Date.now() - d * 24 * 3600 * 1000);
    return dt.toISOString();
  }, [preset]);

  const { data, isLoading, error } = useStrategyHealthByDeploy({
    since: sinceISO,
    pairWindowS: 5,
  });

  const cohortMeta: StrategyMeta[] = data?.strategy_metadata ?? [];
  const { filters } = useFilterState();

  const allowedSids = useMemo(() => {
    if (cohortMeta.length === 0) return null;
    const s = new Set<number>();
    for (const m of cohortMeta) {
      if (passesFilters(m, filters)) s.add(m.strategy_id);
    }
    return s;
  }, [cohortMeta, filters]);

  // Re-aggregate deploy rows from filtered cohort
  const deploys: DeployWindowRow[] = useMemo(() => {
    if (!data) return [];
    if (!allowedSids) return data.deploys;
    return data.deploys.map((d) => {
      const sRows = (data.strategies_by_deploy[d.sha] ?? []).filter((r) =>
        allowedSids.has(r.strategy_id),
      );
      const totA = sRows.reduce((a, b) => a + b.alerts, 0);
      const totBT = sRows.reduce((a, b) => a + b.bt_events, 0);
      const totP = sRows.reduce((a, b) => a + b.paired, 0);
      const totPh = sRows.reduce((a, b) => a + b.phantom, 0);
      const totM = sRows.reduce((a, b) => a + b.missed, 0);
      const denom = totP + totPh + totM;
      const cpct = denom > 0 ? (100 * totP) / denom : null;
      const ranked = sRows.filter((r) => r.rankable);
      const avgCpct =
        ranked.length > 0
          ? ranked.reduce((a, b) => a + b.combined_pct, 0) / ranked.length
          : null;
      return {
        ...d,
        alerts: totA,
        bt_events: totBT,
        paired: totP,
        phantom: totPh,
        missed: totM,
        combined_pct: cpct !== null ? Math.round(cpct * 10) / 10 : null,
        avg_strategy_combined_pct:
          avgCpct !== null ? Math.round(avgCpct * 10) / 10 : null,
        n_active_strategies: sRows.length,
        n_ranked_strategies: ranked.length,
      };
    });
  }, [data, allowedSids]);

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            flexWrap: 'wrap',
            padding: 12,
          }}
        >
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>Date range</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              ±5s pair window. Each row = one commit&apos;s active period.
              The Combined % column aggregates events across all strategies.
              The Avg-Strategy % averages per-strategy combined % equally.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {DATE_PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => setPreset(p.id)}
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
          {data && (
            <div
              style={{
                marginLeft: 'auto',
                fontSize: 11,
                color: 'var(--text-muted)',
              }}
            >
              {data.deploy_count} deploys · cohort: {data.cohort_size}{' '}
              strategies
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

      {isLoading && (
        <Card>
          <div style={{ padding: 24, color: 'var(--text-muted)' }}>
            Loading per-deploy pair-rate analysis…
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
      {data?.note && (
        <Card>
          <div style={{ padding: 12, color: 'var(--orange)' }}>
            {data.note}
          </div>
        </Card>
      )}

      {data && !isLoading && (
        <Card>
          <div style={{ padding: '12px 16px' }}>
            <div
              style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}
            >
              Deploys, newest first
            </div>
            <div
              style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                marginBottom: 10,
              }}
            >
              Combined % = sum-of-paired / sum-of-(paired+phantom+missed).
              Avg-Strategy % = equally-weighted average across strategies
              with both alerts and BT events in the window.
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
                      Commit
                    </th>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>
                      Subject
                    </th>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>
                      Deployed (UTC)
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Window
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Strategies
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Alerts
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      BT events
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Combined %
                    </th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>
                      Avg-Strategy %
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {deploys.map((d) => (
                    <tr
                      key={d.sha + d.timestamp_iso}
                      style={{
                        borderBottom: '1px solid var(--border-muted)',
                      }}
                    >
                      <td style={{ padding: '4px 8px' }}>
                        <code
                          style={{
                            background: 'var(--bg-input)',
                            padding: '1px 4px',
                            borderRadius: 3,
                          }}
                        >
                          {d.sha}
                        </code>
                      </td>
                      <td
                        style={{
                          padding: '4px 8px',
                          maxWidth: 380,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                        title={d.subject}
                      >
                        {d.subject}
                      </td>
                      <td
                        style={{
                          padding: '4px 8px',
                          color: 'var(--text-muted)',
                          fontSize: 11,
                        }}
                      >
                        {fmtTimestamp(d.timestamp_iso)}
                      </td>
                      <td
                        style={{ textAlign: 'right', padding: '4px 8px' }}
                      >
                        {fmtDuration(d.duration_seconds)}
                      </td>
                      <td
                        style={{ textAlign: 'right', padding: '4px 8px' }}
                      >
                        {d.n_ranked_strategies}/{d.n_active_strategies}
                      </td>
                      <td
                        style={{ textAlign: 'right', padding: '4px 8px' }}
                      >
                        {d.alerts}
                      </td>
                      <td
                        style={{ textAlign: 'right', padding: '4px 8px' }}
                      >
                        {d.bt_events}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '4px 8px',
                          ...pctCellColor(d.combined_pct),
                        }}
                      >
                        {fmtPct(d.combined_pct)}
                      </td>
                      <td
                        style={{
                          textAlign: 'right',
                          padding: '4px 8px',
                          ...pctCellColor(d.avg_strategy_combined_pct),
                        }}
                      >
                        {fmtPct(d.avg_strategy_combined_pct)}
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
