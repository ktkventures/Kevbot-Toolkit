'use client';

/**
 * Admin > Parity (Phase B + C, 2026-05-14).
 *
 * Visual model-alignment tooling — surfaces where live / algo / backtest
 * disagree so we can move from "trust the theory" to "see the evidence."
 *
 * Tabs:
 *   - Bars Comparison    (Phase B ✅) cache vs REST OHLCV side-by-side
 *   - Entry Overlay      (Phase C ✅) 3-color markers on shared price chart
 *   - Divergence Heatmap (Phase C ✅) per-minute lane match table
 *   - Ticks              (Phase E stub)
 *
 * See docs/Parity_Plan_2026-05-14.md for the full plan.
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ParityBarComparison from '@/charts/ParityBarComparison';
import ParityEntryOverlay from '@/charts/ParityEntryOverlay';
import ParityDivergenceHeatmap from '@/charts/ParityDivergenceHeatmap';
import ParityObservableComparison from '@/charts/ParityObservableComparison';
import { useStrategies, type StrategyDTO } from '@/hooks/queries/useStrategies';
import { useBars } from '@/hooks/queries/useMarketData';
import {
  useAdminParityBars,
  useAdminParitySnapshot,
  useObservableBars,
} from '@/hooks/queries/useAdminParity';

function ComingSoonStub({ phase, blurb }: { phase: string; blurb: string }) {
  return (
    <div
      className="text-sm p-6 rounded"
      style={{
        background: 'var(--bg-input)',
        border: '1px dashed var(--border)',
        color: 'var(--text-muted)',
      }}
    >
      <div style={{ color: 'var(--text)' }} className="font-semibold mb-1">
        Coming in {phase}
      </div>
      <div>{blurb}</div>
    </div>
  );
}

function todayMidnightUtcIso(): string {
  const d = new Date();
  d.setUTCHours(0, 0, 0, 0);
  return d.toISOString();
}
function nowUtcIso(): string { return new Date().toISOString(); }

function toLocalInput(iso: string): string {
  // Convert ISO UTC → format suitable for <input type="datetime-local">
  // The input is naïve-local; we DO want naïve UTC here so toggling
  // between fields preserves the value Kevin pasted in.
  return iso.slice(0, 16);
}
function fromLocalInput(local: string): string {
  // Treat local input as UTC (no zone conversion).
  return `${local}:00Z`;
}

export default function AdminParityPage() {
  const { data: strategies, isLoading: stratsLoading } = useStrategies();
  const [strategyId, setStrategyId] = useState<number | null>(null);
  const [windowStart, setWindowStart] = useState<string>(todayMidnightUtcIso());
  const [windowEnd, setWindowEnd] = useState<string>(nowUtcIso());

  const selected: StrategyDTO | undefined = strategies?.find(
    (s) => s.id === strategyId,
  );

  // Phase B: bar comparison (existing hook)
  const parityBars = useAdminParityBars({
    strategyId,
    symbol: selected?.symbol ?? null,
    timeframe: selected?.timeframe ?? '1Min',
    days: 2,
  });

  // Phase C: snapshot (live + algo + bt entries) for chosen window
  const snapshot = useAdminParitySnapshot(strategyId, windowStart, windowEnd);

  // Phase C: 1Min bars for the entry-overlay chart
  const overlayBars = useBars(selected?.symbol ?? null, '1Min', 2);

  // Phase E: observable bars for the Ticks tab (aggregated server-side to 1Min)
  const observableQ = useObservableBars(
    selected?.symbol ?? null,
    windowStart,
    windowEnd,
    60,
  );

  const windowBarsForOverlay = useMemo(() => {
    const all = overlayBars.data || [];
    return all.filter((b) => b.timestamp >= windowStart && b.timestamp < windowEnd);
  }, [overlayBars.data, windowStart, windowEnd]);

  return (
    <div>
      <PageHeader
        title="Admin > Parity"
        subtitle="Visual model-alignment tooling. Compare live / algo / backtest data across strategies."
      />

      {/* Controls */}
      <Card>
        <div className="flex items-end gap-3 flex-wrap">
          <div>
            <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
              Strategy
            </label>
            <select
              value={strategyId ?? ''}
              onChange={(e) => {
                const v = e.target.value;
                setStrategyId(v ? Number(v) : null);
              }}
              className="text-sm px-2 py-1.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
                minWidth: 320,
              }}
            >
              <option value="">{stratsLoading ? 'Loading…' : '— select strategy —'}</option>
              {(strategies || []).map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id} · {s.name} ({s.symbol} {s.timeframe})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
              Window start (UTC)
            </label>
            <input
              type="datetime-local"
              value={toLocalInput(windowStart)}
              onChange={(e) => setWindowStart(fromLocalInput(e.target.value))}
              className="text-sm px-2 py-1.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
              }}
            />
          </div>
          <div>
            <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>
              Window end (UTC)
            </label>
            <input
              type="datetime-local"
              value={toLocalInput(windowEnd)}
              onChange={(e) => setWindowEnd(fromLocalInput(e.target.value))}
              className="text-sm px-2 py-1.5 rounded"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
              }}
            />
          </div>
          <div className="flex gap-1">
            <button
              type="button"
              className="text-xs px-2 py-1.5 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text)' }}
              onClick={() => {
                setWindowStart(todayMidnightUtcIso());
                setWindowEnd(nowUtcIso());
              }}
            >
              Today
            </button>
            <button
              type="button"
              className="text-xs px-2 py-1.5 rounded"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text)' }}
              onClick={() => {
                const e = new Date();
                const s = new Date(e.getTime() - 60 * 60 * 1000);
                setWindowStart(s.toISOString());
                setWindowEnd(e.toISOString());
              }}
            >
              Last 1h
            </button>
          </div>
          {selected && (
            <div className="text-xs" style={{ color: 'var(--text-muted)', marginLeft: 'auto' }}>
              <div><strong>Symbol:</strong> {selected.symbol}</div>
              <div><strong>TF:</strong> {selected.timeframe}</div>
              <div><strong>Dir:</strong> {selected.direction}</div>
            </div>
          )}
        </div>
        {snapshot.data && (
          <div className="text-xs mt-3 flex gap-4" style={{ color: 'var(--text-muted)' }}>
            <span>live entries: <strong style={{ color: 'var(--green)' }}>{snapshot.data.counts.live_entries}</strong></span>
            <span>algo entries: <strong style={{ color: 'var(--orange)' }}>{snapshot.data.counts.algo_trades}</strong></span>
            <span>bt entries: <strong style={{ color: '#3b82f6' }}>{snapshot.data.counts.bt_trades}</strong></span>
          </div>
        )}
      </Card>

      <div className="mt-4">
        <Card>
          <TabBar
            tabs={['Bars Comparison', 'Entry Overlay', 'Divergence Heatmap', 'Ticks']}
          >
            {(active) => {
              if (strategyId === null) {
                return (
                  <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>
                    Select a strategy above.
                  </div>
                );
              }

              if (active === 'Bars Comparison') {
                if (parityBars.isLoading) {
                  return <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>Loading…</div>;
                }
                if (parityBars.isError) {
                  return <div className="text-sm py-6" style={{ color: 'var(--red)' }}>Failed to load. Strategy may not have cache bars yet (needs live engine to have run on it).</div>;
                }
                return (
                  <ParityBarComparison
                    cacheBars={parityBars.cacheBars}
                    restBars={parityBars.restBars}
                    rows={parityBars.rows}
                    cacheValueType={parityBars.cacheValueType}
                    cacheNotes={parityBars.cacheNotes}
                    windowStart={windowStart}
                    windowEnd={windowEnd}
                  />
                );
              }

              if (active === 'Entry Overlay') {
                if (snapshot.isLoading || overlayBars.isLoading) {
                  return <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>Loading…</div>;
                }
                if (snapshot.isError) {
                  return <div className="text-sm py-6" style={{ color: 'var(--red)' }}>Failed to load snapshot.</div>;
                }
                return (
                  <ParityEntryOverlay
                    bars={windowBarsForOverlay}
                    liveEntries={snapshot.data?.live_entries || []}
                    algoTrades={snapshot.data?.algo_trades || []}
                    btTrades={snapshot.data?.bt_trades || []}
                    visibleRange={{ from: windowStart, to: windowEnd }}
                  />
                );
              }

              if (active === 'Divergence Heatmap') {
                if (snapshot.isLoading) {
                  return <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>Loading…</div>;
                }
                if (snapshot.isError) {
                  return <div className="text-sm py-6" style={{ color: 'var(--red)' }}>Failed to load snapshot.</div>;
                }
                return (
                  <ParityDivergenceHeatmap
                    liveEntries={snapshot.data?.live_entries || []}
                    algoTrades={snapshot.data?.algo_trades || []}
                    btTrades={snapshot.data?.bt_trades || []}
                  />
                );
              }

              // Ticks (Phase E)
              if (overlayBars.isLoading || parityBars.isLoading || observableQ.isLoading) {
                return <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>Loading…</div>;
              }
              return (
                <ParityObservableComparison
                  symbol={selected?.symbol ?? ''}
                  cacheBars={parityBars.cacheBars}
                  observableBars={observableQ.data?.bars ?? []}
                  restBars={overlayBars.data ?? []}
                  windowStart={windowStart}
                  windowEnd={windowEnd}
                  observableEmpty={!observableQ.isLoading && (observableQ.data?.bars.length ?? 0) === 0}
                  observableLoading={observableQ.isLoading}
                />
              );
            }}
          </TabBar>
        </Card>
      </div>
    </div>
  );
}
