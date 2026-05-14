'use client';

/**
 * Admin > Parity (Phase B, 2026-05-14).
 *
 * Visual model-alignment tooling — surfaces where live / algo / backtest
 * disagree so we can move from "trust the theory" to "see the evidence."
 *
 * Tabs:
 *   - Bars Comparison    (Phase B ✅ — this commit)
 *   - Entry Overlay      (Phase C stub)
 *   - Divergence Heatmap (Phase C stub)
 *   - Ticks              (Phase E stub)
 *
 * See docs/Parity_Plan_2026-05-14.md for the full plan.
 */

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ParityBarComparison from '@/charts/ParityBarComparison';
import { useStrategies, type StrategyDTO } from '@/hooks/queries/useStrategies';
import { useAdminParityBars } from '@/hooks/queries/useAdminParity';

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

export default function AdminParityPage() {
  const { data: strategies, isLoading: stratsLoading } = useStrategies();
  const [strategyId, setStrategyId] = useState<number | null>(null);

  const selected: StrategyDTO | undefined = strategies?.find(
    (s) => s.id === strategyId,
  );

  const parity = useAdminParityBars({
    strategyId,
    symbol: selected?.symbol ?? null,
    timeframe: selected?.timeframe ?? '1Min',
    days: 2,
  });

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
          {selected && (
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
              <div><strong>Symbol:</strong> {selected.symbol}</div>
              <div><strong>TF:</strong> {selected.timeframe}</div>
              <div><strong>Dir:</strong> {selected.direction}</div>
            </div>
          )}
        </div>
      </Card>

      <div className="mt-4">
        <Card>
          <TabBar
            tabs={['Bars Comparison', 'Entry Overlay', 'Divergence Heatmap', 'Ticks']}
          >
            {(active) => {
              if (active === 'Bars Comparison') {
                if (strategyId === null) {
                  return (
                    <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>
                      Select a strategy above to compare its cache vs REST bars.
                    </div>
                  );
                }
                if (parity.isLoading) {
                  return <div className="text-sm py-6 text-center" style={{ color: 'var(--text-muted)' }}>Loading…</div>;
                }
                if (parity.isError) {
                  return <div className="text-sm py-6" style={{ color: 'var(--red)' }}>Failed to load. Strategy may not have cache bars yet (needs live engine to have run on it).</div>;
                }
                return (
                  <ParityBarComparison
                    cacheBars={parity.cacheBars}
                    restBars={parity.restBars}
                    rows={parity.rows}
                    cacheValueType={parity.cacheValueType}
                    cacheNotes={parity.cacheNotes}
                  />
                );
              }
              if (active === 'Entry Overlay') {
                return (
                  <ComingSoonStub
                    phase="Phase C"
                    blurb="Price chart with live / algo / backtest entry markers overlaid, click-to-drill into per-lane state for any disagreement minute."
                  />
                );
              }
              if (active === 'Divergence Heatmap') {
                return (
                  <ComingSoonStub
                    phase="Phase C"
                    blurb="Strategy × minute grid colored by 3-way match state. Click any cell for drill-down."
                  />
                );
              }
              // Ticks
              return (
                <ComingSoonStub
                  phase="Phase E"
                  blurb="Observable-vs-settled comparison rebuilt from Polygon flat-file tick data. Recreates Kevin's external Claude-app analysis inside the app. Requires daily S3 ingestion; gated behind Phase D outcomes."
                />
              );
            }}
          </TabBar>
        </Card>
      </div>
    </div>
  );
}
