'use client';

/**
 * Strategy Builder — Clean API-first page (simplified).
 *
 * Core config form with Run Backtest, results panel with KPIs + equity curve
 * + trade count, and Save button. No mock data.
 */

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useRunBacktest, BacktestRequest } from '@/hooks/queries/useBacktest';
import { useCreateStrategy } from '@/hooks/mutations/useStrategyMutations';
import { useConfluenceGroups, useRiskManagementPacks } from '@/hooks/queries/usePacks';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEFRAMES = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1Hour', '4Hour', '1Day'];
const DIRECTIONS: Array<'LONG' | 'SHORT'> = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  padding: '8px 14px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  width: '100%',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function kpi(v: number | undefined, decimals = 2, suffix = ''): string {
  if (v === undefined || v === null) return '--';
  return `${v >= 0 ? '+' : ''}${v.toFixed(decimals)}${suffix}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function StrategyBuilderPage() {
  const router = useRouter();
  const backtestMutation = useRunBacktest();
  const createMutation = useCreateStrategy();
  const { data: confluenceGroups } = useConfluenceGroups();
  const { data: rmPacks } = useRiskManagementPacks();

  // Config state
  const [symbol, setSymbol] = useState('');
  const [timeframe, setTimeframe] = useState('5Min');
  const [direction, setDirection] = useState<'LONG' | 'SHORT'>('LONG');
  const [session, setSession] = useState('RTH');
  const [lookbackDays, setLookbackDays] = useState(30);
  const [entryTrigger, setEntryTrigger] = useState('');
  const [exitTriggers, setExitTriggers] = useState('');
  const [confluence, setConfluence] = useState('');
  const [stopPackId, setStopPackId] = useState('');
  const [targetPackId, setTargetPackId] = useState('');
  const [strategyName, setStrategyName] = useState('');

  const handleRunBacktest = useCallback(() => {
    if (!symbol.trim()) return;
    if (!entryTrigger.trim()) return;

    const req: BacktestRequest = {
      symbol: symbol.trim().toUpperCase(),
      timeframe,
      direction,
      days: lookbackDays,
      session,
      entry_trigger_confluence_id: entryTrigger.trim(),
      exit_trigger_confluence_ids: exitTriggers.trim() ? exitTriggers.split(',').map((s) => s.trim()) : [],
      confluence: confluence.trim() ? confluence.split(',').map((s) => s.trim()) : [],
      stop_loss_pack_id: stopPackId || undefined,
      take_profit_pack_id: targetPackId || undefined,
      include_chart_data: false,
    };
    backtestMutation.mutate(req);
  }, [symbol, timeframe, direction, session, lookbackDays, entryTrigger, exitTriggers, confluence, stopPackId, targetPackId, backtestMutation]);

  const handleSave = useCallback(() => {
    if (!backtestMutation.data) return;
    const name = strategyName.trim() || `${symbol.toUpperCase()} ${direction} ${timeframe}`;

    createMutation.mutate({
      name,
      symbol: symbol.trim().toUpperCase(),
      timeframe,
      direction,
      trading_session: session,
      data_days: lookbackDays,
      entry_trigger_confluence_id: entryTrigger.trim(),
      exit_trigger_confluence_ids: exitTriggers.trim() ? exitTriggers.split(',').map((s) => s.trim()) : [],
      confluence: confluence.trim() ? confluence.split(',').map((s) => s.trim()) : [],
      stop_loss_pack_id: stopPackId || undefined,
      take_profit_pack_id: targetPackId || undefined,
      kpis: backtestMutation.data.kpis,
    }, {
      onSuccess: () => router.push('/strategies'),
    });
  }, [backtestMutation.data, strategyName, symbol, timeframe, direction, session, lookbackDays, entryTrigger, exitTriggers, confluence, stopPackId, targetPackId, createMutation, router]);

  const result = backtestMutation.data;
  const resultKpis = result?.kpis || {};

  // Derive pack options
  const stopPacks = (rmPacks || []).filter((p: any) => p.base_template?.includes('stop') || p.base_template?.includes('Stop'));
  const targetPacks = (rmPacks || []).filter((p: any) => p.base_template?.includes('target') || p.base_template?.includes('Target') || p.base_template?.includes('reward') || p.base_template?.includes('Reward'));
  const allPacks = rmPacks || [];

  return (
    <div>
      <PageHeader
        title="Strategy Builder"
        subtitle="Configure, backtest, and save a trading strategy"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
        {/* Left: Config */}
        <div className="lg:col-span-1 space-y-4">
          {/* Core config */}
          <Card>
            <h3 className="text-sm font-semibold mb-3">Core Configuration</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Symbol</label>
                <input type="text" placeholder="NVDA" value={symbol} onChange={(e) => setSymbol(e.target.value)} style={inputStyle} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
                  <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} style={inputStyle}>
                    {TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Direction</label>
                  <select value={direction} onChange={(e) => setDirection(e.target.value as 'LONG' | 'SHORT')} style={inputStyle}>
                    {DIRECTIONS.map((d) => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Session</label>
                  <select value={session} onChange={(e) => setSession(e.target.value)} style={inputStyle}>
                    {SESSIONS.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Lookback Days</label>
                  <input type="number" min={1} max={365} value={lookbackDays} onChange={(e) => setLookbackDays(Number(e.target.value))} style={inputStyle} />
                </div>
              </div>
            </div>
          </Card>

          {/* Entry/Exit triggers */}
          <Card>
            <h3 className="text-sm font-semibold mb-3">Triggers</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Entry Trigger (confluence ID)</label>
                <input type="text" placeholder="e.g. EMA_STACK-SML-BULL_CROSS" value={entryTrigger} onChange={(e) => setEntryTrigger(e.target.value)} style={inputStyle} />
              </div>
              <div>
                <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Exit Triggers (comma separated)</label>
                <input type="text" placeholder="e.g. EMA_STACK-SML-BEAR_CROSS" value={exitTriggers} onChange={(e) => setExitTriggers(e.target.value)} style={inputStyle} />
              </div>
              <div>
                <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Confluence conditions (comma separated)</label>
                <input type="text" placeholder="e.g. 5M-EMA_STACK-SML, 1M-RVOL-HIGH" value={confluence} onChange={(e) => setConfluence(e.target.value)} style={inputStyle} />
              </div>
            </div>
          </Card>

          {/* Stop/Target packs */}
          <Card>
            <h3 className="text-sm font-semibold mb-3">Risk Management</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Stop Loss Pack</label>
                <select value={stopPackId} onChange={(e) => setStopPackId(e.target.value)} style={inputStyle}>
                  <option value="">None (default)</option>
                  {allPacks.map((p: any) => (
                    <option key={p.id} value={p.id}>{p.base_template} ({p.version})</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Take Profit Pack</label>
                <select value={targetPackId} onChange={(e) => setTargetPackId(e.target.value)} style={inputStyle}>
                  <option value="">None (signal exit only)</option>
                  {allPacks.map((p: any) => (
                    <option key={p.id} value={p.id}>{p.base_template} ({p.version})</option>
                  ))}
                </select>
              </div>
            </div>
          </Card>

          {/* Run button */}
          <button
            className="w-full py-3 rounded-lg text-sm font-medium"
            style={{
              background: symbol.trim() && entryTrigger.trim() ? 'var(--accent)' : 'var(--bg-input)',
              color: symbol.trim() && entryTrigger.trim() ? '#fff' : 'var(--text-muted)',
              border: 'none',
              cursor: symbol.trim() && entryTrigger.trim() ? 'pointer' : 'not-allowed',
            }}
            disabled={!symbol.trim() || !entryTrigger.trim() || backtestMutation.isPending}
            onClick={handleRunBacktest}
          >
            {backtestMutation.isPending ? 'Running Backtest...' : 'Run Backtest'}
          </button>
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-2 space-y-4">
          {/* Error */}
          {backtestMutation.isError && (
            <Card>
              <div className="text-center py-8" style={{ color: 'var(--red)' }}>
                Backtest failed. Check your configuration and try again.
                <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                  {(backtestMutation.error as any)?.message || 'Unknown error'}
                </p>
              </div>
            </Card>
          )}

          {/* Results */}
          {result && (
            <>
              {/* KPIs */}
              <Card>
                <h3 className="text-sm font-semibold mb-3">Backtest Results</h3>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-4">
                  {[
                    { label: 'Win Rate', value: resultKpis.win_rate != null ? `${resultKpis.win_rate.toFixed(1)}%` : '--' },
                    { label: 'Profit Factor', value: resultKpis.profit_factor != null ? resultKpis.profit_factor.toFixed(2) : '--' },
                    { label: 'Daily R', value: kpi(resultKpis.daily_r) },
                    { label: 'Total R', value: kpi(resultKpis.total_r) },
                    { label: 'Trades', value: resultKpis.total_trades != null ? String(resultKpis.total_trades) : '--' },
                    { label: 'Max DD', value: resultKpis.max_r_drawdown != null ? `${resultKpis.max_r_drawdown.toFixed(1)}R` : '--' },
                  ].map((m) => (
                    <div key={m.label} className="text-center">
                      <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
                      <div className="text-lg font-mono font-bold" style={{ color: 'var(--text-primary)' }}>{m.value}</div>
                    </div>
                  ))}
                </div>

                {/* Secondary KPIs */}
                <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                  {[
                    { label: 'Avg R', value: kpi(resultKpis.avg_r) },
                    { label: 'R-Squared', value: resultKpis.r_squared != null ? resultKpis.r_squared.toFixed(3) : '--' },
                    { label: 'Avg Win', value: kpi(result.secondary_kpis?.avg_win_r) },
                    { label: 'Avg Loss', value: kpi(result.secondary_kpis?.avg_loss_r) },
                    { label: 'Recovery', value: result.secondary_kpis?.recovery_factor != null ? result.secondary_kpis.recovery_factor.toFixed(2) : '--' },
                    { label: 'Data', value: result.data_source || '--' },
                  ].map((m) => (
                    <div key={m.label} className="text-center">
                      <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
                      <div className="text-sm font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{m.value}</div>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Equity curve placeholder */}
              <Card>
                <h3 className="text-sm font-semibold mb-3">Equity Curve</h3>
                {result.equity_curve && result.equity_curve.length > 0 ? (
                  <div className="rounded-lg overflow-hidden" style={{ background: 'var(--bg-input)', height: 200 }}>
                    {/* Simple SVG equity curve */}
                    <svg width="100%" height="100%" viewBox={`0 0 ${result.equity_curve.length} 100`} preserveAspectRatio="none">
                      {(() => {
                        const data = result.equity_curve.map((p: any) => p.cumulative_r);
                        const min = Math.min(...data);
                        const max = Math.max(...data);
                        const range = max - min || 1;
                        const points = data.map((v: number, i: number) =>
                          `${i},${100 - ((v - min) / range) * 90 - 5}`
                        ).join(' ');
                        return (
                          <>
                            <polyline points={points} fill="none" stroke="var(--accent)" strokeWidth="1" />
                          </>
                        );
                      })()}
                    </svg>
                  </div>
                ) : (
                  <div className="rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-input)', height: 200 }}>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>No equity data</span>
                  </div>
                )}
              </Card>

              {/* Trade count + save */}
              <Card>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm">
                      {result.trades?.length || 0} trades over {result.total_bars?.toLocaleString() || 0} bars
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      Source: {result.data_source || 'Unknown'}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <input
                      type="text"
                      placeholder="Strategy name"
                      value={strategyName}
                      onChange={(e) => setStrategyName(e.target.value)}
                      className="w-48"
                      style={inputStyle}
                    />
                    <button
                      className="px-4 py-2 rounded-lg text-sm font-medium"
                      style={{ background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer' }}
                      onClick={handleSave}
                      disabled={createMutation.isPending}
                    >
                      {createMutation.isPending ? 'Saving...' : 'Save Strategy'}
                    </button>
                  </div>
                </div>
              </Card>
            </>
          )}

          {/* Placeholder when no results */}
          {!result && !backtestMutation.isError && (
            <Card>
              <div className="text-center py-20">
                <p className="text-lg font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                  Configure and run a backtest
                </p>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  Set your symbol, timeframe, direction, and triggers on the left, then click Run Backtest.
                  Results with KPIs, equity curve, and trade history will appear here.
                </p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
