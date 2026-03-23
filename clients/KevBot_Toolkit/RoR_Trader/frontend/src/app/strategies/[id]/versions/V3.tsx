'use client';

import { useState, useMemo } from 'react';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

// ---------------------------------------------------------------------------
// V3 Meta: Streamlined Strategy Detail
// ---------------------------------------------------------------------------
// What changed from V2 and why:
//
// TABS REDUCED: 9 -> 3
//   V2: Equity & KPIs, Extended KPIs, Price Chart, Live Chart, Trade History,
//       Confluence Analysis, Configuration, Alerts, Alert Analysis
//   V3: Overview, Trades, Monitoring
//
// REMOVED (from top-level):
// - "Extended KPIs" tab (separate) -> folded into Overview as collapsible section
// - "Price Chart" tab (separate) -> embedded in Overview below equity curve
// - "Configuration" tab (separate) -> shown as compact summary in Overview
// - "Confluence Analysis" tab (separate) -> folded into Trades as collapsible
// - "Alert Analysis" tab (separate) -> merged into Monitoring tab
//
// REORGANIZED:
// - Overview = everything at a glance: KPIs, equity curve, price chart, config summary
// - Trades = full trade history with filters + advanced analysis (streaks, ToD, DoW)
// - Monitoring = alerts config + live chart + position status + alert history + accuracy
//
// WHY:
// A trader looking at a saved strategy asks 3 questions:
//   1. "Is this strategy good?" -> Overview (KPIs + equity curve)
//   2. "What are the trades doing?" -> Trades (history + analysis)
//   3. "Is it running live correctly?" -> Monitoring (alerts + live chart)
// Everything else is secondary detail that belongs inside those 3 buckets.
//
// HEADER:
// - Inline KPI strip in subtitle: "WR 54% | PF 2.05 | Daily R +1.95 | 224 trades"
// - Reduces need to scroll to see key metrics
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const strategy = {
  id: '1',
  name: 'NVDA LONG - Mass #2',
  symbol: 'NVDA',
  direction: 'LONG',
  timeframe: '1Min',
  session: 'RTH',
  trigger: '[C] EMA Bull Cross',
  exitTrigger: '4-bar exit',
  stop: 'Swing (5 bars, $0.03 pad)',
  target: '2R',
  confluence: ['5M-RVOL-HIGH', '1D-MACD_LINE-BULL'],
  generalConfluence: ['GEN-TIME_OF_DAY-MORNING'],
  forwardTesting: true,
  forwardTestStart: '2026-03-07',
  alertTracking: true,
  btDays: 90,
  btStart: '2025-12-07',
  btEnd: '2026-03-07',
};

const kpis = {
  trades: 224, winRate: 54.0, pf: 2.05, avgR: 0.42,
  totalR: 94.1, dailyR: 1.95, rSquared: 0.91, maxDD: -4.5,
  avgWinR: 1.85, avgLossR: -0.90,
};

const extKpis = {
  wins: 121, losses: 103,
  bestTrade: 5.23, worstTrade: -2.10,
  payoffRatio: 2.06, sharpe: 1.82, sortino: 2.65,
  calmar: 4.12, kelly: 0.182,
  recoveryFactor: 20.9, ulcerIndex: 0.032,
  maxConsecWins: 8, maxConsecLosses: 5,
  avgTradeDuration: '4m 23s',
};

const fwdKpis = {
  trades: 18, winRate: 52.1, pf: 1.82, avgR: 0.35,
  totalR: 6.3, dailyR: 0.45,
};

const mockTrades = [
  { id: 1, entryTime: '2026-03-18 09:42:15', exitTime: '2026-03-18 09:46:32', dir: 'LONG', entryPrice: 142.35, exitPrice: 142.98, pnlR: 1.42, execType: '[C]', exitReason: 'Signal' },
  { id: 2, entryTime: '2026-03-18 10:15:03', exitTime: '2026-03-18 10:19:18', dir: 'LONG', entryPrice: 143.10, exitPrice: 142.65, pnlR: -0.85, execType: '[C]', exitReason: 'Stop' },
  { id: 3, entryTime: '2026-03-18 11:02:44', exitTime: '2026-03-18 11:06:44', dir: 'LONG', entryPrice: 143.50, exitPrice: 144.12, pnlR: 2.07, execType: '[L0]', exitReason: 'Target' },
  { id: 4, entryTime: '2026-03-17 09:35:22', exitTime: '2026-03-17 09:39:22', dir: 'LONG', entryPrice: 141.80, exitPrice: 141.45, pnlR: -0.92, execType: '[C]', exitReason: 'Stop' },
  { id: 5, entryTime: '2026-03-17 10:22:11', exitTime: '2026-03-17 10:26:11', dir: 'LONG', entryPrice: 141.60, exitPrice: 142.25, pnlR: 1.63, execType: '[HM]', exitReason: 'Signal' },
  { id: 6, entryTime: '2026-03-17 11:45:08', exitTime: '2026-03-17 11:49:08', dir: 'LONG', entryPrice: 142.90, exitPrice: 143.55, pnlR: 1.95, execType: '[C]', exitReason: 'Target' },
  { id: 7, entryTime: '2026-03-17 13:10:33', exitTime: '2026-03-17 13:14:33', dir: 'LONG', entryPrice: 143.20, exitPrice: 142.80, pnlR: -0.67, execType: '[L0]', exitReason: 'Bar Count' },
  { id: 8, entryTime: '2026-03-14 09:48:55', exitTime: '2026-03-14 09:52:55', dir: 'LONG', entryPrice: 140.50, exitPrice: 141.30, pnlR: 2.38, execType: '[C]', exitReason: 'Signal' },
  { id: 9, entryTime: '2026-03-14 10:30:12', exitTime: '2026-03-14 10:34:12', dir: 'LONG', entryPrice: 141.10, exitPrice: 140.78, pnlR: -0.53, execType: '[HL]', exitReason: 'Stop' },
  { id: 10, entryTime: '2026-03-14 11:15:40', exitTime: '2026-03-14 11:19:40', dir: 'LONG', entryPrice: 140.95, exitPrice: 141.72, pnlR: 1.78, execType: '[C]', exitReason: 'Target' },
  { id: 11, entryTime: '2026-03-13 09:32:18', exitTime: '2026-03-13 09:36:18', dir: 'LONG', entryPrice: 139.80, exitPrice: 140.45, pnlR: 1.15, execType: '[C]', exitReason: 'Signal' },
  { id: 12, entryTime: '2026-03-13 10:55:22', exitTime: '2026-03-13 10:59:22', dir: 'LONG', entryPrice: 140.30, exitPrice: 139.90, pnlR: -0.78, execType: '[L0]', exitReason: 'Stop' },
];

const confluenceAnalysis = [
  { group: '5M-RVOL', state: 'HIGH', winRate: 58.2, trades: 142, avgR: 0.56 },
  { group: '5M-RVOL', state: 'LOW', winRate: 44.1, trades: 82, avgR: -0.12 },
  { group: '1D-MACD_LINE', state: 'BULL', winRate: 56.8, trades: 168, avgR: 0.48 },
  { group: '1D-MACD_LINE', state: 'BEAR', winRate: 42.3, trades: 56, avgR: -0.25 },
];

const todPerformance = [
  { hour: '09:30-10:00', trades: 28, winRate: 64.3, avgR: 0.42 },
  { hour: '10:00-11:00', trades: 31, winRate: 54.8, avgR: 0.18 },
  { hour: '11:00-12:00', trades: 22, winRate: 59.1, avgR: 0.31 },
  { hour: '13:00-14:00', trades: 24, winRate: 62.5, avgR: 0.38 },
  { hour: '14:00-15:00', trades: 15, winRate: 46.7, avgR: -0.12 },
  { hour: '15:00-16:00', trades: 7, winRate: 71.4, avgR: 0.65 },
];

const dowPerformance = [
  { day: 'Monday', trades: 26, winRate: 57.7, avgR: 0.22 },
  { day: 'Tuesday', trades: 29, winRate: 62.1, avgR: 0.41 },
  { day: 'Wednesday', trades: 28, winRate: 60.7, avgR: 0.35 },
  { day: 'Thursday', trades: 25, winRate: 52.0, avgR: 0.08 },
  { day: 'Friday', trades: 19, winRate: 57.9, avgR: 0.28 },
];

const mockAlerts = [
  { time: '2026-03-18 09:42:15', type: 'ENTRY', price: 142.35, status: 'Delivered' },
  { time: '2026-03-18 09:46:32', type: 'EXIT', price: 142.98, status: 'Delivered' },
  { time: '2026-03-18 10:15:03', type: 'ENTRY', price: 143.10, status: 'Delivered' },
  { time: '2026-03-17 09:35:22', type: 'ENTRY', price: 141.80, status: 'Delivered' },
  { time: '2026-03-17 09:39:22', type: 'EXIT', price: 141.45, status: 'Failed' },
];

const alertAccuracy = [
  { time: '2026-03-18 09:42', type: 'ENTRY', alertPrice: 142.35, btPrice: 142.37, drift: 0.02, match: true },
  { time: '2026-03-18 09:46', type: 'EXIT', alertPrice: 142.98, btPrice: 142.95, drift: 0.03, match: true },
  { time: '2026-03-18 10:15', type: 'ENTRY', alertPrice: 143.10, btPrice: 143.10, drift: 0.00, match: true },
  { time: '2026-03-17 09:35', type: 'ENTRY', alertPrice: 141.80, btPrice: 141.82, drift: 0.02, match: true },
  { time: '2026-03-17 09:39', type: 'EXIT', alertPrice: 141.45, btPrice: 141.48, drift: 0.03, match: false },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const execTypeColors: Record<string, string> = {
  '[C]': 'var(--blue)', '[L0]': 'var(--green)', '[L1]': '#8BC34A',
  '[HM]': 'var(--orange)', '[HL]': '#FF5722',
};

const exitReasonColors: Record<string, string> = {
  'Signal': 'var(--green)', 'Target': 'var(--green)',
  'Stop': 'var(--red)', 'Bar Count': '#009688',
};

function daysSince(dateStr: string): number {
  const d = new Date(dateStr);
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Chip({ label, color }: { label: string; color?: string }) {
  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded-md text-xs"
      style={{
        background: color ? `color-mix(in srgb, ${color} 12%, transparent)` : 'var(--bg-input)',
        border: `1px solid ${color ? `color-mix(in srgb, ${color} 30%, transparent)` : 'var(--border)'}`,
        color: color || 'var(--text-secondary)',
      }}
    >
      {label}
    </span>
  );
}

function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mt-4">
      <button
        className="w-full flex items-center justify-between text-sm font-medium py-2"
        style={{ color: 'var(--text-secondary)' }}
        onClick={() => setOpen(!open)}
      >
        <span>{title}</span>
        <span
          className="text-xs transition-transform"
          style={{
            color: 'var(--text-muted)',
            transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        >
          v
        </span>
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const TABS = ['Overview', 'Trades', 'Monitoring'];

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function StrategyDetailV3() {
  const [alertsEnabled, setAlertsEnabled] = useState(strategy.alertTracking);
  const [tradeFilter, setTradeFilter] = useState<'All' | 'Win' | 'Loss'>('All');
  const [execFilter, setExecFilter] = useState('All');

  const fwdDays = daysSince(strategy.forwardTestStart);

  const filteredTrades = useMemo(() => {
    return mockTrades.filter((t) => {
      if (tradeFilter === 'Win' && t.pnlR < 0) return false;
      if (tradeFilter === 'Loss' && t.pnlR >= 0) return false;
      if (execFilter !== 'All' && t.execType !== execFilter) return false;
      return true;
    });
  }, [tradeFilter, execFilter]);

  const btnStyle = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer' as const,
  };

  return (
    <div>
      {/* ---- Header with inline KPIs ---- */}
      <PageHeader
        title={strategy.name}
        subtitle={`WR ${kpis.winRate}% | PF ${kpis.pf.toFixed(2)} | Daily R +${kpis.dailyR.toFixed(2)} | ${kpis.trades} trades`}
        backHref="/strategies"
        actions={
          <>
            <button style={btnStyle}>Edit</button>
            <button style={btnStyle}>Clone</button>
            <button style={{ ...btnStyle, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }}>Delete</button>
          </>
        }
      />

      {/* ---- Strategy quick-info bar ---- */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          {strategy.symbol} | {strategy.direction} | {strategy.timeframe} | {strategy.session}
        </span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>|</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Entry: {strategy.trigger} | Exit: {strategy.exitTrigger} | Stop: {strategy.stop} | TP: {strategy.target}
        </span>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>|</span>
        {strategy.forwardTesting && (
          <span className="text-xs font-medium" style={{ color: 'var(--green)' }}>
            Forward Testing ({fwdDays}d)
          </span>
        )}
      </div>

      {/* ---- Confluence chips ---- */}
      <div className="flex flex-wrap items-center gap-1.5 mb-4">
        {strategy.confluence.map((c) => (
          <Chip key={c} label={c} color="var(--accent)" />
        ))}
        {strategy.generalConfluence.map((c) => (
          <Chip key={c} label={c} color="var(--blue)" />
        ))}
      </div>

      {/* ---- 3 Tabs ---- */}
      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* =========================================================== */}
            {/* OVERVIEW TAB                                                 */}
            {/* =========================================================== */}
            {tab === 'Overview' && (
              <div>
                {/* Primary KPIs */}
                <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-10 gap-2 mb-4">
                  <MetricCard label="Win Rate" value={`${kpis.winRate.toFixed(1)}%`} positive={kpis.winRate > 50} />
                  <MetricCard label="Profit Factor" value={kpis.pf.toFixed(2)} positive={kpis.pf > 1} />
                  <MetricCard label="Daily R" value={`+${kpis.dailyR.toFixed(2)}`} positive />
                  <MetricCard label="Trades" value={String(kpis.trades)} />
                  <MetricCard label="Avg R" value={`+${kpis.avgR.toFixed(2)}`} positive />
                  <MetricCard label="Total R" value={`+${kpis.totalR.toFixed(1)}`} positive />
                  <MetricCard label="R-Squared" value={kpis.rSquared.toFixed(2)} positive={kpis.rSquared > 0.7} />
                  <MetricCard label="Max DD" value={`${kpis.maxDD.toFixed(1)}R`} positive={false} />
                  <MetricCard label="Avg Win" value={`+${kpis.avgWinR.toFixed(2)}R`} positive />
                  <MetricCard label="Avg Loss" value={`${kpis.avgLossR.toFixed(2)}R`} />
                </div>

                {/* Equity Curve */}
                <Card className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium">Equity Curve</h4>
                    <div className="flex gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                      <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'var(--blue)' }} />Backtest</span>
                      {strategy.forwardTesting && (
                        <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'var(--orange)' }} />Forward ({fwdKpis.trades} trades)</span>
                      )}
                    </div>
                  </div>
                  <ChartPlaceholder label="Combined backtest (blue) + forward test (orange) equity curve" height={280} />
                </Card>

                {/* Price Chart */}
                <Card className="mb-4">
                  <ChartPlaceholder label="OHLC Price Chart with trade markers and indicator overlays" height={350} />
                </Card>

                {/* Forward Test Comparison (compact) */}
                {strategy.forwardTesting && (
                  <Card className="mb-4">
                    <h4 className="text-sm font-medium mb-3">Backtest vs Forward Test</h4>
                    <div className="grid grid-cols-5 gap-2 text-center">
                      {[
                        { m: 'Win Rate', bt: `${kpis.winRate}%`, fwd: `${fwdKpis.winRate}%` },
                        { m: 'PF', bt: kpis.pf.toFixed(2), fwd: fwdKpis.pf.toFixed(2) },
                        { m: 'Avg R', bt: `+${kpis.avgR.toFixed(2)}`, fwd: `+${fwdKpis.avgR.toFixed(2)}` },
                        { m: 'Daily R', bt: `+${kpis.dailyR.toFixed(2)}`, fwd: `+${fwdKpis.dailyR.toFixed(2)}` },
                        { m: 'Trades', bt: String(kpis.trades), fwd: String(fwdKpis.trades) },
                      ].map((row) => (
                        <div key={row.m} className="rounded-lg border p-2" style={{ background: 'var(--bg-input)', borderColor: 'var(--border)' }}>
                          <div className="text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>{row.m}</div>
                          <div className="text-xs" style={{ color: 'var(--blue)' }}>{row.bt}</div>
                          <div className="text-xs" style={{ color: 'var(--orange)' }}>{row.fwd}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {/* Extended KPIs (collapsible) */}
                <Card>
                  <CollapsibleSection title="Extended KPIs">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
                      <MetricCard label="Wins" value={String(extKpis.wins)} />
                      <MetricCard label="Losses" value={String(extKpis.losses)} />
                      <MetricCard label="Best Trade" value={`+${extKpis.bestTrade.toFixed(2)}R`} positive />
                      <MetricCard label="Worst Trade" value={`${extKpis.worstTrade.toFixed(2)}R`} />
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
                      <MetricCard label="Sharpe" value={extKpis.sharpe.toFixed(2)} positive={extKpis.sharpe > 1} />
                      <MetricCard label="Sortino" value={extKpis.sortino.toFixed(2)} positive={extKpis.sortino > 1} />
                      <MetricCard label="Calmar" value={extKpis.calmar.toFixed(2)} positive />
                      <MetricCard label="Kelly" value={`${(extKpis.kelly * 100).toFixed(1)}%`} />
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      <MetricCard label="Recovery Factor" value={extKpis.recoveryFactor.toFixed(1)} positive />
                      <MetricCard label="Max Consec Wins" value={String(extKpis.maxConsecWins)} />
                      <MetricCard label="Max Consec Losses" value={String(extKpis.maxConsecLosses)} />
                      <MetricCard label="Avg Duration" value={extKpis.avgTradeDuration} />
                    </div>
                  </CollapsibleSection>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TRADES TAB                                                   */}
            {/* =========================================================== */}
            {tab === 'Trades' && (
              <div>
                {/* Filters */}
                <div className="flex gap-3 mb-4 flex-wrap items-center">
                  <select
                    className="px-3 py-1.5 rounded text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={tradeFilter}
                    onChange={(e) => setTradeFilter(e.target.value as 'All' | 'Win' | 'Loss')}
                  >
                    <option value="All">All Trades</option>
                    <option value="Win">Wins Only</option>
                    <option value="Loss">Losses Only</option>
                  </select>
                  <select
                    className="px-3 py-1.5 rounded text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={execFilter}
                    onChange={(e) => setExecFilter(e.target.value)}
                  >
                    <option value="All">All Exec Types</option>
                    <option value="[C]">[C] Bar Close</option>
                    <option value="[L0]">[L0] Level Cross</option>
                    <option value="[HM]">[HM] Hybrid Market</option>
                    <option value="[HL]">[HL] Hybrid Limit</option>
                  </select>
                  <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    {filteredTrades.length} trades
                  </span>
                </div>

                {/* Trade table */}
                <Card className="mb-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '2px solid var(--border)' }}>
                          {['#', 'Entry Time', 'Exit Time', 'Dir', 'Entry $', 'Exit $', 'P&L (R)', 'Exec', 'Exit Reason'].map((h) => (
                            <th key={h} className="text-left py-2 px-2 font-medium whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredTrades.map((t) => (
                          <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="py-2 px-2" style={{ color: 'var(--text-muted)' }}>{t.id}</td>
                            <td className="py-2 px-2 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{t.entryTime}</td>
                            <td className="py-2 px-2 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{t.exitTime}</td>
                            <td className="py-2 px-2" style={{ color: t.dir === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{t.dir}</td>
                            <td className="py-2 px-2 font-mono" style={{ color: 'var(--text-secondary)' }}>${t.entryPrice.toFixed(2)}</td>
                            <td className="py-2 px-2 font-mono" style={{ color: 'var(--text-secondary)' }}>${t.exitPrice.toFixed(2)}</td>
                            <td className="py-2 px-2 font-mono font-medium" style={{ color: t.pnlR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                              {t.pnlR >= 0 ? '+' : ''}{t.pnlR.toFixed(2)}R
                            </td>
                            <td className="py-2 px-2">
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                                style={{
                                  color: execTypeColors[t.execType] || 'var(--text-secondary)',
                                  background: (execTypeColors[t.execType] || 'var(--text-secondary)') + '20',
                                }}
                              >
                                {t.execType}
                              </span>
                            </td>
                            <td className="py-2 px-2">
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded"
                                style={{
                                  color: exitReasonColors[t.exitReason] || 'var(--text-muted)',
                                  background: (exitReasonColors[t.exitReason] || 'var(--text-muted)') + '20',
                                }}
                              >
                                {t.exitReason}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Confluence Analysis (collapsible) */}
                <Card className="mb-4">
                  <CollapsibleSection title="Confluence Analysis">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border)' }}>
                            {['Group', 'State', 'Win Rate', 'Trades', 'Avg R'].map((h) => (
                              <th key={h} className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {confluenceAnalysis.map((row, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                              <td className="py-2 px-3">{row.group}</td>
                              <td className="py-2 px-3">
                                <span
                                  className="text-xs px-2 py-0.5 rounded-full"
                                  style={{
                                    color: row.state === 'HIGH' || row.state === 'BULL' ? 'var(--green)' : 'var(--red)',
                                    background: (row.state === 'HIGH' || row.state === 'BULL' ? 'var(--green)' : 'var(--red)') + '20',
                                  }}
                                >
                                  {row.state}
                                </span>
                              </td>
                              <td className="py-2 px-3" style={{ color: row.winRate >= 50 ? 'var(--green)' : 'var(--red)' }}>
                                {row.winRate.toFixed(1)}%
                              </td>
                              <td className="py-2 px-3">{row.trades}</td>
                              <td className="py-2 px-3" style={{ color: row.avgR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                                {row.avgR >= 0 ? '+' : ''}{row.avgR.toFixed(2)}R
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>
                </Card>

                {/* Time of Day Performance (collapsible) */}
                <Card className="mb-4">
                  <CollapsibleSection title="Time of Day Performance">
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border)' }}>
                            <th className="text-left px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Hour</th>
                            <th className="text-center px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Trades</th>
                            <th className="text-center px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Win Rate</th>
                            <th className="text-right px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg R</th>
                          </tr>
                        </thead>
                        <tbody>
                          {todPerformance.map((row) => (
                            <tr key={row.hour} style={{ borderBottom: '1px solid var(--border)' }}>
                              <td className="px-2 py-1.5" style={{ color: 'var(--text-secondary)' }}>{row.hour}</td>
                              <td className="px-2 py-1.5 text-center" style={{ color: 'var(--text-secondary)' }}>{row.trades}</td>
                              <td className="px-2 py-1.5 text-center" style={{ color: row.winRate > 55 ? 'var(--green)' : row.winRate < 50 ? 'var(--red)' : 'var(--text-secondary)' }}>
                                {row.winRate.toFixed(1)}%
                              </td>
                              <td className="px-2 py-1.5 text-right font-mono" style={{ color: row.avgR > 0 ? 'var(--green)' : 'var(--red)' }}>
                                {row.avgR >= 0 ? '+' : ''}{row.avgR.toFixed(2)}R
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>
                </Card>

                {/* Day of Week Performance (collapsible) */}
                <Card>
                  <CollapsibleSection title="Day of Week Performance">
                    <div className="grid grid-cols-5 gap-2">
                      {dowPerformance.map((row) => (
                        <div
                          key={row.day}
                          className="rounded-lg border p-3 text-center"
                          style={{
                            background: 'var(--bg-input)',
                            borderColor: row.avgR > 0.3 ? 'var(--green)' : row.avgR < 0 ? 'var(--red)' : 'var(--border)',
                            borderWidth: row.avgR > 0.3 || row.avgR < 0 ? 2 : 1,
                          }}
                        >
                          <div className="text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                            {row.day.slice(0, 3)}
                          </div>
                          <div className="text-sm font-semibold" style={{ color: row.avgR > 0 ? 'var(--green)' : 'var(--red)' }}>
                            {row.avgR >= 0 ? '+' : ''}{row.avgR.toFixed(2)}R
                          </div>
                          <div className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                            {row.trades} trades | {row.winRate.toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </CollapsibleSection>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* MONITORING TAB                                               */}
            {/* =========================================================== */}
            {tab === 'Monitoring' && (
              <div>
                {/* Alert Config + Position Status */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                  {/* Alert Config */}
                  <Card>
                    <h4 className="text-sm font-medium mb-3">Alert Configuration</h4>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Alerts Enabled</span>
                        <div
                          className="relative inline-block w-9 h-5 rounded-full cursor-pointer transition-colors"
                          style={{ background: alertsEnabled ? 'var(--accent)' : 'var(--bg-input)' }}
                          onClick={() => setAlertsEnabled(!alertsEnabled)}
                        >
                          <div
                            className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
                            style={{ background: 'white', left: alertsEnabled ? '18px' : '2px' }}
                          />
                        </div>
                      </div>
                      <div>
                        <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Webhook URL</label>
                        <input
                          type="text"
                          placeholder="https://discord.com/api/webhooks/..."
                          className="w-full px-3 py-2 rounded-lg text-sm"
                          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                          defaultValue=""
                        />
                      </div>
                    </div>
                  </Card>

                  {/* Position Status */}
                  <Card>
                    <h4 className="text-sm font-medium mb-3">Position Status</h4>
                    <div className="grid grid-cols-2 gap-2">
                      <MetricCard label="Status" value="Flat" />
                      <MetricCard label="Last Signal" value="EXIT @ $142.98" />
                      <MetricCard label="Signal Time" value="09:46:32" />
                      <MetricCard label="Current Price" value="$143.25" />
                    </div>
                  </Card>
                </div>

                {/* Live Chart */}
                <Card className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-sm font-medium">Live Chart</h4>
                    <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--green)' + '20', color: 'var(--green)' }}>
                      FLAT
                    </span>
                  </div>
                  <ChartPlaceholder label="Real-time OHLC chart with live candlestick formation and position markers" height={400} />
                </Card>

                {/* Alert History */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Recent Alerts</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          {['Time', 'Type', 'Price', 'Status'].map((h) => (
                            <th key={h} className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mockAlerts.map((a, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="py-2 px-3 text-xs">{a.time}</td>
                            <td className="py-2 px-3">
                              <span
                                className="text-xs px-2 py-0.5 rounded-full"
                                style={{
                                  color: a.type === 'ENTRY' ? 'var(--green)' : 'var(--orange)',
                                  background: (a.type === 'ENTRY' ? 'var(--green)' : 'var(--orange)') + '20',
                                }}
                              >
                                {a.type}
                              </span>
                            </td>
                            <td className="py-2 px-3">${a.price.toFixed(2)}</td>
                            <td className="py-2 px-3" style={{ color: a.status === 'Delivered' ? 'var(--green)' : 'var(--red)' }}>
                              {a.status}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Alert Accuracy (collapsible) */}
                <Card>
                  <CollapsibleSection title="Alert Accuracy Analysis">
                    <div className="grid grid-cols-4 gap-2 mb-4">
                      <MetricCard label="Total Alerts" value="47" />
                      <MetricCard label="Accuracy" value="89.4%" positive />
                      <MetricCard label="Entries Matched" value="21/23" />
                      <MetricCard label="Avg Drift" value="$0.03" />
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border)' }}>
                            {['Time', 'Type', 'Alert $', 'BT $', 'Drift', 'Match'].map((h) => (
                              <th key={h} className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {alertAccuracy.map((row, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                              <td className="py-2 px-3 text-xs">{row.time}</td>
                              <td className="py-2 px-3">
                                <span className="text-xs" style={{ color: row.type === 'ENTRY' ? 'var(--green)' : 'var(--orange)' }}>
                                  {row.type}
                                </span>
                              </td>
                              <td className="py-2 px-3">${row.alertPrice.toFixed(2)}</td>
                              <td className="py-2 px-3">${row.btPrice.toFixed(2)}</td>
                              <td className="py-2 px-3" style={{ color: row.drift > 0.02 ? 'var(--orange)' : 'var(--text-secondary)' }}>
                                ${row.drift.toFixed(2)}
                              </td>
                              <td className="py-2 px-3">
                                <span style={{ color: row.match ? 'var(--green)' : 'var(--red)' }}>
                                  {row.match ? 'Yes' : 'No'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CollapsibleSection>
                </Card>
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
