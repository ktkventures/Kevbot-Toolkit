'use client';

import { useState } from 'react';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

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
  confluence: ['5M-RVOL-HIGH (Default) [PB]', '1D-MACD_LINE-BULL (Default) [PB]'],
  generalConfluence: ['GEN-TIME_OF_DAY-MORNING'],
  forwardTesting: true,
  forwardTestStart: '2026-03-07',
  alertTracking: true,
  riskPerTrade: 100,
  startingBalance: 10000,
  btDays: 90,
  btStart: '2025-12-07',
  btEnd: '2026-03-07',
  barCount: 35280,
};

// Primary KPIs
const kpis = {
  trades: 224, winRate: 54.0, pf: 2.05, avgR: 0.42,
  totalR: 94.1, dailyR: 1.95, rSquared: 0.91, maxDD: -4.5,
  avgWinR: 1.85, avgLossR: -0.90,
};

// Secondary KPIs
const secKpis = {
  wins: 121, losses: 103,
  bestTrade: 5.23, worstTrade: -2.10,
  avgWin: 1.85, avgLoss: -0.90,
  payoffRatio: 2.06, gainPain: 3.42,
  sharpe: 1.82, sortino: 2.65, calmar: 4.12, kelly: 0.182,
  var95: -142.50, cvar95: -198.30, volatility: 12.4,
  skewness: 0.34, kurtosis: 2.18, tailRatio: 1.45,
  outlierWin: 2.8, outlierLoss: 2.3,
  recoveryFactor: 20.9, ulcerIndex: 0.032, serenityIndex: 2.14,
  longestDDTrades: 12, longestDDDays: 5,
  maxConsecWins: 8, maxConsecLosses: 5,
  avgTradeDuration: '4m 23s',
};

// Forward test KPIs
const fwdKpis = {
  trades: 18, winRate: 52.1, pf: 1.82, avgR: 0.35,
  totalR: 6.3, dailyR: 0.45,
};

// Mock trade history
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

// Mock confluence analysis
const confluenceAnalysis = [
  { group: '5M-RVOL', state: 'HIGH', winRate: 58.2, trades: 142, avgR: 0.56 },
  { group: '5M-RVOL', state: 'LOW', winRate: 44.1, trades: 82, avgR: -0.12 },
  { group: '1D-MACD_LINE', state: 'BULL', winRate: 56.8, trades: 168, avgR: 0.48 },
  { group: '1D-MACD_LINE', state: 'BEAR', winRate: 42.3, trades: 56, avgR: -0.25 },
];

// Mock monthly breakdown
const monthlyBreakdown = [
  { month: 'Jan 2026', trades: 42, winRate: 52.4, pf: 1.95, totalR: 18.5 },
  { month: 'Feb 2026', trades: 48, winRate: 56.2, pf: 2.18, totalR: 24.2 },
  { month: 'Mar 2026', trades: 38, winRate: 55.3, pf: 2.05, totalR: 19.8 },
  { month: 'Dec 2025', trades: 45, winRate: 51.1, pf: 1.72, totalR: 14.1 },
  { month: 'Nov 2025', trades: 51, winRate: 54.9, pf: 2.12, totalR: 17.5 },
];

// Mock alerts
const mockAlerts = [
  { time: '2026-03-18 09:42:15', type: 'ENTRY', price: 142.35, status: 'Delivered' },
  { time: '2026-03-18 09:46:32', type: 'EXIT', price: 142.98, status: 'Delivered' },
  { time: '2026-03-18 10:15:03', type: 'ENTRY', price: 143.10, status: 'Delivered' },
  { time: '2026-03-17 09:35:22', type: 'ENTRY', price: 141.80, status: 'Delivered' },
  { time: '2026-03-17 09:39:22', type: 'EXIT', price: 141.45, status: 'Failed' },
];

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

const execTypeColors: Record<string, string> = {
  '[C]': 'var(--blue)',
  '[L0]': 'var(--green)',
  '[L1]': '#8BC34A',
  '[HM]': 'var(--orange)',
  '[HL]': '#FF5722',
};

const exitReasonColors: Record<string, string> = {
  'Signal': 'var(--green)',
  'Target': 'var(--green)',
  'Stop': 'var(--red)',
  'Bar Count': '#009688',
  'HM Unconfirmed': 'var(--orange)',
  'HL Unconfirmed': 'var(--orange)',
};

function daysSince(dateStr: string): number {
  const d = new Date(dateStr);
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

/* ========================================================================= */
/* TABS                                                                        */
/* ========================================================================= */

const TABS = [
  'Equity & KPIs',
  'Extended KPIs',
  'Price Chart',
  'Live Chart',
  'Trade History',
  'Confluence Analysis',
  'Configuration',
  'Alerts',
  'Alert Analysis',
];

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function StrategyDetailV2() {
  const [alertsEnabled, setAlertsEnabled] = useState(strategy.alertTracking);
  const [alertOnEntry, setAlertOnEntry] = useState(true);
  const [alertOnExit, setAlertOnExit] = useState(true);
  const [tradeFilter, setTradeFilter] = useState<'All' | 'Win' | 'Loss'>('All');
  const [execFilter, setExecFilter] = useState('All');

  const fwdDays = daysSince(strategy.forwardTestStart);

  const btnSecondary = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer' as const,
  };

  const filteredTrades = mockTrades.filter((t) => {
    if (tradeFilter === 'Win' && t.pnlR < 0) return false;
    if (tradeFilter === 'Loss' && t.pnlR >= 0) return false;
    if (execFilter !== 'All' && t.execType !== execFilter) return false;
    return true;
  });

  return (
    <div>
      {/* ---- Header ---- */}
      <PageHeader
        title={strategy.name}
        subtitle={`${strategy.symbol} | ${strategy.direction} | ${strategy.timeframe} | ${strategy.session} | Entry: ${strategy.trigger} | Exit: ${strategy.exitTrigger} | Stop: ${strategy.stop} | Target: ${strategy.target}`}
        backHref="/strategies"
        actions={
          <>
            <button style={btnSecondary}>Refresh</button>
            <button style={btnSecondary}>Edit</button>
            <button style={btnSecondary}>Clone</button>
            <button style={{ ...btnSecondary, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }}>Delete</button>
            <button style={btnSecondary}>Reset Alerts</button>
          </>
        }
      />

      {/* ---- Confluence conditions ---- */}
      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
        TF: {strategy.confluence.join(' + ')}
        {strategy.generalConfluence.length > 0 && ` | General: ${strategy.generalConfluence.join(' + ')}`}
      </p>

      {/* ---- Status bar ---- */}
      <div className="flex items-center gap-4 mb-5">
        <span className="text-sm font-medium" style={{ color: 'var(--green)' }}>
          Forward Testing ({fwdDays}d)
        </span>
        <label className="text-xs cursor-pointer flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
          <div
            className="relative inline-block w-8 h-4 rounded-full transition-colors cursor-pointer"
            style={{ background: alertsEnabled ? 'var(--accent)' : 'var(--bg-input)' }}
            onClick={() => setAlertsEnabled(!alertsEnabled)}
          >
            <div
              className="absolute top-0.5 w-3 h-3 rounded-full transition-all"
              style={{ background: 'white', left: alertsEnabled ? '16px' : '2px' }}
            />
          </div>
          Track Alerts
        </label>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Backtest: {strategy.btStart} to {strategy.btEnd} ({strategy.barCount.toLocaleString()} bars)
        </span>
      </div>

      {/* ---- Tabs ---- */}
      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* ============================================================= */}
            {/* TAB 1: EQUITY & KPIs                                          */}
            {/* ============================================================= */}
            {tab === 'Equity & KPIs' && (
              <div>
                {/* Primary KPIs */}
                <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-3 mb-4">
                  <MetricCard label="Win Rate" value={`${kpis.winRate.toFixed(1)}%`} />
                  <MetricCard label="Profit Factor" value={kpis.pf.toFixed(2)} />
                  <MetricCard label="Daily R" value={`+${kpis.dailyR.toFixed(2)}`} />
                  <MetricCard label="Trades" value={String(kpis.trades)} />
                  <MetricCard label="Avg Win (R)" value={`+${kpis.avgWinR.toFixed(2)}`} />
                  <MetricCard label="Avg Loss (R)" value={kpis.avgLossR.toFixed(2)} />
                </div>

                {/* Secondary KPIs */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
                  <MetricCard label="Max DD" value={`${kpis.maxDD.toFixed(1)}R`} />
                  <MetricCard label="Recovery Factor" value={secKpis.recoveryFactor.toFixed(1)} />
                  <MetricCard label="Sharpe" value={secKpis.sharpe.toFixed(2)} />
                  <MetricCard label="Expectancy" value={`+${kpis.avgR.toFixed(2)}R`} />
                </div>

                {/* Backtest equity curve */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Backtest Equity Curve</h4>
                  <ChartPlaceholder label="Cumulative R equity curve (blue line) with max drawdown shading" height={280} />
                </Card>

                {/* Forward test equity curve */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Forward Test Equity Curve</h4>
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                    Forward testing since {strategy.forwardTestStart} ({fwdDays} days, {fwdKpis.trades} trades)
                  </p>
                  <ChartPlaceholder label="Forward test equity curve (orange line)" height={200} />
                </Card>

                {/* Backtest vs Forward comparison */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Backtest vs Forward Test Comparison</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          <th className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>Metric</th>
                          <th className="text-right py-2 px-3" style={{ color: 'var(--blue)' }}>Backtest</th>
                          <th className="text-right py-2 px-3" style={{ color: 'var(--orange)' }}>Forward</th>
                          <th className="text-right py-2 px-3" style={{ color: 'var(--text-muted)' }}>Delta</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { m: 'Win Rate', bt: `${kpis.winRate}%`, fwd: `${fwdKpis.winRate}%`, d: `${(fwdKpis.winRate - kpis.winRate).toFixed(1)}%` },
                          { m: 'Profit Factor', bt: kpis.pf.toFixed(2), fwd: fwdKpis.pf.toFixed(2), d: (fwdKpis.pf - kpis.pf).toFixed(2) },
                          { m: 'Avg R', bt: `+${kpis.avgR.toFixed(2)}`, fwd: `+${fwdKpis.avgR.toFixed(2)}`, d: (fwdKpis.avgR - kpis.avgR).toFixed(2) },
                          { m: 'Daily R', bt: `+${kpis.dailyR.toFixed(2)}`, fwd: `+${fwdKpis.dailyR.toFixed(2)}`, d: (fwdKpis.dailyR - kpis.dailyR).toFixed(2) },
                          { m: 'Trades', bt: String(kpis.trades), fwd: String(fwdKpis.trades), d: '--' },
                        ].map((row, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="py-2 px-3" style={{ color: 'var(--text-secondary)' }}>{row.m}</td>
                            <td className="py-2 px-3 text-right">{row.bt}</td>
                            <td className="py-2 px-3 text-right">{row.fwd}</td>
                            <td className="py-2 px-3 text-right" style={{ color: parseFloat(row.d) >= 0 ? 'var(--green)' : 'var(--red)' }}>{row.d}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 2: EXTENDED KPIs                                           */}
            {/* ============================================================= */}
            {tab === 'Extended KPIs' && (
              <div>
                {/* Performance */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Performance</h4>
                  <div className="grid grid-cols-4 gap-3 mb-4">
                    <MetricCard label="Wins" value={String(secKpis.wins)} />
                    <MetricCard label="Losses" value={String(secKpis.losses)} />
                    <MetricCard label="Best Trade" value={`+${secKpis.bestTrade.toFixed(2)}R`} />
                    <MetricCard label="Worst Trade" value={`${secKpis.worstTrade.toFixed(2)}R`} />
                  </div>
                  <div className="grid grid-cols-4 gap-3">
                    <MetricCard label="Avg Win" value={`+${secKpis.avgWin.toFixed(2)}R`} />
                    <MetricCard label="Avg Loss" value={`${secKpis.avgLoss.toFixed(2)}R`} />
                    <MetricCard label="Payoff Ratio" value={secKpis.payoffRatio.toFixed(2)} />
                    <MetricCard label="Gain/Pain" value={secKpis.gainPain.toFixed(2)} />
                  </div>
                </Card>

                {/* Risk-Adjusted */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Risk-Adjusted</h4>
                  <div className="grid grid-cols-4 gap-3 mb-4">
                    <MetricCard label="Sharpe Ratio" value={secKpis.sharpe.toFixed(2)} />
                    <MetricCard label="Sortino Ratio" value={secKpis.sortino.toFixed(2)} />
                    <MetricCard label="Calmar Ratio" value={secKpis.calmar.toFixed(2)} />
                    <MetricCard label="Kelly Criterion" value={`${(secKpis.kelly * 100).toFixed(1)}%`} />
                  </div>
                  <div className="grid grid-cols-4 gap-3">
                    <MetricCard label="Daily VaR (95%)" value={`$${secKpis.var95.toFixed(2)}`} />
                    <MetricCard label="CVaR (95%)" value={`$${secKpis.cvar95.toFixed(2)}`} />
                    <MetricCard label="Volatility" value={`${secKpis.volatility.toFixed(1)}%`} />
                    <MetricCard label="R-Squared" value={kpis.rSquared.toFixed(2)} />
                  </div>
                </Card>

                {/* Distribution */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Distribution</h4>
                  <div className="grid grid-cols-4 gap-3">
                    <MetricCard label="Skewness" value={secKpis.skewness.toFixed(2)} />
                    <MetricCard label="Kurtosis" value={secKpis.kurtosis.toFixed(2)} />
                    <MetricCard label="Tail Ratio" value={secKpis.tailRatio.toFixed(2)} />
                    <MetricCard label="Outlier Win" value={`${secKpis.outlierWin.toFixed(1)}x`} />
                  </div>
                </Card>

                {/* Drawdown */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Drawdown</h4>
                  <div className="grid grid-cols-4 gap-3 mb-4">
                    <MetricCard label="Max R DD" value={`${kpis.maxDD.toFixed(1)}R`} />
                    <MetricCard label="Recovery Factor" value={secKpis.recoveryFactor.toFixed(1)} />
                    <MetricCard label="Ulcer Index" value={secKpis.ulcerIndex.toFixed(3)} />
                    <MetricCard label="Serenity Index" value={secKpis.serenityIndex.toFixed(2)} />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Longest DD (trades)" value={String(secKpis.longestDDTrades)} />
                    <MetricCard label="Longest DD (days)" value={String(secKpis.longestDDDays)} />
                  </div>
                </Card>

                {/* Streaks */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Streaks</h4>
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard label="Max Consec. Wins" value={String(secKpis.maxConsecWins)} />
                    <MetricCard label="Max Consec. Losses" value={String(secKpis.maxConsecLosses)} />
                    <MetricCard label="Avg Trade Duration" value={secKpis.avgTradeDuration} />
                  </div>
                </Card>

                {/* Monthly Breakdown */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Monthly Breakdown</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          {['Month', 'Trades', 'Win Rate', 'PF', 'Total R'].map((h) => (
                            <th key={h} className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {monthlyBreakdown.map((row, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="py-2 px-3">{row.month}</td>
                            <td className="py-2 px-3">{row.trades}</td>
                            <td className="py-2 px-3">{row.winRate.toFixed(1)}%</td>
                            <td className="py-2 px-3">{row.pf.toFixed(2)}</td>
                            <td className="py-2 px-3" style={{ color: row.totalR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                              +{row.totalR.toFixed(1)}R
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 3: PRICE CHART                                             */}
            {/* ============================================================= */}
            {tab === 'Price Chart' && (
              <div>
                <Card>
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-medium">OHLC Price Chart with Trade Markers</h4>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                      Shows last 90 days of backtested data
                    </span>
                  </div>
                  <ChartPlaceholder
                    label="TradingView-style OHLC chart with candlesticks, EMA overlays, trade entry (+) and exit (x) markers, and confluence heatmap pane"
                    height={500}
                  />
                  <div className="flex gap-4 mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span><span style={{ color: 'var(--green)' }}>+</span> Entry (long)</span>
                    <span><span style={{ color: 'var(--green)' }}>x</span> Exit (win)</span>
                    <span><span style={{ color: 'var(--red)' }}>x</span> Exit (loss)</span>
                    <span><span style={{ color: 'var(--orange)' }}>x</span> Exit (HM/HL unconfirmed)</span>
                    <span><span style={{ color: '#009688' }}>x</span> Exit (bar count)</span>
                  </div>
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 4: LIVE CHART                                              */}
            {/* ============================================================= */}
            {tab === 'Live Chart' && (
              <div>
                <Card className="mb-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-medium">Live Chart</h4>
                    <div className="flex items-center gap-3">
                      <span className="text-xs px-2 py-1 rounded-full" style={{ background: 'var(--green)' + '20', color: 'var(--green)' }}>
                        FLAT
                      </span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Auto-refreshing via Polygon WS
                      </span>
                    </div>
                  </div>
                  <ChartPlaceholder label="Real-time OHLC chart with live candlestick formation, indicators, and position markers" height={500} />
                </Card>

                <Card>
                  <h4 className="text-sm font-medium mb-3">Position Status</h4>
                  <div className="grid grid-cols-4 gap-3">
                    <MetricCard label="Status" value="Flat" />
                    <MetricCard label="Last Signal" value="EXIT @ 142.98" />
                    <MetricCard label="Signal Time" value="09:46:32" />
                    <MetricCard label="Current Price" value="$143.25" />
                  </div>
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 5: TRADE HISTORY                                           */}
            {/* ============================================================= */}
            {tab === 'Trade History' && (
              <div>
                <Card>
                  {/* Filters */}
                  <div className="flex gap-4 mb-4">
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
                      <option value="[L1]">[L1] Level Cross</option>
                      <option value="[HM]">[HM] Hybrid Market</option>
                      <option value="[HL]">[HL] Hybrid Limit</option>
                    </select>
                    <span className="text-sm self-center" style={{ color: 'var(--text-muted)' }}>
                      {filteredTrades.length} trades
                    </span>
                  </div>

                  {/* Trade table */}
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '2px solid var(--border)' }}>
                          {['#', 'Entry Time', 'Exit Time', 'Dir', 'Entry', 'Exit', 'P&L (R)', 'Exec', 'Exit Reason'].map((h) => (
                            <th key={h} className="text-left py-2 px-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredTrades.map((t) => (
                          <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>{t.id}</td>
                            <td className="py-2 px-2 text-xs">{t.entryTime}</td>
                            <td className="py-2 px-2 text-xs">{t.exitTime}</td>
                            <td className="py-2 px-2 text-xs">{t.dir}</td>
                            <td className="py-2 px-2 text-xs">${t.entryPrice.toFixed(2)}</td>
                            <td className="py-2 px-2 text-xs">${t.exitPrice.toFixed(2)}</td>
                            <td
                              className="py-2 px-2 text-xs font-medium"
                              style={{ color: t.pnlR >= 0 ? 'var(--green)' : 'var(--red)' }}
                            >
                              {t.pnlR >= 0 ? '+' : ''}{t.pnlR.toFixed(2)}R
                            </td>
                            <td className="py-2 px-2">
                              <span
                                className="text-xs px-1.5 py-0.5 rounded font-mono"
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
                                className="text-xs px-1.5 py-0.5 rounded"
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
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 6: CONFLUENCE ANALYSIS                                     */}
            {/* ============================================================= */}
            {tab === 'Confluence Analysis' && (
              <div>
                {/* Per-group state at entry analysis */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Per-Group State at Entry</h4>
                  <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                    Which confluence states produce the best wins
                  </p>
                  <div style={{ overflowX: 'auto' }}>
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
                </Card>

                {/* State distribution */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">State Distribution at Entry</h4>
                  <ChartPlaceholder label="Donut chart showing distribution of confluence states at trade entry time" height={300} />
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 7: CONFIGURATION                                           */}
            {/* ============================================================= */}
            {tab === 'Configuration' && (
              <div>
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-4">Strategy Setup</h4>
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      {[
                        { label: 'Ticker', value: strategy.symbol },
                        { label: 'Direction', value: strategy.direction },
                        { label: 'Timeframe', value: strategy.timeframe },
                        { label: 'Session', value: strategy.session },
                        { label: 'Risk Per Trade', value: `$${strategy.riskPerTrade}` },
                        { label: 'Starting Balance', value: `$${strategy.startingBalance.toLocaleString()}` },
                      ].map((item, i) => (
                        <div key={i} className="flex justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                          <span className="text-sm font-medium">{item.value}</span>
                        </div>
                      ))}
                    </div>
                    <div>
                      {[
                        { label: 'Entry Trigger', value: strategy.trigger },
                        { label: 'Exit Trigger', value: strategy.exitTrigger },
                        { label: 'Stop Loss', value: strategy.stop },
                        { label: 'Take Profit', value: strategy.target },
                        { label: 'Backtest Days', value: String(strategy.btDays) },
                        { label: 'Forward Test Start', value: strategy.forwardTestStart },
                      ].map((item, i) => (
                        <div key={i} className="flex justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                          <span className="text-sm font-medium">{item.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>

                <Card>
                  <h4 className="text-sm font-medium mb-4">Confluence Conditions</h4>
                  {strategy.confluence.length > 0 && (
                    <div className="mb-3">
                      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Timeframe Confluence</p>
                      {strategy.confluence.map((c, i) => (
                        <div key={i} className="flex items-center gap-2 py-1.5" style={{ borderBottom: '1px solid var(--border)' }}>
                          <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
                          <span className="text-sm">{c}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  {strategy.generalConfluence.length > 0 && (
                    <div>
                      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>General Confluence</p>
                      {strategy.generalConfluence.map((c, i) => (
                        <div key={i} className="flex items-center gap-2 py-1.5" style={{ borderBottom: '1px solid var(--border)' }}>
                          <span className="w-2 h-2 rounded-full" style={{ background: 'var(--blue)' }} />
                          <span className="text-sm">{c}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 8: ALERTS                                                  */}
            {/* ============================================================= */}
            {tab === 'Alerts' && (
              <div>
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-4">Alert Configuration</h4>
                  <div className="space-y-4">
                    {/* Enable/disable toggle */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Alerts Enabled</span>
                      <div
                        className="relative inline-block w-10 h-5 rounded-full cursor-pointer transition-colors"
                        style={{ background: alertsEnabled ? 'var(--accent)' : 'var(--bg-input)' }}
                        onClick={() => setAlertsEnabled(!alertsEnabled)}
                      >
                        <div
                          className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
                          style={{ background: 'white', left: alertsEnabled ? '22px' : '2px' }}
                        />
                      </div>
                    </div>

                    {/* Alert on entry / exit */}
                    <div className="flex gap-6">
                      <label className="flex items-center gap-2 text-sm cursor-pointer">
                        <input
                          type="checkbox"
                          checked={alertOnEntry}
                          onChange={(e) => setAlertOnEntry(e.target.checked)}
                          style={{ accentColor: 'var(--accent)' }}
                        />
                        Alert on Entry
                      </label>
                      <label className="flex items-center gap-2 text-sm cursor-pointer">
                        <input
                          type="checkbox"
                          checked={alertOnExit}
                          onChange={(e) => setAlertOnExit(e.target.checked)}
                          style={{ accentColor: 'var(--accent)' }}
                        />
                        Alert on Exit
                      </label>
                    </div>

                    {/* Webhook URL */}
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

                {/* Last alerts */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Recent Alerts</h4>
                  <div style={{ overflowX: 'auto' }}>
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
                            <td className="py-2 px-3">
                              <span style={{ color: a.status === 'Delivered' ? 'var(--green)' : 'var(--red)' }}>
                                {a.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* ============================================================= */}
            {/* TAB 9: ALERT ANALYSIS                                          */}
            {/* ============================================================= */}
            {tab === 'Alert Analysis' && (
              <div>
                {/* Summary metrics */}
                <div className="grid grid-cols-4 gap-3 mb-4">
                  <MetricCard label="Alerts Since Monitoring" value="47" />
                  <MetricCard label="Alert Accuracy" value="89.4%" delta="+2.1% vs last week" positive />
                  <MetricCard label="Position Health" value="Good" />
                  <MetricCard label="Discrepancies" value="2" />
                </div>

                {/* Position health */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Position Health</h4>
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <MetricCard label="Entries Matched" value="21/23" />
                    <MetricCard label="Exits Matched" value="20/22" />
                    <MetricCard label="Price Drift Avg" value="$0.03" />
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Alert entry prices compared to backtest entry prices. Small deviations are expected
                    due to bar-close vs WebSocket timing differences.
                  </p>
                </Card>

                {/* Recent alert accuracy */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Recent Alert Accuracy</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          {['Alert Time', 'Type', 'Alert Price', 'BT Price', 'Drift', 'Match'].map((h) => (
                            <th key={h} className="text-left py-2 px-3" style={{ color: 'var(--text-muted)' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { time: '2026-03-18 09:42', type: 'ENTRY', alertPrice: 142.35, btPrice: 142.37, drift: 0.02, match: true },
                          { time: '2026-03-18 09:46', type: 'EXIT', alertPrice: 142.98, btPrice: 142.95, drift: 0.03, match: true },
                          { time: '2026-03-18 10:15', type: 'ENTRY', alertPrice: 143.10, btPrice: 143.10, drift: 0.00, match: true },
                          { time: '2026-03-17 09:35', type: 'ENTRY', alertPrice: 141.80, btPrice: 141.82, drift: 0.02, match: true },
                          { time: '2026-03-17 09:39', type: 'EXIT', alertPrice: 141.45, btPrice: 141.48, drift: 0.03, match: false },
                        ].map((row, i) => (
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
                </Card>
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
