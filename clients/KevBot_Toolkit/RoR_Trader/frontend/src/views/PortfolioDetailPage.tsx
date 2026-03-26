'use client';
/**
 * Portfolio Detail — Clean API-first page.
 * Visual design from V5, data wired to real API hooks. No mock data.
 *
 * V5 QA fixes:
 * - Warning banner: "Dashboard metrics reflect visible strategies from the Strategies tab"
 * - Planned/Executed data toggle + TQ filter dropdown in Live Dashboard
 * - Performance tab: Worst-Case Analysis, Daily P&L vs Limits, Capital Deployed modules
 * - Strategy variable summary above tabs (sigma + variable details per strategy)
 */
import { useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import {
  usePortfolio, usePortfolioCompute, usePortfolioTrades,
  usePortfolioAnomalies, usePortfolioAccount,
} from '@/hooks/queries/usePortfolios';

/* -- Constants -- */
const TABS = ['Live Dashboard', 'Performance', 'Strategies', 'Prop Firm Check', 'Account', 'Webhooks'];
const PULSE_CSS = `@keyframes pulse{0%,100%{transform:scale(1);opacity:.5}50%{transform:scale(2.2);opacity:0}}`;
const btnSec: React.CSSProperties = { background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', padding: '6px 14px', borderRadius: '8px', fontSize: '.875rem', cursor: 'pointer' };

/* -- Helpers -- */
const fmtMoney = (v?: number | null) => v == null ? '--' : `$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const fmtPct = (v?: number | null) => v == null ? '--' : `${v.toFixed(1)}%`;
const Skeleton = ({ h = 200 }: { h?: number }) => <div className="rounded-xl animate-pulse" style={{ background: 'var(--bg-input)', height: h }} />;
const ProgressBar = ({ pct, color = 'var(--accent)' }: { pct: number; color?: string }) => (
  <div className="w-full rounded-full overflow-hidden" style={{ background: 'var(--bg-input)', height: 8 }}>
    <div className="rounded-full transition-all" style={{ width: `${Math.min(100, Math.max(0, pct))}%`, height: '100%', background: color }} />
  </div>
);
const StatusBadge = ({ status }: { status: string }) => {
  const c = ({ Healthy: { bg: 'var(--green-muted)', t: 'var(--green)' }, Warning: { bg: 'var(--orange-muted)', t: 'var(--orange)' }, Critical: { bg: 'var(--red-muted)', t: 'var(--red)' } } as Record<string, { bg: string; t: string }>)[status] || { bg: 'var(--accent-muted)', t: 'var(--accent)' };
  return <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background: c.bg, color: c.t }}>{status}</span>;
};
const TH = ({ headers }: { headers: string[] }) => (
  <tr>{headers.map((h) => <th key={h} className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>)}</tr>
);

interface TabDataProps { portfolio?: any; compute?: any; trades?: any[]; anomalies?: any; account?: any }

/* ======================================================================== */
/* Tab: Live Dashboard                                                       */
/* ======================================================================== */
function LiveDashboardTab({ compute, trades, anomalies }: TabDataProps) {
  const [anomalyTab, setAnomalyTab] = useState('All');
  const [dataMode, setDataMode] = useState<'Planned' | 'Executed'>('Planned');
  const kpis = compute?.live_dashboard_kpis;
  const openPositions: any[] = compute?.open_positions ?? [];
  const buyingPower = compute?.buying_power;
  const anomalyList: any[] = anomalies?.items ?? [];
  const tradeHistory: any[] = trades ?? [];

  return (
    <div>
      {/* Warning banner + data mode toggle + TQ filter (V5) */}
      <div className="flex items-center justify-between mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--accent-muted)', border: '1px solid rgba(99,102,241,0.2)' }}>
        <span className="text-xs" style={{ color: 'var(--accent)' }}>
          Dashboard metrics reflect <strong>visible strategies</strong> from the Strategies tab.
        </span>
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Data:</span>
            {(['Planned', 'Executed'] as const).map((mode) => (
              <button
                key={mode}
                className="text-xs px-2.5 py-1 rounded-full"
                style={{
                  background: dataMode === mode ? 'var(--bg-card)' : 'transparent',
                  color: dataMode === mode ? 'var(--text-primary)' : 'var(--text-muted)',
                  border: dataMode === mode ? '1px solid var(--border)' : '1px solid transparent',
                  cursor: 'pointer',
                  fontWeight: dataMode === mode ? 600 : 400,
                }}
                onClick={() => setDataMode(mode)}
                title={mode === 'Planned' ? 'Planned quantity -- assumes unlimited buying power' : 'Executed quantity -- actual transactions'}
              >
                {mode}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>TQ:</span>
            <select
              className="text-xs px-2 py-1 rounded-full"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', cursor: 'pointer' }}
              defaultValue="None"
            >
              <option value="None">None</option>
              <option value="ttp">Trade The Pool</option>
              <option value="ftmo">FTMO</option>
              <option value="topstep">Topstep</option>
              <option value="custom">My Custom Rules</option>
            </select>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-5 gap-3 mb-6">
        <MetricCard label="Alert Trades" value={kpis?.alert_trades != null ? String(kpis.alert_trades) : '--'} />
        <MetricCard label="Win Rate" value={kpis?.win_rate != null ? fmtPct(kpis.win_rate) : '--'} />
        <MetricCard label="Total P&L" value={kpis?.total_pnl != null ? fmtMoney(kpis.total_pnl) : '--'} positive={kpis?.total_pnl >= 0} delta={kpis?.total_pnl != null ? (kpis.total_pnl >= 0 ? '+' : '') + fmtMoney(kpis.total_pnl) : undefined} />
        <MetricCard label="Expected P&L" value={kpis?.expected_pnl != null ? fmtMoney(kpis.expected_pnl) : '--'} delta="benchmark" />
        <MetricCard label="vs Plan" value={kpis?.vs_plan != null ? fmtMoney(kpis.vs_plan) : '--'} positive={kpis?.vs_plan >= 0} delta={kpis?.vs_plan != null ? (kpis.vs_plan >= 0 ? 'above plan' : 'below plan') : undefined} />
      </div>
      {/* Performance vs Plan */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Performance vs Plan</h3>
        <ChartPlaceholder label="Benchmark equity with confidence bands overlaid with actual equity curve" height={400} />
      </Card>
      {/* Open Positions */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Open Positions <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>{openPositions.length}</span></h3>
        {openPositions.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
              <thead><TH headers={['Strategy', 'Symbol', 'Dir', 'Entry', 'Current', 'Unrealized', 'Duration']} /></thead>
              <tbody>
                {openPositions.map((p: any, i: number) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="py-2.5 px-3 font-medium">{p.strategy}</td>
                    <td className="py-2.5 px-3">{p.symbol}</td>
                    <td className="py-2.5 px-3">{p.direction}</td>
                    <td className="py-2.5 px-3" style={{ color: 'var(--text-secondary)' }}>{fmtMoney(p.entry_price)}</td>
                    <td className="py-2.5 px-3" style={{ color: 'var(--text-secondary)' }}>{fmtMoney(p.current_price)}</td>
                    <td className="py-2.5 px-3 font-medium" style={{ color: (p.unrealized_pnl ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>{fmtMoney(p.unrealized_pnl)}</td>
                    <td className="py-2.5 px-3 text-xs" style={{ color: 'var(--text-muted)' }}>{p.duration ?? '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-sm py-4 text-center" style={{ color: 'var(--text-muted)' }}>No open positions</p>}
      </Card>
      {/* Buying Power */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Buying Power Tracker</h3>
        {buyingPower ? (
          <>
            <div className="grid grid-cols-4 gap-4 mb-4">
              {[{ l: 'Starting Balance', v: fmtMoney(buyingPower.starting_balance) }, { l: 'Current Available', v: fmtMoney(buyingPower.available), c: 'var(--green)' }, { l: 'Currently Allocated', v: fmtMoney(buyingPower.allocated), c: 'var(--orange)' }, { l: 'Utilization', v: fmtPct(buyingPower.utilization) }].map((m) => (
                <div key={m.l}><p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.l}</p><p className="text-lg font-semibold" style={m.c ? { color: m.c } : undefined}>{m.v}</p></div>
              ))}
            </div>
            <ChartPlaceholder label="24-hour buying power chart" height={180} />
          </>
        ) : <ChartPlaceholder label="Buying power tracker -- awaiting data" height={180} />}
      </Card>
      {/* Anomalies */}
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Anomaly Detection {anomalyList.length > 0 && <span className="ml-2 text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>{anomalyList.length}</span>}
          </h3>
          <div className="flex gap-1">
            {['All', 'Alert Issues', 'Performance'].map((t) => (
              <button key={t} className="text-xs px-2.5 py-1 rounded-full" style={{ background: anomalyTab === t ? 'var(--accent-muted)' : 'var(--bg-input)', color: anomalyTab === t ? 'var(--accent)' : 'var(--text-muted)', border: anomalyTab === t ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }} onClick={() => setAnomalyTab(t)}>{t}</button>
            ))}
          </div>
        </div>
        {anomalyList.length > 0 ? (
          <div className="space-y-2">
            {anomalyList.filter((a: any) => anomalyTab === 'All' || a.category === anomalyTab).map((a: any, i: number) => {
              const crit = a.severity === 'critical';
              return (
                <div key={i} className="flex items-start gap-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-xs px-2 py-0.5 rounded font-mono font-medium flex-shrink-0" style={{ background: crit ? 'var(--red-muted)' : 'rgba(255,152,0,0.12)', color: crit ? 'var(--red)' : 'var(--orange)' }}>{crit ? 'CRITICAL' : 'WARNING'}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium mb-0.5" style={{ color: crit ? 'var(--red)' : 'var(--orange)' }}>{a.type}</p>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{a.message}</p>
                  </div>
                  <span className="text-xs flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{a.time}</span>
                </div>
              );
            })}
          </div>
        ) : <p className="text-sm py-4 text-center" style={{ color: 'var(--text-muted)' }}>No anomalies detected</p>}
      </Card>
      {/* Trade History */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Trade History</h3>
        {tradeHistory.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="w-full text-sm" style={{ borderCollapse: 'collapse', minWidth: 800 }}>
              <thead><TH headers={['#', 'Strategy', 'Symbol', 'Dir', 'Entry', 'Exit', 'P&L', 'Status']} /></thead>
              <tbody>
                {tradeHistory.slice(0, 50).map((t: any, i: number) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                    <td className="py-2 px-2 text-xs">{t.strategy_name ?? t.strategy ?? '--'}</td>
                    <td className="py-2 px-2 text-xs">{t.symbol ?? '--'}</td>
                    <td className="py-2 px-2 text-xs">{t.direction?.[0] ?? '--'}</td>
                    <td className="py-2 px-2 text-xs">{fmtMoney(t.entry_price)}</td>
                    <td className="py-2 px-2 text-xs">{t.exit_price ? fmtMoney(t.exit_price) : '--'}</td>
                    <td className="py-2 px-2 text-xs font-medium" style={{ color: t.pnl == null ? 'var(--text-muted)' : t.pnl >= 0 ? 'var(--green)' : 'var(--red)' }}>{t.pnl != null ? fmtMoney(t.pnl) : '--'}</td>
                    <td className="py-2 px-2"><span className="text-xs px-1.5 py-0.5 rounded" style={{ background: t.status === 'Open' ? 'var(--accent-muted)' : 'var(--green-muted)', color: t.status === 'Open' ? 'var(--accent)' : 'var(--green)' }}>{t.status ?? (t.exit_price ? 'Closed' : 'Open')}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-sm py-4 text-center" style={{ color: 'var(--text-muted)' }}>No trades yet</p>}
      </Card>
    </div>
  );
}

/* ======================================================================== */
/* Tab: Performance (V5: added risk analytics modules)                       */
/* ======================================================================== */
function PerformanceTab({ compute }: TabDataProps) {
  const kpis = compute?.performance_kpis;
  return (
    <div>
      <div className="grid grid-cols-6 gap-3 mb-6">
        <MetricCard label="Trades" value={kpis?.trades != null ? String(kpis.trades) : '--'} />
        <MetricCard label="Win Rate" value={kpis?.win_rate != null ? fmtPct(kpis.win_rate) : '--'} />
        <MetricCard label="PF" value={kpis?.profit_factor != null ? kpis.profit_factor.toFixed(2) : '--'} />
        <MetricCard label="Total P&L" value={kpis?.total_pnl != null ? fmtMoney(kpis.total_pnl) : '--'} positive={kpis?.total_pnl >= 0} delta={kpis?.total_pnl != null ? (kpis.total_pnl >= 0 ? '+' : '') + fmtPct(kpis.total_pnl_pct) : undefined} />
        <MetricCard label="Balance" value={kpis?.balance != null ? fmtMoney(kpis.balance) : '--'} />
        <MetricCard label="Max DD" value={kpis?.max_dd != null ? fmtPct(kpis.max_dd) : '--'} />
      </div>
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Combined Equity Curve</h3>
        <ChartPlaceholder label="Combined portfolio equity curve with per-strategy dashed lines" height={320} />
      </Card>
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Drawdown Analysis</h3>
        <ChartPlaceholder label="Drawdown chart with requirement-set threshold line" height={220} />
      </Card>
      <div className="grid grid-cols-2 gap-6 mb-6">
        <Card><h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Daily P&L Distribution</h3><ChartPlaceholder label="Histogram of daily P&L" height={300} /></Card>
        <Card><h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Strategy Correlation</h3><ChartPlaceholder label="Correlation heatmap" height={300} /></Card>
      </div>

      {/* Risk Analytics (V5 missing modules) */}
      <h3 className="text-sm font-semibold mt-6 mb-4" style={{ color: 'var(--text-primary)' }}>Risk Analytics</h3>

      {/* Worst-Case Analysis */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Worst-Case Analysis</h4>
        <div className="grid grid-cols-3 sm:grid-cols-5 gap-3 mb-4">
          {[
            { label: 'Worst Single Day', key: 'worst_day' },
            { label: 'Worst Losing Streak', key: 'worst_streak' },
            { label: 'Worst 5-Day Rolling DD', key: 'worst_5d_dd' },
            { label: 'Days Breaching Pause', key: 'days_breach_pause' },
            { label: 'Days Breaching Max Loss', key: 'days_breach_max' },
          ].map((m) => {
            const val = compute?.worst_case?.[m.key];
            return (
              <div key={m.label}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                <p className="text-sm font-bold">{val ?? '--'}</p>
              </div>
            );
          })}
        </div>
        <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Top 5 Worst Days</h5>
        <p className="text-sm py-4 text-center" style={{ color: 'var(--text-muted)' }}>
          Worst-case analysis will populate once sufficient trade data is available.
        </p>
      </Card>

      {/* Daily P&L vs Limits */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Daily P&L vs Limits</h4>
        <ChartPlaceholder label="Bar chart: daily P&L colored by compliance (blue normal, orange pause breach, red max loss breach) with reference lines for Max Daily Loss and Daily Pause" height={220} />
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          Daily P&L compliance chart will populate once a requirement set is linked and trade data is available.
        </p>
      </Card>

      {/* Daily Peak Capital Deployed */}
      <Card className="mb-4">
        <h4 className="text-sm font-medium mb-3">Daily Peak Capital Deployed</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Shows the maximum buying power used on each day across the date range.
        </p>
        <ChartPlaceholder label="Bar chart by day: peak capital deployed per day. Red dashed line at account balance. X-axis: dates, Y-axis: peak capital ($)" height={220} />
        <div className="grid grid-cols-4 gap-4 mt-3">
          {[
            { label: 'Highest Peak Day', key: 'peak_day' },
            { label: 'Avg Peak / Day', key: 'avg_peak' },
            { label: 'Days Near Limit', key: 'days_near_limit' },
            { label: 'Max Concurrent Positions', key: 'max_concurrent' },
          ].map((m) => {
            const val = compute?.capital_deployed?.[m.key];
            return (
              <div key={m.label}>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                <p className="text-sm font-bold">{val ?? '--'}</p>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

/* ======================================================================== */
/* Tab: Strategies                                                           */
/* ======================================================================== */
function StrategiesTab({ portfolio }: TabDataProps) {
  const strategies: any[] = portfolio?.strategies ?? [];
  return (
    <div>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>{strategies.length} strategies in this portfolio</p>
      {strategies.length > 0 ? (
        <div className="space-y-4">
          {strategies.map((s: any, i: number) => (
            <Card key={s.id ?? i}>
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <p className="font-medium">{s.name ?? `Strategy #${s.id}`}</p>
                    {s.health_status && <StatusBadge status={s.health_status} />}
                    {s.active === false && <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)20', color: 'var(--orange)' }}>Paused</span>}
                  </div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{s.symbol ?? '--'} | {s.direction ?? '--'} | ${s.risk_per_trade ?? '--'}/trade</p>
                </div>
              </div>
              <div className="grid grid-cols-5 gap-4 mb-3">
                {[{ l: 'Win Rate', v: fmtPct(s.win_rate) }, { l: 'Profit Factor', v: s.profit_factor != null ? s.profit_factor.toFixed(2) : '--' }, { l: 'Trades', v: s.trades ?? '--' }, { l: 'P&L Contribution', v: fmtMoney(s.pnl_contribution), c: (s.pnl_contribution ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }, { l: 'Health', v: s.health_score != null ? `${s.health_score}%` : '--' }].map((m) => (
                  <div key={m.l}><p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.l}</p><p className="text-sm font-semibold" style={m.c ? { color: m.c } : undefined}>{m.v}</p></div>
                ))}
              </div>
              <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link href={`/strategies/${s.id}`} className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', textDecoration: 'none' }}>View Strategy</Link>
              </div>
            </Card>
          ))}
        </div>
      ) : <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>No strategies in this portfolio yet.</p></Card>}
    </div>
  );
}

/* ======================================================================== */
/* Tab: Prop Firm Check                                                      */
/* ======================================================================== */
function PropFirmCheckTab({ compute }: TabDataProps) {
  const rs = compute?.requirement_set;
  const rules: any[] = compute?.compliance_rules ?? [];
  const passing = rules.filter((r: any) => r.passing).length;
  return (
    <div>
      <Card className="mb-6">
        {rs ? (
          <div className="flex items-center justify-between">
            <div><h3 className="text-lg font-semibold">{rs.name}</h3><p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>{passing}/{rules.length} rules passing</p></div>
            <span className="text-sm px-3 py-1 rounded-full font-medium" style={{ background: passing === rules.length ? 'var(--green-muted)' : 'var(--red-muted)', color: passing === rules.length ? 'var(--green)' : 'var(--red)' }}>{passing === rules.length ? 'All Passing' : 'Has Violations'}</span>
          </div>
        ) : <p className="text-sm py-4 text-center" style={{ color: 'var(--text-muted)' }}>No requirement set linked.</p>}
      </Card>
      {rules.length > 0 && <div className="space-y-4">
        {rules.map((r: any, i: number) => (
          <Card key={i}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <p className="font-medium">{r.name}</p>
                  <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background: r.passing ? 'var(--green-muted)' : 'var(--red-muted)', color: r.passing ? 'var(--green)' : 'var(--red)' }}>{r.passing ? 'PASS' : 'FAIL'}</span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Type: {r.type} | Threshold: {r.threshold}</p>
              </div>
              <div className="text-right"><p className="text-sm font-semibold">{r.current_value ?? '--'}</p><p className="text-xs" style={{ color: 'var(--text-muted)' }}>current</p></div>
            </div>
            {r.current_pct != null && <div className="mb-2"><p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{fmtPct(r.current_pct)} of limit</p><ProgressBar pct={r.current_pct} color={r.passing ? (r.current_pct > 75 ? 'var(--orange)' : 'var(--green)') : 'var(--red)'} /></div>}
            {r.violations?.length > 0 && <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>{r.violations.map((v: string, vi: number) => <div key={vi} className="flex items-center gap-2 py-1"><span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'var(--red)' }} /><p className="text-xs" style={{ color: 'var(--red)' }}>{v}</p></div>)}</div>}
          </Card>
        ))}
      </div>}
    </div>
  );
}

/* ======================================================================== */
/* Tab: Account                                                              */
/* ======================================================================== */
function AccountTab({ account }: TabDataProps) {
  const metrics = account?.metrics;
  const ledger: any[] = account?.ledger ?? [];
  return (
    <div>
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Current Balance" value={fmtMoney(metrics?.current_balance)} />
        <MetricCard label="Starting Balance" value={fmtMoney(metrics?.starting_balance)} />
        <MetricCard label="Net Deposits" value={fmtMoney(metrics?.net_deposits)} />
        <MetricCard label="Trading P&L" value={fmtMoney(metrics?.trading_pnl)} positive={metrics?.trading_pnl >= 0} delta={metrics?.trading_pnl != null ? (metrics.trading_pnl >= 0 ? '+' : '') + fmtMoney(metrics.trading_pnl) : undefined} />
      </div>
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Balance History</h3>
        <ChartPlaceholder label="Balance history line chart with deposit markers" height={200} />
      </Card>
      <Card className="mb-6">
        <h4 className="text-sm font-medium mb-3">Record Transaction</h4>
        <div className="flex flex-col gap-3">
          <div><label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Amount ($)</label><input type="number" placeholder="0.00" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} /></div>
          <div><label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Date</label><input type="date" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 12px', borderRadius: '8px', fontSize: '0.875rem', width: '100%' }} /></div>
          <div className="flex gap-2">
            <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--green)', color: 'white', border: 'none', cursor: 'pointer' }}>Add Deposit</button>
            <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--red)', color: 'white', border: 'none', cursor: 'pointer' }}>Add Withdrawal</button>
          </div>
        </div>
      </Card>
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Ledger</h3>
        {ledger.length > 0 ? (
          <>
            <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
              {[{ l: 'Date', s: 2 }, { l: 'Type', s: 3 }, { l: 'Amount', s: 2 }, { l: 'Summary', s: 5 }].map((h) => <p key={h.l} className={`col-span-${h.s} text-xs font-medium`} style={{ color: 'var(--text-muted)' }}>{h.l}</p>)}
            </div>
            {ledger.map((e: any, i: number) => (
              <div key={i} className="grid grid-cols-12 gap-2 py-3 border-b items-center" style={{ borderColor: 'var(--border)' }}>
                <p className="col-span-2 text-sm">{e.date}</p>
                <p className="col-span-3 text-sm" style={{ color: 'var(--text-secondary)' }}>{e.type}</p>
                <p className="col-span-2 text-sm font-medium" style={{ color: (e.amount ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>{e.amount != null ? `${e.amount >= 0 ? '+' : ''}${fmtMoney(e.amount)}` : '--'}</p>
                <p className="col-span-5 text-xs" style={{ color: 'var(--text-muted)' }}>{e.summary ?? '--'}</p>
              </div>
            ))}
          </>
        ) : <p className="text-sm py-4 text-center" style={{ color: 'var(--text-muted)' }}>No ledger entries yet</p>}
      </Card>
    </div>
  );
}

/* ======================================================================== */
/* Tab: Webhooks                                                             */
/* ======================================================================== */
function WebhooksTab() {
  return (
    <div>
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3">Webhook Template</h3>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Select an account-based template that defines how this portfolio communicates with your exchange.</p>
        <div className="flex items-center gap-3">
          <select className="flex-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px 14px', borderRadius: '8px', fontSize: '0.875rem' }} defaultValue=""><option value="">No template (webhooks disabled)</option></select>
          <Link href="/alerts/webhook-templates" className="px-3 py-2 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--accent)', textDecoration: 'none', whiteSpace: 'nowrap' }}>Manage Templates</Link>
        </div>
      </Card>
      <Card>
        <h3 className="text-sm font-medium mb-3">Delivery History</h3>
        <p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>Webhook delivery history will appear here once a template is configured.</p>
      </Card>
    </div>
  );
}

/* ======================================================================== */
/* Strategy Variable Summary (V5 — above tabs)                               */
/* ======================================================================== */
function StrategyVariableSummary({ portfolio }: { portfolio?: any }) {
  const strategies: any[] = portfolio?.strategies ?? [];
  if (strategies.length === 0) return null;

  return (
    <Card className="mb-4">
      <h3 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>Strategy Variables</h3>
      <div className="space-y-2">
        {strategies.slice(0, 10).map((s: any, i: number) => (
          <div key={s.id ?? i} className="flex items-center gap-3 flex-wrap">
            <span className="text-xs font-medium min-w-[140px]">{s.name ?? `Strategy #${s.id}`}</span>
            {s.fwd_sigma != null && (
              <span className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{
                background: Math.abs(s.fwd_sigma) <= 1 ? 'var(--green-muted)' : Math.abs(s.fwd_sigma) <= 2 ? 'var(--orange-muted)' : 'var(--red-muted)',
                color: Math.abs(s.fwd_sigma) <= 1 ? 'var(--green)' : Math.abs(s.fwd_sigma) <= 2 ? 'var(--orange)' : 'var(--red)',
              }}>
                {s.fwd_sigma >= 0 ? '+' : ''}{s.fwd_sigma.toFixed(1)} SD
              </span>
            )}
            {s.entry_trigger && (
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                entry: {s.entry_trigger}
              </span>
            )}
            {s.exit_trigger && (
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                exit: {s.exit_trigger}
              </span>
            )}
            {s.stop_desc && (
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ color: 'var(--red)', background: 'rgba(244,67,54,0.1)' }}>
                {s.stop_desc}
              </span>
            )}
            {s.target_desc && (
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ color: 'var(--green)', background: 'rgba(76,175,80,0.1)' }}>
                {s.target_desc}
              </span>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ======================================================================== */
/* Main Component                                                            */
/* ======================================================================== */
interface PortfolioDetailPageProps { portfolioId: number }

export default function PortfolioDetailPage({ portfolioId }: PortfolioDetailPageProps) {
  const { data: portfolio, isLoading: portfolioLoading, error: portfolioError } = usePortfolio(portfolioId);
  const { data: compute } = usePortfolioCompute(portfolioId, ['kpis', 'compliance', 'buying_power', 'open_positions']);
  const { data: trades } = usePortfolioTrades(portfolioId);
  const { data: anomalies } = usePortfolioAnomalies(portfolioId);
  const { data: account } = usePortfolioAccount(portfolioId);

  // Inject pulse animation
  useState(() => {
    if (typeof document !== 'undefined') {
      const id = 'portfolio-detail-pulse-css';
      if (!document.getElementById(id)) { const s = document.createElement('style'); s.id = id; s.textContent = PULSE_CSS; document.head.appendChild(s); }
    }
  });

  if (portfolioLoading) {
    return (
      <div>
        <PageHeader title="Portfolio Detail" backHref="/portfolios" />
        <div className="space-y-4"><Skeleton h={40} /><div className="grid grid-cols-5 gap-3">{[1, 2, 3, 4, 5].map((i) => <Skeleton key={i} h={90} />)}</div><Skeleton h={400} /></div>
      </div>
    );
  }

  if (portfolioError) {
    return (
      <div>
        <PageHeader title="Portfolio Detail" backHref="/portfolios" />
        <Card><div className="text-center py-8" style={{ color: 'var(--red)' }}>Failed to load portfolio. Check your connection and try again.</div></Card>
      </div>
    );
  }

  const name = portfolio?.name ?? 'Portfolio';
  const stratCount = portfolio?.strategies?.length ?? 0;
  const enabled = portfolio?.enabled !== false;
  const tags: string[] = portfolio?.tags ?? [];
  const tabData: TabDataProps = { portfolio, compute, trades: trades ?? [], anomalies, account };

  return (
    <div>
      <PageHeader title={name} backHref="/portfolios" actions={<>
        <button style={btnSec}>Refresh</button><button style={btnSec}>Edit</button><button style={btnSec}>Clone</button>
        <button style={{ ...btnSec, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }}>Delete</button>
      </>} />
      {/* Status badges */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span className="flex items-center gap-1.5 text-xs" style={{ color: enabled ? 'var(--green)' : 'var(--text-muted)' }}>
          <span style={{ position: 'relative', display: 'inline-block', width: 8, height: 8 }}>
            {enabled && <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: 'var(--green)', opacity: 0.5, animation: 'pulse 2s ease-in-out infinite' }} />}
            <span style={{ position: 'absolute', inset: '25%', borderRadius: '50%', background: enabled ? 'var(--green)' : 'var(--text-muted)' }} />
          </span>
          {enabled ? 'Enabled' : 'Disabled'}
        </span>
        <span className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5" style={{ background: 'var(--green)15', color: 'var(--green)' }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />Live
        </span>
        {tags.map((tag) => <span key={tag} className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>{tag}</span>)}
      </div>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        {stratCount} strategies{portfolio?.kpis?.balance != null && <> &middot; {fmtMoney(portfolio.kpis.balance)} balance</>}
      </p>

      {/* Strategy variable summary above tabs (V5) */}
      <StrategyVariableSummary portfolio={portfolio} />

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Live Dashboard' && <LiveDashboardTab {...tabData} />}
            {tab === 'Performance' && <PerformanceTab {...tabData} />}
            {tab === 'Strategies' && <StrategiesTab {...tabData} />}
            {tab === 'Prop Firm Check' && <PropFirmCheckTab {...tabData} />}
            {tab === 'Account' && <AccountTab {...tabData} />}
            {tab === 'Webhooks' && <WebhooksTab />}
          </div>
        )}
      </TabBar>
    </div>
  );
}
