'use client';

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import { useStrategy, useStrategyTrades, useStrategyForwardTest, useStrategyKPIs, useTriggerAnalysis } from '@/hooks/queries/useStrategies';
import { useStrategyAlerts } from '@/hooks/queries/useAlerts';
import { useDeleteStrategy, useDuplicateStrategy, useRefreshStrategy } from '@/hooks/mutations/useStrategyMutations';

/* ========================================================================= */
/* COLOR CONSTANTS                                                            */
/* ========================================================================= */

const EXEC_BADGE_COLOR = '#2196F3';
const FIDELITY_BADGE_COLOR = '#26C6DA';
const EQ_BT_COLOR = '#2196F3';
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

/* ========================================================================= */
/* API DATA MAPPING                                                           */
/* ========================================================================= */

function apiToDetailStrategy(s: any) {
  const k = s.kpis || {};
  return {
    id: String(s.id),
    name: s.name || '--',
    symbol: s.symbol || '--',
    direction: s.direction || 'LONG',
    timeframe: s.timeframe || '1Min',
    session: s.trading_session || 'RTH',
    method: 'Ticker-Based',
    status: s.status || (s.forward_testing ? 'On Track' : 'Insufficient Data'),
    tags: s.tags || [],
    monitored: s.alert_tracking_enabled || false,
    entry: s.entry_trigger_confluence_id || '--',
    exit: s.exit_trigger_confluence_ids || [],
    stop: s.stop_config?.method || '--',
    target: s.target_config?.method || 'Signal exit only',
    confluence: s.confluence || [],
    winRate: k.win_rate ?? 0,
    pf: k.profit_factor ?? 0,
    dailyR: k.daily_r ?? 0,
    dailyROI: 0,  // {{daily_roi}}
    trades: k.total_trades ?? 0,
    maxDD: k.max_r_drawdown ?? 0,
    btDays: s.data_days || 30,
    btStart: '--',  // {{bt_start}}
    btEnd: '--',    // {{bt_end}}
    fwdWinRate: (s.forward_kpis?.win_rate ?? null) as number | null,
    fwdPF: (s.forward_kpis?.profit_factor ?? null) as number | null,
    fwdDailyR: (s.forward_kpis?.daily_r ?? null) as number | null,
    fwdDailyROI: null as number | null, // {{fwd_daily_roi}}
    fwdTrades: s.forward_kpis?.trades ?? 0,
    fwdSince: s.forward_test_start || '',
    fwdMaxDD: (s.forward_kpis?.max_r_drawdown ?? null) as number | null,
    alertWinRate: (s.alert_kpis?.win_rate ?? null) as number | null,
    alertPF: (s.alert_kpis?.profit_factor ?? null) as number | null,
    alertDailyR: (s.alert_kpis?.daily_r ?? null) as number | null,
    alertDailyROI: null as number | null, // {{alert_daily_roi}}
    alertTrades: s.alert_kpis?.trades ?? 0,
    alertMaxDD: (s.alert_kpis?.max_r_drawdown ?? null) as number | null,
    fwdSD: (s.sigma_fwd ?? null) as number | null,
    alertSD: (s.sigma_alert ?? null) as number | null,
    createdAt: s.created_at || '--',
    updatedAt: s.updated_at || '--',
  };
}

// Trades populated from API hooks (useStrategyTrades, useStrategyForwardTest)
const EMPTY_TRADES: any[] = [];

// {{extended_kpis}} — populated from useStrategyKPIs secondary_kpis
const EMPTY_EXTENDED_KPIS = {
  wins: 0, losses: 0, bestTrade: 0, worstTrade: 0,
  avgWin: 0, avgLoss: 0, payoffRatio: 0, expectedDailyR: 0,
  sharpe: 0, sortino: 0, calmar: 0, kelly: 0,
  dailyVaR: 0, cvar: 0, volatility: 0, rSquared: 0,
  skewness: 0, kurtosis: 0, tailRatio: 0, outlierWinPct: 0, outlierLossPct: 0,
  maxRDD: 0, recoveryFactor: 0, ulcerIndex: 0, serenityIndex: 0,
  longestDDTrades: 0, longestDDDays: 0,
  maxConsecWins: 0, maxConsecLosses: 0,
};

const ROLLING_METRIC_OPTIONS = ['Win Rate', 'PF', 'Sharpe'];

// {{alert_analysis}} — populated from API when alert analysis endpoint is available
const EMPTY_ALERT_ANALYSIS = {
  missed: [] as any[],
  phantom: [] as any[],
  summaryMetrics: [] as any[],
  positionHealth: {
    status: '--', entries: 0, exits: 0, avgHoldTime: '--',
    anomalies: [] as any[],
  },
  triggerTiming: [] as any[],
  tradeByTrade: [] as any[],
};

// {{recent_alerts}} — populated from useStrategyAlerts hook
const EMPTY_ALERTS: any[] = [];

// {{trade_alert_mapping}} — populated from API when alert analysis endpoint is available
const EMPTY_TRADE_ALERT_MAPPING: any[] = [];

// {{confluence_groups}} — derived from strategy.confluence when API data is available
const EMPTY_CONFLUENCE_GROUPS: { name: string; pack: string; id: string }[] = [];

// {{confluence_timeline}} — populated from API when confluence analysis endpoint is available
const EMPTY_CONFLUENCE_TIMELINE: any[] = [];

// {{confluence_trigger_events}} — populated from API when confluence analysis endpoint is available
const EMPTY_CONFLUENCE_TRIGGER_EVENTS: any[] = [];

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

const statusColors: Record<string, string> = {
  'On Track': 'var(--green)',
  'Outperforming': 'var(--blue)',
  'Underperforming': 'var(--red)',
  'Insufficient Data': 'var(--text-muted)',
};

const execTypeColors: Record<string, string> = {
  '[C]': EXEC_BADGE_COLOR,
  '[L]': 'var(--green)',
  '[LC]': '#AB47BC',
  '[CC]': '#FF9800',
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

function parseExecTag(entry: string): { exec: string; rest: string } {
  const match = entry.match(/^\[([A-Z]+)\]\s*(.*)/);
  if (match) return { exec: `[${match[1]}]`, rest: match[2] };
  return { exec: '', rest: entry };
}

function parsePack(text: string): { pack: string; trigger: string } {
  const match = text.match(/^(.+?)\s*>\s*(.+)$/);
  if (match) return { pack: match[1].trim(), trigger: match[2].trim() };
  return { pack: '', trigger: text };
}

/* ========================================================================= */
/* PULSE CSS                                                                   */
/* ========================================================================= */

const PULSE_CSS = `@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(2.2); opacity: 0; } }`;

/* ========================================================================= */
/* TABS                                                                        */
/* ========================================================================= */

const TABS = [
  'Equity & KPIs',
  'Chart & Trades',
  'Confluence Analysis',
  'Configuration',
  'Alerts',
  'Alert Analysis',
];

/* ========================================================================= */
/* SUB-COMPONENTS                                                              */
/* ========================================================================= */

function ExecBadge({ exec }: { exec: string }) {
  return (
    <span
      className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full"
      style={{
        color: EXEC_BADGE_COLOR,
        background: EXEC_BADGE_COLOR + '20',
      }}
    >
      {exec}
    </span>
  );
}

function FidelityBadge({ label }: { label: string }) {
  return (
    <span
      className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full"
      style={{
        color: FIDELITY_BADGE_COLOR,
        background: FIDELITY_BADGE_COLOR + '20',
      }}
    >
      {label}
    </span>
  );
}

function ConditionBadge({ text }: { text: string }) {
  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full"
      style={{
        color: 'var(--accent)',
        background: 'var(--accent-muted)',
      }}
    >
      {text}
    </span>
  );
}

function StopBadge({ text }: { text: string }) {
  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full"
      style={{
        color: 'var(--red)',
        background: 'var(--red-muted)',
      }}
    >
      {text}
    </span>
  );
}

function TargetBadge({ text }: { text: string }) {
  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full"
      style={{
        color: 'var(--green)',
        background: 'var(--green-muted)',
      }}
    >
      {text}
    </span>
  );
}

function SigmaBadge({ label, value, color }: { label: string; value: number | null; color: string }) {
  const bgColor = color + '20';
  return (
    <span
      className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full"
      style={{ color, background: bgColor }}
    >
      {label} {value != null ? `${value >= 0 ? '+' : ''}${value.toFixed(1)}` : '--'}&sigma;
    </span>
  );
}

function CollapsibleSection({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-4">
      <button
        className="flex items-center gap-2 text-sm font-medium mb-2 w-full text-left"
        style={{ color: 'var(--text-primary)', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}
        onClick={() => setOpen(!open)}
      >
        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{open ? '\u25BC' : '\u25B6'}</span>
        {title}
      </button>
      {open && children}
    </div>
  );
}

function PillTabs({ tabs, active, onChange }: { tabs: string[]; active: string; onChange: (t: string) => void }) {
  return (
    <div className="flex gap-1 mb-4 flex-wrap">
      {tabs.map((t) => (
        <button
          key={t}
          className="text-xs px-3 py-1.5 rounded-full transition-colors"
          style={{
            background: active === t ? 'var(--accent)' : 'var(--bg-input)',
            color: active === t ? 'white' : 'var(--text-muted)',
            border: active === t ? 'none' : '1px solid var(--border)',
            cursor: 'pointer',
          }}
          onClick={() => onChange(t)}
        >
          {t}
        </button>
      ))}
    </div>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

interface Props {
  strategyId: number;
}

export default function StrategyDetailPage({ strategyId }: Props) {
  const { data: apiStrategy, isLoading, error } = useStrategy(strategyId);
  const { data: trades, isLoading: tradesLoading } = useStrategyTrades(strategyId);
  const { data: fwdData, isLoading: fwdLoading } = useStrategyForwardTest(strategyId);
  const { data: kpiData, isLoading: kpisLoading } = useStrategyKPIs(strategyId);
  const { data: alerts } = useStrategyAlerts(strategyId);
  const { data: triggerAnalysis } = useTriggerAnalysis(strategyId);
  const deleteMut = useDeleteStrategy();
  const dupMut = useDuplicateStrategy();
  const refreshMut = useRefreshStrategy();

  // Map API data to V5 shape
  const strategy = apiStrategy ? apiToDetailStrategy(apiStrategy) : null;
  const extendedKPIs = kpiData?.secondary_kpis || EMPTY_EXTENDED_KPIS;
  // Map API trades (snake_case) to V5 format (camelCase)
  const allTrades = (trades || EMPTY_TRADES).map((t: any, i: number) => ({
    id: t.id ?? i + 1,
    entryTime: t.entry_time || t.entryTime || '--',
    exitTime: t.exit_time || t.exitTime || '--',
    entryPrice: t.entry_price ?? t.entryPrice ?? 0,
    exitPrice: t.exit_price ?? t.exitPrice ?? 0,
    pnlR: t.r_multiple ?? t.pnlR ?? 0,
    execType: t.exec_type ? `[${t.exec_type}]` : (t.execType || '[C]'),
    exitReason: t.exit_reason || t.exitReason || '--',
    isFwd: t.isFwd ?? false,
  }));

  // Inject pulse CSS
  useEffect(() => {
    const id = 'strategy-detail-pulse-css';
    if (!document.getElementById(id)) {
      const s = document.createElement('style');
      s.id = id;
      s.textContent = PULSE_CSS;
      document.head.appendChild(s);
    }
  }, []);

  // State
  const [dateRange, setDateRange] = useState('Strategy Default');
  const [kpiMode, setKpiMode] = useState('Overall');
  const [extKpiTab, setExtKpiTab] = useState('Performance');
  const [showExtKpis, setShowExtKpis] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [eqShowHWM, setEqShowHWM] = useState(false);
  const [eqShowEdge, setEqShowEdge] = useState(false);
  const [eqShowConf, setEqShowConf] = useState(false);
  const [advancedTab, setAdvancedTab] = useState('Rolling Metrics');
  const [rollingWindow, setRollingWindow] = useState(20);
  const [rollingMetric, setRollingMetric] = useState('Win Rate');
  const [returnView, setReturnView] = useState('Histogram');
  const [markovWindow, setMarkovWindow] = useState(20);
  const [edgeDecay, setEdgeDecay] = useState(0.5);
  const [candleCount, setCandleCount] = useState(200);
  const [showConditions, setShowConditions] = useState(true);
  const [showTriggers, setShowTriggers] = useState(true);
  const [btTradesOpen, setBtTradesOpen] = useState(false);
  const [fwdTradesOpen, setFwdTradesOpen] = useState(true);
  const confluenceGroups = triggerAnalysis?.confluence_groups ?? EMPTY_CONFLUENCE_GROUPS;
  const confluenceTimeline = EMPTY_CONFLUENCE_TIMELINE; // State timeline requires backtest instrumentation
  const confluenceTriggerEvents = EMPTY_CONFLUENCE_TRIGGER_EVENTS; // Trigger events require backtest instrumentation
  const recentAlerts = alerts || EMPTY_ALERTS; // from useStrategyAlerts hook
  const tradeAlertMapping = EMPTY_TRADE_ALERT_MAPPING; // {{trade_alert_mapping}} — wire to API
  const alertAnalysis = EMPTY_ALERT_ANALYSIS; // {{alert_analysis}} — wire to API
  const [selectedConfGroup, setSelectedConfGroup] = useState('');
  useEffect(() => {
    if (confluenceGroups.length > 0 && !selectedConfGroup) {
      setSelectedConfGroup(confluenceGroups[0].name);
    }
  }, [confluenceGroups, selectedConfGroup]);
  const [showPosHealth, setShowPosHealth] = useState(false);
  const [showTriggerTiming, setShowTriggerTiming] = useState(false);
  const [showTradeByTrade, setShowTradeByTrade] = useState(false);

  // Early returns after all hooks
  if (isLoading || !strategy) {
    return (
      <div style={{ padding: '32px' }}>
        <div className="animate-pulse space-y-4">
          <div className="h-6 rounded w-1/3" style={{ background: 'var(--border)' }} />
          <div className="h-4 rounded w-2/3" style={{ background: 'var(--border)' }} />
          <div className="h-64 rounded" style={{ background: 'var(--bg-input)' }} />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '32px' }}>
        <div className="text-center py-8" style={{ color: 'var(--red)' }}>Failed to load strategy.</div>
      </div>
    );
  }

  const fwdDays = daysSince(strategy.fwdSince);
  const alertAccuracy = strategy.alertTrades > 0 && strategy.alertWinRate != null && strategy.fwdWinRate
    ? ((strategy.alertWinRate / strategy.fwdWinRate) * 100).toFixed(1)
    : '--';

  // Forward test trades come from the dedicated forward-test endpoint
  // which loads fresh Polygon data and splits at forward_test_start
  const fwdTrades = (fwdData?.forward_trades || []).map((t: any, i: number) => ({
    id: i + 1,
    entryTime: t.entry_time || '--',
    exitTime: t.exit_time || '--',
    entryPrice: t.entry_price ?? 0,
    exitPrice: t.exit_price ?? 0,
    pnlR: t.r_multiple ?? 0,
    execType: t.exec_type ? `[${t.exec_type}]` : '[C]',
    exitReason: t.exit_reason || '--',
    isFwd: true,
  }));
  const btTrades = (fwdData?.backtest_trades || allTrades).map((t: any, i: number) => ({
    id: t.id ?? i + 1,
    entryTime: t.entry_time || t.entryTime || '--',
    exitTime: t.exit_time || t.exitTime || '--',
    entryPrice: t.entry_price ?? t.entryPrice ?? 0,
    exitPrice: t.exit_price ?? t.exitPrice ?? 0,
    pnlR: t.r_multiple ?? t.pnlR ?? 0,
    execType: t.exec_type ? `[${t.exec_type}]` : (t.execType || '[C]'),
    exitReason: t.exit_reason || t.exitReason || '--',
    isFwd: false,
  }));

  // Parse entry for badges
  const entryParsed = parseExecTag(strategy.entry);
  const entryPack = parsePack(entryParsed.rest);
  const exitsParsed = strategy.exit.map((e) => {
    const p = parseExecTag(e);
    return { exec: p.exec, ...parsePack(p.rest) };
  });

  // Styles
  const selectStyle: React.CSSProperties = {
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
  };

  const btnSecondary: React.CSSProperties = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer',
  };

  const thStyle: React.CSSProperties = {
    color: 'var(--text-muted)',
    background: 'var(--bg-secondary)',
    textAlign: 'left' as const,
    padding: '8px 12px',
    fontSize: '0.75rem',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
  };

  const tdStyle: React.CSSProperties = {
    padding: '8px 12px',
    fontSize: '0.8125rem',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text-secondary)',
  };

  /* ======================================================================= */
  /* KPI COMPARISON HELPER                                                     */
  /* ======================================================================= */

  function renderKpiComparison(
    labelA: string, labelB: string, colorA: string, colorB: string,
    dataA: { wr: number | null; pf: number | null; dr: number | null; droi: number | null; tpd: number | null; mdd: number | null },
    dataB: { wr: number | null; pf: number | null; dr: number | null; droi: number | null; tpd: number | null; mdd: number | null },
  ) {
    // Coerce nulls to 0 for display
    const a = { wr: dataA.wr ?? 0, pf: dataA.pf ?? 0, dr: dataA.dr ?? 0, droi: dataA.droi ?? 0, tpd: dataA.tpd ?? 0, mdd: dataA.mdd ?? 0 };
    const b = { wr: dataB.wr ?? 0, pf: dataB.pf ?? 0, dr: dataB.dr ?? 0, droi: dataB.droi ?? 0, tpd: dataB.tpd ?? 0, mdd: dataB.mdd ?? 0 };
    const rows = [
      { label: 'Win Rate', a: `${a.wr.toFixed(1)}%`, b: `${b.wr.toFixed(1)}%`, d: b.wr - a.wr, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` },
      { label: 'PF', a: a.pf.toFixed(2), b: b.pf.toFixed(2), d: b.pf - a.pf, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}` },
      { label: 'Daily R', a: `${a.dr >= 0 ? '+' : ''}${a.dr.toFixed(2)}`, b: `${b.dr >= 0 ? '+' : ''}${b.dr.toFixed(2)}`, d: b.dr - a.dr, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}` },
      { label: 'Daily ROI', a: `${a.droi.toFixed(2)}%`, b: `${b.droi.toFixed(2)}%`, d: b.droi - a.droi, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` },
      { label: 'TPD', a: a.tpd.toFixed(1), b: b.tpd.toFixed(1), d: b.tpd - a.tpd, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}` },
      { label: 'Max DD', a: `${a.mdd.toFixed(1)}R`, b: `${b.mdd.toFixed(1)}R`, d: b.mdd - a.mdd, fmt: (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}R`, invert: true },
    ];

    return (
      <div style={{ overflowX: 'auto' }}>
        <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Metric</th>
              <th style={{ ...thStyle, textAlign: 'right', color: colorA }}>{labelA}</th>
              <th style={{ ...thStyle, textAlign: 'right', color: colorB }}>{labelB}</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>Delta</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const deltaVal = row.d;
              const isGood = row.invert ? deltaVal <= 0 : deltaVal >= 0;
              return (
                <tr key={i}>
                  <td style={tdStyle}>{row.label}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{row.a}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>{row.b}</td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: isGood ? 'var(--green)' : 'var(--red)' }}>{row.fmt(deltaVal)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  /* ======================================================================= */
  /* TRADE TABLE HELPER                                                        */
  /* ======================================================================= */

  function renderTradeTable(trades: any[]) {
    return (
      <div style={{ overflowX: 'auto' }}>
        <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['#', 'Entry Time', 'Exit Time', 'Entry $', 'Exit $', 'R', 'Exec', 'Exit Reason'].map((h) => (
                <th key={h} style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id}>
                <td style={tdStyle}>{t.id}</td>
                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{t.entryTime}</td>
                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{t.exitTime}</td>
                <td style={tdStyle}>${t.entryPrice.toFixed(2)}</td>
                <td style={tdStyle}>${t.exitPrice.toFixed(2)}</td>
                <td style={{ ...tdStyle, color: t.pnlR >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                  {t.pnlR >= 0 ? '+' : ''}{t.pnlR.toFixed(2)}R
                </td>
                <td style={tdStyle}>
                  <span
                    className="text-xs font-mono px-1.5 py-0.5 rounded-full"
                    style={{
                      color: execTypeColors[t.execType] || EXEC_BADGE_COLOR,
                      background: (execTypeColors[t.execType] || EXEC_BADGE_COLOR) + '20',
                    }}
                  >
                    {t.execType}
                  </span>
                </td>
                <td style={tdStyle}>
                  <span style={{ color: exitReasonColors[t.exitReason] || 'var(--text-secondary)' }}>
                    {t.exitReason}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  /* ======================================================================= */
  /* RENDER                                                                    */
  /* ======================================================================= */

  return (
    <div>
      {/* ================================================================= */}
      {/* HEADER SECTION                                                     */}
      {/* ================================================================= */}

      <PageHeader
        title={strategy.name}
        backHref="/strategies"
        actions={
          <>
            <button
              style={{ ...btnSecondary, opacity: refreshMut.isPending ? 0.6 : 1 }}
              disabled={refreshMut.isPending}
              onClick={() => refreshMut.mutate(strategyId)}
            >
              {refreshMut.isPending ? 'Refreshing...' : 'Update Data'}
            </button>
            <button style={btnSecondary} onClick={() => dupMut.mutate(strategyId)}>Clone</button>
            <button
              style={{ ...btnSecondary, background: 'var(--red-muted)', color: 'var(--red)', border: 'none' }}
              onClick={() => { if (confirm(`Delete "${strategy.name}"?`)) deleteMut.mutate(strategyId); }}
            >
              Delete
            </button>
          </>
        }
      />

      {/* Loading indicators */}
      {(fwdLoading || refreshMut.isPending) && (
        <div className="mb-3 px-4 py-2.5 rounded-lg flex items-center gap-3" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)30' }}>
          <div className="w-4 h-4 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
          <span className="text-sm" style={{ color: 'var(--accent)' }}>
            {refreshMut.isPending ? 'Updating strategy data — loading bars from Polygon and running backtest...' : 'Computing forward test trades — this may take 10-30 seconds for 1-minute timeframes...'}
          </span>
        </div>
      )}
      {refreshMut.isSuccess && (
        <div className="mb-3 px-4 py-2.5 rounded-lg flex items-center gap-2" style={{ background: 'var(--green)10', border: '1px solid var(--green)30' }}>
          <span style={{ color: 'var(--green)' }}>&#10003;</span>
          <span className="text-sm" style={{ color: 'var(--green)' }}>
            Data updated — {refreshMut.data?.trades ?? 0} trades refreshed
          </span>
        </div>
      )}
      {refreshMut.isError && (
        <div className="mb-3 px-4 py-2.5 rounded-lg flex items-center gap-2" style={{ background: 'var(--red)10', border: '1px solid var(--red)30' }}>
          <span style={{ color: 'var(--red)' }}>&#10007;</span>
          <span className="text-sm" style={{ color: 'var(--red)' }}>
            Refresh failed — check Polygon API connection
          </span>
        </div>
      )}

      {/* Status badges + sigma + pulse dot */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span
          className="text-xs font-semibold px-2.5 py-1 rounded-full"
          style={{
            color: statusColors[strategy.status],
            background: statusColors[strategy.status] + '20',
          }}
        >
          {strategy.status}
        </span>
        <SigmaBadge label="FWD" value={strategy.fwdSD} color={EQ_FWD_COLOR} />
        <SigmaBadge label="Alert" value={strategy.alertSD} color={EQ_LIVE_COLOR} />
        {strategy.monitored && (
          <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--green)' }}>
            <span style={{ position: 'relative', display: 'inline-block', width: 8, height: 8 }}>
              <span
                style={{
                  position: 'absolute', inset: 0, borderRadius: '50%',
                  background: 'var(--green)', opacity: 0.5,
                  animation: 'pulse 2s ease-in-out infinite',
                }}
              />
              <span
                style={{
                  position: 'absolute', inset: '25%', borderRadius: '50%',
                  background: 'var(--green)',
                }}
              />
            </span>
            Monitored
          </span>
        )}
        {strategy.tags.map((tag) => (
          <span
            key={tag}
            className="text-xs px-2 py-0.5 rounded-full"
            style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Meta line */}
      <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
        {strategy.symbol} &middot; {strategy.direction} &middot; {strategy.timeframe} &middot; {strategy.session}
        &nbsp;&middot;&nbsp;BT {strategy.btDays}d &middot; FWD {fwdDays}d &middot; Alert Accuracy {alertAccuracy}%
      </p>

      {/* Pack-aware variable display */}
      <Card className="mb-4">
        <div className="flex flex-col gap-2">
          {/* Row 1: Entry + Exit */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs" style={{ color: 'var(--text-muted)', minWidth: 48 }}>entry:</span>
            {entryParsed.exec && <ExecBadge exec={entryParsed.exec} />}
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              {entryPack.pack && <>{entryPack.pack} &gt; </>}{entryPack.trigger}
            </span>
            <span className="text-xs" style={{ color: 'var(--border)', margin: '0 4px' }}>|</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>exit:</span>
            {exitsParsed.map((e, i) => (
              <span key={i} className="flex items-center gap-1">
                {e.exec && <ExecBadge exec={e.exec} />}
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {e.pack && <>{e.pack} &gt; </>}{e.trigger}
                </span>
                {i < exitsParsed.length - 1 && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>,</span>}
              </span>
            ))}
          </div>

          {/* Row 2: Stop + Target + Confluence */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs" style={{ color: 'var(--text-muted)', minWidth: 48 }}>stop:</span>
            <StopBadge text={strategy.stop} />
            <span className="text-xs" style={{ color: 'var(--border)', margin: '0 4px' }}>|</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>target:</span>
            <TargetBadge text={strategy.target} />
            <span className="text-xs" style={{ color: 'var(--border)', margin: '0 4px' }}>|</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>confluence:</span>
            <FidelityBadge label="[PB]" />
            {strategy.confluence.map((c) => (
              <ConditionBadge key={c} text={c} />
            ))}
          </div>
        </div>
      </Card>

      {/* ================================================================= */}
      {/* TABS                                                                */}
      {/* ================================================================= */}

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {/* =========================================================== */}
            {/* TAB 1: EQUITY & KPIs                                        */}
            {/* =========================================================== */}
            {tab === 'Equity & KPIs' && (
              <div>
                {/* Date Range + KPI Mode row */}
                <div className="flex items-center gap-6 mb-4 flex-wrap">
                  <div className="flex items-center gap-2">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Date Range:</label>
                    <select
                      style={selectStyle}
                      value={dateRange}
                      onChange={(e) => setDateRange(e.target.value)}
                    >
                      {['Strategy Default', 'All Data', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Backtest Only', 'Forward Only'].map((o) => (
                        <option key={o} value={o}>{o}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>KPI Mode:</label>
                    <select
                      style={selectStyle}
                      value={kpiMode}
                      onChange={(e) => setKpiMode(e.target.value)}
                    >
                      {['Overall', 'BT vs FWD', 'FWD vs Alerts', 'BT vs Alerts'].map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="text-xs" style={{ color: 'var(--text-muted)' }}>Trade Qualification:</label>
                    <select style={selectStyle} defaultValue="None">
                      <option value="None">None</option>
                      <option value="ttp">Trade The Pool</option>
                      <option value="ftmo">FTMO</option>
                      <option value="topstep">Topstep</option>
                      <option value="custom">My Custom Rules</option>
                    </select>
                  </div>
                </div>

                {/* Overall mode: 6-column grid */}
                {kpiMode === 'Overall' && (
                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-6">
                    {[
                      { label: 'Win Rate', value: `${strategy.winRate.toFixed(1)}%` },
                      { label: 'PF', value: strategy.pf.toFixed(2) },
                      { label: 'Daily R', value: `+${strategy.dailyR.toFixed(2)}` },
                      { label: 'Daily ROI', value: `${strategy.dailyROI.toFixed(2)}%` },
                      { label: 'TPD', value: (strategy.trades / strategy.btDays).toFixed(1) },
                      { label: 'Max DD', value: `${strategy.maxDD.toFixed(1)}R` },
                    ].map((kpi) => (
                      <Card key={kpi.label}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                        <p className="text-lg font-bold mt-1">{kpi.value}</p>
                      </Card>
                    ))}
                  </div>
                )}

                {/* BT vs FWD comparison */}
                {kpiMode === 'BT vs FWD' && (
                  <Card className="mb-6">
                    {renderKpiComparison(
                      'Backtest', 'Forward', EQ_BT_COLOR, EQ_FWD_COLOR,
                      { wr: strategy.winRate, pf: strategy.pf, dr: strategy.dailyR, droi: strategy.dailyROI, tpd: strategy.trades / strategy.btDays, mdd: strategy.maxDD },
                      { wr: strategy.fwdWinRate, pf: strategy.fwdPF, dr: strategy.fwdDailyR, droi: strategy.fwdDailyROI, tpd: strategy.fwdTrades / fwdDays, mdd: strategy.fwdMaxDD },
                    )}
                  </Card>
                )}

                {/* FWD vs Alerts comparison */}
                {kpiMode === 'FWD vs Alerts' && (
                  <Card className="mb-6">
                    {renderKpiComparison(
                      'Forward', 'Alerts', EQ_FWD_COLOR, EQ_LIVE_COLOR,
                      { wr: strategy.fwdWinRate, pf: strategy.fwdPF, dr: strategy.fwdDailyR, droi: strategy.fwdDailyROI, tpd: strategy.fwdTrades / fwdDays, mdd: strategy.fwdMaxDD },
                      { wr: strategy.alertWinRate, pf: strategy.alertPF, dr: strategy.alertDailyR, droi: strategy.alertDailyROI, tpd: strategy.alertTrades / fwdDays, mdd: strategy.alertMaxDD },
                    )}
                  </Card>
                )}

                {/* BT vs Alerts comparison */}
                {kpiMode === 'BT vs Alerts' && (
                  <Card className="mb-6">
                    {renderKpiComparison(
                      'Backtest', 'Alerts', EQ_BT_COLOR, EQ_LIVE_COLOR,
                      { wr: strategy.winRate, pf: strategy.pf, dr: strategy.dailyR, droi: strategy.dailyROI, tpd: strategy.trades / strategy.btDays, mdd: strategy.maxDD },
                      { wr: strategy.alertWinRate, pf: strategy.alertPF, dr: strategy.alertDailyR, droi: strategy.alertDailyROI, tpd: strategy.alertTrades / fwdDays, mdd: strategy.alertMaxDD },
                    )}
                  </Card>
                )}

                {/* ---- Extended KPIs (collapsible) ---- */}
                <CollapsibleSection title="Extended KPIs" defaultOpen>
                  <PillTabs
                    tabs={['Performance', 'Risk-Adjusted', 'Distribution', 'Drawdown', 'Streaks']}
                    active={extKpiTab}
                    onChange={setExtKpiTab}
                  />

                  {extKpiTab === 'Performance' && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {[
                        { label: 'Wins', value: String(extendedKPIs.wins) },
                        { label: 'Losses', value: String(extendedKPIs.losses) },
                        { label: 'Best Trade', value: `+${extendedKPIs.bestTrade.toFixed(2)}R` },
                        { label: 'Worst Trade', value: `${extendedKPIs.worstTrade.toFixed(2)}R` },
                        { label: 'Avg Win', value: `+${extendedKPIs.avgWin.toFixed(2)}R` },
                        { label: 'Avg Loss', value: `${extendedKPIs.avgLoss.toFixed(2)}R` },
                        { label: 'Payoff Ratio', value: extendedKPIs.payoffRatio.toFixed(2) },
                        { label: 'Expected Daily R', value: `+${extendedKPIs.expectedDailyR.toFixed(2)}` },
                      ].map((kpi) => (
                        <Card key={kpi.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-base font-bold mt-1">{kpi.value}</p>
                        </Card>
                      ))}
                    </div>
                  )}

                  {extKpiTab === 'Risk-Adjusted' && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {[
                        { label: 'Sharpe', value: extendedKPIs.sharpe.toFixed(2) },
                        { label: 'Sortino', value: extendedKPIs.sortino.toFixed(2) },
                        { label: 'Calmar', value: extendedKPIs.calmar.toFixed(2) },
                        { label: 'Kelly Criterion', value: `${(extendedKPIs.kelly * 100).toFixed(1)}%` },
                        { label: 'Daily VaR', value: `${extendedKPIs.dailyVaR.toFixed(2)}R` },
                        { label: 'CVaR', value: `${extendedKPIs.cvar.toFixed(2)}R` },
                        { label: 'Volatility', value: `${extendedKPIs.volatility.toFixed(1)}%` },
                        { label: 'R\u00B2', value: extendedKPIs.rSquared.toFixed(2) },
                      ].map((kpi) => (
                        <Card key={kpi.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-base font-bold mt-1">{kpi.value}</p>
                        </Card>
                      ))}
                    </div>
                  )}

                  {extKpiTab === 'Distribution' && (
                    <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                      {[
                        { label: 'Skewness', value: extendedKPIs.skewness.toFixed(2) },
                        { label: 'Kurtosis', value: extendedKPIs.kurtosis.toFixed(2) },
                        { label: 'Tail Ratio', value: extendedKPIs.tailRatio.toFixed(2) },
                        { label: 'Outlier Win %', value: `${extendedKPIs.outlierWinPct.toFixed(1)}%` },
                        { label: 'Outlier Loss %', value: `${extendedKPIs.outlierLossPct.toFixed(1)}%` },
                      ].map((kpi) => (
                        <Card key={kpi.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-base font-bold mt-1">{kpi.value}</p>
                        </Card>
                      ))}
                    </div>
                  )}

                  {extKpiTab === 'Drawdown' && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      {[
                        { label: 'Max R DD', value: `${extendedKPIs.maxRDD.toFixed(1)}R` },
                        { label: 'Recovery Factor', value: extendedKPIs.recoveryFactor.toFixed(1) },
                        { label: 'Ulcer Index', value: extendedKPIs.ulcerIndex.toFixed(3) },
                        { label: 'Serenity Index', value: extendedKPIs.serenityIndex.toFixed(2) },
                        { label: 'Longest DD (trades)', value: String(extendedKPIs.longestDDTrades) },
                        { label: 'Longest DD (days)', value: String(extendedKPIs.longestDDDays) },
                      ].map((kpi) => (
                        <Card key={kpi.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-base font-bold mt-1">{kpi.value}</p>
                        </Card>
                      ))}
                    </div>
                  )}

                  {extKpiTab === 'Streaks' && (
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { label: 'Max Consec Wins', value: String(extendedKPIs.maxConsecWins) },
                        { label: 'Max Consec Losses', value: String(extendedKPIs.maxConsecLosses) },
                      ].map((kpi) => (
                        <Card key={kpi.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-base font-bold mt-1">{kpi.value}</p>
                        </Card>
                      ))}
                    </div>
                  )}
                </CollapsibleSection>

                {/* ---- Equity Curve ---- */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Equity Curve</h4>
                  <ChartPlaceholder
                    label="3-segment equity curve: Backtest (blue) | Forward Test (orange) | Alert Actual (green) with DD shading"
                    height={300}
                  />
                  <div className="flex items-center gap-6 mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span className="flex items-center gap-1.5">
                      <span style={{ display: 'inline-block', width: 16, height: 2, background: EQ_BT_COLOR }} /> Backtest
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span style={{ display: 'inline-block', width: 16, height: 2, background: EQ_FWD_COLOR }} /> Forward
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span style={{ display: 'inline-block', width: 16, height: 2, background: EQ_LIVE_COLOR }} /> Alerts
                    </span>
                  </div>
                  <div className="flex items-center gap-4 mt-3">
                    <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                      <input type="checkbox" checked={eqShowHWM} onChange={() => setEqShowHWM(!eqShowHWM)} /> HWM
                    </label>
                    <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                      <input type="checkbox" checked={eqShowEdge} onChange={() => setEqShowEdge(!eqShowEdge)} /> Edge Check
                    </label>
                    <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                      <input type="checkbox" checked={eqShowConf} onChange={() => setEqShowConf(!eqShowConf)} /> Confidence Bands
                    </label>
                  </div>
                </Card>

                {/* ---- R-Distribution ---- */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                  <Card>
                    <h4 className="text-sm font-medium mb-3">BT R-Distribution</h4>
                    <ChartPlaceholder label="R-Multiple histogram (backtest trades)" height={200} />
                  </Card>
                  <Card>
                    <h4 className="text-sm font-medium mb-3">FWD R-Distribution</h4>
                    <ChartPlaceholder label="R-Multiple histogram (forward test trades)" height={200} />
                  </Card>
                </div>

                {/* ---- Advanced Analysis (collapsible, if >= 20 trades) ---- */}
                {strategy.trades >= 20 && (
                  <CollapsibleSection title="Advanced Analysis">
                    <PillTabs
                      tabs={['Rolling Metrics', 'Return Distribution', 'Markov Motor']}
                      active={advancedTab}
                      onChange={setAdvancedTab}
                    />

                    {advancedTab === 'Rolling Metrics' && (
                      <Card>
                        <div className="flex items-center gap-4 mb-4 flex-wrap">
                          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            Window:
                            <input
                              type="range"
                              min={10}
                              max={100}
                              step={5}
                              value={rollingWindow}
                              onChange={(e) => setRollingWindow(Number(e.target.value))}
                              style={{ marginLeft: 8, verticalAlign: 'middle' }}
                            />
                            <span className="font-mono ml-1">{rollingWindow}</span>
                          </label>
                          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            Metric:
                            <select
                              style={{ ...selectStyle, marginLeft: 8 }}
                              value={rollingMetric}
                              onChange={(e) => setRollingMetric(e.target.value)}
                            >
                              {ROLLING_METRIC_OPTIONS.map((m) => (
                                <option key={m} value={m}>{m}</option>
                              ))}
                            </select>
                          </label>
                        </div>
                        <ChartPlaceholder
                          label={`Rolling ${rollingMetric} (window=${rollingWindow}) over trade sequence`}
                          height={250}
                        />
                      </Card>
                    )}

                    {advancedTab === 'Return Distribution' && (
                      <Card>
                        <div className="flex items-center gap-4 mb-4">
                          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            View:
                            <select
                              style={{ ...selectStyle, marginLeft: 8 }}
                              value={returnView}
                              onChange={(e) => setReturnView(e.target.value)}
                            >
                              {['Histogram', 'Box', 'Violin'].map((v) => (
                                <option key={v} value={v}>{v}</option>
                              ))}
                            </select>
                          </label>
                        </div>
                        <ChartPlaceholder
                          label={`Return distribution (${returnView} view) of R-multiples`}
                          height={250}
                        />
                        <div className="grid grid-cols-3 gap-3 mt-4">
                          <Card>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Skewness</p>
                            <p className="text-base font-bold mt-1">{extendedKPIs.skewness.toFixed(2)}</p>
                          </Card>
                          <Card>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Kurtosis</p>
                            <p className="text-base font-bold mt-1">{extendedKPIs.kurtosis.toFixed(2)}</p>
                          </Card>
                          <Card>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Tail Risk</p>
                            <p className="text-base font-bold mt-1">{extendedKPIs.tailRatio.toFixed(2)}</p>
                          </Card>
                        </div>
                      </Card>
                    )}

                    {advancedTab === 'Markov Motor' && (
                      <Card>
                        <div className="flex items-center gap-6 mb-4 flex-wrap">
                          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            Window:
                            <input
                              type="range"
                              min={10}
                              max={100}
                              step={5}
                              value={markovWindow}
                              onChange={(e) => setMarkovWindow(Number(e.target.value))}
                              style={{ marginLeft: 8, verticalAlign: 'middle' }}
                            />
                            <span className="font-mono ml-1">{markovWindow}</span>
                          </label>
                          <label className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            Edge Decay Threshold:
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.1}
                              value={edgeDecay}
                              onChange={(e) => setEdgeDecay(Number(e.target.value))}
                              style={{ marginLeft: 8, verticalAlign: 'middle' }}
                            />
                            <span className="font-mono ml-1">{edgeDecay.toFixed(1)}</span>
                          </label>
                        </div>

                        <ChartPlaceholder
                          label={`Rolling performance with Markov transitions (window=${markovWindow})`}
                          height={250}
                        />

                        {/* Transition probability table */}
                        <h5 className="text-xs font-medium mt-4 mb-2" style={{ color: 'var(--text-muted)' }}>
                          Transition Probability Matrix
                        </h5>
                        <div style={{ overflowX: 'auto' }}>
                          <table className="text-sm" style={{ borderCollapse: 'collapse' }}>
                            <thead>
                              <tr>
                                <th style={{ ...thStyle, minWidth: 80 }}>From / To</th>
                                <th style={{ ...thStyle, textAlign: 'center', minWidth: 80 }}>Win</th>
                                <th style={{ ...thStyle, textAlign: 'center', minWidth: 80 }}>Loss</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Win</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>{'--'}</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>{'--'}</td>
                              </tr>
                              <tr>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Loss</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>{'--'}</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>{'--'}</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>

                        <div className="flex items-center gap-6 mt-4">
                          <div>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trend Score</p>
                            <p className="text-base font-bold" style={{ color: 'var(--text-muted)' }}>{'--'}</p>
                          </div>
                          <div>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Edge Strength</p>
                            <p className="text-base font-bold" style={{ color: 'var(--text-muted)' }}>{'--'}</p>
                          </div>
                        </div>
                      </Card>
                    )}
                  </CollapsibleSection>
                )}
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 2: CHART & TRADES                                       */}
            {/* =========================================================== */}
            {tab === 'Chart & Trades' && (
              <div>
                {/* Chart controls */}
                <div className="flex items-center gap-4 mb-4 flex-wrap">
                  <label className="text-xs flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                    Candles:
                    <input
                      type="number"
                      min={50}
                      max={1000}
                      step={50}
                      value={candleCount}
                      onChange={(e) => setCandleCount(Number(e.target.value))}
                      style={{
                        ...selectStyle,
                        width: 80,
                      }}
                    />
                  </label>
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                    <input type="checkbox" checked={showConditions} onChange={() => setShowConditions(!showConditions)} />
                    Show Conditions
                  </label>
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                    <input type="checkbox" checked={showTriggers} onChange={() => setShowTriggers(!showTriggers)} />
                    Show Triggers
                  </label>
                </div>

                {/* OHLC Chart */}
                <Card className="mb-4">
                  <ChartPlaceholder
                    label="OHLC with indicators, trade markers, confluence heatmap"
                    height={400}
                  />
                  <div className="flex gap-4 mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span><span style={{ color: 'var(--green)' }}>+</span> Entry (long)</span>
                    <span><span style={{ color: 'var(--green)' }}>x</span> Exit (win)</span>
                    <span><span style={{ color: 'var(--red)' }}>x</span> Exit (loss)</span>
                    <span><span style={{ color: 'var(--orange)' }}>x</span> Exit (unconfirmed)</span>
                    <span><span style={{ color: '#009688' }}>x</span> Exit (bar count)</span>
                  </div>
                </Card>

                {/* Position Status */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Position Status</h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Status</p>
                      <p className="text-base font-bold mt-1" style={{ color: 'var(--text-muted)' }}>{'{{status}}'}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Signal</p>
                      <p className="text-base font-bold mt-1">{'{{last_signal}}'}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Signal Time</p>
                      <p className="text-base font-bold mt-1 font-mono">{'{{signal_time}}'}</p>
                    </div>
                    <div>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current Price</p>
                      <p className="text-base font-bold mt-1">{'{{current_price}}'}</p>
                    </div>
                  </div>
                </Card>

                {/* Current Conditions */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Current Conditions</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Condition', 'Current State', 'Needed State', 'Confluence'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {strategy.confluence.length === 0 ? (
                          <tr>
                            <td colSpan={4} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                              No confluence conditions configured
                            </td>
                          </tr>
                        ) : strategy.confluence.map((c, i) => (
                          <tr key={i}>
                            <td style={tdStyle}>{c}</td>
                            <td style={tdStyle}>
                              <span className="text-xs font-mono px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                                {'{{current_state}}'}
                              </span>
                            </td>
                            <td style={tdStyle}>
                              <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{'{{needed_state}}'}</span>
                            </td>
                            <td style={tdStyle}>
                              <span style={{ color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.75rem' }}>
                                {'{{met_status}}'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Alert History + Backtest Trade History side by side */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                  {/* Alert History */}
                  <Card>
                    <h4 className="text-sm font-medium mb-3">Alert History</h4>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Entry Time', 'Exit Time', 'Entry $', 'Exit $', 'R', 'Result'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {recentAlerts.length === 0 ? (
                            <tr>
                              <td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                No alerts available — enable monitoring to populate
                              </td>
                            </tr>
                          ) : recentAlerts.map((row: any, i: number) => (
                            <tr key={i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.entryTime || '--'}</td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.exitTime || '--'}</td>
                              <td style={tdStyle}>{row.entryPrice != null ? `$${Number(row.entryPrice).toFixed(2)}` : '--'}</td>
                              <td style={tdStyle}>{row.exitPrice != null ? `$${Number(row.exitPrice).toFixed(2)}` : '\u2014'}</td>
                              <td style={{ ...tdStyle, color: row.r && row.r >= 0 ? 'var(--green)' : row.r ? 'var(--red)' : 'var(--text-muted)', fontWeight: 600 }}>
                                {row.r != null ? `${row.r >= 0 ? '+' : ''}${Number(row.r).toFixed(2)}` : '\u2014'}
                              </td>
                              <td style={tdStyle}>
                                <span style={{
                                  color: row.result === 'Win' ? 'var(--green)' : row.result === 'Loss' ? 'var(--red)' : 'var(--orange)',
                                  fontWeight: 600, fontSize: '0.75rem',
                                }}>
                                  {row.result || '--'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>

                  {/* Backtest Trade History */}
                  <Card>
                    <h4 className="text-sm font-medium mb-3">Trade History (Backtest)</h4>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Entry Time', 'Exit Time', 'Entry $', 'Exit $', 'R', 'Result'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {btTrades.length === 0 ? (
                            <tr>
                              <td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                No trades available — run backtest to populate
                              </td>
                            </tr>
                          ) : btTrades.slice(0, 20).map((row: any, i: number) => (
                            <tr key={i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.entryTime || '--'}</td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.exitTime || '--'}</td>
                              <td style={tdStyle}>{row.entryPrice != null ? `$${Number(row.entryPrice).toFixed(2)}` : '--'}</td>
                              <td style={tdStyle}>{row.exitPrice != null ? `$${Number(row.exitPrice).toFixed(2)}` : '--'}</td>
                              <td style={{ ...tdStyle, color: row.pnlR >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                                {row.pnlR != null ? `${row.pnlR >= 0 ? '+' : ''}${Number(row.pnlR).toFixed(2)}` : '--'}
                              </td>
                              <td style={tdStyle}>
                                <span style={{
                                  color: row.pnlR >= 0 ? 'var(--green)' : 'var(--red)',
                                  fontWeight: 600, fontSize: '0.75rem',
                                }}>
                                  {row.pnlR != null ? (row.pnlR >= 0 ? 'Win' : 'Loss') : '--'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </div>

                {/* Forward Test Trades */}
                <Card className="mb-4">
                  <button
                    className="flex items-center gap-2 w-full text-left text-sm font-medium mb-3"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-primary)', padding: 0 }}
                    onClick={() => setFwdTradesOpen(!fwdTradesOpen)}
                  >
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{fwdTradesOpen ? '\u25BC' : '\u25B6'}</span>
                    Forward Test Trades
                    <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({fwdTrades.length})</span>
                  </button>
                  {fwdTradesOpen && (
                    fwdLoading ? (
                      <div className="flex items-center gap-3 py-6 justify-center">
                        <div className="w-4 h-4 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
                        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading forward test trades from Polygon...</span>
                      </div>
                    ) : renderTradeTable(fwdTrades)
                  )}
                </Card>

                {/* Backtest Trades */}
                <Card>
                  <button
                    className="flex items-center gap-2 w-full text-left text-sm font-medium mb-3"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-primary)', padding: 0 }}
                    onClick={() => setBtTradesOpen(!btTradesOpen)}
                  >
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{btTradesOpen ? '\u25BC' : '\u25B6'}</span>
                    Backtest Trades
                    <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({btTrades.length})</span>
                  </button>
                  {btTradesOpen && renderTradeTable(btTrades)}
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 3: CONFLUENCE ANALYSIS                                   */}
            {/* =========================================================== */}
            {tab === 'Confluence Analysis' && (
              <div>
                {confluenceGroups.length === 0 ? (
                  <p className="text-sm py-4" style={{ color: 'var(--text-muted)' }}>No confluence groups available — wire to API to populate.</p>
                ) : (
                <>
                <PillTabs
                  tabs={confluenceGroups.map((g) => g.name)}
                  active={selectedConfGroup}
                  onChange={setSelectedConfGroup}
                />

                {confluenceGroups
                  .filter((g) => g.name === selectedConfGroup)
                  .map((group) => (
                    <div key={group.id}>
                      {/* Indicator chart */}
                      <Card className="mb-4">
                        <h4 className="text-sm font-medium mb-3">{group.name} &mdash; Indicator Chart</h4>
                        <ChartPlaceholder
                          label={`${group.name} indicator chart with overlays and state regions`}
                          height={300}
                        />
                      </Card>

                      {/* Interpreter State Timeline + Trigger Events side by side */}
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        {/* Interpreter state timeline */}
                        <Card>
                          <h4 className="text-sm font-medium mb-3">Interpreter State Timeline</h4>
                          <div style={{ overflowX: 'auto' }}>
                            <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                              <thead>
                                <tr>
                                  {['Time', 'State', 'Previous'].map((h) => (
                                    <th key={h} style={thStyle}>{h}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {confluenceTimeline.length === 0 ? (
                                  <tr>
                                    <td colSpan={3} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                      No data — run backtest to populate
                                    </td>
                                  </tr>
                                ) : confluenceTimeline.map((row: any, i: number) => (
                                  <tr key={i}>
                                    <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.time}</td>
                                    <td style={tdStyle}>
                                      <span
                                        className="text-xs font-mono px-2 py-0.5 rounded-full"
                                        style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
                                      >
                                        {row.state}
                                      </span>
                                    </td>
                                    <td style={tdStyle}>
                                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                        {row.previous}
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </Card>

                        {/* Trigger events */}
                        <Card>
                          <h4 className="text-sm font-medium mb-3">Trigger Events</h4>
                          <div style={{ overflowX: 'auto' }}>
                            <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                              <thead>
                                <tr>
                                  {['Time', 'Trigger', 'Sentiment', 'Price', 'Exec'].map((h) => (
                                    <th key={h} style={thStyle}>{h}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {confluenceTriggerEvents.length === 0 ? (
                                  <tr>
                                    <td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                      No data — run backtest to populate
                                    </td>
                                  </tr>
                                ) : confluenceTriggerEvents.map((row: any, i: number) => (
                                  <tr key={i}>
                                    <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.time}</td>
                                    <td style={tdStyle}>{row.trigger}</td>
                                    <td style={tdStyle}>
                                      <span style={{ color: row.sentiment === 'Bullish' ? 'var(--green)' : 'var(--red)' }}>
                                        {row.sentiment}
                                      </span>
                                    </td>
                                    <td style={tdStyle}>${row.price != null ? Number(row.price).toFixed(2) : '--'}</td>
                                    <td style={tdStyle}>
                                      <span
                                        className="text-xs font-mono px-1.5 py-0.5 rounded-full"
                                        style={{
                                          color: execTypeColors[row.exec] || EXEC_BADGE_COLOR,
                                          background: (execTypeColors[row.exec] || EXEC_BADGE_COLOR) + '20',
                                        }}
                                      >
                                        {row.exec}
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </Card>
                      </div>
                    </div>
                  ))}
                </>
                )}

                {/* Exit Reason Breakdown — per-trigger comparison */}
                {triggerAnalysis?.exit_breakdown && triggerAnalysis.exit_breakdown.length > 0 && (
                  <Card className="mt-4">
                    <h4 className="text-sm font-medium mb-3">Exit Reason Breakdown</h4>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Exit Reason', 'Trades', 'Win Rate', 'Total R', 'Avg R', 'Best', 'Worst'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {triggerAnalysis.exit_breakdown.map((row) => (
                            <tr key={row.exit_reason}>
                              <td style={tdStyle}>
                                <span className="text-xs font-mono px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                                  {row.exit_reason.replace(/_/g, ' ')}
                                </span>
                              </td>
                              <td style={tdStyle}>{row.trades}</td>
                              <td style={{ ...tdStyle, color: row.win_rate >= 50 ? 'var(--green)' : 'var(--red)' }}>
                                {row.win_rate.toFixed(1)}%
                              </td>
                              <td style={{ ...tdStyle, color: row.total_r >= 0 ? 'var(--green)' : 'var(--red)', fontFamily: 'monospace' }}>
                                {row.total_r >= 0 ? '+' : ''}{row.total_r.toFixed(2)}R
                              </td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace' }}>
                                {row.avg_r >= 0 ? '+' : ''}{row.avg_r.toFixed(2)}R
                              </td>
                              <td style={{ ...tdStyle, color: 'var(--green)', fontFamily: 'monospace' }}>
                                +{row.best_trade.toFixed(2)}R
                              </td>
                              <td style={{ ...tdStyle, color: 'var(--red)', fontFamily: 'monospace' }}>
                                {row.worst_trade.toFixed(2)}R
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 4: CONFIGURATION                                        */}
            {/* =========================================================== */}
            {tab === 'Configuration' && (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {/* Left: Strategy Setup */}
                  <Card>
                    <h4 className="text-sm font-medium mb-4">Strategy Setup</h4>
                    <div className="flex flex-col gap-3">
                      {[
                        { label: 'Ticker', value: strategy.symbol },
                        { label: 'Direction', value: strategy.direction },
                        { label: 'Timeframe', value: strategy.timeframe },
                        { label: 'Session', value: strategy.session },
                        { label: 'Method', value: strategy.method },
                        { label: 'Created', value: strategy.createdAt },
                        { label: 'Updated', value: strategy.updatedAt },
                        { label: 'FW Start', value: strategy.fwdSince },
                      ].map((row) => (
                        <div key={row.label} className="flex items-center justify-between">
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{row.label}</span>
                          <span className="text-sm font-medium">{row.value}</span>
                        </div>
                      ))}
                    </div>
                  </Card>

                  {/* Right: Variables */}
                  <Card>
                    <h4 className="text-sm font-medium mb-4">Variables</h4>
                    <div className="flex flex-col gap-3">
                      {/* Entry */}
                      <div>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Entry</span>
                        <div className="flex items-center gap-2 mt-1 flex-wrap">
                          {entryParsed.exec && <ExecBadge exec={entryParsed.exec} />}
                          <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                            {entryPack.pack && <>{entryPack.pack} &gt; </>}{entryPack.trigger}
                          </span>
                        </div>
                      </div>
                      {/* Exit */}
                      <div>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Exit</span>
                        {exitsParsed.map((e, i) => (
                          <div key={i} className="flex items-center gap-2 mt-1 flex-wrap">
                            {e.exec && <ExecBadge exec={e.exec} />}
                            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                              {e.pack && <>{e.pack} &gt; </>}{e.trigger}
                            </span>
                          </div>
                        ))}
                      </div>
                      {/* Stop */}
                      <div>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Stop</span>
                        <div className="mt-1">
                          <StopBadge text={strategy.stop} />
                        </div>
                      </div>
                      {/* Target */}
                      <div>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Target</span>
                        <div className="mt-1">
                          <TargetBadge text={strategy.target} />
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>

                {/* Confluence conditions with fidelity badges */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Confluence Conditions</h4>
                  <div className="flex items-center gap-2 flex-wrap">
                    <FidelityBadge label="[PB]" />
                    {strategy.confluence.map((c) => (
                      <ConditionBadge key={c} text={c} />
                    ))}
                  </div>
                  {strategy.confluence.length === 0 && (
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>No confluence conditions configured.</p>
                  )}
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 5: ALERTS                                               */}
            {/* =========================================================== */}
            {tab === 'Alerts' && (
              <div>
                {/* ---- Webhook Execution Flow ---- */}
                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
                  Webhook Execution Flow
                </h3>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  Based on this strategy&apos;s trigger configuration, the system will send the following webhook events to your exchange. Webhooks only fire when an action is needed — all internal logic (hold times, confirmations) is resolved first.
                </p>

                {/* Flow diagram card */}
                <Card className="mb-6">
                  <h4 className="text-sm font-medium mb-4">Entry &rarr; Exit Lifecycle</h4>

                  {/* This strategy uses [C] — show the C flow as primary example */}
                  {/* The flow adapts based on exec type */}
                  <div className="flex flex-col gap-0">
                    {/* Step 1: Entry */}
                    <div className="flex items-start gap-3">
                      <div className="flex flex-col items-center" style={{ minWidth: 24 }}>
                        <span style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--green)', display: 'block' }} />
                        <span style={{ width: 2, height: 40, background: 'var(--border)', display: 'block' }} />
                      </div>
                      <div className="pb-3">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-semibold px-2 py-0.5 rounded-full" style={{ color: 'var(--green)', background: 'var(--green-muted)' }}>
                            entry_long_market
                          </span>
                        </div>
                        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          <strong>Fires:</strong> At bar close when {entryPack.trigger} evaluates TRUE and all confluence conditions are met.
                        </p>
                        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                          <strong>Order:</strong> Market buy at close price &middot; <strong>Delay:</strong> None (immediate)
                        </p>
                      </div>
                    </div>

                    {/* Step 2: Position open */}
                    <div className="flex items-start gap-3">
                      <div className="flex flex-col items-center" style={{ minWidth: 24 }}>
                        <span style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--text-muted)', border: '2px solid var(--border)', display: 'block', boxSizing: 'border-box' }} />
                        <span style={{ width: 2, height: 40, background: 'var(--border)', display: 'block' }} />
                      </div>
                      <div className="pb-3">
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                          Position is open. Stop loss at swing low. Monitoring for: exit trigger (opposite signal), stop loss breach, take profit target, or bar count exit.
                        </p>
                      </div>
                    </div>

                    {/* Step 3: Exit */}
                    <div className="flex items-start gap-3">
                      <div className="flex flex-col items-center" style={{ minWidth: 24 }}>
                        <span style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--red)', display: 'block' }} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-semibold px-2 py-0.5 rounded-full" style={{ color: 'var(--red)', background: 'var(--red-muted)' }}>
                            exit_long_market
                          </span>
                        </div>
                        <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          <strong>Fires:</strong> On exit trigger (opposite signal), stop loss breach, take profit target, or bar count max hold.
                        </p>
                        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                          <strong>Order:</strong> Market sell at current price &middot; <strong>Delay:</strong> None (immediate)
                        </p>
                      </div>
                    </div>
                  </div>
                </Card>

                {/* All possible webhook events for this strategy */}
                <Card className="mb-6">
                  <h4 className="text-sm font-medium mb-3">Webhook Events This Strategy May Fire</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Event Type', 'Order', 'When It Fires', 'Conditions'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          {
                            event: 'entry_long_market',
                            color: 'var(--green)',
                            order: 'Market Buy',
                            when: 'Bar close',
                            conditions: 'Entry trigger TRUE + all confluence conditions met',
                          },
                          {
                            event: 'exit_long_market',
                            color: 'var(--red)',
                            order: 'Market Sell',
                            when: 'Bar close or intra-bar',
                            conditions: 'Exit trigger (opposite signal), stop loss breach, take profit target, or bar count max hold',
                          },
                        ].map((row, i) => (
                          <tr key={i}>
                            <td style={tdStyle}>
                              <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full" style={{ color: row.color, background: row.color + '20' }}>
                                {row.event}
                              </span>
                            </td>
                            <td style={tdStyle}>{row.order}</td>
                            <td style={tdStyle}>{row.when}</td>
                            <td style={{ ...tdStyle, fontSize: '0.75rem' }}>{row.conditions}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                    This strategy&apos;s triggers are configured with market order types. Triggers configured with limit orders may also produce cancel events. Triggers with a confirmation step may produce additional exit events if confirmation fails.
                  </p>
                </Card>

                {/* Execution type reference */}
                <CollapsibleSection title="Trigger Configuration Webhook Reference" defaultOpen>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                    {/* Bar close trigger, no confirmation */}
                    <Card>
                      <div className="flex items-center gap-2 mb-3">
                        <ExecBadge exec="[C]" />
                        <span className="text-sm font-medium">Bar Close Entry</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>&mdash; no confirmation</span>
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>or</span>
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_limit</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>when bar closes</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--red)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--red)' }}>exit_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>on stop loss, exit trigger, target, or bar count</span>
                        </div>
                      </div>
                      <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>2 webhooks. Market or limit depending on trigger&apos;s order_type parameter.</p>
                    </Card>

                    {/* Level cross trigger, no confirmation */}
                    <Card>
                      <div className="flex items-center gap-2 mb-3">
                        <ExecBadge exec="[L]" />
                        <span className="text-sm font-medium">Level Cross Entry</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>&mdash; no confirmation</span>
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>or</span>
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_limit</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>when price crosses level</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--orange)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--orange)' }}>cancel_X</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>if limit order not filled within timeout</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--red)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--red)' }}>exit_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>on stop loss, exit trigger, target, or bar count</span>
                        </div>
                      </div>
                      <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>2-3 webhooks. Market or limit depending on trigger&apos;s order_type. Fires after hold_seconds if configured. Cancel only if limit times out.</p>
                    </Card>

                    {/* Level cross entry + bar close confirmation */}
                    <Card>
                      <div className="flex items-center gap-2 mb-3">
                        <ExecBadge exec="[LC]" />
                        <span className="text-sm font-medium">Level Cross Entry + Bar Close Confirmation</span>
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>or</span>
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_limit</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>when price crosses level</span>
                        </div>
                        <div className="flex items-center gap-2 pl-3.5" style={{ borderLeft: '2px dashed var(--border)', marginLeft: 2 }}>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>waits for bar close to confirm position...</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--orange)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--orange)' }}>exit_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>or</span>
                          <span className="text-xs font-mono" style={{ color: 'var(--orange)' }}>exit_X_limit</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>if bar close does NOT confirm</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--red)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--red)' }}>exit_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>if confirmed: on stop loss, exit trigger, target, or bar count</span>
                        </div>
                      </div>
                      <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Entry at level cross, then bar close must confirm. If unconfirmed, exits immediately. Market or limit depending on trigger&apos;s order_type.</p>
                    </Card>

                    {/* Bar close entry + next bar close confirmation */}
                    <Card>
                      <div className="flex items-center gap-2 mb-3">
                        <ExecBadge exec="[CC]" />
                        <span className="text-sm font-medium">Bar Close Entry + Next Bar Close Confirmation</span>
                      </div>
                      <div className="flex flex-col gap-1.5">
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--green)' }}>entry_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>when bar closes</span>
                        </div>
                        <div className="flex items-center gap-2 pl-3.5" style={{ borderLeft: '2px dashed var(--border)', marginLeft: 2 }}>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>waits for next bar close to confirm position...</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--orange)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--orange)' }}>exit_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>if next bar close does NOT confirm</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--red)', display: 'inline-block' }} />
                          <span className="text-xs font-mono" style={{ color: 'var(--red)' }}>exit_X_market</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>if confirmed: on stop loss, exit trigger, target, or bar count</span>
                        </div>
                      </div>
                      <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>Entry at bar close, then next bar must confirm. If unconfirmed, exits at next bar close.</p>
                    </Card>
                  </div>
                </CollapsibleSection>

                {/* Available Payload Data */}
                <CollapsibleSection title="Available Payload Data">
                  <Card className="mb-4">
                    <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                      These fields are available as <span className="font-mono">{'{{placeholder}}'}</span> tokens in your webhook template JSON payloads. Values shown are from the most recent alert fired by this strategy &mdash; verify they match your expectations.
                    </p>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Placeholder', 'Last Alert Value', 'Source'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { ph: '{{event_type}}', val: '{{event_type}}', src: 'Derived from execution type + direction' },
                            { ph: '{{symbol}}', val: strategy.symbol, src: 'Strategy' },
                            { ph: '{{direction}}', val: strategy.direction, src: 'Strategy' },
                            { ph: '{{order_action}}', val: '{{order_action}}', src: 'Derived (buy/sell/close)' },
                            { ph: '{{order_type}}', val: '{{order_type}}', src: 'Derived from execution type' },
                            { ph: '{{order_price}}', val: '{{order_price}}', src: 'Fill price or limit price' },
                            { ph: '{{stop_price}}', val: '{{stop_price}}', src: 'Calculated stop' },
                            { ph: '{{quantity}}', val: '{{quantity}}', src: 'Portfolio (risk / stop distance)' },
                            { ph: '{{trigger_name}}', val: '{{trigger_name}}', src: 'Strategy trigger' },
                            { ph: '{{atr}}', val: '{{atr}}', src: 'Indicator at signal bar' },
                            { ph: '{{confluence_met}}', val: strategy.confluence.length > 0 ? strategy.confluence.join(', ') : '{{confluence_met}}', src: 'Active conditions at signal' },
                            { ph: '{{portfolio_name}}', val: '{{portfolio_name}}', src: 'Portfolio' },
                            { ph: '{{risk_per_trade}}', val: '{{risk_per_trade}}', src: 'Portfolio' },
                            { ph: '{{timestamp}}', val: '{{timestamp}}', src: 'Signal time' },
                          ].map((row, i) => (
                            <tr key={i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--accent)' }}>{row.ph}</td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.val}</td>
                              <td style={{ ...tdStyle, fontSize: '0.75rem', color: 'var(--text-muted)' }}>{row.src}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </CollapsibleSection>

                {/* Recent alerts table */}
                <Card className="mb-6">
                  <h4 className="text-sm font-medium mb-3">Recent Alerts</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Time', 'Event', 'Trigger', 'Price', 'Order', 'Status'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {recentAlerts.length === 0 ? (
                          <tr>
                            <td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                              No alerts available — enable monitoring to populate
                            </td>
                          </tr>
                        ) : recentAlerts.map((alert: any, i: number) => (
                          <tr key={i}>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{alert.time || '--'}</td>
                            <td style={tdStyle}>
                              <span
                                className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full"
                                style={{
                                  color: alert.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                                  background: alert.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                                }}
                              >
                                {alert.type === 'ENTRY' ? 'entry_long_market' : 'exit_long_market'}
                              </span>
                            </td>
                            <td style={tdStyle}>{alert.trigger || '--'}</td>
                            <td style={tdStyle}>{alert.price != null ? `$${Number(alert.price).toFixed(2)}` : '--'}</td>
                            <td style={tdStyle}>
                              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Market</span>
                            </td>
                            <td style={tdStyle}>
                              <span style={{
                                color: alert.status === 'Delivered' ? 'var(--green)' : 'var(--red)',
                              }}>
                                {alert.status || '--'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Trade-to-Alert mapping */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Trade-to-Alert Mapping</h4>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Trade #', 'BT Entry', 'Alert Entry', 'Entry \u0394', 'BT Exit', 'Alert Exit', 'Exit \u0394'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {tradeAlertMapping.length === 0 ? (
                          <tr>
                            <td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                              No data — enable monitoring to populate
                            </td>
                          </tr>
                        ) : tradeAlertMapping.map((row: any, i: number) => (
                          <tr key={i}>
                            <td style={tdStyle}>{row.tradeNum}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.btEntry}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.alertEntry}</td>
                            <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.entryDelta}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.btExit}</td>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.alertExit}</td>
                            <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.exitDelta}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 6: ALERT ANALYSIS                                       */}
            {/* =========================================================== */}
            {tab === 'Alert Analysis' && (
              <div>
                {/* ---- Discrepancies Section ---- */}
                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Discrepancies</h3>

                {/* Missed Alerts */}
                <Card className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Missed Alerts</h4>
                  {alertAnalysis.missed.length === 0 ? (
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No missed alerts.</p>
                  ) : (
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Trade #', 'Entry Time', 'Monitor Active', 'Status'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {alertAnalysis.missed.map((row: any, i: number) => (
                            <tr key={i}>
                              <td style={tdStyle}>{row.tradeNum}</td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.entryTime}</td>
                              <td style={tdStyle}>
                                <span style={{ color: row.monitorActive ? 'var(--green)' : 'var(--red)' }}>
                                  {row.monitorActive ? 'Yes' : 'No'}
                                </span>
                              </td>
                              <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.status}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </Card>

                {/* Phantom Alerts */}
                <Card className="mb-6">
                  <h4 className="text-sm font-medium mb-3">Phantom Alerts</h4>
                  {alertAnalysis.phantom.length === 0 ? (
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No phantom alerts.</p>
                  ) : (
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Alert ID', 'Type', 'Timestamp', 'Price', 'Status'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {alertAnalysis.phantom.map((row: any, i: number) => (
                            <tr key={i}>
                              <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{row.alertId}</td>
                              <td style={tdStyle}>
                                <span
                                  className="text-xs font-semibold px-2 py-0.5 rounded-full"
                                  style={{
                                    color: row.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                                    background: row.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                                  }}
                                >
                                  {row.type}
                                </span>
                              </td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.timestamp}</td>
                              <td style={tdStyle}>{row.price != null ? `$${Number(row.price).toFixed(2)}` : '--'}</td>
                              <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.status}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </Card>

                {/* ---- Summary Metrics ---- */}
                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Summary Metrics</h3>
                <Card className="mb-6">
                  <div style={{ overflowX: 'auto' }}>
                    <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {['Metric', 'FT (All)', 'FT (Alerts-On)', 'Alert Actual', 'Delta'].map((h) => (
                            <th key={h} style={thStyle}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {alertAnalysis.summaryMetrics.length === 0 ? (
                          <tr>
                            <td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                              No data — enable monitoring to populate
                            </td>
                          </tr>
                        ) : alertAnalysis.summaryMetrics.map((row: any, i: number) => {
                          const deltaNum = parseFloat(row.delta);
                          const isDelta = !isNaN(deltaNum) && row.delta !== '--';
                          const isNeg = isDelta && deltaNum < 0;
                          const isPos = isDelta && deltaNum > 0;
                          // For slippage/missed/phantom, positive delta is bad
                          const isBadIfPositive = ['Avg Slippage (R)', 'Entry Slip $/sh', 'Exit Slip $/sh', 'Missed Count', 'Phantom Count'].includes(row.label);
                          const deltaColor = isDelta
                            ? (isBadIfPositive ? (isPos ? 'var(--red)' : isNeg ? 'var(--green)' : 'var(--text-secondary)') : (isNeg ? 'var(--red)' : isPos ? 'var(--green)' : 'var(--text-secondary)'))
                            : 'var(--text-muted)';

                          return (
                            <tr key={i}>
                              <td style={tdStyle}>{row.label}</td>
                              <td style={{ ...tdStyle, textAlign: 'right' }}>{row.ftAll}</td>
                              <td style={{ ...tdStyle, textAlign: 'right' }}>{row.ftAlertsOn}</td>
                              <td style={{ ...tdStyle, textAlign: 'right' }}>{row.alertActual}</td>
                              <td style={{ ...tdStyle, textAlign: 'right', color: deltaColor, fontWeight: isDelta ? 600 : 400 }}>{row.delta}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* ---- Position Health (collapsible) ---- */}
                <CollapsibleSection title="Position Health">
                  <Card className="mb-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                      {[
                        { label: 'Status', value: alertAnalysis.positionHealth.status },
                        { label: 'Entries', value: String(alertAnalysis.positionHealth.entries) },
                        { label: 'Exits', value: String(alertAnalysis.positionHealth.exits) },
                        { label: 'Avg Hold Time', value: alertAnalysis.positionHealth.avgHoldTime },
                      ].map((kpi) => (
                        <div key={kpi.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-base font-bold mt-1" style={{
                            color: kpi.label === 'Status' && kpi.value === 'Healthy' ? 'var(--green)' : 'var(--text-primary)',
                          }}>
                            {kpi.value}
                          </p>
                        </div>
                      ))}
                    </div>

                    {alertAnalysis.positionHealth.anomalies.length > 0 && (
                      <>
                        <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Anomalies</h5>
                        <div style={{ overflowX: 'auto' }}>
                          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                            <thead>
                              <tr>
                                {['Time', 'Type', 'Detail'].map((h) => (
                                  <th key={h} style={thStyle}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {alertAnalysis.positionHealth.anomalies.map((row, i) => (
                                <tr key={i}>
                                  <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.time}</td>
                                  <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.type}</td>
                                  <td style={tdStyle}>{row.detail}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </>
                    )}
                  </Card>
                </CollapsibleSection>

                {/* ---- Trigger Timing (collapsible) ---- */}
                <CollapsibleSection title="Trigger Timing">
                  <Card className="mb-4">
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Trade #', 'Type', 'Exec', 'Trigger', 'Theo Time', 'Alert Time', 'Time \u0394', 'Theo Price', 'Alert Price', 'Price \u0394', 'Slip (R)'].map((h) => (
                              <th key={h} style={{ ...thStyle, fontSize: '0.6875rem', padding: '6px 8px' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {alertAnalysis.triggerTiming.length === 0 ? (
                            <tr>
                              <td colSpan={11} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                No data — enable monitoring to populate
                              </td>
                            </tr>
                          ) : alertAnalysis.triggerTiming.map((row: any, i: number) => (
                            <tr key={i}>
                              <td style={tdStyle}>{row.tradeNum}</td>
                              <td style={tdStyle}>
                                <span style={{ color: row.type === 'ENTRY' ? 'var(--green)' : 'var(--red)' }}>{row.type}</span>
                              </td>
                              <td style={tdStyle}>
                                <span
                                  className="text-xs font-mono px-1 py-0.5 rounded-full"
                                  style={{
                                    color: execTypeColors[row.exec] || EXEC_BADGE_COLOR,
                                    background: (execTypeColors[row.exec] || EXEC_BADGE_COLOR) + '20',
                                  }}
                                >
                                  {row.exec}
                                </span>
                              </td>
                              <td style={{ ...tdStyle, fontSize: '0.75rem' }}>{row.trigger}</td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.theoTime}</td>
                              <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{row.alertTime}</td>
                              <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.timeDelta}</td>
                              <td style={tdStyle}>{row.theoPrice != null ? `$${Number(row.theoPrice).toFixed(2)}` : '--'}</td>
                              <td style={tdStyle}>{row.alertPrice != null ? `$${Number(row.alertPrice).toFixed(2)}` : '--'}</td>
                              <td style={{ ...tdStyle, color: 'var(--orange)' }}>{row.priceDelta}</td>
                              <td style={{
                                ...tdStyle,
                                color: row.slipR <= 0 ? 'var(--green)' : 'var(--red)',
                                fontWeight: 600,
                              }}>
                                {row.slipR >= 0 ? '+' : ''}{Number(row.slipR).toFixed(2)}R
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </CollapsibleSection>

                {/* ---- Trade-by-Trade (collapsible) ---- */}
                <CollapsibleSection title="Trade-by-Trade Comparison">
                  <Card>
                    <h5 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
                      R-Multiple Comparison &amp; Dollar Slippage
                    </h5>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Trade #', 'FT R', 'Live R', 'Delta R', 'Entry Slip', 'Exit Slip', 'Net Slip'].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {alertAnalysis.tradeByTrade.length === 0 ? (
                            <tr>
                              <td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                No data — enable monitoring to populate
                              </td>
                            </tr>
                          ) : alertAnalysis.tradeByTrade.map((row: any, i: number) => (
                            <tr key={i}>
                              <td style={tdStyle}>{row.tradeNum}</td>
                              <td style={{
                                ...tdStyle,
                                color: row.ftR >= 0 ? 'var(--green)' : 'var(--red)',
                                fontWeight: 600,
                              }}>
                                {row.ftR >= 0 ? '+' : ''}{Number(row.ftR).toFixed(2)}R
                              </td>
                              <td style={{
                                ...tdStyle,
                                color: row.liveR >= 0 ? 'var(--green)' : 'var(--red)',
                                fontWeight: 600,
                              }}>
                                {row.liveR >= 0 ? '+' : ''}{Number(row.liveR).toFixed(2)}R
                              </td>
                              <td style={{
                                ...tdStyle,
                                color: row.deltaR >= 0 ? 'var(--green)' : 'var(--red)',
                                fontWeight: 600,
                              }}>
                                {row.deltaR >= 0 ? '+' : ''}{Number(row.deltaR).toFixed(2)}R
                              </td>
                              <td style={{
                                ...tdStyle,
                                color: row.entrySlip <= 0 ? 'var(--green)' : 'var(--red)',
                              }}>
                                ${row.entrySlip >= 0 ? '+' : ''}{Number(row.entrySlip).toFixed(2)}
                              </td>
                              <td style={{
                                ...tdStyle,
                                color: row.exitSlip <= 0 ? 'var(--green)' : 'var(--red)',
                              }}>
                                ${row.exitSlip >= 0 ? '+' : ''}{Number(row.exitSlip).toFixed(2)}
                              </td>
                              <td style={{
                                ...tdStyle,
                                color: row.netSlip <= 0 ? 'var(--green)' : 'var(--red)',
                                fontWeight: 600,
                              }}>
                                ${row.netSlip >= 0 ? '+' : ''}{Number(row.netSlip).toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </CollapsibleSection>
              </div>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
