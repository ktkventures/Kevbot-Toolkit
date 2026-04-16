'use client';

import { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import type { TradeMarker } from '@/charts/TradingChart';

// Static imports for hooks — these are safe because the page uses ssr:false
import { useStrategy, useStrategyTrades, useStrategyForwardTest, useStrategyKPIs, useTriggerAnalysis, useStrategyChartData, useConfluenceChart, useTradeZoom } from '@/hooks/queries/useStrategies';
import { useStrategyAlerts } from '@/hooks/queries/useAlerts';
import { useBars } from '@/hooks/queries/useMarketData';
import { useLiveBar } from '@/hooks/queries/useLiveBar';
import { useDeleteStrategy, useDuplicateStrategy, useRefreshStrategy } from '@/hooks/mutations/useStrategyMutations';
import { useDisplayStore } from '@/providers/StoreProvider';
import { useChartPrefs } from '@/hooks/useChartPrefs';

// Dynamic imports for chart components — forces separate webpack chunks
const EquityCurve = dynamic(() => import('@/charts/EquityCurve'), { ssr: false });
const PerformanceVsPlan = dynamic(() => import('@/charts/PerformanceVsPlan'), { ssr: false });
const TradeZoomModal = dynamic(() => import('@/components/TradeZoomModal'), { ssr: false });
const DistributionChart = dynamic(() => import('@/charts/DistributionChart'), { ssr: false });
const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });

import { buildStrategyChartPanes } from '@/charts/buildStrategyChartPanes';

type PaneConfig = import('@/charts/SyncedChartPane').PaneConfig;
type SeriesConfig = import('@/charts/SyncedChartPane').SeriesConfig;

/* ========================================================================= */
/* COLOR CONSTANTS                                                            */
/* ========================================================================= */

const INDICATOR_COLORS = [
  '#2196F3', '#FF9800', '#4CAF50', '#E91E63',
  '#00BCD4', '#9C27B0', '#FFC107', '#795548',
];

const EXEC_BADGE_COLOR = '#2196F3';
const FIDELITY_BADGE_COLOR = '#26C6DA';
const EQ_BT_COLOR = '#2196F3';
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

/* ========================================================================= */
/* API DATA MAPPING                                                           */
/* ========================================================================= */

function apiToDetailStrategy(s: any) {
  if (!s) { console.warn('[StrategyDetail] apiToDetailStrategy: null/undefined strategy'); return null; }
  if (!s.kpis) console.warn('[StrategyDetail] Strategy', s.id, 'has no kpis object');
  if (!s.stored_trades?.length) console.warn('[StrategyDetail] Strategy', s.id, 'has 0 stored_trades');
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
    entry: s.entry_trigger_display_name || s.entry_trigger_confluence_id || '--',
    entryId: s.entry_trigger_confluence_id || '',
    exit: (s.exit_trigger_confluence_ids || s.exit_triggers || []).map((eid: string) =>
      s.exit_trigger_display_names?.[eid] || eid
    ),
    exitIds: s.exit_trigger_confluence_ids || s.exit_triggers || [],
    barCountExit: s.bar_count_exit ?? null,
    stop: formatStopDisplay(s.stop_config),
    stopConfig: s.stop_config || null,
    target: formatTargetDisplay(s.target_config),
    targetConfig: s.target_config || null,
    timeExitConfig: s.time_exit_config || null,
    timeExitSummary: s.time_exit_config ? formatTimeExitDisplay(s.time_exit_config) : null,
    confluence: s.confluence || [],
    confluenceEnriched: s.confluence_enriched || [],
    winRate: k.win_rate ?? 0,
    pf: k.profit_factor ?? 0,
    dailyR: k.daily_r ?? 0,
    dailyROI: 0,  // {{daily_roi}} — requires portfolio-level risk context to compute
    trades: k.total_trades ?? 0,
    maxDD: k.max_r_drawdown ?? 0,
    btDays: s.data_days || 30,
    btStart: s.stored_trades?.[0]?.entry_time?.slice(0, 10) || '--',
    btEnd: s.stored_trades?.length ? (s.stored_trades[s.stored_trades.length - 1]?.exit_time?.slice(0, 10) || s.stored_trades[s.stored_trades.length - 1]?.entry_time?.slice(0, 10) || '--') : '--',
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
  avgHold: '--', medianHold: '--',
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
/* ROLLING METRICS + MARKOV COMPUTATION (client-side from trades)            */
/* ========================================================================= */

function computeRollingMetric(
  trades: { pnlR: number }[],
  window: number,
  metric: string,
): number[] {
  if (trades.length < window) return [];
  const result: number[] = [];
  for (let i = window - 1; i < trades.length; i++) {
    const slice = trades.slice(i - window + 1, i + 1);
    const rs = slice.map(t => t.pnlR);
    const wins = rs.filter(r => r > 0).length;
    if (metric === 'Win Rate') {
      result.push(wins / window * 100);
    } else if (metric === 'PF') {
      const grossWin = rs.filter(r => r > 0).reduce((a, b) => a + b, 0);
      const grossLoss = Math.abs(rs.filter(r => r <= 0).reduce((a, b) => a + b, 0));
      result.push(grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? 10 : 0);
    } else if (metric === 'Sharpe') {
      const mean = rs.reduce((a, b) => a + b, 0) / window;
      const variance = rs.reduce((a, r) => a + (r - mean) ** 2, 0) / window;
      const std = Math.sqrt(variance);
      result.push(std > 0 ? mean / std : 0);
    }
  }
  return result;
}

function computeMarkov(trades: { pnlR: number }[]): {
  ww: number; wl: number; lw: number; ll: number;
  trendScore: number; edgeStrength: number;
} {
  let ww = 0, wl = 0, lw = 0, ll = 0;
  for (let i = 1; i < trades.length; i++) {
    const prev = trades[i - 1].pnlR > 0;
    const curr = trades[i].pnlR > 0;
    if (prev && curr) ww++;
    else if (prev && !curr) wl++;
    else if (!prev && curr) lw++;
    else ll++;
  }
  const wTotal = ww + wl || 1;
  const lTotal = lw + ll || 1;
  const pWW = ww / wTotal;
  const pLW = lw / lTotal;
  // Trend score: how much wins cluster (>0.5 = streaky wins, <0.5 = mean-reverting)
  const trendScore = (pWW + (1 - pLW)) / 2;
  // Edge strength: deviation from random walk
  const edgeStrength = Math.abs(pWW - 0.5) + Math.abs(pLW - 0.5);
  return {
    ww: Math.round(pWW * 100),
    wl: Math.round((1 - pWW) * 100),
    lw: Math.round(pLW * 100),
    ll: Math.round((1 - pLW) * 100),
    trendScore: Math.round(trendScore * 100) / 100,
    edgeStrength: Math.round(edgeStrength * 100) / 100,
  };
}

/** Inline rolling line chart (SVG) */
function RollingLineChart({ data, label, height = 250 }: { data: number[]; label: string; height?: number }) {
  if (data.length < 2) {
    return <div className="flex items-center justify-center" style={{ height, color: 'var(--text-muted)' }}><span className="text-xs">Not enough trades for rolling computation</span></div>;
  }
  const width = 700;
  const pad = { top: 20, right: 16, bottom: 28, left: 56 };
  const cW = width - pad.left - pad.right;
  const cH = height - pad.top - pad.bottom;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => ({
    x: pad.left + (i / (data.length - 1)) * cW,
    y: pad.top + cH - ((v - min) / range) * cH,
  }));
  const lineD = `M${pts.map(p => `${p.x},${p.y}`).join(' L')}`;
  const last = pts[pts.length - 1];
  const lastVal = data[data.length - 1];
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      {/* Grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
        const y = pad.top + cH * (1 - pct);
        const val = min + range * pct;
        return (
          <g key={i}>
            <line x1={pad.left} y1={y} x2={pad.left + cW} y2={y} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4 4" />
            <text x={pad.left - 8} y={y + 3} textAnchor="end" fontSize="9" fill="var(--text-muted)" fontFamily="monospace">{val.toFixed(1)}</text>
          </g>
        );
      })}
      <path d={lineD} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
      <circle cx={last.x} cy={last.y} r="4" fill="var(--accent)" />
      <text x={last.x} y={last.y - 10} textAnchor="middle" fontSize="10" fill="var(--accent)" fontWeight="600" fontFamily="monospace">{lastVal.toFixed(1)}</text>
    </svg>
  );
}

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
  if (isNaN(d.getTime())) return 0;
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

// M8.5: small re-rendering pill that shows "Live ·" when Ralph has pushed
// a bar recently, "Not live" otherwise. A 1Hz timer keeps the staleness
// calculation honest without coupling to the chart's render cadence.
function LiveBarStatusPill({
  liveBar, tfSeconds,
}: {
  liveBar: { receivedAt: number; isForming: boolean } | null;
  tfSeconds: number;
}) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  // "Live" window = 5 bar durations, clamped to [15s, 5min].
  const windowMs = Math.max(15_000, Math.min(300_000, tfSeconds * 5 * 1000));
  const isLive = liveBar != null && (now - liveBar.receivedAt) < windowMs;
  return (
    <span
      className="text-[10px] px-2 py-0.5 rounded-full"
      style={{
        background: isLive ? 'var(--green-muted, rgba(76,175,80,0.15))' : 'var(--bg-input)',
        color: isLive ? 'var(--green)' : 'var(--text-muted)',
        border: `1px solid ${isLive ? 'var(--green)' : 'var(--border)'}`,
      }}
      title={
        liveBar
          ? `Last bar ${Math.floor((now - liveBar.receivedAt) / 1000)}s ago`
          : 'Waiting for Ralph broadcast…'
      }
    >
      {isLive ? '● Live' : '○ Not live'}
    </span>
  );
}

/** Safe date → milliseconds. Returns 0 for invalid/missing dates. Logs warnings for debugging. */
function safeDateMs(val: string | null | undefined): number {
  if (!val || val === '--') return 0;
  const d = new Date(val);
  if (isNaN(d.getTime())) {
    console.warn('[StrategyDetail] Invalid date value:', val);
    return 0;
  }
  return d.getTime();
}

function parseExecTag(entry: string): { exec: string; rest: string } {
  const match = entry.match(/^\[([A-Z]+)\]\s*(.*)/);
  if (match) return { exec: `[${match[1]}]`, rest: match[2] };
  return { exec: '', rest: entry };
}

/**
 * Trigger naming convention:
 * - Long name: "Pack (Variation) > Trigger" — used in summary bar, configuration tab
 * - Short name: "Trigger" only — for compact card displays
 * parsePack().pack gives the pack portion, parsePack().trigger gives the short name.
 */
function parsePack(text: string): { pack: string; trigger: string } {
  const match = text.match(/^(.+?)\s*>\s*(.+)$/);
  if (match) return { pack: match[1].trim(), trigger: match[2].trim() };
  return { pack: '', trigger: text };
}

/** Format stop_config into a readable display string. Matches risk_management_packs.py stop_summary templates. */
function formatStopDisplay(stopConfig: any): string {
  if (!stopConfig?.method) return '--';
  const m = stopConfig.method;
  let base = m;
  if (m === 'atr') base = `ATR x${stopConfig.atr_mult ?? 1.5}`;
  else if (m === 'fixed_dollar') base = `$${stopConfig.dollar_amount ?? 1}`;
  else if (m === 'percentage') base = `${stopConfig.percentage ?? 0.5}%`;
  else if (m === 'swing') base = `Swing (${stopConfig.lookback ?? 5} bars, $${stopConfig.padding ?? 0} pad)`;
  if (stopConfig.trailing?.enabled) base += ` → Trail x${stopConfig.trailing.atr_mult ?? 1}`;
  if (stopConfig.breakeven?.enabled) base += ` → BE at ${stopConfig.breakeven.activation_r ?? 1}R`;
  return base;
}

/** Format target_config into a readable display string. Matches risk_management_packs.py target_summary templates. */
function formatTargetDisplay(targetConfig: any): string {
  if (!targetConfig?.method) return 'Signal exit only';
  const m = targetConfig.method;
  if (m === 'atr') return `ATR x${targetConfig.atr_mult ?? 3}`;
  if (m === 'fixed_dollar') return `$${targetConfig.dollar_amount ?? 2}`;
  if (m === 'percentage') return `${targetConfig.percentage ?? 1}%`;
  if (m === 'risk_reward') return `${targetConfig.rr_ratio ?? 2}R`;
  return m;
}

function formatTimeExitDisplay(config: any): string {
  if (!config?.method) return 'None';
  const m = config.method;
  if (m === 'eod_exit') return `EOD ${config.minutes_before_close ?? 15}min`;
  if (m === 'time_of_day_exit') return `Exit at ${config.exit_hour ?? 15}:${String(config.exit_minute ?? 50).padStart(2, '0')}`;
  if (m === 'max_hold_bars') return `Max ${config.max_bars ?? 4} bars`;
  if (m === 'session_exit') return `Window ${config.start_hour ?? 9}:${String(config.start_minute ?? 30).padStart(2, '0')}-${config.end_hour ?? 16}:${String(config.end_minute ?? 0).padStart(2, '0')}`;
  return m;
}

/* ========================================================================= */
/* PULSE CSS                                                                   */
/* ========================================================================= */

/** Format hold time seconds into human-readable string. */
function formatHoldTime(seconds: number | null | undefined, bars: number | null | undefined): string {
  if (seconds != null && seconds > 0) {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
  }
  if (bars != null && bars > 0) return `${bars} bars`;
  return '--';
}

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
  const execColor = useDisplayStore((s) => s.execTypeColor) || EXEC_BADGE_COLOR;
  const shape = useDisplayStore((s) => s.badgeShape);
  const brackets = useDisplayStore((s) => s.showBrackets);
  // exec comes as "[C]" from API — strip brackets and re-apply based on setting
  const raw = exec.replace(/[[\]]/g, '');
  return (
    <span
      className={`text-xs font-mono font-semibold px-2 py-0.5 ${shape === 'square' ? 'rounded' : 'rounded-full'}`}
      style={{
        color: execColor,
        background: execColor + '20',
      }}
    >
      {brackets ? `[${raw}]` : raw}
    </span>
  );
}

function FidelityBadge({ label }: { label: string }) {
  const fidColor = useDisplayStore((s) => s.fidelityColor) || FIDELITY_BADGE_COLOR;
  const shape = useDisplayStore((s) => s.badgeShape);
  const brackets = useDisplayStore((s) => s.showBrackets);
  // label comes as "[PB]" — strip brackets and re-apply based on setting
  const raw = label.replace(/[[\]]/g, '');
  return (
    <span
      className={`text-xs font-mono font-semibold px-2 py-0.5 ${shape === 'square' ? 'rounded' : 'rounded-full'}`}
      style={{
        color: fidColor,
        background: fidColor + '20',
      }}
    >
      {brackets !== false ? `[${raw}]` : raw}
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
  const [dateRange, setDateRange] = useState('Strategy Default');
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');
  // Always load full data — date range filtering happens client-side for instant response
  const { data: apiStrategy, isLoading, error } = useStrategy(strategyId);
  const { data: trades, isLoading: tradesLoading } = useStrategyTrades(strategyId);
  // M8.5 B+: always fetch the forward/backtest split. The endpoint is cheap
  // (just splits stored_trades at forward_test_start — no Polygon round-trip).
  // The `fwdRequested` gate that used to exist referred to a different
  // full-recompute path. Without this, btTrades falls through to all-trades
  // and fwdTrades stays empty → status line shows "FWD 0" even when trades
  // exist past the boundary.
  const [fwdRequested, setFwdRequested] = useState(true);
  const { data: fwdData, isLoading: fwdLoading } = useStrategyForwardTest(strategyId);
  const { data: kpiData, isLoading: kpisLoading } = useStrategyKPIs(strategyId);
  const { data: alerts } = useStrategyAlerts(strategyId);
  const { data: triggerAnalysis } = useTriggerAnalysis(strategyId);
  const deleteMut = useDeleteStrategy();
  const dupMut = useDuplicateStrategy();
  const refreshMut = useRefreshStrategy();
  const chartPrefs = useChartPrefs();

  // Price chart data — fast OHLCV from bars endpoint, slow indicators from chart-data
  const stratSymbol = apiStrategy?.symbol ?? null;
  const stratTimeframe = apiStrategy?.timeframe ?? '1Min';
  // Timeframe duration in milliseconds — used to shift C-type timestamps to next bar open
  const tfMs = useMemo(() => {
    const tf = stratTimeframe;
    if (tf.includes('Min')) return parseInt(tf) * 60 * 1000;
    if (tf.includes('Hour') || tf === '1H') return 3600 * 1000;
    if (tf.includes('Day') || tf === '1D') return 86400 * 1000;
    return 60 * 1000;
  }, [stratTimeframe]);
  // M8.5: seconds form for useLiveBar channel filter. Parses strings like
  // "10Sec", "1Min", "5Min", "1Hour", "1Day" — same convention Ralph uses
  // on the backend (TIMEFRAME_SECONDS dict).
  const tfSeconds = useMemo(() => {
    const tf = stratTimeframe;
    const n = parseInt(tf) || 1;
    if (tf.includes('Sec')) return n;
    if (tf.includes('Min')) return n * 60;
    if (tf.includes('Hour') || tf === '1H') return n * 3600;
    if (tf.includes('Day') || tf === '1D') return n * 86400;
    return 60;
  }, [stratTimeframe]);
  // M8.5: subscribe to Ralph's Supabase Realtime broadcasts for this (symbol, tf).
  // Returns null until the first bar arrives; stays updated on each new broadcast.
  const liveBar = useLiveBar(stratSymbol, tfSeconds);
  // Memoize the formingBar prop so SyncedChartPane (now React.memo'd) only
  // re-renders when liveBar's underlying values genuinely change — NOT on
  // every parent re-render. Reference changes would bust the memo even when
  // the bar snapshot is identical.
  const formingBarProp = useMemo(() => (
    liveBar?.bar ? {
      time: liveBar.bar.timestamp,
      open: liveBar.bar.open,
      high: liveBar.bar.high,
      low: liveBar.bar.low,
      close: liveBar.bar.close,
    } : null
  ), [
    liveBar?.bar?.timestamp, liveBar?.bar?.open, liveBar?.bar?.high,
    liveBar?.bar?.low, liveBar?.bar?.close,
  ]);
  // Pass-through of Ralph's tentative indicator/state payload. Refs are
  // allowed to change on every broadcast — that's the whole point of
  // intra-bar live updates. SyncedChartPane's forming effect re-runs
  // and applies series.update() to each overlay/oscillator/CB-heatmap cell.
  const formingIndicators = liveBar?.indicators ?? null;
  const formingStates = liveBar?.states ?? null;
  const formingStateCrossTf = liveBar?.stateCrossTf ?? null;
  const { data: barsData } = useBars(stratSymbol, stratTimeframe, apiStrategy?.data_days ?? 30);
  const { data: chartDataResp, isLoading: chartDataLoading } = useStrategyChartData(strategyId);

  // Gap detection: useStrategyChartData has a 5-min staleTime and no
  // refetch, so historical bars freeze at page-load time. When the Ralph
  // worker restarts mid-session (or the page is simply left open past
  // staleTime) the forming bar arrives minutes ahead of the last historical
  // bar, leaving a visible void on the chart. When the gap exceeds 2 TF
  // durations, invalidate the query so the backend re-fills the missing
  // bars. A 30-second cooldown prevents thrashing while the refetch is in
  // flight or while the backend is still a tick behind.
  const queryClient = useQueryClient();
  const lastGapRefetchRef = useRef<number>(0);
  useEffect(() => {
    if (!liveBar?.bar?.timestamp || !chartDataResp?.chart_data?.length || !tfSeconds) return;
    const lastHist = chartDataResp.chart_data[chartDataResp.chart_data.length - 1];
    const lastHistMs = new Date(lastHist.timestamp).getTime();
    const liveMs = new Date(liveBar.bar.timestamp).getTime();
    if (!isFinite(lastHistMs) || !isFinite(liveMs)) return;
    const gapMs = liveMs - lastHistMs;
    if (gapMs <= tfSeconds * 1000 * 2) return;
    const now = Date.now();
    if (now - lastGapRefetchRef.current < 30_000) return;
    lastGapRefetchRef.current = now;
    queryClient.invalidateQueries({ queryKey: ['strategy-chart-data', strategyId] });
  }, [liveBar?.bar?.timestamp, chartDataResp, tfSeconds, strategyId, queryClient]);

  // ---- Client-side date range filtering (must be before strategy/KPI derivation) ----
  const isDateFiltered = dateRange && dateRange !== 'Strategy Default' && dateRange !== 'All Data';

  const filteredStoredTrades = useMemo(() => {
    const raw = apiStrategy?.stored_trades || [];
    if (!raw.length || !isDateFiltered) return raw;

    const now = Date.now();
    const fwdStart = apiStrategy?.forward_test_start;
    const fwdMs = fwdStart ? safeDateMs(fwdStart) : 0;

    return raw.filter((t: any) => {
      const ms = safeDateMs(t.entry_time);
      if (!ms) return true;
      if (dateRange === 'Last 7 Days') return ms >= now - 7 * 86400000;
      if (dateRange === 'Last 30 Days') return ms >= now - 30 * 86400000;
      if (dateRange === 'Last 90 Days') return ms >= now - 90 * 86400000;
      if (dateRange === 'Backtest Only') return fwdMs ? ms < fwdMs : true;
      if (dateRange === 'Forward Only') return fwdMs ? ms >= fwdMs : false;
      if (dateRange === 'Custom') {
        const startMs = customStart ? new Date(customStart + 'T00:00:00').getTime() : 0;
        const endMs = customEnd ? new Date(customEnd + 'T23:59:59').getTime() : Infinity;
        return ms >= startMs && ms <= endMs;
      }
      return true;
    });
  }, [apiStrategy, dateRange, isDateFiltered, customStart, customEnd]);

  // Date range text: show actual date range of currently visible trades
  const dateRangeText = useMemo(() => {
    const trades = filteredStoredTrades;
    if (!trades.length) return '';
    const times = trades.map((t: any) => safeDateMs(t.entry_time)).filter((ms: number) => ms > 0);
    if (times.length < 1) return '';
    const fmt = (ms: number) => new Date(ms).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    return `${fmt(Math.min(...times))} — ${fmt(Math.max(...times))}`;
  }, [filteredStoredTrades]);

  const clientKPIs = useMemo(() => {
    const trades = filteredStoredTrades;
    if (!trades.length || !isDateFiltered) return null;
    const rValues = trades.map((t: any) => t.r_multiple ?? 0);
    const wins = rValues.filter((r: number) => r > 0);
    const losses = rValues.filter((r: number) => r <= 0);
    const totalR = rValues.reduce((s: number, r: number) => s + r, 0);
    const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0;
    const grossWin = wins.reduce((s: number, r: number) => s + r, 0);
    const grossLoss = Math.abs(losses.reduce((s: number, r: number) => s + r, 0));
    const pf = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0;
    const avgR = trades.length > 0 ? totalR / trades.length : 0;
    const times = trades.map((t: any) => safeDateMs(t.exit_time || t.entry_time)).filter((ms: number) => ms > 0);
    const daySpan = times.length >= 2 ? Math.max(1, Math.ceil((Math.max(...times) - Math.min(...times)) / 86400000)) : 1;
    const dailyR = totalR / daySpan;
    let peak = 0, maxDD = 0, cum = 0;
    for (const r of rValues) { cum += r; if (cum > peak) peak = cum; const dd = cum - peak; if (dd < maxDD) maxDD = dd; }
    const avgWin = wins.length > 0 ? grossWin / wins.length : 0;
    const avgLoss = losses.length > 0 ? grossLoss / losses.length : 0;
    const payoff = avgLoss > 0 ? avgWin / avgLoss : 0;
    const bestTrade = rValues.length > 0 ? Math.max(...rValues) : 0;
    const worstTrade = rValues.length > 0 ? Math.min(...rValues) : 0;
    let rSquared = 0;
    if (rValues.length >= 3) {
      const cumR = rValues.reduce((acc: number[], r: number, i: number) => { acc.push((acc[i - 1] ?? 0) + r); return acc; }, [] as number[]);
      const n = cumR.length; const xMean = (n - 1) / 2;
      const yMean = cumR.reduce((s: number, v: number) => s + v, 0) / n;
      const slope = cumR.reduce((s: number, y: number, i: number) => s + (i - xMean) * (y - yMean), 0) / cumR.reduce((s: number, _: number, i: number) => s + (i - xMean) ** 2, 0);
      const intercept = yMean - slope * xMean;
      let ssReg = 0, ssTot = 0;
      for (let i = 0; i < n; i++) { ssReg += (cumR[i] - (slope * i + intercept)) ** 2; ssTot += (cumR[i] - yMean) ** 2; }
      rSquared = ssTot > 0 ? 1 - ssReg / ssTot : 0;
    }
    return {
      primary: { win_rate: winRate, profit_factor: isFinite(pf) ? pf : 0, daily_r: dailyR, total_trades: trades.length, max_r_drawdown: maxDD, avg_r: avgR },
      extended: { wins: wins.length, losses: losses.length, bestTrade, worstTrade, avgWin, avgLoss: -avgLoss, payoffRatio: payoff, expectedDailyR: dailyR * trades.length / daySpan, rSquared },
    };
  }, [filteredStoredTrades, isDateFiltered]);

  // Map API data to V5 shape (client KPIs override when date range is active)
  const strategyRaw = apiStrategy ? apiToDetailStrategy(apiStrategy) : null;
  const strategy = useMemo(() => {
    if (!strategyRaw) return null;
    if (!isDateFiltered || !clientKPIs) return strategyRaw;
    const k = clientKPIs.primary;
    return {
      ...strategyRaw,
      winRate: k.win_rate,
      pf: k.profit_factor,
      dailyR: k.daily_r,
      trades: k.total_trades,
      maxDD: k.max_r_drawdown,
    };
  }, [strategyRaw, isDateFiltered, clientKPIs]);

  // Compute hold time stats from stored trades
  const holdTimeStats = useMemo(() => {
    const raw = isDateFiltered ? filteredStoredTrades : (apiStrategy?.stored_trades || []);
    const holdSeconds = raw
      .map((t: any) => t.hold_time_seconds)
      .filter((s: any) => s != null && s > 0) as number[];
    if (holdSeconds.length === 0) {
      const holdBars = raw.map((t: any) => t.bars_held).filter((b: any) => b != null && b > 0) as number[];
      if (holdBars.length === 0) return { avgHold: '--', medianHold: '--' };
      const avg = holdBars.reduce((s, b) => s + b, 0) / holdBars.length;
      const sorted = [...holdBars].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      return { avgHold: `${avg.toFixed(1)} bars`, medianHold: `${median} bars` };
    }
    const avg = holdSeconds.reduce((s, v) => s + v, 0) / holdSeconds.length;
    const sorted = [...holdSeconds].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    return { avgHold: formatHoldTime(avg, null), medianHold: formatHoldTime(median, null) };
  }, [apiStrategy, filteredStoredTrades, isDateFiltered]);

  const extendedKPIs = useMemo(() => {
    // When date range is active, use client-computed extended KPIs
    if (isDateFiltered && clientKPIs) {
      const e = clientKPIs.extended;
      return {
        ...EMPTY_EXTENDED_KPIS,
        wins: e.wins, losses: e.losses, bestTrade: e.bestTrade, worstTrade: e.worstTrade,
        avgWin: e.avgWin, avgLoss: e.avgLoss, payoffRatio: e.payoffRatio,
        expectedDailyR: e.expectedDailyR, rSquared: e.rSquared,
        avgHold: holdTimeStats.avgHold, medianHold: holdTimeStats.medianHold,
      };
    }
    const s = kpiData?.secondary_kpis;
    if (!s) return EMPTY_EXTENDED_KPIS;
    return {
      wins: s.win_count ?? s.wins ?? 0,
      losses: s.loss_count ?? s.losses ?? 0,
      bestTrade: s.best_trade_r ?? s.bestTrade ?? 0,
      worstTrade: s.worst_trade_r ?? s.worstTrade ?? 0,
      avgWin: s.avg_win_r ?? s.avgWin ?? 0,
      avgLoss: s.avg_loss_r ?? s.avgLoss ?? 0,
      payoffRatio: s.payoff_ratio ?? s.payoffRatio ?? 0,
      expectedDailyR: s.expected_daily ?? s.expectedDailyR ?? 0,
      sharpe: s.sharpe_ratio ?? s.sharpe ?? 0,
      sortino: s.sortino_ratio ?? s.sortino ?? 0,
      calmar: s.calmar_ratio ?? s.calmar ?? 0,
      kelly: s.kelly_criterion ?? s.kelly ?? 0,
      dailyVaR: s.daily_var_95 ?? s.dailyVaR ?? 0,
      cvar: s.cvar_95 ?? s.cvar ?? 0,
      volatility: s.volatility ?? 0,
      rSquared: s.r_squared ?? s.rSquared ?? 0,
      skewness: s.skewness ?? 0,
      kurtosis: s.kurtosis ?? 0,
      tailRatio: s.tail_ratio ?? s.tailRatio ?? 0,
      outlierWinPct: s.outlier_win_ratio ?? s.outlierWinPct ?? 0,
      outlierLossPct: s.outlier_loss_ratio ?? s.outlierLossPct ?? 0,
      maxRDD: s.max_r_drawdown ?? s.maxRDD ?? 0,
      recoveryFactor: s.recovery_factor ?? s.recoveryFactor ?? 0,
      ulcerIndex: s.ulcer_index ?? s.ulcerIndex ?? 0,
      serenityIndex: s.serenity_index ?? s.serenityIndex ?? 0,
      longestDDTrades: s.longest_dd_trades ?? s.longestDDTrades ?? 0,
      longestDDDays: s.longest_dd_days ?? s.longestDDDays ?? 0,
      maxConsecWins: s.max_consec_wins ?? s.maxConsecWins ?? 0,
      maxConsecLosses: s.max_consec_losses ?? s.maxConsecLosses ?? 0,
      avgHold: holdTimeStats.avgHold, medianHold: holdTimeStats.medianHold,
    };
  }, [kpiData, isDateFiltered, clientKPIs, holdTimeStats]);
  // Map API trades (snake_case) to V5 format — useMemo prevents Terser const-chaining TDZ
  const allTrades = useMemo(() => (trades || EMPTY_TRADES).map((t: any, i: number) => ({
    id: t.id ?? i + 1,
    entryTime: t.entry_time || t.entryTime || '--',
    exitTime: t.exit_time || t.exitTime || '--',
    entryPrice: t.entry_price ?? t.entryPrice ?? 0,
    exitPrice: t.exit_price ?? t.exitPrice ?? 0,
    pnlR: t.r_multiple ?? t.pnlR ?? 0,
    execType: t.exec_type ? `[${t.exec_type}]` : (t.execType || '[C]'),
    exitReason: t.exit_reason || t.exitReason || '--',
    holdTime: formatHoldTime(t.hold_time_seconds, t.bars_held),
    isFwd: t.isFwd ?? false,
  })), [trades]);

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
  const [kpiMode, setKpiMode] = useState('Overall');
  const [extKpiTab, setExtKpiTab] = useState('Performance');
  const [showExtKpis, setShowExtKpis] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [eqShowHWM, setEqShowHWM] = useState(false);
  const [eqShowEdge, setEqShowEdge] = useState(false);
  const [eqShowConf, setEqShowConf] = useState(false);
  const [eqXAxisLocal, setEqXAxisLocal] = useState<'trade' | 'time' | null>(null);
  const [pvpViewMode, setPvpViewMode] = useState<'forward' | 'alerts'>('forward');
  const [zoomTrade, setZoomTrade] = useState<{ idx: number; side: 'entry' | 'exit'; trade: any; alertMatch?: any } | null>(null);
  const zoomQuery = useTradeZoom(
    zoomTrade ? strategyId : null,
    zoomTrade ? zoomTrade.idx : null,
    zoomTrade?.side ?? 'exit',
  );
  const [advancedTab, setAdvancedTab] = useState('Rolling Metrics');
  const [rollingWindow, setRollingWindow] = useState(20);
  const [rollingMetric, setRollingMetric] = useState('Win Rate');
  const [returnView, setReturnView] = useState('Histogram');
  const [markovWindow, setMarkovWindow] = useState(20);
  const [edgeDecay, setEdgeDecay] = useState(0.5);
  const [candleCount, setCandleCount] = useState(200);
  const [manualExitLoading, setManualExitLoading] = useState(false);
  const [manualExitResult, setManualExitResult] = useState<{ ok: boolean; msg: string } | null>(null);

  const handleManualExit = useCallback(async () => {
    if (!strategyId || manualExitLoading) return;
    if (!window.confirm('Are you sure you want to manually exit this position? This will fire exit webhooks to all linked portfolios.')) return;
    setManualExitLoading(true);
    setManualExitResult(null);
    try {
      const token = localStorage.getItem('ror_access_token');
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/api/strategies/${strategyId}/manual-exit`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      });
      const data = await res.json();
      if (res.ok) {
        setManualExitResult({ ok: true, msg: `Exit sent — ${data.webhooks_delivered} webhook(s) delivered${data.price ? ` @ $${Number(data.price).toFixed(2)}` : ''}` });
      } else {
        setManualExitResult({ ok: false, msg: data.detail || 'Failed to send exit' });
      }
    } catch (e: any) {
      setManualExitResult({ ok: false, msg: e.message || 'Network error' });
    } finally {
      setManualExitLoading(false);
    }
  }, [strategyId, manualExitLoading]);
  const [showConditions, setShowConditions] = useState(true);
  const [showTriggers, setShowTriggers] = useState(true);
  const [btTradesOpen, setBtTradesOpen] = useState(false);
  const [fwdTradesOpen, setFwdTradesOpen] = useState(false);
  // M8.5 B+: expand beyond 100-row cap for algo history (off by default to
  // keep the DOM light; user opts in and accepts the render cost).
  const [showAllAlgoHistory, setShowAllAlgoHistory] = useState(false);
  const confluenceGroups = triggerAnalysis?.confluence_groups ?? EMPTY_CONFLUENCE_GROUPS;
  const confluenceTimeline = EMPTY_CONFLUENCE_TIMELINE; // State timeline requires backtest instrumentation
  const confluenceTriggerEvents = EMPTY_CONFLUENCE_TRIGGER_EVENTS; // Trigger events require backtest instrumentation
  // Map raw alerts to event-level format (for Alerts tab)
  // Map raw alerts — handle both flattened (from _row_to_alert) and nested (data JSONB) formats
  const recentAlertEvents = useMemo(() => (alerts || EMPTY_ALERTS).map((a: any) => {
    const d = a.data || {};
    return {
      time: a.timestamp || a.time || '--',
      type: (a.type || '').toLowerCase().includes('entry') ? 'ENTRY' : 'EXIT',
      trigger: a.trigger || d.trigger || '--',
      price: a.price ?? d.price ?? null,
      stopPrice: a.stop_price ?? d.stop_price ?? a.entry_stop_price ?? d.entry_stop_price ?? null,
      entryPrice: a.entry_price ?? d.entry_price ?? null,
      status: a.webhook_sent ? 'Delivered' : a.acknowledged ? 'Acknowledged' : 'Pending',
    };
  }), [alerts]);

  // Pair entry/exit alerts into trade rows (for Alert History table on Chart & Trades tab)
  const recentAlerts = useMemo(() => {
    const entries: any[] = [];
    const paired: any[] = [];
    const sorted = [...recentAlertEvents].sort((a, b) => safeDateMs(a.time) - safeDateMs(b.time));
    for (const evt of sorted) {
      if (evt.type === 'ENTRY') {
        entries.push(evt);
      } else if (evt.type === 'EXIT' && entries.length > 0) {
        const entry = entries.shift();
        const entryP = entry?.price ?? 0;
        const exitP = evt.price ?? 0;
        const stopP = entry?.stopPrice;
        let rMult: number | null = null;
        if (stopP != null && stopP !== 0 && entryP && exitP) {
          rMult = (exitP - entryP) / Math.abs(entryP - stopP);
        }
        paired.push({
          entryTime: entry?.time || '--', exitTime: evt.time || '--',
          entryPrice: entryP, exitPrice: exitP,
          r: rMult != null ? Math.round(rMult * 100) / 100 : null,
          result: rMult != null ? (rMult >= 0 ? 'Win' : 'Loss') : exitP > entryP ? 'Win' : exitP < entryP ? 'Loss' : '--',
          exitReason: evt.trigger || '--',
        });
      }
    }
    for (const entry of entries) {
      paired.push({ entryTime: entry.time || '--', exitTime: null, entryPrice: entry.price, exitPrice: null, r: null, result: 'Open', exitReason: null });
    }
    return paired.reverse();
  }, [recentAlertEvents]);
  // Alert analysis computed lazily — avoid complex useMemo chains that cause TDZ in production
  const alertAnalysis = EMPTY_ALERT_ANALYSIS;
  const [selectedConfGroup, setSelectedConfGroup] = useState('');
  const [selectedCondition, setSelectedCondition] = useState<string | null>(null);
  useEffect(() => {
    if (confluenceGroups.length > 0 && !selectedConfGroup) {
      setSelectedConfGroup(confluenceGroups[0].name);
    }
  }, [confluenceGroups, selectedConfGroup]);
  // Auto-select first confluence condition for Confluence Analysis tab
  const strategyConfluence: string[] = apiStrategy?.confluence || [];
  useEffect(() => {
    if (strategyConfluence.length > 0 && !selectedCondition) {
      setSelectedCondition(strategyConfluence[0]);
    }
  }, [strategyConfluence, selectedCondition]);
  const { data: confChartData, isLoading: confChartLoading } = useConfluenceChart(strategyId, selectedCondition);
  // Confluence Analysis panes — memoized so the parent re-rendering (e.g. a
  // live-bar tick on the adjacent Chart & Trades state) doesn't produce a
  // fresh panes ref and retrigger SyncedChartPane's structure effect. That
  // effect saves/restores the visible logical range; retriggered with a
  // stale range on a small dataset it squishes bars into the right edge.
  const confPanesData = useMemo(() => {
    if (!confChartData || confChartData.bars.length === 0) return null;
    const bars = confChartData.bars;

    const stateRanges: any[] = [];
    let regionStart: number | null = null;
    let regionMet = false;
    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const met = bar._met === true || (bar._state != null && bar._state === confChartData.needed_state);
      const t = Math.floor(safeDateMs(bar.timestamp) / 1000);
      if (regionStart === null) { regionStart = t; regionMet = met; }
      else if (met !== regionMet) {
        stateRanges.push({ startTime: regionStart, endTime: t, color: regionMet ? 'rgba(76,175,80,0.15)' : 'rgba(244,67,54,0.10)' });
        regionStart = t; regionMet = met;
      }
    }
    if (regionStart !== null) {
      const lastT = Math.floor(safeDateMs(bars[bars.length - 1].timestamp) / 1000);
      stateRanges.push({ startTime: regionStart, endTime: lastT, color: regionMet ? 'rgba(76,175,80,0.15)' : 'rgba(244,67,54,0.10)' });
    }
    const primitives: any[] = stateRanges.length > 0
      ? [{ type: 'sessionHighlighting' as const, seriesIndex: 0, options: { ranges: stateRanges } }]
      : [];

    const confOverlays: string[] = (confChartData as any).overlay_indicators || [];
    const confOscillators: string[] = (confChartData as any).oscillator_indicators || [];
    const COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63'];

    const panes: PaneConfig[] = [];
    panes.push({
      id: 'conf-price',
      height: 300,
      series: [
        {
          type: 'Candlestick',
          data: bars.map((b: any) => ({ time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close })),
        },
        ...confOverlays.map((col: string, i: number) => ({
          type: 'Line' as const,
          data: bars.filter((b: any) => b[col] != null).map((b: any) => ({ time: b.timestamp, value: b[col] })),
          options: { color: COLORS[i % COLORS.length], lineWidth: 2, title: col.replace(/_/g, ' ') },
        })),
      ],
      primitives,
    });

    if (confOscillators.length > 0) {
      const oscSeries: SeriesConfig[] = [];
      for (const col of confOscillators.filter(c => c.includes('hist'))) {
        oscSeries.push({
          type: 'Histogram',
          data: bars.filter((b: any) => b[col] != null).map((b: any) => ({
            time: b.timestamp, value: b[col], color: b[col] >= 0 ? '#4CAF50' : '#f44336',
          })),
          options: { priceLineVisible: false, title: 'Hist' },
        });
      }
      for (const col of confOscillators.filter(c => !c.includes('hist'))) {
        oscSeries.push({
          type: 'Line',
          data: bars.filter((b: any) => b[col] != null).map((b: any) => ({ time: b.timestamp, value: b[col] })),
          options: { color: col.includes('signal') ? '#FF9800' : '#2196F3', lineWidth: 1, priceLineVisible: false, title: col.replace(/_/g, ' ') },
        });
      }
      oscSeries.push({
        type: 'Line',
        data: [{ time: bars[0].timestamp, value: 0 }, { time: bars[bars.length - 1].timestamp, value: 0 }],
        options: { color: 'rgba(128,128,128,0.3)', lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false },
      });
      panes.push({ id: 'conf-oscillator', height: 150, series: oscSeries });
    }

    return panes;
  }, [confChartData]);
  const [showPosHealth, setShowPosHealth] = useState(false);
  const [showTriggerTiming, setShowTriggerTiming] = useState(false);
  const [showTradeByTrade, setShowTradeByTrade] = useState(false);

  // Forward test trades — wrapped in useMemo to prevent Terser from chaining
  // const declarations (which causes TDZ errors in production builds)
  // Shift a C-type timestamp to the next bar open (more accurate to reality)
  const shiftCType = useMemo(() => (iso: string, execType: string): string => {
    if (!iso || iso === '--') return iso;
    try {
      const isL = execType.includes('L') || execType.includes('HM') || execType.includes('HL');
      if (isL) return iso;
      const d = new Date(iso);
      if (isNaN(d.getTime())) {
        console.warn('[StrategyDetail] shiftCType: invalid date:', iso);
        return iso;
      }
      return new Date(d.getTime() + tfMs).toISOString();
    } catch (e) {
      console.warn('[StrategyDetail] shiftCType error:', iso, e);
      return iso;
    }
  }, [tfMs]);

  // Derive exit exec type from exit_reason:
  // L-type: stop_loss, target, unconfirmed_hl, unconfirmed_hm (price crosses a level)
  // C-type: bar_count_exit, opposite_signal, time_exit, signal (evaluated at bar close)
  const L_TYPE_EXITS = new Set(['stop_loss', 'stop', 'target', 'unconfirmed_hl']);
  const exitExecTypeOf = (reason: string) => L_TYPE_EXITS.has(reason) ? 'L' : 'C';

  const fwdTrades = useMemo(() => (fwdData?.forward_trades || []).map((t: any, i: number) => {
    const rawExec = t.exec_type || 'C';
    const exitReason = t.exit_reason || '--';
    const exitExec = exitExecTypeOf(exitReason);
    return {
      id: i + 1,
      entryTime: t.entry_time || '--',
      exitTime: t.exit_time || '--',
      entryTimeDisplay: shiftCType(t.entry_time || '--', rawExec),
      exitTimeDisplay: shiftCType(t.exit_time || '--', exitExec),
      entryPrice: t.entry_price ?? 0,
      exitPrice: t.exit_price ?? 0,
      pnlR: t.r_multiple ?? 0,
      execType: rawExec,
      exitExecType: exitExec,
      exitReason,
      holdTime: formatHoldTime(t.hold_time_seconds, t.bars_held),
      isFwd: true,
    };
  }), [fwdData, shiftCType]);

  const btTrades = useMemo(() => (fwdData?.backtest_trades || allTrades).map((t: any, i: number) => {
    const rawExec = t.exec_type || t.execType?.replace(/[\[\]]/g, '') || 'C';
    const exitReason = t.exit_reason || t.exitReason || '--';
    const exitExec = exitExecTypeOf(exitReason);
    return {
      id: t.id ?? i + 1,
      entryTime: t.entry_time || t.entryTime || '--',
      exitTime: t.exit_time || t.exitTime || '--',
      entryTimeDisplay: shiftCType(t.entry_time || t.entryTime || '--', rawExec),
      exitTimeDisplay: shiftCType(t.exit_time || t.exitTime || '--', exitExec),
      entryPrice: t.entry_price ?? t.entryPrice ?? 0,
      exitPrice: t.exit_price ?? t.exitPrice ?? 0,
      pnlR: t.r_multiple ?? t.pnlR ?? 0,
      execType: rawExec,
      exitExecType: exitExec,
      exitReason,
      holdTime: formatHoldTime(t.hold_time_seconds ?? t.holdTimeSeconds, t.bars_held ?? t.barsHeld),
      isFwd: false,
    };
  }), [fwdData, allTrades, shiftCType]);

  // Trade-to-Alert mapping — MUST be after btTrades declaration
  const tradeAlertMapping = useMemo(() => {
    if (recentAlerts.length === 0) return [];
    return recentAlerts.filter((a: any) => a.entryTime && a.entryTime !== '--').map((alertTrade: any, i: number) => {
      const alertEntryMs = safeDateMs(alertTrade.entryTime);
      let closestBt: any = null;
      let closestDist = Infinity;
      for (const bt of btTrades) {
        if (!bt.entryTime || bt.entryTime === '--') continue;
        const dist = Math.abs(safeDateMs(bt.entryTime) - alertEntryMs);
        if (dist < closestDist) { closestDist = dist; closestBt = bt; }
      }
      const entryDelta = closestBt ? `${Math.round(closestDist / 1000)}s` : '--';
      const exitDelta = closestBt && alertTrade.exitTime && closestBt.exitTime !== '--'
        ? `${Math.round(Math.abs(safeDateMs(alertTrade.exitTime) - safeDateMs(closestBt.exitTime)) / 1000)}s` : '--';
      return {
        tradeNum: i + 1,
        btEntry: closestBt?.entryTime ? new Date(closestBt.entryTime).toLocaleString() : '--',
        alertEntry: new Date(alertTrade.entryTime).toLocaleString(),
        entryDelta,
        btExit: closestBt?.exitTime && closestBt.exitTime !== '--' ? new Date(closestBt.exitTime).toLocaleString() : '--',
        alertExit: alertTrade.exitTime ? new Date(alertTrade.exitTime).toLocaleString() : '--',
        exitDelta,
      };
    });
  }, [recentAlerts, btTrades]);

  // Match sets: which algo trades have matching alerts, and vice versa
  // Uses shifted (display) timestamps for C-type trades so deltas reflect real slippage
  // Match threshold = user's alertSlippage setting (not a fixed 10 minutes)
  const { alertMatches, algoMatches } = useMemo(() => {
    const algoAll = [...fwdTrades, ...btTrades];
    const slipMs = (chartPrefs.alertSlippage || 5) * 1000;
    // Search window: max of slippage tolerance or 2× timeframe (to find the closest match)
    const searchWindow = Math.max(slipMs, tfMs * 2);

    // For each alert, find closest algo trade by entry time (using shifted display time)
    const alertResults: { matched: boolean; entryDelta: number | null; exitDelta: number | null }[] = [];
    for (let ai = 0; ai < recentAlerts.length; ai++) {
      const a = recentAlerts[ai];
      if (!a.entryTime || a.entryTime === '--') {
        alertResults.push({ matched: false, entryDelta: null, exitDelta: null });
        continue;
      }
      const aEntryMs = safeDateMs(a.entryTime);
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let ti = 0; ti < algoAll.length; ti++) {
        const t = algoAll[ti];
        if (!t.entryTimeDisplay || t.entryTimeDisplay === '--') continue;
        const dist = Math.abs(safeDateMs(t.entryTimeDisplay) - aEntryMs);
        if (dist < bestDist) { bestDist = dist; bestIdx = ti; }
      }
      if (bestIdx >= 0 && bestDist <= searchWindow) {
        const algo = algoAll[bestIdx];
        // Delta from alert's perspective: negative = algo was earlier, positive = algo was later
        const entryDelta = (safeDateMs(algo.entryTimeDisplay) - aEntryMs) / 1000;
        let exitDelta: number | null = null;
        if (a.exitTime && a.exitTime !== '--' && algo.exitTimeDisplay && algo.exitTimeDisplay !== '--') {
          exitDelta = (safeDateMs(algo.exitTimeDisplay) - safeDateMs(a.exitTime)) / 1000;
        }
        alertResults.push({ matched: Math.abs(entryDelta) <= chartPrefs.alertSlippage, entryDelta, exitDelta });
      } else {
        alertResults.push({ matched: false, entryDelta: null, exitDelta: null });
      }
    }

    // For each algo trade, find closest alert by entry time
    const algoResults: { matched: boolean; entryDelta: number | null; exitDelta: number | null; alertEntryPrice?: number; alertExitPrice?: number }[] = [];
    for (let ti = 0; ti < algoAll.length; ti++) {
      const t = algoAll[ti];
      if (!t.entryTimeDisplay || t.entryTimeDisplay === '--') {
        algoResults.push({ matched: false, entryDelta: null, exitDelta: null });
        continue;
      }
      const tEntryMs = safeDateMs(t.entryTimeDisplay);
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let ai = 0; ai < recentAlerts.length; ai++) {
        const a = recentAlerts[ai];
        if (!a.entryTime || a.entryTime === '--') continue;
        const dist = Math.abs(safeDateMs(a.entryTime) - tEntryMs);
        if (dist < bestDist) { bestDist = dist; bestIdx = ai; }
      }
      if (bestIdx >= 0 && bestDist <= searchWindow) {
        const alert = recentAlerts[bestIdx];
        const entryDelta = (safeDateMs(alert.entryTime) - tEntryMs) / 1000;
        let exitDelta: number | null = null;
        if (alert.exitTime && alert.exitTime !== '--' && t.exitTimeDisplay && t.exitTimeDisplay !== '--') {
          exitDelta = (safeDateMs(alert.exitTime) - safeDateMs(t.exitTimeDisplay)) / 1000;
        }
        algoResults.push({
          matched: Math.abs(entryDelta) <= chartPrefs.alertSlippage,
          entryDelta, exitDelta,
          alertEntryPrice: alert.entryPrice ?? undefined,
          alertExitPrice: alert.exitPrice ?? undefined,
        });
      } else {
        algoResults.push({ matched: false, entryDelta: null, exitDelta: null });
      }
    }

    return { alertMatches: alertResults, algoMatches: algoResults };
  }, [recentAlerts, fwdTrades, btTrades, chartPrefs.alertSlippage, tfMs]);

  // Timezone-aware timestamp formatter
  const formatTime = useMemo(() => {
    // Map common aliases to IANA names
    const TZ_MAP: Record<string, string> = {
      'US/Eastern': 'America/New_York', 'US/Central': 'America/Chicago',
      'US/Mountain': 'America/Denver', 'US/Pacific': 'America/Los_Angeles',
      'US/Alaska': 'America/Anchorage', 'US/Hawaii': 'Pacific/Honolulu',
    };
    const rawTz = chartPrefs.timezone || 'US/Mountain';
    const tz = TZ_MAP[rawTz] || rawTz;
    return (iso: string | null | undefined) => {
      if (!iso || iso === '--') return '--';
      try {
        return new Date(iso).toLocaleString('en-US', {
          timeZone: tz, month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
        });
      } catch { return iso; }
    };
  }, [chartPrefs.timezone]);

  // Two-line time renderer for table cells: "Mar 27" on top, "13:54:00" below
  const renderTime = useMemo(() => {
    const TZ_MAP: Record<string, string> = {
      'US/Eastern': 'America/New_York', 'US/Central': 'America/Chicago',
      'US/Mountain': 'America/Denver', 'US/Pacific': 'America/Los_Angeles',
      'US/Alaska': 'America/Anchorage', 'US/Hawaii': 'Pacific/Honolulu',
    };
    const rawTz = chartPrefs.timezone || 'US/Mountain';
    const tz = TZ_MAP[rawTz] || rawTz;
    return (iso: string | null | undefined, badge?: string) => {
      if (!iso || iso === '--') return <span>--</span>;
      try {
        const d = new Date(iso);
        const datePart = d.toLocaleString('en-US', { timeZone: tz, month: 'short', day: 'numeric' });
        const timePart = d.toLocaleString('en-US', { timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
        const isL = badge === 'L';
        return (
          <span style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', lineHeight: 1.3 }}>
            <span>
              {badge && <span style={{ color: isL ? '#FF9800' : 'var(--accent)', fontSize: '0.6rem', fontWeight: 700, marginRight: 2 }}>{badge}</span>}
              {datePart}
            </span>
            <span style={{ fontSize: '0.7rem' }}>{timePart}</span>
          </span>
        );
      } catch { return <span>{iso}</span>; }
    };
  }, [chartPrefs.timezone]);

  // Build equity curve data — always from stored_trades for instant load (no Polygon dependency)
  const equityPoints = useMemo(() => {
    const raw = isDateFiltered ? filteredStoredTrades : (apiStrategy?.stored_trades || []);
    if (raw.length > 0) {
      let cum = 0;
      return raw.map((t: any, i: number) => {
        cum += (t.r_multiple ?? 0);
        return { trade_number: i + 1, cumulative_r: cum, timestamp: t.exit_time || t.entry_time || '--' };
      });
    }
    // Fallback: use stored equity_curve_data if no stored_trades
    const ecd = apiStrategy?.equity_curve_data;
    if (ecd?.cumulative_r?.length) {
      return ecd.cumulative_r.map((cr: number, i: number) => ({
        trade_number: i + 1,
        cumulative_r: cr,
        timestamp: ecd.exit_times?.[i] ?? undefined,
      }));
    }
    return [];
  }, [apiStrategy, filteredStoredTrades, isDateFiltered]);

  // Compute boundary index from forward_test_start date (MUST be before alertEquityPoints)
  const equityBoundaryIndex = useMemo(() => {
    const fwdStart = apiStrategy?.forward_test_start;
    if (!fwdStart) return null;
    const fwdMs = safeDateMs(fwdStart);
    if (!fwdMs) return null;
    const raw = isDateFiltered ? filteredStoredTrades : (apiStrategy?.stored_trades || []);
    for (let i = 0; i < raw.length; i++) {
      const ms = safeDateMs(raw[i].entry_time);
      if (ms && ms >= fwdMs) return i;
    }
    return null;
  }, [apiStrategy, filteredStoredTrades, isDateFiltered]);

  // Build alert equity points — offset by FWD cumulative R so the green line
  // starts at the same level as the FWD curve (overlays to show slippage/gaps)
  const alertEquityPoints = useMemo(() => {
    const closedAlerts = recentAlerts.filter((a: any) => a.r != null && a.exitTime);
    if (closedAlerts.length === 0) return [];
    const sorted = [...closedAlerts].sort((a: any, b: any) => safeDateMs(a.entryTime) - safeDateMs(b.entryTime));

    let fwdStartR = 0;
    if (equityPoints.length > 0 && equityBoundaryIndex != null && equityBoundaryIndex > 0) {
      fwdStartR = equityPoints[equityBoundaryIndex - 1]?.cumulative_r ?? 0;
    }

    let cum = fwdStartR;
    return sorted.map((a: any, i: number) => {
      cum += (a.r ?? 0);
      return { trade_number: i + 1, cumulative_r: cum, timestamp: a.exitTime || a.entryTime || '--' };
    });
  }, [recentAlerts, equityPoints, equityBoundaryIndex]);

  // Split stored_trades into BT and FWD for Performance vs Plan chart
  const { pvpBtTrades, pvpFwdTrades } = useMemo(() => {
    const raw = apiStrategy?.stored_trades || [];
    if (!equityBoundaryIndex || equityBoundaryIndex >= raw.length) return { pvpBtTrades: raw, pvpFwdTrades: [] };
    return {
      pvpBtTrades: raw.slice(0, equityBoundaryIndex),
      pvpFwdTrades: raw.slice(equityBoundaryIndex),
    };
  }, [apiStrategy, equityBoundaryIndex]);

  // Compute sigma badges from PvP formula (same as Performance vs Plan chart)
  const clientSigma = useMemo(() => {
    if (pvpBtTrades.length < 10) return { fwd: null, alert: null };
    const btR = pvpBtTrades.map((t: any) => t.r_multiple ?? 0);
    const n = btR.length;
    const avgR = btR.reduce((s: number, r: number) => s + r, 0) / n;
    const varR = btR.reduce((s: number, r: number) => s + (r - avgR) ** 2, 0) / (n - 1);

    // FWD sigma
    let fwdSigma: number | null = null;
    if (pvpFwdTrades.length >= 3) {
      const fwdCum = pvpFwdTrades.reduce((s: number, t: any) => s + (t.r_multiple ?? 0), 0);
      const expected = pvpFwdTrades.length * avgR;
      const stdAtN = Math.sqrt(pvpFwdTrades.length * varR);
      fwdSigma = stdAtN > 0 ? (fwdCum - expected) / stdAtN : 0;
    }

    // Alert sigma
    let alertSigma: number | null = null;
    const closedAlerts = recentAlerts.filter((a: any) => a.r != null);
    if (closedAlerts.length >= 3) {
      const alertCum = closedAlerts.reduce((s: number, a: any) => s + (a.r ?? 0), 0);
      const expected = closedAlerts.length * avgR;
      const stdAtN = Math.sqrt(closedAlerts.length * varR);
      alertSigma = stdAtN > 0 ? (alertCum - expected) / stdAtN : 0;
    }

    return { fwd: fwdSigma, alert: alertSigma };
  }, [pvpBtTrades, pvpFwdTrades, recentAlerts]);

  // Use the shared buildStrategyChartPanes helper so Strategy Detail and
  // Strategy Builder render charts through one code path. candle coloring
  // (Swing 1-2-3 etc.), per-family oscillator panes, and internal-column
  // filtering all come from the helper.
  const chartTabData = useMemo(() => {
    const chartSrc = chartDataResp?.chart_data;
    const rawBars = chartSrc && chartSrc.length > 0
      ? chartSrc
      : barsData
        ? barsData.map(b => ({ timestamp: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume }))
        : [];
    const bars = candleCount > 0 && rawBars.length > candleCount
      ? rawBars.slice(-candleCount)
      : rawBars;

    if (bars.length === 0) {
      return { chartPanes: [] as PaneConfig[], overlayNames: [] as string[], hasBars: false };
    }

    const overlayNames: string[] = (chartDataResp as any)?.overlay_indicators || [];
    const oscNames: string[] = (chartDataResp as any)?.oscillator_indicators || [];
    const heatmapConds: any[] = ((chartDataResp as any)?.heatmap_conditions || []).filter((c: any) => c.has_data);
    const candleColorColumn: string | undefined = (chartDataResp as any)?.candle_color_column || undefined;

    // Normalize trade rows for the helper — raw times + entry_trigger so the
    // helper can detect exec type and apply C-type shift itself.
    const markerTrades = (btTrades.length > 0 || fwdTrades.length > 0)
      ? [...btTrades, ...fwdTrades]
      : (apiStrategy?.stored_trades || []).map((t: any) => ({
          entry_time: t.entry_time,
          exit_time: t.exit_time,
          entry_price: t.entry_price ?? 0,
          exit_price: t.exit_price ?? 0,
          r_multiple: t.r_multiple ?? 0,
          exit_reason: t.exit_reason || '',
          entry_trigger: t.entry_trigger || '',
          exec_type: t.exec_type || 'C',
          stop_exec_type: t.stop_exec_type,
          target_exec_type: t.target_exec_type,
        }));

    const chartPanes = buildStrategyChartPanes({
      bars,
      trades: markerTrades,
      alerts: recentAlerts,
      direction: strategy?.direction || 'LONG',
      overlayNames,
      oscNames,
      heatmapConds,
      showConditions,
      showTriggers,
      tfMs,
      candleColorColumn,
      chartPrefs,
    });

    return { chartPanes, overlayNames, hasBars: true };
  }, [
    chartDataResp, barsData, candleCount, showConditions, showTriggers,
    chartPrefs, btTrades, fwdTrades, apiStrategy, recentAlerts, tfMs,
    strategy?.direction,
  ]);

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

  // Derive exec type from trigger ID suffix
  const deriveExecTag = (id: string): string => {
    if (id.endsWith('_lc')) return '[LC]';
    if (id.endsWith('_cc')) return '[CC]';
    if (id.endsWith('_ib') || id.endsWith('_hm') || id.endsWith('_hl')) return '[L]';
    return '[C]';
  };

  // Parse entry for badges — derive exec type from the raw trigger ID
  const entryParsed = parseExecTag(strategy.entry);
  if (!entryParsed.exec && strategy.entryId) {
    entryParsed.exec = deriveExecTag(strategy.entryId);
  }
  const entryPack = parsePack(entryParsed.rest);
  const exitsParsed = strategy.exit.map((e: string, i: number) => {
    const p = parseExecTag(e);
    const rawId = strategy.exitIds?.[i] || e;
    if (!p.exec && rawId && rawId !== '--') p.exec = deriveExecTag(rawId);
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
    textAlign: 'center' as const,
    padding: '6px 8px',
    fontSize: '0.7rem',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.05em',
    whiteSpace: 'nowrap' as const,
  };

  const tdStyle: React.CSSProperties = {
    padding: '6px 8px',
    fontSize: '0.8125rem',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    textAlign: 'center' as const,
    verticalAlign: 'middle' as const,
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
      <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
        <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['#', 'Entry Time', 'Exit Time', 'Hold', 'Entry $', 'Exit $', 'R', 'Exec', 'Exit Reason'].map((h) => (
                <th key={h} style={{ ...thStyle, position: 'sticky' as const, top: 0, zIndex: 1, background: 'var(--bg-card)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id}>
                <td style={tdStyle}>{t.id}</td>
                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{t.entryTime}</td>
                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{t.exitTime}</td>
                <td style={{ ...tdStyle, fontSize: '0.75rem' }}>{t.holdTime || '--'}</td>
                <td style={tdStyle}>${t.entryPrice.toFixed(2)}</td>
                <td style={tdStyle}>${t.exitPrice.toFixed(2)}</td>
                <td style={{ ...tdStyle, color: t.pnlR >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                  {t.pnlR >= 0 ? '+' : ''}{t.pnlR.toFixed(2)}R
                </td>
                <td style={tdStyle}>
                  <span
                    className="text-xs font-mono px-1.5 py-0.5 rounded-full"
                    style={{
                      color: EXEC_BADGE_COLOR,
                      background: EXEC_BADGE_COLOR + '20',
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
              style={{ ...btnSecondary, opacity: (fwdLoading || fwdRequested) ? 0.6 : 1 }}
              disabled={fwdLoading}
              onClick={() => setFwdRequested(true)}
            >
              {fwdLoading ? 'Loading FWD...' : 'Update Forward Tests'}
            </button>
            <button
              style={{ ...btnSecondary, opacity: refreshMut.isPending ? 0.6 : 1 }}
              disabled={refreshMut.isPending}
              onClick={() => refreshMut.mutate(strategyId)}
            >
              {refreshMut.isPending ? 'Refreshing...' : 'Update All Data'}
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
        <SigmaBadge label="FWD" value={clientSigma.fwd ?? strategy.fwdSD} color={EQ_FWD_COLOR} />
        <SigmaBadge label="Alert" value={clientSigma.alert ?? strategy.alertSD} color={EQ_LIVE_COLOR} />
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
        &nbsp;&middot;&nbsp;Algo: BT {strategy.btDays}d ({btTrades.length}) + FWD {fwdDays}d ({fwdTrades.length}) &middot; Alerts: {recentAlerts.length} &middot; Alert Accuracy {alertAccuracy}%
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
            {exitsParsed.length > 0 ? exitsParsed.map((e, i) => (
              <span key={i} className="flex items-center gap-1">
                {e.exec && <ExecBadge exec={e.exec} />}
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                  {e.pack && <>{e.pack} &gt; </>}{e.trigger}
                </span>
                {i < exitsParsed.length - 1 && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>,</span>}
              </span>
            )) : null}
            {strategy.barCountExit && (
              <span className="flex items-center gap-1">
                {exitsParsed.length > 0 && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>,</span>}
                <ExecBadge exec="[C]" />
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>Bar Count Exit (Default) &gt; {strategy.barCountExit} bars</span>
              </span>
            )}
            {exitsParsed.length === 0 && !strategy.barCountExit && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>--</span>
            )}
          </div>

          {/* Row 2: Stop + Target + Confluence */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs" style={{ color: 'var(--text-muted)', minWidth: 48 }}>stop:</span>
            <ExecBadge exec="[L]" />
            <StopBadge text={strategy.stop} />
            <span className="text-xs" style={{ color: 'var(--border)', margin: '0 4px' }}>|</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>target:</span>
            {strategy.target !== 'Signal exit only' && strategy.target !== '--' && <ExecBadge exec="[L]" />}
            <TargetBadge text={strategy.target} />
            {strategy.timeExitSummary && (
              <>
                <span className="text-xs" style={{ color: 'var(--border)', margin: '0 4px' }}>|</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>time exit:</span>
                <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange-muted)' }}>
                  {strategy.timeExitSummary}
                </span>
              </>
            )}
            <span className="text-xs" style={{ color: 'var(--border)', margin: '0 4px' }}>|</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>confluence:</span>
            {(strategy.confluenceEnriched?.length > 0 ? strategy.confluenceEnriched : strategy.confluence.map((c: string) => ({ id: c, fidelity: null, label: c }))).map((c: any) => (
              <span key={c.id} className="flex items-center gap-1">
                {c.fidelity && <FidelityBadge label={`[${c.fidelity}]`} />}
                <ConditionBadge text={c.id} />
              </span>
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
                      {['Strategy Default', 'All Data', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Backtest Only', 'Forward Only', 'Custom'].map((o) => (
                        <option key={o} value={o}>{o}</option>
                      ))}
                    </select>
                    {dateRange === 'Custom' && (
                      <>
                        <input type="date" value={customStart} onChange={(e) => setCustomStart(e.target.value)} style={selectStyle} />
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>to</span>
                        <input type="date" value={customEnd} onChange={(e) => setCustomEnd(e.target.value)} style={selectStyle} />
                      </>
                    )}
                    {dateRangeText && (
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{dateRangeText}</span>
                    )}
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
                        { label: 'Avg Hold', value: extendedKPIs.avgHold || '--' },
                        { label: 'Median Hold', value: extendedKPIs.medianHold || '--' },
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
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-medium">Equity Curve</h4>
                    {/* X-axis toggle */}
                    <div className="flex items-center gap-1 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                      {([{ id: 'trade' as const, label: 'Per Trade' }, { id: 'time' as const, label: 'Per Day' }]).map((opt) => {
                        const active = (eqXAxisLocal ?? chartPrefs.eqXAxis) === opt.id;
                        return (
                          <button
                            key={opt.id}
                            className="text-[10px] px-3 py-1 transition-colors"
                            style={{ background: active ? 'var(--accent-muted)' : 'transparent', color: active ? 'var(--accent)' : 'var(--text-muted)' }}
                            onClick={() => setEqXAxisLocal(opt.id)}
                          >
                            {opt.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <EquityCurve
                    data={equityPoints}
                    boundaryIndex={equityBoundaryIndex}
                    height={300}
                    showZeroLine={chartPrefs.eqShowZeroLine}
                    showHWM={eqShowHWM || chartPrefs.eqShowHWM}
                    showEdgeCheck={eqShowEdge}
                    xAxis={eqXAxisLocal ?? chartPrefs.eqXAxis}
                    btColor={chartPrefs.eqBacktestColor || EQ_BT_COLOR}
                    fwdColor={chartPrefs.eqForwardColor || EQ_FWD_COLOR}
                    liveColor={chartPrefs.eqLiveColor || EQ_LIVE_COLOR}
                    lineStyle={chartPrefs.eqLineStyle || 'solid'}
                    showGradient={chartPrefs.eqFillGradient !== false}
                    alertOverlayData={alertEquityPoints}
                  />
                  <div className="flex items-center gap-6 mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span className="flex items-center gap-1.5">
                      <span style={{ display: 'inline-block', width: 16, height: 2, background: chartPrefs.eqBacktestColor || EQ_BT_COLOR }} /> Backtest
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span style={{ display: 'inline-block', width: 16, height: 2, background: chartPrefs.eqForwardColor || EQ_FWD_COLOR }} /> Forward
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span style={{ display: 'inline-block', width: 16, height: 2, background: chartPrefs.eqLiveColor || EQ_LIVE_COLOR }} /> Alerts
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
                    <DistributionChart values={btTrades.map(t => t.pnlR)} bins={20} height={200} />
                  </Card>
                  <Card>
                    <h4 className="text-sm font-medium mb-3">FWD R-Distribution</h4>
                    <DistributionChart values={fwdTrades.map(t => t.pnlR)} bins={20} height={200} />
                  </Card>
                </div>

                {/* ---- Performance vs Plan ---- */}
                {pvpFwdTrades.length < 3 ? (
                  <Card className="mb-4">
                    <h4 className="text-sm font-medium mb-3">Performance vs Plan</h4>
                    <div className="flex items-center justify-center py-8">
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Collecting forward test data — needs at least 3 forward trades to compare against backtest predictions.
                        {pvpFwdTrades.length > 0 && ` (${pvpFwdTrades.length}/3 trades so far)`}
                      </p>
                    </div>
                  </Card>
                ) : (
                  <Card className="mb-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-medium">Performance vs Plan</h4>
                      {recentAlerts.filter((a: any) => a.r != null).length >= 3 && (
                        <div className="flex items-center gap-1 rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                          {([{ id: 'forward' as const, label: 'Forward Test' }, { id: 'alerts' as const, label: 'Alert Trades' }]).map((opt) => (
                            <button
                              key={opt.id}
                              className="text-[10px] px-3 py-1 transition-colors"
                              style={{ background: pvpViewMode === opt.id ? 'var(--accent-muted)' : 'transparent', color: pvpViewMode === opt.id ? 'var(--accent)' : 'var(--text-muted)' }}
                              onClick={() => setPvpViewMode(opt.id)}
                            >
                              {opt.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                    <PerformanceVsPlan
                      btTrades={pvpBtTrades}
                      fwdTrades={pvpFwdTrades}
                      alertTrades={recentAlerts}
                      viewMode={pvpViewMode}
                      height={280}
                    />
                  </Card>
                )}

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
                        <RollingLineChart
                          data={computeRollingMetric(allTrades, rollingWindow, rollingMetric)}
                          label={`Rolling ${rollingMetric} (window=${rollingWindow})`}
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

                    {advancedTab === 'Markov Motor' && (() => {
                      const windowedTrades = allTrades.slice(-markovWindow);
                      const markov = computeMarkov(windowedTrades.length >= 2 ? windowedTrades : allTrades);
                      const rollingWR = computeRollingMetric(allTrades, markovWindow, 'Win Rate');
                      return (
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

                        <RollingLineChart
                          data={rollingWR}
                          label={`Rolling Win Rate with Markov transitions (window=${markovWindow})`}
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
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--green)' }}>{markov.ww}%</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--red)' }}>{markov.wl}%</td>
                              </tr>
                              <tr>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Loss</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--green)' }}>{markov.lw}%</td>
                                <td style={{ ...tdStyle, textAlign: 'center', color: 'var(--red)' }}>{markov.ll}%</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>

                        <div className="flex items-center gap-6 mt-4">
                          <div>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Trend Score</p>
                            <p className="text-base font-bold" style={{ color: markov.trendScore > 0.5 ? 'var(--green)' : 'var(--red)' }}>{markov.trendScore.toFixed(2)}</p>
                          </div>
                          <div>
                            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Edge Strength</p>
                            <p className="text-base font-bold" style={{ color: markov.edgeStrength > edgeDecay ? 'var(--green)' : 'var(--text-muted)' }}>{markov.edgeStrength.toFixed(2)}</p>
                          </div>
                        </div>
                      </Card>
                      );
                    })()}
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
                  {/* M8.5: live-broadcast status pill. Green when Ralph has
                      pushed a bar within the last 5× the timeframe duration,
                      gray otherwise. */}
                  <LiveBarStatusPill liveBar={liveBar} tfSeconds={tfSeconds} />
                </div>

                {/* ---- Synchronized Multi-Pane Chart ----
                   M8.5 B+: chartPanes is memoized above (chartTabData) so
                   parent re-renders from useLiveBar broadcasts (every
                   ~250ms) no longer produce a fresh `panes` reference.
                   SyncedChartPane's setup effect stays cold and the
                   imperative formingBar update path handles live ticks
                   without rebuilding the chart. */}
                {!chartTabData.hasBars ? (
                  <Card className="mb-4">
                    <ChartPlaceholder
                      label={stratSymbol ? `Loading ${stratSymbol} bars...` : 'OHLC chart'}
                      height={400}
                    />
                  </Card>
                ) : (
                  <>
                    {chartDataLoading && !chartDataResp && (
                      <div className="text-xs px-3 py-1.5 mb-2 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                        Loading indicators & heatmap data...
                      </div>
                    )}
                    <Card className="mb-4">
                      <SyncedChartPane
                        panes={chartTabData.chartPanes}
                        upColor={chartPrefs.candleUp}
                        downColor={chartPrefs.candleDown}
                        upBorderColor={chartPrefs.candleUpBorder}
                        gridLines={chartPrefs.gridLines}
                        rightOffset={chartPrefs.rightOffset}
                        timezone={chartPrefs.timezone}
                        formingBar={formingBarProp}
                        formingIndicators={formingIndicators}
                        formingStates={formingStates}
                        formingStateCrossTf={formingStateCrossTf}
                      />
                      {/* Legend */}
                      <div className="flex flex-wrap gap-3 mt-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                        {chartTabData.overlayNames.map((name: string, i: number) => (
                          <span key={name} className="flex items-center gap-1">
                            <span className="inline-block w-3 h-0.5 rounded" style={{ background: INDICATOR_COLORS[i % INDICATOR_COLORS.length] }} />
                            {name.replace(/_/g, ' ')}
                          </span>
                        ))}
                        <span className="flex items-center gap-1"><span style={{ color: chartPrefs.entryColor }}>&#9650;</span> Entry</span>
                        <span className="flex items-center gap-1"><span style={{ color: chartPrefs.exitWinColor }}>&#9679;</span> Win</span>
                        <span className="flex items-center gap-1"><span style={{ color: chartPrefs.exitLossColor }}>&#9679;</span> Loss</span>
                        <span className="flex items-center gap-1"><span style={{ color: chartPrefs.exitStopColor }}>&#9679;</span> Stop</span>
                        <span className="flex items-center gap-1"><span style={{ color: chartPrefs.entryColor }}>+</span> Algo price</span>
                        <span className="flex items-center gap-1"><span style={{ color: chartPrefs.entryColor }}>&times;</span> Alert price</span>
                      </div>
                    </Card>
                  </>
                )}

                {/* Position Status */}
                <Card className="mb-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-medium">Position Status</h4>
                    {(() => {
                      const hasOpenPosition = recentAlerts.some((t: any) => t.result === 'Open');
                      return (
                        <button
                          onClick={handleManualExit}
                          disabled={!hasOpenPosition || manualExitLoading}
                          className="px-3 py-1.5 rounded text-xs font-medium transition-colors"
                          style={{
                            background: hasOpenPosition ? 'var(--red)' : 'var(--bg-input)',
                            color: hasOpenPosition ? 'white' : 'var(--text-muted)',
                            border: hasOpenPosition ? 'none' : '1px solid var(--border)',
                            opacity: manualExitLoading ? 0.6 : 1,
                            cursor: hasOpenPosition ? 'pointer' : 'not-allowed',
                          }}
                        >
                          {manualExitLoading ? 'Exiting...' : hasOpenPosition ? 'Manual Exit' : 'No Position'}
                        </button>
                      );
                    })()}
                  </div>
                  {manualExitResult && (
                    <div className="mb-3 px-3 py-2 rounded text-xs" style={{
                      background: manualExitResult.ok ? 'rgba(76,175,80,0.1)' : 'rgba(244,67,54,0.1)',
                      color: manualExitResult.ok ? 'var(--green)' : 'var(--red)',
                      border: `1px solid ${manualExitResult.ok ? 'var(--green)' : 'var(--red)'}`,
                    }}>
                      {manualExitResult.msg}
                    </div>
                  )}
                  {(() => {
                    const lastBar = chartDataResp?.chart_data?.length ? chartDataResp.chart_data[chartDataResp.chart_data.length - 1] : null;
                    const lastAlert = recentAlertEvents.length > 0 ? recentAlertEvents[0] : null;
                    const hasOpenPosition = recentAlerts.some((t: any) => t.result === 'Open');
                    const currentPrice = lastBar ? `$${Number(lastBar.close).toFixed(2)}` : '--';
                    const lastSignal = lastAlert ? `${lastAlert.type} (${lastAlert.trigger})` : '--';
                    const signalTime = lastAlert?.time ? formatTime(lastAlert.time) : '--';
                    const status = hasOpenPosition ? 'In Position' : 'Flat';
                    return (
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <div>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Status</p>
                          <p className="text-base font-bold mt-1" style={{ color: hasOpenPosition ? 'var(--green)' : 'var(--text-muted)' }}>{status}</p>
                        </div>
                        <div>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last Signal</p>
                          <p className="text-base font-bold mt-1">{lastSignal}</p>
                        </div>
                        <div>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Signal Time</p>
                          <p className="text-base font-bold mt-1 font-mono">{signalTime}</p>
                        </div>
                        <div>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Current Price</p>
                          <p className="text-base font-bold mt-1">{currentPrice}</p>
                        </div>
                      </div>
                    );
                  })()}
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
                        {(() => {
                          const heatmapConds = ((chartDataResp as any)?.heatmap_conditions || []).filter((c: any) => c.has_data);
                          const lastBar = chartDataResp?.chart_data?.length ? chartDataResp.chart_data[chartDataResp.chart_data.length - 1] : null;
                          const allConds = heatmapConds.length > 0 ? heatmapConds : strategy.confluence.map((c: string) => {
                            const parts = c.split('-', 3);
                            return parts.length >= 3 ? { label: c, column: parts[1], needed_state: parts[2], has_data: false } : null;
                          }).filter(Boolean);

                          if (allConds.length === 0) {
                            return (
                              <tr>
                                <td colSpan={4} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                  No confluence conditions configured
                                </td>
                              </tr>
                            );
                          }

                          return allConds.map((cond: any, i: number) => {
                            const currentState = lastBar ? (lastBar[`_state_${cond.column}`] ?? '--') : '--';
                            const neededState = cond.needed_state;
                            const isMet = currentState === neededState;
                            return (
                              <tr key={i}>
                                <td style={tdStyle}>{cond.label}</td>
                                <td style={tdStyle}>
                                  <span className="text-xs font-mono px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
                                    {currentState}
                                  </span>
                                </td>
                                <td style={tdStyle}>
                                  <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{neededState}</span>
                                </td>
                                <td style={tdStyle}>
                                  <span style={{ color: isMet ? 'var(--green)' : 'var(--red)', fontWeight: 600, fontSize: '0.75rem' }}>
                                    {isMet ? 'Met' : 'Not met'}
                                  </span>
                                </td>
                              </tr>
                            );
                          });
                        })()}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Algo History + Alert History side by side */}
                {(() => {
                  const slipTol = chartPrefs.alertSlippage;
                  const fmtDelta = (d: number | null) => {
                    if (d == null) return '--';
                    const sign = d >= 0 ? '+' : '';
                    return Math.abs(d) < 1 ? `${sign}${d.toFixed(1)}s` : `${sign}${Math.round(d)}s`;
                  };
                  const deltaColor = (d: number | null) => {
                    if (d == null) return 'var(--text-muted)';
                    return Math.abs(d) <= slipTol ? 'var(--green)' : 'var(--red)';
                  };
                  // Sort algo trades by entry time descending (most recent first)
                  const sortedAlgoFull = [...fwdTrades, ...btTrades]
                    .map((t, origIdx) => ({ ...t, _origIdx: origIdx }))
                    .sort((a, b) => {
                      const aMs = a.entryTime && a.entryTime !== '--' ? safeDateMs(a.entryTime) : 0;
                      const bMs = b.entryTime && b.entryTime !== '--' ? safeDateMs(b.entryTime) : 0;
                      return bMs - aMs;
                    });
                  // M8.5 B+: cap rendered rows. 7,000+ rows of algo history
                  // would create 70,000+ DOM nodes per table and freeze Edge
                  // on scroll/reflow. Most recent 100 covers the visible
                  // verification use case; full list is available via
                  // Equity & KPIs tab / export if needed later.
                  const ALGO_DISPLAY_CAP = showAllAlgoHistory ? Infinity : 100;
                  const sortedAlgo = sortedAlgoFull.slice(0, ALGO_DISPLAY_CAP);
                  return (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                  {/* Algo History (+ on chart — BT + FWD programmatic trades) — LEFT */}
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-medium">
                        Algo History{' '}
                        <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>
                          (showing {sortedAlgo.length.toLocaleString()} of {sortedAlgoFull.length.toLocaleString()})
                        </span>
                      </h4>
                      {sortedAlgoFull.length > 100 && (
                        <button
                          onClick={() => setShowAllAlgoHistory(v => !v)}
                          className="text-xs"
                          style={{
                            color: 'var(--accent)', background: 'transparent',
                            border: 'none', cursor: 'pointer', padding: '2px 6px',
                          }}
                          title="Rendering 7,000+ rows can slow page interactions"
                        >
                          {showAllAlgoHistory ? 'Show recent 100' : `Show all ${sortedAlgoFull.length.toLocaleString()}`}
                        </button>
                      )}
                    </div>
                    <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Entry Time', '\u0394', 'Exit Time', '\u0394', 'Hold', 'Entry $', 'Exit $', 'R', 'Result', 'Exit Reason'].map((h, hi) => (
                              <th key={hi} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {sortedAlgo.length === 0 ? (
                            <tr>
                              <td colSpan={10} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                No trades available — run backtest to populate
                              </td>
                            </tr>
                          ) : sortedAlgo.map((row: any, si: number) => {
                            const m = algoMatches[row._origIdx] || { matched: false, entryDelta: null, exitDelta: null };
                            const execBadge = row.execType || 'C';
                            const isL = execBadge.includes('L') || execBadge.includes('HM') || execBadge.includes('HL');
                            const exitLabel = (row.exitReason || '--').replace(/_/g, ' ');
                            return (
                              <tr key={si}>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem', cursor: 'pointer' }}
                                    onClick={() => setZoomTrade({ idx: row._origIdx ?? si, side: 'entry', trade: row, alertMatch: m.matched ? m : null })}
                                    title="Click to drill down into 1-second candles">
                                  {renderTime(row.entryTimeDisplay, isL ? 'L' : 'C')}
                                </td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', color: deltaColor(m.entryDelta) }}>{fmtDelta(m.entryDelta)}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem', cursor: 'pointer' }}
                                    onClick={() => setZoomTrade({ idx: row._origIdx ?? si, side: 'exit', trade: row, alertMatch: m.matched ? m : null })}
                                    title="Click to drill down into 1-second candles">
                                  {renderTime(row.exitTimeDisplay, row.exitExecType === 'L' ? 'L' : 'C')}
                                </td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', color: deltaColor(m.exitDelta) }}>{fmtDelta(m.exitDelta)}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', color: 'var(--text-muted)' }}>{row.holdTime || '--'}</td>
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
                                <td style={{ ...tdStyle, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                  {exitLabel}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </Card>

                  {/* Alert History (× on chart — actual alert executions) — RIGHT */}
                  <Card>
                    <h4 className="text-sm font-medium mb-3">Alert History <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({recentAlerts.length})</span></h4>
                    <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
                      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Entry Time', '\u0394', 'Exit Time', '\u0394', 'Hold', 'Entry $', 'Exit $', 'R', 'Result', 'Exit Reason'].map((h, hi) => (
                              <th key={hi} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {recentAlerts.length === 0 ? (
                            <tr>
                              <td colSpan={10} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                                No alerts available — enable monitoring to populate
                              </td>
                            </tr>
                          ) : recentAlerts.map((row: any, i: number) => {
                            const m = alertMatches[i] || { matched: false, entryDelta: null, exitDelta: null };
                            const exitLabel = (row.exitReason || '--').replace(/_/g, ' ');
                            // Compute alert hold time from entry/exit timestamps
                            const alertHoldMs = safeDateMs(row.exitTime) && safeDateMs(row.entryTime) ? safeDateMs(row.exitTime) - safeDateMs(row.entryTime) : 0;
                            const alertHold = alertHoldMs > 0 ? formatHoldTime(alertHoldMs / 1000, null) : '--';
                            return (
                              <tr key={i}>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{renderTime(row.entryTime)}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', color: deltaColor(m.entryDelta) }}>{fmtDelta(m.entryDelta)}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{renderTime(row.exitTime)}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', color: deltaColor(m.exitDelta) }}>{fmtDelta(m.exitDelta)}</td>
                                <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.7rem', color: 'var(--text-muted)' }}>{alertHold}</td>
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
                                <td style={{ ...tdStyle, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                  {exitLabel}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </div>
                  );
                })()}
              </div>
            )}

            {/* =========================================================== */}
            {/* TAB 3: CONFLUENCE ANALYSIS                                   */}
            {/* =========================================================== */}
            {tab === 'Confluence Analysis' && (
              <div>
                {strategyConfluence.length === 0 ? (
                  <p className="text-sm py-4" style={{ color: 'var(--text-muted)' }}>No confluence conditions configured for this strategy.</p>
                ) : (
                <>
                {/* Condition selector buttons */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {strategyConfluence.map((cond) => {
                    const parts = cond.split('-', 3);
                    const tf = parts[0] || '';
                    const interp = parts[1] || '';
                    const state = parts[2] || '';
                    const isSelected = selectedCondition === cond;
                    return (
                      <button
                        key={cond}
                        onClick={() => setSelectedCondition(cond)}
                        className="px-3 py-2 rounded-lg text-xs font-medium transition-colors"
                        style={{
                          background: isSelected ? 'var(--accent)' : 'var(--bg-card)',
                          color: isSelected ? 'white' : 'var(--text-secondary)',
                          border: isSelected ? 'none' : '1px solid var(--border)',
                        }}
                      >
                        <span className="font-mono">[{tf}]</span>{' '}
                        <span>{interp.replace(/_/g, ' ')}</span>{' '}
                        <span style={{ color: isSelected ? 'rgba(255,255,255,0.7)' : 'var(--text-muted)' }}>{state}</span>
                      </button>
                    );
                  })}
                </div>

                {/* Confluence condition chart */}
                {selectedCondition && (
                  <div>
                    <Card className="mb-4">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-sm font-medium">
                          {selectedCondition.split('-')[1]?.replace(/_/g, ' ') || 'Indicator'} — {confChartData?.timeframe || '...'} Chart
                        </h4>
                        <div className="flex items-center gap-3">
                          <span className="text-xs font-mono px-2 py-1 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                            Looking for: <strong style={{ color: 'var(--accent)' }}>{selectedCondition.split('-').slice(2).join('-')}</strong>
                          </span>
                          {confChartData && confChartData.bars.length > 0 && (() => {
                            const lastBar = confChartData.bars[confChartData.bars.length - 1];
                            const currentState = lastBar._state || '--';
                            const isMet = lastBar._met === true || currentState === confChartData.needed_state;
                            return (
                              <span className="text-xs font-mono px-2 py-1 rounded" style={{
                                background: isMet ? 'var(--green-muted)' : 'var(--red-muted)',
                                color: isMet ? 'var(--green)' : 'var(--red)',
                              }}>
                                Current: <strong>{currentState}</strong> {isMet ? '✓ Met' : '✗ Not met'}
                              </span>
                            );
                          })()}
                        </div>
                      </div>
                      {confChartLoading ? (
                        <ChartPlaceholder label={`Loading ${selectedCondition.split('-')[0]} chart...`} height={350} />
                      ) : confPanesData ? (
                        <SyncedChartPane
                          panes={confPanesData}
                          upColor={chartPrefs.candleUp}
                          downColor={chartPrefs.candleDown}
                          upBorderColor={chartPrefs.candleUpBorder}
                          gridLines={chartPrefs.gridLines}
                          rightOffset={chartPrefs.rightOffset}
                          timezone={chartPrefs.timezone}
                        />
                      ) : (
                        <ChartPlaceholder label="No data for this condition" height={350} />
                      )}
                      {/* State legend */}
                      <div className="flex items-center gap-4 mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                        <span className="flex items-center gap-1">
                          <span className="inline-block w-4 h-3 rounded" style={{ background: 'rgba(76,175,80,0.15)' }} />
                          Condition met
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="inline-block w-4 h-3 rounded" style={{ background: 'rgba(244,67,54,0.1)' }} />
                          Condition not met
                        </span>
                        {((confChartData as any)?.overlay_indicators || []).map((col: string, i: number) => (
                          <span key={col} className="flex items-center gap-1">
                            <span className="inline-block w-3 h-0.5 rounded" style={{ background: ['#2196F3', '#FF9800', '#4CAF50', '#E91E63'][i % 4] }} />
                            {col.replace(/_/g, ' ')}
                          </span>
                        ))}
                        {((confChartData as any)?.oscillator_indicators || []).map((col: string) => (
                          <span key={col} className="flex items-center gap-1">
                            <span className="inline-block w-3 h-0.5 rounded" style={{ background: col.includes('signal') ? '#FF9800' : '#2196F3' }} />
                            {col.replace(/_/g, ' ')} (oscillator)
                          </span>
                        ))}
                      </div>
                    </Card>
                  </div>
                )}
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
                        {exitsParsed.length > 0 && exitsParsed.map((e, i) => (
                          <div key={i} className="flex items-center gap-2 mt-1 flex-wrap">
                            {e.exec && <ExecBadge exec={e.exec} />}
                            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                              {e.pack && <>{e.pack} &gt; </>}{e.trigger}
                            </span>
                          </div>
                        ))}
                        {strategy.barCountExit != null && (
                          <div className="flex items-center gap-2 mt-1 flex-wrap">
                            <ExecBadge exec="[C]" />
                            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                              Bar Count Exit (Default) &gt; {strategy.barCountExit} bars
                            </span>
                          </div>
                        )}
                        {exitsParsed.length === 0 && !strategy.barCountExit && (
                          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>--</p>
                        )}
                      </div>
                      {/* Stop */}
                      <div>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Stop</span>
                        <div className="mt-1 flex items-center gap-1">
                          <ExecBadge exec="[L]" />
                          <StopBadge text={strategy.stop} />
                        </div>
                      </div>
                      {/* Target */}
                      <div>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Target</span>
                        <div className="mt-1 flex items-center gap-1">
                          {strategy.target !== 'Signal exit only' && strategy.target !== '--' && <ExecBadge exec="[L]" />}
                          <TargetBadge text={strategy.target} />
                        </div>
                      </div>
                      {/* Time Exit */}
                      {strategy.timeExitSummary && (
                        <div>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Time Exit</span>
                          <div className="mt-1">
                            <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange-muted)' }}>
                              {strategy.timeExitSummary}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </Card>
                </div>

                {/* Confluence conditions with fidelity badges */}
                <Card>
                  <h4 className="text-sm font-medium mb-3">Confluence Conditions</h4>
                  <div className="flex items-center gap-2 flex-wrap">
                    {(strategy.confluenceEnriched?.length > 0 ? strategy.confluenceEnriched : strategy.confluence.map((c: string) => ({ id: c, fidelity: null, label: c }))).map((c: any) => (
                      <span key={c.id} className="flex items-center gap-1">
                        {c.fidelity && <FidelityBadge label={`[${c.fidelity}]`} />}
                        <ConditionBadge text={c.id} />
                      </span>
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
                            { ph: '{{event_type}}', val: recentAlertEvents[0]?.type ? `${recentAlertEvents[0].type.toLowerCase()}_${strategy.direction.toLowerCase()}_market` : '--', src: 'Derived from execution type + direction' },
                            { ph: '{{symbol}}', val: strategy.symbol, src: 'Strategy' },
                            { ph: '{{direction}}', val: strategy.direction, src: 'Strategy' },
                            { ph: '{{order_action}}', val: strategy.direction === 'LONG' ? (recentAlertEvents[0]?.type === 'ENTRY' ? 'buy' : 'sell') : (recentAlertEvents[0]?.type === 'ENTRY' ? 'sell' : 'buy'), src: 'Derived (buy/sell/close)' },
                            { ph: '{{order_type}}', val: 'market', src: 'Derived from execution type' },
                            { ph: '{{order_price}}', val: recentAlertEvents[0]?.price != null ? `$${Number(recentAlertEvents[0].price).toFixed(2)}` : '--', src: 'Fill price or limit price' },
                            { ph: '{{stop_price}}', val: recentAlertEvents[0]?.stopPrice != null ? `$${Number(recentAlertEvents[0].stopPrice).toFixed(2)}` : '--', src: 'Calculated stop' },
                            { ph: '{{quantity}}', val: '--', src: 'Portfolio (risk / stop distance)' },
                            { ph: '{{trigger_name}}', val: recentAlertEvents[0]?.trigger || '--', src: 'Strategy trigger' },
                            { ph: '{{atr}}', val: '--', src: 'Indicator at signal bar' },
                            { ph: '{{confluence_met}}', val: strategy.confluence.length > 0 ? strategy.confluence.join(', ') : '--', src: 'Active conditions at signal' },
                            { ph: '{{portfolio_name}}', val: '--', src: 'Portfolio' },
                            { ph: '{{risk_per_trade}}', val: '--', src: 'Portfolio' },
                            { ph: '{{timestamp}}', val: recentAlertEvents[0]?.time ? new Date(recentAlertEvents[0].time).toLocaleString() : '--', src: 'Signal time' },
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
                        {recentAlertEvents.length === 0 ? (
                          <tr>
                            <td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: 'var(--text-muted)' }}>
                              No alerts available — enable monitoring to populate
                            </td>
                          </tr>
                        ) : recentAlertEvents.map((alert: any, i: number) => (
                          <tr key={i}>
                            <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: '0.75rem' }}>{alert.time ? new Date(alert.time).toLocaleString() : '--'}</td>
                            <td style={tdStyle}>
                              <span
                                className="text-xs font-mono font-semibold px-2 py-0.5 rounded-full"
                                style={{
                                  color: alert.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                                  background: alert.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                                }}
                              >
                                {alert.type === 'ENTRY' ? `entry_${strategy.direction.toLowerCase()}_market` : `exit_${strategy.direction.toLowerCase()}_market`}
                              </span>
                            </td>
                            <td style={tdStyle}>{alert.trigger || '--'}</td>
                            <td style={tdStyle}>{alert.price != null ? `$${Number(alert.price).toFixed(2)}` : '--'}</td>
                            <td style={tdStyle}>
                              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Market</span>
                            </td>
                            <td style={tdStyle}>
                              <span style={{
                                color: alert.status === 'Delivered' ? 'var(--green)' : alert.status === 'Pending' ? 'var(--orange)' : 'var(--text-muted)',
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
                                    color: EXEC_BADGE_COLOR,
                                    background: EXEC_BADGE_COLOR + '20',
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

      {/* Trade Drill-Down Modal */}
      {zoomTrade && (
        <TradeZoomModal
          isOpen={!!zoomTrade}
          onClose={() => setZoomTrade(null)}
          tradeIdx={zoomTrade.idx}
          side={zoomTrade.side}
          trade={zoomTrade.trade}
          zoomData={zoomQuery.data ?? null}
          isLoading={zoomQuery.isLoading}
          error={zoomQuery.error ? String(zoomQuery.error) : null}
          alertMatch={zoomTrade.alertMatch?.matched ? {
            entryPrice: zoomTrade.alertMatch.alertEntryPrice,
            exitPrice: zoomTrade.alertMatch.alertExitPrice,
          } : null}
        />
      )}
    </div>
  );
}
