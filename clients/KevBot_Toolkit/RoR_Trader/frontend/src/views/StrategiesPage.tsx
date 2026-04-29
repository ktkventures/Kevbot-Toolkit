'use client';

/**
 * My Strategies — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: src/app/strategies/versions/V5.tsx (885 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useMemo, useEffect } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { useDeleteStrategy, useDuplicateStrategy, useBulkDeleteStrategies } from '@/hooks/mutations/useStrategyMutations';

/* ========================================================================= */
/* TYPES                                                                       */
/* ========================================================================= */

interface Strategy {
  id: string;
  name: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  timeframe: string;
  session: string;
  status: 'On Track' | 'Outperforming' | 'Underperforming' | 'Insufficient Data';
  tags: string[];
  winRate: number;
  pf: number;
  dailyR: number;
  dailyROI: number;
  trades: number;
  maxDD: number;
  fwdWinRate: number | null;
  fwdPF: number | null;
  fwdDailyROI: number | null;
  fwdTrades: number;
  fwdSince: string;
  alertWinRate: number | null;
  alertPF: number | null;
  alertDailyR: number | null;
  alertDailyROI: number | null;
  alertTrades: number;
  alertMaxDD: number | null;
  entry: string;
  exit: string[];
  stop: string;
  target: string;
  confluence: string[];
  alertTracking: boolean;
  monitored: boolean;
  btDays: number;
  sigmaFwd: number;
  sigmaAlert: number;
  equityCurveData?: { cumulative_r: number[]; exit_times?: string[]; boundary_index?: number | null };
  updatedAt: string;
  portfolioCount: number;
  health?: StrategyHealth;
  // Backtest fidelity (services.compute_strategy_health)
  dataSource?: 'rapid' | 'full' | 'hifi';
  // Algo history fidelity — derived from stored_trades' behavior + hifi_resolved
  algoHistoryFidelity?: 'Bar' | 'Hi-Fi' | 'Mixed' | 'Unknown';
  // Auto-parity status — set by recompute_and_persist_stored_trades after
  // every refresh; null if the strategy has not been refreshed since the
  // 2026-04-28 dbe0cb3 cut-over.
  parityStatus?: ParityStatus | null;
}

// Strategy Health Badge — see services.compute_strategy_health.
// Severity: 'healthy' (no issues) | 'minor' (yellow info) | 'action' (orange,
// fix recommended) | 'broken' (red, can't run as-is).
export type HealthSeverity = 'healthy' | 'minor' | 'action' | 'broken';

export interface HealthIssue {
  severity: 'minor' | 'action' | 'broken';
  code: string;
  title: string;
  detail: string;
  fix_action: string | null;
  fix_action_label: string | null;
}

export interface StrategyHealth {
  severity: HealthSeverity;
  issues: HealthIssue[];
}

// Auto-parity score (services.forward_test_service writes this after every
// recompute_and_persist_stored_trades). Verdicts come from
// api.services.parity_service._build_report.
export type ParityVerdict =
  | 'PASS'
  | 'PARTIAL'
  | 'FAIL_LIVE_BLOCKED'
  | 'FAIL_OVER_FIRES'
  | 'NO_TRADES'
  | 'NO_DATA'
  | 'ERROR';

export interface ParityStatus {
  score: number | null;            // matched_count / total (0..1)
  verdict: ParityVerdict;
  stored_count: number;
  matched_count: number;
  stored_only_count: number;
  replay_only_count: number;
  most_common_failing_gate: string | null;
  computed_at: string;             // ISO8601
  duration_s: number;
  error?: string;                  // present iff verdict === 'ERROR'
  params: { last_n: number; forward_test_only: boolean };
}

/* ========================================================================= */
/* API → V5 Strategy Mapping                                                   */
/* ========================================================================= */

function formatTimeExit(config: any): string {
  if (!config?.method) return '';
  const m = config.method;
  if (m === 'eod_exit') return `EOD ${config.minutes_before_close ?? 15}m`;
  if (m === 'time_of_day_exit') return `@${config.exit_hour ?? 15}:${String(config.exit_minute ?? 50).padStart(2, '0')}`;
  if (m === 'max_hold_bars') return `Max ${config.max_bars ?? 4} bars`;
  if (m === 'session_exit') return 'Session window';
  return m;
}

function apiToStrategy(s: any): Strategy {
  const k = s.kpis || {};
  return {
    id: String(s.id),
    name: s.name || '--',
    symbol: s.symbol || '--',
    direction: s.direction || 'LONG',
    timeframe: s.timeframe || '1Min',
    session: s.trading_session || 'RTH',
    status: s.status || (s.forward_testing ? 'On Track' : 'Insufficient Data'),
    tags: s.tags || [],
    winRate: k.win_rate ?? 0,
    pf: k.profit_factor ?? 0,
    dailyR: k.daily_r ?? 0,
    dailyROI: 0,  // {{daily_roi}} — requires portfolio-level risk context to compute
    trades: k.total_trades ?? 0,
    maxDD: k.max_r_drawdown ?? 0,
    // Forward test KPIs (from enrich_strategy)
    fwdWinRate: s.forward_kpis?.win_rate ?? null,
    fwdPF: s.forward_kpis?.profit_factor ?? null,
    fwdDailyROI: null, // {{fwd_daily_roi}} — needs computation
    // fwdTrades: prefer the server-computed forward_trades_count (list endpoint
    // attaches this by counting trades with entry_fill_ts >= forward_test_start).
    // Falls back to enrich's forward_kpis.trades when ?enrich=true is used.
    fwdTrades: s.forward_trades_count ?? s.forward_kpis?.trades ?? 0,
    fwdSince: s.forward_test_start || '',
    // Alert KPIs (from enrich_strategy)
    alertWinRate: s.alert_kpis?.win_rate ?? null,
    alertPF: s.alert_kpis?.profit_factor ?? null,
    alertDailyR: s.alert_kpis?.daily_r ?? null,
    alertDailyROI: null, // {{alert_daily_roi}}
    // Prefer server-computed alert_trades_count (COUNT(*) from alerts table).
    // Falls back to alert_kpis.trades (from live_executions, only populated
    // by enrich_strategy).
    alertTrades: s.alert_trades_count ?? s.alert_kpis?.trades ?? 0,
    alertMaxDD: s.alert_kpis?.max_r_drawdown ?? null,
    entry: s.entry_trigger_confluence_id || '--',
    exit: s.exit_trigger_confluence_ids || [],
    stop: s.stop_config?.method ? `${s.stop_config.method}${s.stop_config.atr_mult ? ` ${s.stop_config.atr_mult}x` : ''}` : '--',
    target: s.target_config?.method || 'Signal exit only',
    timeExit: s.time_exit_config?.method ? formatTimeExit(s.time_exit_config) : null,
    confluence: s.confluence || [],
    alertTracking: s.alert_tracking_enabled || false,
    monitored: s.alert_tracking_enabled || false,
    btDays: s.data_days || 30,
    sigmaFwd: s.sigma_fwd ?? 0,
    sigmaAlert: s.sigma_alert ?? 0,
    equityCurveData: s.equity_curve_data || undefined,
    updatedAt: s.updated_at || s.created_at || '',
    // Attached server-side by the list endpoint (count of portfolios whose
    // `strategies` array contains this strategy's id).
    portfolioCount: s.portfolio_count ?? 0,
    health: s.health || undefined,
    dataSource: s.data_source || undefined,
    algoHistoryFidelity: s.algo_history_fidelity || 'Unknown',
    parityStatus: s.parity_status || null,
  };
}

/* ========================================================================= */
/* Strategy Fidelity Badges                                                    */
/* ========================================================================= */
// Always-visible row of small chips showing the fidelity layer of each
// strategy: Backtest tier (rapid/full/hifi) and Algo history resolution
// (Bar/Hi-Fi). Distinct from the Health Badge — that surfaces ISSUES;
// these surface INFORMATION about how the strategy was computed.

const FIDELITY_BACKTEST_COLORS: Record<string, { bg: string; fg: string; label: string }> = {
  rapid: { bg: 'var(--orange-muted)', fg: 'var(--orange)', label: 'Rapid' },
  full:  { bg: 'var(--bg-input)',     fg: 'var(--text-secondary)', label: 'Full' },
  hifi:  { bg: 'var(--green-muted)',  fg: 'var(--green)', label: 'Hi-Fi' },
};

const FIDELITY_ALGO_COLORS: Record<string, { bg: string; fg: string; label: string }> = {
  'Bar':     { bg: 'var(--bg-input)',    fg: 'var(--text-secondary)', label: 'Bar' },
  'Hi-Fi':   { bg: 'var(--green-muted)', fg: 'var(--green)',          label: 'Hi-Fi' },
  'Mixed':   { bg: 'var(--orange-muted)',fg: 'var(--orange)',         label: 'Mixed' },
  'Unknown': { bg: 'var(--bg-input)',    fg: 'var(--text-muted)',     label: 'Unknown' },
};

export function StrategyFidelityBadges({
  strategy,
  variant = 'compact',
}: {
  strategy: Pick<Strategy, 'dataSource' | 'algoHistoryFidelity'>;
  variant?: 'compact' | 'detail';
}) {
  const bt = strategy.dataSource ? FIDELITY_BACKTEST_COLORS[strategy.dataSource] : null;
  const algo = strategy.algoHistoryFidelity
    ? FIDELITY_ALGO_COLORS[strategy.algoHistoryFidelity]
    : null;
  if (!bt && !algo) return null;

  const sizeClass = variant === 'compact' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2 py-0.5';

  return (
    <div className="inline-flex items-center gap-1" title="Backtest fidelity / Algo history resolution">
      {bt && (
        <span
          className={`${sizeClass} rounded font-medium`}
          style={{ background: bt.bg, color: bt.fg, border: `1px solid ${bt.fg}40` }}
        >
          BT: {bt.label}
        </span>
      )}
      {algo && (
        <span
          className={`${sizeClass} rounded font-medium`}
          style={{ background: algo.bg, color: algo.fg, border: `1px solid ${algo.fg}40` }}
        >
          Algo: {algo.label}
        </span>
      )}
    </div>
  );
}

/* ========================================================================= */
/* Strategy Parity Badge                                                       */
/* ========================================================================= */
// Always-visible chip: "does this strategy fire live?" Sourced from
// strategies.parity_status which auto-recomputes on every refresh.
//
// Verdict → colour:
//   PASS              → green     ("trust this — backtest reproduces live")
//   PARTIAL           → orange    ("some trades match, some don't")
//   FAIL_LIVE_BLOCKED → red       ("nothing fires live")
//   FAIL_OVER_FIRES   → red       ("live fires extras the backtest didn't")
//   NO_TRADES         → muted     ("strategy hasn't produced any trades yet")
//   NO_DATA / ERROR   → orange    ("couldn't compute — usually transient")
//   null              → muted     ("not refreshed since auto-parity launched")

const PARITY_COLORS: Record<ParityVerdict,
  { bg: string; fg: string; icon: string; label: string }> = {
  PASS:              { bg: 'var(--green-muted)',  fg: 'var(--green)',  icon: '✓', label: 'Live' },
  PARTIAL:           { bg: 'var(--orange-muted)', fg: 'var(--orange)', icon: '~', label: 'Partial' },
  FAIL_LIVE_BLOCKED: { bg: 'var(--red-muted)',    fg: 'var(--red)',    icon: '✕', label: 'Blocked' },
  FAIL_OVER_FIRES:   { bg: 'var(--red-muted)',    fg: 'var(--red)',    icon: '+', label: 'Over' },
  NO_TRADES:         { bg: 'var(--bg-input)',     fg: 'var(--text-muted)', icon: '·', label: 'No trades' },
  NO_DATA:           { bg: 'var(--orange-muted)', fg: 'var(--orange)', icon: '!', label: 'No data' },
  ERROR:             { bg: 'var(--orange-muted)', fg: 'var(--orange)', icon: '!', label: 'Error' },
};

export function StrategyParityBadge({
  parity,
  variant = 'pill',
  onClick,
}: {
  parity?: ParityStatus | null;
  variant?: 'pill' | 'compact';
  onClick?: () => void;
}) {
  const sizeClass = variant === 'compact'
    ? 'text-[10px] px-1.5 py-0.5'
    : 'text-xs px-2 py-0.5';

  // Not yet computed — show a muted placeholder so the user knows the
  // feature exists and a refresh will populate it.
  if (!parity) {
    return (
      <span
        title="No parity score yet — refresh the strategy to compute one"
        className={`${sizeClass} rounded font-medium inline-flex items-center gap-1`}
        style={{
          color: 'var(--text-muted)',
          background: 'var(--bg-input)',
          border: '1px solid var(--border)',
        }}
      >
        <span>?</span>
        <span>Parity —</span>
      </span>
    );
  }

  const c = PARITY_COLORS[parity.verdict] ?? PARITY_COLORS.ERROR;
  const pct = parity.score != null
    ? `${Math.round(parity.score * 100)}%`
    : null;
  const display = pct ?? c.label;
  const tooltip = (() => {
    const parts: string[] = [];
    parts.push(`Parity: ${parity.verdict}`);
    if (parity.score != null) {
      parts.push(`${parity.matched_count}/${parity.stored_count} stored trades match live engine`);
    }
    if (parity.most_common_failing_gate) {
      parts.push(`Most common failing gate: ${parity.most_common_failing_gate}`);
    }
    if (parity.error) {
      parts.push(`Error: ${parity.error}`);
    }
    return parts.join(' — ');
  })();

  return (
    <button
      type="button"
      onClick={onClick ? (e) => { e.stopPropagation(); onClick(); } : undefined}
      title={tooltip}
      className={`${sizeClass} rounded-full font-medium inline-flex items-center gap-1 ${onClick ? 'cursor-pointer' : 'cursor-default'}`}
      style={{ color: c.fg, background: c.bg, border: `1px solid ${c.fg}40` }}
    >
      <span style={{ fontSize: 10, lineHeight: 1 }}>{c.icon}</span>
      <span>{display}</span>
    </button>
  );
}

/* ========================================================================= */
/* Strategy Health Badge                                                       */
/* ========================================================================= */

const HEALTH_SEVERITY_COLORS: Record<HealthSeverity,
  { bg: string; fg: string; label: string }> = {
  healthy: { bg: 'var(--green-muted)', fg: 'var(--green)', label: 'Healthy' },
  minor:   { bg: 'var(--orange-muted)', fg: 'var(--orange)', label: 'Minor' },
  action:  { bg: 'var(--orange-muted)', fg: 'var(--orange)', label: 'Action recommended' },
  broken:  { bg: 'var(--red-muted)', fg: 'var(--red)', label: 'Broken' },
};

const HEALTH_SEVERITY_ICONS: Record<HealthSeverity, string> = {
  healthy: '✓',
  minor: 'i',
  action: '!',
  broken: '✕',
};

export function StrategyHealthBadge({
  health,
  variant = 'pill',
  onOpenDrawer,
}: {
  health?: StrategyHealth;
  variant?: 'pill' | 'banner';
  onOpenDrawer?: () => void;
}) {
  if (!health || health.severity === 'healthy') return null;
  const c = HEALTH_SEVERITY_COLORS[health.severity];
  const icon = HEALTH_SEVERITY_ICONS[health.severity];

  // Tooltip: count by severity
  const counts: Record<string, number> = {};
  for (const i of health.issues) counts[i.severity] = (counts[i.severity] || 0) + 1;
  const tooltipParts = Object.entries(counts)
    .map(([s, n]) => `${n} ${s}`)
    .join(', ');
  const tooltip = `${tooltipParts} — click for details`;

  if (variant === 'banner') {
    return (
      <div
        className="rounded p-3 mb-3 cursor-pointer"
        style={{ background: c.bg, border: `1px solid ${c.fg}40` }}
        onClick={(e) => { e.stopPropagation(); onOpenDrawer?.(); }}
        role="button"
      >
        <div className="flex items-center gap-3">
          <span
            className="inline-flex items-center justify-center rounded-full font-bold"
            style={{ background: c.fg, color: 'white', width: 24, height: 24, fontSize: 12 }}
          >
            {icon}
          </span>
          <div className="flex-1">
            <div className="text-sm font-semibold" style={{ color: c.fg }}>
              {c.label}: {health.issues.length} issue{health.issues.length !== 1 ? 's' : ''}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {health.issues.slice(0, 2).map((i) => i.title).join(' · ')}
              {health.issues.length > 2 && ` (+${health.issues.length - 2} more)`}
            </div>
          </div>
          <span className="text-xs" style={{ color: c.fg }}>Click for details →</span>
        </div>
      </div>
    );
  }

  // pill variant — for use inline in strategy card header
  return (
    <button
      type="button"
      onClick={(e) => { e.stopPropagation(); onOpenDrawer?.(); }}
      title={tooltip}
      className="text-xs px-2 py-0.5 rounded-full font-semibold inline-flex items-center gap-1"
      style={{ background: c.bg, color: c.fg, border: `1px solid ${c.fg}40` }}
    >
      <span style={{ fontSize: 10, lineHeight: 1 }}>{icon}</span>
      <span>{health.issues.length}</span>
    </button>
  );
}

export function StrategyHealthDrawer({
  health,
  isOpen,
  onClose,
  strategyName,
}: {
  health?: StrategyHealth;
  isOpen: boolean;
  onClose: () => void;
  strategyName?: string;
}) {
  if (!health) return null;
  return (
    <Modal
      title={`Strategy Health${strategyName ? ` — ${strategyName}` : ''}`}
      isOpen={isOpen}
      onClose={onClose}
      width="560px"
    >
      {health.severity === 'healthy' ? (
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          No issues detected. This strategy looks good.
        </p>
      ) : (
        <div className="space-y-3">
          {health.issues.map((issue, idx) => {
            const c = HEALTH_SEVERITY_COLORS[issue.severity];
            return (
              <div
                key={idx}
                className="rounded p-3"
                style={{ background: c.bg, border: `1px solid ${c.fg}30` }}
              >
                <div className="flex items-start gap-2">
                  <span
                    className="inline-flex items-center justify-center rounded-full font-bold flex-shrink-0"
                    style={{ background: c.fg, color: 'white', width: 20, height: 20, fontSize: 10, marginTop: 1 }}
                  >
                    {HEALTH_SEVERITY_ICONS[issue.severity]}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold" style={{ color: c.fg }}>
                      {issue.title}
                    </div>
                    <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                      {issue.detail}
                    </div>
                    <div className="text-[10px] mt-1 font-mono" style={{ color: 'var(--text-muted)' }}>
                      code: {issue.code}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
          <p className="text-[11px] mt-3" style={{ color: 'var(--text-muted)' }}>
            Fix-action buttons are coming in the next update — for now, the
            issue codes above are mappable to the relevant action (e.g.
            `rapid_test_data` → re-run a full backtest from the Strategy
            Builder, `live_alert_divergence` → check live engine logs, etc.).
          </p>
        </div>
      )}
    </Modal>
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

function daysSince(dateStr: string): number {
  const d = new Date(dateStr);
  const now = new Date();
  return Math.floor((now.getTime() - d.getTime()) / 86400000);
}

/** Compact "x ago" for the meta line. Falls back to `--` on empty. */
function timeAgo(dateStr: string): string {
  if (!dateStr) return '--';
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return '--';
  const diff = (Date.now() - d.getTime()) / 1000; // seconds
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  // Older than a week — show the date
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** Generate a mini equity curve SVG from mock cumulative R data */
// Display settings V5 equity curve colors
const EXEC_BADGE_COLOR = '#2196F3';
const EQ_BT_COLOR = '#2196F3';
const EQ_FWD_COLOR = '#FF9800';
const EQ_LIVE_COLOR = '#4CAF50';

function MiniEquityCurve({ equityCurveData, fwdStartPct, hasAlerts, showHWM, showEdgeMA, showConfBands, height = 64 }: { equityCurveData?: { cumulative_r: number[]; boundary_index?: number | null }; fwdStartPct: number; hasAlerts: boolean; showHWM: boolean; showEdgeMA: boolean; showConfBands: boolean; height?: number }) {
  // Drop any non-finite values (NaN/Infinity/null) — one of them in cumR
  // makes Math.min/max return NaN, which propagates into every toY() call
  // and Recharts logs "Expected number" errors for every SVG attribute.
  const cumR = (equityCurveData?.cumulative_r || []).filter((v) => Number.isFinite(v));
  const totalPoints = cumR.length;
  if (totalPoints < 2) {
    return <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>--</span></div>;
  }
  const rawBoundary = equityCurveData?.boundary_index;
  const fwdIdx = Number.isFinite(rawBoundary)
    ? Math.min(Math.max(rawBoundary as number, 1), totalPoints - 1)
    : Math.max(1, Math.floor(totalPoints * (Number.isFinite(fwdStartPct) ? fwdStartPct : 0.5)));
  const w = 320;
  const h = height;
  const pad = 3;

  const btPoints = cumR.slice(0, fwdIdx + 1);
  const fwdPoints = cumR.slice(fwdIdx);
  const livePoints: number[] = [];

  const allVals = [...btPoints, ...fwdPoints, ...livePoints];
  const min = Math.min(...allVals);
  const max = Math.max(...allVals);
  const range = max - min || 1;

  const toY = (v: number) => h - ((v - min) / range) * (h - pad * 2) - pad;
  const toX = (idx: number) => (idx / totalPoints) * w;

  const buildLine = (pts: number[], startIdx: number) =>
    pts.map((p, i) => `${toX(startIdx + i).toFixed(1)},${toY(p).toFixed(1)}`).join(' ');
  const buildFill = (pts: number[], startIdx: number) => {
    const line = buildLine(pts, startIdx);
    return `${toX(startIdx).toFixed(1)},${h} ${line} ${toX(startIdx + pts.length - 1).toFixed(1)},${h}`;
  };

  // HWM
  let hwmPeak = -Infinity;
  const hwmLine = btPoints.map((p, i) => { hwmPeak = Math.max(hwmPeak, p); return `${toX(i).toFixed(1)},${toY(hwmPeak).toFixed(1)}`; }).join(' ');

  const zeroY = toY(0);
  const bndX = toX(fwdIdx);

  const uid = totalPoints + fwdIdx; // stable unique ID for gradient defs

  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`btG${uid}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={EQ_BT_COLOR} stopOpacity="0.2" /><stop offset="100%" stopColor={EQ_BT_COLOR} stopOpacity="0" /></linearGradient>
        <linearGradient id={`fwG${uid}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={EQ_FWD_COLOR} stopOpacity="0.15" /><stop offset="100%" stopColor={EQ_FWD_COLOR} stopOpacity="0" /></linearGradient>
      </defs>
      {/* Zero line */}
      <line x1="0" y1={zeroY} x2={w} y2={zeroY} stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="4 3" opacity="0.4" />
      {/* HWM */}
      {showHWM && <polyline points={hwmLine} fill="none" stroke="var(--green)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.4" />}
      {/* Edge MA (simple moving average of equity) */}
      {showEdgeMA && (() => {
        const maLine = btPoints.map((_, i) => {
          const window = btPoints.slice(Math.max(0, i - 7), i + 1);
          const avg = window.reduce((s, v) => s + v, 0) / window.length;
          return `${toX(i).toFixed(1)},${toY(avg).toFixed(1)}`;
        }).join(' ');
        return <polyline points={maLine} fill="none" stroke="#808000" strokeWidth="0.8" strokeDasharray="3 2" opacity="0.6" />;
      })()}
      {/* Confidence bands (1SD around forward test) */}
      {showConfBands && fwdPoints.length > 1 && (() => {
        const sdOffset = 2.5;
        const upperBand = fwdPoints.map((p, i) => `${toX(fwdIdx + i).toFixed(1)},${toY(p + sdOffset).toFixed(1)}`).join(' ');
        const lowerBand = fwdPoints.map((p, i) => `${toX(fwdIdx + i).toFixed(1)},${toY(p - sdOffset).toFixed(1)}`).join(' ');
        return (
          <>
            <polyline points={upperBand} fill="none" stroke={EQ_BT_COLOR} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" />
            <polyline points={lowerBand} fill="none" stroke={EQ_BT_COLOR} strokeWidth="0.5" strokeDasharray="2 2" opacity="0.3" />
          </>
        );
      })()}
      {/* BT gradient fill */}
      <polygon points={buildFill(btPoints.slice(0, fwdIdx + 1), 0)} fill={`url(#btG${uid})`} />
      {/* FWD gradient fill */}
      <polygon points={buildFill(fwdPoints, fwdIdx)} fill={`url(#fwG${uid})`} />
      {/* FWD boundary */}
      <line x1={bndX} y1="0" x2={bndX} y2={h} stroke={EQ_FWD_COLOR} strokeWidth="0.5" strokeDasharray="3 2" opacity="0.5" />
      {/* BT line */}
      <polyline points={buildLine(btPoints.slice(0, fwdIdx + 1), 0)} fill="none" stroke={EQ_BT_COLOR} strokeWidth="1.5" />
      {/* FWD line */}
      <polyline points={buildLine(fwdPoints, fwdIdx)} fill="none" stroke={EQ_FWD_COLOR} strokeWidth="1.5" />
      {/* Live alerts — overlaid on FWD x-range, shows slippage */}
      {livePoints.length > 0 && <polyline points={buildLine(livePoints, fwdIdx)} fill="none" stroke={EQ_LIVE_COLOR} strokeWidth="1.5" />}
    </svg>
  );
}

// SD values from enriched strategy API (sigma_fwd, sigma_alert)
function getStrategySD(strat: Strategy): { fwd: number; alert: number } {
  return { fwd: strat.sigmaFwd, alert: strat.sigmaAlert };
}

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

const PULSE_CSS = `@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(2.2); opacity: 0; } }`;

export default function StrategiesPage() {
  // ---- API Hooks ----
  const { data: apiData, isLoading, error } = useStrategies();
  const deleteMut = useDeleteStrategy();
  const dupMut = useDuplicateStrategy();
  const bulkDeleteMut = useBulkDeleteStrategies();

  // Bulk refresh — uses plain fetch to avoid production bundler issues
  const [refreshing, setRefreshing] = useState(false);
  const [refreshCount, setRefreshCount] = useState(0);
  const [refreshTotal, setRefreshTotal] = useState(0);

  useEffect(() => {
    const id = 'strategies-pulse-css';
    if (!document.getElementById(id)) {
      const s = document.createElement('style'); s.id = id; s.textContent = PULSE_CSS; document.head.appendChild(s);
    }
  }, []);

  // Map API data to V5 Strategy interface
  const strategies: Strategy[] = useMemo(() => {
    if (!apiData) return [];
    return apiData.map(apiToStrategy);
  }, [apiData]);

  // Filter state (must be before any early returns — React hooks rule)
  const [tickerFilter, setTickerFilter] = useState('All');
  const [directionFilter, setDirectionFilter] = useState('All');
  const [tagFilter, setTagFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [tqFilter, setTqFilter] = useState('None');
  const [dataView, setDataView] = useState('Strategy Default');
  const [sortBy, setSortBy] = useState('Newest First');
  const [eqXAxis, setEqXAxis] = useState<'time' | 'trade'>('time');
  const [eqShowHWM, setEqShowHWM] = useState(true);
  const [eqShowEdgeMA, setEqShowEdgeMA] = useState(false);
  const [eqShowConfBands, setEqShowConfBands] = useState(false);
  const [kpiMode, setKpiMode] = useState('Overall');
  const [chartHeight, setChartHeight] = useState(64);
  // Strategy Health drawer — opens when a badge is clicked. We track the
  // strategy whose drawer is open (or null when closed).
  const [healthDrawerFor, setHealthDrawerFor] = useState<string | null>(null);
  // Health-aggregate filter: 'all' shows everything; otherwise filter cards
  // to only strategies whose health.severity matches.
  const [healthFilter, setHealthFilter] = useState<'all' | HealthSeverity>('all');

  // Bulk select (always available, no select mode toggle needed)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Delete confirmation
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false);

  // Derived filter options
  const allTickers = useMemo(() => Array.from(new Set(strategies.map((s) => s.symbol))).sort(), [strategies]);
  const allTags = useMemo(() => Array.from(new Set(strategies.flatMap((s) => s.tags))).sort(), [strategies]);
  const allStatuses = ['On Track', 'Outperforming', 'Underperforming', 'Insufficient Data'];

  // Filtered + sorted strategies
  const filteredStrategies = useMemo(() => {
    let result = [...strategies];
    if (tickerFilter !== 'All') result = result.filter((s) => s.symbol === tickerFilter);
    if (directionFilter !== 'All') result = result.filter((s) => s.direction === directionFilter);
    if (tagFilter !== 'All') result = result.filter((s) => s.tags.includes(tagFilter));
    if (statusFilter !== 'All') result = result.filter((s) => s.status === statusFilter);
    if (healthFilter !== 'all') {
      // Treat strategies with no health field as 'healthy'
      result = result.filter((s) => (s.health?.severity || 'healthy') === healthFilter);
    }

    // Sort
    switch (sortBy) {
      case 'Newest First': result.sort((a, b) => parseInt(b.id) - parseInt(a.id)); break;
      case 'Oldest First': result.sort((a, b) => parseInt(a.id) - parseInt(b.id)); break;
      case 'Name A-Z': result.sort((a, b) => a.name.localeCompare(b.name)); break;
      case 'Win Rate (High)': result.sort((a, b) => b.winRate - a.winRate); break;
      case 'Profit Factor (High)': result.sort((a, b) => b.pf - a.pf); break;
      case 'Daily R (High)': result.sort((a, b) => b.dailyR - a.dailyR); break;
      case 'Max DD (Best)': result.sort((a, b) => b.maxDD - a.maxDD); break;
    }
    return result;
  }, [strategies, tickerFilter, directionFilter, tagFilter, statusFilter, healthFilter, sortBy]);

  // Health aggregate counts (for the chip row above the filters)
  const healthCounts = useMemo(() => {
    const counts: Record<HealthSeverity, number> = {
      healthy: 0, minor: 0, action: 0, broken: 0,
    };
    for (const s of strategies) {
      const sev = (s.health?.severity || 'healthy') as HealthSeverity;
      counts[sev]++;
    }
    return counts;
  }, [strategies]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    setSelectedIds(new Set(filteredStrategies.map((s) => s.id)));
  };

  const clearSelection = () => {
    setSelectedIds(new Set());
  };

  const selectStyle = {
    background: 'var(--bg-input)',
    border: '1px solid var(--border)',
    color: 'var(--text-primary)',
    padding: '6px 12px',
    borderRadius: '8px',
    fontSize: '0.875rem',
  };

  const btnSecondary = {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    cursor: 'pointer' as const,
  };

  const btnPrimary = {
    background: 'var(--accent)',
    border: 'none',
    color: 'white',
    padding: '6px 14px',
    borderRadius: '8px',
    fontSize: '0.875rem',
    fontWeight: 500,
    cursor: 'pointer' as const,
  };

  // Loading / error states (after all hooks)
  if (isLoading) {
    return (
      <div style={{ padding: '32px' }}>
        <h1 className="text-xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>My Strategies</h1>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}><div className="animate-pulse space-y-3"><div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} /><div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} /><div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} /></div></Card>
          ))}
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div style={{ padding: '32px' }}>
        <h1 className="text-xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>My Strategies</h1>
        <Card><div className="text-center py-8" style={{ color: 'var(--red)' }}>Failed to load strategies.</div></Card>
      </div>
    );
  }

  return (
    <div>
      {/* ---- Header ---- */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">My Strategies</h1>
        <div className="flex gap-3">
          <button style={btnSecondary}>Reset All Alerts</button>
          <button
            style={{ ...btnSecondary, opacity: refreshing ? 0.6 : 1 }}
            disabled={refreshing}
            onClick={async () => {
              if (refreshing || strategies.length === 0) return;
              setRefreshing(true);
              setRefreshCount(0);
              setRefreshTotal(strategies.length);
              const token = localStorage.getItem('ror_access_token') || '';
              const base = process.env.NEXT_PUBLIC_API_URL || '';
              for (let i = 0; i < strategies.length; i++) {
                try {
                  await fetch(`${base}/api/strategies/${strategies[i].id}/refresh`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                  });
                } catch (e) { /* skip failures */ }
                setRefreshCount(i + 1);
              }
              setRefreshing(false);
              window.location.reload();
            }}
          >
            {refreshing ? `Refreshing ${refreshCount}/${refreshTotal}...` : 'Update All Data'}
          </button>
          <Link href="/strategy-builder"><button style={btnPrimary}>+ New Strategy</button></Link>
        </div>
      </div>

      {/* ---- Bulk action bar (shows when any cards are checked) ---- */}
      {selectedIds.size > 0 && (
        <div
          className="flex items-center gap-4 mb-4 px-4 py-3 rounded-lg"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
            {selectedIds.size} selected
          </span>
          <button
            className="text-sm px-3 py-1 rounded"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
            onClick={selectAll}
          >
            Select All
          </button>
          <button
            className="text-sm px-3 py-1 rounded"
            style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', cursor: 'pointer' }}
            onClick={() => setBulkDeleteConfirm(true)}
          >
            Delete Selected
          </button>
          <button style={btnPrimary} className="text-sm">
            Create Portfolio
          </button>
          <button style={btnSecondary} className="text-sm">
            Update Portfolio
          </button>
          <button
            style={{ ...btnSecondary, opacity: refreshing ? 0.6 : 1 }}
            className="text-sm"
            disabled={refreshing || selectedIds.size === 0}
            onClick={async () => {
              if (refreshing || selectedIds.size === 0) return;
              const ids = Array.from(selectedIds);
              setRefreshing(true);
              setRefreshCount(0);
              setRefreshTotal(ids.length);
              const token = localStorage.getItem('ror_access_token') || '';
              const base = process.env.NEXT_PUBLIC_API_URL || '';
              for (let i = 0; i < ids.length; i++) {
                try {
                  await fetch(`${base}/api/strategies/${ids[i]}/refresh`, {
                    method: 'POST',
                    headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                  });
                } catch (e) { /* skip failures */ }
                setRefreshCount(i + 1);
              }
              setRefreshing(false);
              window.location.reload();
            }}
          >
            {refreshing ? `Refreshing ${refreshCount}/${refreshTotal}...` : 'Update Data'}
          </button>
          <button
            style={{ ...btnSecondary, opacity: refreshing ? 0.6 : 1 }}
            className="text-sm"
            disabled={refreshing || selectedIds.size === 0}
            onClick={async () => {
              if (refreshing || selectedIds.size === 0) return;
              const ids = Array.from(selectedIds).map(Number);
              setRefreshing(true);
              setRefreshCount(0);
              setRefreshTotal(ids.length);
              const token = localStorage.getItem('ror_access_token') || '';
              const base = process.env.NEXT_PUBLIC_API_URL || '';
              try {
                const resp = await fetch(`${base}/api/strategies/run-hifi-pass2-bulk`, {
                  method: 'POST',
                  headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                  body: JSON.stringify({ strategy_ids: ids }),
                });
                const result = await resp.json();
                alert(`Hi-Fi Pass 2 complete\n\n` +
                      `Strategies: ${result.total_strategies}\n` +
                      `Entry timestamps refined: ${result.total_entries_refined}\n` +
                      `Exits refined: ${result.total_exits_refined}\n` +
                      `Failed: ${result.total_failed}`);
              } catch (e: any) {
                alert(`Hi-Fi Pass 2 failed: ${e?.message || e}`);
              }
              setRefreshing(false);
              window.location.reload();
            }}
          >
            Run Hi-Fi
          </button>
          <button style={btnSecondary} className="text-sm">
            Add Tag
          </button>
          <span className="flex-1" />
          <button
            className="text-sm px-3 py-1 rounded"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}
            onClick={clearSelection}
          >
            Clear
          </button>
        </div>
      )}

      {/* ---- Strategy Health drawer ---- */}
      {healthDrawerFor && (() => {
        const target = strategies.find((s) => s.id === healthDrawerFor);
        return (
          <StrategyHealthDrawer
            health={target?.health}
            isOpen={true}
            onClose={() => setHealthDrawerFor(null)}
            strategyName={target?.name}
          />
        );
      })()}

      {/* ---- Bulk delete confirmation modal ---- */}
      <Modal
        title="Delete Selected Strategies"
        isOpen={bulkDeleteConfirm}
        onClose={() => setBulkDeleteConfirm(false)}
        width="480px"
      >
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Delete {selectedIds.size} strategies? This cannot be undone.
        </p>
        <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
          {strategies
            .filter((s) => selectedIds.has(s.id))
            .map((s) => s.name)
            .join(', ')}
        </p>
        <div className="flex gap-3 justify-end">
          <button style={btnSecondary} onClick={() => setBulkDeleteConfirm(false)}>Cancel</button>
          <button
            style={{ ...btnPrimary, background: 'var(--red)' }}
            onClick={() => { bulkDeleteMut.mutate(Array.from(selectedIds).map(Number)); setBulkDeleteConfirm(false); clearSelection(); }}
          >
            Yes, Delete All
          </button>
        </div>
      </Modal>

      {/* ---- Health Aggregate Chip Row ----
           Shows total counts by severity across the strategy fleet.
           Each non-zero severity is clickable to filter the cards below to
           only that severity. Click an active filter again (or All) to clear.
           Hidden entirely when every strategy is healthy. */}
      {(healthCounts.minor + healthCounts.action + healthCounts.broken > 0) && (
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Strategy health:
          </span>
          {(['all', 'broken', 'action', 'minor', 'healthy'] as const).map((sev) => {
            // Build label + color per severity
            const labels: Record<typeof sev, string> = {
              all: `All (${strategies.length})`,
              healthy: `${healthCounts.healthy} healthy`,
              minor: `${healthCounts.minor} minor`,
              action: `${healthCounts.action} action`,
              broken: `${healthCounts.broken} broken`,
            };
            const colors: Record<typeof sev, { bg: string; fg: string }> = {
              all:     { bg: 'var(--bg-input)', fg: 'var(--text-secondary)' },
              healthy: { bg: 'var(--green-muted)', fg: 'var(--green)' },
              minor:   { bg: 'var(--orange-muted)', fg: 'var(--orange)' },
              action:  { bg: 'var(--orange-muted)', fg: 'var(--orange)' },
              broken:  { bg: 'var(--red-muted)', fg: 'var(--red)' },
            };
            const isActive = healthFilter === sev;
            // Hide zero-count severities (except 'all', which always shows)
            if (sev !== 'all'
                && healthCounts[sev as HealthSeverity] === 0) {
              return null;
            }
            return (
              <button
                key={sev}
                type="button"
                onClick={() => setHealthFilter(sev === 'all' ? 'all' : sev as HealthSeverity)}
                className="text-xs px-2.5 py-1 rounded-full font-medium"
                style={{
                  background: isActive ? colors[sev].fg : colors[sev].bg,
                  color: isActive ? 'white' : colors[sev].fg,
                  border: `1px solid ${colors[sev].fg}40`,
                  cursor: 'pointer',
                }}
              >
                {labels[sev]}
              </button>
            );
          })}
          {healthFilter !== 'all' && (
            <button
              type="button"
              onClick={() => setHealthFilter('all')}
              className="text-xs px-2 py-1 rounded"
              style={{ color: 'var(--text-muted)', background: 'transparent', border: '1px solid var(--border)' }}
            >
              Clear filter
            </button>
          )}
        </div>
      )}

      {/* ---- Filters ---- */}
      <div className="grid grid-cols-3 lg:grid-cols-7 gap-3 mb-5">
        <select style={selectStyle} value={tickerFilter} onChange={(e) => setTickerFilter(e.target.value)}>
          <option value="All">Ticker: All</option>
          {allTickers.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={directionFilter} onChange={(e) => setDirectionFilter(e.target.value)}>
          <option value="All">Direction: All</option>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </select>
        <select style={selectStyle} value={tagFilter} onChange={(e) => setTagFilter(e.target.value)}>
          <option value="All">Tag: All</option>
          {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <select style={selectStyle} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="All">Status: All</option>
          {allStatuses.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select style={selectStyle} value={tqFilter} onChange={(e) => setTqFilter(e.target.value)}>
          <option value="None">TQ: None</option>
          <option value="ttp">TQ: Trade The Pool</option>
          <option value="ftmo">TQ: FTMO</option>
          <option value="topstep">TQ: Topstep</option>
          <option value="custom">TQ: My Custom Rules</option>
        </select>
        <select style={selectStyle} value={dataView} onChange={(e) => setDataView(e.target.value)}>
          {['Strategy Default', 'All Data', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 'Backtest Only', 'Forward Test Only'].map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <select style={selectStyle} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          {['Newest First', 'Oldest First', 'Name A-Z', 'Win Rate (High)', 'Profit Factor (High)', 'Daily R (High)', 'Max DD (Best)'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* Viewing preferences row */}
      <div className="flex items-center gap-4 mb-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <div className="flex items-center gap-1.5">
          <span>KPIs:</span>
          <select
            className="px-2 py-1 rounded text-[10px] font-medium"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            value={kpiMode}
            onChange={(e) => setKpiMode(e.target.value)}
          >
            <option value="Overall">Overall</option>
            <option value="BT vs FWD">Backtest vs Forward</option>
            <option value="FWD vs Alerts">Forward vs Alerts</option>
            <option value="BT vs Alerts">Backtest vs Alerts</option>
          </select>
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Chart Height:</span>
          {([
            { value: 48, label: 'S' },
            { value: 64, label: 'M' },
            { value: 96, label: 'L' },
            { value: 140, label: 'XL' },
          ]).map((opt) => (
            <button
              key={opt.value}
              onClick={() => setChartHeight(opt.value)}
              className="px-2 py-1 rounded font-medium"
              style={{
                background: chartHeight === opt.value ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: chartHeight === opt.value ? 'var(--accent)' : 'var(--text-muted)',
                border: chartHeight === opt.value ? '1px solid var(--accent)' : '1px solid var(--border)',
              }}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>X-axis:</span>
          {(['time', 'trade'] as const).map((mode) => (
            <button key={mode} onClick={() => setEqXAxis(mode)} className="px-2 py-1 rounded font-medium" style={{ background: eqXAxis === mode ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: eqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {mode === 'time' ? 'Time' : 'Trade #'}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>High Water Mark:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowHWM(v === 'On')} className="px-2 py-1 rounded font-medium" style={{ background: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowHWM ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Edge Check:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowEdgeMA(v === 'On')} className="px-2 py-1 rounded font-medium" style={{ background: (eqShowEdgeMA ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowEdgeMA ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowEdgeMA ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
        <div className="w-px h-4" style={{ background: 'var(--border)' }} />
        <div className="flex items-center gap-1.5">
          <span>Confidence Bands:</span>
          {(['On', 'Off'] as const).map((v) => (
            <button key={v} onClick={() => setEqShowConfBands(v === 'On')} className="px-2 py-1 rounded font-medium" style={{ background: (eqShowConfBands ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowConfBands ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowConfBands ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)' }}>
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* ---- Strategy count ---- */}
      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {filteredStrategies.length} strateg{filteredStrategies.length === 1 ? 'y' : 'ies'}
        {(tickerFilter !== 'All' || directionFilter !== 'All' || tagFilter !== 'All' || statusFilter !== 'All')
          ? ` (filtered from ${strategies.length})`
          : ''}
      </p>

      {/* ---- Empty state ---- */}
      {filteredStrategies.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {strategies.length === 0
                ? 'No strategies yet. Create your first strategy!'
                : 'No strategies match the current filters.'}
            </p>
            {strategies.length === 0 && (
              <button style={btnPrimary}>Go to Strategy Builder</button>
            )}
          </div>
        </Card>
      )}

      {/* ---- Strategy cards (2-column grid) ---- */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredStrategies.map((strat) => {
          const fwdDays = daysSince(strat.fwdSince);

          return (
            <Card key={strat.id}>
              {/* Name + status badge + monitored dot | SD top-right */}
              <div className="flex items-center gap-2 mb-1">
                {/* Pulsing dot if monitored */}
                {strat.monitored && (
                  <div className="relative flex-shrink-0 w-2.5 h-2.5">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ background: 'var(--green)' }} />
                    <div className="w-2.5 h-2.5 rounded-full absolute top-0 left-0" style={{ background: 'var(--green)', animation: 'pulse 2s ease-out infinite', opacity: 0.5 }} />
                  </div>
                )}
                <h3 className="font-semibold text-base">
                  <span style={{ color: 'var(--text-muted)', fontWeight: 400, marginRight: '0.4rem' }}>
                    #{strat.id}
                  </span>
                  {strat.name}
                </h3>
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{
                    color: statusColors[strat.status],
                    background: statusColors[strat.status] + '20',
                  }}
                >
                  {strat.status}
                </span>
                <StrategyHealthBadge
                  health={strat.health}
                  onOpenDrawer={() => setHealthDrawerFor(strat.id)}
                />
                <StrategyParityBadge parity={strat.parityStatus} />
                <span className="flex-1" />
                {strat.fwdTrades > 0 && (() => {
                  const { fwd, alert } = getStrategySD(strat);
                  const fmtSD = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}\u03c3`;
                  return (
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_FWD_COLOR, background: EQ_FWD_COLOR + '18' }}>
                        {fmtSD(fwd)}
                      </span>
                      {strat.alertTracking && (
                        <>
                          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>|</span>
                          <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ color: EQ_LIVE_COLOR, background: EQ_LIVE_COLOR + '18' }}>
                            {fmtSD(alert)}
                          </span>
                        </>
                      )}
                    </div>
                  );
                })()}
              </div>

              {/* Tags */}
              {strat.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1" style={{ marginTop: '-2px' }}>
                  {strat.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-0.5 rounded-full"
                      style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', fontSize: '0.72rem' }}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {/* Meta line: symbol · direction · timeframe · session · BT days (trades) · Fwd days (trades) · Alerts (count) · Portfolios (count) · updated */}
              {/* Matches the subtitle on the Strategy Detail page so counts are consistent across views. */}
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {strat.symbol} {strat.direction}
                <span> · {strat.timeframe}</span>
                {strat.session !== 'RTH' && (
                  <span style={{ color: '#9C27B0' }}> · {strat.session}</span>
                )}
                <span style={{ color: EQ_BT_COLOR }}> · BT {strat.btDays}d ({strat.trades.toLocaleString()})</span>
                <span style={{ color: EQ_FWD_COLOR }}> · Fwd {fwdDays}d ({strat.fwdTrades.toLocaleString()})</span>
                <span style={{ color: EQ_LIVE_COLOR }}> · Alerts ({strat.alertTrades.toLocaleString()})</span>
                <span style={{ color: strat.portfolioCount > 0 ? '#9C27B0' : 'var(--text-muted)' }}>
                  {' '}· Portfolios ({strat.portfolioCount})
                </span>
                <span> · Updated {timeAgo(strat.updatedAt)}</span>
              </p>
              {/* Fidelity badges — always visible row above the equity curve */}
              <div className="mb-2">
                <StrategyFidelityBadges strategy={strat} variant="compact" />
              </div>

              {/* Mini equity curve (3-segment per display settings V5) */}
              <div className="rounded-lg mb-2 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <MiniEquityCurve
                  equityCurveData={strat.equityCurveData}
                  fwdStartPct={strat.btDays / (strat.btDays + fwdDays)}
                  hasAlerts={strat.alertTracking}
                  showHWM={eqShowHWM}
                  showEdgeMA={eqShowEdgeMA}
                  showConfBands={eqShowConfBands}
                  height={chartHeight}
                />
              </div>
              {/* Legend */}
              <div className="flex gap-3 mb-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span><span style={{ color: EQ_BT_COLOR }}>{'\u2014'}</span> Backtest</span>
                <span><span style={{ color: EQ_FWD_COLOR }}>{'\u2014'}</span> Forward</span>
                {strat.alertTracking && <span><span style={{ color: EQ_LIVE_COLOR }}>{'\u2014'}</span> Alerts</span>}
              </div>

              {/* KPIs — mode-dependent */}
              {(() => {
                const fmt = (v: number | null, suffix = '') => v !== null ? `${v >= 0 && suffix !== '%' && suffix !== 'R' ? '+' : ''}${v.toFixed(suffix === '%' ? 1 : 2)}${suffix}` : '--';
                const fmtD = (a: number | null, b: number | null) => {
                  if (a === null || b === null) return '--';
                  const d = a - b;
                  const color = d > 0 ? 'var(--green)' : d < 0 ? 'var(--red)' : 'var(--text-muted)';
                  return <span style={{ color }}>{d >= 0 ? '+' : ''}{d.toFixed(1)}</span>;
                };

                const btTPD = strat.btDays > 0 ? strat.trades / strat.btDays : 0;
                const fwdTPD = fwdDays > 0 ? strat.fwdTrades / fwdDays : null;
                const alertTPD = strat.alertTrades > 0 && fwdDays > 0 ? strat.alertTrades / fwdDays : null;

                const btRow = { label: 'Backtest', color: EQ_BT_COLOR, wr: strat.winRate, pf: strat.pf, dr: strat.dailyR, roi: strat.dailyROI as number | null, tpd: btTPD as number | null, dd: strat.maxDD };
                const fwdRow = { label: 'Forward', color: EQ_FWD_COLOR, wr: strat.fwdWinRate, pf: strat.fwdPF, dr: null as number | null, roi: strat.fwdDailyROI, tpd: fwdTPD, dd: null as number | null };
                const alertRow = { label: 'Alerts', color: EQ_LIVE_COLOR, wr: strat.alertWinRate, pf: strat.alertPF, dr: strat.alertDailyR, roi: strat.alertDailyROI, tpd: alertTPD, dd: strat.alertMaxDD };

                if (kpiMode === 'Overall') {
                  return (
                    <div className="grid grid-cols-6 gap-2 mb-2">
                      {[
                        { label: 'WR', value: `${strat.winRate.toFixed(1)}%` },
                        { label: 'PF', value: strat.pf.toFixed(2) },
                        { label: 'Daily R', value: `${strat.dailyR >= 0 ? '+' : ''}${strat.dailyR.toFixed(2)}` },
                        { label: 'Daily ROI', value: `${strat.dailyROI >= 0 ? '+' : ''}${strat.dailyROI.toFixed(2)}%` },
                        { label: 'TPD', value: btTPD.toFixed(1) },
                        { label: 'Max DD', value: `${strat.maxDD.toFixed(1)}R` },
                      ].map((kpi, j) => (
                        <div key={j}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                          <p className="text-sm font-medium">{kpi.value}</p>
                        </div>
                      ))}
                    </div>
                  );
                }

                // Comparison table mode
                const rowA = kpiMode === 'FWD vs Alerts' ? fwdRow : btRow;
                const rowB = kpiMode === 'BT vs FWD' ? fwdRow : alertRow;

                return (
                  <div className="mb-2 rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)' }}>
                    {/* Header */}
                    <div className="grid grid-cols-7 text-[10px] font-medium px-2 py-1.5" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
                      <span></span><span>WR</span><span>PF</span><span>Daily R</span><span>Daily ROI</span><span>TPD</span><span>Max DD</span>
                    </div>
                    {/* Row A */}
                    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
                      <span className="text-[10px] font-medium" style={{ color: rowA.color }}>{rowA.label}</span>
                      <span>{rowA.wr !== null ? `${rowA.wr.toFixed(1)}%` : '--'}</span>
                      <span>{rowA.pf !== null ? rowA.pf.toFixed(2) : '--'}</span>
                      <span>{rowA.dr !== null ? fmt(rowA.dr) : '--'}</span>
                      <span>{rowA.roi !== null ? `${rowA.roi.toFixed(2)}%` : '--'}</span>
                      <span>{rowA.tpd !== null ? rowA.tpd.toFixed(1) : '--'}</span>
                      <span>{rowA.dd !== null ? `${rowA.dd.toFixed(1)}R` : '--'}</span>
                    </div>
                    {/* Row B */}
                    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)' }}>
                      <span className="text-[10px] font-medium" style={{ color: rowB.color }}>{rowB.label}</span>
                      <span>{rowB.wr !== null ? `${rowB.wr.toFixed(1)}%` : '--'}</span>
                      <span>{rowB.pf !== null ? rowB.pf.toFixed(2) : '--'}</span>
                      <span>{rowB.dr !== null ? fmt(rowB.dr) : '--'}</span>
                      <span>{rowB.roi !== null ? `${rowB.roi.toFixed(2)}%` : '--'}</span>
                      <span>{rowB.tpd !== null ? rowB.tpd.toFixed(1) : '--'}</span>
                      <span>{rowB.dd !== null ? `${rowB.dd.toFixed(1)}R` : '--'}</span>
                    </div>
                    {/* Delta row */}
                    <div className="grid grid-cols-7 text-xs px-2 py-1.5 border-t" style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
                      <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{'\u0394'}</span>
                      <span>{fmtD(rowB.wr, rowA.wr)}</span>
                      <span>{fmtD(rowB.pf, rowA.pf)}</span>
                      <span>{fmtD(rowB.dr, rowA.dr)}</span>
                      <span>{fmtD(rowB.roi, rowA.roi)}</span>
                      <span>{fmtD(rowB.tpd, rowA.tpd)}</span>
                      <span>{fmtD(rowB.dd, rowA.dd)}</span>
                    </div>
                  </div>
                );
              })()}

              {/* Strategy variables — pack-aware display */}
              <div className="space-y-1.5 mt-2">
                {/* Row 1: Entry */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>entry:</span>
                  {(() => {
                    const match = strat.entry.match(/^(\[[A-Z]+\])\s*(.+)$/);
                    const execTag = match?.[1] || '';
                    const rest = match?.[2] || strat.entry;
                    return (
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {execTag && <span className="font-medium px-1 py-0.5 rounded-full mr-1" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20', fontSize: '10px' }}>{execTag}</span>}
                        {rest}
                      </span>
                    );
                  })()}
                </div>
                {/* Row 2: Exit(s) */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>exit:</span>
                  {strat.exit.map((ex, i) => {
                    const match = ex.match(/^(\[[A-Z]+\])\s*(.+)$/);
                    const execTag = match?.[1] || '';
                    const rest = match?.[2] || ex;
                    return (
                      <span key={i} className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
                        {execTag && <span className="font-medium px-1 py-0.5 rounded-full mr-1" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20', fontSize: '10px' }}>{execTag}</span>}
                        {rest}
                        {i < strat.exit.length - 1 && <span style={{ color: 'var(--text-muted)' }}>{' , '}</span>}
                      </span>
                    );
                  })}
                </div>
                {/* Row 3: Stop + Target */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>stop:</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--red)', background: 'var(--red)' + '18' }}>
                    {strat.stop}
                  </span>
                  <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>target:</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green)' + '18' }}>
                    {strat.target}
                  </span>
                  {strat.timeExit && (
                    <>
                      <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>time exit:</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange)' + '18' }}>
                        {strat.timeExit}
                      </span>
                    </>
                  )}
                </div>
                {/* Row 4: Confluence conditions with fidelity badges */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>confluence:</span>
                  {strat.confluence.length > 0 ? strat.confluence.map((c) => (
                    <span key={c} className="text-[10px] font-mono flex items-center gap-1">
                      <span className="px-1 py-0.5 rounded-full font-medium" style={{ color: '#26C6DA', background: '#26C6DA20', fontSize: '9px' }}>[PB]</span>
                      <span className="px-1.5 py-0.5 rounded" style={{ color: 'var(--accent)', background: 'var(--accent)' + '15' }}>{c}</span>
                    </span>
                  )) : (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>none</span>
                  )}
                </div>
              </div>

              {/* Action buttons + alert toggle + checkbox */}
              <div className="flex items-center gap-2 mt-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                <Link
                  href={`/strategies/${strat.id}`}
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}
                >
                  View
                </Link>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                  Edit
                </button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}
                  onClick={() => dupMut.mutate(Number(strat.id))}>
                  Clone
                </button>
                <button
                  className="px-3 py-1.5 rounded text-xs"
                  style={{ background: 'var(--red-muted)', color: 'var(--red)', border: 'none', fontSize: '0.75rem', padding: '4px 12px', cursor: 'pointer' }}
                  onClick={() => setDeleteId(strat.id)}
                >
                  Delete
                </button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}>
                  Tags
                </button>
                <span className="flex-1" />
                {/* Alert tracking toggle removed — alerts always fire for any
                    monitored strategy. Webhook delivery is gated separately via
                    portfolio.webhook_group_id. */}
                <input
                  type="checkbox"
                  checked={selectedIds.has(strat.id)}
                  onChange={() => toggleSelect(strat.id)}
                  className="w-4 h-4 rounded cursor-pointer flex-shrink-0"
                  style={{ accentColor: 'var(--accent)' }}
                  title="Select for bulk actions"
                />
              </div>

              {/* Inline delete confirmation */}
              {deleteId === strat.id && (
                <div className="mt-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>
                    Delete &apos;{strat.name}&apos;? This cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button
                      style={{ ...btnPrimary, background: 'var(--red)', fontSize: '0.75rem', padding: '4px 12px' }}
                      onClick={() => { deleteMut.mutate(Number(deleteId)); setDeleteId(null); }}
                    >
                      Yes, Delete
                    </button>
                    <button
                      style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 12px' }}
                      onClick={() => setDeleteId(null)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
