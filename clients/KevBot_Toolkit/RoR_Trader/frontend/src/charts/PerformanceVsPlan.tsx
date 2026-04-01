'use client';

/**
 * Performance vs Plan Chart
 *
 * Compares forward test actual performance against backtest-derived expectations.
 * Ported from Streamlit's compute_portfolio_benchmark() in portfolios.py.
 *
 * - Plan line: N × avg_r (expected cumulative R per trade)
 * - 1SD band: ±√(N × var_r) — 68% confidence, centered on plan
 * - 2SD band: ±2√(N × var_r) — 95% confidence, centered on plan
 * - Actual line: cumulative R of forward test trades, color-coded by deviation
 */

import { useMemo, useState } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface Trade {
  r_multiple: number;
  entry_time?: string;
  exit_time?: string;
}

interface AlertTrade {
  r: number | null;
  entryTime?: string;
  exitTime?: string;
}

interface PerformanceVsPlanProps {
  btTrades: Trade[];
  fwdTrades: Trade[];
  alertTrades?: AlertTrade[];
  height?: number;
  projectionMultiplier?: number;
  /** Controlled view mode from parent */
  viewMode?: 'forward' | 'alerts';
}

export default function PerformanceVsPlan({
  btTrades,
  fwdTrades,
  alertTrades,
  height = 280,
  projectionMultiplier = 1.4,
  viewMode: controlledViewMode,
}: PerformanceVsPlanProps) {
  const [localViewMode, setLocalViewMode] = useState<'forward' | 'alerts'>('forward');
  const hasAlertData = useMemo(() => (alertTrades || []).filter(a => a.r != null).length >= 3, [alertTrades]);
  const activeMode = hasAlertData ? (controlledViewMode ?? localViewMode) : 'forward';

  const chartData = useMemo(() => {
    if (btTrades.length < 10 || fwdTrades.length < 3) return null;

    // Step 1: Compute R distribution from backtest trades
    const btR = btTrades.map(t => t.r_multiple);
    const n = btR.length;
    const avgR = btR.reduce((s, r) => s + r, 0) / n;
    const varR = btR.reduce((s, r) => s + (r - avgR) ** 2, 0) / (n - 1);

    // Actual cumulative R (from FWD trades or Alert trades based on mode)
    const sourceTrades = activeMode === 'alerts' && alertTrades
      ? alertTrades.filter(a => a.r != null).sort((a, b) => {
          const aMs = a.entryTime ? new Date(a.entryTime).getTime() || 0 : 0;
          const bMs = b.entryTime ? new Date(b.entryTime).getTime() || 0 : 0;
          return aMs - bMs;
        })
      : fwdTrades;
    let cumActual = 0;
    const actualCum: number[] = [];
    for (const t of sourceTrades) {
      const r = 'r_multiple' in t ? (t as Trade).r_multiple : ((t as AlertTrade).r ?? 0);
      cumActual += r;
      actualCum.push(cumActual);
    }

    // Step 2: Build plan line and confidence bands centered on plan
    const actualTradeCount = actualCum.length || fwdTrades.length;
    const maxTrades = Math.ceil(actualTradeCount * projectionMultiplier);

    const points: any[] = [];
    for (let i = 0; i <= maxTrades; i++) {
      const plan = i * avgR;
      const std = Math.sqrt(i * varR);

      // For Recharts band rendering: we use stacked areas
      // band2sd_base = lower2sd, band2sd_height = upper2sd - lower2sd (the full 2SD band)
      // band1sd_base = lower1sd, band1sd_height = upper1sd - lower1sd (the full 1SD band)
      const upper2sd = plan + 2 * std;
      const lower2sd = plan - 2 * std;
      const upper1sd = plan + 1 * std;
      const lower1sd = plan - 1 * std;

      points.push({
        x: i,
        plan,
        actual: i > 0 && i <= actualCum.length ? actualCum[i - 1] : null,
        // For band rendering: store upper and lower directly
        upper2sd,
        lower2sd,
        upper1sd,
        lower1sd,
        // Recharts trick: use base + range for stacked area bands
        band2sdBase: lower2sd,
        band2sdRange: upper2sd - lower2sd,
        band1sdBase: lower1sd,
        band1sdRange: upper1sd - lower1sd,
      });
    }

    // Summary metrics
    const lastActual = actualCum.length > 0 ? actualCum[actualCum.length - 1] : 0;
    const tradeCount = actualCum.length;
    const expectedAtN = tradeCount * avgR;
    const vsPlan = lastActual - expectedAtN;
    const stdAtN = Math.sqrt(tradeCount * varR);
    const deviationSD = stdAtN > 0 ? (lastActual - expectedAtN) / stdAtN : 0;

    let status: 'on_track' | 'outperforming' | 'underperforming' | 'severe' = 'on_track';
    if (deviationSD > 1) status = 'outperforming';
    else if (deviationSD < -1 && deviationSD >= -2) status = 'underperforming';
    else if (deviationSD < -2) status = 'severe';

    return { points, summary: { trades: tradeCount, actual: lastActual, expected: expectedAtN, vsPlan, deviationSD, status } };
  }, [btTrades, fwdTrades, alertTrades, activeMode, projectionMultiplier]);

  const statusColors: Record<string, string> = {
    on_track: '#4CAF50',
    outperforming: '#2196F3',
    underperforming: '#FF9800',
    severe: '#F44336',
  };
  const statusLabels: Record<string, string> = {
    on_track: 'On Track',
    outperforming: 'Outperforming',
    underperforming: 'Below Plan',
    severe: 'Severe Deviation',
  };

  if (!chartData) return null;

  const { points, summary } = chartData;
  const statusColor = statusColors[summary.status];

  return (
    <div>
      {/* Summary KPIs — left aligned */}
      <div className="flex items-center gap-6 mb-3 flex-wrap">
        <div>
          <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>FWD Trades</span>
          <span className="text-sm font-semibold">{summary.trades}</span>
        </div>
        <div>
          <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>Actual</span>
          <span className="text-sm font-semibold" style={{ color: summary.actual >= 0 ? 'var(--green)' : 'var(--red)' }}>
            {summary.actual >= 0 ? '+' : ''}{summary.actual.toFixed(2)}R
          </span>
        </div>
        <div>
          <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>Expected</span>
          <span className="text-sm font-semibold">
            {summary.expected >= 0 ? '+' : ''}{summary.expected.toFixed(2)}R
          </span>
        </div>
        <div>
          <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>vs Plan</span>
          <span className="text-sm font-semibold" style={{ color: statusColor }}>
            {summary.vsPlan >= 0 ? '+' : ''}{summary.vsPlan.toFixed(2)}R
          </span>
        </div>
        <div>
          <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>Status</span>
          <span
            className="text-xs font-medium px-2 py-0.5 rounded-full"
            style={{ color: statusColor, background: statusColor + '20' }}
          >
            {statusLabels[summary.status]} ({summary.deviationSD >= 0 ? '+' : ''}{summary.deviationSD.toFixed(1)}σ)
          </span>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={points} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
          <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />

          <XAxis
            dataKey="x"
            type="number"
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
            tickLine={false}
            label={{ value: 'Trade #', position: 'insideBottom', offset: -8, fill: 'var(--text-muted)', fontSize: 10 }}
            domain={[0, 'dataMax']}
          />
          <YAxis
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
            tickLine={false}
            tickFormatter={(v: number) => `${v.toFixed(1)}R`}
          />

          <Tooltip
            contentStyle={{
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: '8px', fontSize: '12px', color: 'var(--text-primary)',
            }}
            formatter={(value: number | null, name: string) => {
              if (value == null) return ['-', ''];
              const labels: Record<string, string> = { plan: 'Plan', actual: 'Actual', upper1sd: '+1σ', lower1sd: '-1σ', upper2sd: '+2σ', lower2sd: '-2σ' };
              return [`${value.toFixed(2)}R`, labels[name] || ''];
            }}
            labelFormatter={(v: number) => `Trade #${v}`}
          />

          <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.3} />

          {/* 2SD band: stacked area — invisible base + visible range */}
          <Area
            type="monotone" dataKey="band2sdBase" stackId="band2sd"
            stroke="none" fill="transparent" isAnimationActive={false}
          />
          <Area
            type="monotone" dataKey="band2sdRange" stackId="band2sd"
            stroke="none" fill="rgba(33, 150, 243, 0.06)" isAnimationActive={false}
          />

          {/* 1SD band: stacked area — invisible base + visible range */}
          <Area
            type="monotone" dataKey="band1sdBase" stackId="band1sd"
            stroke="none" fill="transparent" isAnimationActive={false}
          />
          <Area
            type="monotone" dataKey="band1sdRange" stackId="band1sd"
            stroke="none" fill="rgba(33, 150, 243, 0.15)" isAnimationActive={false}
          />

          {/* Plan line (white dashed) */}
          <Line
            type="monotone" dataKey="plan"
            stroke="rgba(255,255,255,0.6)" strokeWidth={2} strokeDasharray="6 4"
            dot={false} isAnimationActive={false}
          />

          {/* Actual FWD line (color-coded by status) */}
          <Line
            type="monotone" dataKey="actual"
            stroke={statusColor} strokeWidth={2.5}
            dot={false} isAnimationActive={false}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 16, height: 2, background: statusColor }} /> Actual
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 16, height: 2, borderTop: '2px dashed rgba(255,255,255,0.6)' }} /> Plan
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 12, height: 8, background: 'rgba(33,150,243,0.15)', borderRadius: 2 }} /> 68% Band (1SD)
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 12, height: 8, background: 'rgba(33,150,243,0.06)', borderRadius: 2 }} /> 95% Band (2SD)
        </span>
      </div>
    </div>
  );
}
