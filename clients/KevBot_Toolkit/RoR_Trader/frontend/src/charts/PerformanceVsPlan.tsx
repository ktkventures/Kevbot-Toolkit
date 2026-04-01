'use client';

/**
 * Performance vs Plan Chart
 *
 * Compares forward test actual performance against backtest-derived expectations.
 * Ported from Streamlit's compute_portfolio_benchmark() in portfolios.py.
 *
 * - Plan line: N × avg_r (expected cumulative R per trade)
 * - 1SD band: ±√(N × var_r) — 68% confidence
 * - 2SD band: ±2√(N × var_r) — 95% confidence
 * - Actual line: cumulative R of forward test trades, color-coded by deviation
 */

import { useMemo } from 'react';
import {
  AreaChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface Trade {
  r_multiple: number;
  entry_time?: string;
  exit_time?: string;
}

interface PerformanceVsPlanProps {
  /** Backtest trades — used to compute avg_r, std_r for the plan line */
  btTrades: Trade[];
  /** Forward test trades — the "actual" line */
  fwdTrades: Trade[];
  height?: number;
  /** How far to project the plan line beyond current FWD trades (in trade count) */
  projectionMultiplier?: number;
}

export default function PerformanceVsPlan({
  btTrades,
  fwdTrades,
  height = 280,
  projectionMultiplier = 1.4,
}: PerformanceVsPlanProps) {
  const chartData = useMemo(() => {
    if (btTrades.length < 10 || fwdTrades.length < 3) return null;

    // Step 1: Compute R distribution from backtest trades
    const btR = btTrades.map(t => t.r_multiple);
    const n = btR.length;
    const avgR = btR.reduce((s, r) => s + r, 0) / n;
    const varR = btR.reduce((s, r) => s + (r - avgR) ** 2, 0) / (n - 1);

    // Step 2: Build plan line and confidence bands
    const maxTrades = Math.ceil(fwdTrades.length * projectionMultiplier);
    const points: any[] = [];

    // Actual cumulative FWD R
    let cumActual = 0;
    const actualCum: number[] = [];
    for (const t of fwdTrades) {
      cumActual += t.r_multiple;
      actualCum.push(cumActual);
    }

    for (let i = 0; i <= maxTrades; i++) {
      const tradeNum = i;
      const plan = tradeNum * avgR;
      const std = Math.sqrt(tradeNum * varR);

      const point: any = {
        x: tradeNum,
        plan,
        upper2sd: plan + 2 * std,
        lower2sd: plan - 2 * std,
        upper1sd: plan + 1 * std,
        lower1sd: plan - 1 * std,
        actual: i > 0 && i <= fwdTrades.length ? actualCum[i - 1] : null,
      };

      points.push(point);
    }

    // Compute summary metrics
    const lastActual = actualCum.length > 0 ? actualCum[actualCum.length - 1] : 0;
    const expectedAtN = fwdTrades.length * avgR;
    const vsPlan = lastActual - expectedAtN;
    const stdAtN = Math.sqrt(fwdTrades.length * varR);
    const deviationSD = stdAtN > 0 ? (lastActual - expectedAtN) / stdAtN : 0;

    let status: 'on_track' | 'outperforming' | 'underperforming' | 'severe' = 'on_track';
    if (deviationSD > 1) status = 'outperforming';
    else if (deviationSD < -1 && deviationSD >= -2) status = 'underperforming';
    else if (deviationSD < -2) status = 'severe';

    return { points, summary: { trades: fwdTrades.length, actual: lastActual, expected: expectedAtN, vsPlan, deviationSD, status } };
  }, [btTrades, fwdTrades, projectionMultiplier]);

  if (!chartData) {
    return null; // Not enough data — don't render
  }

  const { points, summary } = chartData;

  const statusColors = {
    on_track: '#4CAF50',
    outperforming: '#2196F3',
    underperforming: '#FF9800',
    severe: '#F44336',
  };
  const statusLabels = {
    on_track: 'On Track',
    outperforming: 'Outperforming',
    underperforming: 'Below Plan',
    severe: 'Severe Deviation',
  };
  const statusColor = statusColors[summary.status];

  return (
    <div>
      {/* Summary KPIs */}
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
        <AreaChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
          <defs>
            <linearGradient id="band2sdGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2196F3" stopOpacity={0.08} />
              <stop offset="100%" stopColor="#2196F3" stopOpacity={0.04} />
            </linearGradient>
            <linearGradient id="band1sdGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2196F3" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#2196F3" stopOpacity={0.08} />
            </linearGradient>
          </defs>

          <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />

          <XAxis
            dataKey="x"
            type="number"
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
            tickLine={false}
            label={{ value: 'Trade #', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 10 }}
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
              if (value == null) return ['-', name];
              const labels: Record<string, string> = { plan: 'Plan', actual: 'Actual', upper1sd: '1σ Upper', lower1sd: '1σ Lower', upper2sd: '2σ Upper', lower2sd: '2σ Lower' };
              return [`${value.toFixed(2)}R`, labels[name] || name];
            }}
          />

          <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.3} />

          {/* 2SD band (outermost, lightest) */}
          <Area type="monotone" dataKey="upper2sd" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="lower2sd" stroke="none" fill="url(#band2sdGrad)" isAnimationActive={false} />

          {/* 1SD band (inner, slightly darker) */}
          <Area type="monotone" dataKey="upper1sd" stroke="none" fill="transparent" isAnimationActive={false} />
          <Area type="monotone" dataKey="lower1sd" stroke="none" fill="url(#band1sdGrad)" isAnimationActive={false} />

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
        </AreaChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 16, height: 2, background: statusColor }} /> Actual
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 16, height: 2, background: 'rgba(255,255,255,0.6)', borderTop: '1px dashed rgba(255,255,255,0.6)' }} /> Plan
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 12, height: 8, background: 'rgba(33,150,243,0.18)', borderRadius: 2 }} /> 68% Band (1SD)
        </span>
        <span className="flex items-center gap-1">
          <span style={{ display: 'inline-block', width: 12, height: 8, background: 'rgba(33,150,243,0.08)', borderRadius: 2 }} /> 95% Band (2SD)
        </span>
      </div>
    </div>
  );
}
