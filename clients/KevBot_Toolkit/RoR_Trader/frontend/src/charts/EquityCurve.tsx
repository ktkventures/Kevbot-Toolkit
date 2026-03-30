'use client';

import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart,
} from 'recharts';

interface EquityPoint {
  trade_number: number;
  timestamp?: string;
  cumulative_r: number;
}

interface EquityCurveProps {
  data: EquityPoint[];
  /** Index where backtest ends and forward test begins */
  boundaryIndex?: number | null;
  /** Index where forward test ends and live alerts begin */
  alertBoundaryIndex?: number | null;
  height?: number;
  showZeroLine?: boolean;
  showHWM?: boolean;
  xAxis?: 'trade' | 'time';
  mini?: boolean;
}

export default function EquityCurve({
  data,
  boundaryIndex,
  alertBoundaryIndex,
  height = 300,
  showZeroLine = true,
  showHWM = false,
  xAxis = 'trade',
  mini = false,
}: EquityCurveProps) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    if (xAxis === 'time') {
      // Per-day mode: group trades by exit date, sum daily R, build cumulative
      const dailyR = new Map<string, { label: string; ts: number; totalR: number }>();
      for (const pt of data) {
        if (!pt.timestamp) continue;
        const d = new Date(pt.timestamp);
        const dateKey = d.toISOString().slice(0, 10); // YYYY-MM-DD for sorting
        const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        // r_multiple for this trade = difference from previous cumulative
        const existing = dailyR.get(dateKey);
        if (existing) {
          // This trade's individual R = current cumulative - previous point's cumulative
          // But we only have cumulative_r, so daily total = last cumulative of this day
          dailyR.set(dateKey, { label, ts: d.getTime(), totalR: pt.cumulative_r });
        } else {
          dailyR.set(dateKey, { label, ts: d.getTime(), totalR: pt.cumulative_r });
        }
      }
      // Sort by date key (YYYY-MM-DD sorts correctly)
      const sorted = Array.from(dailyR.entries()).sort((a, b) => a[0].localeCompare(b[0]));
      return sorted.map(([, { label, totalR }]) => ({
        x: label,
        r: totalR,
        bt: totalR,
        fwd: null,
        live: null,
        btBridge: null,
        fwdBridge: null,
      }));
    }

    // Per-trade mode: sequential trade numbers
    return data.map((pt, i) => {
      let segment: 'backtest' | 'forward' | 'live' = 'backtest';
      if (boundaryIndex != null && i >= boundaryIndex) segment = 'forward';
      if (alertBoundaryIndex != null && i >= alertBoundaryIndex) segment = 'live';

      return {
        x: pt.trade_number,
        r: pt.cumulative_r,
        bt: segment === 'backtest' ? pt.cumulative_r : null,
        fwd: segment === 'forward' ? pt.cumulative_r : null,
        live: segment === 'live' ? pt.cumulative_r : null,
        btBridge: segment === 'forward' && i === (boundaryIndex ?? 0) ? pt.cumulative_r : null,
        fwdBridge: segment === 'live' && i === (alertBoundaryIndex ?? 0) ? pt.cumulative_r : null,
      };
    });
  }, [data, boundaryIndex, alertBoundaryIndex, xAxis]);

  // HWM line
  const hwm = useMemo(() => {
    if (!showHWM || chartData.length === 0) return [];
    let max = -Infinity;
    return chartData.map((pt) => {
      const val = pt.r ?? pt.bt ?? pt.fwd ?? pt.live ?? 0;
      max = Math.max(max, val);
      return { x: pt.x, hwm: max };
    });
  }, [chartData, showHWM]);

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No trade data
      </div>
    );
  }

  // Debug: log chartData to diagnose per-day doubling issue
  if (xAxis === 'time') {
    console.log('[EquityCurve] Per-day chartData:', chartData.length, 'points', chartData.slice(0, 5), '...', chartData.slice(-3));
    // Check for duplicate x values
    const xVals = chartData.map((d: any) => d.x);
    const uniqueX = new Set(xVals);
    if (xVals.length !== uniqueX.size) {
      console.warn('[EquityCurve] DUPLICATE x values detected!', xVals.length, 'total vs', uniqueX.size, 'unique');
      // Log the duplicates
      const counts = new Map<string, number>();
      for (const x of xVals) { counts.set(x, (counts.get(x) || 0) + 1); }
      const dupes = Array.from(counts.entries()).filter(([, c]) => c > 1);
      console.warn('[EquityCurve] Duplicates:', dupes);
    }
  }

  if (mini) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <defs>
            <linearGradient id="btGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2196F3" stopOpacity={0.15} />
              <stop offset="100%" stopColor="#2196F3" stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type="monotone" dataKey="r" stroke="#2196F3" strokeWidth={1.5}
            fill="url(#btGrad)" dot={false} isAnimationActive={false}
          />
          {showZeroLine && <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="3 3" strokeOpacity={0.3} />}
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="btGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2196F3" stopOpacity={0.12} />
            <stop offset="100%" stopColor="#2196F3" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="fwdGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FF9800" stopOpacity={0.12} />
            <stop offset="100%" stopColor="#FF9800" stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />

        <XAxis
          dataKey="x"
          type={xAxis === 'trade' ? 'number' : 'category'}
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          domain={xAxis === 'trade' ? ['dataMin', 'dataMax'] : undefined}
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
          formatter={(value: number | null) => value != null ? [`${value.toFixed(2)}R`, ''] : ['-', '']}
        />

        {showZeroLine && (
          <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.4} />
        )}

        {/* Per-day mode: single combined line (no segment splitting) */}
        {xAxis === 'time' ? (
          <Area
            type="monotone" dataKey="r" stroke="#2196F3" strokeWidth={2}
            fill="url(#btGradient)" dot={false} isAnimationActive={false}
          />
        ) : (
          <>
            {/* Backtest segment (blue) */}
            <Area
              type="monotone" dataKey="bt" stroke="#2196F3" strokeWidth={2}
              fill="url(#btGradient)" dot={false} isAnimationActive={false}
              connectNulls={false}
            />
            {/* Forward test segment (orange) */}
            <Area
              type="monotone" dataKey="fwd" stroke="#FF9800" strokeWidth={2}
              fill="url(#fwdGradient)" dot={false} isAnimationActive={false}
              connectNulls={false}
            />
            {/* Live alert segment (green, no fill) */}
            <Line
              type="monotone" dataKey="live" stroke="#4CAF50" strokeWidth={2}
              dot={false} isAnimationActive={false} connectNulls={false}
            />
          </>
        )}

        {/* HWM line */}
        {showHWM && hwm.length > 0 && (
          <Line
            data={hwm} type="stepAfter" dataKey="hwm"
            stroke="var(--text-secondary)" strokeWidth={1} strokeDasharray="4 2"
            dot={false} isAnimationActive={false} strokeOpacity={0.4}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
}
