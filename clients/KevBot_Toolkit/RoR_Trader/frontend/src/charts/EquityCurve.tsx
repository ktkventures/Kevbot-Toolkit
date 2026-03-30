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

    return data.map((pt, i) => {
      // Determine which segment this point belongs to
      let segment: 'backtest' | 'forward' | 'live' = 'backtest';
      if (boundaryIndex != null && i >= boundaryIndex) {
        segment = 'forward';
      }
      if (alertBoundaryIndex != null && i >= alertBoundaryIndex) {
        segment = 'live';
      }

      const timeLabel = pt.timestamp
        ? new Date(pt.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        : `#${i + 1}`;

      return {
        x: xAxis === 'trade' ? pt.trade_number : timeLabel,
        r: pt.cumulative_r,
        // Split into segment-specific fields for multi-color rendering
        bt: segment === 'backtest' ? pt.cumulative_r : null,
        fwd: segment === 'forward' ? pt.cumulative_r : null,
        live: segment === 'live' ? pt.cumulative_r : null,
        // Bridge point: duplicate last of prev segment as first of next
        btBridge: segment === 'forward' && i === (boundaryIndex ?? 0) ? pt.cumulative_r : null,
        fwdBridge: segment === 'live' && i === (alertBoundaryIndex ?? 0) ? pt.cumulative_r : null,
      };
    });
  }, [data, boundaryIndex, alertBoundaryIndex, xAxis]);

  // HWM line
  const hwm = useMemo(() => {
    if (!showHWM || data.length === 0) return [];
    let max = -Infinity;
    return data.map((pt, i) => {
      max = Math.max(max, pt.cumulative_r);
      const timeLabel = pt.timestamp
        ? new Date(pt.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        : `#${i + 1}`;
      return { x: xAxis === 'trade' ? pt.trade_number : timeLabel, hwm: max };
    });
  }, [data, showHWM, xAxis]);

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No trade data
      </div>
    );
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
