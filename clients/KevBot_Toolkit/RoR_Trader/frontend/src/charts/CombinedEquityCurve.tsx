'use client';

import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface StrategyOverlay {
  name: string;
  color: string;
  points: number[];
}

interface CombinedEquityCurveProps {
  combined: number[];
  strategyOverlays?: StrategyOverlay[];
  timestamps?: (string | null)[];
  height?: number;
  xAxis?: 'trade' | 'time';
  showZeroLine?: boolean;
}

const DEFAULT_STRATEGY_COLORS = [
  'var(--accent)', 'var(--green)', 'var(--orange)',
  '#8B5CF6', '#06B6D4', '#EC4899', '#14B8A6', '#F59E0B',
];

export default function CombinedEquityCurve({
  combined,
  strategyOverlays = [],
  timestamps,
  height = 320,
  xAxis = 'trade',
  showZeroLine = true,
}: CombinedEquityCurveProps) {
  const chartData = useMemo(() => {
    if (!combined || combined.length === 0) return [];
    return combined.map((val, i) => {
      const row: Record<string, any> = {
        x: xAxis === 'time' && timestamps?.[i]
          ? (() => {
              try { return new Date(timestamps[i] as string).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }); }
              catch { return i + 1; }
            })()
          : i + 1,
        combined: val,
      };
      for (const o of strategyOverlays) {
        row[`s_${o.name}`] = o.points[i] ?? null;
      }
      return row;
    });
  }, [combined, strategyOverlays, timestamps, xAxis]);

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No equity data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
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
          tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '8px', fontSize: '12px', color: 'var(--text-primary)',
          }}
          formatter={(value: number | null, name: string) => {
            if (value == null) return ['-', ''];
            const label = name === 'combined' ? 'Combined' : name.replace(/^s_/, '');
            return [`$${value.toFixed(0)}`, label];
          }}
        />
        {showZeroLine && <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.4} />}

        {/* Per-strategy dashed overlays (60% opacity) — drawn first so combined sits on top */}
        {strategyOverlays.map((o, i) => (
          <Line
            key={o.name}
            type="monotone"
            dataKey={`s_${o.name}`}
            stroke={o.color || DEFAULT_STRATEGY_COLORS[i % DEFAULT_STRATEGY_COLORS.length]}
            strokeWidth={1.5}
            strokeDasharray="4 3"
            strokeOpacity={0.6}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        ))}

        {/* Bold combined line */}
        <Line
          type="monotone"
          dataKey="combined"
          stroke="var(--text-primary)"
          strokeWidth={2.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
