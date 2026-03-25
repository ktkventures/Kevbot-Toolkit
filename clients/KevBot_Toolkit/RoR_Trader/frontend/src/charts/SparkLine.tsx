'use client';

import { AreaChart, Area, ResponsiveContainer, ReferenceLine } from 'recharts';

interface SparkLineProps {
  /** Array of cumulative R values */
  data: number[];
  /** Height in pixels */
  height?: number;
  /** Index where backtest ends (show forward test in orange after) */
  boundaryIndex?: number | null;
  /** Line color (default blue) */
  color?: string;
}

/**
 * Tiny inline equity sparkline for strategy/portfolio cards.
 * No axes, no labels — just the shape of the equity curve.
 */
export default function SparkLine({
  data,
  height = 60,
  boundaryIndex,
  color = '#2196F3',
}: SparkLineProps) {
  if (!data || data.length < 2) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-secondary)',
          fontSize: '11px',
        }}
      >
        No data
      </div>
    );
  }

  const chartData = data.map((r, i) => ({
    i,
    r,
    bt: boundaryIndex == null || i < boundaryIndex ? r : null,
    fwd: boundaryIndex != null && i >= boundaryIndex ? r : null,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <defs>
          <linearGradient id={`spark-grad-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.15} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="spark-grad-fwd" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FF9800" stopOpacity={0.15} />
            <stop offset="100%" stopColor="#FF9800" stopOpacity={0} />
          </linearGradient>
        </defs>

        <ReferenceLine y={0} stroke="var(--text-secondary)" strokeOpacity={0.15} />

        {boundaryIndex != null ? (
          <>
            <Area
              type="monotone" dataKey="bt" stroke={color} strokeWidth={1.5}
              fill={`url(#spark-grad-${color.replace('#', '')})`}
              dot={false} isAnimationActive={false} connectNulls={false}
            />
            <Area
              type="monotone" dataKey="fwd" stroke="#FF9800" strokeWidth={1.5}
              fill="url(#spark-grad-fwd)"
              dot={false} isAnimationActive={false} connectNulls={false}
            />
          </>
        ) : (
          <Area
            type="monotone" dataKey="r" stroke={color} strokeWidth={1.5}
            fill={`url(#spark-grad-${color.replace('#', '')})`}
            dot={false} isAnimationActive={false}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
}
