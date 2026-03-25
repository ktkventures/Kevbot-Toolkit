'use client';

import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';

interface DistributionChartProps {
  /** Array of R-multiple values */
  values: number[];
  /** Number of histogram bins */
  bins?: number;
  height?: number;
  title?: string;
}

/**
 * R-distribution histogram — shows win/loss distribution.
 * Green bars for positive R, red for negative.
 */
export default function DistributionChart({
  values,
  bins = 20,
  height = 250,
}: DistributionChartProps) {
  const chartData = useMemo(() => {
    if (!values || values.length === 0) return [];

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const binWidth = range / bins;

    // Create bins
    const histogram: { center: number; count: number; isPositive: boolean }[] = [];
    for (let i = 0; i < bins; i++) {
      const low = min + i * binWidth;
      const high = low + binWidth;
      const center = (low + high) / 2;
      const count = values.filter((v) => v >= low && (i === bins - 1 ? v <= high : v < high)).length;
      histogram.push({ center: Math.round(center * 100) / 100, count, isPositive: center >= 0 });
    }

    return histogram;
  }, [values, bins]);

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No trade data
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />
        <XAxis
          dataKey="center"
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          tickFormatter={(v: number) => `${v.toFixed(1)}R`}
        />
        <YAxis
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '8px', fontSize: '12px', color: 'var(--text-primary)',
          }}
          formatter={(value: number) => [value, 'Trades']}
          labelFormatter={(label: number) => `${label}R`}
        />
        <ReferenceLine x={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.4} />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.isPositive ? 'var(--green, #4CAF50)' : 'var(--red, #f44336)'}
              fillOpacity={0.7}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
