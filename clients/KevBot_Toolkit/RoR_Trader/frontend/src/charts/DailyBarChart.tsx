'use client';

import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts';

interface DailyBarPoint {
  date: string;
  value: number;
  /** Optional per-bar fill override */
  fill?: string;
}

interface ReferenceLineSpec {
  value: number;
  color: string;
  label?: string;
  dashed?: boolean;
}

interface DailyBarChartProps {
  data: DailyBarPoint[];
  referenceLines?: ReferenceLineSpec[];
  /** Coloring rule when no per-bar fill is set */
  colorRule?: 'pnl' | 'single';
  singleColor?: string;
  height?: number;
  yTickFormatter?: (v: number) => string;
}

export default function DailyBarChart({
  data,
  referenceLines = [],
  colorRule = 'pnl',
  singleColor = 'var(--accent)',
  height = 220,
  yTickFormatter,
}: DailyBarChartProps) {
  const chartData = useMemo(
    () => data.map((d) => ({ ...d })),
    [data],
  );

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No daily data
      </div>
    );
  }

  const fmtY = yTickFormatter ?? ((v: number) => `$${Math.round(v).toLocaleString()}`);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />
        <XAxis
          dataKey="date"
          tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          tickFormatter={fmtY}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '8px', fontSize: '12px', color: 'var(--text-primary)',
          }}
          formatter={(value: number) => [fmtY(value), '']}
        />
        <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.4} />
        {referenceLines.map((rl, i) => (
          <ReferenceLine
            key={i}
            y={rl.value}
            stroke={rl.color}
            strokeDasharray={rl.dashed === false ? undefined : '4 3'}
            strokeOpacity={0.7}
            label={rl.label ? {
              value: rl.label,
              position: 'insideTopRight',
              fill: rl.color,
              fontSize: 10,
            } : undefined}
          />
        ))}
        <Bar dataKey="value" isAnimationActive={false}>
          {chartData.map((d, i) => {
            const fill = d.fill
              ?? (colorRule === 'pnl'
                ? (d.value >= 0 ? 'var(--green)' : 'var(--red)')
                : singleColor);
            return <Cell key={i} fill={fill} />;
          })}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
