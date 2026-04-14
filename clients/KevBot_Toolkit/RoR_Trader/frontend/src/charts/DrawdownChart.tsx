'use client';

import { useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface DrawdownPoint {
  exit_time?: string;
  drawdown_pct: number;
}

interface DrawdownChartProps {
  data: DrawdownPoint[];
  /** Optional max drawdown limit from requirement set (positive % value, rendered as negative) */
  maxDDLimit?: number | null;
  height?: number;
}

export default function DrawdownChart({ data, maxDDLimit, height = 220 }: DrawdownChartProps) {
  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map((pt, i) => ({
      x: i + 1,
      dd: pt.drawdown_pct,
      timestamp: pt.exit_time,
    }));
  }, [data]);

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No drawdown data
      </div>
    );
  }

  const vals = chartData.map((d) => d.dd);
  const minV = Math.min(...vals, maxDDLimit != null ? -maxDDLimit : 0);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--red)" stopOpacity={0.05} />
            <stop offset="100%" stopColor="var(--red)" stopOpacity={0.25} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />
        <XAxis
          dataKey="x"
          type="number"
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          domain={['dataMin', 'dataMax']}
        />
        <YAxis
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          tickFormatter={(v: number) => `${v.toFixed(1)}%`}
          domain={[Math.min(minV * 1.1, -0.5), 0]}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '8px', fontSize: '12px', color: 'var(--text-primary)',
          }}
          formatter={(value: number) => [`${value.toFixed(2)}%`, 'Drawdown']}
        />
        <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="4 4" strokeOpacity={0.4} />
        {maxDDLimit != null && (
          <ReferenceLine
            y={-maxDDLimit}
            stroke="var(--orange)"
            strokeDasharray="4 3"
            strokeOpacity={0.7}
            label={{
              value: `Limit: -${maxDDLimit}%`,
              position: 'insideTopRight',
              fill: 'var(--orange)',
              fontSize: 10,
            }}
          />
        )}
        <Area
          type="monotone"
          dataKey="dd"
          stroke="var(--red)"
          strokeWidth={1.5}
          fill="url(#ddGrad)"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
