'use client';

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts';

interface BalancePoint {
  date: string;
  balance: number;
}

interface BalanceHistoryChartProps {
  data: BalancePoint[];
  height?: number;
}

export default function BalanceHistoryChart({ data, height = 200 }: BalanceHistoryChartProps) {
  if (data.length < 2) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '12px' }}>
        No balance history yet
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="balHistGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.15} />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
          </linearGradient>
        </defs>
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
          tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: '8px', fontSize: '12px', color: 'var(--text-primary)',
          }}
          formatter={(value: number) => [`$${value.toFixed(2)}`, 'Balance']}
        />
        <Area
          type="monotone"
          dataKey="balance"
          stroke="var(--accent)"
          strokeWidth={2}
          fill="url(#balHistGrad)"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
