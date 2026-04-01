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
  showEdgeCheck?: boolean;
  xAxis?: 'trade' | 'time';
  mini?: boolean;
  /** Segment colors from display settings */
  btColor?: string;
  fwdColor?: string;
  liveColor?: string;
  /** Line style: solid (default), smooth (monotone), stepped */
  lineStyle?: 'solid' | 'smooth' | 'stepped';
  /** Show gradient fill under BT and FWD segments */
  showGradient?: boolean;
}

export default function EquityCurve({
  data,
  boundaryIndex,
  alertBoundaryIndex,
  height = 300,
  showZeroLine = true,
  showHWM = false,
  showEdgeCheck = false,
  xAxis = 'trade',
  mini = false,
  btColor = '#2196F3',
  fwdColor = '#FF9800',
  liveColor = '#4CAF50',
  lineStyle = 'solid',
  showGradient = true,
}: EquityCurveProps) {
  // Map lineStyle to Recharts curve type
  const curveType = lineStyle === 'stepped' ? 'stepAfter' : lineStyle === 'smooth' ? 'monotone' : 'linear';

  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    if (xAxis === 'time') {
      // Per-day mode: group trades by exit date, sum daily R, build cumulative
      const dailyR = new Map<string, { label: string; ts: number; totalR: number; idx: number }>();
      let tradeIdx = 0;
      for (const pt of data) {
        if (!pt.timestamp) { tradeIdx++; continue; }
        const d = new Date(pt.timestamp);
        if (isNaN(d.getTime())) { console.warn('[EquityCurve] Invalid timestamp:', pt.timestamp); tradeIdx++; continue; }
        const dateKey = d.toISOString().slice(0, 10);
        const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        dailyR.set(dateKey, { label, ts: d.getTime(), totalR: pt.cumulative_r, idx: tradeIdx });
        tradeIdx++;
      }
      const sorted = Array.from(dailyR.entries()).sort((a, b) => a[0].localeCompare(b[0]));

      return sorted.map(([, { label, totalR, idx }]) => {
        let segment: 'backtest' | 'forward' | 'live' = 'backtest';
        if (boundaryIndex != null && idx >= boundaryIndex) segment = 'forward';
        if (alertBoundaryIndex != null && idx >= alertBoundaryIndex) segment = 'live';
        return {
          x: label,
          r: totalR,
          bt: segment === 'backtest' ? totalR : null,
          fwd: segment === 'forward' ? totalR : null,
          live: segment === 'live' ? totalR : null,
          btBridge: segment === 'forward' && dailyR.size > 0 ? totalR : null,
          fwdBridge: segment === 'live' ? totalR : null,
        };
      });
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

  if (chartData.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No trade data
      </div>
    );
  }

  // Merge HWM into chartData
  const chartDataWithHwm = useMemo(() => {
    if (!showHWM || chartData.length === 0) return chartData;
    let max = -Infinity;
    return chartData.map((pt: any) => {
      const val = pt.r ?? pt.bt ?? pt.fwd ?? pt.live ?? 0;
      max = Math.max(max, val);
      return { ...pt, hwm: max };
    });
  }, [chartData, showHWM]);

  // Edge Check: 21-period moving average on cumulative R
  const chartDataFinal = useMemo(() => {
    if (!showEdgeCheck || chartDataWithHwm.length < 21) return chartDataWithHwm;
    const window = 21;
    return chartDataWithHwm.map((pt: any, i: number) => {
      if (i < window - 1) return { ...pt, edgeMA: null };
      let sum = 0;
      for (let j = i - window + 1; j <= i; j++) sum += (chartDataWithHwm[j].r ?? 0);
      return { ...pt, edgeMA: sum / window };
    });
  }, [chartDataWithHwm, showEdgeCheck]);

  if (mini) {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <defs>
            <linearGradient id="btGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={btColor} stopOpacity={0.15} />
              <stop offset="100%" stopColor={btColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type={curveType} dataKey="r" stroke={btColor} strokeWidth={1.5}
            fill={showGradient ? 'url(#btGrad)' : 'none'} dot={false} isAnimationActive={false}
          />
          {showZeroLine && <ReferenceLine y={0} stroke="var(--text-secondary)" strokeDasharray="3 3" strokeOpacity={0.3} />}
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  // Unique gradient IDs to avoid SVG conflicts
  const btGradId = `btGradient-${height}`;
  const fwdGradId = `fwdGradient-${height}`;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartDataFinal} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id={btGradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={btColor} stopOpacity={0.12} />
            <stop offset="100%" stopColor={btColor} stopOpacity={0} />
          </linearGradient>
          <linearGradient id={fwdGradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={fwdColor} stopOpacity={0.12} />
            <stop offset="100%" stopColor={fwdColor} stopOpacity={0} />
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

        {/* Backtest segment */}
        <Area
          type={curveType} dataKey="bt" stroke={btColor} strokeWidth={2}
          fill={showGradient ? `url(#${btGradId})` : 'none'} dot={false} isAnimationActive={false}
          connectNulls={false}
        />
        {/* Bridge: connect BT end to FWD start */}
        <Line type={curveType} dataKey="btBridge" stroke={btColor} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />

        {/* Forward test segment */}
        <Area
          type={curveType} dataKey="fwd" stroke={fwdColor} strokeWidth={2}
          fill={showGradient ? `url(#${fwdGradId})` : 'none'} dot={false} isAnimationActive={false}
          connectNulls={false}
        />
        {/* Bridge: connect FWD end to Live start */}
        <Line type={curveType} dataKey="fwdBridge" stroke={fwdColor} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />

        {/* Live alert segment (no fill) */}
        <Line
          type={curveType} dataKey="live" stroke={liveColor} strokeWidth={2}
          dot={false} isAnimationActive={false} connectNulls={false}
        />

        {/* HWM line */}
        {showHWM && (
          <Line
            type="stepAfter" dataKey="hwm"
            stroke="var(--text-secondary)" strokeWidth={1} strokeDasharray="4 2"
            dot={false} isAnimationActive={false} strokeOpacity={0.4}
          />
        )}

        {/* Edge Check: 21-period MA on cumulative R */}
        {showEdgeCheck && (
          <Line
            type="monotone" dataKey="edgeMA"
            stroke="#AB47BC" strokeWidth={1.5} strokeDasharray="6 3"
            dot={false} isAnimationActive={false} strokeOpacity={0.7}
          />
        )}
      </AreaChart>
    </ResponsiveContainer>
  );
}
