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
  /** Alert trades as separate overlay data (green line on top of FWD) */
  alertOverlayData?: EquityPoint[];
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
  alertOverlayData,
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
      // Per-day mode: group trades by exit date, then fill non-trading days with flat lines
      const dailyR = new Map<string, { totalR: number; maxIdx: number }>();
      for (let i = 0; i < data.length; i++) {
        const pt = data[i];
        if (!pt.timestamp) continue;
        const d = new Date(pt.timestamp);
        if (isNaN(d.getTime())) { console.warn('[EquityCurve] Invalid timestamp:', pt.timestamp); continue; }
        const dateKey = d.toISOString().slice(0, 10);
        dailyR.set(dateKey, { totalR: pt.cumulative_r, maxIdx: i });
      }
      if (dailyR.size === 0) return [];

      // Get date range and fill ALL calendar days (weekdays only for equities)
      const sortedKeys = Array.from(dailyR.keys()).sort();
      const startDate = new Date(sortedKeys[0] + 'T00:00:00');
      const endDate = new Date(sortedKeys[sortedKeys.length - 1] + 'T00:00:00');
      const allDays: { dateKey: string; label: string; totalR: number; maxIdx: number }[] = [];
      let lastR = 0;
      let lastIdx = 0;
      const cursor = new Date(startDate);
      while (cursor <= endDate) {
        const dow = cursor.getDay();
        if (dow !== 0 && dow !== 6) { // Skip weekends
          const dk = cursor.toISOString().slice(0, 10);
          const label = cursor.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          const entry = dailyR.get(dk);
          if (entry) {
            lastR = entry.totalR;
            lastIdx = entry.maxIdx;
            allDays.push({ dateKey: dk, label, totalR: entry.totalR, maxIdx: entry.maxIdx });
          } else {
            // Non-trading day: carry forward last known cumulative R
            allDays.push({ dateKey: dk, label, totalR: lastR, maxIdx: lastIdx });
          }
        }
        cursor.setDate(cursor.getDate() + 1);
      }

      return allDays.map(({ label, totalR, maxIdx }, dayIdx) => {
        let segment: 'backtest' | 'forward' | 'live' = 'backtest';
        if (boundaryIndex != null && maxIdx >= boundaryIndex) segment = 'forward';
        if (alertBoundaryIndex != null && maxIdx >= alertBoundaryIndex) segment = 'live';

        // At segment transitions, include value in both segments for seamless connection
        let prevSegment: string | null = null;
        if (dayIdx > 0) {
          const pi = allDays[dayIdx - 1].maxIdx;
          prevSegment = 'backtest';
          if (boundaryIndex != null && pi >= boundaryIndex) prevSegment = 'forward';
          if (alertBoundaryIndex != null && pi >= alertBoundaryIndex) prevSegment = 'live';
        }
        const isBtToFwd = prevSegment === 'backtest' && segment === 'forward';
        const isFwdToLive = prevSegment === 'forward' && segment === 'live';

        return {
          x: label,
          r: totalR,
          bt: (segment === 'backtest' || isBtToFwd) ? totalR : null,
          fwd: (segment === 'forward' || isBtToFwd || isFwdToLive) ? totalR : null,
          live: (segment === 'live' || isFwdToLive) ? totalR : null,
          btBridge: null,
          fwdBridge: null,
        };
      });
    }

    // Per-trade mode: sequential trade numbers with overlap at segment boundaries
    return data.map((pt, i) => {
      let segment: 'backtest' | 'forward' | 'live' = 'backtest';
      if (boundaryIndex != null && i >= boundaryIndex) segment = 'forward';
      if (alertBoundaryIndex != null && i >= alertBoundaryIndex) segment = 'live';

      // At segment boundaries, include value in BOTH segments so lines connect seamlessly
      const isLastBT = boundaryIndex != null && i === boundaryIndex - 1;
      const isFirstFWD = boundaryIndex != null && i === boundaryIndex;
      const isLastFWD = alertBoundaryIndex != null && i === alertBoundaryIndex - 1;
      const isFirstLive = alertBoundaryIndex != null && i === alertBoundaryIndex;

      return {
        x: pt.trade_number,
        r: pt.cumulative_r,
        bt: (segment === 'backtest' || isFirstFWD) ? pt.cumulative_r : null,
        fwd: (segment === 'forward' || isLastBT || isFirstLive) ? pt.cumulative_r : null,
        live: (segment === 'live' || isLastFWD) ? pt.cumulative_r : null,
        btBridge: null,
        fwdBridge: null,
      };
    });
  }, [data, boundaryIndex, alertBoundaryIndex, xAxis]);

  // Merge HWM into chartData (MUST be before any early return — React hooks rule)
  const chartDataWithHwm = useMemo(() => {
    if (!showHWM || chartData.length === 0) return chartData;
    let max = -Infinity;
    return chartData.map((pt: any) => {
      const val = pt.r ?? pt.bt ?? pt.fwd ?? pt.live ?? 0;
      max = Math.max(max, val);
      return { ...pt, hwm: max };
    });
  }, [chartData, showHWM]);

  // Edge Check: 21-period moving average on cumulative R (MUST be before early return)
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

  // Merge alert overlay data into chart (green line overlaying FWD segment)
  const chartDataWithAlerts = useMemo(() => {
    if (!alertOverlayData || alertOverlayData.length === 0) return chartDataFinal;
    if (xAxis === 'trade' && boundaryIndex != null) {
      // In per-trade mode: map alert trades to positions starting at boundaryIndex
      return chartDataFinal.map((pt: any, i: number) => {
        const alertIdx = i - boundaryIndex;
        if (alertIdx >= 0 && alertIdx < alertOverlayData.length) {
          return { ...pt, alertR: alertOverlayData[alertIdx].cumulative_r };
        }
        return { ...pt, alertR: null };
      });
    }
    // In per-day mode: map by matching timestamps
    if (xAxis === 'time') {
      const alertByDate = new Map<string, number>();
      for (const apt of alertOverlayData) {
        if (!apt.timestamp) continue;
        const d = new Date(apt.timestamp);
        if (!isNaN(d.getTime())) {
          const label = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
          alertByDate.set(label, apt.cumulative_r);
        }
      }
      return chartDataFinal.map((pt: any) => ({
        ...pt,
        alertR: alertByDate.get(pt.x) ?? null,
      }));
    }
    return chartDataFinal;
  }, [chartDataFinal, alertOverlayData, xAxis, boundaryIndex]);

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
      <AreaChart data={chartDataWithAlerts} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
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
        {/* Forward test segment */}
        <Area
          type={curveType} dataKey="fwd" stroke={fwdColor} strokeWidth={2}
          fill={showGradient ? `url(#${fwdGradId})` : 'none'} dot={false} isAnimationActive={false}
          connectNulls={false}
        />

        {/* Live alert segment (no fill) */}
        <Line
          type={curveType} dataKey="live" stroke={liveColor} strokeWidth={2}
          dot={false} isAnimationActive={false} connectNulls={false}
        />

        {/* Alert overlay (green, overlays on FWD segment to show execution quality) */}
        <Line
          type={curveType} dataKey="alertR" stroke={liveColor} strokeWidth={2}
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
