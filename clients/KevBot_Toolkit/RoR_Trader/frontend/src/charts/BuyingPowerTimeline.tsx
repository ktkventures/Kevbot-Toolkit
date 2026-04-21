'use client';

/**
 * BuyingPowerTimeline — 24-hour intraday view of portfolio buying power.
 *
 * Consumes the capital-utilization timeline from
 * POST /api/portfolios/{id}/capital-utilization?source=alerts which returns
 * one event per position entry/exit:
 *   { time, buying_power, capital_deployed, concurrent_positions,
 *     event_type: 'entry' | 'exit', strategy }
 *
 * Filters events to the selected date, builds a step-line series so the
 * chart reflects "state held until next event" (vs a misleading smooth
 * interpolation). Markers at each event with hover tooltip naming the
 * strategy + event type.
 */

import { useMemo } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';

interface TimelineEvent {
  time: string;
  buying_power: number;
  capital_deployed: number;
  concurrent_positions: number;
  event_type?: 'entry' | 'exit';
  strategy?: string;
}

interface BuyingPowerTimelineProps {
  timeline: TimelineEvent[];
  date: string;           // YYYY-MM-DD — filter events to this day
  startingBalance: number; // shown as upper reference line
  height?: number;
}

const GREEN = '#4CAF50';
const ORANGE = '#FF9800';

export default function BuyingPowerTimeline({
  timeline, date, startingBalance, height = 220,
}: BuyingPowerTimelineProps) {
  const { data, eventCount } = useMemo(() => {
    if (!Array.isArray(timeline) || timeline.length === 0) {
      return { data: [], eventCount: 0 };
    }
    const dayStart = new Date(`${date}T00:00:00`).getTime();
    const dayEnd = dayStart + 86400000;

    // Sort by time, filter to selected date
    const sorted = [...timeline]
      .map((e) => ({ ...e, _ms: new Date(e.time).getTime() }))
      .filter((e) => Number.isFinite(e._ms) && e._ms >= dayStart && e._ms < dayEnd)
      .sort((a, b) => a._ms - b._ms);

    if (sorted.length === 0) return { data: [], eventCount: 0 };

    // Build step-line points: at each event, emit a "before" sample with the
    // prior state + an "after" sample with the new state. This produces a
    // step chart where each state lasts until the next event.
    const points: Array<{
      t: number; time: string; buying_power: number;
      capital_deployed: number; event?: string;
    }> = [];

    // Anchor the day at midnight with the first event's prior state so the
    // chart starts from a known baseline.
    const first = sorted[0];
    points.push({
      t: dayStart, time: new Date(dayStart).toISOString(),
      buying_power: startingBalance,
      capital_deployed: 0,
    });

    for (let i = 0; i < sorted.length; i++) {
      const e = sorted[i];
      // Pre-event point carries the prior state
      if (i > 0) {
        const prev = sorted[i - 1];
        points.push({
          t: e._ms - 1, time: new Date(e._ms - 1).toISOString(),
          buying_power: Number(prev.buying_power) || 0,
          capital_deployed: Number(prev.capital_deployed) || 0,
        });
      }
      // Event point
      points.push({
        t: e._ms,
        time: e.time,
        buying_power: Number(e.buying_power) || 0,
        capital_deployed: Number(e.capital_deployed) || 0,
        event: e.event_type && e.strategy
          ? `${e.event_type}: ${e.strategy}`
          : e.event_type,
      });
    }

    // Trailing point at end of day to extend the step-line visually.
    const last = sorted[sorted.length - 1];
    points.push({
      t: dayEnd - 1, time: new Date(dayEnd - 1).toISOString(),
      buying_power: Number(last.buying_power) || 0,
      capital_deployed: Number(last.capital_deployed) || 0,
    });

    return { data: points, eventCount: sorted.length };
  }, [timeline, date, startingBalance]);

  if (data.length === 0) {
    return (
      <div
        style={{
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: '12px',
        }}
      >
        No position activity on {date}
      </div>
    );
  }

  const fmtHHMM = (t: number) => {
    const d = new Date(t);
    return d.toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', hour12: false,
    });
  };

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="bpAvail" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={GREEN} stopOpacity={0.2} />
            <stop offset="100%" stopColor={GREEN} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="bpDeployed" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={ORANGE} stopOpacity={0.25} />
            <stop offset="100%" stopColor={ORANGE} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--border, rgba(255,255,255,0.06))" strokeDasharray="3 3" />
        <XAxis
          dataKey="t"
          type="number"
          domain={['dataMin', 'dataMax']}
          tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          tickFormatter={fmtHHMM}
        />
        <YAxis
          tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
          axisLine={{ stroke: 'var(--border, rgba(255,255,255,0.1))' }}
          tickLine={false}
          tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: '8px', fontSize: '12px',
            color: 'var(--text-primary)',
          }}
          labelFormatter={(label: number) => fmtHHMM(label)}
          formatter={(value: number, name: string) => {
            const label = name === 'buying_power' ? 'Available' : 'Deployed';
            return [`$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, label];
          }}
        />
        {startingBalance > 0 && (
          <ReferenceLine
            y={startingBalance}
            stroke="rgba(255,255,255,0.25)"
            strokeDasharray="4 4"
            label={{
              value: `Balance $${Math.round(startingBalance).toLocaleString()}`,
              fill: 'var(--text-muted)', fontSize: 10, position: 'insideTopRight',
            }}
          />
        )}
        <Area
          type="stepAfter"
          dataKey="buying_power"
          stroke={GREEN}
          strokeWidth={1.5}
          fill="url(#bpAvail)"
          dot={false}
          isAnimationActive={false}
        />
        <Area
          type="stepAfter"
          dataKey="capital_deployed"
          stroke={ORANGE}
          strokeWidth={1.5}
          fill="url(#bpDeployed)"
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
