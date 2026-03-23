'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Alert Command Center
// =============================================================================
// Dark terminal-inspired alert management with:
//
// 1. ALERT COMMAND CENTER — terminal-style feed with color-coded entries
// 2. SIGNAL RADAR — circular visualization of active symbols + directions
// 3. ALERT FREQUENCY CHART — sparkline showing signal density over time
// 4. LIVE POSITION CARDS — animated cards with entry, current, stop-to-target bar
// 5. STRATEGY SIGNAL MATRIX — grid of last signals per strategy
// 6. ALERT STATISTICS — accuracy, avg R, best trade, total alerts
// 7. NOTIFICATION TIMELINE — vertical timeline with connecting lines
// 8. PULSE ANIMATIONS — animated glow on new alerts
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
  0% { transform: scale(1); opacity: 0.6; }
  100% { transform: scale(2.5); opacity: 0; }
}
@keyframes slide-in-left {
  0% { transform: translateX(-16px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes slide-in-right {
  0% { transform: translateX(16px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes terminal-blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}
@keyframes radar-sweep {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
@keyframes radar-ping {
  0% { transform: scale(0.8); opacity: 1; }
  100% { transform: scale(2); opacity: 0; }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 20px rgba(0,255,136,0.35); }
}
@keyframes alert-flash {
  0% { background: rgba(0,255,136,0.15); }
  100% { background: transparent; }
}
@keyframes progress-fill {
  0% { width: 0%; }
}
@keyframes heartbeat {
  0%, 100% { transform: scale(1); }
  14% { transform: scale(1.08); }
  28% { transform: scale(1); }
  42% { transform: scale(1.04); }
  70% { transform: scale(1); }
}
@keyframes scanline {
  0% { top: -10%; }
  100% { top: 110%; }
}
`;

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const MONITOR_STATUS = {
  running: true,
  uptime: '6h 42m',
  pid: 14923,
  lastHeartbeat: '2s ago',
  cpuUsage: 3.2,
  memUsage: 128,
};

const STRATEGIES = [
  {
    id: '1', name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG',
    lastSignal: 'ENTRY' as const, lastSignalTime: '10:32 AM', position: 'FLAT' as const,
    alertEntry: true, alertExit: true, webhook: true, totalAlerts: 42,
  },
  {
    id: '2', name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG',
    lastSignal: 'IN_POSITION' as const, lastSignalTime: '09:45 AM', position: 'IN_POSITION' as const,
    alertEntry: true, alertExit: true, webhook: true, totalAlerts: 78,
  },
  {
    id: '3', name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG',
    lastSignal: 'EXIT' as const, lastSignalTime: '09:15 AM', position: 'FLAT' as const,
    alertEntry: true, alertExit: true, webhook: false, totalAlerts: 35,
  },
  {
    id: '4', name: 'TSLA LONG - Mass #5', symbol: 'TSLA', direction: 'LONG',
    lastSignal: 'FLAT' as const, lastSignalTime: '08:50 AM', position: 'FLAT' as const,
    alertEntry: true, alertExit: true, webhook: true, totalAlerts: 28,
  },
  {
    id: '5', name: 'META LONG - Mass #13', symbol: 'META', direction: 'LONG',
    lastSignal: 'FLAT' as const, lastSignalTime: '--', position: 'FLAT' as const,
    alertEntry: false, alertExit: false, webhook: false, totalAlerts: 0,
  },
];

const OPEN_POSITIONS = [
  {
    strategy: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG',
    entryPrice: 562.10, currentPrice: 563.45, stopPrice: 560.50, targetPrice: 566.20,
    unrealizedPnL: 67.50, unrealizedR: 0.84, duration: '47m', entryTime: '09:45 AM',
    execType: '[L0]',
  },
];

const ALERTS: {
  id: string;
  time: string;
  type: 'ENTRY' | 'EXIT' | 'SYSTEM';
  strategy: string;
  symbol: string;
  price: string;
  execType: string;
  pnl: string | null;
  rValue: number | null;
  isNew: boolean;
}[] = [
  { id: '1', time: '10:32:14', type: 'ENTRY', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: '$487.50', execType: '[C]', pnl: null, rValue: null, isNew: true },
  { id: '2', time: '10:15:02', type: 'EXIT', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: '$487.23', execType: '[C]', pnl: '+$120', rValue: 1.2, isNew: true },
  { id: '3', time: '09:45:30', type: 'ENTRY', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', price: '$562.10', execType: '[L0]', pnl: null, rValue: null, isNew: false },
  { id: '4', time: '09:31:18', type: 'ENTRY', strategy: 'TSLA LONG - Mass #5', symbol: 'TSLA', price: '$178.90', execType: '[HM]', pnl: null, rValue: null, isNew: false },
  { id: '5', time: '09:15:44', type: 'EXIT', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', price: '$192.45', execType: '[C]', pnl: '-$80', rValue: -0.8, isNew: false },
  { id: '6', time: '09:02:12', type: 'ENTRY', strategy: 'AAPL LONG - Mass #5', symbol: 'AAPL', price: '$191.80', execType: '[L1]', pnl: null, rValue: null, isNew: false },
  { id: '7', time: '08:50:55', type: 'EXIT', strategy: 'TSLA LONG - Mass #5', symbol: 'TSLA', price: '$176.50', execType: '[HL]', pnl: '+$210', rValue: 2.1, isNew: false },
  { id: '8', time: '08:42:03', type: 'ENTRY', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', price: '$561.30', execType: '[C]', pnl: null, rValue: null, isNew: false },
  { id: '9', time: '08:35:21', type: 'EXIT', strategy: 'SPY LONG - Mass #1', symbol: 'SPY', price: '$561.90', execType: '[C]', pnl: '+$30', rValue: 0.3, isNew: false },
  { id: '10', time: '08:15:10', type: 'ENTRY', strategy: 'NVDA LONG - Mass #2', symbol: 'NVDA', price: '$485.20', execType: '[C]', pnl: null, rValue: null, isNew: false },
];

// Alert frequency (alerts per 15-min bucket)
const FREQUENCY_DATA = [0, 1, 0, 2, 3, 1, 0, 0, 1, 2, 4, 2, 1, 3, 2, 1, 0, 1, 2, 1];

const ALERT_STATS = {
  totalAlerts: 183,
  todayAlerts: 10,
  accuracy: 58.4,
  avgR: 0.34,
  bestTrade: { symbol: 'TSLA', r: 3.8 },
  worstTrade: { symbol: 'META', r: -2.1 },
  totalR: 12.6,
  streakWins: 3,
};

// Radar data: symbols with their angle positions and signal recency
const RADAR_SYMBOLS = [
  { symbol: 'NVDA', angle: 30, distance: 0.9, direction: 'LONG' as const, lastSignal: 'ENTRY' as const, active: true },
  { symbol: 'SPY', angle: 120, distance: 0.7, direction: 'LONG' as const, lastSignal: 'IN_POSITION' as const, active: true },
  { symbol: 'AAPL', angle: 200, distance: 0.5, direction: 'LONG' as const, lastSignal: 'EXIT' as const, active: false },
  { symbol: 'TSLA', angle: 280, distance: 0.4, direction: 'LONG' as const, lastSignal: 'FLAT' as const, active: false },
  { symbol: 'META', angle: 340, distance: 0.2, direction: 'LONG' as const, lastSignal: 'FLAT' as const, active: false },
];

// ---------------------------------------------------------------------------
// SVG Components
// ---------------------------------------------------------------------------

/** Signal radar visualization */
function SignalRadar({ symbols, size = 200 }: { symbols: typeof RADAR_SYMBOLS; size?: number }) {
  const c = size / 2;
  const maxR = (size - 40) / 2;

  const signalColors: Record<string, string> = {
    ENTRY: 'var(--green)',
    EXIT: 'var(--red)',
    IN_POSITION: 'var(--accent)',
    FLAT: 'var(--text-muted)',
  };

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Concentric rings */}
        {[0.25, 0.5, 0.75, 1].map((r) => (
          <circle
            key={r}
            cx={c}
            cy={c}
            r={maxR * r}
            fill="none"
            stroke="var(--border)"
            strokeWidth="0.5"
            opacity={0.4}
          />
        ))}
        {/* Crosshairs */}
        <line x1={c} y1={c - maxR} x2={c} y2={c + maxR} stroke="var(--border)" strokeWidth="0.5" opacity={0.3} />
        <line x1={c - maxR} y1={c} x2={c + maxR} y2={c} stroke="var(--border)" strokeWidth="0.5" opacity={0.3} />

        {/* Sweep line */}
        <line
          x1={c}
          y1={c}
          x2={c}
          y2={c - maxR}
          stroke="var(--accent)"
          strokeWidth="1"
          opacity={0.4}
          style={{ transformOrigin: `${c}px ${c}px`, animation: 'radar-sweep 8s linear infinite' }}
        />

        {/* Symbol dots */}
        {symbols.map((sym) => {
          const rad = (sym.angle * Math.PI) / 180;
          const dist = maxR * sym.distance;
          const x = c + dist * Math.cos(rad - Math.PI / 2);
          const y = c + dist * Math.sin(rad - Math.PI / 2);
          const color = signalColors[sym.lastSignal];

          return (
            <g key={sym.symbol}>
              {/* Ping animation for active */}
              {sym.active && (
                <circle
                  cx={x}
                  cy={y}
                  r={8}
                  fill={color}
                  opacity={0.3}
                  style={{ animation: 'radar-ping 2s ease-out infinite' }}
                />
              )}
              {/* Dot */}
              <circle
                cx={x}
                cy={y}
                r={sym.active ? 5 : 3}
                fill={color}
                style={{ filter: sym.active ? `drop-shadow(0 0 4px ${color})` : 'none' }}
              />
              {/* Label */}
              <text
                x={x}
                y={y - 10}
                textAnchor="middle"
                fill="var(--text-primary)"
                fontSize="9"
                fontWeight="600"
                fontFamily="monospace"
              >
                {sym.symbol}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Alert frequency sparkline */
function FrequencySparkline({ data, width = 300, height = 40 }: {
  data: number[];
  width?: number;
  height?: number;
}) {
  const max = Math.max(...data, 1);
  const barW = width / data.length - 1;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {data.map((v, i) => {
        const barH = (v / max) * (height - 4);
        const intensity = v / max;
        return (
          <rect
            key={i}
            x={i * (barW + 1)}
            y={height - barH - 2}
            width={barW}
            height={barH}
            rx={1}
            fill="var(--accent)"
            opacity={0.3 + intensity * 0.6}
          />
        );
      })}
    </svg>
  );
}

/** Live position card with stop-to-target progress bar */
function PositionCard({ position }: { position: typeof OPEN_POSITIONS[0] }) {
  const range = position.targetPrice - position.stopPrice;
  const progress = ((position.currentPrice - position.stopPrice) / range) * 100;
  const entryPct = ((position.entryPrice - position.stopPrice) / range) * 100;
  const isProfit = position.unrealizedPnL >= 0;

  return (
    <Card
      className="relative overflow-hidden"
    >
      {/* Scanline effect */}
      <div
        className="absolute inset-x-0 h-px pointer-events-none"
        style={{
          background: 'linear-gradient(90deg, transparent, var(--accent), transparent)',
          opacity: 0.15,
          animation: 'scanline 4s linear infinite',
        }}
      />

      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold">{position.symbol}</span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-mono"
              style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
            >
              {position.direction}
            </span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-mono"
              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
            >
              {position.execType}
            </span>
          </div>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{position.strategy}</p>
        </div>
        <div className="text-right">
          <p
            className="text-lg font-bold font-mono"
            style={{
              color: isProfit ? 'var(--green)' : 'var(--red)',
              animation: 'heartbeat 3s ease-in-out infinite',
            }}
          >
            {isProfit ? '+' : ''}{position.unrealizedR.toFixed(2)}R
          </p>
          <p className="text-xs" style={{ color: isProfit ? 'var(--green)' : 'var(--red)' }}>
            {isProfit ? '+' : ''}${position.unrealizedPnL.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Price row */}
      <div className="flex items-center justify-between text-xs mb-3">
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Entry </span>
          <span className="font-mono font-medium">${position.entryPrice.toFixed(2)}</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Current </span>
          <span className="font-mono font-medium" style={{ color: isProfit ? 'var(--green)' : 'var(--red)' }}>
            ${position.currentPrice.toFixed(2)}
          </span>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Duration </span>
          <span className="font-mono">{position.duration}</span>
        </div>
      </div>

      {/* Stop-to-Target bar */}
      <div>
        <div className="flex items-center justify-between text-[10px] mb-1 font-mono">
          <span style={{ color: 'var(--red)' }}>Stop ${position.stopPrice.toFixed(2)}</span>
          <span style={{ color: 'var(--green)' }}>Target ${position.targetPrice.toFixed(2)}</span>
        </div>
        <div className="relative h-3 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
          {/* Entry marker */}
          <div
            className="absolute top-0 bottom-0 w-0.5"
            style={{
              left: `${Math.max(0, Math.min(100, entryPct))}%`,
              background: 'var(--text-muted)',
              zIndex: 2,
            }}
          />
          {/* Current price fill */}
          <div
            className="absolute left-0 top-0 bottom-0 rounded-full transition-all"
            style={{
              width: `${Math.max(0, Math.min(100, progress))}%`,
              background: isProfit
                ? 'linear-gradient(90deg, var(--orange), var(--green))'
                : 'linear-gradient(90deg, var(--red), var(--orange))',
              animation: 'progress-fill 1s ease-out forwards',
            }}
          />
          {/* Current price dot */}
          <div
            className="absolute top-0 bottom-0 w-3 h-3 rounded-full border-2"
            style={{
              left: `calc(${Math.max(0, Math.min(100, progress))}% - 6px)`,
              background: 'var(--bg-card)',
              borderColor: isProfit ? 'var(--green)' : 'var(--red)',
              boxShadow: `0 0 6px ${isProfit ? 'var(--green)' : 'var(--red)'}`,
              zIndex: 3,
            }}
          />
        </div>
      </div>
    </Card>
  );
}

/** Strategy signal matrix */
function SignalMatrix({ strategies }: { strategies: typeof STRATEGIES }) {
  const signalColors: Record<string, { bg: string; text: string }> = {
    ENTRY: { bg: 'var(--green-muted)', text: 'var(--green)' },
    EXIT: { bg: 'var(--red-muted)', text: 'var(--red)' },
    IN_POSITION: { bg: 'var(--accent-muted)', text: 'var(--accent)' },
    FLAT: { bg: 'var(--bg-input)', text: 'var(--text-muted)' },
  };

  return (
    <div className="grid grid-cols-5 gap-2">
      {strategies.map((s, i) => {
        const colors = signalColors[s.lastSignal];
        return (
          <div
            key={s.id}
            className="rounded-lg p-3 text-center transition-all cursor-pointer"
            style={{
              background: colors.bg,
              border: `1px solid ${s.lastSignal === 'ENTRY' || s.lastSignal === 'IN_POSITION' ? colors.text : 'var(--border)'}`,
              animation: `fade-in-up 0.4s ease-out ${i * 0.08}s both`,
              ...(s.lastSignal === 'ENTRY' ? { boxShadow: '0 0 12px rgba(0,255,136,0.15)' } : {}),
            }}
            title={`${s.name} — Last: ${s.lastSignal} at ${s.lastSignalTime}`}
          >
            <p className="text-sm font-bold mb-0.5">{s.symbol}</p>
            <p
              className="text-[10px] font-mono font-semibold"
              style={{ color: colors.text }}
            >
              {s.lastSignal === 'IN_POSITION' ? 'IN POS' : s.lastSignal}
            </p>
            <p className="text-[9px] mt-1" style={{ color: 'var(--text-muted)' }}>{s.lastSignalTime}</p>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Toggle Switch (reused from V3 style)
// ---------------------------------------------------------------------------

function ToggleSwitch({ on, onChange, size = 'sm' }: { on: boolean; onChange: () => void; size?: 'sm' | 'lg' }) {
  const w = size === 'lg' ? 44 : 28;
  const h = size === 'lg' ? 24 : 14;
  const dot = size === 'lg' ? 18 : 10;
  const offset = size === 'lg' ? 3 : 2;

  return (
    <button
      onClick={onChange}
      className="relative rounded-full transition-colors"
      style={{
        width: w,
        height: h,
        background: on ? 'var(--accent)' : 'var(--bg-input)',
        border: on ? 'none' : '1px solid var(--border)',
        cursor: 'pointer',
        flexShrink: 0,
      }}
    >
      <span
        className="absolute rounded-full transition-all"
        style={{
          width: dot,
          height: dot,
          top: offset,
          left: on ? w - dot - offset : offset,
          background: on ? 'white' : 'var(--text-muted)',
        }}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Alert Detail Modal
// ---------------------------------------------------------------------------

function AlertDetailModal({ alert, onClose }: {
  alert: typeof ALERTS[0];
  onClose: () => void;
}) {
  return (
    <Modal title="Alert Detail" isOpen onClose={onClose} width="500px">
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <span
            className="text-sm px-3 py-1 rounded-full font-semibold"
            style={{
              background: alert.type === 'ENTRY' ? 'var(--green-muted)' : alert.type === 'EXIT' ? 'var(--red-muted)' : 'var(--accent-muted)',
              color: alert.type === 'ENTRY' ? 'var(--green)' : alert.type === 'EXIT' ? 'var(--red)' : 'var(--accent)',
            }}
          >
            {alert.type}
          </span>
          <span className="font-mono text-sm" style={{ color: 'var(--text-muted)' }}>{alert.execType}</span>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {[
            { label: 'Time', value: alert.time },
            { label: 'Symbol', value: alert.symbol },
            { label: 'Price', value: alert.price },
            { label: 'Strategy', value: alert.strategy },
            { label: 'Exec Type', value: alert.execType },
            { label: 'P&L', value: alert.pnl || '--' },
          ].map((field) => (
            <div key={field.label} className="p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{field.label}</p>
              <p className="text-sm font-medium font-mono">{field.value}</p>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function AlertsV4() {
  const [monitorOn, setMonitorOn] = useState(MONITOR_STATUS.running);
  const [strategies, setStrategies] = useState(STRATEGIES);
  const [filterType, setFilterType] = useState<'ALL' | 'ENTRY' | 'EXIT'>('ALL');
  const [selectedAlert, setSelectedAlert] = useState<typeof ALERTS[0] | null>(null);

  const filteredAlerts = useMemo(() => {
    if (filterType === 'ALL') return ALERTS;
    return ALERTS.filter((a) => a.type === filterType);
  }, [filterType]);

  const toggleField = useCallback((id: string, field: 'alertEntry' | 'alertExit' | 'webhook') => {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, [field]: !s[field] } : s))
    );
  }, []);

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      <h1 className="text-2xl font-bold mb-5">Alerts & Signals</h1>

      {/* ── Row 1: Monitor Status + Stats + Signal Matrix ── */}
      <div className="grid grid-cols-12 gap-4 mb-5">
        {/* Monitor Status Card */}
        <div className="col-span-3" style={{ animation: 'fade-in-up 0.4s ease-out both' }}>
          <Card>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{
                      background: monitorOn ? 'var(--green)' : 'var(--text-muted)',
                      boxShadow: monitorOn ? '0 0 10px var(--green)' : 'none',
                    }}
                  />
                  {monitorOn && (
                    <div
                      className="absolute inset-0 w-3 h-3 rounded-full"
                      style={{
                        background: 'var(--green)',
                        animation: 'pulse-ring 2s ease-out infinite',
                      }}
                    />
                  )}
                </div>
                <div>
                  <p className="font-semibold text-sm">Monitor Engine</p>
                  <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {monitorOn ? `PID ${MONITOR_STATUS.pid}` : 'Stopped'}
                  </p>
                </div>
              </div>
              <ToggleSwitch on={monitorOn} onChange={() => setMonitorOn(!monitorOn)} size="lg" />
            </div>

            {monitorOn && (
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Uptime', value: MONITOR_STATUS.uptime },
                  { label: 'Heartbeat', value: MONITOR_STATUS.lastHeartbeat },
                  { label: 'CPU', value: `${MONITOR_STATUS.cpuUsage}%` },
                  { label: 'Memory', value: `${MONITOR_STATUS.memUsage}MB` },
                ].map((stat) => (
                  <div key={stat.label} className="text-center p-2 rounded" style={{ background: 'var(--bg-input)' }}>
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{stat.label}</p>
                    <p className="text-xs font-mono font-medium">{stat.value}</p>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* Alert Stats */}
        <div className="col-span-4" style={{ animation: 'fade-in-up 0.4s ease-out 0.08s both' }}>
          <div className="grid grid-cols-2 gap-3 h-full">
            <MetricCard label="Today's Alerts" value={String(ALERT_STATS.todayAlerts)} delta={`${ALERT_STATS.totalAlerts} total`} />
            <MetricCard label="Accuracy" value={`${ALERT_STATS.accuracy}%`} delta={`${ALERT_STATS.streakWins}W streak`} positive />
            <MetricCard label="Avg R" value={`${ALERT_STATS.avgR}R`} delta={`${ALERT_STATS.totalR}R total`} positive />
            <MetricCard
              label="Best Trade"
              value={`+${ALERT_STATS.bestTrade.r}R`}
              delta={ALERT_STATS.bestTrade.symbol}
              positive
            />
          </div>
        </div>

        {/* Signal Matrix */}
        <div className="col-span-5" style={{ animation: 'fade-in-up 0.4s ease-out 0.16s both' }}>
          <Card>
            <h3 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Strategy Signal Matrix
            </h3>
            <SignalMatrix strategies={strategies} />
          </Card>
        </div>
      </div>

      {/* ── Row 2: Radar + Positions + Alert Frequency ── */}
      <div className="grid grid-cols-12 gap-4 mb-5">
        {/* Signal Radar */}
        <div className="col-span-3" style={{ animation: 'fade-in-up 0.4s ease-out 0.2s both' }}>
          <Card>
            <h3 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
              Signal Radar
            </h3>
            <div className="flex justify-center">
              <SignalRadar symbols={RADAR_SYMBOLS} size={190} />
            </div>
            {/* Legend */}
            <div className="flex items-center justify-center gap-4 mt-2">
              {[
                { label: 'Entry', color: 'var(--green)' },
                { label: 'In Pos', color: 'var(--accent)' },
                { label: 'Exit', color: 'var(--red)' },
                { label: 'Flat', color: 'var(--text-muted)' },
              ].map((l) => (
                <span key={l.label} className="flex items-center gap-1 text-[9px]" style={{ color: 'var(--text-muted)' }}>
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: l.color }} />
                  {l.label}
                </span>
              ))}
            </div>
          </Card>
        </div>

        {/* Live Positions */}
        <div className="col-span-5" style={{ animation: 'fade-in-up 0.4s ease-out 0.24s both' }}>
          {OPEN_POSITIONS.length > 0 ? (
            <div className="space-y-3">
              <h3 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                Open Positions ({OPEN_POSITIONS.length})
              </h3>
              {OPEN_POSITIONS.map((pos, i) => (
                <PositionCard key={i} position={pos} />
              ))}
            </div>
          ) : (
            <Card className="h-full flex items-center justify-center">
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No open positions</p>
            </Card>
          )}

          {/* Alert frequency below positions */}
          <Card className="mt-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Alert Frequency</h3>
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>15m intervals</span>
            </div>
            <FrequencySparkline data={FREQUENCY_DATA} height={32} />
          </Card>
        </div>

        {/* Strategy Config */}
        <div className="col-span-4" style={{ animation: 'fade-in-up 0.4s ease-out 0.28s both' }}>
          <Card className="h-full">
            <h3 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Strategy Configuration
            </h3>
            <div className="space-y-0">
              {strategies.map((strat) => (
                <div
                  key={strat.id}
                  className="py-2.5 border-b last:border-0"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-medium">{strat.symbol}</span>
                    <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                      {strat.totalAlerts} alerts
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <label className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      <ToggleSwitch on={strat.alertEntry} onChange={() => toggleField(strat.id, 'alertEntry')} />
                      Entry
                    </label>
                    <label className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      <ToggleSwitch on={strat.alertExit} onChange={() => toggleField(strat.id, 'alertExit')} />
                      Exit
                    </label>
                    <label className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      <ToggleSwitch on={strat.webhook} onChange={() => toggleField(strat.id, 'webhook')} />
                      Hook
                    </label>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* ── Row 3: Terminal-style Alert Feed ── */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Alert Feed
            </h3>
            <span className="flex items-center gap-1">
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: 'var(--green)', animation: 'terminal-blink 1s ease-in-out infinite' }}
              />
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>LIVE</span>
            </span>
          </div>

          <div className="flex items-center gap-2">
            {(['ALL', 'ENTRY', 'EXIT'] as const).map((type) => (
              <button
                key={type}
                onClick={() => setFilterType(type)}
                className="text-xs px-2.5 py-1 rounded-md transition-all font-mono"
                style={{
                  background: filterType === type ? 'var(--accent-muted)' : 'var(--bg-input)',
                  color: filterType === type ? 'var(--accent)' : 'var(--text-muted)',
                  border: `1px solid ${filterType === type ? 'var(--accent)' : 'var(--border)'}`,
                }}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Terminal-style feed */}
        <div
          className="rounded-lg overflow-hidden font-mono text-xs"
          style={{
            background: 'rgba(0,0,0,0.3)',
            border: '1px solid var(--border)',
            maxHeight: 400,
            overflowY: 'auto',
          }}
        >
          {/* Header row */}
          <div
            className="grid grid-cols-12 gap-2 px-3 py-2 border-b sticky top-0"
            style={{ background: 'rgba(0,0,0,0.5)', borderColor: 'var(--border)' }}
          >
            <span className="col-span-2" style={{ color: 'var(--text-muted)' }}>TIME</span>
            <span className="col-span-1" style={{ color: 'var(--text-muted)' }}>EXEC</span>
            <span className="col-span-1" style={{ color: 'var(--text-muted)' }}>TYPE</span>
            <span className="col-span-1" style={{ color: 'var(--text-muted)' }}>SYM</span>
            <span className="col-span-2" style={{ color: 'var(--text-muted)' }}>PRICE</span>
            <span className="col-span-3" style={{ color: 'var(--text-muted)' }}>STRATEGY</span>
            <span className="col-span-2 text-right" style={{ color: 'var(--text-muted)' }}>P&L</span>
          </div>

          {filteredAlerts.length === 0 ? (
            <div className="text-center py-10" style={{ color: 'var(--text-muted)' }}>
              No alerts match filter.
            </div>
          ) : (
            filteredAlerts.map((alert, i) => (
              <div
                key={alert.id}
                className="grid grid-cols-12 gap-2 px-3 py-2 border-b items-center cursor-pointer transition-all"
                style={{
                  borderColor: 'rgba(255,255,255,0.03)',
                  animation: alert.isNew
                    ? `alert-flash 2s ease-out, slide-in-left 0.3s ease-out ${i * 0.05}s both`
                    : `slide-in-left 0.3s ease-out ${i * 0.05}s both`,
                }}
                onClick={() => setSelectedAlert(alert)}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.03)')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
              >
                {/* Time */}
                <span className="col-span-2" style={{ color: 'var(--text-muted)' }}>
                  {alert.time}
                </span>

                {/* Exec type */}
                <span
                  className="col-span-1 px-1 py-0.5 rounded text-center"
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    color: 'var(--text-muted)',
                    fontSize: '0.6rem',
                  }}
                >
                  {alert.execType}
                </span>

                {/* Type */}
                <span
                  className="col-span-1 font-semibold"
                  style={{
                    color: alert.type === 'ENTRY' ? 'var(--green)' : alert.type === 'EXIT' ? 'var(--red)' : 'var(--accent)',
                  }}
                >
                  {alert.type === 'ENTRY' ? '>> ENT' : alert.type === 'EXIT' ? '<< EXT' : '-- SYS'}
                </span>

                {/* Symbol */}
                <span className="col-span-1 font-semibold">{alert.symbol}</span>

                {/* Price */}
                <span className="col-span-2">{alert.price}</span>

                {/* Strategy */}
                <span className="col-span-3 truncate" style={{ color: 'var(--text-muted)' }}>
                  {alert.strategy}
                </span>

                {/* P&L */}
                <span className="col-span-2 text-right">
                  {alert.pnl ? (
                    <span
                      className="font-semibold"
                      style={{
                        color: alert.rValue && alert.rValue >= 0 ? 'var(--green)' : 'var(--red)',
                      }}
                    >
                      {alert.pnl} ({alert.rValue && alert.rValue >= 0 ? '+' : ''}{alert.rValue}R)
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>--</span>
                  )}
                </span>

                {/* New indicator */}
                {alert.isNew && (
                  <div
                    className="absolute right-3 w-1.5 h-1.5 rounded-full"
                    style={{
                      background: 'var(--green)',
                      boxShadow: '0 0 6px var(--green)',
                    }}
                  />
                )}
              </div>
            ))
          )}
        </div>
      </Card>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <AlertDetailModal alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
      )}
    </div>
  );
}
