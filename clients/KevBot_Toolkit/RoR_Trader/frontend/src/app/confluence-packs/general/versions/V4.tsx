'use client';

import { useState, useCallback, useMemo, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';

/* ─────────────────────────── Types ─────────────────────────── */

interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float' | 'bool' | 'select';
  value: number | boolean | string;
  min?: number;
  max?: number;
  options?: string[];
}

interface PackTrigger {
  name: string;
  direction: 'BOTH';
  type: 'ENTRY' | 'EXIT';
  exec: string;
}

interface GeneralPack {
  id: string;
  name: string;
  category: 'Session' | 'Calendar';
  enabled: boolean;
  isDefault: boolean;
  params: PackParam[];
  outputs: string[];
  triggers: PackTrigger[];
  currentState: string;
}

/* ─────────────────────────── Mock Data ─────────────────────────── */

const initialPacks: GeneralPack[] = [
  {
    id: 'tod-default',
    name: 'Time of Day',
    category: 'Session',
    enabled: true,
    isDefault: true,
    currentState: 'IN_WINDOW',
    params: [
      { key: 'start_hour', label: 'Start Hour', type: 'int', value: 9, min: 0, max: 23 },
      { key: 'start_minute', label: 'Start Min', type: 'int', value: 30, min: 0, max: 59 },
      { key: 'end_hour', label: 'End Hour', type: 'int', value: 12, min: 0, max: 23 },
      { key: 'end_minute', label: 'End Min', type: 'int', value: 0, min: 0, max: 59 },
    ],
    outputs: ['IN_WINDOW', 'OUT_OF_WINDOW'],
    triggers: [
      { name: 'Window Opens', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
      { name: 'Window Closes', direction: 'BOTH', type: 'EXIT', exec: '[C]' },
    ],
  },
  {
    id: 'session-default',
    name: 'Trading Session',
    category: 'Session',
    enabled: true,
    isDefault: true,
    currentState: 'IN_SESSION',
    params: [
      { key: 'session', label: 'Session', type: 'select', value: 'regular', options: ['pre_market', 'regular', 'after_hours', 'extended'] },
    ],
    outputs: ['IN_SESSION', 'OUT_OF_SESSION'],
    triggers: [
      { name: 'Session Opens', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
      { name: 'Session Closes', direction: 'BOTH', type: 'EXIT', exec: '[C]' },
    ],
  },
  {
    id: 'dow-default',
    name: 'Day of Week',
    category: 'Calendar',
    enabled: true,
    isDefault: true,
    currentState: 'ALLOWED_DAY',
    params: [
      { key: 'monday', label: 'Mon', type: 'bool', value: true },
      { key: 'tuesday', label: 'Tue', type: 'bool', value: true },
      { key: 'wednesday', label: 'Wed', type: 'bool', value: true },
      { key: 'thursday', label: 'Thu', type: 'bool', value: true },
      { key: 'friday', label: 'Fri', type: 'bool', value: true },
    ],
    outputs: ['ALLOWED_DAY', 'BLOCKED_DAY'],
    triggers: [],
  },
  {
    id: 'cal-default',
    name: 'Calendar Filter',
    category: 'Calendar',
    enabled: false,
    isDefault: true,
    currentState: 'CLEAR',
    params: [
      { key: 'avoid_fomc', label: 'Avoid FOMC', type: 'bool', value: true },
      { key: 'avoid_opex', label: 'Avoid OpEx', type: 'bool', value: false },
      { key: 'avoid_nfp', label: 'Avoid NFP', type: 'bool', value: true },
      { key: 'buffer_minutes', label: 'Buffer (min)', type: 'int', value: 30, min: 0, max: 120 },
    ],
    outputs: ['CLEAR', 'BLOCKED'],
    triggers: [
      { name: 'Event Block Starts', direction: 'BOTH', type: 'EXIT', exec: '[C]' },
      { name: 'Event Block Clears', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
    ],
  },
];

/* ─────────────────────────── SVG Icons ─────────────────────────── */

function ClockIcon({ size = 64, hour = 10, minute = 30 }: { size?: number; hour?: number; minute?: number }) {
  const cx = size / 2;
  const cy = size / 2;
  const r = (size - 8) / 2;
  const hourAngle = ((hour % 12) + minute / 60) * 30 - 90;
  const minAngle = minute * 6 - 90;
  const hourLen = r * 0.5;
  const minLen = r * 0.7;
  const hx = cx + hourLen * Math.cos((hourAngle * Math.PI) / 180);
  const hy = cy + hourLen * Math.sin((hourAngle * Math.PI) / 180);
  const mx = cx + minLen * Math.cos((minAngle * Math.PI) / 180);
  const my = cy + minLen * Math.sin((minAngle * Math.PI) / 180);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} fill="none">
      <circle cx={cx} cy={cy} r={r} stroke="var(--accent)" strokeWidth="2" opacity="0.3" />
      {/* Hour ticks */}
      {Array.from({ length: 12 }, (_, i) => {
        const a = i * 30 - 90;
        const x1 = cx + (r - 3) * Math.cos((a * Math.PI) / 180);
        const y1 = cy + (r - 3) * Math.sin((a * Math.PI) / 180);
        const x2 = cx + r * Math.cos((a * Math.PI) / 180);
        const y2 = cy + r * Math.sin((a * Math.PI) / 180);
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--text-muted)" strokeWidth="1" opacity="0.5" />;
      })}
      <line x1={cx} y1={cy} x2={hx} y2={hy} stroke="var(--text-primary)" strokeWidth="2.5" strokeLinecap="round" />
      <line x1={cx} y1={cy} x2={mx} y2={my} stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="2" fill="var(--accent)" />
    </svg>
  );
}

function SessionIcon({ size = 64 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      {/* Pre-market */}
      <rect x="4" y="38" width="12" height="14" rx="2" fill="var(--orange)" opacity="0.3" />
      {/* Regular */}
      <rect x="18" y="18" width="20" height="34" rx="2" fill="var(--green)" opacity="0.6" />
      {/* After-hours */}
      <rect x="40" y="30" width="12" height="22" rx="2" fill="var(--blue)" opacity="0.3" />
      {/* Extended */}
      <rect x="54" y="42" width="6" height="10" rx="1" fill="var(--accent)" opacity="0.2" />
      {/* Labels */}
      <text x="10" y="36" textAnchor="middle" fontSize="5" fill="var(--text-muted)">PRE</text>
      <text x="28" y="16" textAnchor="middle" fontSize="5" fill="var(--green)">REG</text>
      <text x="46" y="28" textAnchor="middle" fontSize="5" fill="var(--text-muted)">AH</text>
    </svg>
  );
}

function CalendarIcon({ size = 64 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      <rect x="6" y="12" width="52" height="44" rx="4" stroke="var(--accent)" strokeWidth="1.5" opacity="0.4" />
      <line x1="6" y1="22" x2="58" y2="22" stroke="var(--accent)" strokeWidth="1" opacity="0.3" />
      {/* Day cells */}
      {[0, 1, 2, 3, 4].map((col) => (
        <g key={col}>
          {[0, 1, 2, 3].map((row) => {
            const blocked = (col === 2 && row === 1) || (col === 4 && row === 2);
            return (
              <rect
                key={row}
                x={9 + col * 10}
                y={25 + row * 8}
                width="7"
                height="5"
                rx="1"
                fill={blocked ? 'var(--red)' : 'var(--green)'}
                opacity={blocked ? 0.6 : 0.25}
              />
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function DayOfWeekIcon({ size = 64, activeDays = [true, true, true, true, true] }: { size?: number; activeDays?: boolean[] }) {
  const labels = ['M', 'T', 'W', 'T', 'F'];
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      {labels.map((label, i) => {
        const x = 6 + i * 11;
        const active = activeDays[i];
        return (
          <g key={i}>
            <rect
              x={x}
              y="20"
              width="9"
              height="28"
              rx="3"
              fill={active ? 'var(--green)' : 'var(--red)'}
              opacity={active ? 0.5 : 0.2}
            />
            <text
              x={x + 4.5}
              y="16"
              textAnchor="middle"
              fontSize="7"
              fill={active ? 'var(--green)' : 'var(--red)'}
              fontWeight="bold"
              opacity={active ? 1 : 0.4}
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function getPackIcon(pack: GeneralPack, size = 64) {
  if (pack.id === 'tod-default') return <ClockIcon size={size} hour={Number(pack.params[0]?.value) || 9} minute={Number(pack.params[1]?.value) || 30} />;
  if (pack.id === 'session-default') return <SessionIcon size={size} />;
  if (pack.id === 'dow-default') {
    const days = pack.params.filter((p) => p.type === 'bool').map((p) => p.value as boolean);
    return <DayOfWeekIcon size={size} activeDays={days} />;
  }
  if (pack.id === 'cal-default') return <CalendarIcon size={size} />;
  return <ClockIcon size={size} />;
}

/* ─────────────────────────── Helpers ─────────────────────────── */

const categoryColors: Record<string, { color: string; bg: string }> = {
  Session: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Calendar: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

function getStateColor(state: string): string {
  if (['IN_WINDOW', 'IN_SESSION', 'ALLOWED_DAY', 'CLEAR'].includes(state)) return 'var(--green)';
  return 'var(--red)';
}

/* ─────────────────────────── Live Clock ─────────────────────────── */

function LiveClock() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const h = time.getHours();
  const m = time.getMinutes();
  const s = time.getSeconds();
  const formatted = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;

  // Determine session
  let session = 'Out of Session';
  let sessionColor = 'var(--text-muted)';
  if (h >= 4 && h < 9 || (h === 9 && m < 30)) {
    session = 'Pre-Market';
    sessionColor = 'var(--orange)';
  } else if ((h === 9 && m >= 30) || (h > 9 && h < 16)) {
    session = 'Regular Hours';
    sessionColor = 'var(--green)';
  } else if (h >= 16 && h < 20) {
    session = 'After Hours';
    sessionColor = 'var(--blue)';
  }

  return (
    <div className="flex items-center gap-3">
      <div className="relative">
        <div className="w-3 h-3 rounded-full" style={{ background: sessionColor }} />
        <div
          className="w-3 h-3 rounded-full absolute top-0 left-0"
          style={{ background: sessionColor, animation: 'pulse-ring 2s ease-out infinite' }}
        />
      </div>
      <div>
        <span className="text-lg font-mono font-bold" style={{ color: 'var(--text-primary)' }}>{formatted}</span>
        <span className="text-xs ml-2" style={{ color: sessionColor }}>{session}</span>
      </div>
    </div>
  );
}

/* ─────────────────────────── Timeline Visualization ─────────────────────────── */

function DayTimeline({ packs }: { packs: GeneralPack[] }) {
  const totalMinutes = 24 * 60;
  const todPack = packs.find((p) => p.id === 'tod-default');
  const sessionPack = packs.find((p) => p.id === 'session-default');

  // Extract windows from ToD pack
  const todStart = todPack
    ? (Number(todPack.params[0]?.value) || 0) * 60 + (Number(todPack.params[1]?.value) || 0)
    : 0;
  const todEnd = todPack
    ? (Number(todPack.params[2]?.value) || 0) * 60 + (Number(todPack.params[3]?.value) || 0)
    : 0;

  // Session times (approximate for visual)
  const sessionType = sessionPack ? String(sessionPack.params[0]?.value) : 'regular';
  const sessionRanges: Record<string, [number, number]> = {
    pre_market: [4 * 60, 9 * 60 + 30],
    regular: [9 * 60 + 30, 16 * 60],
    after_hours: [16 * 60, 20 * 60],
    extended: [4 * 60, 20 * 60],
  };
  const [sessStart, sessEnd] = sessionRanges[sessionType] || [9 * 60 + 30, 16 * 60];

  // Current time marker
  const now = new Date();
  const nowMinutes = now.getHours() * 60 + now.getMinutes();
  const nowPct = (nowMinutes / totalMinutes) * 100;

  // Time labels
  const hourLabels = [0, 4, 8, 9.5, 12, 16, 20, 24];

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
        24-Hour Activity Timeline
      </h3>
      <div className="space-y-4">
        {/* Hour axis labels */}
        <div className="relative h-4">
          {hourLabels.map((h) => (
            <span
              key={h}
              className="absolute text-[9px] font-mono -translate-x-1/2"
              style={{ left: `${(h / 24) * 100}%`, color: 'var(--text-muted)' }}
            >
              {h === 9.5 ? '9:30' : `${h}:00`}
            </span>
          ))}
        </div>

        {/* Trading Session band */}
        {sessionPack?.enabled && (
          <div className="relative">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-medium w-24" style={{ color: 'var(--blue)' }}>Trading Session</span>
              <div className="flex-1 relative h-5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <div
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    left: `${(sessStart / totalMinutes) * 100}%`,
                    width: `${((sessEnd - sessStart) / totalMinutes) * 100}%`,
                    background: 'var(--blue)',
                    opacity: 0.35,
                  }}
                />
                {/* Now marker */}
                <div
                  className="absolute top-0 h-full w-0.5"
                  style={{ left: `${nowPct}%`, background: 'var(--text-primary)', opacity: 0.8 }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Time of Day window */}
        {todPack?.enabled && (
          <div className="relative">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-medium w-24" style={{ color: 'var(--green)' }}>Time Window</span>
              <div className="flex-1 relative h-5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <div
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    left: `${(todStart / totalMinutes) * 100}%`,
                    width: `${((todEnd - todStart) / totalMinutes) * 100}%`,
                    background: 'var(--green)',
                    opacity: 0.35,
                  }}
                />
                <div
                  className="absolute top-0 h-full w-0.5"
                  style={{ left: `${nowPct}%`, background: 'var(--text-primary)', opacity: 0.8 }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Overlap zone */}
        {todPack?.enabled && sessionPack?.enabled && (
          <div className="relative">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[10px] font-medium w-24" style={{ color: 'var(--accent)' }}>Overlap (Active)</span>
              <div className="flex-1 relative h-5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                {(() => {
                  const overlapStart = Math.max(todStart, sessStart);
                  const overlapEnd = Math.min(todEnd, sessEnd);
                  if (overlapEnd <= overlapStart) return null;
                  return (
                    <div
                      className="absolute top-0 h-full rounded-full"
                      style={{
                        left: `${(overlapStart / totalMinutes) * 100}%`,
                        width: `${((overlapEnd - overlapStart) / totalMinutes) * 100}%`,
                        background: 'linear-gradient(90deg, var(--green), var(--accent), var(--blue))',
                        opacity: 0.5,
                      }}
                    />
                  );
                })()}
                <div
                  className="absolute top-0 h-full w-0.5"
                  style={{ left: `${nowPct}%`, background: 'var(--text-primary)', opacity: 0.8 }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="flex items-center gap-4 text-[10px] pt-1" style={{ color: 'var(--text-muted)' }}>
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ background: 'var(--text-primary)' }} />
            Now
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-2 rounded" style={{ background: 'var(--green)', opacity: 0.4 }} />
            Trade Window
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-2 rounded" style={{ background: 'var(--blue)', opacity: 0.4 }} />
            Session
          </span>
          <span className="flex items-center gap-1">
            <div className="w-4 h-2 rounded" style={{ background: 'var(--accent)', opacity: 0.4 }} />
            Active Overlap
          </span>
        </div>
      </div>
    </Card>
  );
}

/* ─────────────────────────── Week View ─────────────────────────── */

function WeekView({ packs }: { packs: GeneralPack[] }) {
  const dowPack = packs.find((p) => p.id === 'dow-default');
  const calPack = packs.find((p) => p.id === 'cal-default');
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
  const dayKeys = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'];

  // Mock upcoming events
  const events: Record<string, string> = { Wed: 'FOMC', Fri: 'NFP' };

  const today = new Date().getDay(); // 0=Sun, 1=Mon...
  const todayIdx = today >= 1 && today <= 5 ? today - 1 : -1;

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-secondary)' }}>
        Week at a Glance
      </h3>
      <div className="grid grid-cols-5 gap-2">
        {days.map((day, i) => {
          const allowed = dowPack ? dowPack.params.find((p) => p.key === dayKeys[i])?.value === true : true;
          const event = events[day];
          const blocked = event && calPack?.enabled && calPack.params.find((p) => {
            if (event === 'FOMC' && p.key === 'avoid_fomc') return p.value === true;
            if (event === 'NFP' && p.key === 'avoid_nfp') return p.value === true;
            return false;
          });
          const isToday = i === todayIdx;

          return (
            <div
              key={day}
              className="rounded-xl p-3 text-center transition-all relative overflow-hidden"
              style={{
                background: isToday ? `${allowed && !blocked ? 'var(--green)' : 'var(--red)'}10` : 'var(--bg-input)',
                border: isToday ? `2px solid ${allowed && !blocked ? 'var(--green)' : 'var(--red)'}60` : '1px solid var(--border)',
              }}
            >
              <span
                className="text-sm font-bold block mb-1"
                style={{ color: allowed && !blocked ? 'var(--text-primary)' : 'var(--text-muted)' }}
              >
                {day}
              </span>
              {isToday && (
                <span className="text-[8px] font-semibold px-1.5 py-0.5 rounded-full" style={{ background: 'var(--accent)', color: 'white' }}>
                  TODAY
                </span>
              )}
              <div className="mt-2">
                {allowed && !blocked ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2" strokeLinecap="round" className="mx-auto">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--red)" strokeWidth="2" strokeLinecap="round" className="mx-auto">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                )}
              </div>
              {!allowed && (
                <span className="text-[8px] mt-1 block" style={{ color: 'var(--red)' }}>
                  Day Off
                </span>
              )}
              {event && (
                <span
                  className="text-[8px] mt-1 block font-mono"
                  style={{ color: blocked ? 'var(--red)' : 'var(--orange)' }}
                >
                  {event}{blocked ? ' [BLOCKED]' : ''}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/* ─────────────────────────── State Pulse ─────────────────────────── */

function StatePulse({ state, pack }: { state: string; pack: GeneralPack }) {
  const color = getStateColor(state);
  const label = state.replace(/_/g, ' ');
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
        <div
          className="w-2.5 h-2.5 rounded-full absolute top-0 left-0"
          style={{ background: color, animation: 'pulse-ring 2s ease-out infinite' }}
        />
      </div>
      <span className="text-xs font-mono font-semibold" style={{ color }}>{state}</span>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  );
}

/* ─────────────────────────── Pack Detail Modal ─────────────────────────── */

function PackDetailModal({ pack, onClose, onUpdate }: { pack: GeneralPack; onClose: () => void; onUpdate: (id: string, params: PackParam[]) => void }) {
  const [localParams, setLocalParams] = useState<PackParam[]>(pack.params);
  const [detailTab, setDetailTab] = useState<'overview' | 'params' | 'triggers'>('overview');

  const updateParam = (key: string, val: number | boolean | string) => {
    setLocalParams((prev) => prev.map((p) => (p.key === key ? { ...p, value: val } : p)));
  };

  return (
    <Modal title="" isOpen onClose={onClose} width="700px">
      {/* Custom header with icon */}
      <div className="flex items-start gap-4 -mt-2 mb-6">
        <div
          className="w-20 h-20 rounded-2xl flex items-center justify-center flex-shrink-0"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          {getPackIcon(pack, 56)}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-xl font-bold">{pack.name}</h2>
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                color: categoryColors[pack.category]?.color,
                background: categoryColors[pack.category]?.bg,
              }}
            >
              {pack.category}
            </span>
            {pack.isDefault && (
              <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
                DEFAULT
              </span>
            )}
          </div>
          <StatePulse state={pack.currentState} pack={pack} />
        </div>
      </div>

      {/* Tab pills */}
      <div className="flex gap-1 mb-5 p-1 rounded-xl" style={{ background: 'var(--bg-input)' }}>
        {(['overview', 'params', 'triggers'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setDetailTab(tab)}
            className="flex-1 py-2 text-xs font-medium rounded-lg transition-all"
            style={{
              background: detailTab === tab ? 'var(--bg-card)' : 'transparent',
              color: detailTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: detailTab === tab ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {detailTab === 'overview' && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="Current State" value={pack.currentState} delta={getStateColor(pack.currentState) === 'var(--green)' ? 'Active' : 'Blocked'} positive={getStateColor(pack.currentState) === 'var(--green)'} />
            <MetricCard label="Trigger Count" value={String(pack.triggers.length)} delta={pack.triggers.length > 0 ? 'Has transitions' : 'Condition only'} positive={pack.triggers.length > 0} />
          </div>
          {/* Visual preview */}
          <div className="rounded-xl p-8 flex items-center justify-center" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            {getPackIcon(pack, 120)}
          </div>
          {/* Output badges */}
          <div>
            <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Output States</h4>
            <div className="flex flex-wrap gap-1.5">
              {pack.outputs.map((output) => {
                const isActive = output === pack.currentState;
                return (
                  <span
                    key={output}
                    className="text-xs font-mono px-2.5 py-1 rounded-lg transition-all"
                    style={{
                      background: isActive ? getStateColor(output) : 'var(--bg-input)',
                      color: isActive ? 'white' : 'var(--text-secondary)',
                      boxShadow: isActive ? `0 0 12px ${getStateColor(output)}40` : 'none',
                      border: isActive ? 'none' : '1px solid var(--border)',
                    }}
                  >
                    {output}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {detailTab === 'params' && (
        <div className="space-y-4">
          {localParams.map((param) => (
            <div key={param.key}>
              {param.type === 'bool' ? (
                <div
                  className="flex items-center justify-between px-4 py-3 rounded-xl"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                >
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{param.label}</span>
                  <button
                    onClick={() => updateParam(param.key, !param.value)}
                    className="w-10 h-6 rounded-full relative transition-colors"
                    style={{
                      background: param.value ? 'var(--accent)' : 'var(--bg-card)',
                      border: param.value ? 'none' : '1px solid var(--border)',
                    }}
                  >
                    <div
                      className="w-4 h-4 rounded-full absolute top-1 transition-all"
                      style={{
                        background: param.value ? 'white' : 'var(--text-muted)',
                        left: param.value ? '22px' : '4px',
                      }}
                    />
                  </button>
                </div>
              ) : param.type === 'select' ? (
                <div
                  className="px-4 py-3 rounded-xl"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                >
                  <label className="text-xs mb-2 block" style={{ color: 'var(--text-muted)' }}>{param.label}</label>
                  <div className="flex gap-2">
                    {param.options?.map((opt) => (
                      <button
                        key={opt}
                        onClick={() => updateParam(param.key, opt)}
                        className="flex-1 py-2 rounded-lg text-xs font-medium transition-all"
                        style={{
                          background: param.value === opt ? 'var(--accent)' : 'var(--bg-card)',
                          color: param.value === opt ? 'white' : 'var(--text-secondary)',
                          border: param.value === opt ? 'none' : '1px solid var(--border)',
                        }}
                      >
                        {opt.replace(/_/g, ' ')}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div
                  className="flex items-center gap-3 px-4 py-3 rounded-xl"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                >
                  <label className="text-xs w-24" style={{ color: 'var(--text-muted)' }}>{param.label}</label>
                  <div className="flex-1 relative h-8 flex items-center">
                    <div className="absolute inset-x-0 h-1.5 rounded-full" style={{ background: 'var(--bg-primary)' }} />
                    {(() => {
                      const min = param.min ?? 0;
                      const max = param.max ?? 100;
                      const pct = ((Number(param.value) - min) / (max - min)) * 100;
                      return (
                        <>
                          <div className="absolute left-0 h-1.5 rounded-full" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
                          <input
                            type="range"
                            min={min}
                            max={max}
                            step={param.type === 'float' ? 0.1 : 1}
                            value={Number(param.value)}
                            onChange={(e) => updateParam(param.key, Number(e.target.value))}
                            className="absolute inset-x-0 w-full h-full opacity-0 cursor-pointer"
                            style={{ zIndex: 2 }}
                          />
                          <div
                            className="absolute w-4 h-4 rounded-full border-2 shadow-lg pointer-events-none"
                            style={{ left: `calc(${pct}% - 8px)`, background: 'var(--bg-card)', borderColor: 'var(--accent)' }}
                          />
                        </>
                      );
                    })()}
                  </div>
                  <span className="text-sm font-mono w-10 text-center rounded-md py-0.5" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
                    {String(param.value)}
                  </span>
                </div>
              )}
            </div>
          ))}
          <div className="flex gap-3 pt-2">
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
              style={{ background: 'var(--accent)', color: 'white' }}
              onClick={() => { onUpdate(pack.id, localParams); onClose(); }}
            >
              Save Parameters
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
              onClick={() => setLocalParams(pack.params)}
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {detailTab === 'triggers' && (
        <div className="space-y-2">
          {pack.triggers.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>This pack is condition-only (no triggers)</p>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>It gates other triggers but does not generate signals by itself.</p>
            </div>
          ) : (
            pack.triggers.map((trigger, i) => (
              <div
                key={i}
                className="flex items-center gap-3 px-4 py-3 rounded-xl"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-sm"
                  style={{
                    background: trigger.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)',
                    color: trigger.type === 'ENTRY' ? 'var(--green)' : 'var(--red)',
                  }}
                >
                  {trigger.type === 'ENTRY' ? '\u2191' : '\u2193'}
                </div>
                <div className="flex-1">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: trigger.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)', color: trigger.type === 'ENTRY' ? 'var(--green)' : 'var(--red)' }}>
                      {trigger.type}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      {trigger.exec}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </Modal>
  );
}

/* ═══════════════════════════ Main Component ═══════════════════════════ */

export default function GeneralPacksV4() {
  const [packs, setPacks] = useState<GeneralPack[]>(initialPacks);
  const [detailPack, setDetailPack] = useState<GeneralPack | null>(null);
  const [showNewModal, setShowNewModal] = useState(false);
  const [filterCategory, setFilterCategory] = useState<'All' | 'Session' | 'Calendar'>('All');

  const togglePack = useCallback((id: string) => {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)));
  }, []);

  const updatePackParams = useCallback((id: string, params: PackParam[]) => {
    setPacks((prev) => prev.map((p) => (p.id === id ? { ...p, params } : p)));
  }, []);

  const enabledCount = packs.filter((p) => p.enabled).length;
  const conditionOnlyCount = packs.filter((p) => p.triggers.length === 0).length;
  const totalTriggers = packs.reduce((sum, p) => sum + p.triggers.length, 0);

  const filtered = filterCategory === 'All' ? packs : packs.filter((p) => p.category === filterCategory);

  const grouped = useMemo(() => {
    const map = new Map<string, GeneralPack[]>();
    filtered.forEach((p) => {
      const key = p.category;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(p);
    });
    return map;
  }, [filtered]);

  return (
    <div>
      <style>{`
        @keyframes pulse-ring {
          0% { transform: scale(1); opacity: 0.6; }
          100% { transform: scale(2.5); opacity: 0; }
        }
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes tick {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        .pack-card { animation: fade-in-up 0.4s ease forwards; }
        .pack-card:nth-child(1) { animation-delay: 0.0s; }
        .pack-card:nth-child(2) { animation-delay: 0.05s; }
        .pack-card:nth-child(3) { animation-delay: 0.1s; }
        .pack-card:nth-child(4) { animation-delay: 0.15s; }
      `}</style>

      <PageHeader
        title="General Packs"
        subtitle="Time, session, and calendar conditions -- see when trading is active at a glance"
        actions={
          <div className="flex items-center gap-3">
            <LiveClock />
            <button
              onClick={() => setShowNewModal(true)}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              + New Pack
            </button>
          </div>
        }
      />

      {/* Summary Dashboard */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Total Packs" value={String(packs.length)} delta={`${enabledCount} enabled`} positive />
        <MetricCard label="Total Triggers" value={String(totalTriggers)} delta={`${conditionOnlyCount} condition-only`} positive />
        <MetricCard
          label="Active State"
          value={
            packs.filter((p) => p.enabled && getStateColor(p.currentState) === 'var(--green)').length === enabledCount
              ? 'All Clear'
              : 'Blocked'
          }
          delta={`${packs.filter((p) => p.enabled && getStateColor(p.currentState) === 'var(--green)').length}/${enabledCount} passing`}
          positive={packs.filter((p) => p.enabled && getStateColor(p.currentState) === 'var(--green)').length === enabledCount}
        />
        <MetricCard label="Net Result" value={packs.every((p) => !p.enabled || getStateColor(p.currentState) === 'var(--green)') ? 'TRADE' : 'WAIT'} delta="Combined gate" positive={packs.every((p) => !p.enabled || getStateColor(p.currentState) === 'var(--green)')} />
      </div>

      {/* Timeline Visualization */}
      <div className="mb-6">
        <DayTimeline packs={packs} />
      </div>

      {/* Week View */}
      <div className="mb-6">
        <WeekView packs={packs} />
      </div>

      {/* Category filter */}
      <div className="flex items-center gap-2 mb-5">
        {(['All', 'Session', 'Calendar'] as const).map((cat) => (
          <button
            key={cat}
            onClick={() => setFilterCategory(cat)}
            className="px-4 py-2 rounded-lg text-xs font-medium transition-all"
            style={{
              background: filterCategory === cat ? (cat === 'All' ? 'var(--accent)' : categoryColors[cat]?.color || 'var(--accent)') : 'var(--bg-input)',
              color: filterCategory === cat ? 'white' : 'var(--text-secondary)',
              border: filterCategory === cat ? 'none' : '1px solid var(--border)',
            }}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Pack Cards */}
      <div className="space-y-8">
        {Array.from(grouped.entries()).map(([groupName, groupPacks]) => (
          <div key={groupName}>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-1.5 h-5 rounded-full" style={{ background: categoryColors[groupName]?.color || 'var(--accent)' }} />
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>{groupName}</h3>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{groupPacks.length} pack{groupPacks.length !== 1 ? 's' : ''}</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {groupPacks.map((pack) => (
                <div key={pack.id} className="pack-card">
                  <Card className={`relative overflow-hidden ${!pack.enabled ? 'opacity-60' : ''}`}>
                    {/* Top accent line */}
                    <div
                      className="absolute top-0 left-0 right-0 h-0.5"
                      style={{
                        background: pack.enabled
                          ? `linear-gradient(90deg, ${categoryColors[pack.category]?.color || 'var(--accent)'}, transparent)`
                          : 'var(--border)',
                      }}
                    />

                    <div className="flex gap-4">
                      {/* Indicator Icon */}
                      <div
                        className="w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0 cursor-pointer transition-transform hover:scale-105"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                        onClick={() => setDetailPack(pack)}
                      >
                        {getPackIcon(pack, 48)}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-1.5">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <button
                                onClick={() => setDetailPack(pack)}
                                className="font-semibold text-sm hover:underline"
                                style={{ color: 'var(--text-primary)' }}
                              >
                                {pack.name}
                              </button>
                              <span
                                className="text-[10px] px-1.5 py-0.5 rounded"
                                style={{ background: categoryColors[pack.category]?.bg, color: categoryColors[pack.category]?.color }}
                              >
                                {pack.category}
                              </span>
                            </div>
                            <StatePulse state={pack.currentState} pack={pack} />
                          </div>

                          {/* Toggle */}
                          <button
                            onClick={() => togglePack(pack.id)}
                            className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                            style={{
                              background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)',
                              border: pack.enabled ? 'none' : '1px solid var(--border)',
                            }}
                          >
                            <div
                              className="w-4 h-4 rounded-full absolute top-1 transition-all"
                              style={{
                                background: pack.enabled ? 'white' : 'var(--text-muted)',
                                left: pack.enabled ? '22px' : '4px',
                              }}
                            />
                          </button>
                        </div>

                        {/* Stats row */}
                        <div className="flex items-center gap-4 mt-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                          <span className="flex items-center gap-1">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                            </svg>
                            {pack.triggers.length} triggers
                          </span>
                          <span className="flex items-center gap-1">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <circle cx="12" cy="12" r="10" />
                              <line x1="12" y1="8" x2="12" y2="12" />
                              <line x1="12" y1="12" x2="16" y2="12" />
                            </svg>
                            {pack.outputs.length} states
                          </span>
                        </div>

                        {/* Params preview */}
                        <div className="flex flex-wrap gap-1.5 mt-2.5">
                          {pack.params.map((p) => (
                            <span
                              key={p.key}
                              className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
                            >
                              {p.label}: {p.type === 'bool' ? (p.value ? 'ON' : 'OFF') : String(p.value)}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Detail Modal */}
      {detailPack && (
        <PackDetailModal
          pack={detailPack}
          onClose={() => setDetailPack(null)}
          onUpdate={updatePackParams}
        />
      )}

      {/* New Pack Modal */}
      <Modal title="Create General Pack" isOpen={showNewModal} onClose={() => setShowNewModal(false)} width="520px">
        <NewPackForm onClose={() => setShowNewModal(false)} />
      </Modal>
    </div>
  );
}

/* ─────────────────────────── New Pack Form ─────────────────────────── */

function NewPackForm({ onClose }: { onClose: () => void }) {
  const [template, setTemplate] = useState('Time of Day');
  const [versionName, setVersionName] = useState('');

  const templates = [
    { name: 'Time of Day', category: 'Session', desc: 'Define custom trading windows based on clock time' },
    { name: 'Trading Session', category: 'Session', desc: 'Gate trades by pre-market, regular, or after-hours' },
    { name: 'Day of Week', category: 'Calendar', desc: 'Block specific days from trading' },
    { name: 'Calendar Filter', category: 'Calendar', desc: 'Avoid high-impact events like FOMC, NFP, OpEx' },
  ];

  const selected = templates.find((t) => t.name === template);

  return (
    <div className="space-y-5">
      <div>
        <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Template</label>
        <div className="grid grid-cols-1 gap-2">
          {templates.map((t) => {
            const isSelected = template === t.name;
            return (
              <button
                key={t.name}
                onClick={() => setTemplate(t.name)}
                className="flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all"
                style={{
                  background: isSelected ? `${categoryColors[t.category]?.color}10` : 'var(--bg-input)',
                  border: isSelected ? `1px solid ${categoryColors[t.category]?.color}60` : '1px solid var(--border)',
                }}
              >
                <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: isSelected ? categoryColors[t.category]?.bg : 'var(--bg-primary)' }}>
                  {t.name === 'Time of Day' && <ClockIcon size={32} />}
                  {t.name === 'Trading Session' && <SessionIcon size={32} />}
                  {t.name === 'Day of Week' && <DayOfWeekIcon size={32} />}
                  {t.name === 'Calendar Filter' && <CalendarIcon size={32} />}
                </div>
                <div>
                  <span className="text-sm font-medium" style={{ color: isSelected ? categoryColors[t.category]?.color : 'var(--text-primary)' }}>{t.name}</span>
                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{t.desc}</p>
                </div>
                {isSelected && (
                  <svg className="ml-auto" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={categoryColors[t.category]?.color} strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Version Name</label>
        <input
          className="w-full px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="e.g. Morning Only, No Fridays..."
          value={versionName}
          onChange={(e) => setVersionName(e.target.value)}
        />
      </div>

      <div
        className="rounded-xl p-4 flex items-center gap-4"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div className="w-12 h-12 rounded-lg flex items-center justify-center" style={{ background: 'var(--bg-primary)' }}>
          {selected?.name === 'Time of Day' && <ClockIcon size={36} />}
          {selected?.name === 'Trading Session' && <SessionIcon size={36} />}
          {selected?.name === 'Day of Week' && <DayOfWeekIcon size={36} />}
          {selected?.name === 'Calendar Filter' && <CalendarIcon size={36} />}
        </div>
        <div>
          <p className="text-sm font-medium">{selected?.name || template}</p>
          <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
            {(selected?.name || template).toLowerCase().replace(/\s+/g, '-')}-{versionName ? versionName.toLowerCase().replace(/\s+/g, '-') : 'custom'}
          </p>
        </div>
      </div>

      <div className="flex gap-3 pt-1">
        <button
          className="px-4 py-2.5 rounded-lg text-sm font-medium flex-1"
          style={{ background: 'var(--accent)', color: 'white' }}
          onClick={onClose}
        >
          Create Pack
        </button>
        <button
          className="px-4 py-2.5 rounded-lg text-sm flex-1"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          onClick={onClose}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
