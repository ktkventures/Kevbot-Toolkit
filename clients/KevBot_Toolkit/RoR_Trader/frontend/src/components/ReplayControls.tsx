'use client';

/**
 * ReplayControls — compact toolbar for scenario replay scrubbing.
 *
 * Layout: [◀] [HH:MM:SS] [▶]  Step: [1s ▾]
 *         [═══════════░░░░░░░] ← thin progress bar (clickable)
 */

interface ReplayControlsProps {
  currentTime: number;   // Unix seconds
  startTime: number;
  endTime: number;
  interval: number;      // 1, 5, 10, or 60
  isAtStart: boolean;
  isAtEnd: boolean;
  onStepBack: () => void;
  onStepForward: () => void;
  onSeek: (time: number) => void;
  onIntervalChange: (n: number) => void;
  onReset: () => void;
}

function formatTime(unixSec: number): string {
  const d = new Date(unixSec * 1000);
  return d.toISOString().slice(11, 19); // HH:MM:SS in UTC
}

const INTERVALS = [1, 5, 10, 60] as const;

const btnStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  borderRadius: '4px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 24,
  height: 24,
  fontSize: '11px',
  padding: 0,
  flexShrink: 0,
};

const disabledBtnStyle: React.CSSProperties = {
  ...btnStyle,
  opacity: 0.35,
  cursor: 'default',
};

export default function ReplayControls({
  currentTime, startTime, endTime, interval,
  isAtStart, isAtEnd,
  onStepBack, onStepForward, onSeek, onIntervalChange, onReset,
}: ReplayControlsProps) {
  const progress = endTime > startTime
    ? (currentTime - startTime) / (endTime - startTime)
    : 0;

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(Math.round(startTime + frac * (endTime - startTime)));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {/* Controls row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'nowrap' }}>
        {/* Reset */}
        <button onClick={onReset} style={isAtStart ? disabledBtnStyle : btnStyle} disabled={isAtStart} title="Reset">
          &#x23EE;
        </button>
        {/* Step back */}
        <button onClick={onStepBack} style={isAtStart ? disabledBtnStyle : btnStyle} disabled={isAtStart} title="Step back">
          &#x25C0;
        </button>
        {/* Time display */}
        <span
          className="text-[10px] font-mono"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            borderRadius: '4px',
            padding: '2px 6px',
            color: 'var(--text-primary)',
            minWidth: 64,
            textAlign: 'center',
            flexShrink: 0,
          }}
        >
          {formatTime(currentTime)}
        </span>
        {/* Step forward */}
        <button onClick={onStepForward} style={isAtEnd ? disabledBtnStyle : btnStyle} disabled={isAtEnd} title="Step forward">
          &#x25B6;
        </button>
        {/* Skip to end */}
        <button onClick={() => onSeek(endTime)} style={isAtEnd ? disabledBtnStyle : btnStyle} disabled={isAtEnd} title="Skip to end">
          &#x23ED;
        </button>

        {/* Spacer */}
        <div style={{ width: 8 }} />

        {/* Interval selector */}
        <span className="text-[9px]" style={{ color: 'var(--text-muted)', flexShrink: 0 }}>Step:</span>
        <select
          value={interval}
          onChange={e => onIntervalChange(Number(e.target.value))}
          className="text-[10px]"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
            borderRadius: '4px',
            padding: '2px 4px',
            cursor: 'pointer',
            flexShrink: 0,
          }}
        >
          {INTERVALS.map(v => (
            <option key={v} value={v}>{v}s</option>
          ))}
        </select>
      </div>

      {/* Progress bar */}
      <div
        onClick={handleProgressClick}
        style={{
          width: '100%',
          height: 4,
          borderRadius: 2,
          background: 'var(--bg-input)',
          cursor: 'pointer',
          position: 'relative',
          overflow: 'hidden',
        }}
        title={`${Math.round(progress * 100)}%`}
      >
        <div
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            height: '100%',
            width: `${progress * 100}%`,
            background: 'var(--accent)',
            borderRadius: 2,
            transition: 'width 0.05s linear',
          }}
        />
      </div>
    </div>
  );
}
