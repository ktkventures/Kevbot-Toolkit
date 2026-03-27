'use client';

/**
 * Confluence condition heatmap — shows green/red blocks per condition per bar.
 * Renders as a simple HTML grid synchronized to the visible bars in the price chart.
 */

interface HeatmapCondition {
  label: string;
  column: string;
  neededState: string;
}

interface HeatmapProps {
  /** Chart data bars (each bar has _state_{column} fields) */
  bars: Record<string, any>[];
  /** Condition definitions */
  conditions: HeatmapCondition[];
  /** Number of visible bars (rightmost slice) */
  visibleBars?: number;
}

const GREEN = '#4CAF50';
const RED = 'rgba(244,67,54,0.5)';

export default function ConfluenceHeatmap({ bars, conditions, visibleBars }: HeatmapProps) {
  if (conditions.length === 0 || bars.length === 0) return null;

  const slicedBars = visibleBars && visibleBars < bars.length
    ? bars.slice(-visibleBars)
    : bars;

  // Sample bars for performance (max ~300 cells per row)
  const sampleRate = Math.max(1, Math.floor(slicedBars.length / 300));
  const sampledBars = sampleRate > 1
    ? slicedBars.filter((_, i) => i % sampleRate === 0)
    : slicedBars;

  const cellW = Math.max(2, Math.min(6, Math.floor(800 / sampledBars.length)));
  const rowH = 14;
  const labelW = 160;

  return (
    <div style={{ overflowX: 'hidden', marginBottom: 4 }}>
      {conditions.map((cond) => {
        const stateKey = `_state_${cond.column}`;
        return (
          <div key={cond.label} style={{ display: 'flex', alignItems: 'center', height: rowH + 2 }}>
            <div
              style={{
                width: labelW,
                flexShrink: 0,
                fontSize: 9,
                fontFamily: 'monospace',
                color: 'var(--text-muted)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                paddingRight: 6,
                textAlign: 'right',
              }}
              title={cond.label}
            >
              {cond.label}
            </div>
            <div style={{ display: 'flex', gap: 0, flex: 1 }}>
              {sampledBars.map((bar, i) => {
                const stateVal = bar[stateKey];
                const isMet = stateVal != null && stateVal === cond.neededState;
                return (
                  <div
                    key={i}
                    style={{
                      width: cellW,
                      height: rowH,
                      background: isMet ? GREEN : RED,
                      borderRadius: 1,
                    }}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
