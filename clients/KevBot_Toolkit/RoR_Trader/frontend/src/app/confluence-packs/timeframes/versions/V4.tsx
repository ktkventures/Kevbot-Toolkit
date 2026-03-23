'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

// =============================================================================
// V4 Meta: Creative — Timeframe Hierarchy Explorer
// =============================================================================
// Visual timeframe system with:
//
// 1. HIERARCHY TREE — visual parent/child tree showing TF relationships
// 2. BAR COUNT CALCULATOR — how many bars fit in a session per TF
// 3. CROSS-TF DATA FLOW — animated arrows showing confluence data flow
// 4. TF RECOMMENDATION ENGINE — suggests TF combos based on strategy type
// 5. SESSION VISUALIZER — timeline ruler with bar marks per TF
// 6. TF METRICS STRIP — bars/session, avg volatility, data density
// 7. ANIMATED FLOW LINES — SVG arrows between related timeframes
// 8. STRATEGY FIT CARDS — shows which strategies use each TF combo
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-node {
  0%, 100% { box-shadow: 0 0 6px rgba(0,255,136,0.15); }
  50% { box-shadow: 0 0 18px rgba(0,255,136,0.4); }
}
@keyframes flow-dash {
  0% { stroke-dashoffset: 20; }
  100% { stroke-dashoffset: 0; }
}
@keyframes bar-grow {
  0% { transform: scaleY(0); }
  100% { transform: scaleY(1); }
}
@keyframes ripple {
  0% { transform: scale(1); opacity: 0.5; }
  100% { transform: scale(2.5); opacity: 0; }
}
@keyframes slide-in {
  0% { opacity: 0; transform: translateX(-16px); }
  100% { opacity: 1; transform: translateX(0); }
}
@keyframes glow-border {
  0%, 100% { border-color: rgba(0,255,136,0.2); }
  50% { border-color: rgba(0,255,136,0.6); }
}
@keyframes typewriter {
  0% { width: 0; }
  100% { width: 100%; }
}
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TimeframeConfig {
  tf: string;
  label: string;
  seconds: number;
  enabled: boolean;
  primary: boolean;
  barsPerSession: number;
  avgVolatility: number; // avg ATR-based movement per bar
  strategyCount: number; // how many strategies use this TF
}

type StrategyType = 'scalping' | 'dayTrading' | 'swingIntraday' | 'momentum';

interface Recommendation {
  strategyType: StrategyType;
  label: string;
  primaryTf: string;
  confluenceTfs: string[];
  reason: string;
  confidence: number;
}

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const mockTimeframes: TimeframeConfig[] = [
  { tf: '1Min', label: '1 Minute', seconds: 60, enabled: true, primary: true, barsPerSession: 390, avgVolatility: 0.08, strategyCount: 5 },
  { tf: '2Min', label: '2 Minutes', seconds: 120, enabled: false, primary: false, barsPerSession: 195, avgVolatility: 0.12, strategyCount: 1 },
  { tf: '3Min', label: '3 Minutes', seconds: 180, enabled: true, primary: false, barsPerSession: 130, avgVolatility: 0.15, strategyCount: 2 },
  { tf: '5Min', label: '5 Minutes', seconds: 300, enabled: true, primary: false, barsPerSession: 78, avgVolatility: 0.22, strategyCount: 4 },
  { tf: '10Min', label: '10 Minutes', seconds: 600, enabled: false, primary: false, barsPerSession: 39, avgVolatility: 0.31, strategyCount: 0 },
  { tf: '15Min', label: '15 Minutes', seconds: 900, enabled: true, primary: false, barsPerSession: 26, avgVolatility: 0.38, strategyCount: 3 },
  { tf: '30Min', label: '30 Minutes', seconds: 1800, enabled: false, primary: false, barsPerSession: 13, avgVolatility: 0.55, strategyCount: 1 },
  { tf: '1H', label: '1 Hour', seconds: 3600, enabled: true, primary: false, barsPerSession: 7, avgVolatility: 0.72, strategyCount: 2 },
  { tf: '4H', label: '4 Hours', seconds: 14400, enabled: false, primary: false, barsPerSession: 2, avgVolatility: 1.45, strategyCount: 0 },
  { tf: '1D', label: '1 Day', seconds: 86400, enabled: true, primary: false, barsPerSession: 1, avgVolatility: 2.10, strategyCount: 1 },
];

const SESSION_HOURS = 6.5; // RTH: 9:30 - 16:00
const SESSION_MINUTES = SESSION_HOURS * 60;

const RECOMMENDATIONS: Recommendation[] = [
  {
    strategyType: 'scalping',
    label: 'Scalping',
    primaryTf: '1Min',
    confluenceTfs: ['5Min', '15Min'],
    reason: 'Fast entries on 1Min with 5Min trend + 15Min structure for confluence',
    confidence: 92,
  },
  {
    strategyType: 'dayTrading',
    label: 'Day Trading',
    primaryTf: '5Min',
    confluenceTfs: ['15Min', '1H'],
    reason: '5Min gives good signal-to-noise with 15Min trend and 1H structure',
    confidence: 88,
  },
  {
    strategyType: 'swingIntraday',
    label: 'Swing Intraday',
    primaryTf: '15Min',
    confluenceTfs: ['1H', '1D'],
    reason: '15Min for positioning with 1H trend and 1D key levels',
    confidence: 85,
  },
  {
    strategyType: 'momentum',
    label: 'Momentum',
    primaryTf: '3Min',
    confluenceTfs: ['1Min', '5Min'],
    reason: '3Min captures momentum bursts with 1Min timing and 5Min direction',
    confidence: 78,
  },
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Session timeline ruler showing bar divisions */
function SessionTimeline({ timeframes }: { timeframes: TimeframeConfig[] }) {
  const enabledTfs = timeframes.filter((t) => t.enabled);

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
        Session Bar Density
      </h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        RTH Session: 6.5 hours (9:30 AM - 4:00 PM ET) = {SESSION_MINUTES} minutes
      </p>
      <div className="space-y-3">
        {enabledTfs.map((tf, i) => {
          const fillRatio = Math.min(tf.barsPerSession / 400, 1);
          return (
            <div key={tf.tf} style={{ animation: `slide-in 0.4s ease ${i * 0.08}s both` }}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold w-12" style={{ color: tf.primary ? 'var(--accent)' : 'var(--text-primary)' }}>
                    {tf.tf}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{tf.label}</span>
                </div>
                <span className="text-xs font-mono font-bold" style={{ color: 'var(--accent)' }}>
                  {tf.barsPerSession} bars
                </span>
              </div>
              {/* Bar density visualization */}
              <div className="relative h-5 rounded-md overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <div
                  className="absolute inset-y-0 left-0 rounded-md"
                  style={{
                    width: `${fillRatio * 100}%`,
                    background: tf.primary
                      ? 'linear-gradient(90deg, var(--accent), rgba(0,255,136,0.3))'
                      : 'linear-gradient(90deg, var(--text-muted), rgba(128,128,128,0.15))',
                    animation: `bar-grow 0.6s ease ${i * 0.08 + 0.2}s both`,
                    transformOrigin: 'left',
                  }}
                />
                {/* tick marks for individual bars (show up to 40 marks) */}
                {tf.barsPerSession <= 40 && (
                  <div className="absolute inset-0 flex">
                    {Array.from({ length: tf.barsPerSession }).map((_, bi) => (
                      <div
                        key={bi}
                        className="flex-1"
                        style={{
                          borderRight: bi < tf.barsPerSession - 1 ? '1px solid rgba(255,255,255,0.08)' : 'none',
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

/** Hierarchy tree: shows TF relationships */
function HierarchyTree({ timeframes, onToggle, onSetPrimary }: {
  timeframes: TimeframeConfig[];
  onToggle: (tf: string) => void;
  onSetPrimary: (tf: string) => void;
}) {
  const primary = timeframes.find((t) => t.primary);
  const confluenceTfs = timeframes.filter((t) => t.enabled && !t.primary);
  const disabledTfs = timeframes.filter((t) => !t.enabled);

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
        Timeframe Hierarchy
      </h3>

      {/* Primary node */}
      {primary && (
        <div className="flex flex-col items-center mb-2">
          <div
            className="px-5 py-3 rounded-xl border-2 text-center"
            style={{
              borderColor: 'var(--accent)',
              background: 'var(--accent-muted)',
              animation: 'pulse-node 2s ease-in-out infinite',
            }}
          >
            <div className="text-sm font-bold font-mono" style={{ color: 'var(--accent)' }}>
              {primary.tf}
            </div>
            <div className="text-[10px] mt-0.5" style={{ color: 'var(--accent)' }}>
              PRIMARY
            </div>
          </div>

          {/* Animated connector line */}
          {confluenceTfs.length > 0 && (
            <svg width="2" height="28" className="my-1">
              <line x1="1" y1="0" x2="1" y2="28" stroke="var(--accent)" strokeWidth="2" strokeDasharray="4 3" style={{ animation: 'flow-dash 1s linear infinite' }} />
            </svg>
          )}
        </div>
      )}

      {/* Confluence TFs */}
      {confluenceTfs.length > 0 && (
        <div className="mb-4">
          <div className="text-[10px] text-center mb-2" style={{ color: 'var(--text-muted)' }}>
            CONFLUENCE
          </div>
          <div className="flex flex-wrap gap-2 justify-center">
            {confluenceTfs.map((tf, i) => (
              <div
                key={tf.tf}
                className="group relative"
                style={{ animation: `fade-in-up 0.3s ease ${i * 0.06}s both` }}
              >
                <button
                  onClick={() => onSetPrimary(tf.tf)}
                  className="px-3 py-2 rounded-lg border text-center transition-all"
                  style={{
                    borderColor: 'var(--border)',
                    background: 'var(--bg-card)',
                  }}
                  title="Click to set as primary"
                >
                  <div className="text-xs font-bold font-mono" style={{ color: 'var(--text-primary)' }}>
                    {tf.tf}
                  </div>
                  <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                    {tf.barsPerSession}b/s
                  </div>
                </button>
                {/* Disable button */}
                <button
                  onClick={() => onToggle(tf.tf)}
                  className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full flex items-center justify-center text-[8px] opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ background: 'var(--red)', color: 'white' }}
                  title="Disable"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Divider */}
      {disabledTfs.length > 0 && (
        <div className="border-t pt-3 mt-3" style={{ borderColor: 'var(--border)' }}>
          <div className="text-[10px] text-center mb-2" style={{ color: 'var(--text-muted)' }}>
            AVAILABLE
          </div>
          <div className="flex flex-wrap gap-1.5 justify-center">
            {disabledTfs.map((tf, i) => (
              <button
                key={tf.tf}
                onClick={() => onToggle(tf.tf)}
                className="px-2.5 py-1.5 rounded-lg text-xs font-mono transition-all"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px dashed var(--border)',
                  color: 'var(--text-muted)',
                  opacity: 0.6,
                  animation: `fade-in-up 0.3s ease ${i * 0.05}s both`,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--accent)';
                  e.currentTarget.style.color = 'var(--accent)';
                  e.currentTarget.style.opacity = '1';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.color = 'var(--text-muted)';
                  e.currentTarget.style.opacity = '0.6';
                }}
                title="Click to enable"
              >
                + {tf.tf}
              </button>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

/** Cross-TF data flow diagram */
function DataFlowDiagram({ timeframes }: { timeframes: TimeframeConfig[] }) {
  const enabled = timeframes.filter((t) => t.enabled).sort((a, b) => a.seconds - b.seconds);
  const primary = enabled.find((t) => t.primary);
  if (!primary || enabled.length < 2) return null;

  // Group: TFs faster than primary, primary, TFs slower than primary
  const faster = enabled.filter((t) => t.seconds < primary.seconds);
  const slower = enabled.filter((t) => t.seconds > primary.seconds);

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
        Cross-TF Data Flow
      </h3>
      <p className="text-[10px] mb-4" style={{ color: 'var(--text-muted)' }}>
        How confluence data flows from secondary TFs into the primary decision TF
      </p>

      <div className="flex items-center justify-center gap-2 flex-wrap">
        {/* Faster TFs */}
        {faster.map((tf, i) => (
          <div key={tf.tf} className="flex items-center gap-2" style={{ animation: `slide-in 0.3s ease ${i * 0.1}s both` }}>
            <div
              className="px-2 py-1.5 rounded-lg border text-center"
              style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}
            >
              <div className="text-[10px] font-mono font-bold" style={{ color: 'var(--text-secondary)' }}>{tf.tf}</div>
              <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>Timing</div>
            </div>
            <svg width="32" height="12">
              <line x1="0" y1="6" x2="28" y2="6" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="4 2" style={{ animation: 'flow-dash 0.8s linear infinite' }} />
              <polygon points="28,2 34,6 28,10" fill="var(--text-muted)" />
            </svg>
          </div>
        ))}

        {/* Primary (center) */}
        <div
          className="px-4 py-3 rounded-xl border-2 text-center relative"
          style={{
            borderColor: 'var(--accent)',
            background: 'var(--accent-muted)',
            animation: 'pulse-node 2.5s ease-in-out infinite',
          }}
        >
          <div className="text-sm font-bold font-mono" style={{ color: 'var(--accent)' }}>{primary.tf}</div>
          <div className="text-[9px]" style={{ color: 'var(--accent)' }}>Decisions</div>
          {/* Ripple */}
          <div
            className="absolute inset-0 rounded-xl border-2"
            style={{
              borderColor: 'var(--accent)',
              animation: 'ripple 2s ease-out infinite',
              pointerEvents: 'none',
            }}
          />
        </div>

        {/* Slower TFs */}
        {slower.map((tf, i) => (
          <div key={tf.tf} className="flex items-center gap-2" style={{ animation: `slide-in 0.3s ease ${(faster.length + i) * 0.1}s both` }}>
            <svg width="32" height="12">
              <line x1="6" y1="6" x2="32" y2="6" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray="4 2" style={{ animation: 'flow-dash 0.8s linear infinite reverse' }} />
              <polygon points="0,6 6,2 6,10" fill="var(--text-muted)" />
            </svg>
            <div
              className="px-2 py-1.5 rounded-lg border text-center"
              style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}
            >
              <div className="text-[10px] font-mono font-bold" style={{ color: 'var(--text-secondary)' }}>{tf.tf}</div>
              <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>Trend</div>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/** TF recommendation cards based on strategy type */
function RecommendationEngine({ onApply }: { onApply: (primary: string, confluenceTfs: string[]) => void }) {
  const [selectedType, setSelectedType] = useState<StrategyType | null>(null);
  const selected = RECOMMENDATIONS.find((r) => r.strategyType === selectedType);

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
        TF Recommendation Engine
      </h3>
      <p className="text-[10px] mb-3" style={{ color: 'var(--text-muted)' }}>
        Select your strategy style to get optimal timeframe configurations
      </p>

      {/* Strategy type selector */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {RECOMMENDATIONS.map((rec) => (
          <button
            key={rec.strategyType}
            onClick={() => setSelectedType(rec.strategyType)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
            style={{
              background: selectedType === rec.strategyType ? 'var(--accent-muted)' : 'var(--bg-input)',
              color: selectedType === rec.strategyType ? 'var(--accent)' : 'var(--text-muted)',
              border: selectedType === rec.strategyType ? '1px solid var(--accent)' : '1px solid var(--border)',
            }}
          >
            {rec.label}
          </button>
        ))}
      </div>

      {/* Recommendation card */}
      {selected && (
        <div
          className="rounded-lg border p-4"
          style={{
            borderColor: 'var(--accent)',
            background: 'linear-gradient(135deg, var(--accent-muted), transparent)',
            animation: 'fade-in-up 0.3s ease both',
          }}
        >
          <div className="flex items-start justify-between mb-3">
            <div>
              <h4 className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                {selected.label} Configuration
              </h4>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {selected.reason}
              </p>
            </div>
            {/* Confidence gauge */}
            <div className="text-center flex-shrink-0 ml-3">
              <div
                className="w-11 h-11 rounded-full flex items-center justify-center border-2"
                style={{ borderColor: 'var(--accent)' }}
              >
                <span className="text-xs font-bold" style={{ color: 'var(--accent)' }}>
                  {selected.confidence}%
                </span>
              </div>
              <div className="text-[9px] mt-0.5" style={{ color: 'var(--text-muted)' }}>Fit</div>
            </div>
          </div>

          {/* TF layout */}
          <div className="flex items-center gap-3 mb-3">
            <div
              className="px-3 py-2 rounded-lg border-2 text-center"
              style={{ borderColor: 'var(--accent)', background: 'var(--accent-muted)' }}
            >
              <div className="text-xs font-bold font-mono" style={{ color: 'var(--accent)' }}>
                {selected.primaryTf}
              </div>
              <div className="text-[9px]" style={{ color: 'var(--accent)' }}>Primary</div>
            </div>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>+</span>
            {selected.confluenceTfs.map((tf) => (
              <div
                key={tf}
                className="px-2.5 py-1.5 rounded-lg border text-center"
                style={{ borderColor: 'var(--border)', background: 'var(--bg-input)' }}
              >
                <div className="text-xs font-mono font-medium" style={{ color: 'var(--text-secondary)' }}>
                  {tf}
                </div>
                <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Conf.</div>
              </div>
            ))}
          </div>

          <button
            onClick={() => onApply(selected.primaryTf, selected.confluenceTfs)}
            className="w-full py-2 rounded-lg text-xs font-medium transition-opacity"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            Apply Configuration
          </button>
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function TimeframesV4() {
  const [timeframes, setTimeframes] = useState<TimeframeConfig[]>(mockTimeframes);

  const enabledCount = timeframes.filter((t) => t.enabled).length;
  const primaryTf = timeframes.find((t) => t.primary);
  const totalBars = useMemo(
    () => timeframes.filter((t) => t.enabled).reduce((sum, t) => sum + t.barsPerSession, 0),
    [timeframes],
  );

  const toggleTimeframe = useCallback((tf: string) => {
    setTimeframes((prev) =>
      prev.map((t) => {
        if (t.tf === tf) {
          if (t.primary) return t;
          return { ...t, enabled: !t.enabled };
        }
        return t;
      }),
    );
  }, []);

  const setPrimary = useCallback((tf: string) => {
    setTimeframes((prev) =>
      prev.map((t) => ({
        ...t,
        primary: t.tf === tf,
        enabled: t.tf === tf ? true : t.enabled,
      })),
    );
  }, []);

  const applyRecommendation = useCallback((primary: string, confluenceTfs: string[]) => {
    setTimeframes((prev) =>
      prev.map((t) => ({
        ...t,
        primary: t.tf === primary,
        enabled: t.tf === primary || confluenceTfs.includes(t.tf),
      })),
    );
  }, []);

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      <PageHeader
        title="Timeframe Explorer"
        subtitle={`${enabledCount} of ${timeframes.length} enabled \u00b7 Primary: ${primaryTf?.tf ?? '---'} \u00b7 ${totalBars} total bars/session`}
      />

      {/* Metrics strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard label="Primary TF" value={primaryTf?.tf ?? '---'} delta={`${primaryTf?.barsPerSession ?? 0} bars/session`} positive />
        <MetricCard label="Confluence TFs" value={String(enabledCount - 1)} delta="active cross-TF feeds" positive />
        <MetricCard label="Data Density" value={`${totalBars}`} delta="bars processed/session" positive />
        <MetricCard
          label="Avg Volatility"
          value={`${(timeframes.filter((t) => t.enabled).reduce((s, t) => s + t.avgVolatility, 0) / enabledCount).toFixed(2)}%`}
          delta="per bar (enabled avg)"
          positive
        />
      </div>

      {/* Main layout: 2 columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
        <HierarchyTree timeframes={timeframes} onToggle={toggleTimeframe} onSetPrimary={setPrimary} />
        <SessionTimeline timeframes={timeframes} />
      </div>

      {/* Data flow diagram */}
      <div className="mb-5">
        <DataFlowDiagram timeframes={timeframes} />
      </div>

      {/* Recommendation engine */}
      <RecommendationEngine onApply={applyRecommendation} />
    </div>
  );
}
