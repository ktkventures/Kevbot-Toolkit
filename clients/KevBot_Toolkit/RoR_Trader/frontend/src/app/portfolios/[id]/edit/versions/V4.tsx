'use client';

import { useState, useMemo, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

const STRATEGY_COLORS = ['var(--green)', 'var(--blue)', 'var(--orange)', 'var(--red)', '#AB47BC', '#26C6DA'];

interface StrategyData {
  id: string;
  name: string;
  symbol: string;
  direction: string;
  displayName: string;
  trades: number;
  winRate: number;
  pf: number;
  totalR: number;
  maxDD: number;
  avgR: number;
  equityCurve: number[];
}

const ALL_STRATEGIES: StrategyData[] = [
  {
    id: 's1', name: 'Mass #2', symbol: 'NVDA', direction: 'LONG',
    displayName: 'NVDA LONG - Mass #2',
    trades: 224, winRate: 54.0, pf: 2.05, totalR: 94.1, maxDD: -4.5, avgR: 0.42,
    equityCurve: [0, 5, 8, 6, 15, 20, 28, 25, 35, 40, 45, 52, 58, 65, 70, 78, 85, 92],
  },
  {
    id: 's2', name: 'Mass #1', symbol: 'SPY', direction: 'LONG',
    displayName: 'SPY LONG - Mass #1',
    trades: 186, winRate: 52.1, pf: 1.78, totalR: 61.2, maxDD: -3.8, avgR: 0.33,
    equityCurve: [0, 3, 5, 4, 10, 15, 18, 20, 25, 30, 32, 38, 42, 48, 52, 56, 62, 68],
  },
  {
    id: 's3', name: 'Mass #5', symbol: 'AAPL', direction: 'LONG',
    displayName: 'AAPL LONG - Mass #5',
    trades: 142, winRate: 56.3, pf: 1.92, totalR: 48.5, maxDD: -2.9, avgR: 0.34,
    equityCurve: [0, 2, 4, 6, 8, 12, 15, 18, 22, 28, 30, 35, 38, 42, 48, 52, 58, 62],
  },
  {
    id: 's4', name: 'Mass #5', symbol: 'TSLA', direction: 'LONG',
    displayName: 'TSLA LONG - Mass #5',
    trades: 198, winRate: 48.5, pf: 1.65, totalR: 38.2, maxDD: -6.1, avgR: 0.19,
    equityCurve: [0, 4, 2, 8, 5, 12, 18, 14, 22, 25, 20, 28, 32, 35, 30, 38, 42, 48],
  },
  {
    id: 's5', name: 'Mass #13', symbol: 'META', direction: 'LONG',
    displayName: 'META LONG - Mass #13',
    trades: 168, winRate: 55.4, pf: 1.88, totalR: 52.8, maxDD: -3.4, avgR: 0.31,
    equityCurve: [0, 3, 7, 10, 14, 18, 22, 25, 30, 34, 38, 42, 45, 50, 55, 60, 65, 70],
  },
  {
    id: 's6', name: 'Mass #7', symbol: 'AMZN', direction: 'LONG',
    displayName: 'AMZN LONG - Mass #7',
    trades: 155, winRate: 53.5, pf: 1.82, totalR: 44.6, maxDD: -3.1, avgR: 0.29,
    equityCurve: [0, 2, 5, 8, 12, 16, 20, 23, 26, 30, 34, 38, 40, 44, 48, 52, 56, 60],
  },
];

const REQUIREMENT_SETS = ['None', 'TTP 50k', 'FTMO 100k', 'Apex 150k'];

interface PortfolioStrategy {
  id: string;
  name: string;
  symbol: string;
  direction: string;
  displayName: string;
  riskPerTrade: number;
  trades: number;
  winRate: number;
  pf: number;
  totalR: number;
  maxDD: number;
  avgR: number;
  equityCurve: number[];
}

// Pre-filled mock data for editing
const INITIAL_PORTFOLIO = {
  name: 'Momentum Alpha',
  startingBalance: 80000,
  riskScaling: 85,
  riskTolerance: 'aggressive' as const,
  requirementSet: 'TTP 50k',
  strategies: [
    { ...ALL_STRATEGIES[0], riskPerTrade: 120 },
    { ...ALL_STRATEGIES[1], riskPerTrade: 200 },
    { ...ALL_STRATEGIES[2], riskPerTrade: 90 },
    { ...ALL_STRATEGIES[3], riskPerTrade: 80 },
  ] as PortfolioStrategy[],
};

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function buildEquityPath(curve: number[], maxVal: number, svgWidth: number, svgHeight: number, padTop: number, padBot: number): string {
  if (curve.length === 0) return '';
  const usableHeight = svgHeight - padTop - padBot;
  const stepX = svgWidth / (curve.length - 1);
  return curve.map((val, i) => {
    const x = i * stepX;
    const y = padTop + usableHeight - (val / maxVal) * usableHeight;
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function buildCombinedCurve(strategies: PortfolioStrategy[], riskScaling: number): number[] {
  if (strategies.length === 0) return [];
  const maxLen = Math.max(...strategies.map(s => s.equityCurve.length));
  const combined: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    let total = 0;
    for (const s of strategies) {
      const idx = Math.min(i, s.equityCurve.length - 1);
      const scaledVal = s.equityCurve[idx] * (s.riskPerTrade / 100);
      total += scaledVal * (1 + riskScaling / 100 * 0.5);
    }
    combined.push(total);
  }
  return combined;
}

function getRiskLevel(strategies: PortfolioStrategy[], riskScaling: number): { label: string; color: string; pct: number } {
  if (strategies.length === 0) return { label: 'N/A', color: 'var(--text-muted)', pct: 0 };
  const avgDD = strategies.reduce((s, st) => s + Math.abs(st.maxDD), 0) / strategies.length;
  const scaledDD = avgDD * (1 + riskScaling / 100 * 0.5);
  if (scaledDD < 3) return { label: 'Conservative', color: 'var(--green)', pct: scaledDD / 10 * 100 };
  if (scaledDD < 5) return { label: 'Moderate', color: 'var(--yellow, #e5a813)', pct: scaledDD / 10 * 100 };
  return { label: 'Aggressive', color: 'var(--red)', pct: Math.min(scaledDD / 10 * 100, 100) };
}

/* ========================================================================= */
/* STYLES                                                                     */
/* ========================================================================= */

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  borderRadius: '0.5rem',
  padding: '0.5rem 0.75rem',
  fontSize: '0.875rem',
  width: '100%',
  outline: 'none',
};

const selectStyle: React.CSSProperties = {
  ...inputStyle,
  cursor: 'pointer',
  appearance: 'none' as const,
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M3 5l3 3 3-3'/%3E%3C/svg%3E")`,
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.75rem center',
  paddingRight: '2rem',
};

/* ========================================================================= */
/* WIZARD STEPS                                                               */
/* ========================================================================= */

const STEPS = [
  { id: 'setup', label: 'Setup', icon: '1' },
  { id: 'strategies', label: 'Strategies', icon: '2' },
  { id: 'risk', label: 'Risk', icon: '3' },
  { id: 'review', label: 'Review', icon: '4' },
] as const;

type StepId = typeof STEPS[number]['id'];

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function PortfolioEditV4() {
  const router = useRouter();
  const initialRef = useRef(INITIAL_PORTFOLIO);

  // --- Wizard state ---
  const [currentStep, setCurrentStep] = useState<StepId>('setup');

  // --- Settings state (pre-filled) ---
  const [portfolioName, setPortfolioName] = useState(INITIAL_PORTFOLIO.name);
  const [startingBalance, setStartingBalance] = useState(INITIAL_PORTFOLIO.startingBalance);
  const [riskTolerance, setRiskTolerance] = useState<'conservative' | 'moderate' | 'aggressive'>(INITIAL_PORTFOLIO.riskTolerance);

  // --- Strategy builder state ---
  const [strategies, setStrategies] = useState<PortfolioStrategy[]>([...INITIAL_PORTFOLIO.strategies]);
  const [hoveredStrategy, setHoveredStrategy] = useState<string | null>(null);

  // --- Risk state ---
  const [riskScaling, setRiskScaling] = useState(INITIAL_PORTFOLIO.riskScaling);
  const [requirementSet, setRequirementSet] = useState(INITIAL_PORTFOLIO.requirementSet);

  // --- Change tracking ---
  const hasChanges = useMemo(() => {
    const init = initialRef.current;
    if (portfolioName !== init.name) return true;
    if (startingBalance !== init.startingBalance) return true;
    if (riskScaling !== init.riskScaling) return true;
    if (requirementSet !== init.requirementSet) return true;
    if (riskTolerance !== init.riskTolerance) return true;
    if (strategies.length !== init.strategies.length) return true;
    for (let i = 0; i < strategies.length; i++) {
      if (strategies[i].id !== init.strategies[i]?.id) return true;
      if (strategies[i].riskPerTrade !== init.strategies[i]?.riskPerTrade) return true;
    }
    return false;
  }, [portfolioName, startingBalance, riskScaling, requirementSet, riskTolerance, strategies]);

  // --- Changed fields for indicators ---
  const changedFields = useMemo(() => {
    const init = initialRef.current;
    const fields: string[] = [];
    if (portfolioName !== init.name) fields.push('name');
    if (startingBalance !== init.startingBalance) fields.push('balance');
    if (riskScaling !== init.riskScaling) fields.push('risk_scaling');
    if (requirementSet !== init.requirementSet) fields.push('requirement_set');
    if (riskTolerance !== init.riskTolerance) fields.push('risk_tolerance');
    if (strategies.length !== init.strategies.length) fields.push('strategies');
    else {
      for (let i = 0; i < strategies.length; i++) {
        if (strategies[i].id !== init.strategies[i]?.id || strategies[i].riskPerTrade !== init.strategies[i]?.riskPerTrade) {
          fields.push('strategies');
          break;
        }
      }
    }
    return fields;
  }, [portfolioName, startingBalance, riskScaling, requirementSet, riskTolerance, strategies]);

  // --- Computed values ---
  const combinedCurve = useMemo(
    () => buildCombinedCurve(strategies, riskScaling),
    [strategies, riskScaling]
  );

  const riskLevel = useMemo(
    () => getRiskLevel(strategies, riskScaling),
    [strategies, riskScaling]
  );

  const metrics = useMemo(() => {
    if (strategies.length === 0) {
      return { trades: 0, winRate: 0, pf: 0, totalPnl: 0, finalBalance: 0, maxDD: 0 };
    }
    const totalTrades = strategies.reduce((sum, s) => sum + s.trades, 0);
    const weightedWR = strategies.reduce((sum, s) => sum + s.winRate * s.trades, 0) / totalTrades;
    const weightedPF = strategies.reduce((sum, s) => sum + s.pf * s.riskPerTrade, 0)
      / strategies.reduce((sum, s) => sum + s.riskPerTrade, 0);
    const totalPnl = strategies.reduce((sum, s) => sum + s.totalR * s.riskPerTrade, 0);
    const scaledPnl = totalPnl * (1 + riskScaling / 100 * 0.5);
    const finalBalance = startingBalance + scaledPnl;
    const maxDD = Math.min(...strategies.map(s => s.maxDD));
    return { trades: totalTrades, winRate: weightedWR, pf: weightedPF, totalPnl: scaledPnl, finalBalance, maxDD };
  }, [strategies, startingBalance, riskScaling]);

  // Donut chart data
  const donutSegments = useMemo(() => {
    if (strategies.length === 0) return [];
    const totalRisk = strategies.reduce((s, st) => s + st.riskPerTrade, 0);
    let cumAngle = 0;
    return strategies.map((s, i) => {
      const pct = s.riskPerTrade / totalRisk;
      const startAngle = cumAngle;
      cumAngle += pct * 360;
      return {
        id: s.id,
        label: s.symbol,
        pct,
        startAngle,
        endAngle: cumAngle,
        color: STRATEGY_COLORS[ALL_STRATEGIES.findIndex(a => a.id === s.id) % STRATEGY_COLORS.length],
      };
    });
  }, [strategies]);

  // Smart defaults based on risk tolerance
  const applySmartDefaults = useCallback((tolerance: 'conservative' | 'moderate' | 'aggressive') => {
    setRiskTolerance(tolerance);
    if (tolerance === 'conservative') {
      setRiskScaling(25);
    } else if (tolerance === 'moderate') {
      setRiskScaling(50);
    } else {
      setRiskScaling(85);
    }
  }, []);

  // --- Handlers ---
  const handleToggleStrategy = useCallback((stratId: string) => {
    setStrategies(prev => {
      const exists = prev.find(s => s.id === stratId);
      if (exists) return prev.filter(s => s.id !== stratId);
      const data = ALL_STRATEGIES.find(s => s.id === stratId);
      if (!data) return prev;
      const defaultRisk = riskTolerance === 'conservative' ? 50 : riskTolerance === 'moderate' ? 100 : 200;
      return [...prev, { ...data, riskPerTrade: defaultRisk }];
    });
  }, [riskTolerance]);

  const handleStrategyRiskChange = useCallback((id: string, newRisk: string) => {
    setStrategies(prev => prev.map(s =>
      s.id === id ? { ...s, riskPerTrade: Number(newRisk) || 0 } : s
    ));
  }, []);

  const handleResetChanges = useCallback(() => {
    const init = initialRef.current;
    setPortfolioName(init.name);
    setStartingBalance(init.startingBalance);
    setRiskScaling(init.riskScaling);
    setRequirementSet(init.requirementSet);
    setRiskTolerance(init.riskTolerance);
    setStrategies([...init.strategies]);
  }, []);

  // --- Navigation helpers ---
  const stepIndex = STEPS.findIndex(s => s.id === currentStep);
  const canGoNext = () => {
    if (currentStep === 'setup') return portfolioName.trim().length > 0;
    if (currentStep === 'strategies') return strategies.length > 0;
    if (currentStep === 'risk') return true;
    return true;
  };
  const goNext = () => {
    if (stepIndex < STEPS.length - 1) setCurrentStep(STEPS[stepIndex + 1].id);
  };
  const goBack = () => {
    if (stepIndex > 0) setCurrentStep(STEPS[stepIndex - 1].id);
  };

  // --- SVG helpers ---
  const svgW = 500;
  const svgH = 200;
  const padTop = 20;
  const padBot = 10;
  const maxEquityVal = Math.max(1, ...combinedCurve);
  const combinedPath = buildEquityPath(combinedCurve, maxEquityVal, svgW, svgH, padTop, padBot);
  const combinedFillPath = combinedPath ? `${combinedPath} L${svgW},${svgH} L0,${svgH} Z` : '';

  // Donut arc path helper
  function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
    const start = ((startAngle - 90) * Math.PI) / 180;
    const end = ((endAngle - 90) * Math.PI) / 180;
    const x1 = cx + r * Math.cos(start);
    const y1 = cy + r * Math.sin(start);
    const x2 = cx + r * Math.cos(end);
    const y2 = cy + r * Math.sin(end);
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`;
  }

  // Mini sparkline for strategy cards
  function renderSparkline(curve: number[], color: string, w: number, h: number) {
    if (curve.length < 2) return null;
    const max = Math.max(...curve, 1);
    const stepX = w / (curve.length - 1);
    const path = curve.map((v, i) => {
      const x = i * stepX;
      const y = h - (v / max) * (h - 4) - 2;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const fillPath = `${path} L${w},${h} L0,${h} Z`;
    return (
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ display: 'block' }}>
        <defs>
          <linearGradient id={`sparkE-${color.replace(/[^a-zA-Z0-9]/g, '')}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={fillPath} fill={`url(#sparkE-${color.replace(/[^a-zA-Z0-9]/g, '')})`} />
        <path d={path} fill="none" stroke={color} strokeWidth="1.5" />
      </svg>
    );
  }

  // Risk gauge SVG
  function renderRiskGauge() {
    const cx = 80;
    const cy = 80;
    const r = 60;
    const startAngle = 135;
    const endAngle = 405;
    const totalAngle = endAngle - startAngle;
    const filledAngle = startAngle + (riskLevel.pct / 100) * totalAngle;

    function polarToCartesian(cxP: number, cyP: number, rP: number, angleDeg: number) {
      const rad = ((angleDeg - 90) * Math.PI) / 180;
      return { x: cxP + rP * Math.cos(rad), y: cyP + rP * Math.sin(rad) };
    }

    function describeGaugeArc(cxP: number, cyP: number, rP: number, startA: number, endA: number) {
      const s = polarToCartesian(cxP, cyP, rP, startA);
      const e = polarToCartesian(cxP, cyP, rP, endA);
      const largeArc = endA - startA > 180 ? 1 : 0;
      return `M ${s.x} ${s.y} A ${rP} ${rP} 0 ${largeArc} 1 ${e.x} ${e.y}`;
    }

    return (
      <svg width="160" height="120" viewBox="0 0 160 120">
        <path
          d={describeGaugeArc(cx, cy, r, startAngle, endAngle)}
          fill="none"
          stroke="var(--border)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {riskLevel.pct > 0 && (
          <path
            d={describeGaugeArc(cx, cy, r, startAngle, filledAngle)}
            fill="none"
            stroke={riskLevel.color}
            strokeWidth="12"
            strokeLinecap="round"
          />
        )}
        <text x={cx} y={cy - 4} textAnchor="middle" fill={riskLevel.color} fontSize="16" fontWeight="700">
          {riskLevel.label}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill="var(--text-muted)" fontSize="10">
          Portfolio Risk
        </text>
      </svg>
    );
  }

  // Helper: is a field changed?
  function fieldChanged(field: string): React.CSSProperties {
    if (changedFields.includes(field)) {
      return { boxShadow: '0 0 0 2px var(--orange)', borderColor: 'var(--orange)' };
    }
    return {};
  }

  return (
    <div>
      <PageHeader
        title="Edit Portfolio"
        subtitle={`Modifying: ${initialRef.current.name}`}
        backHref="/portfolios/1"
        actions={
          hasChanges ? (
            <span
              className="text-xs px-2.5 py-1 rounded-full"
              style={{ background: 'var(--orange)', color: 'white', opacity: 0.9 }}
            >
              {changedFields.length} unsaved change{changedFields.length !== 1 ? 's' : ''}
            </span>
          ) : undefined
        }
      />

      {/* ============ Wizard Progress Bar ============ */}
      <Card className="mb-6">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'relative' }}>
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '40px',
              right: '40px',
              height: '2px',
              background: 'var(--border)',
              transform: 'translateY(-50%)',
              zIndex: 0,
            }}
          />
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '40px',
              width: `${(stepIndex / (STEPS.length - 1)) * (100 - 10)}%`,
              height: '2px',
              background: 'var(--accent)',
              transform: 'translateY(-50%)',
              zIndex: 1,
              transition: 'width 0.4s ease',
            }}
          />

          {STEPS.map((step, i) => {
            const isActive = i === stepIndex;
            const isCompleted = i < stepIndex;
            // Show change dot on step if relevant fields changed
            const stepHasChange =
              (step.id === 'setup' && (changedFields.includes('name') || changedFields.includes('balance') || changedFields.includes('risk_tolerance'))) ||
              (step.id === 'strategies' && changedFields.includes('strategies')) ||
              (step.id === 'risk' && (changedFields.includes('risk_scaling') || changedFields.includes('requirement_set')));

            return (
              <button
                key={step.id}
                onClick={() => {
                  if (i <= stepIndex || (i === stepIndex + 1 && canGoNext())) {
                    setCurrentStep(step.id);
                  }
                }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '0.5rem',
                  cursor: i <= stepIndex + 1 ? 'pointer' : 'default',
                  background: 'transparent',
                  border: 'none',
                  zIndex: 2,
                  opacity: i > stepIndex + 1 ? 0.4 : 1,
                  transition: 'opacity 0.3s ease',
                  position: 'relative',
                }}
              >
                {/* Change indicator dot */}
                {stepHasChange && (
                  <div style={{
                    position: 'absolute',
                    top: -2,
                    right: 'calc(50% - 24px)',
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: 'var(--orange)',
                    border: '2px solid var(--bg-card)',
                    zIndex: 3,
                  }} />
                )}
                <div
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.875rem',
                    fontWeight: 600,
                    transition: 'all 0.3s ease',
                    background: isCompleted ? 'var(--accent)' : isActive ? 'var(--accent)' : 'var(--bg-card)',
                    color: isCompleted || isActive ? 'white' : 'var(--text-muted)',
                    border: isCompleted || isActive ? '2px solid var(--accent)' : '2px solid var(--border)',
                    boxShadow: isActive ? '0 0 0 4px var(--accent-muted)' : 'none',
                  }}
                >
                  {isCompleted ? (
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                      <path d="M3.5 8L6.5 11L12.5 5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : (
                    step.icon
                  )}
                </div>
                <span
                  className="text-xs font-medium"
                  style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}
                >
                  {step.label}
                </span>
              </button>
            );
          })}
        </div>
      </Card>

      {/* ============ STEP 1: Setup ============ */}
      {currentStep === 'setup' && (
        <div style={{ animation: 'fadeInEdit 0.3s ease' }}>
          <Card className="mb-6">
            <h3 className="text-lg font-semibold mb-1">Portfolio Setup</h3>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
              Modify your portfolio name and account parameters.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', maxWidth: '600px' }}>
              <div style={{ gridColumn: '1 / -1' }}>
                <label className="text-xs block mb-1.5" style={{ color: changedFields.includes('name') ? 'var(--orange)' : 'var(--text-muted)' }}>
                  Portfolio Name {changedFields.includes('name') && '(changed)'}
                </label>
                <input
                  type="text"
                  placeholder="e.g. Momentum Alpha"
                  value={portfolioName}
                  onChange={(e) => setPortfolioName(e.target.value)}
                  style={{ ...inputStyle, fontSize: '1rem', padding: '0.75rem 1rem', ...fieldChanged('name') }}
                  autoFocus
                />
              </div>
              <div>
                <label className="text-xs block mb-1.5" style={{ color: changedFields.includes('balance') ? 'var(--orange)' : 'var(--text-muted)' }}>
                  Starting Balance ($) {changedFields.includes('balance') && '(changed)'}
                </label>
                <input
                  type="number"
                  min={1000}
                  step={1000}
                  value={startingBalance}
                  onChange={(e) => setStartingBalance(Number(e.target.value) || 1000)}
                  style={{ ...inputStyle, ...fieldChanged('balance') }}
                />
              </div>
              <div>
                <label className="text-xs block mb-2" style={{ color: changedFields.includes('risk_tolerance') ? 'var(--orange)' : 'var(--text-muted)' }}>
                  Risk Tolerance {changedFields.includes('risk_tolerance') && '(changed)'}
                </label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {(['conservative', 'moderate', 'aggressive'] as const).map(t => (
                    <button
                      key={t}
                      onClick={() => applySmartDefaults(t)}
                      className="flex-1 py-2 rounded-lg text-xs font-medium transition-all"
                      style={{
                        background: riskTolerance === t
                          ? t === 'conservative' ? 'var(--green)' : t === 'moderate' ? 'var(--accent)' : 'var(--red)'
                          : 'var(--bg-input)',
                        color: riskTolerance === t ? 'white' : 'var(--text-secondary)',
                        border: riskTolerance === t ? 'none' : '1px solid var(--border)',
                        textTransform: 'capitalize',
                      }}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ============ STEP 2: Strategy Picker ============ */}
      {currentStep === 'strategies' && (
        <div style={{ animation: 'fadeInEdit 0.3s ease' }}>
          <div className="grid grid-cols-5 gap-6">
            <div className="col-span-3">
              <Card className="mb-0">
                <h3 className="text-lg font-semibold mb-1">Select Strategies</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                  Click strategies to add or remove them. Pre-selected strategies from the original portfolio are highlighted.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem' }}>
                  {ALL_STRATEGIES.map((s, idx) => {
                    const isSelected = strategies.some(st => st.id === s.id);
                    const wasOriginal = initialRef.current.strategies.some(st => st.id === s.id);
                    const isHovered = hoveredStrategy === s.id;
                    const color = STRATEGY_COLORS[idx % STRATEGY_COLORS.length];
                    const isNewlyAdded = isSelected && !wasOriginal;
                    const isRemoved = !isSelected && wasOriginal;

                    return (
                      <div
                        key={s.id}
                        onClick={() => handleToggleStrategy(s.id)}
                        onMouseEnter={() => setHoveredStrategy(s.id)}
                        onMouseLeave={() => setHoveredStrategy(null)}
                        className="rounded-xl p-4 cursor-pointer transition-all"
                        style={{
                          background: isSelected ? 'var(--accent-muted)' : 'var(--bg-input)',
                          border: isNewlyAdded
                            ? '2px solid var(--green)'
                            : isRemoved
                              ? '2px dashed var(--red)'
                              : isSelected
                                ? '2px solid var(--accent)'
                                : '2px solid var(--border)',
                          transform: isHovered ? 'translateY(-2px)' : 'translateY(0)',
                          boxShadow: isHovered ? '0 4px 12px rgba(0,0,0,0.2)' : 'none',
                          position: 'relative',
                          overflow: 'hidden',
                          opacity: isRemoved ? 0.5 : 1,
                        }}
                      >
                        {/* Selection check / change badge */}
                        {isSelected && (
                          <div
                            style={{
                              position: 'absolute',
                              top: 8,
                              right: 8,
                              width: 22,
                              height: 22,
                              borderRadius: '50%',
                              background: isNewlyAdded ? 'var(--green)' : 'var(--accent)',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                              <path d="M3.5 8L6.5 11L12.5 5" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                          </div>
                        )}
                        {isNewlyAdded && (
                          <span
                            className="text-[9px] px-1.5 py-0.5 rounded font-semibold"
                            style={{ position: 'absolute', top: 8, left: 8, background: 'var(--green)', color: 'white' }}
                          >
                            NEW
                          </span>
                        )}

                        <div className="flex items-center gap-2 mb-2">
                          <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                          <span className="text-sm font-semibold truncate">{s.displayName}</span>
                        </div>

                        <div className="mb-3" style={{ height: 40 }}>
                          {renderSparkline(s.equityCurve, color, 200, 40)}
                        </div>

                        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                          <div>
                            <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>Win Rate</span>
                            <span className="text-xs font-semibold">{s.winRate}%</span>
                          </div>
                          <div>
                            <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>PF</span>
                            <span className="text-xs font-semibold">{s.pf}</span>
                          </div>
                          <div>
                            <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>Total R</span>
                            <span className="text-xs font-semibold" style={{ color: s.totalR >= 0 ? 'var(--green)' : 'var(--red)' }}>
                              {s.totalR >= 0 ? '+' : ''}{s.totalR.toFixed(1)}
                            </span>
                          </div>
                          <div>
                            <span className="text-[10px] block" style={{ color: 'var(--text-muted)' }}>Trades</span>
                            <span className="text-xs font-semibold">{s.trades}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </div>

            {/* Right panel: Selected + Donut */}
            <div className="col-span-2">
              <Card className="mb-4">
                <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                  Portfolio Composition
                  <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)' }}>({strategies.length} selected)</span>
                </h3>

                {strategies.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Click strategies on the left to build your portfolio</p>
                  </div>
                ) : (
                  <>
                    {/* Donut chart */}
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                      <svg width="140" height="140" viewBox="0 0 160 160">
                        {donutSegments.length === 1 ? (
                          <circle cx="80" cy="80" r="55" fill={donutSegments[0].color} opacity="0.85" />
                        ) : (
                          donutSegments.map((seg) => (
                            <path
                              key={seg.id}
                              d={describeArc(80, 80, 55, seg.startAngle, seg.endAngle - 0.5)}
                              fill={seg.color}
                              opacity="0.85"
                            />
                          ))
                        )}
                        <circle cx="80" cy="80" r="32" fill="var(--bg-card)" />
                        <text x="80" y="78" textAnchor="middle" fill="var(--text-primary)" fontSize="18" fontWeight="700">
                          {strategies.length}
                        </text>
                        <text x="80" y="93" textAnchor="middle" fill="var(--text-muted)" fontSize="9">
                          {strategies.length === 1 ? 'strategy' : 'strategies'}
                        </text>
                      </svg>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      {strategies.map((s) => {
                        const wasOriginal = initialRef.current.strategies.some(st => st.id === s.id);
                        const origRisk = initialRef.current.strategies.find(st => st.id === s.id)?.riskPerTrade;
                        const riskChanged = origRisk !== undefined && origRisk !== s.riskPerTrade;

                        return (
                          <div
                            key={s.id}
                            className="flex items-center gap-2 rounded-lg px-3 py-2"
                            style={{
                              background: 'var(--bg-input)',
                              border: !wasOriginal ? '1px solid var(--green)' : riskChanged ? '1px solid var(--orange)' : '1px solid var(--border)',
                            }}
                          >
                            <div style={{ width: 8, height: 8, borderRadius: '50%', background: STRATEGY_COLORS[ALL_STRATEGIES.findIndex(a => a.id === s.id) % STRATEGY_COLORS.length], flexShrink: 0 }} />
                            <span className="text-xs font-medium flex-1 truncate">{s.symbol}</span>
                            {!wasOriginal && (
                              <span className="text-[9px] px-1 rounded" style={{ background: 'var(--green)', color: 'white' }}>NEW</span>
                            )}
                            <div className="flex items-center gap-1" style={{ flexShrink: 0 }}>
                              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>$</span>
                              <input
                                type="number"
                                min={1}
                                step={10}
                                value={s.riskPerTrade}
                                onChange={(e) => handleStrategyRiskChange(s.id, e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                                style={{
                                  ...inputStyle,
                                  width: '60px',
                                  padding: '0.125rem 0.375rem',
                                  textAlign: 'right' as const,
                                  fontSize: '0.75rem',
                                  ...(riskChanged ? { borderColor: 'var(--orange)' } : {}),
                                }}
                              />
                              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>/t</span>
                            </div>
                            <button
                              onClick={(e) => { e.stopPropagation(); handleToggleStrategy(s.id); }}
                              className="w-5 h-5 rounded flex items-center justify-center text-[10px]"
                              style={{ color: 'var(--red)', background: 'transparent', flexShrink: 0 }}
                              onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--red-muted)')}
                              onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                            >
                              x
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* ============ STEP 3: Risk Configuration ============ */}
      {currentStep === 'risk' && (
        <div style={{ animation: 'fadeInEdit 0.3s ease' }}>
          <div className="grid grid-cols-2 gap-6 mb-6">
            <Card>
              <h3 className="text-lg font-semibold mb-1">Risk Level</h3>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                Adjust risk scaling to control your overall exposure.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                {renderRiskGauge()}

                <div style={{ width: '100%', maxWidth: 300 }}>
                  <label className="text-xs block mb-1.5 text-center" style={{ color: changedFields.includes('risk_scaling') ? 'var(--orange)' : 'var(--text-muted)' }}>
                    Risk Scaling: {riskScaling}% {changedFields.includes('risk_scaling') && `(was ${initialRef.current.riskScaling}%)`}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={riskScaling}
                    onChange={(e) => setRiskScaling(Number(e.target.value))}
                    style={{ width: '100%', accentColor: riskLevel.color }}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem' }}>
                    <span className="text-[10px]" style={{ color: 'var(--green)' }}>Conservative</span>
                    <span className="text-[10px]" style={{ color: 'var(--red)' }}>Aggressive</span>
                  </div>
                </div>
              </div>
            </Card>

            <Card>
              <h3 className="text-lg font-semibold mb-1">Risk Rules</h3>
              <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                Apply prop firm rules or custom requirement sets.
              </p>

              <div className="mb-6">
                <label className="text-xs block mb-1.5" style={{ color: changedFields.includes('requirement_set') ? 'var(--orange)' : 'var(--text-muted)' }}>
                  Requirement Set {changedFields.includes('requirement_set') && `(was: ${initialRef.current.requirementSet})`}
                </label>
                <select
                  value={requirementSet}
                  onChange={(e) => setRequirementSet(e.target.value)}
                  style={{ ...selectStyle, ...fieldChanged('requirement_set') }}
                >
                  {REQUIREMENT_SETS.map(r => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
              </div>

              <div>
                <h4 className="text-xs font-semibold mb-3" style={{ color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Stress Scenarios
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {[
                    { label: 'Market drops 5%', impact: metrics.totalPnl * -0.3, ddImpact: metrics.maxDD * 1.5 },
                    { label: '3-loss streak', impact: -strategies.reduce((s, st) => s + st.riskPerTrade * 3, 0), ddImpact: metrics.maxDD * 1.2 },
                    { label: 'Best case (all win)', impact: metrics.totalPnl * 1.5, ddImpact: metrics.maxDD * 0.5 },
                  ].map((scenario) => (
                    <div
                      key={scenario.label}
                      className="rounded-lg p-3"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium">{scenario.label}</span>
                        <span
                          className="text-xs font-semibold"
                          style={{ color: scenario.impact >= 0 ? 'var(--green)' : 'var(--red)' }}
                        >
                          {scenario.impact >= 0 ? '+' : ''}${Math.abs(scenario.impact).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </span>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Projected Drawdown</span>
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                          {scenario.ddImpact.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </div>

          {strategies.length > 0 && (
            <div className="grid grid-cols-6 gap-3">
              <MetricCard label="Trades" value={metrics.trades.toLocaleString()} />
              <MetricCard label="Win Rate" value={`${metrics.winRate.toFixed(1)}%`} />
              <MetricCard label="Profit Factor" value={metrics.pf.toFixed(2)} />
              <MetricCard
                label="Total P&L"
                value={`${metrics.totalPnl >= 0 ? '+' : ''}$${Math.abs(metrics.totalPnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                positive={metrics.totalPnl >= 0}
              />
              <MetricCard
                label="Final Balance"
                value={`$${metrics.finalBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              />
              <MetricCard
                label="Max Drawdown"
                value={`${metrics.maxDD.toFixed(1)}%`}
              />
            </div>
          )}
        </div>
      )}

      {/* ============ STEP 4: Review ============ */}
      {currentStep === 'review' && (
        <div style={{ animation: 'fadeInEdit 0.3s ease' }}>
          {/* Summary header with change summary */}
          <Card className="mb-6">
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div>
                <h3 className="text-lg font-semibold mb-1">
                  {portfolioName || 'Untitled Portfolio'}
                </h3>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  ${startingBalance.toLocaleString()} starting balance
                  {requirementSet !== 'None' ? ` | ${requirementSet}` : ''}
                  {' | '}Risk scaling: {riskScaling}%
                </p>
                {hasChanges && (
                  <div className="flex flex-wrap gap-1.5 mt-2">
                    {changedFields.map(f => (
                      <span
                        key={f}
                        className="text-[10px] px-2 py-0.5 rounded-full font-medium"
                        style={{ background: 'var(--orange)', color: 'white', opacity: 0.85 }}
                      >
                        {f.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <div
                className="px-3 py-1.5 rounded-full text-xs font-semibold"
                style={{ background: riskLevel.color, color: 'white', opacity: 0.9 }}
              >
                {riskLevel.label}
              </div>
            </div>
          </Card>

          <div className="grid grid-cols-6 gap-3 mb-6">
            <MetricCard label="Trades" value={metrics.trades.toLocaleString()} />
            <MetricCard label="Win Rate" value={`${metrics.winRate.toFixed(1)}%`} />
            <MetricCard label="Profit Factor" value={metrics.pf.toFixed(2)} />
            <MetricCard
              label="Total P&L"
              value={`${metrics.totalPnl >= 0 ? '+' : ''}$${Math.abs(metrics.totalPnl).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              positive={metrics.totalPnl >= 0}
            />
            <MetricCard
              label="Final Balance"
              value={`$${metrics.finalBalance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            />
            <MetricCard
              label="Max Drawdown"
              value={`${metrics.maxDD.toFixed(1)}%`}
            />
          </div>

          <div className="grid grid-cols-5 gap-6 mb-6">
            <div className="col-span-3">
              <Card>
                <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Combined Equity Curve</h3>
                <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', height: 220 }}>
                  <svg width="100%" height="100%" viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="none">
                    <defs>
                      <linearGradient id="eqGradEditV4" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
                        <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                    <line x1="0" y1="50" x2={svgW} y2="50" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                    <line x1="0" y1="100" x2={svgW} y2="100" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                    <line x1="0" y1="150" x2={svgW} y2="150" stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                    {strategies.map((s) => {
                      const path = buildEquityPath(
                        s.equityCurve.map(v => v * (s.riskPerTrade / 100)),
                        maxEquityVal,
                        svgW, svgH, padTop, padBot
                      );
                      return (
                        <path
                          key={s.id}
                          d={path}
                          fill="none"
                          stroke={STRATEGY_COLORS[ALL_STRATEGIES.findIndex(a => a.id === s.id) % STRATEGY_COLORS.length]}
                          strokeWidth="1"
                          strokeDasharray="4,3"
                          opacity="0.5"
                        />
                      );
                    })}
                    {combinedFillPath && (
                      <path d={combinedFillPath} fill="url(#eqGradEditV4)" stroke="none" />
                    )}
                    {combinedPath && (
                      <path d={combinedPath} fill="none" stroke="var(--accent)" strokeWidth="2" />
                    )}
                  </svg>
                </div>
              </Card>
            </div>

            <div className="col-span-2">
              <Card>
                <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                  Strategies ({strategies.length})
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
                  {strategies.map((s) => {
                    const sIdx = ALL_STRATEGIES.findIndex(a => a.id === s.id);
                    const wasOriginal = initialRef.current.strategies.some(st => st.id === s.id);
                    return (
                      <div
                        key={s.id}
                        className="flex items-center gap-2 rounded-lg px-3 py-2"
                        style={{ background: 'var(--bg-input)' }}
                      >
                        <div style={{ width: 6, height: 6, borderRadius: '50%', background: STRATEGY_COLORS[sIdx % STRATEGY_COLORS.length], flexShrink: 0 }} />
                        <span className="text-xs font-medium flex-1 truncate">{s.displayName}</span>
                        {!wasOriginal && (
                          <span className="text-[9px] px-1 rounded" style={{ background: 'var(--green)', color: 'white' }}>NEW</span>
                        )}
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>${s.riskPerTrade}/t</span>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </div>
          </div>
        </div>
      )}

      {/* ============ Bottom Navigation Bar ============ */}
      <div
        className="rounded-xl border p-4 mt-6"
        style={{
          background: 'var(--bg-card)',
          borderColor: 'var(--border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          {stepIndex > 0 && (
            <button
              onClick={goBack}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
              }}
            >
              Back
            </button>
          )}
          {hasChanges && (
            <button
              onClick={handleResetChanges}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--orange)',
                color: 'var(--orange)',
              }}
            >
              Reset Changes
            </button>
          )}
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            onClick={() => router.push('/portfolios/1')}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
            }}
          >
            Cancel
          </button>

          {currentStep === 'review' ? (
            <button
              className="px-6 py-2 rounded-lg text-sm font-medium transition-opacity"
              style={{
                background: 'var(--accent)',
                color: 'white',
                opacity: hasChanges && strategies.length > 0 && portfolioName.trim() ? 1 : 0.5,
                cursor: hasChanges && strategies.length > 0 && portfolioName.trim() ? 'pointer' : 'not-allowed',
              }}
              disabled={!hasChanges || strategies.length === 0 || !portfolioName.trim()}
            >
              Update Portfolio
            </button>
          ) : (
            <button
              onClick={goNext}
              className="px-6 py-2 rounded-lg text-sm font-medium transition-opacity"
              style={{
                background: 'var(--accent)',
                color: 'white',
                opacity: canGoNext() ? 1 : 0.5,
                cursor: canGoNext() ? 'pointer' : 'not-allowed',
              }}
              disabled={!canGoNext()}
            >
              Next Step
            </button>
          )}
        </div>
      </div>

      <style>{`
        @keyframes fadeInEdit {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
