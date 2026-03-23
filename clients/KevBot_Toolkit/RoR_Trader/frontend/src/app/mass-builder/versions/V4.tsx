'use client';

import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import Modal from '@/components/Modal';

// =============================================================================
// V4 Meta: Creative — Strategy Discovery Lab
// =============================================================================
// Immersive strategy discovery experience with:
//
// 1. DISCOVERY LAB INTERFACE — visual search with parameter cards
// 2. SCATTER PLOT RESULTS — interactive plot (x=WR, y=PF, size=trades)
// 3. ANIMATED TICKER PROGRESS — "Evaluating NVDA LONG 1Min..." ticker
// 4. HEATMAP GRID — which ticker/TF combos produce the best strategies
// 5. INSIGHT CARDS — AI-style summary cards on results
// 6. VISUAL CONFIG — icon-based parameter cards instead of form inputs
// 7. COMBINATION COUNTER — live count of total combos to evaluate
// 8. TOP PERFORMERS SPOTLIGHT — hero cards for best discoveries
// =============================================================================

// ---------------------------------------------------------------------------
// CSS Animations
// ---------------------------------------------------------------------------

const ANIMATION_STYLES = `
@keyframes fade-in-up {
  0% { opacity: 0; transform: translateY(12px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 8px rgba(0,255,136,0.1); }
  50% { box-shadow: 0 0 24px rgba(0,255,136,0.35); }
}
@keyframes slide-in-left {
  0% { transform: translateX(-20px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes slide-in-right {
  0% { transform: translateX(20px); opacity: 0; }
  100% { transform: translateX(0); opacity: 1; }
}
@keyframes scan-bar {
  0% { left: 0%; width: 0%; }
  50% { width: 40%; }
  100% { left: 100%; width: 0%; }
}
@keyframes ticker-scroll {
  0% { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
@keyframes dot-bounce {
  0%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(-6px); }
}
@keyframes pop-in {
  0% { transform: scale(0.5); opacity: 0; }
  70% { transform: scale(1.05); }
  100% { transform: scale(1); opacity: 1; }
}
@keyframes progress-fill {
  0% { width: 0%; }
}
@keyframes heatmap-reveal {
  0% { opacity: 0; transform: scale(0.8); }
  100% { opacity: 1; transform: scale(1); }
}
@keyframes sparkle {
  0%, 100% { opacity: 0; }
  50% { opacity: 1; }
}
`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface MassResult {
  rank: number;
  ticker: string;
  direction: 'LONG' | 'SHORT';
  tf: string;
  trigger: string;
  winRate: number;
  pf: number;
  dailyR: number;
  trades: number;
  status: 'active' | 'saved';
  equityCurve: number[];
}

interface HeatmapCell {
  ticker: string;
  tf: string;
  score: number; // 0-100 composite score
  count: number;
}

interface Insight {
  id: string;
  type: 'discovery' | 'warning' | 'recommendation';
  title: string;
  body: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AVAILABLE_TFS = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1H'];

const ENTRY_TRIGGERS = [
  'EMA Bull Cross', 'EMA Bear Cross', 'EMA Fan Open Bull',
  'MACD Cross Bull', 'MACD Cross Bear',
  'VWAP Cross Above', 'VWAP Cross Below',
  'UT Bot Buy', 'UT Bot Sell',
];

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

function generateEquityCurve(): number[] {
  const curve: number[] = [0];
  for (let i = 1; i < 30; i++) {
    curve.push(curve[i - 1] + (Math.random() - 0.4) * 0.5);
  }
  return curve;
}

const mockResults: MassResult[] = [
  { rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 2, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 3, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'UT Bot Buy', winRate: 54.0, pf: 2.05, dailyR: 1.78, trades: 201, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 4, ticker: 'TSLA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 178, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 5, ticker: 'AAPL', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 55.0, pf: 1.92, dailyR: 1.45, trades: 95, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 6, ticker: 'MSFT', direction: 'LONG', tf: '5Min', trigger: 'MACD Cross Bull', winRate: 49.5, pf: 1.65, dailyR: 0.98, trades: 210, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 7, ticker: 'GOOG', direction: 'SHORT', tf: '1Min', trigger: 'EMA Bear Cross', winRate: 53.2, pf: 1.71, dailyR: 1.05, trades: 145, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 8, ticker: 'META', direction: 'LONG', tf: '3Min', trigger: 'EMA Bull Cross', winRate: 56.8, pf: 2.11, dailyR: 1.65, trades: 112, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 9, ticker: 'AMZN', direction: 'LONG', tf: '5Min', trigger: 'MACD Cross Bull', winRate: 50.2, pf: 1.55, dailyR: 0.88, trades: 183, status: 'active', equityCurve: generateEquityCurve() },
  { rank: 10, ticker: 'SPY', direction: 'SHORT', tf: '15Min', trigger: 'EMA Bear Cross', winRate: 48.5, pf: 1.42, dailyR: 0.72, trades: 62, status: 'active', equityCurve: generateEquityCurve() },
];

const mockHeatmap: HeatmapCell[] = [
  { ticker: 'NVDA', tf: '1Min', score: 92, count: 3 },
  { ticker: 'NVDA', tf: '5Min', score: 65, count: 1 },
  { ticker: 'SPY', tf: '1Min', score: 45, count: 1 },
  { ticker: 'SPY', tf: '5Min', score: 82, count: 2 },
  { ticker: 'SPY', tf: '15Min', score: 38, count: 1 },
  { ticker: 'TSLA', tf: '1Min', score: 58, count: 2 },
  { ticker: 'TSLA', tf: '5Min', score: 42, count: 1 },
  { ticker: 'AAPL', tf: '5Min', score: 70, count: 1 },
  { ticker: 'MSFT', tf: '5Min', score: 48, count: 1 },
  { ticker: 'GOOG', tf: '1Min', score: 55, count: 1 },
  { ticker: 'META', tf: '3Min', score: 74, count: 1 },
  { ticker: 'AMZN', tf: '5Min', score: 50, count: 1 },
];

const mockInsights: Insight[] = [
  { id: 'i1', type: 'discovery', title: 'NVDA 1Min Dominance', body: 'NVDA on 1Min shows the highest profit factor (3.12) across all combinations. EMA-based triggers outperform MACD for this ticker.' },
  { id: 'i2', type: 'recommendation', title: 'SPY Favors 5Min', body: 'SPY strategies perform significantly better on 5Min vs 1Min. VWAP Cross is the optimal entry trigger with 58.3% WR.' },
  { id: 'i3', type: 'warning', title: 'Low Trade Count', body: 'SPY SHORT 15Min has only 62 trades. Consider extending the backtest period for more statistically significant results.' },
];

const EVAL_STEPS = [
  'NVDA LONG 1Min EMA Bull Cross',
  'NVDA LONG 5Min VWAP Cross Above',
  'SPY LONG 1Min MACD Cross Bull',
  'SPY LONG 5Min VWAP Cross Above',
  'TSLA LONG 1Min EMA Bull Cross',
  'AAPL LONG 5Min VWAP Cross Above',
  'GOOG SHORT 1Min EMA Bear Cross',
  'META LONG 3Min EMA Bull Cross',
];

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ToggleChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all"
      style={{
        background: active ? 'var(--accent-muted)' : 'var(--bg-input)',
        color: active ? 'var(--accent)' : 'var(--text-muted)',
        border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
        transform: active ? 'scale(1.02)' : 'scale(1)',
      }}
    >
      {label}
    </button>
  );
}

/** Mini equity sparkline SVG */
function Sparkline({ data, width = 80, height = 24 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');
  const isPositive = data[data.length - 1] > data[0];
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline
        points={points}
        fill="none"
        stroke={isPositive ? 'var(--green)' : 'var(--red)'}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** Animated progress ticker bar */
function ProgressTicker({ isRunning, progress, currentStep }: {
  isRunning: boolean;
  progress: number;
  currentStep: string;
}) {
  if (!isRunning) return null;

  return (
    <Card>
      <div className="flex items-center gap-3 mb-2">
        <div className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: 'var(--accent)',
                animation: `dot-bounce 1.4s ease-in-out ${i * 0.16}s infinite`,
              }}
            />
          ))}
        </div>
        <span className="text-xs font-medium" style={{ color: 'var(--accent)' }}>
          Evaluating Strategies...
        </span>
        <span className="text-xs font-mono ml-auto" style={{ color: 'var(--text-muted)' }}>
          {Math.round(progress)}%
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 rounded-full overflow-hidden mb-2" style={{ background: 'var(--bg-input)' }}>
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${progress}%`,
            background: 'linear-gradient(90deg, var(--accent), rgba(0,255,136,0.5))',
          }}
        />
      </div>

      {/* Current evaluation ticker */}
      <div
        className="overflow-hidden h-5 rounded"
        style={{ background: 'rgba(0,0,0,0.2)' }}
      >
        <div className="whitespace-nowrap text-xs font-mono px-2 py-0.5" style={{ color: 'var(--text-muted)' }}>
          {currentStep}
        </div>
      </div>
    </Card>
  );
}

/** Interactive scatter plot — WR vs PF with size = trades */
function ScatterPlot({ results, onSelect }: {
  results: MassResult[];
  onSelect: (r: MassResult) => void;
}) {
  const [hovered, setHovered] = useState<number | null>(null);

  const plotW = 400;
  const plotH = 260;
  const pad = { top: 20, right: 20, bottom: 35, left: 45 };
  const w = plotW - pad.left - pad.right;
  const h = plotH - pad.top - pad.bottom;

  const wrMin = 40;
  const wrMax = 70;
  const pfMin = 0;
  const pfMax = 4;

  function scaleX(wr: number) { return pad.left + ((Math.min(Math.max(wr, wrMin), wrMax) - wrMin) / (wrMax - wrMin)) * w; }
  function scaleY(pf: number) { return pad.top + h - ((Math.min(Math.max(pf, pfMin), pfMax) - pfMin) / (pfMax - pfMin)) * h; }
  function scaleR(trades: number) { return Math.max(4, Math.min(16, trades / 15)); }

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
        Strategy Scatter Plot
      </h3>
      <p className="text-[10px] mb-2" style={{ color: 'var(--text-muted)' }}>
        X = Win Rate % | Y = Profit Factor | Size = Trade Count
      </p>

      <svg width="100%" viewBox={`0 0 ${plotW} ${plotH}`} style={{ overflow: 'visible' }}>
        {/* Grid lines */}
        {[0, 1, 2, 3, 4].map((pf) => (
          <g key={`pf-${pf}`}>
            <line
              x1={pad.left} y1={scaleY(pf)} x2={plotW - pad.right} y2={scaleY(pf)}
              stroke="var(--border)" strokeWidth="0.5" strokeDasharray="3 3"
            />
            <text x={pad.left - 6} y={scaleY(pf) + 3} textAnchor="end" fontSize="9" fill="var(--text-muted)">
              {pf.toFixed(1)}
            </text>
          </g>
        ))}
        {[40, 45, 50, 55, 60, 65, 70].map((wr) => (
          <g key={`wr-${wr}`}>
            <line
              x1={scaleX(wr)} y1={pad.top} x2={scaleX(wr)} y2={plotH - pad.bottom}
              stroke="var(--border)" strokeWidth="0.5" strokeDasharray="3 3"
            />
            <text x={scaleX(wr)} y={plotH - pad.bottom + 14} textAnchor="middle" fontSize="9" fill="var(--text-muted)">
              {wr}%
            </text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={plotW / 2} y={plotH - 2} textAnchor="middle" fontSize="10" fill="var(--text-muted)">Win Rate %</text>
        <text x={10} y={plotH / 2} textAnchor="middle" fontSize="10" fill="var(--text-muted)" transform={`rotate(-90, 10, ${plotH / 2})`}>Profit Factor</text>

        {/* Quality zone */}
        <rect
          x={scaleX(50)} y={scaleY(4)} width={scaleX(70) - scaleX(50)} height={scaleY(1.5) - scaleY(4)}
          fill="rgba(0,255,136,0.04)" stroke="rgba(0,255,136,0.1)" strokeDasharray="4 2" rx="4"
        />
        <text x={scaleX(60)} y={scaleY(3.8) + 10} textAnchor="middle" fontSize="8" fill="rgba(0,255,136,0.3)">Quality Zone</text>

        {/* Data points */}
        {results.map((r, i) => {
          const cx = scaleX(r.winRate);
          const cy = scaleY(r.pf);
          const radius = scaleR(r.trades);
          const isHov = hovered === r.rank;
          return (
            <g
              key={r.rank}
              style={{
                cursor: 'pointer',
                animation: `pop-in 0.4s ease ${i * 0.05}s both`,
              }}
              onClick={() => onSelect(r)}
              onMouseEnter={() => setHovered(r.rank)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Glow ring */}
              {isHov && (
                <circle cx={cx} cy={cy} r={radius + 4} fill="none" stroke="var(--accent)" strokeWidth="1.5" opacity="0.5" />
              )}
              <circle
                cx={cx} cy={cy} r={isHov ? radius + 2 : radius}
                fill={r.direction === 'LONG' ? 'var(--green)' : 'var(--red)'}
                opacity={isHov ? 1 : 0.7}
                style={{ transition: 'all 0.15s ease' }}
              />
              <text x={cx} y={cy + 3} textAnchor="middle" fontSize="8" fill="white" fontWeight="bold">
                {r.rank}
              </text>
              {/* Tooltip */}
              {isHov && (
                <g>
                  <rect x={cx + radius + 6} y={cy - 22} width="110" height="38" rx="4" fill="var(--bg-secondary)" stroke="var(--border)" />
                  <text x={cx + radius + 12} y={cy - 8} fontSize="9" fontWeight="bold" fill="var(--text-primary)">
                    {r.ticker} {r.direction} {r.tf}
                  </text>
                  <text x={cx + radius + 12} y={cy + 5} fontSize="8" fill="var(--text-muted)">
                    WR:{r.winRate}% PF:{r.pf} R:{r.dailyR}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </Card>
  );
}

/** Ticker/TF heatmap */
function TickerTfHeatmap({ cells }: { cells: HeatmapCell[] }) {
  const tickers = Array.from(new Set(cells.map((c) => c.ticker)));
  const tfs = Array.from(new Set(cells.map((c) => c.tf))).sort((a, b) => {
    const order = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1H'];
    return order.indexOf(a) - order.indexOf(b);
  });

  function getCell(ticker: string, tf: string) {
    return cells.find((c) => c.ticker === ticker && c.tf === tf);
  }

  function scoreColor(score: number): string {
    if (score >= 80) return 'rgba(0,255,136,0.35)';
    if (score >= 60) return 'rgba(0,255,136,0.18)';
    if (score >= 40) return 'rgba(255,255,0,0.12)';
    return 'rgba(255,80,80,0.1)';
  }

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
        Ticker / Timeframe Heatmap
      </h3>
      <p className="text-[10px] mb-3" style={{ color: 'var(--text-muted)' }}>
        Composite score by ticker + timeframe combination
      </p>

      <div className="overflow-x-auto">
        <table className="w-full" style={{ borderCollapse: 'separate', borderSpacing: '2px' }}>
          <thead>
            <tr>
              <th className="text-[10px] font-medium p-1 text-left" style={{ color: 'var(--text-muted)' }}></th>
              {tfs.map((tf) => (
                <th key={tf} className="text-[10px] font-mono font-medium p-1 text-center" style={{ color: 'var(--text-muted)' }}>
                  {tf}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((ticker, ti) => (
              <tr key={ticker}>
                <td className="text-xs font-medium p-1 pr-2" style={{ color: 'var(--text-primary)' }}>
                  {ticker}
                </td>
                {tfs.map((tf, tfi) => {
                  const cell = getCell(ticker, tf);
                  return (
                    <td key={tf} className="p-0.5">
                      <div
                        className="rounded text-center py-1.5 px-1"
                        style={{
                          background: cell ? scoreColor(cell.score) : 'var(--bg-input)',
                          minWidth: '48px',
                          animation: cell ? `heatmap-reveal 0.3s ease ${(ti * tfs.length + tfi) * 0.04}s both` : undefined,
                        }}
                      >
                        {cell ? (
                          <>
                            <div className="text-xs font-bold" style={{ color: cell.score >= 60 ? 'var(--green)' : 'var(--text-secondary)' }}>
                              {cell.score}
                            </div>
                            <div className="text-[8px]" style={{ color: 'var(--text-muted)' }}>
                              {cell.count} strat{cell.count !== 1 ? 's' : ''}
                            </div>
                          </>
                        ) : (
                          <div className="text-[10px]" style={{ color: 'var(--text-muted)', opacity: 0.3 }}>-</div>
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/** AI-style insight cards */
function InsightCards({ insights }: { insights: Insight[] }) {
  const iconMap = { discovery: 'D', warning: '!', recommendation: 'R' };
  const colorMap = {
    discovery: { bg: 'var(--accent-muted)', color: 'var(--accent)', border: 'var(--accent)' },
    warning: { bg: 'var(--red-muted)', color: 'var(--red)', border: 'var(--red)' },
    recommendation: { bg: 'rgba(100,149,237,0.12)', color: 'rgb(100,149,237)', border: 'rgb(100,149,237)' },
  };

  return (
    <Card>
      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
        Analysis Insights
      </h3>
      <div className="space-y-2.5">
        {insights.map((ins, i) => {
          const c = colorMap[ins.type];
          return (
            <div
              key={ins.id}
              className="flex gap-3 p-3 rounded-lg border"
              style={{
                borderColor: c.border,
                background: c.bg,
                borderLeftWidth: '3px',
                animation: `slide-in-left 0.3s ease ${i * 0.1}s both`,
              }}
            >
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0"
                style={{ background: c.bg, color: c.color, border: `1px solid ${c.border}` }}
              >
                {iconMap[ins.type]}
              </div>
              <div className="min-w-0">
                <h4 className="text-xs font-bold" style={{ color: c.color }}>{ins.title}</h4>
                <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>{ins.body}</p>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function MassBuilderV4() {
  // Config state
  const [tickerInput, setTickerInput] = useState('');
  const [selectedTickers, setSelectedTickers] = useState<string[]>(['NVDA', 'SPY', 'TSLA', 'AAPL', 'MSFT', 'GOOG', 'META', 'AMZN']);
  const [selectedTFs, setSelectedTFs] = useState<string[]>(['1Min', '3Min', '5Min', '15Min']);
  const [direction, setDirection] = useState<'LONG' | 'SHORT' | 'BOTH'>('BOTH');
  const [selectedTriggers, setSelectedTriggers] = useState<string[]>(['EMA Bull Cross', 'EMA Bear Cross', 'VWAP Cross Above', 'MACD Cross Bull', 'UT Bot Buy']);
  const [minTrades, setMinTrades] = useState(10);
  const [minWR, setMinWR] = useState(0);
  const [minPF, setMinPF] = useState(0);

  // Run state
  const [results, setResults] = useState<MassResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [evalStep, setEvalStep] = useState('');
  const [selectedResult, setSelectedResult] = useState<MassResult | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const totalCombinations = useMemo(() => {
    const dirs = direction === 'BOTH' ? 2 : 1;
    return selectedTickers.length * selectedTFs.length * dirs * selectedTriggers.length;
  }, [selectedTickers, selectedTFs, direction, selectedTriggers]);

  function addTicker() {
    const ticker = tickerInput.trim().toUpperCase();
    if (ticker && !selectedTickers.includes(ticker)) {
      setSelectedTickers((prev) => [...prev, ticker]);
    }
    setTickerInput('');
  }

  function removeTicker(ticker: string) {
    setSelectedTickers((prev) => prev.filter((t) => t !== ticker));
  }

  function toggleTF(tf: string) {
    setSelectedTFs((prev) => prev.includes(tf) ? prev.filter((t) => t !== tf) : [...prev, tf]);
  }

  function toggleTrigger(trigger: string) {
    setSelectedTriggers((prev) => prev.includes(trigger) ? prev.filter((t) => t !== trigger) : [...prev, trigger]);
  }

  function runAnalysis() {
    setIsRunning(true);
    setProgress(0);
    setResults([]);
    let step = 0;

    intervalRef.current = setInterval(() => {
      step++;
      const pct = Math.min((step / 30) * 100, 99);
      setProgress(pct);
      setEvalStep(EVAL_STEPS[step % EVAL_STEPS.length]);

      if (step >= 30) {
        if (intervalRef.current) clearInterval(intervalRef.current);
        setProgress(100);
        setResults(mockResults);
        setIsRunning(false);
      }
    }, 120);
  }

  // Clean up interval on unmount
  useEffect(() => {
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  function saveResult(rank: number) {
    setResults((prev) => prev.map((r) => r.rank === rank ? { ...r, status: 'saved' as const } : r));
  }

  const canRun = selectedTickers.length > 0 && selectedTriggers.length > 0 && selectedTFs.length > 0;

  // Sorted results
  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) => b.dailyR - a.dailyR);
  }, [results]);

  // Top performer
  const topResult = sortedResults[0] ?? null;

  return (
    <div>
      <style dangerouslySetInnerHTML={{ __html: ANIMATION_STYLES }} />

      <PageHeader
        title="Strategy Discovery Lab"
        subtitle={`${totalCombinations} combinations \u00b7 ${results.length} discoveries`}
      />

      {/* Combination counter strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <MetricCard label="Tickers" value={String(selectedTickers.length)} />
        <MetricCard label="Timeframes" value={String(selectedTFs.length)} />
        <MetricCard label="Triggers" value={String(selectedTriggers.length)} />
        <MetricCard
          label="Total Combinations"
          value={totalCombinations.toLocaleString()}
          delta={direction === 'BOTH' ? 'LONG + SHORT' : direction}
          positive
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-5">
        {/* ====== LEFT: Configuration ====== */}
        <div className="lg:col-span-1 space-y-4">
          {/* Tickers */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Tickers</h3>
            <div className="flex gap-1.5 flex-wrap mb-2">
              {selectedTickers.map((t) => (
                <span
                  key={t}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  {t}
                  <button onClick={() => removeTicker(t)} className="text-[10px]" style={{ color: 'var(--accent)' }}>x</button>
                </span>
              ))}
            </div>
            <div className="flex gap-1.5">
              <input
                type="text" value={tickerInput}
                onChange={(e) => setTickerInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') addTicker(); }}
                placeholder="Add ticker..."
                className="flex-1 px-2 py-1.5 rounded text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
              <button onClick={addTicker} className="px-2 py-1.5 rounded text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>+</button>
            </div>
          </Card>

          {/* Timeframes + Direction */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Timeframes</h3>
            <div className="flex gap-1.5 flex-wrap mb-3">
              {AVAILABLE_TFS.map((tf) => (
                <ToggleChip key={tf} label={tf} active={selectedTFs.includes(tf)} onClick={() => toggleTF(tf)} />
              ))}
            </div>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Direction</h3>
            <div className="flex gap-1.5">
              {(['LONG', 'SHORT', 'BOTH'] as const).map((d) => (
                <ToggleChip key={d} label={d} active={direction === d} onClick={() => setDirection(d)} />
              ))}
            </div>
          </Card>

          {/* Triggers */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Entry Triggers</h3>
            <div className="flex gap-1.5 flex-wrap">
              {ENTRY_TRIGGERS.map((tr) => (
                <ToggleChip key={tr} label={tr} active={selectedTriggers.includes(tr)} onClick={() => toggleTrigger(tr)} />
              ))}
            </div>
          </Card>

          {/* Criteria + Run */}
          <Card>
            <h3 className="text-xs font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Min Criteria</h3>
            <div className="flex gap-3 mb-4">
              <div>
                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Trades</label>
                <input type="number" min={0} value={minTrades} onChange={(e) => setMinTrades(parseInt(e.target.value) || 0)}
                  className="w-16 px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
              </div>
              <div>
                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>WR%</label>
                <input type="number" min={0} max={100} value={minWR} onChange={(e) => setMinWR(parseInt(e.target.value) || 0)}
                  className="w-16 px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
              </div>
              <div>
                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>PF</label>
                <input type="number" min={0} step={0.1} value={minPF} onChange={(e) => setMinPF(parseFloat(e.target.value) || 0)}
                  className="w-16 px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
              </div>
            </div>
            <button
              onClick={runAnalysis}
              disabled={!canRun || isRunning}
              className="w-full py-2.5 rounded-lg text-sm font-medium transition-all"
              style={{
                background: canRun && !isRunning ? 'var(--accent)' : 'var(--bg-input)',
                color: canRun && !isRunning ? 'white' : 'var(--text-muted)',
                cursor: canRun && !isRunning ? 'pointer' : 'not-allowed',
                animation: canRun && !isRunning ? 'pulse-glow 2s ease-in-out infinite' : 'none',
              }}
            >
              {isRunning ? 'Running...' : `Launch Discovery (${totalCombinations} combos)`}
            </button>
          </Card>
        </div>

        {/* ====== RIGHT: Results ====== */}
        <div className="lg:col-span-2 space-y-4">
          {/* Progress ticker */}
          <ProgressTicker isRunning={isRunning} progress={progress} currentStep={evalStep} />

          {results.length === 0 && !isRunning && (
            <Card>
              <div className="flex flex-col items-center justify-center py-16">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl mb-4"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  S
                </div>
                <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
                  Ready to Discover
                </h3>
                <p className="text-xs text-center max-w-xs" style={{ color: 'var(--text-muted)' }}>
                  Configure your parameters and launch discovery to find the best strategy combinations across tickers, timeframes, and triggers.
                </p>
              </div>
            </Card>
          )}

          {results.length > 0 && (
            <>
              {/* Top performer spotlight */}
              {topResult && (
                <div
                  className="rounded-xl border-2 p-4"
                  style={{
                    borderColor: 'var(--accent)',
                    background: 'linear-gradient(135deg, var(--accent-muted), transparent)',
                    animation: 'fade-in-up 0.4s ease both',
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded font-bold"
                          style={{ background: 'var(--accent)', color: 'white' }}
                        >
                          #1 DISCOVERY
                        </span>
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{
                            background: topResult.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                            color: topResult.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                          }}
                        >
                          {topResult.direction}
                        </span>
                      </div>
                      <h3 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
                        {topResult.ticker} {topResult.tf} &mdash; {topResult.trigger}
                      </h3>
                    </div>
                    <div className="flex gap-5">
                      <div className="text-center">
                        <div className="text-lg font-bold" style={{ color: 'var(--green)' }}>
                          {topResult.winRate}%
                        </div>
                        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Win Rate</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold" style={{ color: 'var(--accent)' }}>
                          {topResult.pf.toFixed(2)}
                        </div>
                        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>PF</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold" style={{ color: 'var(--green)' }}>
                          +{topResult.dailyR.toFixed(2)}R
                        </div>
                        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Daily R</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Scatter plot */}
              <ScatterPlot results={results} onSelect={setSelectedResult} />

              {/* Results table */}
              <Card>
                <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
                  All Results ({results.length})
                </h3>
                {/* Header */}
                <div
                  className="grid items-center gap-2 py-2 px-2 text-[10px] font-medium border-b"
                  style={{ gridTemplateColumns: '28px 56px 44px 48px 1fr 52px 52px 56px 48px 64px 72px', borderColor: 'var(--border)', color: 'var(--text-muted)' }}
                >
                  <span>#</span><span>Ticker</span><span>Dir</span><span>TF</span><span>Trigger</span>
                  <span className="text-right">WR%</span><span className="text-right">PF</span>
                  <span className="text-right">Daily R</span><span className="text-right">Tr</span>
                  <span className="text-center">Curve</span><span className="text-right">Action</span>
                </div>
                {/* Rows */}
                <div>
                  {sortedResults.map((r, i) => (
                    <div
                      key={r.rank}
                      className="grid items-center gap-2 py-2 px-2 text-xs border-b transition-colors"
                      style={{
                        gridTemplateColumns: '28px 56px 44px 48px 1fr 52px 52px 56px 48px 64px 72px',
                        borderColor: 'var(--border)',
                        animation: `fade-in-up 0.2s ease ${i * 0.03}s both`,
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                    >
                      <span className="font-mono" style={{ color: r.rank <= 3 ? 'var(--accent)' : 'var(--text-muted)' }}>{r.rank}</span>
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{r.ticker}</span>
                      <span className="text-[10px] px-1 py-0.5 rounded text-center" style={{
                        background: r.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                        color: r.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                      }}>{r.direction}</span>
                      <span style={{ color: 'var(--text-secondary)' }}>{r.tf}</span>
                      <span className="truncate" style={{ color: 'var(--text-secondary)' }}>{r.trigger}</span>
                      <span className="text-right font-medium" style={{ color: r.winRate >= 50 ? 'var(--green)' : 'var(--red)' }}>{r.winRate.toFixed(1)}</span>
                      <span className="text-right font-medium" style={{ color: r.pf >= 1.5 ? 'var(--green)' : 'var(--text-secondary)' }}>{r.pf.toFixed(2)}</span>
                      <span className="text-right font-medium" style={{ color: r.dailyR > 0 ? 'var(--green)' : 'var(--red)' }}>+{r.dailyR.toFixed(2)}</span>
                      <span className="text-right" style={{ color: 'var(--text-secondary)' }}>{r.trades}</span>
                      <span className="flex justify-center"><Sparkline data={r.equityCurve} width={56} height={18} /></span>
                      <span className="text-right">
                        {r.status === 'saved' ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>Saved</span>
                        ) : (
                          <button onClick={() => saveResult(r.rank)} className="text-[10px] px-2 py-1 rounded font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Save</button>
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Bottom: Heatmap + Insights (only after results) */}
      {results.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <TickerTfHeatmap cells={mockHeatmap} />
          <InsightCards insights={mockInsights} />
        </div>
      )}

      {/* Detail modal */}
      <Modal
        title={selectedResult ? `${selectedResult.ticker} ${selectedResult.direction} ${selectedResult.tf}` : ''}
        isOpen={!!selectedResult}
        onClose={() => setSelectedResult(null)}
        width="480px"
      >
        {selectedResult && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <MetricCard label="Win Rate" value={`${selectedResult.winRate}%`} positive={selectedResult.winRate >= 50} />
              <MetricCard label="Profit Factor" value={selectedResult.pf.toFixed(2)} positive={selectedResult.pf >= 1.5} />
              <MetricCard label="Daily R" value={`+${selectedResult.dailyR.toFixed(2)}`} positive />
              <MetricCard label="Total Trades" value={String(selectedResult.trades)} />
            </div>
            <div>
              <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Equity Curve</h4>
              <Sparkline data={selectedResult.equityCurve} width={400} height={80} />
            </div>
            <div className="flex justify-end gap-2 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
              <button onClick={() => setSelectedResult(null)} className="px-4 py-2 rounded-lg text-sm" style={{ border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Close</button>
              <button
                onClick={() => { saveResult(selectedResult.rank); setSelectedResult(null); }}
                className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}
              >
                Save Strategy
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
