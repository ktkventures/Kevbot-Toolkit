'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Modal from '@/components/Modal';

/* ========================================================================
   Types
   ======================================================================== */

interface MassConfig {
  tickers: string[];
  timeframes: string[];
  directions: ('LONG' | 'SHORT')[];
  entryTriggers: string[];
  exitTriggers: string[];
  exitDepth: number;
  tfConfluences: string[];
  tfConfluenceDepth: number;
  generalConfluences: string[];
  generalConfluenceDepth: number;
  session: string;
  dateRange: { mode: 'days' | 'range'; days: number; start?: string; end?: string };
  stopConfig: { method: string; atrMult?: number; dollarAmount?: number; percentage?: number; lookback?: number; padding?: number };
  targetConfig: { method: string; rrRatio?: number; atrMult?: number } | null;
  requiredPerformance: {
    sortBy: string;
    minTrades: number;
    minWinRate: number | null;
    minProfitFactor: number | null;
    minDailyR: number | null;
    minRSquared: number | null;
  };
  maxResults: number;
}

interface MassResult {
  rank: number;
  ticker: string;
  direction: 'LONG' | 'SHORT';
  tf: string;
  trigger: string;
  confluence: string[];
  winRate: number;
  pf: number;
  dailyR: number;
  trades: number;
  maxDD: number;
  totalR: number;
  avgR: number;
  rSquared: number;
  equityCurve: number[];
  exitTrigger: string;
  stopDesc: string;
  targetDesc: string;
  dateRange: string;
  status: 'active' | 'saved' | 'passed';
}

/* ========================================================================
   Mock Data
   ======================================================================== */

const TICKER_PRESETS: Record<string, string[]> = {
  'Mag 7': ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA'],
  'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA'],
  'Crypto': ['BTC/USD', 'ETH/USD'],
};

const AVAILABLE_TFS = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1Hour', '2Hour', '4Hour'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];
const SORT_OPTIONS = ['Daily R', 'Win Rate', 'Profit Factor', 'R-Squared', 'Total R', 'Trades'];

const EXEC_TYPES = ['[C]', '[L]', '[LC]', '[CC]'] as const;
const EXEC_BADGE_COLOR = '#2196F3';

interface TriggerDef {
  name: string;
  pack: string;
  variation: string;
  execTypes: string[]; // which exec types are available for this trigger
}

const ENTRY_TRIGGER_DEFS: TriggerDef[] = [
  { name: 'Short > Mid Cross', pack: 'EMA Stack', variation: 'Default', execTypes: ['[C]', '[L]', '[LC]', '[CC]'] },
  { name: 'Short < Mid Cross', pack: 'EMA Stack', variation: 'Default', execTypes: ['[C]', '[L]', '[LC]'] },
  { name: 'Fan Open Bull', pack: 'EMA Stack', variation: 'Default', execTypes: ['[C]'] },
  { name: 'Short > Mid Cross', pack: 'EMA Stack', variation: 'Scalping', execTypes: ['[C]', '[L]', '[LC]'] },
  { name: 'Short < Mid Cross', pack: 'EMA Stack', variation: 'Scalping', execTypes: ['[C]', '[L]'] },
  { name: 'Bullish Cross', pack: 'MACD Line', variation: 'Default', execTypes: ['[C]'] },
  { name: 'Bearish Cross', pack: 'MACD Line', variation: 'Default', execTypes: ['[C]'] },
  { name: 'Cross Above VWAP', pack: 'VWAP', variation: 'Default', execTypes: ['[C]', '[L]', '[LC]'] },
  { name: 'Cross Below VWAP', pack: 'VWAP', variation: 'Default', execTypes: ['[C]', '[L]', '[LC]'] },
  { name: 'Buy Signal', pack: 'UT Bot', variation: 'Default', execTypes: ['[L]', '[LC]'] },
  { name: 'Sell Signal', pack: 'UT Bot', variation: 'Default', execTypes: ['[L]', '[LC]'] },
  { name: 'Buy Signal', pack: 'UT Bot', variation: 'Confirmed', execTypes: ['[L]', '[LC]'] },
  { name: 'Price > Short EMA', pack: 'EMA Price Position', variation: 'Default', execTypes: ['[C]', '[L]', '[LC]'] },
  { name: 'Price > Mid EMA', pack: 'EMA Price Position', variation: 'Default', execTypes: ['[C]', '[L]'] },
];

// Flatten for backward compat with existing selection logic
const ENTRY_TRIGGERS = ENTRY_TRIGGER_DEFS.flatMap((t) =>
  t.execTypes.map((exec) => ({
    id: `${t.pack}-${t.variation}-${t.name}-${exec}`.replace(/\s+/g, '-').toLowerCase(),
    name: t.name,
    tag: exec,
    pack: t.pack,
    variation: t.variation,
  }))
);

const EXIT_TRIGGER_DEFS: TriggerDef[] = [
  { name: 'Short < Mid Cross', pack: 'EMA Stack', variation: 'Default', execTypes: ['[C]', '[L]', '[LC]'] },
  { name: 'Short > Mid Cross', pack: 'EMA Stack', variation: 'Default', execTypes: ['[C]', '[L]'] },
  { name: 'Short < Mid Cross', pack: 'EMA Stack', variation: 'Scalping', execTypes: ['[C]', '[L]'] },
  { name: 'Bearish Cross', pack: 'MACD Line', variation: 'Default', execTypes: ['[C]'] },
  { name: 'Cross Below VWAP', pack: 'VWAP', variation: 'Default', execTypes: ['[C]', '[L]'] },
  { name: 'Bar Count Exit', pack: 'Bar Count Exit', variation: 'Default', execTypes: ['[C]'] },
];

const EXIT_TRIGGERS = EXIT_TRIGGER_DEFS.flatMap((t) =>
  t.execTypes.map((exec) => ({
    id: `exit-${t.pack}-${t.variation}-${t.name}-${exec}`.replace(/\s+/g, '-').toLowerCase(),
    name: t.name, tag: exec, pack: t.pack, variation: t.variation,
  }))
);

const FIDELITY_BADGE_COLOR = '#26C6DA';

const TF_CONFLUENCES = [
  { id: '_TF_-EMA_STACK-SML-PB', display: 'SML', pack: 'EMA Stack', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-EMA_STACK-SML-CB', display: 'SML', pack: 'EMA Stack', variation: 'Default', fidelity: '[CB]' },
  { id: '_TF_-EMA_STACK-LMS-PB', display: 'LMS', pack: 'EMA Stack', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-EMA_STACK-SML-SCALP-PB', display: 'SML', pack: 'EMA Stack', variation: 'Scalping', fidelity: '[PB]' },
  { id: '_TF_-MACD_LINE-M>S+-PB', display: 'M>S+', pack: 'MACD Line', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-MACD_LINE-M<S--PB', display: 'M<S-', pack: 'MACD Line', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-VWAP_POSITION->V-PB', display: '>VWAP', pack: 'VWAP', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-VWAP_POSITION-<V-PB', display: '<VWAP', pack: 'VWAP', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-RVOL-HIGH-PB', display: 'HIGH', pack: 'RVOL', variation: 'Default', fidelity: '[PB]' },
  { id: '_TF_-RVOL-LOW-PB', display: 'LOW', pack: 'RVOL', variation: 'Default', fidelity: '[PB]' },
];

const GENERAL_CONFLUENCES = [
  { id: 'GEN-time-rth-open', display: 'RTH Open (9:30-10:30)', pack: 'Time of Day', variation: 'Default' },
  { id: 'GEN-time-midday', display: 'Midday (11:00-13:00)', pack: 'Time of Day', variation: 'Default' },
  { id: 'GEN-time-power-hour', display: 'Power Hour (15:00-16:00)', pack: 'Time of Day', variation: 'Power Hour' },
  { id: 'GEN-day-mon-wed', display: 'Mon-Wed', pack: 'Day of Week', variation: 'Default' },
  { id: 'GEN-day-tue-thu', display: 'Tue-Thu', pack: 'Day of Week', variation: 'Midweek Only' },
  { id: 'GEN-session-rth', display: 'Regular Trading Hours', pack: 'Trading Session', variation: 'Default' },
  { id: 'GEN-session-extended', display: 'Extended Hours', pack: 'Trading Session', variation: 'Default' },
];

const STOP_METHODS = ['ATR', 'Fixed $', 'Pct %', 'Swing'];
const TARGET_METHODS = ['None', 'R:R', 'ATR', 'Fixed $', 'Pct %', 'Swing'];

function generateEquityCurve(totalR: number, trades: number): number[] {
  const curve: number[] = [0];
  const avgStep = totalR / trades;
  for (let i = 1; i <= trades; i++) {
    const prev = curve[i - 1];
    const noise = (Math.random() - 0.4) * Math.abs(avgStep) * 2;
    curve.push(prev + avgStep * 0.7 + noise);
  }
  // Normalize so last point is totalR
  const scale = totalR / (curve[curve.length - 1] || 1);
  return curve.map((v) => Math.round(v * scale * 100) / 100);
}

const mockResults: MassResult[] = [
  {
    rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min',
    trigger: '[C] EMA Bull Cross', confluence: ['SML', 'M>S+'],
    winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, maxDD: -1.8,
    totalR: 34.2, avgR: 0.38, rSquared: 0.87,
    equityCurve: generateEquityCurve(34.2, 89),
    exitTrigger: 'EMA Bear Cross', stopDesc: 'ATR x1.5', targetDesc: 'R:R 2.0',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
  {
    rank: 2, ticker: 'SPY', direction: 'LONG', tf: '5Min',
    trigger: '[C] VWAP Cross Above', confluence: ['SML'],
    winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, maxDD: -2.1,
    totalR: 28.5, avgR: 0.23, rSquared: 0.82,
    equityCurve: generateEquityCurve(28.5, 124),
    exitTrigger: 'VWAP Cross Below', stopDesc: 'Swing 5', targetDesc: 'None',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
  {
    rank: 3, ticker: 'NVDA', direction: 'LONG', tf: '1Min',
    trigger: '[HM] UT Bot Buy', confluence: ['SML', 'M>S+', '>V'],
    winRate: 54.0, pf: 2.05, dailyR: 1.78, trades: 201, maxDD: -2.5,
    totalR: 25.1, avgR: 0.12, rSquared: 0.79,
    equityCurve: generateEquityCurve(25.1, 201),
    exitTrigger: 'MACD Cross Bear', stopDesc: 'ATR x1.5', targetDesc: 'R:R 3.0',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
  {
    rank: 4, ticker: 'TSLA', direction: 'LONG', tf: '1Min',
    trigger: '[C] EMA Bull Cross', confluence: ['SML'],
    winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 178, maxDD: -2.9,
    totalR: 18.7, avgR: 0.11, rSquared: 0.71,
    equityCurve: generateEquityCurve(18.7, 178),
    exitTrigger: 'EMA Bear Cross', stopDesc: 'Fixed $1.00', targetDesc: 'R:R 2.0',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
  {
    rank: 5, ticker: 'AAPL', direction: 'LONG', tf: '5Min',
    trigger: '[L0] VWAP Cross Above', confluence: ['M>S+'],
    winRate: 55.0, pf: 1.92, dailyR: 1.45, trades: 95, maxDD: -2.3,
    totalR: 14.9, avgR: 0.16, rSquared: 0.74,
    equityCurve: generateEquityCurve(14.9, 95),
    exitTrigger: 'VWAP Cross Below', stopDesc: 'ATR x2.0', targetDesc: 'ATR x3.0',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
  {
    rank: 6, ticker: 'MSFT', direction: 'LONG', tf: '5Min',
    trigger: '[C] MACD Cross Bull', confluence: ['SML', '>V'],
    winRate: 49.5, pf: 1.65, dailyR: 0.98, trades: 210, maxDD: -3.1,
    totalR: 12.3, avgR: 0.06, rSquared: 0.68,
    equityCurve: generateEquityCurve(12.3, 210),
    exitTrigger: 'MACD Cross Bear', stopDesc: 'ATR x1.5', targetDesc: 'R:R 2.5',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
  {
    rank: 7, ticker: 'GOOG', direction: 'SHORT', tf: '1Min',
    trigger: '[C] EMA Bear Cross', confluence: ['LMS', 'M<S-'],
    winRate: 53.2, pf: 1.71, dailyR: 1.05, trades: 145, maxDD: -2.7,
    totalR: 10.8, avgR: 0.07, rSquared: 0.65,
    equityCurve: generateEquityCurve(10.8, 145),
    exitTrigger: 'EMA Bull Cross', stopDesc: 'Swing 5', targetDesc: 'None',
    dateRange: '2025-12-20 to 2026-03-20', status: 'active',
  },
];

/* ========================================================================
   Sub-Components
   ======================================================================== */

function MiniEquityCurve({ data, height = 80 }: { data: number[]; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 200;
  const h = height;
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * (h - 8) - 4;
    return `${x},${y}`;
  }).join(' ');

  const final = data[data.length - 1];
  const color = final >= 0 ? 'var(--green, #4CAF50)' : 'var(--red, #f44336)';
  const fillColor = final >= 0 ? 'rgba(76,175,80,0.12)' : 'rgba(244,67,54,0.12)';

  // Build fill polygon
  const fillPoints = `0,${h} ${points} ${w},${h}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={height} preserveAspectRatio="none">
      <polygon points={fillPoints} fill={fillColor} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
      {/* Zero line */}
      {min < 0 && max > 0 && (
        <line
          x1="0" x2={w}
          y1={h - ((0 - min) / range) * (h - 8) - 4}
          y2={h - ((0 - min) / range) * (h - 8) - 4}
          stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="3,3"
        />
      )}
    </svg>
  );
}

function ToggleChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
      style={{
        background: active ? 'var(--accent-muted)' : 'var(--bg-input)',
        color: active ? 'var(--accent)' : 'var(--text-muted)',
        border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
      }}
    >
      {label}
    </button>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function MassBuilderV5() {
  // Config state
  const [searchName, setSearchName] = useState(`Search ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`);
  const [selectedTickers, setSelectedTickers] = useState<string[]>(['NVDA', 'SPY']);
  const [tickerInput, setTickerInput] = useState('');
  const [selectedTFs, setSelectedTFs] = useState<string[]>(['1Min', '5Min']);
  const [selectedDirections, setSelectedDirections] = useState<('LONG' | 'SHORT')[]>(['LONG']);
  const [selectedEntries, setSelectedEntries] = useState<string[]>(['ema-bull-cross', 'vwap-cross-up']);
  const [selectedExits, setSelectedExits] = useState<string[]>(['ema-bear-cross-exit']);
  const [exitDepth, setExitDepth] = useState(1);
  const [selectedTfConf, setSelectedTfConf] = useState<string[]>(['_TF_-EMA_STACK-SML', '_TF_-MACD_LINE-M>S+']);
  const [tfConfDepth, setTfConfDepth] = useState(2);
  const [selectedGenConf, setSelectedGenConf] = useState<string[]>([]);
  const [genConfDepth, setGenConfDepth] = useState(1);
  const [session, setSession] = useState('RTH');
  const [lookbackDays, setLookbackDays] = useState(90);
  const [stopMethod, setStopMethod] = useState('ATR');
  const [stopAtrMult, setStopAtrMult] = useState(1.5);
  const [targetMethod, setTargetMethod] = useState('R:R');
  const [targetRR, setTargetRR] = useState(2.0);
  const [sortBy, setSortBy] = useState('Daily R');
  const [minTrades, setMinTrades] = useState(10);
  const [minWR, setMinWR] = useState(0);
  const [minPF, setMinPF] = useState(0);
  const [maxResults, setMaxResults] = useState(500);

  // UI state
  const [activeConfigTab, setActiveConfigTab] = useState('Tickers');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState('');
  const [results, setResults] = useState<MassResult[]>([]);
  const [expandedResult, setExpandedResult] = useState<number | null>(null);

  // Post-analysis filter state
  const [filterWR, setFilterWR] = useState(0);
  const [filterPF, setFilterPF] = useState(0);
  const [filterTrades, setFilterTrades] = useState(0);
  const [filterSort, setFilterSort] = useState('Daily R');
  const [showPassed, setShowPassed] = useState(false);

  const CONFIG_TABS = ['Tickers', 'Timeframes', 'Direction', 'Entry', 'Exit', 'TF Confluence', 'General', 'Stop Loss', 'Take Profit'];

  // Combination estimates
  const estimate = useMemo(() => {
    const nTickers = selectedTickers.length;
    const nTFs = selectedTFs.length;
    const nDirs = selectedDirections.length;
    const nEntries = selectedEntries.length;
    const nExits = Math.max(selectedExits.length, 1);
    const baseConfigs = nTickers * nTFs * nDirs * nEntries;
    // TF confluence combos (simplified)
    let tfCombos = 1;
    for (let d = 1; d <= tfConfDepth && d <= selectedTfConf.length; d++) {
      tfCombos += Math.round(factorial(selectedTfConf.length) / (factorial(d) * factorial(selectedTfConf.length - d)));
    }
    const total = baseConfigs * nExits * tfCombos;
    const estSeconds = total * 0.35; // ~350ms per eval
    return { nTickers, nTFs, nDirs, nEntries, nExits, baseConfigs, total, estSeconds };
  }, [selectedTickers, selectedTFs, selectedDirections, selectedEntries, selectedExits, selectedTfConf, tfConfDepth]);

  function factorial(n: number): number {
    if (n <= 1) return 1;
    let r = 1;
    for (let i = 2; i <= n; i++) r *= i;
    return r;
  }

  function formatTime(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  }

  const canAnalyze = selectedTickers.length > 0 && selectedEntries.length > 0;

  function runAnalysis() {
    setIsAnalyzing(true);
    setProgress(0);
    setProgressLabel('Starting analysis...');

    // Simulate analysis with progress
    let step = 0;
    const totalSteps = 20;
    const interval = setInterval(() => {
      step++;
      const pct = step / totalSteps;
      setProgress(pct);
      if (step < totalSteps * 0.3) {
        setProgressLabel(`Loading data for ${selectedTickers.join(', ')}...`);
      } else if (step < totalSteps * 0.8) {
        setProgressLabel(`Testing combinations... ${Math.round(pct * estimate.total).toLocaleString()} / ${estimate.total.toLocaleString()}`);
      } else {
        setProgressLabel('Ranking results...');
      }

      if (step >= totalSteps) {
        clearInterval(interval);
        setResults(mockResults);
        setIsAnalyzing(false);
        setProgress(1);
        setProgressLabel(`Complete - ${mockResults.length} strategies found`);
      }
    }, 200);
  }

  // Apply post-analysis filters
  const filteredResults = useMemo(() => {
    let filtered = results.filter((r) => {
      if (!showPassed && r.status === 'passed') return false;
      if (filterWR > 0 && r.winRate < filterWR) return false;
      if (filterPF > 0 && r.pf < filterPF) return false;
      if (filterTrades > 0 && r.trades < filterTrades) return false;
      return true;
    });
    // Sort
    const sortKeyMap: Record<string, keyof MassResult> = {
      'Daily R': 'dailyR', 'Win Rate': 'winRate', 'Profit Factor': 'pf',
      'R-Squared': 'rSquared', 'Total R': 'totalR', 'Trades': 'trades',
    };
    const sk = sortKeyMap[filterSort] || 'dailyR';
    filtered.sort((a, b) => (b[sk] as number) - (a[sk] as number));
    return filtered;
  }, [results, filterSort, filterWR, filterPF, filterTrades, showPassed]);

  function toggleItem<T>(arr: T[], item: T, setter: (v: T[]) => void) {
    if (arr.includes(item)) {
      setter(arr.filter((x) => x !== item));
    } else {
      setter([...arr, item]);
    }
  }

  function saveResult(idx: number) {
    setResults((prev) => prev.map((r, i) => (i === idx ? { ...r, status: 'saved' as const } : r)));
  }

  function passResult(idx: number) {
    setResults((prev) =>
      prev.map((r, i) =>
        i === idx ? { ...r, status: r.status === 'passed' ? 'active' as const : 'passed' as const } : r
      )
    );
  }

  return (
    <div>
      <PageHeader title="Mass Strategy Builder" subtitle="Bulk strategy discovery and optimization engine" />

      {/* ====== Header Row: Search Name + Save ====== */}
      <div className="flex items-end gap-4 mb-5">
        <div className="flex-1">
          <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
            Search Name
          </label>
          <input
            type="text"
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          />
        </div>
        <button
          className="px-5 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
        >
          Save Search
        </button>
      </div>

      {/* ====== Configuration Panel ====== */}
      <Card className="mb-5">
        {/* Config tab bar */}
        <div className="flex gap-1 overflow-x-auto pb-1 mb-4 border-b" style={{ borderColor: 'var(--border)' }}>
          {CONFIG_TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveConfigTab(tab)}
              className="px-3 py-2 text-xs font-medium whitespace-nowrap transition-colors relative"
              style={{
                color: activeConfigTab === tab ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: activeConfigTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                marginBottom: '-1px',
              }}
            >
              {tab}
              {/* Badge for selected count */}
              {tab === 'Tickers' && selectedTickers.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedTickers.length}
                </span>
              )}
              {tab === 'Entry' && selectedEntries.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedEntries.length}
                </span>
              )}
              {tab === 'Exit' && selectedExits.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedExits.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="min-h-[180px]">
          {/* --- Tickers --- */}
          {activeConfigTab === 'Tickers' && (
            <div>
              <p className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                Tickers to analyze
              </p>
              {/* Quick-add presets */}
              <div className="flex gap-2 mb-3">
                {Object.entries(TICKER_PRESETS).map(([name, tickers]) => (
                  <button
                    key={name}
                    onClick={() => {
                      const merged = new Set([...selectedTickers, ...tickers]);
                      setSelectedTickers(Array.from(merged).sort());
                    }}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                  >
                    + {name}
                  </button>
                ))}
                {selectedTickers.length > 0 && (
                  <button
                    onClick={() => setSelectedTickers([])}
                    className="px-3 py-1.5 rounded-lg text-xs"
                    style={{ color: 'var(--red)' }}
                  >
                    Clear All
                  </button>
                )}
              </div>
              {/* Text input */}
              <div className="flex gap-2 mb-3">
                <input
                  type="text"
                  placeholder="Enter tickers (comma separated)..."
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const raw = tickerInput.replace(/\n/g, ',').replace(/;/g, ',');
                      const tickers = raw.split(',').map((t) => t.trim().toUpperCase()).filter(Boolean);
                      setSelectedTickers((prev) => Array.from(new Set([...prev, ...tickers])).sort());
                      setTickerInput('');
                    }
                  }}
                  className="flex-1 px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
                <button
                  onClick={() => {
                    const raw = tickerInput.replace(/\n/g, ',').replace(/;/g, ',');
                    const tickers = raw.split(',').map((t) => t.trim().toUpperCase()).filter(Boolean);
                    setSelectedTickers((prev) => Array.from(new Set([...prev, ...tickers])).sort());
                    setTickerInput('');
                  }}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--accent)', color: 'white' }}
                >
                  Add
                </button>
              </div>
              {/* Selected chips */}
              <div className="flex flex-wrap gap-1.5">
                {selectedTickers.map((t) => (
                  <span
                    key={t}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: '1px solid var(--accent)' }}
                  >
                    {t}
                    <button
                      onClick={() => setSelectedTickers((prev) => prev.filter((x) => x !== t))}
                      className="ml-0.5 text-xs opacity-60 hover:opacity-100"
                    >
                      x
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* --- Timeframes --- */}
          {activeConfigTab === 'Timeframes' && (
            <div>
              <p className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                Select timeframes to test
              </p>
              <div className="flex flex-wrap gap-2">
                {AVAILABLE_TFS.map((tf) => (
                  <ToggleChip
                    key={tf}
                    label={tf}
                    active={selectedTFs.includes(tf)}
                    onClick={() => toggleItem(selectedTFs, tf, setSelectedTFs)}
                  />
                ))}
              </div>
              <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                {selectedTFs.length} timeframe{selectedTFs.length !== 1 ? 's' : ''} selected
              </p>
            </div>
          )}

          {/* --- Direction --- */}
          {activeConfigTab === 'Direction' && (
            <div>
              <p className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                Select directions to test
              </p>
              <div className="flex gap-3">
                {(['LONG', 'SHORT'] as const).map((dir) => (
                  <label key={dir} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedDirections.includes(dir)}
                      onChange={() => {
                        if (selectedDirections.includes(dir)) {
                          if (selectedDirections.length > 1) {
                            setSelectedDirections((prev) => prev.filter((d) => d !== dir));
                          }
                        } else {
                          setSelectedDirections((prev) => [...prev, dir]);
                        }
                      }}
                      className="rounded"
                    />
                    <span className="text-sm font-medium" style={{ color: dir === 'LONG' ? 'var(--green, #4CAF50)' : 'var(--red, #f44336)' }}>
                      {dir}
                    </span>
                  </label>
                ))}
              </div>
              <div className="mt-4">
                <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Session</p>
                <div className="flex flex-wrap gap-2">
                  {SESSIONS.map((s) => (
                    <ToggleChip key={s} label={s} active={session === s} onClick={() => setSession(s)} />
                  ))}
                </div>
              </div>
              <div className="mt-4">
                <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Lookback</p>
                <div className="flex items-center gap-3">
                  <input
                    type="number"
                    min={7}
                    max={1825}
                    value={lookbackDays}
                    onChange={(e) => setLookbackDays(parseInt(e.target.value, 10) || 90)}
                    className="w-24 px-3 py-1.5 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>days</span>
                </div>
              </div>
            </div>
          )}

          {/* --- Entry Triggers --- */}
          {activeConfigTab === 'Entry' && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select entry triggers to test</p>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>Full backtest each</span>
                </div>
                <span className="text-xs" style={{ color: 'var(--accent)' }}>{selectedEntries.length} selected</span>
              </div>
              {/* Multi-column scrollable grid grouped by pack > variation > trigger with inline exec type checkboxes */}
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {Object.entries(
                  ENTRY_TRIGGER_DEFS.reduce<Record<string, Record<string, TriggerDef[]>>>((acc, t) => {
                    if (!acc[t.pack]) acc[t.pack] = {};
                    if (!acc[t.pack][t.variation]) acc[t.pack][t.variation] = [];
                    acc[t.pack][t.variation].push(t);
                    return acc;
                  }, {})
                ).map(([pack, variations]) => (
                  <div key={pack} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    {Object.entries(variations).map(([variation, triggers], vi) => (
                      <div key={variation} className={vi > 0 ? 'mt-3 pt-3 border-t' : ''} style={vi > 0 ? { borderColor: 'var(--border)' } : undefined}>
                        {/* Pack name + variation in parenthetical display style */}
                        <p className="text-xs font-semibold mb-2">
                          <span style={{ color: 'var(--text-primary)' }}>{pack}</span>
                          {' '}<span style={{ color: 'var(--text-muted)' }}>({variation})</span>
                        </p>
                        <div className="space-y-1.5">
                          {triggers.map((t, ti) => (
                            <div key={ti} className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-xs min-w-[100px]" style={{ color: 'var(--text-secondary)' }}>{t.name}</span>
                              {t.execTypes.map((exec) => {
                                const triggerId = `${t.pack}-${t.variation}-${t.name}-${exec}`.replace(/\s+/g, '-').toLowerCase();
                                const isChecked = selectedEntries.includes(triggerId);
                                return (
                                  <label key={exec} className="flex items-center gap-0.5 cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={isChecked}
                                      onChange={() => toggleItem(selectedEntries, triggerId, setSelectedEntries)}
                                      className="w-3 h-3 rounded"
                                      style={{ accentColor: EXEC_BADGE_COLOR }}
                                    />
                                    <span className="text-[10px] font-mono font-semibold px-1 py-0.5 rounded" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>
                                      {exec}
                                    </span>
                                  </label>
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* --- Exit Triggers --- */}
          {activeConfigTab === 'Exit' && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select exit triggers to test</p>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>Full backtest each</span>
                </div>
                <span className="text-xs" style={{ color: 'var(--accent)' }}>{selectedExits.length} selected</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-4" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {Object.entries(
                  EXIT_TRIGGER_DEFS.reduce<Record<string, Record<string, TriggerDef[]>>>((acc, t) => {
                    if (!acc[t.pack]) acc[t.pack] = {};
                    if (!acc[t.pack][t.variation]) acc[t.pack][t.variation] = [];
                    acc[t.pack][t.variation].push(t);
                    return acc;
                  }, {})
                ).map(([pack, variations]) => (
                  <div key={pack} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    {Object.entries(variations).map(([variation, triggers], vi) => (
                      <div key={variation} className={vi > 0 ? 'mt-3 pt-3 border-t' : ''} style={vi > 0 ? { borderColor: 'var(--border)' } : undefined}>
                        <p className="text-xs font-semibold mb-2">
                          <span style={{ color: 'var(--text-primary)' }}>{pack}</span>
                          {' '}<span style={{ color: 'var(--text-muted)' }}>({variation})</span>
                        </p>
                        <div className="space-y-1.5">
                          {triggers.map((t, ti) => (
                            <div key={ti} className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-xs min-w-[100px]" style={{ color: 'var(--text-secondary)' }}>{t.name}</span>
                              {t.execTypes.map((exec) => {
                                const triggerId = `exit-${t.pack}-${t.variation}-${t.name}-${exec}`.replace(/\s+/g, '-').toLowerCase();
                                const isChecked = selectedExits.includes(triggerId);
                                return (
                                  <label key={exec} className="flex items-center gap-0.5 cursor-pointer">
                                    <input type="checkbox" checked={isChecked} onChange={() => toggleItem(selectedExits, triggerId, setSelectedExits)} className="w-3 h-3 rounded" style={{ accentColor: EXEC_BADGE_COLOR }} />
                                    <span className="text-[10px] font-mono font-semibold px-1 py-0.5 rounded" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>{exec}</span>
                                  </label>
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-muted)' }}>Exit combination depth</label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4].map((d) => (
                    <ToggleChip key={d} label={`${d}`} active={exitDepth === d} onClick={() => setExitDepth(d)} />
                  ))}
                </div>
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Depth 1 = test individually. 2+ = test combinations.</p>
              </div>
            </div>
          )}

          {/* --- TF Confluence --- */}
          {activeConfigTab === 'TF Confluence' && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select TF confluences for auto-search</p>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>Fast filter</span>
                </div>
                <div className="flex items-center gap-2">
                  <button className="text-[10px] px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)', cursor: 'pointer', border: 'none' }}
                    onClick={() => setSelectedTfConf(TF_CONFLUENCES.map((c) => c.id))}>Select All</button>
                  {selectedTfConf.length > 0 && (
                    <button className="text-[10px] px-2 py-0.5 rounded" style={{ color: 'var(--text-muted)', cursor: 'pointer', border: 'none', background: 'transparent' }}
                      onClick={() => setSelectedTfConf([])}>Clear</button>
                  )}
                  <span className="text-xs" style={{ color: 'var(--accent)' }}>{selectedTfConf.length} selected</span>
                </div>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {Object.entries(
                  TF_CONFLUENCES.reduce<Record<string, Record<string, typeof TF_CONFLUENCES>>>((acc, c) => {
                    if (!acc[c.pack]) acc[c.pack] = {};
                    if (!acc[c.pack][c.variation]) acc[c.pack][c.variation] = [];
                    acc[c.pack][c.variation].push(c);
                    return acc;
                  }, {})
                ).map(([pack, variations]) => (
                  <div key={pack} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    {Object.entries(variations).map(([variation, confs], vi) => (
                      <div key={variation} className={vi > 0 ? 'mt-3 pt-3 border-t' : ''} style={vi > 0 ? { borderColor: 'var(--border)' } : undefined}>
                        <p className="text-xs font-semibold mb-2">
                          <span style={{ color: 'var(--text-primary)' }}>{pack}</span>
                          {' '}<span style={{ color: 'var(--text-muted)' }}>({variation})</span>
                        </p>
                        <div className="space-y-1.5">
                          {/* Group by state name, show fidelity badges inline */}
                          {Object.entries(
                            confs.reduce<Record<string, typeof confs>>((acc, c) => {
                              (acc[c.display] = acc[c.display] || []).push(c);
                              return acc;
                            }, {})
                          ).map(([display, fidelityConfs]) => (
                            <div key={display} className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-xs font-mono min-w-[60px]" style={{ color: 'var(--text-secondary)' }}>{display}</span>
                              {fidelityConfs.map((c) => (
                                <label key={c.id} className="flex items-center gap-0.5 cursor-pointer">
                                  <input type="checkbox" checked={selectedTfConf.includes(c.id)} onChange={() => toggleItem(selectedTfConf, c.id, setSelectedTfConf)} className="w-3 h-3 rounded" style={{ accentColor: FIDELITY_BADGE_COLOR }} />
                                  <span className="text-[10px] font-mono font-semibold px-1 py-0.5 rounded" style={{ color: FIDELITY_BADGE_COLOR, background: FIDELITY_BADGE_COLOR + '20' }}>{c.fidelity}</span>
                                </label>
                              ))}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-muted)' }}>TF confluence depth</label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4].map((d) => (
                    <ToggleChip key={d} label={`${d}`} active={tfConfDepth === d} onClick={() => setTfConfDepth(d)} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* --- General Confluence --- */}
          {activeConfigTab === 'General' && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select general confluences</p>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>Fast filter</span>
                </div>
                <div className="flex items-center gap-2">
                  <button className="text-[10px] px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)', cursor: 'pointer', border: 'none' }}
                    onClick={() => setSelectedGenConf(GENERAL_CONFLUENCES.map((c) => c.id))}>Select All</button>
                  {selectedGenConf.length > 0 && (
                    <button className="text-[10px] px-2 py-0.5 rounded" style={{ color: 'var(--text-muted)', cursor: 'pointer', border: 'none', background: 'transparent' }}
                      onClick={() => setSelectedGenConf([])}>Clear</button>
                  )}
                  <span className="text-xs" style={{ color: 'var(--accent)' }}>{selectedGenConf.length} selected</span>
                </div>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {Object.entries(
                  GENERAL_CONFLUENCES.reduce<Record<string, Record<string, typeof GENERAL_CONFLUENCES>>>((acc, c) => {
                    if (!acc[c.pack]) acc[c.pack] = {};
                    if (!acc[c.pack][c.variation]) acc[c.pack][c.variation] = [];
                    acc[c.pack][c.variation].push(c);
                    return acc;
                  }, {})
                ).map(([pack, variations]) => (
                  <div key={pack} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    {Object.entries(variations).map(([variation, confs], vi) => (
                      <div key={variation} className={vi > 0 ? 'mt-3 pt-3 border-t' : ''} style={vi > 0 ? { borderColor: 'var(--border)' } : undefined}>
                        <p className="text-xs font-semibold mb-2">
                          <span style={{ color: 'var(--text-primary)' }}>{pack}</span>
                          {' '}<span style={{ color: 'var(--text-muted)' }}>({variation})</span>
                        </p>
                        <div className="space-y-1">
                          {confs.map((c) => (
                            <label key={c.id} className="flex items-center gap-2 cursor-pointer py-0.5">
                              <input type="checkbox" checked={selectedGenConf.includes(c.id)} onChange={() => toggleItem(selectedGenConf, c.id, setSelectedGenConf)} className="rounded" style={{ accentColor: 'var(--accent)' }} />
                              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{c.display}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
                <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-muted)' }}>General confluence depth</label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4].map((d) => (
                    <ToggleChip key={d} label={`${d}`} active={genConfDepth === d} onClick={() => setGenConfDepth(d)} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* --- Stop Loss --- */}
          {activeConfigTab === 'Stop Loss' && (() => {
            const STOP_PACKS = [
              { id: 'atr-stop-def', name: '1.5x ATR', pack: 'ATR Stop', variation: 'Default' },
              { id: 'atr-stop-wide', name: '2.0x ATR', pack: 'ATR Stop', variation: 'Wide' },
              { id: 'atr-stop-tight', name: '1.0x ATR', pack: 'ATR Stop', variation: 'Tight' },
              { id: 'swing-stop-def', name: '5-bar, $0.05 pad', pack: 'Swing Stop', variation: 'Default' },
              { id: 'swing-stop-wide', name: '10-bar, $0.10 pad', pack: 'Swing Stop', variation: 'Wide' },
              { id: 'fixed-stop-def', name: '$1.00 stop', pack: 'Fixed Dollar', variation: 'Default' },
              { id: 'fixed-stop-half', name: '$0.50 stop', pack: 'Fixed Dollar', variation: 'Example' },
              { id: 'pct-stop-def', name: '0.5% stop', pack: 'Percentage', variation: 'Default' },
              { id: 'atr-trail-def', name: '1.5x ATR, trails', pack: 'ATR Trailing', variation: 'Default' },
              { id: 'breakeven-def', name: 'Move to BE at 1R', pack: 'Breakeven', variation: 'Default' },
              { id: 'breakeven-half', name: 'Move to BE at 0.5R', pack: 'Breakeven', variation: 'Example' },
            ];
            return (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select stop loss packs to test</p>
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>Full backtest each</span>
                  </div>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Select multiple to test in combination</span>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3" style={{ maxHeight: 400, overflowY: 'auto' }}>
                  {Object.entries(
                    STOP_PACKS.reduce<Record<string, Record<string, typeof STOP_PACKS>>>((acc, p) => {
                      if (!acc[p.pack]) acc[p.pack] = {};
                      if (!acc[p.pack][p.variation]) acc[p.pack][p.variation] = [];
                      acc[p.pack][p.variation].push(p);
                      return acc;
                    }, {})
                  ).map(([pack, variations]) => (
                    <div key={pack} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                      {Object.entries(variations).map(([variation, packs], vi) => (
                        <div key={variation} className={vi > 0 ? 'mt-3 pt-3 border-t' : ''} style={vi > 0 ? { borderColor: 'var(--border)' } : undefined}>
                          <p className="text-xs font-semibold mb-2">
                            <span style={{ color: 'var(--text-primary)' }}>{pack}</span>
                            {' '}<span style={{ color: 'var(--text-muted)' }}>({variation})</span>
                          </p>
                          <div className="space-y-1">
                            {packs.map((p) => (
                              <label key={p.id} className="flex items-center gap-2 cursor-pointer py-0.5">
                                <input type="checkbox" defaultChecked={p.id === 'atr-stop-def'} className="rounded" style={{ accentColor: 'var(--red)' }} />
                                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{p.name}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}

          {/* --- Take Profit --- */}
          {activeConfigTab === 'Take Profit' && (() => {
            const TARGET_PACKS = [
              { id: 'rr-def', name: '2:1 risk/reward', pack: 'Risk:Reward', variation: 'Default' },
              { id: 'rr-3to1', name: '3:1 risk/reward', pack: 'Risk:Reward', variation: 'Aggressive' },
              { id: 'rr-1.5to1', name: '1.5:1 risk/reward', pack: 'Risk:Reward', variation: 'Conservative' },
              { id: 'atr-target-def', name: '2.0x ATR target', pack: 'ATR Target', variation: 'Default' },
              { id: 'atr-target-wide', name: '3.0x ATR target', pack: 'ATR Target', variation: 'Example' },
              { id: 'fixed-target-def', name: '$2.00 target', pack: 'Fixed Dollar', variation: 'Default' },
              { id: 'pct-target-def', name: '1% target', pack: 'Percentage', variation: 'Default' },
              { id: 'swing-target-def', name: '5-bar swing high', pack: 'Swing Target', variation: 'Default' },
              { id: 'no-target', name: 'Signal or bar count only', pack: 'No Target', variation: 'Default' },
            ];
            return (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select take profit packs to test</p>
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>Full backtest each</span>
                  </div>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Select multiple to test in combination</span>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3" style={{ maxHeight: 400, overflowY: 'auto' }}>
                  {Object.entries(
                    TARGET_PACKS.reduce<Record<string, Record<string, typeof TARGET_PACKS>>>((acc, p) => {
                      if (!acc[p.pack]) acc[p.pack] = {};
                      if (!acc[p.pack][p.variation]) acc[p.pack][p.variation] = [];
                      acc[p.pack][p.variation].push(p);
                      return acc;
                    }, {})
                  ).map(([pack, variations]) => (
                    <div key={pack} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                      {Object.entries(variations).map(([variation, packs], vi) => (
                        <div key={variation} className={vi > 0 ? 'mt-3 pt-3 border-t' : ''} style={vi > 0 ? { borderColor: 'var(--border)' } : undefined}>
                          <p className="text-xs font-semibold mb-2">
                            <span style={{ color: 'var(--text-primary)' }}>{pack}</span>
                            {' '}<span style={{ color: 'var(--text-muted)' }}>({variation})</span>
                          </p>
                          <div className="space-y-1">
                            {packs.map((p) => (
                              <label key={p.id} className="flex items-center gap-2 cursor-pointer py-0.5">
                                <input type="checkbox" defaultChecked={p.id === 'rr-def'} className="rounded" style={{ accentColor: 'var(--green)' }} />
                                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{p.name}</span>
                              </label>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>
      </Card>

      {/* ====== Estimation + Required Performance + Analyze ====== */}
      <div className="grid grid-cols-12 gap-4 mb-5">
        {/* Preview — compact */}
        <Card className="col-span-3">
          <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Preview</p>
          <div className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
            {estimate.total.toLocaleString()}
          </div>
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>total evaluations &middot; Est. {formatTime(estimate.estSeconds)}</p>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-2 text-[10px]" style={{ color: 'var(--text-secondary)' }}>
            <span>{estimate.nTickers} ticker{estimate.nTickers !== 1 ? 's' : ''}</span>
            <span>{estimate.nTFs} TF{estimate.nTFs !== 1 ? 's' : ''}</span>
            <span>{estimate.nDirs} dir</span>
            <span>{estimate.nEntries} entry</span>
            <span>{estimate.nExits} exit</span>
          </div>
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>Last {lookbackDays} days</p>
        </Card>

        {/* Required Performance — 4-column compact grid */}
        <Card className="col-span-7">
          <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Required Performance</p>
          <div className="grid grid-cols-4 gap-x-3 gap-y-2">
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Prioritize By</label>
              <select className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                {SORT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min Trades</label>
              <input type="number" min={1} max={500} value={minTrades} onChange={(e) => setMinTrades(parseInt(e.target.value, 10) || 10)}
                className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min WR %</label>
              <input type="number" min={0} max={100} step={5} value={minWR} onChange={(e) => setMinWR(parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min PF</label>
              <input type="number" min={0} max={20} step={0.25} value={minPF} onChange={(e) => setMinPF(parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min Daily R</label>
              <input type="number" min={0} max={10} step={0.1} defaultValue={0}
                className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min R²</label>
              <input type="number" min={0} max={1} step={0.05} defaultValue={0}
                className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
            </div>
            <div>
              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Max Results</label>
              <input type="number" min={10} max={5000} step={50} value={maxResults} onChange={(e) => setMaxResults(parseInt(e.target.value, 10) || 500)}
                className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
            </div>
          </div>
        </Card>

        {/* Analyze button */}
        <Card className="col-span-2 flex flex-col items-center justify-center">
          <button
            onClick={runAnalysis}
            disabled={!canAnalyze || isAnalyzing}
            className="w-full py-4 rounded-xl text-base font-bold transition-opacity hover:opacity-80 disabled:opacity-40"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze'}
          </button>
          {!canAnalyze && (
            <p className="text-[10px] mt-2 text-center" style={{ color: 'var(--text-muted)' }}>
              Select tickers + entry triggers
            </p>
          )}
        </Card>
      </div>

      {/* ====== Progress Bar ====== */}
      {(isAnalyzing || progress > 0) && (
        <Card className="mb-5">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
              {progressLabel}
            </p>
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
              {Math.round(progress * 100)}%
            </span>
          </div>
          <div
            className="h-2 rounded-full overflow-hidden"
            style={{ background: 'var(--bg-input)' }}
          >
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${progress * 100}%`,
                background: progress >= 1 ? 'var(--green, #4CAF50)' : 'var(--accent)',
              }}
            />
          </div>
          {results.length > 0 && progress >= 1 && (
            <div className="flex gap-4 mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <span>{results.length} strategies found</span>
              <span>Best WR: {Math.max(...results.map((r) => r.winRate)).toFixed(1)}%</span>
              <span>Best PF: {Math.max(...results.map((r) => r.pf)).toFixed(2)}</span>
              <span>Best Daily R: {Math.max(...results.map((r) => r.dailyR)).toFixed(2)}</span>
            </div>
          )}
        </Card>
      )}

      {/* ====== Results Section ====== */}
      {results.length > 0 && (
        <>
          {/* Post-analysis filters */}
          <div className="grid grid-cols-2 md:grid-cols-8 gap-3 mb-4">
            <div>
              <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Sort by</label>
              <select
                value={filterSort}
                onChange={(e) => setFilterSort(e.target.value)}
                className="w-full px-2 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                {SORT_OPTIONS.map((o) => <option key={o}>{o}</option>)}
              </select>
            </div>
            <div>
              <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Min WR%</label>
              <input
                type="number" min={0} max={100} step={5} value={filterWR}
                onChange={(e) => setFilterWR(parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
            </div>
            <div>
              <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Min PF</label>
              <input
                type="number" min={0} max={20} step={0.25} value={filterPF}
                onChange={(e) => setFilterPF(parseFloat(e.target.value) || 0)}
                className="w-full px-2 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
            </div>
            <div>
              <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Min Trades</label>
              <input
                type="number" min={0} max={500} value={filterTrades}
                onChange={(e) => setFilterTrades(parseInt(e.target.value, 10) || 0)}
                className="w-full px-2 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
            </div>
            <div>
              <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Min R²</label>
              <input
                type="number" min={0} max={1} step={0.05} defaultValue={0}
                className="w-full px-2 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
            </div>
            <div>
              <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Trade Qual.</label>
              <select className="w-full px-2 py-1.5 rounded-lg text-xs"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                defaultValue="None">
                <option value="None">None</option>
                <option value="ttp">Trade The Pool</option>
                <option value="ftmo">FTMO</option>
                <option value="topstep">Topstep</option>
                <option value="custom">My Custom Rules</option>
              </select>
            </div>
            <div className="flex items-end gap-3">
              <label className="flex items-center gap-2 cursor-pointer pb-1">
                <input type="checkbox" checked={showPassed} onChange={(e) => setShowPassed(e.target.checked)} className="rounded" />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Show passed</span>
              </label>
              <p className="text-xs pb-1.5" style={{ color: 'var(--text-muted)' }}>
                {filteredResults.length}/{results.length}
              </p>
            </div>
          </div>

          {/* Result cards */}
          <div className="space-y-3">
            {filteredResults.map((result, idx) => {
              const isPassed = result.status === 'passed';
              const isSaved = result.status === 'saved';
              const isExpanded = expandedResult === idx;

              return (
                <div key={idx} style={{ opacity: isPassed ? 0.5 : 1 }}>
                <Card>
                  <div className="flex gap-4">
                    {/* Left: info */}
                    <div className="flex-1 min-w-0">
                      {/* Header */}
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span
                          className="text-xs font-bold px-2 py-0.5 rounded"
                          style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                        >
                          #{result.rank}
                        </span>
                        <span className="text-sm font-bold" style={{ color: isPassed ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                          {result.ticker}
                        </span>
                        <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                          {result.tf}
                        </span>
                        <span
                          className="text-xs font-medium"
                          style={{ color: result.direction === 'LONG' ? 'var(--green, #4CAF50)' : 'var(--red, #f44336)' }}
                        >
                          {result.direction}
                        </span>
                      </div>

                      {/* Trigger + confluence */}
                      <p className="text-xs mb-0.5" style={{ color: 'var(--text-secondary)' }}>
                        Entry: {result.trigger}
                        {result.confluence.length > 0 && (
                          <> + {result.confluence.map((c) => (
                            <span
                              key={c}
                              className="inline-block mx-0.5 px-1.5 py-0 rounded font-mono text-[10px]"
                              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
                            >
                              {c}
                            </span>
                          ))}</>
                        )}
                      </p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Exit: {result.exitTrigger} &middot; Stop: {result.stopDesc} &middot; Target: {result.targetDesc}
                      </p>
                      <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                        Range: {result.dateRange}
                      </p>

                      {/* KPI row */}
                      <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mt-3">
                        {[
                          { label: 'Trades', value: result.trades.toString() },
                          { label: 'WR', value: `${result.winRate.toFixed(1)}%` },
                          { label: 'PF', value: result.pf.toFixed(2) },
                          { label: 'Avg R', value: `${result.avgR >= 0 ? '+' : ''}${result.avgR.toFixed(2)}` },
                          { label: 'Total R', value: `${result.totalR >= 0 ? '+' : ''}${result.totalR.toFixed(1)}` },
                          { label: 'Daily R', value: `${result.dailyR >= 0 ? '+' : ''}${result.dailyR.toFixed(2)}` },
                          { label: 'R\u00B2', value: result.rSquared.toFixed(2) },
                          { label: 'Max DD', value: `${result.maxDD.toFixed(1)}R` },
                        ].map((kpi) => (
                          <div key={kpi.label} className="text-center">
                            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                            <p className="text-xs font-bold" style={{ color: 'var(--text-primary)' }}>{kpi.value}</p>
                          </div>
                        ))}
                      </div>

                      {/* Expandable details */}
                      <button
                        onClick={() => setExpandedResult(isExpanded ? null : idx)}
                        className="text-[10px] mt-2 underline"
                        style={{ color: 'var(--accent)' }}
                      >
                        {isExpanded ? 'Hide Details' : 'View Details'}
                      </button>
                      {isExpanded && (
                        <div
                          className="mt-2 p-3 rounded-lg text-xs space-y-1"
                          style={{ background: 'var(--bg-input)' }}
                        >
                          <p style={{ color: 'var(--text-secondary)' }}>
                            <strong>Avg Win:</strong> {(result.avgR * 1.5).toFixed(2)}R &middot;
                            <strong> Avg Loss:</strong> {(-Math.abs(result.avgR) * 0.8).toFixed(2)}R
                          </p>
                          <p style={{ color: 'var(--text-secondary)' }}>
                            <strong>Win Count:</strong> {Math.round(result.trades * result.winRate / 100)} &middot;
                            <strong> Loss Count:</strong> {result.trades - Math.round(result.trades * result.winRate / 100)}
                          </p>
                          <p style={{ color: 'var(--text-secondary)' }}>
                            <strong>Expectancy:</strong> {result.avgR.toFixed(3)}R per trade
                          </p>
                          <p style={{ color: 'var(--text-secondary)' }}>
                            <strong>Recovery Factor:</strong> {Math.abs(result.totalR / (result.maxDD || -1)).toFixed(2)}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Right: equity curve + actions */}
                    <div className="w-48 flex-shrink-0 flex flex-col justify-between">
                      <MiniEquityCurve data={result.equityCurve} height={80} />
                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => saveResult(idx)}
                          disabled={isSaved}
                          className="flex-1 px-2 py-1.5 rounded-lg text-xs font-medium transition-opacity hover:opacity-80 disabled:opacity-40"
                          style={{
                            background: isSaved ? 'var(--bg-input)' : 'var(--accent)',
                            color: isSaved ? 'var(--text-muted)' : 'white',
                          }}
                        >
                          {isSaved ? 'Saved' : 'Save'}
                        </button>
                        <button
                          onClick={() => passResult(idx)}
                          className="flex-1 px-2 py-1.5 rounded-lg text-xs font-medium"
                          style={{
                            background: isPassed ? 'var(--accent-muted)' : 'var(--bg-input)',
                            border: '1px solid var(--border)',
                            color: isPassed ? 'var(--accent)' : 'var(--text-muted)',
                          }}
                        >
                          {isPassed ? 'Un-pass' : 'Pass'}
                        </button>
                      </div>
                    </div>
                  </div>
                </Card>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Empty state */}
      {results.length === 0 && !isAnalyzing && (
        <Card className="mt-4">
          <div className="text-center py-8">
            <p className="text-sm mb-1" style={{ color: 'var(--text-secondary)' }}>
              No results yet
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Configure your search parameters and click Analyze to discover strategies.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}
