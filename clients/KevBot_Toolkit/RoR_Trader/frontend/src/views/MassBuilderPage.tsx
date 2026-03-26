'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Modal from '@/components/Modal';
import { useRunMassSearch } from '@/hooks/queries/useMassBuilder';
import { useConfluenceGroups, useConfluenceTriggers, useRiskManagementPacks, useGeneralPacks } from '@/hooks/queries/usePacks';

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
   Constants
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
const FIDELITY_BADGE_COLOR = '#26C6DA';

interface TriggerDef {
  name: string;
  pack: string;
  variation: string;
  execTypes: string[]; // which exec types are available for this trigger
}

interface TfConfDef {
  id: string;
  display: string;
  pack: string;
  variation: string;
  fidelity: string;
}

interface GenConfDef {
  id: string;
  display: string;
  pack: string;
  variation: string;
}

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

export default function MassBuilderPage() {
  // ---- API hooks (MUST come before any early returns) ----
  const runMassSearch = useRunMassSearch();
  const { data: apiEntryTriggers } = useConfluenceTriggers('LONG');
  const { data: apiExitTriggers } = useConfluenceTriggers('EXIT');
  const { data: apiConfluenceGroups } = useConfluenceGroups();
  const { data: apiRmPacks } = useRiskManagementPacks();
  const { data: apiGeneralPacks } = useGeneralPacks();

  // ---- Derive trigger/pack data from API ----
  const ENTRY_TRIGGER_DEFS: TriggerDef[] = useMemo(() => {
    if (!apiEntryTriggers) return [];
    return Object.entries(apiEntryTriggers).map(([id, name]) => {
      const parts = id.split('_');
      const pack = parts[0] || 'unknown';
      return {
        name: String(name),
        pack,
        variation: 'Default',
        execTypes: ['[C]'], // Base exec type; full exec support comes from templates
      };
    });
  }, [apiEntryTriggers]);

  const EXIT_TRIGGER_DEFS: TriggerDef[] = useMemo(() => {
    if (!apiExitTriggers) return [];
    return Object.entries(apiExitTriggers).map(([id, name]) => {
      const parts = id.split('_');
      const pack = parts[0] || 'unknown';
      return {
        name: String(name),
        pack,
        variation: 'Default',
        execTypes: ['[C]'],
      };
    });
  }, [apiExitTriggers]);

  const TF_CONFLUENCES: TfConfDef[] = useMemo(() => {
    if (!apiConfluenceGroups) return [];
    const confs: TfConfDef[] = [];
    for (const g of apiConfluenceGroups) {
      // Create PB (previous bar) confluence entries per group
      confs.push({
        id: `_TF_-${g.id.toUpperCase()}-BULL-PB`,
        display: `${g.base_template} Bull`,
        pack: g.base_template,
        variation: g.version || 'Default',
        fidelity: '[PB]',
      });
      confs.push({
        id: `_TF_-${g.id.toUpperCase()}-BEAR-PB`,
        display: `${g.base_template} Bear`,
        pack: g.base_template,
        variation: g.version || 'Default',
        fidelity: '[PB]',
      });
    }
    return confs;
  }, [apiConfluenceGroups]);

  const GENERAL_CONFLUENCES: GenConfDef[] = useMemo(() => {
    if (!apiGeneralPacks) return [];
    return apiGeneralPacks.map((p: any) => ({
      id: `GEN-${p.id.toUpperCase()}-IN`,
      display: `${p.base_template} (${p.version || 'Default'})`,
      pack: p.base_template,
      variation: p.version || 'Default',
    }));
  }, [apiGeneralPacks]);

  /** Stop packs derived from API risk-management packs */
  const STOP_PACKS = useMemo(() => {
    if (!apiRmPacks) return [];
    return apiRmPacks
      .filter((p: any) => !p.base_template?.includes('target') && p.base_template !== 'rr_ratio')
      .map((p: any) => ({
        id: p.id,
        name: Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version || 'Default',
        pack: p.base_template,
        variation: p.version || 'Default',
      }));
  }, [apiRmPacks]);

  /** Target packs derived from API risk-management packs */
  const TARGET_PACKS = useMemo(() => {
    if (!apiRmPacks) return [];
    return apiRmPacks
      .filter((p: any) => p.base_template === 'rr_ratio' || p.base_template?.includes('target'))
      .map((p: any) => ({
        id: p.id,
        name: Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version || 'Default',
        pack: p.base_template,
        variation: p.version || 'Default',
      }));
  }, [apiRmPacks]);

  // Config state
  const [searchName, setSearchName] = useState(`Search ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tickerInput, setTickerInput] = useState('');
  const [selectedTFs, setSelectedTFs] = useState<string[]>([]);
  const [selectedDirections, setSelectedDirections] = useState<('LONG' | 'SHORT')[]>(['LONG']);
  const [selectedEntries, setSelectedEntries] = useState<string[]>([]);
  const [selectedExits, setSelectedExits] = useState<string[]>([]);
  const [exitDepth, setExitDepth] = useState(1);
  const [selectedTfConf, setSelectedTfConf] = useState<string[]>([]);
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

  // Layout & display state
  const [resultColumns, setResultColumns] = useState(2);
  const [eqShowHWM, setEqShowHWM] = useState(false);
  const [chartHeight, setChartHeight] = useState(96);
  const [eqXAxis, setEqXAxis] = useState<'trade' | 'time'>('trade');

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
    setProgressLabel('Submitting search...');

    const config = {
      name: searchName,
      tickers: selectedTickers,
      timeframes: selectedTFs,
      directions: selectedDirections,
      entry_triggers: selectedEntries,
      exit_triggers: selectedExits,
      exit_depth: exitDepth,
      tf_confluences: selectedTfConf,
      tf_confluence_depth: tfConfDepth,
      general_confluences: selectedGenConf,
      general_confluence_depth: genConfDepth,
      session,
      lookback_days: lookbackDays,
      sort_by: sortBy,
      min_trades: minTrades,
      min_win_rate: minWR > 0 ? minWR : null,
      min_profit_factor: minPF > 0 ? minPF : null,
      max_results: maxResults,
    };

    runMassSearch.mutate(config, {
      onSuccess: (data) => {
        setIsAnalyzing(false);
        setProgress(1);
        setProgressLabel(`Search submitted — ID: ${data.search_id}`);
      },
      onError: (err: any) => {
        setIsAnalyzing(false);
        setProgress(0);
        setProgressLabel(`Error: ${err.message || 'Search failed'}`);
      },
    });
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
                                <input type="checkbox" className="rounded" style={{ accentColor: 'var(--red)' }} />
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
                                <input type="checkbox" className="rounded" style={{ accentColor: 'var(--green)' }} />
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

          {/* Viewing preferences */}
          <div className="flex items-center gap-4 mb-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
            <div className="flex items-center gap-1.5">
              <span>Columns:</span>
              {[1, 2, 3].map((n) => (
                <button key={n} onClick={() => setResultColumns(n)} className="px-2 py-1 rounded font-medium"
                  style={{ background: resultColumns === n ? 'var(--accent-muted)' : 'var(--bg-input)', color: resultColumns === n ? 'var(--accent)' : 'var(--text-muted)', border: resultColumns === n ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                  {n}
                </button>
              ))}
            </div>
            <div className="w-px h-4" style={{ background: 'var(--border)' }} />
            <div className="flex items-center gap-1.5">
              <span>Chart:</span>
              {[{ v: 48, l: 'S' }, { v: 64, l: 'M' }, { v: 96, l: 'L' }, { v: 140, l: 'XL' }].map((o) => (
                <button key={o.v} onClick={() => setChartHeight(o.v)} className="px-2 py-1 rounded font-medium"
                  style={{ background: chartHeight === o.v ? 'var(--accent-muted)' : 'var(--bg-input)', color: chartHeight === o.v ? 'var(--accent)' : 'var(--text-muted)', border: chartHeight === o.v ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                  {o.l}
                </button>
              ))}
            </div>
            <div className="w-px h-4" style={{ background: 'var(--border)' }} />
            <div className="flex items-center gap-1.5">
              <span>X-axis:</span>
              {(['trade', 'time'] as const).map((mode) => (
                <button key={mode} onClick={() => setEqXAxis(mode)} className="px-2 py-1 rounded font-medium"
                  style={{ background: eqXAxis === mode ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: eqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                  {mode === 'trade' ? 'Trade #' : 'Date'}
                </button>
              ))}
            </div>
            <div className="w-px h-4" style={{ background: 'var(--border)' }} />
            <div className="flex items-center gap-1.5">
              <span>HWM:</span>
              {(['On', 'Off'] as const).map((v) => (
                <button key={v} onClick={() => setEqShowHWM(v === 'On')} className="px-2 py-1 rounded font-medium"
                  style={{ background: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent-muted)' : 'var(--bg-input)', color: (eqShowHWM ? 'On' : 'Off') === v ? 'var(--accent)' : 'var(--text-muted)', border: (eqShowHWM ? 'On' : 'Off') === v ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}>
                  {v}
                </button>
              ))}
            </div>
          </div>

          {/* Result cards — My Strategies style */}
          <div className={`grid gap-4 ${resultColumns === 1 ? 'grid-cols-1' : resultColumns === 2 ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3'}`}>
            {filteredResults.map((result, idx) => {
              const isPassed = result.status === 'passed';
              const isSaved = result.status === 'saved';

              return (
                <div key={idx} style={{ opacity: isPassed ? 0.5 : 1 }}>
                <Card>
                  {/* Row 1: Rank + Name + Ticker + TF + Direction */}
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                      #{result.rank}
                    </span>
                    <h3 className="font-semibold text-sm">{searchName} — {result.ticker} {result.direction}</h3>
                    <span className="flex-1" />
                    <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{result.tf}</span>
                  </div>

                  {/* Meta line */}
                  <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                    {result.ticker} {result.direction} &middot; {result.tf} &middot; {result.dateRange}
                  </p>

                  {/* Equity curve */}
                  <div className="rounded-lg mb-2 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                    <MiniEquityCurve data={result.equityCurve} height={chartHeight} />
                  </div>

                  {/* KPIs — two rows of 4 */}
                  <div className="grid grid-cols-4 gap-2 mb-2">
                    {[
                      { label: 'Win Rate', value: `${result.winRate.toFixed(1)}%` },
                      { label: 'Profit Factor', value: result.pf.toFixed(2) },
                      { label: 'Daily R', value: `${result.dailyR >= 0 ? '+' : ''}${result.dailyR.toFixed(2)}` },
                      { label: 'R\u00B2', value: result.rSquared.toFixed(2) },
                      { label: 'Avg R', value: `${result.avgR >= 0 ? '+' : ''}${result.avgR.toFixed(2)}` },
                      { label: 'Total R', value: `${result.totalR >= 0 ? '+' : ''}${result.totalR.toFixed(0)}` },
                      { label: 'Max DD', value: `${result.maxDD.toFixed(1)}R` },
                      { label: 'Trades', value: String(result.trades) },
                    ].map((kpi) => (
                      <div key={kpi.label}>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                        <p className="text-sm font-medium">{kpi.value}</p>
                      </div>
                    ))}
                  </div>

                  {/* Strategy variables — pack-aware display */}
                  <div className="space-y-1 mb-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>entry:</span>
                      <span className="text-[10px] font-mono font-medium px-1 py-0.5 rounded-full" style={{ color: '#2196F3', background: '#2196F320', fontSize: '10px' }}>[C]</span>
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>{result.trigger}</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>exit:</span>
                      <span className="text-[10px] font-mono font-medium px-1 py-0.5 rounded-full" style={{ color: '#2196F3', background: '#2196F320', fontSize: '10px' }}>[C]</span>
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>{result.exitTrigger}</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>stop:</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--red)', background: 'var(--red)18' }}>{result.stopDesc}</span>
                      <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>target:</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green)18' }}>{result.targetDesc}</span>
                    </div>
                    {result.confluence.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>confluence:</span>
                        <span className="text-[10px] font-mono px-1 py-0.5 rounded-full font-medium" style={{ color: '#26C6DA', background: '#26C6DA20', fontSize: '9px' }}>[PB]</span>
                        {result.confluence.map((c) => (
                          <span key={c} className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ color: 'var(--accent)', background: 'var(--accent)15' }}>{c}</span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Action row */}
                  <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
                    <button
                      onClick={() => saveResult(idx)}
                      disabled={isSaved}
                      className="px-3 py-1 rounded text-xs font-medium transition-opacity hover:opacity-80 disabled:opacity-40"
                      style={{ background: isSaved ? 'var(--bg-input)' : 'var(--accent)', color: isSaved ? 'var(--text-muted)' : 'white', cursor: 'pointer' }}
                    >
                      {isSaved ? 'Saved' : 'Save Strategy'}
                    </button>
                    <button
                      onClick={() => passResult(idx)}
                      className="px-3 py-1 rounded text-xs"
                      style={{ background: isPassed ? 'var(--accent-muted)' : 'var(--bg-input)', border: '1px solid var(--border)', color: isPassed ? 'var(--accent)' : 'var(--text-muted)', cursor: 'pointer' }}
                    >
                      {isPassed ? 'Un-pass' : 'Pass'}
                    </button>
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
