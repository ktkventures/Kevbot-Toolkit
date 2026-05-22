'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Modal from '@/components/Modal';
import { useRunMassSearch, useMassProgress, useMassResult } from '@/hooks/queries/useMassBuilder';
import { useCreateStrategy } from '@/hooks/mutations/useStrategyMutations';
import { useConfluenceGroups, useConfluenceTemplates, useConfluenceTriggers, useStopLossPacks, useTakeProfitPacks, useGeneralPacks, useGeneralTemplates, useTimeExitPacks } from '@/hooks/queries/usePacks';
import { useSettings } from '@/hooks/queries/useSettings';
import { useStrategyModels } from '@/hooks/queries/useStrategies';

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
  // Out-of-Sample gate results — present only when the search ran with OOS on.
  hasOos?: boolean;
  oosWinRate?: number;
  oosPf?: number;
  oosDailyR?: number;
  oosTrades?: number;
  oosSigmaStatus?: 'green' | 'amber' | 'red' | 'unknown';
  oosSigmaZ?: number | null;
  _raw?: any;  // Raw result from backend for save flow
}

/* ========================================================================
   Constants
   ======================================================================== */

const TICKER_PRESETS: Record<string, string[]> = {
  'Mag 7': ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA'],
  'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA'],
  'Crypto': ['BTC/USD', 'ETH/USD'],
};

// All possible timeframes in order — filtered by user settings at runtime
const ALL_TIMEFRAMES = [
  '5Sec', '10Sec', '15Sec', '30Sec',
  '1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min',
  '1Hour', '2Hour', '4Hour', '1Day',
];
const TF_LABELS: Record<string, string> = {
  '5Sec': '5s', '10Sec': '10s', '15Sec': '15s', '30Sec': '30s',
  '1Min': '1m', '2Min': '2m', '3Min': '3m', '5Min': '5m', '10Min': '10m',
  '15Min': '15m', '30Min': '30m', '1Hour': '1h', '2Hour': '2h', '4Hour': '4h',
  '1Day': '1d',
};
const TIMEFRAMES_FALLBACK = ['1Min', '5Min', '15Min', '1Hour', '4Hour'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended Hours', '24/7'];
const SORT_OPTIONS = ['Daily R', 'Win Rate', 'Profit Factor', 'R-Squared', 'Total R', 'Trades'];

const EXEC_TYPES = ['[C]', '[L]', '[LC]', '[CC]'] as const;
const EXEC_BADGE_COLOR = '#2196F3';
const FIDELITY_BADGE_COLOR = '#26C6DA';

// OOS robustness chip — how the held-out window compares to the
// in-sample projection (docs/Spec_OOS_Test_Periods.md §9.4).
function oosSigmaChip(status?: string): { label: string; bg: string; color: string } {
  switch (status) {
    case 'green':
      return { label: 'OOS holding', bg: 'rgba(76,175,80,0.18)', color: '#7fd081' };
    case 'amber':
      return { label: 'OOS soft', bg: 'rgba(255,193,7,0.18)', color: '#ffc107' };
    case 'red':
      return { label: 'OOS degraded', bg: 'rgba(244,67,54,0.18)', color: '#ef5350' };
    default:
      return { label: 'OOS n/a', bg: 'rgba(120,120,120,0.18)', color: 'var(--text-muted)' };
  }
}

// Out-of-sample result strip on a Mass Builder result card.
function OosStrip({ result }: { result: MassResult }) {
  const chip = oosSigmaChip(result.oosSigmaStatus);
  const dr = result.oosDailyR ?? 0;
  return (
    <div className="flex items-center gap-2 mb-2 px-2 py-1.5 rounded flex-wrap" style={{ background: 'var(--bg-input)' }}>
      <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded" style={{ background: chip.bg, color: chip.color }}>
        {chip.label}
      </span>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
        Out-of-sample:&nbsp;
        WR {(result.oosWinRate ?? 0).toFixed(1)}% &middot;{' '}
        PF {(result.oosPf ?? 0).toFixed(2)} &middot;{' '}
        Daily R {dr >= 0 ? '+' : ''}{dr.toFixed(2)} &middot;{' '}
        {result.oosTrades ?? 0} trades
        {result.oosSigmaZ != null && <> &middot; z {result.oosSigmaZ.toFixed(2)}</>}
      </span>
    </div>
  );
}

interface TriggerDef {
  id: string;   // confluence trigger ID from backend (e.g. "swing_123_default_bull_c2")
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
  state: string;
  direction: 'BULL' | 'BEAR' | 'NEUTRAL';
  description: string;
  groupId: string;
}

interface GenConfDef {
  id: string;
  display: string;
  pack: string;
  variation: string;
  state: string;
  description: string;
  packId: string;
}

/** Format the bottom-bar (fine-grain) label based on current phase. */
function formatInnerLabel(phase: string | undefined | null, step: number, total: number): string {
  const pct = Math.round((step / total) * 100);
  const stepFmt = step.toLocaleString();
  const totalFmt = total.toLocaleString();
  switch (phase) {
    case 'load':
      return `${stepFmt} / ${totalFmt} bars loaded (${pct}%)`;
    case 'backtest':
      return `Bar ${stepFmt} / ${totalFmt} (${pct}%)`;
    case 'confluence':
      return `Combo ${stepFmt} / ${totalFmt} (${pct}%)`;
    case 'prep':
      return `Preparing… (${pct}%)`;
    default:
      return `${stepFmt} / ${totalFmt}`;
  }
}

/** Short, human-readable phase badge text. */
function phaseBadge(phase: string | undefined | null): string {
  switch (phase) {
    case 'load': return 'Loading';
    case 'prep': return 'Preparing';
    case 'backtest': return 'Backtesting';
    case 'confluence': return 'Searching confluences';
    case 'save': return 'Saving';
    default: return '';
  }
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

// algo_model split (2026-05-08): per-model dropdown for Mass Builder spec.
function MassBuilderModelDropdown({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Record<string, { label: string; available: boolean; description: string }>;
  onChange: (v: string) => void;
}) {
  const entries = Object.entries(options);
  const currentDesc = options[value]?.description;
  return (
    <div>
      <label className="text-[10px] font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-1.5 rounded-lg text-sm"
        style={{
          background: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
        }}
      >
        {entries.map(([key, opt]) => (
          <option key={key} value={key} disabled={!opt.available && key !== value}>
            {opt.label}{opt.available ? '' : ' (coming soon)'}
          </option>
        ))}
      </select>
      {currentDesc && (
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)', lineHeight: 1.4 }}>
          {currentDesc.length > 120 ? currentDesc.slice(0, 117) + '…' : currentDesc}
        </p>
      )}
    </div>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function MassBuilderPage() {
  // ---- API hooks (MUST come before any early returns) ----
  const runMassSearch = useRunMassSearch();
  const createStrategy = useCreateStrategy();
  const { data: apiEntryTriggers } = useConfluenceTriggers('LONG');
  const { data: apiExitTriggers } = useConfluenceTriggers('EXIT');
  const { data: apiConfluenceGroups } = useConfluenceGroups();
  const { data: apiConfluenceTemplates } = useConfluenceTemplates();
  const { data: apiStopPacks } = useStopLossPacks();
  const { data: apiTargetPacks } = useTakeProfitPacks();
  const { data: apiGeneralPacks } = useGeneralPacks();
  const { data: apiGeneralTemplates } = useGeneralTemplates();
  const { data: apiTimeExitPacks } = useTimeExitPacks();
  const { data: userSettings } = useSettings();

  // ---- Derive available timeframes from user settings (Timeframes pack) ----
  const AVAILABLE_TFS: string[] = useMemo(() => {
    const enabledTfs = userSettings?.enabled_timeframes;
    if (!enabledTfs || Object.keys(enabledTfs).length === 0) return TIMEFRAMES_FALLBACK;
    const primary = ALL_TIMEFRAMES.filter((tf) => enabledTfs[tf]?.primaryEnabled);
    return primary.length > 0 ? primary : TIMEFRAMES_FALLBACK;
  }, [userSettings]);

  // ---- Derive trigger/pack data from API ----
  const ENTRY_TRIGGER_DEFS: TriggerDef[] = useMemo(() => {
    if (!apiEntryTriggers) return [];
    return Object.entries(apiEntryTriggers).map(([id, name]) => {
      const parts = id.split('_');
      const pack = parts[0] || 'unknown';
      return {
        id,
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
        id,
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
      const tmpl = apiConfluenceTemplates?.[g.base_template] as any | undefined;
      // `outputs` is an array of state codes; `output_descriptions` is state→text;
      // `output_directions` is state→"BULL"|"BEAR"|"NEUTRAL" (added by backend).
      const outputs: string[] = Array.isArray(tmpl?.outputs) ? tmpl.outputs : [];
      const outDesc: Record<string, string> = tmpl?.output_descriptions || {};
      const outDir: Record<string, string> = tmpl?.output_directions || {};
      if (outputs.length === 0) {
        // Template missing or has no outputs (e.g. user-custom template with no
        // classification). Fall back to direction-summary checkboxes so the
        // pack is still selectable.
        for (const dir of ['BULL', 'BEAR'] as const) {
          confs.push({
            id: `_TF_-${g.id.toUpperCase()}-${dir}-PB`,
            display: dir === 'BULL' ? 'Bull' : 'Bear',
            pack: g.base_template,
            variation: g.version || 'Default',
            fidelity: '[PB]',
            state: dir,
            direction: dir,
            description: `All ${dir.toLowerCase()} states`,
            groupId: g.id,
          });
        }
        continue;
      }
      for (const state of outputs) {
        const direction = (outDir[state] || 'NEUTRAL') as 'BULL' | 'BEAR' | 'NEUTRAL';
        confs.push({
          id: `_TF_-${g.id.toUpperCase()}-${state}-PB`,
          display: state,
          pack: g.base_template,
          variation: g.version || 'Default',
          fidelity: '[PB]',
          state,
          direction,
          description: outDesc[state] || state,
          groupId: g.id,
        });
      }
    }
    return confs;
  }, [apiConfluenceGroups, apiConfluenceTemplates]);

  const GENERAL_CONFLUENCES: GenConfDef[] = useMemo(() => {
    if (!apiGeneralPacks) return [];
    const confs: GenConfDef[] = [];
    for (const p of apiGeneralPacks as any[]) {
      const tmpl = apiGeneralTemplates?.[p.base_template] as any | undefined;
      const outputs: string[] = Array.isArray(tmpl?.outputs) ? tmpl.outputs : [];
      const outDesc: Record<string, string> = tmpl?.output_descriptions || {};
      if (outputs.length === 0) {
        confs.push({
          id: `GEN-${p.id.toUpperCase()}-IN`,
          display: 'Active',
          pack: p.base_template,
          variation: p.version || 'Default',
          state: 'IN',
          description: 'Pack condition active',
          packId: p.id,
        });
        continue;
      }
      for (const state of outputs) {
        confs.push({
          id: `GEN-${p.id.toUpperCase()}-${state}`,
          display: state,
          pack: p.base_template,
          variation: p.version || 'Default',
          state,
          description: outDesc[state] || state,
          packId: p.id,
        });
      }
    }
    return confs;
  }, [apiGeneralPacks, apiGeneralTemplates]);

  /** Stop packs — sourced from dedicated /stop-loss endpoint which filters by
   * template has_stop flag (avoids misclassifying combined stop+target packs). */
  const STOP_PACKS = useMemo(() => {
    if (!apiStopPacks) return [];
    return (apiStopPacks as any[]).map((p: any) => ({
      id: p.id,
      name: p.stop_summary || Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version || 'Default',
      pack: p.base_template,
      variation: p.version || 'Default',
    }));
  }, [apiStopPacks]);

  /** Target packs — sourced from dedicated /take-profit endpoint. */
  const TARGET_PACKS = useMemo(() => {
    if (!apiTargetPacks) return [];
    return (apiTargetPacks as any[]).map((p: any) => ({
      id: p.id,
      name: p.target_summary || Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version || 'Default',
      pack: p.base_template,
      variation: p.version || 'Default',
    }));
  }, [apiTargetPacks]);

  /** Time exit packs derived from API */
  const TIME_EXIT_PACKS = useMemo(() => {
    if (!apiTimeExitPacks) return [];
    return apiTimeExitPacks.map((p: any) => ({
      id: p.id,
      name: p.exit_summary || p.version || 'Default',
      pack: p.base_template,
      variation: p.version || 'Default',
    }));
  }, [apiTimeExitPacks]);

  // Config state
  const [searchName, setSearchName] = useState(`Search ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} ${new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`);
  const [triggerHiFi, setTriggerHiFi] = useState(false);
  const [confluenceHiFi, setConfluenceHiFi] = useState(false);
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
  const [selectedStopPacks, setSelectedStopPacks] = useState<string[]>([]);
  const [selectedTargetPacks, setSelectedTargetPacks] = useState<string[]>([]);
  const [selectedTimeExitPacks, setSelectedTimeExitPacks] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState('Daily R');
  const [minTrades, setMinTrades] = useState(10);
  const [minWR, setMinWR] = useState(0);
  const [minPF, setMinPF] = useState(0);
  const [minDailyR, setMinDailyR] = useState(0);
  const [maxResults, setMaxResults] = useState(500);

  // Out-of-Sample gate (docs/Spec_OOS_Test_Periods.md §9). When enabled,
  // every combo is also backtested through today and split at
  // `oosInSampleEnd`: ranked on the in-sample window, gated on the OOS
  // window (raw thresholds + optional sigma band).
  const [oosEnabled, setOosEnabled] = useState(false);
  const [oosInSampleEnd, setOosInSampleEnd] = useState('');  // YYYY-MM-DD
  const [oosMinTrades, setOosMinTrades] = useState(0);
  const [oosMinWR, setOosMinWR] = useState(0);
  const [oosMinPF, setOosMinPF] = useState(0);
  const [oosMinDailyR, setOosMinDailyR] = useState(0);
  const [oosSigmaEnabled, setOosSigmaEnabled] = useState(true);
  const [oosSigmaN, setOosSigmaN] = useState(2);
  const [activeSearchId, setActiveSearchId] = useState<string | number | null>(null);

  // Model selection (algo_model split — 2026-05-08). Spawned strategies
  // inherit these values via mass_builder.py:build_strategy_config.
  const { data: modelsResp } = useStrategyModels();
  const [backtestModel, setBacktestModel] = useState<string>('rest_hifi');
  const [algoModel, setAlgoModel] = useState<string>('cache_locked');
  const [liveModel, setLiveModel] = useState<string>('ws_agg_locked');

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

  // ---- Progress polling & result fetching ----
  const { data: progressData } = useMassProgress(activeSearchId);
  const { data: searchResult } = useMassResult(
    progressData?.status === 'completed' ? activeSearchId : null
  );

  // ---- Edit / Clone-from-search param handling ----
  // Mass Results links here with either ?edit={id} (view saved search with
  // its results) or ?clone={id} (just pre-fill config as a new search).
  // Both fetch the same saved-search payload. Edit additionally populates
  // the results panel below.
  const searchParams = useSearchParams();
  const editId = searchParams?.get('edit') || null;
  const cloneId = searchParams?.get('clone') || null;
  const sourceId = editId || cloneId;
  const isEditMode = !!editId;
  const { data: cloneSource } = useMassResult(sourceId);
  const [cloneApplied, setCloneApplied] = useState(false);
  useEffect(() => {
    if (!cloneSource || cloneApplied) return;
    // Apply config (same logic for both edit and clone)
    const cfg = (cloneSource as any).config_data || (cloneSource as any).config || cloneSource || {};
    if (Array.isArray(cfg.tickers)) setSelectedTickers(cfg.tickers);
    if (Array.isArray(cfg.timeframes)) setSelectedTFs(cfg.timeframes);
    if (Array.isArray(cfg.directions)) setSelectedDirections(cfg.directions);
    if (Array.isArray(cfg.entry_triggers)) setSelectedEntries(cfg.entry_triggers);
    if (Array.isArray(cfg.exit_triggers)) setSelectedExits(cfg.exit_triggers);
    if (Array.isArray(cfg.tf_confluences)) setSelectedTfConf(cfg.tf_confluences);
    if (Array.isArray(cfg.general_confluences)) setSelectedGenConf(cfg.general_confluences);
    if (typeof cfg.tf_confluence_depth === 'number') setTfConfDepth(cfg.tf_confluence_depth);
    if (typeof cfg.general_confluence_depth === 'number') setGenConfDepth(cfg.general_confluence_depth);
    if (Array.isArray(cfg.stop_packs)) setSelectedStopPacks(cfg.stop_packs);
    if (Array.isArray(cfg.target_packs)) setSelectedTargetPacks(cfg.target_packs);
    if (Array.isArray(cfg.time_exit_packs)) setSelectedTimeExitPacks(cfg.time_exit_packs);
    if (typeof cfg.session === 'string') setSession(cfg.session);
    if (cfg.date_range?.days) setLookbackDays(cfg.date_range.days);
    if (cfg.name) {
      setSearchName(isEditMode ? cfg.name : `Clone of ${cfg.name}`);
    }
    setCloneApplied(true);
  }, [cloneSource, cloneApplied, isEditMode]);

  // When in edit mode, feed the saved results into the existing results
  // panel by setting activeSearchId. The rawResults useEffect downstream
  // maps them into MassResult[] just like a completed live search.
  useEffect(() => {
    if (isEditMode && editId && !activeSearchId) {
      setActiveSearchId(editId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isEditMode, editId]);

  // Confluence sub-progress state
  const [confProgress, setConfProgress] = useState(0);
  const [confLabel, setConfLabel] = useState('');
  const [phase, setPhase] = useState<string | null>(null);
  const [phaseDetail, setPhaseDetail] = useState<string>('');

  // React to progress changes
  useEffect(() => {
    if (!progressData) return;
    const { status, progress: p, total, current_label } = progressData;
    const innerStep = (progressData as any).inner_step;
    const innerTotal = (progressData as any).inner_total;
    const pd = (progressData as any);
    if (status === 'running') {
      setProgress(total > 0 ? p / total : 0);
      setPhase(pd.phase || null);
      setPhaseDetail(pd.phase_detail || current_label || '');
      setProgressLabel(current_label || `Processing ${p}/${total}...`);
      // Bottom bar — fine-grain inside current phase
      if (innerStep != null && innerTotal > 0) {
        setConfProgress(innerStep / innerTotal);
        setConfLabel(formatInnerLabel(pd.phase, innerStep, innerTotal));
      } else {
        setConfProgress(0);
        setConfLabel('');
      }
    } else if (status === 'completed') {
      setIsAnalyzing(false);
      setProgress(1);
      setProgressLabel('Search complete');
      setPhase('save');
      setPhaseDetail('Complete');
      setConfProgress(0);
      setConfLabel('');
    } else if (status === 'failed' || status === 'cancelled') {
      setIsAnalyzing(false);
      setProgress(0);
      setProgressLabel(`Search ${status}`);
      setPhase(null);
      setPhaseDetail('');
      setConfProgress(0);
      setConfLabel('');
    } else if (status === 'orphaned') {
      // Worker thread died (usually on API deploy). Tier 2 roadmap will
      // add checkpoint/resume; for now the user can Edit the saved row
      // and re-run from scratch.
      setIsAnalyzing(false);
      setProgress(0);
      setProgressLabel('Search interrupted — the API restarted mid-run. Re-run from Mass Results.');
      setPhase(null);
      setPhaseDetail('');
      setConfProgress(0);
      setConfLabel('');
    }
  }, [progressData]);

  // Populate results when search completes — check both progressData
  // (in-memory, has stored_trades) and searchResult (DB fallback).
  const rawResults = progressData?.results ?? searchResult?.results;
  useEffect(() => {
    if (!rawResults || !Array.isArray(rawResults) || rawResults.length === 0) return;
    const stopMethod = (c: any) => {
      const m = c?.stop_config?.method;
      if (m === 'atr') return `ATR ${c.stop_config.atr_mult ?? 1.5}x`;
      return m || 'ATR 1.5x';
    };
    const targetMethod = (c: any) => {
      const m = c?.target_config?.method;
      if (m === 'risk_reward') return `${c.target_config.rr_ratio ?? 2}R`;
      return m || 'Signal exit';
    };
    const mapped: MassResult[] = rawResults.map((r: any, i: number) => ({
      rank: i + 1,
      ticker: r.config?.symbol || '--',
      direction: (r.config?.direction || 'LONG') as 'LONG' | 'SHORT',
      tf: r.config?.timeframe || '1Min',
      trigger: r.config?.entry_trigger_name || r.config?.entry_trigger_confluence_id || '--',
      exitTrigger: (r.config?.exit_trigger_names || []).join(', ') || 'Signal',
      confluence: r.confluence_str ? r.confluence_str.split(' + ') : [],
      stopDesc: stopMethod(r.config),
      targetDesc: targetMethod(r.config),
      dateRange: `${r.config?.data_days || 90}d`,
      winRate: r.kpis?.win_rate ?? 0,
      pf: r.kpis?.profit_factor ?? 0,
      dailyR: r.kpis?.daily_r ?? 0,
      avgR: r.kpis?.avg_r ?? 0,
      totalR: r.kpis?.total_r ?? 0,
      trades: r.kpis?.total_trades ?? 0,
      maxDD: r.kpis?.max_r_drawdown ?? 0,
      rSquared: r.kpis?.r_squared ?? 0,
      equityCurve: r.equity_curve || [],
      status: (r.status || 'active') as 'active' | 'saved' | 'passed',
      hasOos: !!r.oos_kpis,
      oosWinRate: r.oos_kpis?.win_rate,
      oosPf: r.oos_kpis?.profit_factor,
      oosDailyR: r.oos_kpis?.daily_r,
      oosTrades: r.oos_kpis?.total_trades,
      oosSigmaStatus: r.oos_sigma?.status,
      oosSigmaZ: r.oos_sigma?.z ?? null,
      _raw: r,
    }));
    setResults(mapped);
    setActiveSearchId(null);
  }, [rawResults]);

  // Post-analysis filter state
  const [filterWR, setFilterWR] = useState(0);
  const [filterPF, setFilterPF] = useState(0);
  const [filterTrades, setFilterTrades] = useState(0);
  const [filterSort, setFilterSort] = useState('Daily R');
  const [showPassed, setShowPassed] = useState(false);

  const CONFIG_TABS = ['Tickers', 'Timeframes', 'Direction', 'Entry', 'Exit', 'TF Confluence', 'General', 'Stop Loss', 'Take Profit', 'Time Exit'];

  // Combination estimates. Calibrated 2026-04-12 against two real runs
  // (90d/1Min/1 BT and 180d/{10Sec,5Min,1Min}/12 BTs) — fit both within ~5%:
  //
  //   trigger_bt_ms = BASE_MS_PER_BT + PER_BAR_US × bars
  //   confluence_bt = flat CONFLUENCE_MS per combo
  //   overhead_ms   = BASE_MS_PER_GROUP × nGroups + PER_BAR_US_OVERHEAD × total_bars
  //
  // `bars` for a single BT = bars_per_day[tf] × lookback_days.
  // A "group" = one (ticker, TF) data-load unit.
  const EST_BASE_MS_PER_BT = 7043;             // fixed startup per trigger backtest
  const EST_PER_BAR_US = 343;                  // per-bar engine processing cost
  const EST_CONFLUENCE_MS = 0.9;               // confluence subset filter + KPI compute
  const EST_OVERHEAD_BASE_MS_PER_GROUP = 8965; // data load + initial indicator setup
  const EST_OVERHEAD_PER_BAR_US = 397;         // shadow-TF prep / indicator warmup per bar
  // Approx bars per trading day by timeframe. Market hours (6.5h) for stocks.
  const BARS_PER_DAY: Record<string, number> = {
    '5Sec': 4680, '10Sec': 2340, '15Sec': 1560, '30Sec': 780,
    '1Min': 390, '2Min': 195, '3Min': 130, '5Min': 78,
    '10Min': 39, '15Min': 26, '30Min': 13,
    '1Hour': 7, '2Hour': 3.5, '4Hour': 2, '1Day': 1,
  };
  // Each TF confluence selection (e.g. "EMA_STACK BULL") expands into multiple
  // actual interpreter-states in the record pool (SML, SLM, MSL), each of
  // which may appear at every TF (primary + secondaries). Adjust the effective
  // pool size accordingly so the confluence combo count isn't undercounted.
  const AVG_STATES_PER_TF_SELECTION = 2;
  const estimate = useMemo(() => {
    const nTickers = selectedTickers.length;
    const nTFs = selectedTFs.length;
    const nDirs = selectedDirections.length;
    const nEntries = selectedEntries.length;
    const nExits = Math.max(selectedExits.length, 1);
    const nStops = Math.max(selectedStopPacks.length, 1);
    const nTargets = Math.max(selectedTargetPacks.length, 1);
    const nTimeExits = Math.max(selectedTimeExitPacks.length, 1);
    // BTs per TF group = tickers × dirs × entries × exits × packs
    const perTfGroup = nTickers * nDirs * nEntries * nExits * nStops * nTargets * nTimeExits;
    const triggerBacktests = perTfGroup * nTFs;

    // Confluence pool = effective number of unique labels in the record pool.
    // TF selections fan out across interpreter states AND across selected TFs.
    // General selections are 1:1 with the labels they produce.
    const effectiveTfLabels = selectedTfConf.length * AVG_STATES_PER_TF_SELECTION * Math.max(nTFs, 1);
    const effectivePoolSize = effectiveTfLabels + selectedGenConf.length;
    let comboCount = 0;
    for (let d = 1; d <= tfConfDepth && d <= effectivePoolSize; d++) {
      // Safe binomial coefficient: C(n, d) = ∏(n-i)/(i+1) for i=0..d-1.
      // Avoids factorial() overflow when nLabels is large (e.g. 50+),
      // which was producing NaN in the preview.
      let c = 1;
      for (let i = 0; i < d; i++) {
        c = c * (effectivePoolSize - i) / (i + 1);
      }
      comboCount += Math.round(c);
    }
    const confluenceBacktests = triggerBacktests * comboCount;

    // Per-TF trigger BT time scales with bars processed at that TF.
    let triggerMs = 0;
    let avgTriggerMs = 0;
    let totalBarsLoaded = 0;
    if (nTFs > 0) {
      for (const tf of selectedTFs) {
        const bars = (BARS_PER_DAY[tf] || 390) * lookbackDays;
        const msPerBt = EST_BASE_MS_PER_BT + EST_PER_BAR_US * bars / 1000;
        triggerMs += msPerBt * perTfGroup;
        avgTriggerMs += msPerBt;
        totalBarsLoaded += bars * Math.max(nTickers, 1);
      }
      avgTriggerMs = avgTriggerMs / nTFs;
    } else {
      avgTriggerMs = EST_BASE_MS_PER_BT + EST_PER_BAR_US * 390 * lookbackDays / 1000;
    }
    const triggerSeconds = triggerMs / 1000;
    const confluenceSeconds = (confluenceBacktests * EST_CONFLUENCE_MS) / 1000;
    // Overhead = per-group fixed + per-bar prep across all data loaded.
    const nGroups = Math.max(nTickers, 1) * Math.max(nTFs, 1);
    const overheadMs = EST_OVERHEAD_BASE_MS_PER_GROUP * nGroups + EST_OVERHEAD_PER_BAR_US * totalBarsLoaded / 1000;
    const overheadSeconds = overheadMs / 1000;
    const totalSeconds = triggerSeconds + confluenceSeconds + overheadSeconds;
    return {
      nTickers, nTFs, nDirs, nEntries, nExits,
      triggerBacktests, confluenceBacktests,
      triggerSeconds, confluenceSeconds, overheadSeconds, totalSeconds,
      avgTriggerMs,
    };
  }, [selectedTickers, selectedTFs, selectedDirections, selectedEntries, selectedExits, selectedTfConf, tfConfDepth, selectedGenConf, selectedStopPacks, selectedTargetPacks, selectedTimeExitPacks, lookbackDays]);

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

    // When re-running from edit mode with an unchanged name, auto-suffix so
    // the new search is visually distinct from the original on Mass Results.
    // Pattern: "Original Name (v2)", "Original Name (v3)", etc.
    let submittedName = searchName;
    if (isEditMode) {
      const existingSuffix = submittedName.match(/\(v(\d+)\)$/);
      if (existingSuffix) {
        const n = parseInt(existingSuffix[1], 10) + 1;
        submittedName = submittedName.replace(/\(v\d+\)$/, `(v${n})`);
      } else {
        submittedName = `${submittedName} (v2)`;
      }
      // Update the visible name field too so the user sees the change
      setSearchName(submittedName);
    }

    const config = {
      name: submittedName,
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
      stop_packs: selectedStopPacks.length > 0 ? selectedStopPacks : undefined,
      target_packs: selectedTargetPacks.length > 0 ? selectedTargetPacks : undefined,
      time_exit_packs: selectedTimeExitPacks.length > 0 ? selectedTimeExitPacks : undefined,
      session,
      date_range: { mode: 'days', days: lookbackDays },
      sort_by: sortBy,
      required_performance: {
        min_trades: minTrades,
        min_win_rate: minWR > 0 ? minWR : undefined,
        min_profit_factor: minPF > 0 ? minPF : undefined,
        min_daily_r: minDailyR > 0 ? minDailyR : undefined,
      },
      // Out-of-Sample gate (docs/Spec_OOS_Test_Periods.md §9).
      oos: (oosEnabled && oosInSampleEnd) ? {
        enabled: true,
        in_sample_end: oosInSampleEnd,
        required_performance: {
          min_trades: oosMinTrades > 0 ? oosMinTrades : undefined,
          min_win_rate: oosMinWR > 0 ? oosMinWR : undefined,
          min_profit_factor: oosMinPF > 0 ? oosMinPF : undefined,
          min_daily_r: oosMinDailyR > 0 ? oosMinDailyR : undefined,
        },
        sigma: { enabled: oosSigmaEnabled, n: oosSigmaN },
      } : undefined,
      max_results: maxResults,
      // Model fields (algo_model split — 2026-05-08). Spawned strategies
      // inherit these via mass_builder.py:build_strategy_config.
      backtest_model: backtestModel,
      algo_model: algoModel,
      live_model: liveModel,
    };

    runMassSearch.mutate(config, {
      onSuccess: (data) => {
        setActiveSearchId(data.search_id);
        setProgressLabel('Search started — polling for progress...');
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
    // Sort — in-sample KPIs, or the OOS-side KPIs (OOS-* options).
    const sortKeyMap: Record<string, keyof MassResult> = {
      'Daily R': 'dailyR', 'Win Rate': 'winRate', 'Profit Factor': 'pf',
      'R-Squared': 'rSquared', 'Total R': 'totalR', 'Trades': 'trades',
      'OOS Daily R': 'oosDailyR', 'OOS Win Rate': 'oosWinRate',
      'OOS Profit Factor': 'oosPf', 'OOS Trades': 'oosTrades',
    };
    const sk = sortKeyMap[filterSort] || 'dailyR';
    // OOS fields are undefined on non-OOS results — sort those last.
    filtered.sort((a, b) =>
      ((b[sk] as number) ?? -Infinity) - ((a[sk] as number) ?? -Infinity));
    return filtered;
  }, [results, filterSort, filterWR, filterPF, filterTrades, showPassed]);

  // Sort dropdown gains OOS-* options only when the results carry OOS data.
  const sortOptionsAvailable = useMemo(() => (
    results.some((r) => r.hasOos)
      ? [...SORT_OPTIONS, 'OOS Daily R', 'OOS Win Rate', 'OOS Profit Factor', 'OOS Trades']
      : SORT_OPTIONS
  ), [results]);

  function toggleItem<T>(arr: T[], item: T, setter: (v: T[]) => void) {
    if (arr.includes(item)) {
      setter(arr.filter((x) => x !== item));
    } else {
      setter([...arr, item]);
    }
  }

  // Identify results by `_raw` reference (the original backend object), not
  // by array index. The filteredResults array is a sorted/filtered view, so
  // the index in the view does not correspond to the index in `results`.
  // Using reference equality ensures the action applies to the clicked row
  // regardless of current sort order.
  function saveResult(r: MassResult) {
    if (!r?._raw) {
      setResults((prev) => prev.map((x) => (x._raw === r._raw ? { ...x, status: 'saved' as const } : x)));
      return;
    }
    const raw = r._raw;
    const cfg = raw.config || {};
    createStrategy.mutate({
      name: `${r.ticker} ${r.direction} ${r.tf} Mass #${r.rank}`,
      symbol: cfg.symbol,
      direction: cfg.direction,
      timeframe: cfg.timeframe,
      trading_session: cfg.trading_session || 'RTH',
      data_days: cfg.data_days || 90,
      entry_trigger_confluence_id: cfg.entry_trigger_confluence_id,
      exit_trigger_confluence_ids: cfg.exit_trigger_confluence_ids || [],
      confluence: cfg.confluence || [],
      general_confluences: cfg.general_confluences || [],
      stop_config: cfg.stop_config,
      target_config: cfg.target_config,
      time_exit_config: cfg.time_exit_config,
      kpis: raw.kpis || {},
      stored_trades: raw.stored_trades || [],
      equity_curve_data: {
        cumulative_r: raw.equity_curve || [],
      },
    });
    setResults((prev) => prev.map((x) => (x._raw === r._raw ? { ...x, status: 'saved' as const } : x)));
  }

  function passResult(r: MassResult) {
    setResults((prev) =>
      prev.map((x) =>
        x._raw === r._raw
          ? { ...x, status: x.status === 'passed' ? 'active' as const : 'passed' as const }
          : x
      )
    );
  }

  return (
    <div>
      <PageHeader title="Mass Strategy Builder" subtitle="Bulk strategy discovery and optimization engine" />

      {/* Preview-KPIs warning — Mass Builder runs a base backtest and
          post-filters by confluence, which can produce KPIs that differ
          from engine-accurate numbers (the base's position-state blocking
          hides trades the live engine would take under gating). Always
          verify strategy KPIs via "Update All Data" after saving. */}
      <div
        className="mb-4 px-3 py-2 rounded flex items-start gap-2"
        style={{ background: 'rgba(255, 152, 0, 0.12)', border: '1px solid var(--orange)' }}
      >
        <span style={{ color: 'var(--orange)', fontSize: '1rem', lineHeight: 1 }}>⚠</span>
        <p className="text-xs" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <span className="font-semibold" style={{ color: 'var(--orange)' }}>Preview KPIs.</span>{' '}
          Mass Builder ranks strategies using a post-filter on a single base backtest.
          Real engine performance can differ — the base run&apos;s position-state blocking hides
          trades the live engine takes under confluence gating. After saving any strategy
          from here, click <strong>Update All Data</strong> on the Strategy Detail page for
          engine-accurate KPIs.
        </p>
      </div>

      {/* Edit-mode banner — shown when loaded via ?edit={id} from Mass Results */}
      {isEditMode && cloneApplied && (
        <div className="mb-4 px-3 py-2 rounded flex items-center justify-between" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)' }}>
          <p className="text-xs" style={{ color: 'var(--accent)' }}>
            <span className="font-semibold">Editing saved search.</span>{' '}
            Results from the original run are shown below. Running Analyze will create a new search — the original is preserved.
          </p>
        </div>
      )}

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
        {/* Fidelity toggles */}
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div
              className="relative w-9 h-5 rounded-full transition-colors"
              style={{ background: triggerHiFi ? 'var(--accent)' : 'var(--bg-input)', border: '1px solid var(--border)' }}
              onClick={() => setTriggerHiFi(!triggerHiFi)}
            >
              <div
                className="absolute top-0.5 w-3.5 h-3.5 rounded-full transition-all"
                style={{ background: triggerHiFi ? '#fff' : 'var(--text-muted)', left: triggerHiFi ? '18px' : '2px' }}
              />
            </div>
            <span className="text-xs font-medium whitespace-nowrap" style={{ color: triggerHiFi ? 'var(--accent)' : 'var(--text-muted)' }}>
              Trigger HiFi
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <div
              className="relative w-9 h-5 rounded-full transition-colors"
              style={{ background: confluenceHiFi ? 'var(--accent)' : 'var(--bg-input)', border: '1px solid var(--border)' }}
              onClick={() => setConfluenceHiFi(!confluenceHiFi)}
            >
              <div
                className="absolute top-0.5 w-3.5 h-3.5 rounded-full transition-all"
                style={{ background: confluenceHiFi ? '#fff' : 'var(--text-muted)', left: confluenceHiFi ? '18px' : '2px' }}
              />
            </div>
            <span className="text-xs font-medium whitespace-nowrap" style={{ color: confluenceHiFi ? 'var(--accent)' : 'var(--text-muted)' }}>
              Confluence HiFi
            </span>
          </label>
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
              {tab === 'Timeframes' && selectedTFs.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedTFs.length}
                </span>
              )}
              {tab === 'Direction' && selectedDirections.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedDirections.length}
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
              {tab === 'TF Confluence' && selectedTfConf.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedTfConf.length}
                </span>
              )}
              {tab === 'General' && selectedGenConf.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedGenConf.length}
                </span>
              )}
              {tab === 'Stop Loss' && selectedStopPacks.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedStopPacks.length}
                </span>
              )}
              {tab === 'Take Profit' && selectedTargetPacks.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedTargetPacks.length}
                </span>
              )}
              {tab === 'Time Exit' && selectedTimeExitPacks.length > 0 && (
                <span className="ml-1 text-[10px] px-1 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {selectedTimeExitPacks.length}
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
                    label={TF_LABELS[tf] || tf}
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

              {/* Models — algo_model split (2026-05-08) */}
              <div className="mt-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Models</p>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Backtest = KPI baseline · Algo = live accountability · Live = engine reality
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <MassBuilderModelDropdown
                    label="Backtest Model"
                    value={backtestModel}
                    options={modelsResp?.backtest_models || {}}
                    onChange={setBacktestModel}
                  />
                  <MassBuilderModelDropdown
                    label="Algo Model"
                    value={algoModel}
                    options={(modelsResp as any)?.algo_models || modelsResp?.backtest_models || {}}
                    onChange={setAlgoModel}
                  />
                  <MassBuilderModelDropdown
                    label="Live Model"
                    value={liveModel}
                    options={modelsResp?.live_models || {}}
                    onChange={setLiveModel}
                  />
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
                          {triggers.map((t) => (
                            <div key={t.id} className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-xs min-w-[100px]" style={{ color: 'var(--text-secondary)' }}>{t.name}</span>
                              {t.execTypes.map((exec) => {
                                const isChecked = selectedEntries.includes(t.id);
                                return (
                                  <label key={exec} className="flex items-center gap-0.5 cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={isChecked}
                                      onChange={() => toggleItem(selectedEntries, t.id, setSelectedEntries)}
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
                          {triggers.map((t) => (
                            <div key={t.id} className="flex items-center gap-1.5 flex-wrap">
                              <span className="text-xs min-w-[100px]" style={{ color: 'var(--text-secondary)' }}>{t.name}</span>
                              {t.execTypes.map((exec) => {
                                const isChecked = selectedExits.includes(t.id);
                                return (
                                  <label key={exec} className="flex items-center gap-0.5 cursor-pointer">
                                    <input type="checkbox" checked={isChecked} onChange={() => toggleItem(selectedExits, t.id, setSelectedExits)} className="w-3 h-3 rounded" style={{ accentColor: EXEC_BADGE_COLOR }} />
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
                  <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Select TF confluence states for auto-search</p>
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
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 mb-4" style={{ maxHeight: 500, overflowY: 'auto' }}>
                {Object.entries(
                  TF_CONFLUENCES.reduce<Record<string, typeof TF_CONFLUENCES>>((acc, c) => {
                    (acc[c.groupId] = acc[c.groupId] || []).push(c);
                    return acc;
                  }, {})
                ).map(([groupId, confs]) => {
                  const first = confs[0];
                  const byDirection: Record<'BULL' | 'BEAR' | 'NEUTRAL', typeof confs> = {
                    BULL: confs.filter((c) => c.direction === 'BULL'),
                    BEAR: confs.filter((c) => c.direction === 'BEAR'),
                    NEUTRAL: confs.filter((c) => c.direction === 'NEUTRAL'),
                  };
                  const selectedCount = confs.filter((c) => selectedTfConf.includes(c.id)).length;
                  const DIR_META: Record<'BULL' | 'BEAR' | 'NEUTRAL', { label: string; color: string; bg: string }> = {
                    BULL: { label: 'Bull', color: 'var(--green)', bg: 'var(--green-muted)' },
                    BEAR: { label: 'Bear', color: 'var(--red)', bg: 'var(--red-muted)' },
                    NEUTRAL: { label: 'Neutral', color: 'var(--text-muted)', bg: 'var(--bg-elevated)' },
                  };
                  const toggleDirection = (dir: 'BULL' | 'BEAR' | 'NEUTRAL') => {
                    const ids = byDirection[dir].map((c) => c.id);
                    const allSelected = ids.every((id) => selectedTfConf.includes(id));
                    if (allSelected) {
                      setSelectedTfConf(selectedTfConf.filter((id) => !ids.includes(id)));
                    } else {
                      setSelectedTfConf(Array.from(new Set([...selectedTfConf, ...ids])));
                    }
                  };
                  return (
                    <div key={groupId} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                      <div className="flex items-center justify-between mb-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
                        <div className="min-w-0 flex items-center gap-1.5">
                          <div className="min-w-0">
                            <p className="text-xs font-semibold truncate" style={{ color: 'var(--text-primary)' }}>{first.pack}</p>
                            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{first.variation}</p>
                          </div>
                          <span className="text-[9px] font-mono font-semibold px-1 py-0.5 rounded flex-shrink-0" style={{ color: FIDELITY_BADGE_COLOR, background: FIDELITY_BADGE_COLOR + '20' }}>PB</span>
                        </div>
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: selectedCount > 0 ? 'var(--accent)' : 'var(--text-muted)', background: selectedCount > 0 ? 'var(--accent-muted)' : 'transparent' }}>{selectedCount}/{confs.length}</span>
                      </div>
                      {(['BULL', 'BEAR', 'NEUTRAL'] as const).map((dir) => {
                        if (byDirection[dir].length === 0) return null;
                        const meta = DIR_META[dir];
                        const dirSelected = byDirection[dir].filter((c) => selectedTfConf.includes(c.id)).length;
                        return (
                          <div key={dir} className="mb-2 last:mb-0">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: meta.color }}>{meta.label}</span>
                              <button className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: meta.bg, color: meta.color, border: 'none', cursor: 'pointer' }}
                                onClick={() => toggleDirection(dir)}>
                                {dirSelected === byDirection[dir].length ? 'clear' : 'all'}
                              </button>
                            </div>
                            <div className="space-y-0.5">
                              {byDirection[dir].map((c) => (
                                <label key={c.id} title={c.description} className="flex items-center gap-1.5 cursor-pointer py-0.5 px-1 rounded hover:bg-black/10">
                                  <input type="checkbox" checked={selectedTfConf.includes(c.id)} onChange={() => toggleItem(selectedTfConf, c.id, setSelectedTfConf)} className="w-3 h-3 rounded flex-shrink-0" style={{ accentColor: meta.color }} />
                                  <span className="text-[11px] font-mono flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{c.state}</span>
                                  <span className="text-[9px] truncate" style={{ color: 'var(--text-muted)' }}>{c.description}</span>
                                </label>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  );
                })}
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
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 mb-4" style={{ maxHeight: 500, overflowY: 'auto' }}>
                {Object.entries(
                  GENERAL_CONFLUENCES.reduce<Record<string, typeof GENERAL_CONFLUENCES>>((acc, c) => {
                    (acc[c.packId] = acc[c.packId] || []).push(c);
                    return acc;
                  }, {})
                ).map(([packId, confs]) => {
                  const first = confs[0];
                  const packIds = confs.map((c) => c.id);
                  const packSelected = confs.filter((c) => selectedGenConf.includes(c.id)).length;
                  const allSelected = packSelected === confs.length;
                  const togglePack = () => {
                    if (allSelected) {
                      setSelectedGenConf(selectedGenConf.filter((id) => !packIds.includes(id)));
                    } else {
                      setSelectedGenConf(Array.from(new Set([...selectedGenConf, ...packIds])));
                    }
                  };
                  return (
                    <div key={packId} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                      <div className="flex items-center justify-between mb-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
                        <div className="min-w-0">
                          <p className="text-xs font-semibold truncate" style={{ color: 'var(--text-primary)' }}>{first.pack}</p>
                          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{first.variation}</p>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <button className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: 'none', cursor: 'pointer' }} onClick={togglePack}>
                            {allSelected ? 'clear' : 'all'}
                          </button>
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: packSelected > 0 ? 'var(--accent)' : 'var(--text-muted)', background: packSelected > 0 ? 'var(--accent-muted)' : 'transparent' }}>{packSelected}/{confs.length}</span>
                        </div>
                      </div>
                      <div className="space-y-0.5">
                        {confs.map((c) => (
                          <label key={c.id} title={c.description} className="flex items-center gap-1.5 cursor-pointer py-0.5 px-1 rounded hover:bg-black/10">
                            <input type="checkbox" checked={selectedGenConf.includes(c.id)} onChange={() => toggleItem(selectedGenConf, c.id, setSelectedGenConf)} className="w-3 h-3 rounded flex-shrink-0" style={{ accentColor: 'var(--accent)' }} />
                            <span className="text-[11px] font-mono flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>{c.state}</span>
                            <span className="text-[9px] truncate" style={{ color: 'var(--text-muted)' }}>{c.description}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  );
                })}
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
                                <input type="checkbox" className="rounded" style={{ accentColor: 'var(--red)' }}
                                  checked={selectedStopPacks.includes(p.id)}
                                  onChange={() => setSelectedStopPacks((prev) => prev.includes(p.id) ? prev.filter((x) => x !== p.id) : [...prev, p.id])}
                                />
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
                                <input type="checkbox" className="rounded" style={{ accentColor: 'var(--green)' }}
                                  checked={selectedTargetPacks.includes(p.id)}
                                  onChange={() => setSelectedTargetPacks((prev) => prev.includes(p.id) ? prev.filter((x) => x !== p.id) : [...prev, p.id])}
                                />
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

          {activeConfigTab === 'Time Exit' && (
            <div>
              <p className="text-sm font-medium mb-2" style={{ color: 'var(--text-primary)' }}>
                Time Exit Packs <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({TIME_EXIT_PACKS.length} available)</span>
              </p>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {TIME_EXIT_PACKS.map((p) => (
                  <label key={p.id} className="flex items-center gap-2 cursor-pointer py-1 px-3 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <input type="checkbox" className="rounded" style={{ accentColor: 'var(--orange)' }}
                      checked={selectedTimeExitPacks.includes(p.id)}
                      onChange={() => setSelectedTimeExitPacks((prev) => prev.includes(p.id) ? prev.filter((x) => x !== p.id) : [...prev, p.id])}
                    />
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{p.name}</span>
                  </label>
                ))}
                {TIME_EXIT_PACKS.length === 0 && (
                  <p className="text-xs italic col-span-3" style={{ color: 'var(--text-muted)' }}>No time exit packs configured. <a href="/confluence-packs/time-exit" className="underline" style={{ color: 'var(--accent)' }}>Create one</a></p>
                )}
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* ====== Estimation + Required Performance + Analyze ====== */}
      <div className="grid grid-cols-12 gap-4 mb-5">
        {/* Preview — compact */}
        <Card className="col-span-3">
          <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Preview</p>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <div>
              <div className="text-base font-bold" style={{ color: 'var(--text-primary)' }}>
                {estimate.triggerBacktests.toLocaleString()}
              </div>
              <p className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }} title="Full engine backtests. Scales with bars (TF × lookback).">Trigger Backtests</p>
              <p className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }}>Avg ~{(estimate.avgTriggerMs / 1000).toFixed(1)}s ea</p>
              <p className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }}>Subtotal {formatTime(estimate.triggerSeconds)}</p>
            </div>
            <div>
              <div className="text-base font-bold" style={{ color: 'var(--accent)' }}>
                {estimate.confluenceBacktests.toLocaleString()}
              </div>
              <p className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }} title="Confluence subset filters per Trigger Backtest trade list.">Confluence Backtests</p>
              <p className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }}>Avg ~{EST_CONFLUENCE_MS}ms ea</p>
              <p className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }}>Subtotal {formatTime(estimate.confluenceSeconds)}</p>
            </div>
          </div>
          <p className="text-[9px] leading-tight mb-1" style={{ color: 'var(--text-muted)' }}>Overhead (data load + warmup): {formatTime(estimate.overheadSeconds)}</p>
          <div className="pt-2 mt-1 border-t" style={{ borderColor: 'var(--border)' }}>
            <p className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>Total Est. Runtime: {formatTime(estimate.totalSeconds)}</p>
          </div>
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

      {/* ====== Out-of-Sample Validation (OOS gate) ====== */}
      <Card className="mb-5">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Out-of-Sample Validation</p>
            <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>overfit gate</span>
          </div>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
            <input type="checkbox" checked={oosEnabled} onChange={(e) => setOosEnabled(e.target.checked)} />
            Enabled
          </label>
        </div>
        {!oosEnabled ? (
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            Off — combos are ranked and gated on the full window only. Enable to hold out a
            recent slice the optimizer never sees, then gate every combo on how it performs there.
          </p>
        ) : (
          <div className="space-y-3">
            <div className="grid grid-cols-4 gap-x-3 gap-y-2">
              <div>
                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>In-Sample Ends</label>
                <input type="date" value={oosInSampleEnd} onChange={(e) => setOosInSampleEnd(e.target.value)}
                  className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
              </div>
              <div className="col-span-3 flex items-end">
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {oosInSampleEnd
                    ? `Ranked on the in-sample window (lookback start → ${oosInSampleEnd}). Gated on the out-of-sample window (${oosInSampleEnd} → today) — held out, never optimized on.`
                    : 'Pick the date the in-sample window ends. Everything after it, through today, becomes the held-out OOS window.'}
                </p>
              </div>
            </div>
            <div>
              <p className="text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                OOS Required Performance — a combo must clear these on the held-out window too (0 = no constraint)
              </p>
              <div className="grid grid-cols-4 gap-x-3 gap-y-2">
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>OOS Min Trades</label>
                  <input type="number" min={0} max={500} value={oosMinTrades} onChange={(e) => setOosMinTrades(parseInt(e.target.value, 10) || 0)}
                    className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                </div>
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>OOS Min WR %</label>
                  <input type="number" min={0} max={100} step={5} value={oosMinWR} onChange={(e) => setOosMinWR(parseFloat(e.target.value) || 0)}
                    className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                </div>
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>OOS Min PF</label>
                  <input type="number" min={0} max={20} step={0.25} value={oosMinPF} onChange={(e) => setOosMinPF(parseFloat(e.target.value) || 0)}
                    className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                </div>
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>OOS Min Daily R</label>
                  <input type="number" min={0} max={10} step={0.1} value={oosMinDailyR} onChange={(e) => setOosMinDailyR(parseFloat(e.target.value) || 0)}
                    className="w-full px-2 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4 flex-wrap">
              <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
                <input type="checkbox" checked={oosSigmaEnabled} onChange={(e) => setOosSigmaEnabled(e.target.checked)} />
                Sigma-band gate
              </label>
              <div className="flex items-center gap-1.5">
                <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>N&sigma;</label>
                <input type="number" min={1} max={3} step={0.5} value={oosSigmaN} disabled={!oosSigmaEnabled}
                  onChange={(e) => setOosSigmaN(parseFloat(e.target.value) || 2)}
                  className="w-16 px-2 py-1 rounded text-xs disabled:opacity-40" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
              </div>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                Rejects combos whose OOS cumulative R falls more than N&sigma; below the in-sample projection.
              </p>
            </div>
          </div>
        )}
      </Card>

      {/* ====== Progress Bar ====== */}
      {(isAnalyzing || progress > 0) && (
        <Card className="mb-5">
          {/* Top bar: phase + subphase detail + overall fill */}
          <div className="flex items-center justify-between mb-1 gap-2">
            <div className="flex items-center gap-2 min-w-0 flex-1">
              {phase && (
                <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded flex-shrink-0"
                      style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  {phaseBadge(phase)}
                </span>
              )}
              <p className="text-xs font-medium truncate" style={{ color: 'var(--text-secondary)' }}>
                {phaseDetail || progressLabel}
              </p>
            </div>
            <span className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
              {Math.round(progress * 100)}%
            </span>
          </div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${progress * 100}%`,
                background: progress >= 1 ? 'var(--green, #4CAF50)' : 'var(--accent)',
              }}
            />
          </div>
          {/* Bottom bar: fine-grain inside current phase (bars / combos) */}
          {confLabel && (
            <div className="mt-3">
              <div className="flex items-center justify-between mb-1">
                <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {confLabel}
                </p>
                <span className="text-[11px] font-mono" style={{ color: 'var(--text-muted)' }}>
                  {Math.round(confProgress * 100)}%
                </span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                <div
                  className="h-full rounded-full transition-all duration-150"
                  style={{
                    width: `${confProgress * 100}%`,
                    background: 'var(--text-muted)',
                  }}
                />
              </div>
            </div>
          )}
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
                {sortOptionsAvailable.map((o) => <option key={o}>{o}</option>)}
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
                <div key={result._raw?.id || `${result.ticker}-${result.tf}-${result.direction}-${result.rank}`} style={{ opacity: isPassed ? 0.5 : 1 }}>
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

                  {/* OOS strip — held-out window result + robustness chip */}
                  {result.hasOos && <OosStrip result={result} />}

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
                      onClick={() => saveResult(result)}
                      disabled={isSaved}
                      className="px-3 py-1 rounded text-xs font-medium transition-opacity hover:opacity-80 disabled:opacity-40"
                      style={{ background: isSaved ? 'var(--bg-input)' : 'var(--accent)', color: isSaved ? 'var(--text-muted)' : 'white', cursor: 'pointer' }}
                    >
                      {isSaved ? 'Saved' : 'Save Strategy'}
                    </button>
                    <button
                      onClick={() => passResult(result)}
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
