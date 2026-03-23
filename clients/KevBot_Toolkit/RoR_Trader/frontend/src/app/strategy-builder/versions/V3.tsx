'use client';

import { useState, useMemo, useCallback } from 'react';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Modal from '@/components/Modal';

// ---------------------------------------------------------------------------
// V3 Meta: Streamlined Strategy Builder
// ---------------------------------------------------------------------------
// What changed from V2 and why:
//
// REMOVED:
// - Tabs for charts (Equity/Price) -- now stacked vertically, always visible
// - Separate "Analysis Tabs" sidebar (Entry/Exit/TF/General/Stop/Target drill-down)
//   -> Moved to Strategy Detail V3's "Trades" tab where it belongs (post-save analysis)
// - Confluence scoring panel (weights/depth limit) -- rare feature, hidden in V3
// - Optimizable Variables summary card -- redundant with the config bar + exit summary
// - "Analyzer" drill-down with mock entry/exit comparison tables
// - Markov Motor analysis section -> moved to Strategy Detail V3
// - Separate "General Conditions" modal -- merged into one unified confluence picker
// - Asset type selector -- auto-detected from ticker (BTC/USD -> crypto)
// - Lookback mode selector -- simplified to just "days" with a single input
// - Strategy name input in top bar -- auto-generated, editable inline before save
//
// KEPT (reorganized):
// - Compact config bar: ticker, direction, timeframe, session, lookback
// - Trigger search with clickable results
// - Exit config (collapsed by default, summary line visible)
// - Confluence as chip-based selector
// - KPIs as compact horizontal strip
// - Equity curve + price chart stacked
// - Trade history table (always visible)
// - Run Backtest / Save Strategy actions
//
// RATIONALE:
// A trader building a strategy does 90% of the work in: pick ticker, pick trigger,
// set stop/target, run backtest, check KPIs. The V2 analysis tabs are useful AFTER
// the strategy is saved -- that's what Strategy Detail is for. V3 removes the
// build-time clutter and makes the feedback loop (config -> backtest -> results)
// as tight as possible.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TriggerDef {
  id: string;
  name: string;
  execType: 'C' | 'L0' | 'L1' | 'HM' | 'HL';
  pack: string;
}

interface ConfluenceCondition {
  id: string;
  label: string;
  pack: string;
}

interface TradeRow {
  id: number;
  entryTime: string;
  exitTime: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  rMultiple: number;
  execType: string;
  exitReason: string;
}

// ---------------------------------------------------------------------------
// Mock Data (reused from V2, trimmed)
// ---------------------------------------------------------------------------

const TIMEFRAMES = ['1Min', '5Min', '15Min', '1H', '4H', '1D'];
const DIRECTIONS = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];
const STOP_METHODS = ['Swing', 'ATR', 'Fixed $', 'Pct %'];
const TARGET_METHODS = ['None', 'R:R', 'ATR', 'Fixed $', 'Pct %'];

const MOCK_TRIGGERS: TriggerDef[] = [
  { id: 'ema-price-cross-up', name: 'EMA Price Cross Up', execType: 'C', pack: 'EMA Stack' },
  { id: 'ema-price-cross-down', name: 'EMA Price Cross Down', execType: 'C', pack: 'EMA Stack' },
  { id: 'ema-fan-open-bull', name: 'EMA Fan Open Bull', execType: 'C', pack: 'EMA Stack' },
  { id: 'ema-fan-open-bear', name: 'EMA Fan Open Bear', execType: 'C', pack: 'EMA Stack' },
  { id: 'macd-cross-bull', name: 'MACD Cross Bull', execType: 'C', pack: 'MACD' },
  { id: 'macd-cross-bear', name: 'MACD Cross Bear', execType: 'C', pack: 'MACD' },
  { id: 'macd-zero-cross-up', name: 'MACD Zero Cross Up', execType: 'C', pack: 'MACD' },
  { id: 'vwap-cross-up', name: 'VWAP Cross Up', execType: 'L0', pack: 'VWAP' },
  { id: 'vwap-cross-down', name: 'VWAP Cross Down', execType: 'L0', pack: 'VWAP' },
  { id: 'rsi-oversold-exit', name: 'RSI Oversold Exit', execType: 'C', pack: 'RSI' },
  { id: 'rsi-overbought-exit', name: 'RSI Overbought Exit', execType: 'C', pack: 'RSI' },
  { id: 'ut-bot-buy', name: 'UT Bot Buy', execType: 'HM', pack: 'UT Bot' },
  { id: 'ut-bot-sell', name: 'UT Bot Sell', execType: 'HM', pack: 'UT Bot' },
  { id: 'supertrend-flip-bull', name: 'Supertrend Flip Bull', execType: 'L1', pack: 'Supertrend' },
  { id: 'supertrend-flip-bear', name: 'Supertrend Flip Bear', execType: 'L1', pack: 'Supertrend' },
  { id: 'bar-count-exit-4', name: '4-Bar Exit', execType: 'C', pack: 'Bar Count' },
  { id: 'keltner-bounce-long', name: 'Keltner Bounce Long', execType: 'HL', pack: 'Keltner' },
  { id: 'keltner-bounce-short', name: 'Keltner Bounce Short', execType: 'HL', pack: 'Keltner' },
];

const ALL_CONFLUENCE: ConfluenceCondition[] = [
  { id: '5M-EMA_STACK-BULL', label: '5M EMA Stack Bullish', pack: 'EMA Stack' },
  { id: '5M-EMA_STACK-SML', label: '5M EMA Stack S>M>L', pack: 'EMA Stack' },
  { id: '15M-EMA_STACK-BULL', label: '15M EMA Stack Bullish', pack: 'EMA Stack' },
  { id: '1H-EMA_STACK-BULL', label: '1H EMA Stack Bullish', pack: 'EMA Stack' },
  { id: '1D-MACD_LINE-BULL', label: '1D MACD Line Bullish', pack: 'MACD' },
  { id: '1D-MACD_LINE-BEAR', label: '1D MACD Line Bearish', pack: 'MACD' },
  { id: '5M-MACD_LINE-BULL', label: '5M MACD Line Bullish', pack: 'MACD' },
  { id: '1H-VWAP_POS-ABOVE', label: '1H Above VWAP', pack: 'VWAP' },
  { id: '1H-VWAP_POS-BELOW', label: '1H Below VWAP', pack: 'VWAP' },
  { id: '5M-RSI_ZONE-NEUTRAL', label: '5M RSI Neutral Zone', pack: 'RSI' },
  { id: '1D-RSI_ZONE-BULL', label: '1D RSI Bullish Zone', pack: 'RSI' },
  { id: '15M-SUPERTREND-BULL', label: '15M Supertrend Bullish', pack: 'Supertrend' },
  { id: '1H-SUPERTREND-BULL', label: '1H Supertrend Bullish', pack: 'Supertrend' },
  // General packs (merged into one list)
  { id: 'GEN-TOD-NY_MORNING', label: 'NY Morning (9:30-11:30)', pack: 'Time / Session' },
  { id: 'GEN-TOD-NY_AFTERNOON', label: 'NY Afternoon (13:00-15:30)', pack: 'Time / Session' },
  { id: 'GEN-DOW-MON_TUE_WED', label: 'Mon/Tue/Wed', pack: 'Time / Session' },
  { id: 'GEN-DOW-TUE_THU', label: 'Tue/Thu', pack: 'Time / Session' },
  { id: 'GEN-SESSION-RTH', label: 'Regular Trading Hours', pack: 'Time / Session' },
];

const MOCK_KPIS = {
  winRate: 58.3,
  profitFactor: 2.14,
  dailyR: 0.47,
  totalTrades: 127,
  avgR: 0.22,
  totalR: 28.4,
  rSquared: 0.87,
  maxDD: -4.2,
};

const MOCK_TRADES: TradeRow[] = [
  { id: 1, entryTime: '03/18 09:35', exitTime: '03/18 10:12', direction: 'LONG', entryPrice: 567.42, exitPrice: 569.18, rMultiple: 1.84, execType: 'C', exitReason: 'signal' },
  { id: 2, entryTime: '03/18 10:45', exitTime: '03/18 11:30', direction: 'LONG', entryPrice: 568.90, exitPrice: 567.15, rMultiple: -1.00, execType: 'C', exitReason: 'stop' },
  { id: 3, entryTime: '03/18 13:15', exitTime: '03/18 14:02', direction: 'LONG', entryPrice: 567.80, exitPrice: 570.25, rMultiple: 2.56, execType: 'L0', exitReason: 'target' },
  { id: 4, entryTime: '03/19 09:32', exitTime: '03/19 09:58', direction: 'LONG', entryPrice: 571.10, exitPrice: 570.20, rMultiple: -0.85, execType: 'HM', exitReason: 'stop' },
  { id: 5, entryTime: '03/19 10:15', exitTime: '03/19 11:45', direction: 'LONG', entryPrice: 570.55, exitPrice: 573.40, rMultiple: 3.12, execType: 'C', exitReason: 'signal' },
  { id: 6, entryTime: '03/19 13:05', exitTime: '03/19 13:42', direction: 'LONG', entryPrice: 572.80, exitPrice: 573.60, rMultiple: 0.84, execType: 'C', exitReason: 'bar_count' },
  { id: 7, entryTime: '03/19 14:10', exitTime: '03/19 15:15', direction: 'LONG', entryPrice: 573.20, exitPrice: 572.05, rMultiple: -1.00, execType: 'L1', exitReason: 'stop' },
  { id: 8, entryTime: '03/20 09:35', exitTime: '03/20 10:50', direction: 'LONG', entryPrice: 570.15, exitPrice: 572.90, rMultiple: 2.89, execType: 'C', exitReason: 'target' },
  { id: 9, entryTime: '03/20 11:20', exitTime: '03/20 12:05', direction: 'LONG', entryPrice: 572.45, exitPrice: 573.80, rMultiple: 1.41, execType: 'HM', exitReason: 'signal' },
  { id: 10, entryTime: '03/20 13:30', exitTime: '03/20 14:45', direction: 'LONG', entryPrice: 573.10, exitPrice: 571.90, rMultiple: -1.00, execType: 'C', exitReason: 'stop' },
  { id: 11, entryTime: '03/20 15:00', exitTime: '03/20 15:45', direction: 'LONG', entryPrice: 572.20, exitPrice: 574.50, rMultiple: 2.41, execType: 'C', exitReason: 'signal' },
  { id: 12, entryTime: '03/21 09:33', exitTime: '03/21 10:20', direction: 'LONG', entryPrice: 574.80, exitPrice: 575.95, rMultiple: 1.20, execType: 'L0', exitReason: 'signal' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EXEC_TYPE_COLORS: Record<string, string> = {
  C: 'var(--blue)', L0: 'var(--green)', L1: 'var(--orange)',
  HM: 'var(--accent)', HL: 'var(--red)',
};

const EXIT_REASON_COLORS: Record<string, string> = {
  signal: 'var(--green)', target: 'var(--green)',
  stop: 'var(--red)', bar_count: 'var(--orange)',
};

function ExecBadge({ type }: { type: string }) {
  const color = EXEC_TYPE_COLORS[type] || 'var(--text-muted)';
  return (
    <span
      className="inline-flex items-center justify-center px-1.5 py-0.5 rounded text-[10px] font-mono font-bold leading-none"
      style={{
        color,
        background: `color-mix(in srgb, ${color} 15%, transparent)`,
        border: `1px solid color-mix(in srgb, ${color} 30%, transparent)`,
      }}
    >
      [{type}]
    </span>
  );
}

// ---------------------------------------------------------------------------
// Inline Components
// ---------------------------------------------------------------------------

function ConfigSelect({
  value,
  options,
  onChange,
}: {
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      className="px-2 py-1.5 rounded-lg text-sm"
      style={{
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        color: 'var(--text-primary)',
      }}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  );
}

function ConfigInput({
  value,
  onChange,
  placeholder,
  width,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  width?: string;
}) {
  return (
    <input
      type="text"
      className="px-2 py-1.5 rounded-lg text-sm"
      style={{
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        color: 'var(--text-primary)',
        width: width || '80px',
      }}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}

function NumberField({
  value,
  onChange,
  min,
  max,
  step,
  width,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  width?: string;
}) {
  return (
    <input
      type="number"
      className="px-2 py-1.5 rounded-lg text-sm"
      style={{
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        color: 'var(--text-primary)',
        width: width || '70px',
      }}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
    />
  );
}

function Chip({
  label,
  onRemove,
  color,
}: {
  label: string;
  onRemove?: () => void;
  color?: string;
}) {
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs"
      style={{
        background: color ? `color-mix(in srgb, ${color} 12%, transparent)` : 'var(--bg-input)',
        border: `1px solid ${color ? `color-mix(in srgb, ${color} 30%, transparent)` : 'var(--border)'}`,
        color: color || 'var(--text-secondary)',
      }}
    >
      {label}
      {onRemove && (
        <button
          onClick={onRemove}
          className="hover:opacity-70 transition-opacity ml-0.5"
          style={{ color: 'var(--text-muted)', fontSize: '10px' }}
        >
          x
        </button>
      )}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Confluence Picker Modal (unified TF + General)
// ---------------------------------------------------------------------------

function ConfluencePickerModal({
  isOpen,
  onClose,
  selected,
  onToggle,
}: {
  isOpen: boolean;
  onClose: () => void;
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    if (!search) return ALL_CONFLUENCE;
    const q = search.toLowerCase();
    return ALL_CONFLUENCE.filter(
      (c) => c.label.toLowerCase().includes(q) || c.id.toLowerCase().includes(q) || c.pack.toLowerCase().includes(q)
    );
  }, [search]);

  const grouped = useMemo(() => {
    const map: Record<string, ConfluenceCondition[]> = {};
    for (const c of filtered) {
      if (!map[c.pack]) map[c.pack] = [];
      map[c.pack].push(c);
    }
    return map;
  }, [filtered]);

  return (
    <Modal title="Confluence Conditions" isOpen={isOpen} onClose={onClose} width="480px">
      <input
        type="text"
        className="w-full px-3 py-2 rounded-lg text-sm mb-3"
        style={{
          background: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
        }}
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search conditions..."
      />
      <div className="space-y-0.5 overflow-y-auto" style={{ maxHeight: 400 }}>
        {Object.entries(grouped).map(([pack, conds]) => (
          <div key={pack}>
            <div
              className="text-[10px] font-semibold uppercase tracking-wider px-2 py-1.5 sticky top-0"
              style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}
            >
              {pack}
            </div>
            {conds.map((c) => {
              const active = selected.has(c.id);
              return (
                <button
                  key={c.id}
                  className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
                  style={{
                    background: active ? 'var(--accent-muted)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--text-secondary)',
                  }}
                  onClick={() => onToggle(c.id)}
                >
                  <span
                    className="w-4 h-4 rounded border flex items-center justify-center text-[10px]"
                    style={{
                      borderColor: active ? 'var(--accent)' : 'var(--border)',
                      background: active ? 'var(--accent)' : 'transparent',
                      color: active ? '#000' : 'transparent',
                    }}
                  >
                    {active ? '\u2713' : ''}
                  </span>
                  <span className="flex-1">{c.label}</span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{c.id}</span>
                </button>
              );
            })}
          </div>
        ))}
        {Object.keys(grouped).length === 0 && (
          <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>No conditions match.</p>
        )}
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function StrategyBuilderV3() {
  // ---- Config State ----
  const [symbol, setSymbol] = useState('SPY');
  const [direction, setDirection] = useState('LONG');
  const [timeframe, setTimeframe] = useState('5Min');
  const [session, setSession] = useState('RTH');
  const [lookbackDays, setLookbackDays] = useState(30);

  // ---- Entry/Exit ----
  const [entryTrigger, setEntryTrigger] = useState('ema-price-cross-up');
  const [triggerSearch, setTriggerSearch] = useState('');
  const [exitTrigger, setExitTrigger] = useState('ema-price-cross-down');

  // ---- Exit Config ----
  const [exitExpanded, setExitExpanded] = useState(false);
  const [stopMethod, setStopMethod] = useState('Swing');
  const [stopValue, setStopValue] = useState(5);
  const [targetMethod, setTargetMethod] = useState('R:R');
  const [targetValue, setTargetValue] = useState(2.0);
  const [oppositeSignalExit, setOppositeSignalExit] = useState(true);

  // ---- Confluence ----
  const [selectedConditions, setSelectedConditions] = useState<Set<string>>(
    new Set(['5M-EMA_STACK-BULL', '1D-MACD_LINE-BULL'])
  );
  const [confluenceModalOpen, setConfluenceModalOpen] = useState(false);

  // ---- Backtest State ----
  const [backtestRan, setBacktestRan] = useState(false);

  // ---- Derived ----
  const isCrypto = symbol.includes('/');
  const effectiveSession = isCrypto ? '24/7' : session;
  const entryDef = MOCK_TRIGGERS.find((t) => t.id === entryTrigger);
  const exitDef = MOCK_TRIGGERS.find((t) => t.id === exitTrigger);

  const filteredTriggers = useMemo(() => {
    const q = triggerSearch.toLowerCase();
    return MOCK_TRIGGERS.filter((t) => {
      // Direction filtering
      if (direction === 'LONG') {
        if (t.name.toLowerCase().match(/down|bear|sell|short|overbought/)) return false;
      } else {
        if (t.name.toLowerCase().match(/up|bull|buy|long|oversold/)) return false;
      }
      // Search filtering
      if (q && !t.name.toLowerCase().includes(q) && !t.pack.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [triggerSearch, direction]);

  const handleToggleCondition = useCallback((id: string) => {
    setSelectedConditions((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleRunBacktest = useCallback(() => {
    setBacktestRan(true);
  }, []);

  const exitSummary = `Stop: ${stopMethod}${stopMethod === 'Swing' ? ` (${stopValue} bars)` : stopMethod === 'ATR' ? ` x${stopValue}` : stopMethod === 'Fixed $' ? ` $${stopValue}` : ` ${stopValue}%`} | TP: ${targetMethod === 'None' ? 'None' : `${targetValue}${targetMethod === 'R:R' ? 'R' : targetMethod === 'ATR' ? 'x ATR' : targetMethod === 'Fixed $' ? '$' : '%'}`}${oppositeSignalExit ? ' | Exit: Opposite Signal' : ''}`;

  // ---- RENDER ----
  return (
    <div className="max-w-[1200px] mx-auto">
      {/* ================================================================= */}
      {/* COMPACT CONFIG BAR                                                */}
      {/* ================================================================= */}
      <Card className="mb-3">
        <div className="flex flex-wrap items-center gap-3">
          {/* Ticker */}
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Ticker</span>
            <ConfigInput
              value={symbol}
              onChange={(v) => setSymbol(v.toUpperCase())}
              placeholder="SPY"
              width="90px"
            />
          </div>

          {/* Direction */}
          <ConfigSelect value={direction} options={DIRECTIONS} onChange={setDirection} />

          {/* Timeframe */}
          <ConfigSelect value={timeframe} options={TIMEFRAMES} onChange={setTimeframe} />

          {/* Session */}
          {isCrypto ? (
            <span className="text-xs px-2 py-1.5 rounded-lg" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
              24/7
            </span>
          ) : (
            <ConfigSelect value={session} options={SESSIONS} onChange={setSession} />
          )}

          {/* Lookback */}
          <div className="flex items-center gap-1.5">
            <NumberField value={lookbackDays} onChange={setLookbackDays} min={7} max={365} step={7} width="60px" />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>days</span>
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Run Backtest */}
          <button
            className="px-4 py-1.5 rounded-lg text-sm font-medium transition-all"
            style={{
              background: 'var(--accent)',
              color: '#000',
              cursor: 'pointer',
            }}
            onClick={handleRunBacktest}
          >
            {backtestRan ? 'Re-run' : 'Run Backtest'}
          </button>
        </div>
      </Card>

      {/* ================================================================= */}
      {/* TRIGGER SELECTOR                                                  */}
      {/* ================================================================= */}
      <Card className="mb-3">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Entry</span>
          {entryDef && (
            <div className="flex items-center gap-1.5">
              <ExecBadge type={entryDef.execType} />
              <span className="text-sm font-medium">{entryDef.name}</span>
            </div>
          )}
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>|</span>
          <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Exit</span>
          {exitDef && (
            <div className="flex items-center gap-1.5">
              <ExecBadge type={exitDef.execType} />
              <span className="text-sm">{exitDef.name}</span>
            </div>
          )}
        </div>

        {/* Search box */}
        <input
          type="text"
          className="w-full px-3 py-2 rounded-lg text-sm mb-2"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
          }}
          value={triggerSearch}
          onChange={(e) => setTriggerSearch(e.target.value)}
          placeholder="Search triggers... (click to set entry, shift+click for exit)"
        />

        {/* Trigger chips */}
        <div className="flex flex-wrap gap-1.5">
          {filteredTriggers.map((t) => {
            const isEntry = t.id === entryTrigger;
            const isExit = t.id === exitTrigger;
            return (
              <button
                key={t.id}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-all"
                style={{
                  background: isEntry
                    ? 'var(--accent-muted)'
                    : isExit
                    ? `color-mix(in srgb, var(--orange) 15%, transparent)`
                    : 'var(--bg-input)',
                  border: `1px solid ${isEntry ? 'var(--accent)' : isExit ? 'var(--orange)' : 'var(--border)'}`,
                  color: isEntry ? 'var(--accent)' : isExit ? 'var(--orange)' : 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
                onClick={(e) => {
                  if (e.shiftKey) {
                    setExitTrigger(t.id);
                  } else {
                    setEntryTrigger(t.id);
                  }
                }}
                title={isEntry ? 'Current entry' : isExit ? 'Current exit' : 'Click = set entry, Shift+click = set exit'}
              >
                <ExecBadge type={t.execType} />
                <span>{t.name}</span>
                {isEntry && <span className="text-[9px] font-semibold ml-0.5">ENTRY</span>}
                {isExit && <span className="text-[9px] font-semibold ml-0.5">EXIT</span>}
              </button>
            );
          })}
        </div>
      </Card>

      {/* ================================================================= */}
      {/* EXIT CONFIG (collapsed summary line)                              */}
      {/* ================================================================= */}
      <Card className="mb-3">
        <button
          className="w-full flex items-center justify-between"
          onClick={() => setExitExpanded(!exitExpanded)}
        >
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            {exitSummary}
          </span>
          <span
            className="text-xs transition-transform"
            style={{
              color: 'var(--text-muted)',
              transform: exitExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
            }}
          >
            v
          </span>
        </button>

        {exitExpanded && (
          <div className="mt-3 pt-3 grid grid-cols-1 sm:grid-cols-3 gap-4" style={{ borderTop: '1px solid var(--border)' }}>
            {/* Stop Loss */}
            <div>
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
                Stop Loss
              </label>
              <ConfigSelect value={stopMethod} options={STOP_METHODS} onChange={setStopMethod} />
              <div className="mt-1.5">
                <NumberField
                  value={stopValue}
                  onChange={setStopValue}
                  min={stopMethod === 'Swing' ? 2 : 0.1}
                  max={stopMethod === 'Swing' ? 50 : 100}
                  step={stopMethod === 'Swing' ? 1 : 0.1}
                  width="90px"
                />
                <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>
                  {stopMethod === 'Swing' ? 'bars' : stopMethod === 'ATR' ? 'x mult' : stopMethod === 'Fixed $' ? '$' : '%'}
                </span>
              </div>
            </div>

            {/* Take Profit */}
            <div>
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
                Take Profit
              </label>
              <ConfigSelect value={targetMethod} options={TARGET_METHODS} onChange={setTargetMethod} />
              {targetMethod !== 'None' && (
                <div className="mt-1.5">
                  <NumberField
                    value={targetValue}
                    onChange={setTargetValue}
                    min={0.5}
                    max={10}
                    step={0.5}
                    width="90px"
                  />
                  <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>
                    {targetMethod === 'R:R' ? 'R' : targetMethod === 'ATR' ? 'x mult' : targetMethod === 'Fixed $' ? '$' : '%'}
                  </span>
                </div>
              )}
            </div>

            {/* Exit Signal */}
            <div>
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
                Exit on Opposite Signal
              </label>
              <div
                className="relative inline-block w-9 h-5 rounded-full cursor-pointer transition-colors"
                style={{ background: oppositeSignalExit ? 'var(--accent)' : 'var(--bg-input)' }}
                onClick={() => setOppositeSignalExit(!oppositeSignalExit)}
              >
                <div
                  className="absolute top-0.5 w-4 h-4 rounded-full transition-all"
                  style={{ background: 'white', left: oppositeSignalExit ? '18px' : '2px' }}
                />
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* CONFLUENCE CHIPS                                                  */}
      {/* ================================================================= */}
      <div className="flex flex-wrap items-center gap-1.5 mb-3">
        <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Confluence:</span>
        {selectedConditions.size === 0 && (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>None</span>
        )}
        {Array.from(selectedConditions).sort().map((cid) => {
          const cDef = ALL_CONFLUENCE.find((c) => c.id === cid);
          return (
            <Chip
              key={cid}
              label={cDef?.label || cid}
              color="var(--accent)"
              onRemove={() => handleToggleCondition(cid)}
            />
          );
        })}
        <button
          className="text-xs px-2 py-1 rounded-md transition-colors"
          style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
          onClick={() => setConfluenceModalOpen(true)}
        >
          + Add
        </button>
      </div>

      {/* ================================================================= */}
      {/* PRE-BACKTEST STATE                                                */}
      {/* ================================================================= */}
      {!backtestRan && (
        <Card className="text-center py-16">
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Configure your strategy above, then click <strong>Run Backtest</strong>.
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {symbol} | {direction} | {timeframe} | {effectiveSession} | {lookbackDays} days
          </p>
        </Card>
      )}

      {/* ================================================================= */}
      {/* RESULTS (after backtest)                                          */}
      {/* ================================================================= */}
      {backtestRan && (
        <>
          {/* ---- KPI Strip ---- */}
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 mb-3">
            <MetricCard label="Trades" value={String(MOCK_KPIS.totalTrades)} />
            <MetricCard label="Win Rate" value={`${MOCK_KPIS.winRate.toFixed(1)}%`} positive={MOCK_KPIS.winRate > 50} />
            <MetricCard label="Profit Factor" value={MOCK_KPIS.profitFactor.toFixed(2)} positive={MOCK_KPIS.profitFactor > 1} />
            <MetricCard label="Daily R" value={`+${MOCK_KPIS.dailyR.toFixed(2)}`} positive={MOCK_KPIS.dailyR > 0} />
            <MetricCard label="Total R" value={`+${MOCK_KPIS.totalR.toFixed(1)}`} positive={MOCK_KPIS.totalR > 0} />
            <MetricCard label="Avg R" value={`+${MOCK_KPIS.avgR.toFixed(2)}`} positive={MOCK_KPIS.avgR > 0} />
            <MetricCard label="R-Squared" value={MOCK_KPIS.rSquared.toFixed(2)} positive={MOCK_KPIS.rSquared > 0.7} />
            <MetricCard label="Max DD" value={`${MOCK_KPIS.maxDD.toFixed(1)}R`} positive={false} />
          </div>

          {/* ---- Equity Curve ---- */}
          <Card className="mb-3">
            <ChartPlaceholder label="Equity Curve -- cumulative R over time" height={280} />
          </Card>

          {/* ---- Price Chart ---- */}
          <Card className="mb-3">
            <ChartPlaceholder label="OHLC Price Chart with trade entry/exit markers and indicator overlays" height={350} />
            <div className="flex gap-3 mt-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
              <span><span style={{ color: 'var(--green)' }}>+</span> Entry</span>
              <span><span style={{ color: 'var(--green)' }}>x</span> Exit (win)</span>
              <span><span style={{ color: 'var(--red)' }}>x</span> Exit (loss)</span>
              <span><span style={{ color: 'var(--orange)' }}>x</span> Exit (unconfirmed)</span>
              <span><span style={{ color: '#009688' }}>x</span> Exit (bar count)</span>
            </div>
          </Card>

          {/* ---- Trade History ---- */}
          <Card className="mb-3">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Trade History
              </h3>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {MOCK_TRADES.length} trades
              </span>
            </div>
            <div className="overflow-x-auto" style={{ maxHeight: 320 }}>
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    {['#', 'Entry', 'Exit', 'Entry $', 'Exit $', 'P&L (R)', 'Exec', 'Reason'].map((h) => (
                      <th
                        key={h}
                        className="text-left px-2 py-1.5 font-medium whitespace-nowrap sticky top-0"
                        style={{ color: 'var(--text-muted)', background: 'var(--bg-card)' }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {MOCK_TRADES.map((t) => (
                    <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>{t.id}</td>
                      <td className="px-2 py-1.5 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{t.entryTime}</td>
                      <td className="px-2 py-1.5 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{t.exitTime}</td>
                      <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-secondary)' }}>${t.entryPrice.toFixed(2)}</td>
                      <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-secondary)' }}>${t.exitPrice.toFixed(2)}</td>
                      <td className="px-2 py-1.5 font-mono font-medium" style={{ color: t.rMultiple >= 0 ? 'var(--green)' : 'var(--red)' }}>
                        {t.rMultiple >= 0 ? '+' : ''}{t.rMultiple.toFixed(2)}
                      </td>
                      <td className="px-2 py-1.5"><ExecBadge type={t.execType} /></td>
                      <td className="px-2 py-1.5">
                        <span
                          className="text-[10px] px-1.5 py-0.5 rounded"
                          style={{
                            color: EXIT_REASON_COLORS[t.exitReason] || 'var(--text-muted)',
                            background: `color-mix(in srgb, ${EXIT_REASON_COLORS[t.exitReason] || 'var(--text-muted)'} 12%, transparent)`,
                          }}
                        >
                          {t.exitReason}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* ---- Save / Actions ---- */}
          <div className="flex items-center gap-3">
            <button
              className="px-5 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: '#000', cursor: 'pointer' }}
            >
              Save Strategy
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
            >
              Save as Draft
            </button>
          </div>
        </>
      )}

      {/* ---- Confluence Modal ---- */}
      <ConfluencePickerModal
        isOpen={confluenceModalOpen}
        onClose={() => setConfluenceModalOpen(false)}
        selected={selectedConditions}
        onToggle={handleToggleCondition}
      />
    </div>
  );
}
