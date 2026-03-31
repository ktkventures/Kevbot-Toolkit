'use client';

/**
 * Strategy Builder — Copy of V5.tsx with mock data replaced by API hooks.
 *
 * Mock data references replaced:
 * - API_TRIGGERS → useConfluenceTriggers() from API
 * - API_STOP_PACKS / API_TARGET_PACKS → useRiskManagementPacks()
 * - API_CONFLUENCE_CONDITIONS → useConfluenceGroups() derived
 * - API_GENERAL_CONDITIONS → useGeneralPacks() derived
 * - EMPTY_KPIS / EMPTY_TRADES → useRunBacktest() mutation results
 * - Save → useCreateStrategy() mutation
 */

import { useState, useMemo, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import EquityCurve from '@/charts/EquityCurve';
import TradingChart from '@/charts/TradingChart';
const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });
import type { CandleData, TradeMarker } from '@/charts/TradingChart';
import { buildStrategyChartPanes } from '@/charts/buildStrategyChartPanes';
import { useChartPrefs } from '@/hooks/useChartPrefs';
import TabBar from '@/components/TabBar';
import Modal from '@/components/Modal';
import { useRunBacktest, useAnalyzeTriggers, type AnalyzeResult } from '@/hooks/queries/useBacktest';
import { useCreateStrategy } from '@/hooks/mutations/useStrategyMutations';
import { useConfluenceGroups, useConfluenceTriggers, useGeneralPacks, useRiskManagementPacks } from '@/hooks/queries/usePacks';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TriggerDef {
  id: string;
  name: string;
  execType: 'C' | 'L' | 'LC' | 'CC';
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
  confluences: string;
}

interface KPIs {
  winRate: number;
  profitFactor: number;
  dailyR: number;
  totalTrades: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  recoveryFactor: number;
  sharpeRatio: number;
  expectancy: number;
  totalR: number;
  avgR: number;
  rSquared: number;
  maxRDrawdown: number;
}

// ---------------------------------------------------------------------------
// Mock Data
// ---------------------------------------------------------------------------

const TIMEFRAMES = ['1Min', '5Min', '15Min', '1H', '4H', '1D'];
const DIRECTIONS = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];
const ASSET_TYPES = ['Equity', 'Crypto'];
const LOOKBACK_MODES = ['Days', 'Bars/Candles', 'Date Range'];
// Stop/Target packs — selectable saved variations
interface RiskPack { id: string; name: string; version: string; summary: string }
const API_STOP_PACKS: RiskPack[] = [
  { id: 'atr-default', name: 'ATR Stop', version: 'Default', summary: '1.5x ATR' },
  { id: 'atr-tight', name: 'ATR Stop', version: 'Tight', summary: '1.0x ATR' },
  { id: 'fixed-default', name: 'Fixed Dollar', version: 'Default', summary: '$1.00 fixed' },
  { id: 'pct-default', name: 'Percentage', version: 'Default', summary: '0.5% from entry' },
  { id: 'swing-default', name: 'Swing Stop', version: 'Default', summary: '5-bar swing, $0.05 pad' },
  { id: 'swing-wide', name: 'Swing Stop', version: 'Wide', summary: '8-bar swing, $0.10 pad' },
  { id: 'trail-default', name: 'ATR Trailing', version: 'Default', summary: '1.5x ATR \u2192 trail 1.0x @ 1.0R' },
  { id: 'be-default', name: 'Breakeven', version: 'Default', summary: '1.5x ATR \u2192 BE @ 1.0R' },
];
const API_TARGET_PACKS: RiskPack[] = [
  { id: 'rr-default', name: 'Risk:Reward', version: 'Default', summary: '2:1 R:R' },
  { id: 'rr-3to1', name: 'Risk:Reward', version: '3:1 Aggressive', summary: '3:1 R:R' },
  { id: 'atr-target-default', name: 'ATR Target', version: 'Default', summary: '2.0x ATR' },
  { id: 'fixed-target-default', name: 'Fixed Dollar', version: 'Default', summary: '$2.00 fixed' },
  { id: 'swing-target-default', name: 'Swing Target', version: 'Default', summary: '5-bar swing, $0.05 pad' },
  { id: 'no-target', name: 'No Target', version: 'Default', summary: 'Exit via stop, signal, or bar count' },
];

// Each trigger is a base trigger + execution type variant (4 per base trigger)
const API_TRIGGERS: TriggerDef[] = [
  // EMA Stack — 4 base triggers × 4 exec types
  { id: 'ema-cross-bull-c', name: 'Short > Mid Cross', execType: 'C', pack: 'EMA Stack (Default)' },
  { id: 'ema-cross-bull-l', name: 'Short > Mid Cross', execType: 'L', pack: 'EMA Stack (Default)' },
  { id: 'ema-cross-bull-lc', name: 'Short > Mid Cross', execType: 'LC', pack: 'EMA Stack (Default)' },
  { id: 'ema-cross-bull-cc', name: 'Short > Mid Cross', execType: 'CC', pack: 'EMA Stack (Default)' },
  { id: 'ema-cross-bear-c', name: 'Short < Mid Cross', execType: 'C', pack: 'EMA Stack (Default)' },
  { id: 'ema-cross-bear-l', name: 'Short < Mid Cross', execType: 'L', pack: 'EMA Stack (Default)' },
  { id: 'ema-mid-cross-bull-c', name: 'Mid > Long Cross', execType: 'C', pack: 'EMA Stack (Default)' },
  { id: 'ema-mid-cross-bear-c', name: 'Mid < Long Cross', execType: 'C', pack: 'EMA Stack (Default)' },
  // MACD Line
  { id: 'macd-cross-bull-c', name: 'Bullish Cross', execType: 'C', pack: 'MACD Line (Default)' },
  { id: 'macd-cross-bull-l', name: 'Bullish Cross', execType: 'L', pack: 'MACD Line (Default)' },
  { id: 'macd-cross-bear-c', name: 'Bearish Cross', execType: 'C', pack: 'MACD Line (Default)' },
  { id: 'macd-zero-cross-up-c', name: 'Zero Line Cross Up', execType: 'C', pack: 'MACD Line (Default)' },
  // VWAP — shows all 4 exec types
  { id: 'vwap-cross-above-c', name: 'Cross Above VWAP', execType: 'C', pack: 'VWAP (Default)' },
  { id: 'vwap-cross-above-l', name: 'Cross Above VWAP', execType: 'L', pack: 'VWAP (Default)' },
  { id: 'vwap-cross-above-lc', name: 'Cross Above VWAP', execType: 'LC', pack: 'VWAP (Default)' },
  { id: 'vwap-cross-below-c', name: 'Cross Below VWAP', execType: 'C', pack: 'VWAP (Default)' },
  { id: 'vwap-cross-below-l', name: 'Cross Below VWAP', execType: 'L', pack: 'VWAP (Default)' },
  // UT Bot
  { id: 'utbot-buy-c', name: 'Buy Signal', execType: 'C', pack: 'UT Bot (Default)' },
  { id: 'utbot-buy-l', name: 'Buy Signal', execType: 'L', pack: 'UT Bot (Default)' },
  { id: 'utbot-buy-lc', name: 'Buy Signal', execType: 'LC', pack: 'UT Bot (Default)' },
  { id: 'utbot-sell-c', name: 'Sell Signal', execType: 'C', pack: 'UT Bot (Default)' },
  { id: 'utbot-sell-l', name: 'Sell Signal', execType: 'L', pack: 'UT Bot (Default)' },
  // EMA Price Position
  { id: 'ema-pp-cross-short-up-c', name: 'Price Cross Short EMA Up', execType: 'C', pack: 'EMA Price Position (Default)' },
  { id: 'ema-pp-cross-short-up-l', name: 'Price Cross Short EMA Up', execType: 'L', pack: 'EMA Price Position (Default)' },
  { id: 'ema-pp-cross-short-down-c', name: 'Price Cross Short EMA Down', execType: 'C', pack: 'EMA Price Position (Default)' },
  { id: 'ema-pp-cross-mid-up-c', name: 'Price Cross Mid EMA Up', execType: 'C', pack: 'EMA Price Position (Default)' },
  // MACD Histogram
  { id: 'macd-hist-flip-pos-c', name: 'Histogram Flip Positive', execType: 'C', pack: 'MACD Histogram (Default)' },
  { id: 'macd-hist-flip-neg-c', name: 'Histogram Flip Negative', execType: 'C', pack: 'MACD Histogram (Default)' },
  // RVOL
  { id: 'rvol-spike-c', name: 'Volume Spike', execType: 'C', pack: 'Relative Volume (Default)' },
  { id: 'rvol-extreme-c', name: 'Extreme Volume', execType: 'C', pack: 'Relative Volume (Default)' },
  // Bar Count Exit
  { id: 'bar-count-exit-c', name: 'Bar Count Exit', execType: 'C', pack: 'Bar Count Exit (Default)' },
];

const API_CONFLUENCE_CONDITIONS: ConfluenceCondition[] = [
  { id: '5M-EMA_STACK-BULL', label: '5M EMA Stack Bullish', pack: 'EMA Stack (Default)' },
  { id: '5M-EMA_STACK-SML', label: '5M EMA Stack S>M>L', pack: 'EMA Stack (Default)' },
  { id: '15M-EMA_STACK-BULL', label: '15M EMA Stack Bullish', pack: 'EMA Stack (Default)' },
  { id: '1H-EMA_STACK-BULL', label: '1H EMA Stack Bullish', pack: 'EMA Stack (Default)' },
  { id: '1D-MACD_LINE-BULL', label: '1D MACD Line Bullish', pack: 'MACD Line (Default)' },
  { id: '1D-MACD_LINE-BEAR', label: '1D MACD Line Bearish', pack: 'MACD Line (Default)' },
  { id: '5M-MACD_LINE-BULL', label: '5M MACD Line Bullish', pack: 'MACD Line (Default)' },
  { id: '1H-VWAP_POS-ABOVE', label: '1H Above VWAP', pack: 'VWAP (Default)' },
  { id: '1H-VWAP_POS-BELOW', label: '1H Below VWAP', pack: 'VWAP (Default)' },
  { id: '5M-RSI_ZONE-NEUTRAL', label: '5M RSI Neutral Zone', pack: 'RSI (Default)' },
  { id: '1D-RSI_ZONE-BULL', label: '1D RSI Bullish Zone', pack: 'RSI (Default)' },
  { id: '15M-SUPERTREND-BULL', label: '15M Supertrend Bullish', pack: 'Supertrend (Default)' },
  { id: '1H-SUPERTREND-BULL', label: '1H Supertrend Bullish', pack: 'Supertrend (Default)' },
];

const API_GENERAL_CONDITIONS: ConfluenceCondition[] = [
  { id: 'GEN-TOD-NY_MORNING', label: 'NY Morning (9:30-11:30)', pack: 'Time of Day' },
  { id: 'GEN-TOD-NY_AFTERNOON', label: 'NY Afternoon (13:00-15:30)', pack: 'Time of Day' },
  { id: 'GEN-DOW-MON_TUE_WED', label: 'Mon/Tue/Wed', pack: 'Day of Week' },
  { id: 'GEN-DOW-TUE_THU', label: 'Tue/Thu', pack: 'Day of Week' },
  { id: 'GEN-SESSION-RTH', label: 'Regular Trading Hours', pack: 'Session Filter' },
];

// Empty KPIs — shown before a backtest is run. No fake numbers.
const EMPTY_KPIS: KPIs = {
  winRate: 0, profitFactor: 0, dailyR: 0, totalTrades: 0,
  avgWin: 0, avgLoss: 0, maxDrawdown: 0, recoveryFactor: 0,
  sharpeRatio: 0, expectancy: 0, totalR: 0, avgR: 0,
  rSquared: 0, maxRDrawdown: 0,
};

// Empty trades — shown before a backtest is run. No fake data.
const EMPTY_TRADES: TradeRow[] = [];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Display settings V5: uniform blue for all exec types
const EXEC_BADGE_COLOR = '#2196F3';

function ExecBadge({ type }: { type: string }) {
  return (
    <span
      className="text-xs font-mono font-medium px-2 py-0.5 rounded-full"
      style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}
    >
      [{type}]
    </span>
  );
}

function ExitReasonBadge({ reason }: { reason: string }) {
  const colors: Record<string, string> = {
    signal: 'var(--green)',
    target: 'var(--green)',
    stop: 'var(--red)',
    bar_count: 'var(--orange)',
  };
  const c = colors[reason] || 'var(--text-muted)';
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded font-medium"
      style={{
        color: c,
        background: `color-mix(in srgb, ${c} 12%, transparent)`,
      }}
    >
      {reason}
    </span>
  );
}

function ConditionPill({
  label,
  onRemove,
}: {
  label: string;
  onRemove?: () => void;
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs"
      style={{
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        color: 'var(--text-secondary)',
      }}
    >
      {label}
      {onRemove && (
        <button
          onClick={onRemove}
          className="ml-0.5 hover:opacity-70 transition-opacity"
          style={{ color: 'var(--text-muted)' }}
        >
          x
        </button>
      )}
    </span>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-muted)' }}>
      {children}
    </label>
  );
}

function SelectInput({
  options,
  value,
  onChange,
}: {
  options: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <select
      className="w-full px-3 py-2 rounded-lg text-sm"
      style={{
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        color: 'var(--text-primary)',
      }}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  suffix,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-1">
      <input
        type="number"
        className="w-full px-3 py-2 rounded-lg text-sm"
        style={{
          background: 'var(--bg-input)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
        }}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
      />
      {suffix && (
        <span className="text-xs whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
          {suffix}
        </span>
      )}
    </div>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      className="w-full px-3 py-2 rounded-lg text-sm"
      style={{
        background: 'var(--bg-input)',
        border: '1px solid var(--border)',
        color: 'var(--text-primary)',
      }}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
    />
  );
}

function PrimaryButton({
  children,
  onClick,
  disabled,
  fullWidth,
  size = 'md',
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  fullWidth?: boolean;
  size?: 'sm' | 'md' | 'lg';
}) {
  const py = size === 'sm' ? 'py-1.5' : size === 'lg' ? 'py-3' : 'py-2';
  const text = size === 'sm' ? 'text-xs' : size === 'lg' ? 'text-base' : 'text-sm';
  return (
    <button
      className={`${py} px-4 rounded-lg font-medium ${text} transition-all ${fullWidth ? 'w-full' : ''}`}
      style={{
        background: disabled ? 'var(--bg-input)' : 'var(--accent)',
        color: disabled ? 'var(--text-muted)' : '#000',
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
      }}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

function SecondaryButton({
  children,
  onClick,
  size = 'sm',
}: {
  children: React.ReactNode;
  onClick?: () => void;
  size?: 'sm' | 'md';
}) {
  const py = size === 'sm' ? 'py-1' : 'py-2';
  const text = size === 'sm' ? 'text-xs' : 'text-sm';
  return (
    <button
      className={`${py} px-3 rounded-lg ${text} transition-all`}
      style={{
        background: 'var(--accent-muted)',
        color: 'var(--accent)',
      }}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TriggerGroupedList({
  triggers,
  searchQuery,
  selectedId,
  onSelect,
}: {
  triggers: TriggerDef[];
  searchQuery: string;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const filtered = useMemo(() => {
    if (!searchQuery) return triggers;
    const q = searchQuery.toLowerCase();
    return triggers.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.pack.toLowerCase().includes(q) ||
        t.execType.toLowerCase().includes(q)
    );
  }, [triggers, searchQuery]);

  const grouped = useMemo(() => {
    const map: Record<string, TriggerDef[]> = {};
    for (const t of filtered) {
      if (!map[t.pack]) map[t.pack] = [];
      map[t.pack].push(t);
    }
    return map;
  }, [filtered]);

  return (
    <div
      className="space-y-1 overflow-y-auto pr-1"
      style={{ maxHeight: 600 }}
    >
      {Object.entries(grouped).map(([pack, trigs]) => (
        <div key={pack}>
          <div
            className="text-[10px] font-semibold uppercase tracking-wider px-2 py-1.5 sticky top-0"
            style={{ color: 'var(--text-muted)', background: 'var(--bg-card)' }}
          >
            {pack}
          </div>
          {trigs.map((t) => (
            <button
              key={t.id}
              className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
              style={{
                background: t.id === selectedId ? 'var(--accent-muted)' : 'transparent',
                color: t.id === selectedId ? 'var(--accent)' : 'var(--text-secondary)',
              }}
              onClick={() => onSelect(t.id)}
            >
              <ExecBadge type={t.execType} />
              <span className="flex-1">{t.name}</span>
              {t.id === selectedId && (
                <span className="text-[10px]" style={{ color: 'var(--accent)' }}>
                  selected
                </span>
              )}
            </button>
          ))}
        </div>
      ))}
      {Object.keys(grouped).length === 0 && (
        <p className="text-xs p-3" style={{ color: 'var(--text-muted)' }}>
          No triggers match your search.
        </p>
      )}
    </div>
  );
}

function ConfluencePickerModal({
  isOpen,
  onClose,
  conditions,
  selected,
  onToggle,
}: {
  isOpen: boolean;
  onClose: () => void;
  conditions: ConfluenceCondition[];
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    if (!search) return conditions;
    const q = search.toLowerCase();
    return conditions.filter(
      (c) => c.label.toLowerCase().includes(q) || c.id.toLowerCase().includes(q) || c.pack.toLowerCase().includes(q)
    );
  }, [conditions, search]);

  const grouped = useMemo(() => {
    const map: Record<string, ConfluenceCondition[]> = {};
    for (const c of filtered) {
      if (!map[c.pack]) map[c.pack] = [];
      map[c.pack].push(c);
    }
    return map;
  }, [filtered]);

  return (
    <Modal title="Add Confluence Conditions" isOpen={isOpen} onClose={onClose} width="560px">
      <TextInput value={search} onChange={setSearch} placeholder="Search conditions..." />
      <div className="mt-4 space-y-1 overflow-y-auto" style={{ maxHeight: 400 }}>
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
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {c.id}
                  </span>
                </button>
              );
            })}
          </div>
        ))}
        {Object.keys(grouped).length === 0 && (
          <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>
            No conditions match your search.
          </p>
        )}
      </div>
    </Modal>
  );
}

function KPIDashboard({ kpis, backtestRan }: { kpis: KPIs; backtestRan: boolean }) {
  if (!backtestRan) return null;

  return (
    <div className="space-y-3">
      {/* Primary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8 gap-2">
        <MetricCard label="Trades" value={String(kpis.totalTrades)} />
        <MetricCard
          label="Win Rate"
          value={`${kpis.winRate.toFixed(1)}%`}
          positive={kpis.winRate > 50}
        />
        <MetricCard
          label="Profit Factor"
          value={kpis.profitFactor.toFixed(2)}
          positive={kpis.profitFactor > 1}
        />
        <MetricCard
          label="Avg R"
          value={`${kpis.avgR >= 0 ? '+' : ''}${kpis.avgR.toFixed(2)}`}
          positive={kpis.avgR > 0}
        />
        <MetricCard
          label="Total R"
          value={`${kpis.totalR >= 0 ? '+' : ''}${kpis.totalR.toFixed(1)}`}
          positive={kpis.totalR > 0}
        />
        <MetricCard
          label="Daily R"
          value={`${kpis.dailyR >= 0 ? '+' : ''}${kpis.dailyR.toFixed(2)}`}
          positive={kpis.dailyR > 0}
        />
        <MetricCard
          label="R-Squared"
          value={kpis.rSquared.toFixed(2)}
          positive={kpis.rSquared > 0.7}
        />
        <MetricCard
          label="Max R DD"
          value={`${kpis.maxRDrawdown.toFixed(1)}R`}
          positive={false}
        />
      </div>
      {/* Secondary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        <MetricCard
          label="Avg Win"
          value={`+${kpis.avgWin.toFixed(2)}R`}
          positive
        />
        <MetricCard label="Avg Loss" value={`${kpis.avgLoss.toFixed(2)}R`} />
        <MetricCard
          label="Max Drawdown"
          value={`${kpis.maxDrawdown.toFixed(1)}%`}
        />
        <MetricCard
          label="Recovery Factor"
          value={kpis.recoveryFactor.toFixed(2)}
          positive={kpis.recoveryFactor > 2}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={kpis.sharpeRatio.toFixed(2)}
          positive={kpis.sharpeRatio > 1}
        />
        <MetricCard
          label="Expectancy"
          value={`${kpis.expectancy >= 0 ? '+' : ''}${kpis.expectancy.toFixed(2)}R`}
          positive={kpis.expectancy > 0}
        />
      </div>
    </div>
  );
}

function OptimizableVariables({
  entryTrigger,
  exitTriggers,
  selectedConditions,
  generalConditions,
  selectedStopPack,
  selectedTargetPack,
  allTriggers,
  allConditions,
  allGeneralConditions,
  stopPacks,
  targetPacks,
  onRemoveCondition,
  onRemoveGeneral,
  onRemoveExit,
  onReload,
  isLoading,
}: {
  entryTrigger: string;
  exitTriggers: string[];
  selectedConditions: Set<string>;
  generalConditions: Set<string>;
  selectedStopPack: string;
  selectedTargetPack: string;
  allTriggers: TriggerDef[];
  allConditions: ConfluenceCondition[];
  allGeneralConditions: ConfluenceCondition[];
  stopPacks: RiskPack[];
  targetPacks: RiskPack[];
  onRemoveCondition: (id: string) => void;
  onRemoveGeneral: (id: string) => void;
  onRemoveExit: (idx: number) => void;
  onReload?: () => void;
  isLoading?: boolean;
}) {
  const entryDef = allTriggers.find((t) => t.id === entryTrigger);
  const stopPack = stopPacks.find((p) => p.id === selectedStopPack);
  const targetPack = targetPacks.find((p) => p.id === selectedTargetPack);

  function formatStopDisplay() {
    return stopPack ? `${stopPack.name} (${stopPack.version})` : 'None';
  }

  function formatTargetDisplay() {
    return targetPack ? `${targetPack.name} (${targetPack.version})` : 'None';
  }

  return (
    <Card>
      <div
        className="text-xs font-medium mb-3 flex items-center justify-between"
        style={{ color: 'var(--text-muted)' }}
      >
        <span>Optimizable Variables</span>
        {onReload && (
          <button
            onClick={onReload}
            disabled={isLoading}
            className="px-3 py-1 rounded text-xs font-medium transition-colors"
            style={{
              background: 'var(--accent)',
              color: 'white',
              opacity: isLoading ? 0.6 : 1,
              cursor: isLoading ? 'not-allowed' : 'pointer',
            }}
          >
            {isLoading ? 'Loading...' : 'Reload'}
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {/* Entry */}
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
            Entry
          </div>
          {entryDef ? (
            <div className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
              <ExecBadge type={entryDef.execType} />
              <span className="italic">{entryDef.name}</span>
            </div>
          ) : (
            <span className="text-xs italic" style={{ color: 'var(--text-muted)' }}>None</span>
          )}
        </div>

        {/* Exit(s) */}
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
            Exit(s)
          </div>
          {exitTriggers.length > 0 ? (
            <div className="space-y-1">
              {exitTriggers.map((eid, idx) => {
                const eDef = allTriggers.find((t) => t.id === eid);
                return (
                  <div key={eid} className="flex items-center gap-1 text-xs group">
                    {eDef && <ExecBadge type={eDef.execType} />}
                    <span className="italic flex-1" style={{ color: 'var(--text-secondary)' }}>
                      {eDef?.name || eid}
                    </span>
                    <button
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px]"
                      style={{ color: 'var(--text-muted)' }}
                      onClick={() => onRemoveExit(idx)}
                    >
                      x
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <span className="text-xs italic" style={{ color: 'var(--text-muted)' }}>None</span>
          )}
        </div>

        {/* TF Conditions */}
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
            TF Conditions
          </div>
          {selectedConditions.size > 0 ? (
            <div className="space-y-1">
              {Array.from(selectedConditions).sort().map((cid) => {
                const cDef = allConditions.find((c) => c.id === cid);
                return (
                  <div key={cid} className="flex items-center gap-1 text-xs group">
                    <span className="italic flex-1" style={{ color: 'var(--text-secondary)' }}>
                      {cDef?.label || cid}
                    </span>
                    <button
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px]"
                      style={{ color: 'var(--text-muted)' }}
                      onClick={() => onRemoveCondition(cid)}
                    >
                      x
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <span className="text-xs italic" style={{ color: 'var(--text-muted)' }}>None</span>
          )}
        </div>

        {/* General */}
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
            General
          </div>
          {generalConditions.size > 0 ? (
            <div className="space-y-1">
              {Array.from(generalConditions).sort().map((gid) => {
                const gDef = allGeneralConditions.find((c) => c.id === gid);
                return (
                  <div key={gid} className="flex items-center gap-1 text-xs group">
                    <span className="italic flex-1" style={{ color: 'var(--text-secondary)' }}>
                      {gDef?.label || gid}
                    </span>
                    <button
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px]"
                      style={{ color: 'var(--text-muted)' }}
                      onClick={() => onRemoveGeneral(gid)}
                    >
                      x
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <span className="text-xs italic" style={{ color: 'var(--text-muted)' }}>None</span>
          )}
        </div>

        {/* Stop Loss */}
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
            Stop Loss
          </div>
          <span className="text-xs italic" style={{ color: 'var(--text-secondary)' }}>
            {formatStopDisplay()}
          </span>
        </div>

        {/* Take Profit */}
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-muted)' }}>
            Take Profit
          </div>
          <span className="text-xs italic" style={{ color: 'var(--text-secondary)' }}>
            {formatTargetDisplay()}
          </span>
        </div>
      </div>
    </Card>
  );
}

function TradeHistoryTable({ trades }: { trades: TradeRow[] }) {
  const [sortCol, setSortCol] = useState<keyof TradeRow>('id');
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    const arr = [...trades];
    arr.sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return arr;
  }, [trades, sortCol, sortAsc]);

  function handleSort(col: keyof TradeRow) {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  }

  const columns: { key: keyof TradeRow; label: string; align?: 'right' | 'center' }[] = [
    { key: 'id', label: '#', align: 'center' },
    { key: 'entryTime', label: 'Entry Time' },
    { key: 'exitTime', label: 'Exit Time' },
    { key: 'direction', label: 'Dir', align: 'center' },
    { key: 'entryPrice', label: 'Entry $', align: 'right' },
    { key: 'exitPrice', label: 'Exit $', align: 'right' },
    { key: 'rMultiple', label: 'P&L (R)', align: 'right' },
    { key: 'execType', label: 'Exec', align: 'center' },
    { key: 'exitReason', label: 'Exit Reason', align: 'center' },
  ];

  const sortIndicator = (col: keyof TradeRow) => {
    if (sortCol !== col) return '';
    return sortAsc ? ' \u2191' : ' \u2193';
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {columns.map((col) => (
              <th
                key={col.key}
                className={`px-2 py-2 font-medium cursor-pointer hover:opacity-80 transition-opacity whitespace-nowrap ${
                  col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                }`}
                style={{ color: 'var(--text-muted)' }}
                onClick={() => handleSort(col.key)}
              >
                {col.label}{sortIndicator(col.key)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((trade) => (
            <tr
              key={trade.id}
              className="hover:opacity-90 transition-opacity"
              style={{ borderBottom: '1px solid var(--border)' }}
            >
              <td className="px-2 py-2 text-center" style={{ color: 'var(--text-muted)' }}>
                {trade.id}
              </td>
              <td className="px-2 py-2 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>
                {trade.entryTime}
              </td>
              <td className="px-2 py-2 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>
                {trade.exitTime}
              </td>
              <td className="px-2 py-2 text-center">
                <span
                  className="text-[10px] font-medium"
                  style={{ color: trade.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}
                >
                  {trade.direction}
                </span>
              </td>
              <td className="px-2 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)' }}>
                ${trade.entryPrice.toFixed(2)}
              </td>
              <td className="px-2 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)' }}>
                ${trade.exitPrice.toFixed(2)}
              </td>
              <td className="px-2 py-2 text-right font-mono font-medium">
                <span style={{ color: trade.rMultiple >= 0 ? 'var(--green)' : 'var(--red)' }}>
                  {trade.rMultiple >= 0 ? '+' : ''}{trade.rMultiple.toFixed(2)}
                </span>
              </td>
              <td className="px-2 py-2 text-center">
                <ExecBadge type={trade.execType} />
              </td>
              <td className="px-2 py-2 text-center">
                <ExitReasonBadge reason={trade.exitReason} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AdvancedAnalysis() {
  const [expanded, setExpanded] = useState(false);

  // {{advanced_analysis}} — not wired yet. Needs backtest API to compute these.
  const winStreaks: number[] = [];
  const lossStreaks: number[] = [];
  const todPerformance: { hour: string; trades: number; winRate: number; avgR: number }[] = [];
  const dowPerformance: { day: string; trades: number; winRate: number; avgR: number }[] = [];

  return (
    <Card>
      <button
        className="w-full flex items-center justify-between text-sm font-medium"
        style={{ color: 'var(--text-secondary)' }}
        onClick={() => setExpanded(!expanded)}
      >
        <span>Advanced Analysis</span>
        <span
          className="text-xs transition-transform"
          style={{
            color: 'var(--text-muted)',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        >
          v
        </span>
      </button>

      {expanded && (
        <div className="mt-4 space-y-6">
          {/* Win/Loss Streaks */}
          <div>
            <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Win/Loss Streaks
            </h4>
            <div className="flex gap-4">
              <div className="flex-1">
                <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                  Win Streaks
                </div>
                <div className="flex items-end gap-1" style={{ height: 60 }}>
                  {winStreaks.map((s, i) => (
                    <div
                      key={i}
                      className="flex-1 rounded-t"
                      style={{
                        height: `${(s / Math.max(...winStreaks)) * 100}%`,
                        background: 'var(--green)',
                        opacity: 0.6 + (s / Math.max(...winStreaks)) * 0.4,
                        minHeight: 4,
                      }}
                    />
                  ))}
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Max: {Math.max(...winStreaks)}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Avg: {(winStreaks.reduce((a, b) => a + b, 0) / winStreaks.length).toFixed(1)}
                  </span>
                </div>
              </div>
              <div className="flex-1">
                <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>
                  Loss Streaks
                </div>
                <div className="flex items-end gap-1" style={{ height: 60 }}>
                  {lossStreaks.map((s, i) => (
                    <div
                      key={i}
                      className="flex-1 rounded-t"
                      style={{
                        height: `${(s / Math.max(...lossStreaks)) * 100}%`,
                        background: 'var(--red)',
                        opacity: 0.6 + (s / Math.max(...lossStreaks)) * 0.4,
                        minHeight: 4,
                      }}
                    />
                  ))}
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Max: {Math.max(...lossStreaks)}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    Avg: {(lossStreaks.reduce((a, b) => a + b, 0) / lossStreaks.length).toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Time of Day Performance */}
          <div>
            <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Time of Day Performance
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <th className="text-left px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Hour</th>
                    <th className="text-center px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Trades</th>
                    <th className="text-center px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Win Rate</th>
                    <th className="text-right px-2 py-1.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg R</th>
                  </tr>
                </thead>
                <tbody>
                  {todPerformance.map((row) => (
                    <tr key={row.hour} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="px-2 py-1.5" style={{ color: 'var(--text-secondary)' }}>{row.hour}</td>
                      <td className="px-2 py-1.5 text-center" style={{ color: 'var(--text-secondary)' }}>{row.trades}</td>
                      <td className="px-2 py-1.5 text-center">
                        <span style={{ color: row.winRate > 55 ? 'var(--green)' : row.winRate < 50 ? 'var(--red)' : 'var(--text-secondary)' }}>
                          {row.winRate.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-2 py-1.5 text-right font-mono">
                        <span style={{ color: row.avgR > 0 ? 'var(--green)' : 'var(--red)' }}>
                          {row.avgR >= 0 ? '+' : ''}{row.avgR.toFixed(2)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Day of Week Performance */}
          <div>
            <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Day of Week Performance
            </h4>
            <div className="grid grid-cols-5 gap-2">
              {dowPerformance.map((row) => (
                <div
                  key={row.day}
                  className="rounded-lg border p-3 text-center"
                  style={{
                    background: 'var(--bg-input)',
                    borderColor: row.avgR > 0.3 ? 'var(--green)' : row.avgR < 0 ? 'var(--red)' : 'var(--border)',
                    borderWidth: row.avgR > 0.3 || row.avgR < 0 ? 2 : 1,
                  }}
                >
                  <div className="text-[10px] font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                    {row.day.slice(0, 3)}
                  </div>
                  <div className="text-sm font-semibold" style={{ color: row.avgR > 0 ? 'var(--green)' : 'var(--red)' }}>
                    {row.avgR >= 0 ? '+' : ''}{row.avgR.toFixed(2)}R
                  </div>
                  <div className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                    {row.trades} trades | {row.winRate.toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Markov Motor — {{markov_transitions}} not wired yet */}
          <div>
            <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-muted)' }}>
              Markov Motor Analysis
            </h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border p-3" style={{ background: 'var(--bg-input)', borderColor: 'var(--border)' }}>
                <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>After Win</div>
                <div className="flex justify-between text-xs mb-1">
                  <span style={{ color: 'var(--text-secondary)' }}>P(Win)</span>
                  <span style={{ color: 'var(--text-muted)' }}>{'{{win_after_win}}'}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span style={{ color: 'var(--text-secondary)' }}>P(Loss)</span>
                  <span style={{ color: 'var(--text-muted)' }}>{'{{loss_after_win}}'}</span>
                </div>
                <div className="mt-2 h-2 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                  <div className="h-full rounded-full" style={{ width: '50%', background: 'var(--text-muted)', opacity: 0.3 }} />
                </div>
              </div>
              <div className="rounded-lg border p-3" style={{ background: 'var(--bg-input)', borderColor: 'var(--border)' }}>
                <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>After Loss</div>
                <div className="flex justify-between text-xs mb-1">
                  <span style={{ color: 'var(--text-secondary)' }}>P(Win)</span>
                  <span style={{ color: 'var(--text-muted)' }}>{'{{win_after_loss}}'}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span style={{ color: 'var(--text-secondary)' }}>P(Loss)</span>
                  <span style={{ color: 'var(--text-muted)' }}>{'{{loss_after_loss}}'}</span>
                </div>
                <div className="mt-2 h-2 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
                  <div className="h-full rounded-full" style={{ width: '50%', background: 'var(--text-muted)', opacity: 0.3 }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Analyzer Drill-Down Tab Content
// ---------------------------------------------------------------------------

interface AnalyzerResult {
  triggerId: string;
  triggerName: string;
  execType: string;
  totalTrades: number;
  profitFactor: number;
  winRate: number;
  avgR: number;
  dailyR: number;
  rSquared: number;
}

// Empty analysis results — populated by the Analyze button (TODO: wire to backtest API)
const EMPTY_ENTRY_ANALYSIS: AnalyzerResult[] = [];
const EMPTY_EXIT_ANALYSIS: AnalyzerResult[] = [];

function AnalyzerResultCard({
  result,
  isCurrent,
  actionLabel,
  onAction,
}: {
  result: AnalyzerResult;
  isCurrent: boolean;
  actionLabel: string;
  onAction?: () => void;
}) {
  return (
    <div
      className="rounded-lg border p-3"
      style={{
        background: isCurrent ? 'var(--accent-muted)' : 'var(--bg-card)',
        borderColor: isCurrent ? 'var(--accent)' : 'var(--border)',
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <ExecBadge type={result.execType} />
          <span
            className={`text-sm ${isCurrent ? 'font-semibold' : ''}`}
            style={{ color: isCurrent ? 'var(--accent)' : 'var(--text-primary)' }}
          >
            {result.triggerName}
          </span>
          {isCurrent && (
            <span className="text-[10px] italic" style={{ color: 'var(--accent)' }}>
              (current)
            </span>
          )}
        </div>
        {!isCurrent && onAction && (
          <SecondaryButton onClick={onAction}>{actionLabel}</SecondaryButton>
        )}
      </div>
      <div className="grid grid-cols-6 gap-2">
        <div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Trades</span>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{result.totalTrades}</div>
        </div>
        <div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>PF</span>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{result.profitFactor.toFixed(1)}</div>
        </div>
        <div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>WR</span>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{result.winRate.toFixed(1)}%</div>
        </div>
        <div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Avg R</span>
          <div className="text-xs" style={{ color: result.avgR > 0 ? 'var(--green)' : 'var(--red)' }}>
            {result.avgR >= 0 ? '+' : ''}{result.avgR.toFixed(2)}
          </div>
        </div>
        <div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Daily R</span>
          <div className="text-xs" style={{ color: result.dailyR > 0 ? 'var(--green)' : 'var(--red)' }}>
            {result.dailyR >= 0 ? '+' : ''}{result.dailyR.toFixed(2)}
          </div>
        </div>
        <div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>R-sq</span>
          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{result.rSquared.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Depth Selector (button group for confluence depth)
// ---------------------------------------------------------------------------

function DepthSelector({ depth, maxDepth, onChange }: { depth: number; maxDepth: number; onChange: (d: number) => void }) {
  return (
    <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Confluence Depth:</span>
      <div className="flex gap-1">
        {Array.from({ length: maxDepth }, (_, i) => i + 1).map((n) => (
          <button
            key={n}
            onClick={() => onChange(n)}
            className="w-7 h-7 rounded text-xs font-medium transition-colors"
            style={{
              background: n === depth ? 'var(--accent)' : 'var(--bg-input)',
              color: n === depth ? 'white' : 'var(--text-muted)',
              border: n === depth ? 'none' : '1px solid var(--border)',
            }}
          >
            {n}
          </button>
        ))}
      </div>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
        {depth === 1 ? 'Individual' : `Combinations of up to ${depth}`}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Filter & Sort Modal
// ---------------------------------------------------------------------------

interface FilterConfig {
  sortBy: string;
  sortDir: 'asc' | 'desc';
  minPF: string;
  minWR: string;
  minTrades: string;
  minR2: string;
  minDailyR: string;
  minAvgR: string;
  tqFilter: string;
}

const defaultFilters: FilterConfig = { sortBy: 'profitFactor', sortDir: 'desc', minPF: '', minWR: '', minTrades: '', minR2: '', minDailyR: '', minAvgR: '', tqFilter: 'None' };

function applyAnalysisFilters(results: AnalyzeResult[], filters: FilterConfig, search?: string): AnalyzeResult[] {
  let filtered = [...results];
  // Text search
  if (search) {
    const q = search.toLowerCase();
    filtered = filtered.filter(r => r.trigger_name.toLowerCase().includes(q));
  }
  // KPI thresholds
  const minPF = parseFloat(filters.minPF); if (!isNaN(minPF)) filtered = filtered.filter(r => r.profit_factor >= minPF);
  const minWR = parseFloat(filters.minWR); if (!isNaN(minWR)) filtered = filtered.filter(r => r.win_rate >= minWR);
  const minTrades = parseInt(filters.minTrades); if (!isNaN(minTrades)) filtered = filtered.filter(r => r.total_trades >= minTrades);
  const minR2 = parseFloat(filters.minR2); if (!isNaN(minR2)) filtered = filtered.filter(r => r.r_squared >= minR2);
  const minDR = parseFloat(filters.minDailyR); if (!isNaN(minDR)) filtered = filtered.filter(r => r.daily_r >= minDR);
  const minAR = parseFloat(filters.minAvgR); if (!isNaN(minAR)) filtered = filtered.filter(r => r.avg_r >= minAR);
  // Sort
  const keyMap: Record<string, keyof AnalyzeResult> = {
    profitFactor: 'profit_factor', winRate: 'win_rate', trades: 'total_trades',
    rSquared: 'r_squared', dailyR: 'daily_r', avgR: 'avg_r',
  };
  const sortKey = keyMap[filters.sortBy] || 'daily_r';
  const dir = filters.sortDir === 'asc' ? 1 : -1;
  filtered.sort((a, b) => ((a[sortKey] as number) - (b[sortKey] as number)) * dir);
  return filtered;
}

function FilterSortModal({ isOpen, onClose, filters, onApply }: { isOpen: boolean; onClose: () => void; filters: FilterConfig; onApply: (f: FilterConfig) => void }) {
  const [local, setLocal] = useState<FilterConfig>(filters);

  return (
    <Modal title="Filter & Sort" isOpen={isOpen} onClose={onClose} width="400px">
      <div className="space-y-5">
        {/* Sort */}
        <div>
          <label className="text-xs block mb-1.5" style={{ color: 'var(--text-secondary)' }}>Sort by</label>
          <select
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            value={local.sortBy}
            onChange={(e) => setLocal({ ...local, sortBy: e.target.value })}
          >
            <option value="profitFactor">Profit Factor</option>
            <option value="winRate">Win Rate</option>
            <option value="avgR">Avg R</option>
            <option value="dailyR">Daily R</option>
            <option value="rSquared">R-Squared</option>
            <option value="totalTrades">Total Trades</option>
          </select>
        </div>
        <div className="flex gap-3">
          {(['asc', 'desc'] as const).map((dir) => (
            <button
              key={dir}
              onClick={() => setLocal({ ...local, sortDir: dir })}
              className="flex-1 py-2 rounded-lg text-xs font-medium"
              style={{
                background: local.sortDir === dir ? 'var(--accent-muted)' : 'var(--bg-input)',
                color: local.sortDir === dir ? 'var(--accent)' : 'var(--text-muted)',
                border: `1px solid ${local.sortDir === dir ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              {dir === 'asc' ? 'Ascending' : 'Descending'}
            </button>
          ))}
        </div>

        {/* Minimum Thresholds */}
        <div>
          <label className="text-xs block mb-2" style={{ color: 'var(--text-secondary)' }}>Minimum Thresholds</label>
          <div className="grid grid-cols-3 gap-3">
            {[
              { key: 'minPF', label: 'Min PF', placeholder: 'e.g. 1.5' },
              { key: 'minWR', label: 'Min WR %', placeholder: 'e.g. 50' },
              { key: 'minTrades', label: 'Min Trades', placeholder: 'e.g. 20' },
              { key: 'minR2', label: 'Min R\u00b2', placeholder: 'e.g. 0.7' },
              { key: 'minDailyR', label: 'Min Daily R', placeholder: 'e.g. 0.2' },
              { key: 'minAvgR', label: 'Min Avg R', placeholder: 'e.g. 0.1' },
            ].map((f) => (
              <div key={f.key}>
                <label className="text-[10px] block mb-1" style={{ color: 'var(--text-muted)' }}>{f.label}</label>
                <input
                  className="w-full px-2 py-1.5 rounded text-xs font-mono"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  placeholder={f.placeholder}
                  value={local[f.key as keyof FilterConfig]}
                  onChange={(e) => setLocal({ ...local, [f.key]: e.target.value })}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Trade Qualification Filter */}
        <div>
          <label className="text-xs block mb-1.5" style={{ color: 'var(--text-secondary)' }}>Trade Qualification</label>
          <select
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            value={local.tqFilter || 'None'}
            onChange={(e) => setLocal({ ...local, tqFilter: e.target.value })}
          >
            <option value="None">None (all trades count)</option>
            <option value="ttp">Trade The Pool — $50k</option>
            <option value="ftmo">FTMO — $100k Challenge</option>
            <option value="topstep">Topstep — $150k Combine</option>
            <option value="custom">My Custom Rules</option>
          </select>
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
            Applies trade qualification rules from the selected requirement set. See Portfolio Requirements for details on how each rule affects wins, losses, or all trades.
          </p>
        </div>

        <button
          onClick={() => { onApply(local); onClose(); }}
          className="w-full py-2.5 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          Apply Filters
        </button>
      </div>
    </Modal>
  );
}

// ---------------------------------------------------------------------------
// Analysis Toolbar (Search + Analyze + Filter button)
// ---------------------------------------------------------------------------

function AnalyzeProgressBar({ label }: { label: string }) {
  return (
    <div className="py-8 px-4">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{label}</span>
      </div>
      <div className="w-full rounded-full overflow-hidden" style={{ height: 4, background: 'var(--bg-input)' }}>
        <div className="h-full rounded-full" style={{
          background: 'var(--accent)',
          width: '100%',
          animation: 'analyzeProgress 2s ease-in-out infinite',
        }} />
      </div>
      <style>{`
        @keyframes analyzeProgress {
          0% { width: 5%; opacity: 0.6; }
          50% { width: 80%; opacity: 1; }
          100% { width: 5%; opacity: 0.6; }
        }
      `}</style>
    </div>
  );
}

function AnalysisToolbar({ search, onSearch, onAnalyze, onFilterClick, placeholder }: {
  search: string; onSearch: (v: string) => void; onAnalyze: () => void; onFilterClick: () => void; placeholder: string;
}) {
  return (
    <div className="flex gap-2 mb-3">
      <div className="flex-1">
        <TextInput value={search} onChange={onSearch} placeholder={placeholder} />
      </div>
      <PrimaryButton onClick={onAnalyze}>Analyze</PrimaryButton>
      <button
        onClick={onFilterClick}
        className="px-2.5 py-2 rounded-lg transition-colors"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
        title="Filter & Sort"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="4" y1="6" x2="20" y2="6" />
          <line x1="7" y1="12" x2="17" y2="12" />
          <line x1="10" y1="18" x2="14" y2="18" />
        </svg>
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export interface BacktestResult {
  trades: TradeRow[];
  kpis: KPIs;
  equityCurve: { trade_number: number; timestamp: string; cumulative_r: number }[];
  chartData?: any[];
  totalBars: number;
  dataSource: string;
}

export interface StrategyBuilderV5Props {
  /** Real backtest result from API (overrides mock data when present) */
  backtestResult?: BacktestResult | null;
  /** Called when user clicks "Run Backtest" */
  onRunBacktest?: (config: Record<string, any>) => void;
  /** Called when user clicks "Save Strategy" */
  onSave?: (strategy: Record<string, any>) => void;
  /** Whether a backtest is currently running */
  isBacktesting?: boolean;
  /** Real stop loss packs from API */
  stopPacks?: RiskPack[];
  /** Real target packs from API */
  targetPacks?: RiskPack[];
}

export default function StrategyBuilderPage() {
  const router = useRouter();

  // ---- Chart prefs for shared chart rendering ----
  const chartPrefs = useChartPrefs();

  // ---- API Hooks ----
  const backtestMut = useRunBacktest();
  const analyzeMut = useAnalyzeTriggers();
  const createMut = useCreateStrategy();
  const { data: apiEntryTriggers } = useConfluenceTriggers('LONG');
  const { data: apiExitTriggers } = useConfluenceTriggers('EXIT');
  const { data: apiConfluenceGroups } = useConfluenceGroups();
  const { data: apiGeneralPacks } = useGeneralPacks();
  const { data: apiRmPacks } = useRiskManagementPacks();

  // Derive trigger/pack/condition lists from API data
  const API_TRIGGERS: TriggerDef[] = useMemo(() => {
    if (!apiEntryTriggers && !apiExitTriggers) return [];
    const triggers: TriggerDef[] = [];
    const deriveExecType = (id: string): 'C' | 'L' | 'LC' | 'CC' => {
      if (id.endsWith('_lc')) return 'LC';
      if (id.endsWith('_cc')) return 'CC';
      if (id.endsWith('_ib')) return 'L';
      if (id.endsWith('_hm')) return 'LC'; // HM is functionally LC with same-bar confirm
      if (id.endsWith('_hl')) return 'LC'; // HL is functionally LC with limit bail
      return 'C';
    };
    for (const [id, name] of Object.entries(apiEntryTriggers || {})) {
      triggers.push({ id, name: String(name), execType: deriveExecType(id), pack: id.split('_')[0] || 'unknown' });
    }
    for (const [id, name] of Object.entries(apiExitTriggers || {})) {
      if (!triggers.find(t => t.id === id)) {
        triggers.push({ id, name: String(name), execType: deriveExecType(id), pack: id.split('_')[0] || 'unknown' });
      }
    }
    return triggers.length > 0 ? triggers : API_TRIGGERS;
  }, [apiEntryTriggers, apiExitTriggers]);

  const API_STOP_PACKS: RiskPack[] = useMemo(() => {
    if (!apiRmPacks) return [];
    return apiRmPacks.map((p: any) => ({
      id: p.id, name: `${p.base_template}`, version: p.version || 'Default',
      summary: Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version,
    }));
  }, [apiRmPacks]);

  const API_TARGET_PACKS: RiskPack[] = useMemo(() => {
    if (!apiRmPacks) return [];
    // All RM packs are potential target packs — the backend analyze endpoint
    // filters to only those with valid target configs
    return apiRmPacks.map((p: any) => ({
      id: p.id, name: `${p.base_template}`, version: p.version || 'Default',
      summary: Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version,
    }));
  }, [apiRmPacks]);


  const API_CONFLUENCE_CONDITIONS: ConfluenceCondition[] = useMemo(() => {
    if (!apiConfluenceGroups) return [];
    const conds: ConfluenceCondition[] = [];
    for (const g of apiConfluenceGroups) {
      // Generate conditions from group outputs (e.g., "1M-EMA_STACK_DEFAULT-SML")
      conds.push({ id: `1M-${g.id.toUpperCase()}-BULL`, label: `${g.base_template} (${g.version}) — Bullish`, pack: g.base_template });
      conds.push({ id: `1M-${g.id.toUpperCase()}-BEAR`, label: `${g.base_template} (${g.version}) — Bearish`, pack: g.base_template });
    }
    return conds.length > 0 ? conds : API_CONFLUENCE_CONDITIONS;
  }, [apiConfluenceGroups]);

  const API_GENERAL_CONDITIONS: ConfluenceCondition[] = useMemo(() => {
    if (!apiGeneralPacks) return [];
    return apiGeneralPacks.map((p: any) => ({
      id: `GEN-${p.id.toUpperCase()}-IN`,
      label: `${p.base_template} (${p.version}) — In Window`,
      pack: p.base_template,
    }));
  }, [apiGeneralPacks]);

  // Backtest result from API mutation
  const backtestResult: BacktestResult | null = useMemo(() => {
    if (!backtestMut.data) return null;
    const d = backtestMut.data;
    return {
      trades: (d.trades || []).map((t: any, i: number) => ({
        id: i + 1,
        entryTime: t.entry_time ? new Date(t.entry_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
        exitTime: t.exit_time ? new Date(t.exit_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
        direction: (t.direction || 'LONG') as 'LONG' | 'SHORT',
        entryPrice: t.entry_price || 0, exitPrice: t.exit_price || 0,
        rMultiple: t.r_multiple || 0, execType: ({'HM': 'LC', 'HL': 'LC', 'L0': 'L', 'L1': 'L'} as Record<string,string>)[t.exec_type] || t.exec_type || 'C',
        exitReason: t.exit_reason || '', confluences: '',
      })),
      kpis: {
        winRate: d.kpis?.win_rate ?? 0, profitFactor: d.kpis?.profit_factor ?? 0,
        dailyR: d.kpis?.daily_r ?? 0, totalTrades: d.kpis?.total_trades ?? 0,
        avgWin: d.secondary_kpis?.avg_win_r ?? 0, avgLoss: d.secondary_kpis?.avg_loss_r ?? 0,
        maxDrawdown: d.kpis?.max_r_drawdown ?? 0, recoveryFactor: d.secondary_kpis?.recovery_factor ?? 0,
        sharpeRatio: d.secondary_kpis?.sharpe_ratio ?? 0, expectancy: d.kpis?.avg_r ?? 0,
        totalR: d.kpis?.total_r ?? 0, avgR: d.kpis?.avg_r ?? 0,
        rSquared: d.kpis?.r_squared ?? 0, maxRDrawdown: d.kpis?.max_r_drawdown ?? 0,
      },
      equityCurve: d.equity_curve || [],
      chartData: d.chart_data, totalBars: d.total_bars || 0, dataSource: d.data_source || '',
      overlay_indicators: d.overlay_indicators || [],
      oscillator_indicators: d.oscillator_indicators || [],
      heatmap_conditions: d.heatmap_conditions || [],
      rawTrades: d.trades || [],  // Raw trade data with ISO timestamps for chart rendering
    };
  }, [backtestMut.data]);

  const isBacktesting = backtestMut.isPending;
  const onRunBacktest = useCallback((config: Record<string, any>) => {
    backtestMut.mutate({
      symbol: config.symbol, timeframe: config.timeframe, direction: config.direction,
      days: config.days, session: config.session,
      entry_trigger_confluence_id: config.entry_trigger_confluence_id,
      exit_trigger_confluence_ids: config.exit_trigger_confluence_ids || [],
      confluence: config.confluence || [],
      stop_loss_pack_id: config.stop_loss_pack_id, take_profit_pack_id: config.take_profit_pack_id,
      bar_count_exit: config.bar_count_exit,
      secondary_tfs: config.secondary_tfs || [],
      include_chart_data: true,
    });
  }, [backtestMut]);

  const onSave = useCallback((strategy: Record<string, any>) => {
    createMut.mutate(strategy, { onSuccess: () => router.push('/strategies') });
  }, [createMut, router]);

  // ---- Config State ----
  const [symbol, setSymbol] = useState('SPY');
  const [strategyMethod, setStrategyMethod] = useState<'standard' | 'webhook' | 'scanner'>('standard');
  const [assetType, setAssetType] = useState('Equity');
  const [direction, setDirection] = useState('LONG');
  const [timeframe, setTimeframe] = useState('5Min');
  const [session, setSession] = useState('RTH');
  const [lookbackMode, setLookbackMode] = useState('Days');
  const [lookbackDays, setLookbackDays] = useState(30);
  const [lookbackBars, setLookbackBars] = useState(1000);
  const [strategyName, setStrategyName] = useState('SPY LONG - 1');

  // ---- Entry/Exit ----
  const [entryTrigger, setEntryTrigger] = useState('');
  const [exitTriggers, setExitTriggers] = useState<string[]>([]);
  const [triggerSearch, setTriggerSearch] = useState('');

  // ---- Stop Loss (pack selection) ----
  const [selectedStopPack, setSelectedStopPack] = useState('');
  const [eqXAxis, setEqXAxis] = useState<'trade' | 'time'>('trade');

  // ---- Target (pack selection) ----
  const [selectedTargetPack, setSelectedTargetPack] = useState('');

  // ---- Confluence ----
  const [selectedConditions, setSelectedConditions] = useState<Set<string>>(new Set());
  const [selectedGenerals, setSelectedGenerals] = useState<Set<string>>(new Set());
  const [confluenceModalOpen, setConfluenceModalOpen] = useState(false);
  const [generalModalOpen, setGeneralModalOpen] = useState(false);
  // ---- Depth selectors per tab ----
  const [entryDepth] = useState(1);
  const [exitDepth, setExitDepth] = useState(1);
  const [tfDepth, setTfDepth] = useState(1);
  const [generalDepth, setGeneralDepth] = useState(1);
  const [stopDepth] = useState(1);
  const [targetDepth] = useState(1);

  // ---- Filter & Sort ----
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [activeAnalysisTab, setActiveAnalysisTab] = useState('Entry');
  const [filters, setFilters] = useState<FilterConfig>(defaultFilters);

  // ---- Backtest State ----
  const [backtestRan, setBacktestRan] = useState(false);
  const [configExpanded, setConfigExpanded] = useState(false);

  // ---- Analysis State ----
  const [entryAnalysisRan, setEntryAnalysisRan] = useState(false);
  const [exitAnalysisRan, setExitAnalysisRan] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<Record<string, AnalyzeResult[]>>({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzingMode, setAnalyzingMode] = useState('');

  const handleAnalyze = useCallback((mode: string = 'entry') => {
    if (!entryTrigger || isAnalyzing) return;
    setIsAnalyzing(true);
    setAnalyzingMode(mode);
    const analyzeDepth = mode === 'combinations' ? tfDepth : mode === 'general_combinations' ? generalDepth : undefined;
    analyzeMut.mutate({
      mode,
      depth: analyzeDepth,
      symbol, timeframe, direction: direction as 'LONG' | 'SHORT',
      days: lookbackDays, session, data_feed: 'sip',
      entry_trigger_confluence_id: entryTrigger,
      exit_trigger_confluence_ids: exitTriggers,
      confluence: Array.from(selectedConditions),
      stop_loss_pack_id: selectedStopPack || undefined,
      take_profit_pack_id: selectedTargetPack || undefined,
      stop_atr_mult: 1.5,
      lookback_mode: lookbackMode,
      bar_count_exit: exitTriggers.some(t => t.includes('bar_count')) ? 4 : undefined,
      secondary_tfs: secondaryTfs,
    } as any, {
      onSuccess: (data) => {
        setAnalysisResults(prev => ({ ...prev, [mode]: data.results || [] }));
        if (mode === 'entry') setEntryAnalysisRan(true);
        if (mode === 'exit') setExitAnalysisRan(true);
        setIsAnalyzing(false);
        setAnalyzingMode('');
      },
      onError: () => {
        setIsAnalyzing(false);
        setAnalyzingMode('');
      },
    });
  }, [analyzeMut, symbol, timeframe, direction, lookbackDays, session, lookbackMode,
      entryTrigger, exitTriggers, selectedConditions, selectedStopPack, selectedTargetPack, isAnalyzing]);
  const [analyzerSearch, setAnalyzerSearch] = useState('');

  // Compute derived values
  const estimatedBars = useMemo(() => {
    const barsPerDay = timeframe === '1Min' ? 390 : timeframe === '5Min' ? 78 : timeframe === '15Min' ? 26 : timeframe === '1H' ? 7 : timeframe === '4H' ? 2 : 1;
    const sessionMult = session === 'RTH' ? 1 : session === 'Extended' ? 2.4 : session === '24/7' ? 3.7 : 1.5;
    if (lookbackMode === 'Bars/Candles') return lookbackBars;
    return Math.round(lookbackDays * barsPerDay * sessionMult);
  }, [lookbackDays, lookbackBars, timeframe, session, lookbackMode]);

  // Derive secondary timeframes: load ALL timeframes from enabled TF packs
  // (not just selected conditions — we need cross-TF data to DISCOVER conditions)
  const secondaryTfs = useMemo(() => {
    const tfMap: Record<string, string> = {
      '1M': '1Min', '2M': '2Min', '3M': '3Min', '5M': '5Min', '10M': '10Min',
      '15M': '15Min', '30M': '30Min', '1H': '1Hour', '4H': '4Hour', '1D': '1Day',
    };
    const primaryLabel = timeframe === '1Min' ? '1M' : timeframe === '5Min' ? '5M' :
      timeframe === '15Min' ? '15M' : timeframe === '1Hour' ? '1H' : timeframe === '1Day' ? '1D' : '1M';

    // Always include standard cross-TF timeframes that are coarser than primary
    const standardTfs = ['5M', '15M', '1H', '1D'];
    const tfs = new Set<string>();
    for (const tf of standardTfs) {
      if (tf !== primaryLabel) {
        const mapped = tfMap[tf];
        if (mapped) tfs.add(mapped);
      }
    }
    // Also add any from selected conditions
    for (const cond of selectedConditions) {
      const parts = cond.split('-');
      if (parts.length >= 2 && parts[0] !== 'GEN' && parts[0] !== primaryLabel) {
        const mapped = tfMap[parts[0]];
        if (mapped) tfs.add(mapped);
      }
    }
    return Array.from(tfs);
  }, [selectedConditions, timeframe]);

  // Use API packs when provided, fall back to mocks
  const activeStopPacks = API_STOP_PACKS;
  const activeTargetPacks = API_TARGET_PACKS;

  const entryDef = API_TRIGGERS.find((t) => t.id === entryTrigger);

  const exitDefs = useMemo(
    () => exitTriggers.map((eid) => API_TRIGGERS.find((t) => t.id === eid)).filter(Boolean) as TriggerDef[],
    [exitTriggers]
  );

  // Handlers
  const handleRunBacktest = useCallback(() => {
    setBacktestRan(true);
    if (onRunBacktest) {
      onRunBacktest({
        symbol, timeframe, direction, session,
        days: lookbackDays,
        lookback_mode: lookbackMode,
        entry_trigger_confluence_id: entryTrigger,
        exit_trigger_confluence_ids: exitTriggers,
        confluence: Array.from(selectedConditions),
        stop_loss_pack_id: selectedStopPack,
        take_profit_pack_id: selectedTargetPack,
        // Extract bar_count_exit if a bar count exit trigger is selected
        bar_count_exit: exitTriggers.some(t => t.includes('bar_count')) ? 4 : undefined,
        secondary_tfs: secondaryTfs,
      });
    }
  }, [onRunBacktest, symbol, timeframe, direction, session, lookbackDays, lookbackMode, entryTrigger, exitTriggers, selectedConditions, selectedStopPack, selectedTargetPack, secondaryTfs]);

  const handleToggleCondition = useCallback((id: string) => {
    setSelectedConditions((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleToggleGeneral = useCallback((id: string) => {
    setSelectedGenerals((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleRemoveExit = useCallback((idx: number) => {
    setExitTriggers((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const allSelected = useMemo(
    () => new Set(Array.from(selectedConditions).concat(Array.from(selectedGenerals))),
    [selectedConditions, selectedGenerals]
  );

  // ---------------------------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------------------------

  return (
    <div>
      {/* ================================================================= */}
      {/* STRATEGY METHOD SELECTOR                                          */}
      {/* ================================================================= */}
      <Card className="mb-4">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Strategy Method:</span>
          <div className="flex gap-2">
            {([
              { id: 'standard' as const, label: 'Standard', desc: 'Ticker-specific backtesting and alerts', available: true },
              { id: 'webhook' as const, label: 'Inbound Webhook', desc: 'Fires on incoming webhook signals', available: false },
              { id: 'scanner' as const, label: 'Scanner', desc: 'Scans across a collection of tickers', available: false },
            ]).map((method) => (
              <button
                key={method.id}
                onClick={() => method.available && setStrategyMethod(method.id)}
                className="px-4 py-2 rounded-lg text-sm font-medium transition-colors relative"
                style={{
                  background: strategyMethod === method.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                  color: strategyMethod === method.id ? 'var(--accent)' : method.available ? 'var(--text-primary)' : 'var(--text-muted)',
                  border: strategyMethod === method.id ? '1px solid var(--accent)' : '1px solid var(--border)',
                  cursor: method.available ? 'pointer' : 'not-allowed',
                  opacity: method.available ? 1 : 0.6,
                }}
                title={method.desc}
              >
                {method.label}
                {!method.available && (
                  <span className="ml-1.5 text-[9px] px-1 py-0.5 rounded" style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                    Soon
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* ================================================================= */}
      {/* TOP ROW: Core config fields in a horizontal bar                   */}
      {/* ================================================================= */}
      <Card className="mb-4">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-10 gap-3">
          {/* Asset Type */}
          <div>
            <SectionLabel>Asset</SectionLabel>
            <SelectInput options={ASSET_TYPES} value={assetType} onChange={(v) => {
              setAssetType(v);
              if (v === 'Crypto') {
                setSession('24/7');
                if (symbol === 'SPY') setSymbol('BTC/USD');
              } else {
                setSession('RTH');
                if (symbol === 'BTC/USD') setSymbol('SPY');
              }
            }} />
          </div>

          {/* Ticker */}
          <div>
            <SectionLabel>Ticker</SectionLabel>
            <TextInput
              value={symbol}
              onChange={(v) => setSymbol(v.toUpperCase())}
              placeholder={assetType === 'Crypto' ? 'BTC/USD' : 'SPY'}
            />
          </div>

          {/* Timeframe */}
          <div>
            <SectionLabel>Timeframe</SectionLabel>
            <SelectInput options={TIMEFRAMES} value={timeframe} onChange={setTimeframe} />
          </div>

          {/* Direction */}
          <div>
            <SectionLabel>Direction</SectionLabel>
            <SelectInput options={DIRECTIONS} value={direction} onChange={setDirection} />
          </div>

          {/* Session */}
          <div>
            <SectionLabel>Session</SectionLabel>
            {assetType === 'Crypto' ? (
              <input
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-muted)',
                }}
                value="24/7"
                disabled
              />
            ) : (
              <SelectInput options={SESSIONS} value={session} onChange={setSession} />
            )}
          </div>

          {/* Lookback Mode */}
          <div>
            <SectionLabel>Lookback</SectionLabel>
            <SelectInput options={LOOKBACK_MODES} value={lookbackMode} onChange={setLookbackMode} />
          </div>

          {/* Lookback Value */}
          <div>
            <SectionLabel>{lookbackMode === 'Bars/Candles' ? 'Bars' : 'Days'}</SectionLabel>
            <NumberInput
              value={lookbackMode === 'Bars/Candles' ? lookbackBars : lookbackDays}
              onChange={lookbackMode === 'Bars/Candles' ? setLookbackBars : setLookbackDays}
              min={lookbackMode === 'Bars/Candles' ? 100 : 7}
              max={lookbackMode === 'Bars/Candles' ? 500000 : 1825}
              step={lookbackMode === 'Bars/Candles' ? 100 : 7}
            />
          </div>

          {/* Strategy Name */}
          <div className="xl:col-span-2">
            <SectionLabel>Name</SectionLabel>
            <TextInput value={strategyName} onChange={setStrategyName} />
          </div>

          {/* Load Data */}
          <div className="flex items-end">
            <PrimaryButton fullWidth onClick={handleRunBacktest}>
              {backtestRan ? 'Reload' : 'Load Data'}
            </PrimaryButton>
          </div>
        </div>
        {/* Status line */}
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            ~{estimatedBars.toLocaleString()} bars
          </span>
          {estimatedBars > 200000 && (
            <span className="text-xs" style={{ color: 'var(--red)' }}>
              Very large dataset -- may be slow
            </span>
          )}
          {estimatedBars > 50000 && estimatedBars <= 200000 && (
            <span className="text-xs" style={{ color: 'var(--orange)' }}>
              Large dataset
            </span>
          )}
        </div>
      </Card>

      {/* ================================================================= */}
      {/* STRATEGY CONFIG EXPANDER: Entry, Exit, Stop, Target               */}
      {/* ================================================================= */}
      <Card className="mb-4">
        <button
          className="w-full flex items-center justify-between text-sm font-medium"
          style={{ color: 'var(--text-secondary)' }}
          onClick={() => setConfigExpanded(!configExpanded)}
        >
          <div className="flex items-center gap-3">
            <span>Strategy Config</span>
            {/* Quick summary when collapsed */}
            {!configExpanded && entryDef && (
              <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>
                <ExecBadge type={entryDef.execType} />{' '}
                {entryDef.name} {'\u2192'}{' '}
                {exitDefs.map((e) => e.name).join(' / ') || 'No exit'}{' '}
                | Stop: {API_STOP_PACKS.find((p) => p.id === selectedStopPack)?.summary || 'None'}{' '}
                | Target: {API_TARGET_PACKS.find((p) => p.id === selectedTargetPack)?.summary || 'None'}
              </span>
            )}
          </div>
          <span
            className="text-xs transition-transform"
            style={{
              color: 'var(--text-muted)',
              transform: configExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
            }}
          >
            v
          </span>
        </button>

        {configExpanded && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Entry Trigger */}
            <div>
              <SectionLabel>Entry Trigger</SectionLabel>
              <TextInput
                value={triggerSearch}
                onChange={setTriggerSearch}
                placeholder="Search triggers..."
              />
              <div className="mt-2">
                <TriggerGroupedList
                  triggers={API_TRIGGERS.filter((t) => {
                    // Show direction-appropriate triggers
                    if (direction === 'LONG') {
                      return !t.name.toLowerCase().includes('down') &&
                             !t.name.toLowerCase().includes('bear') &&
                             !t.name.toLowerCase().includes('sell') &&
                             !t.name.toLowerCase().includes('short') &&
                             !t.name.toLowerCase().includes('overbought');
                    }
                    return !t.name.toLowerCase().includes('up') &&
                           !t.name.toLowerCase().includes('bull') &&
                           !t.name.toLowerCase().includes('buy') &&
                           !t.name.toLowerCase().includes('long') &&
                           !t.name.toLowerCase().includes('oversold');
                  })}
                  searchQuery={triggerSearch}
                  selectedId={entryTrigger}
                  onSelect={setEntryTrigger}
                />
              </div>
            </div>

            {/* Exit Trigger(s) */}
            <div>
              <SectionLabel>
                Exit Trigger(s){' '}
                <span style={{ color: 'var(--text-muted)' }}>({exitTriggers.length}/3)</span>
              </SectionLabel>
              {exitTriggers.length > 0 && (
                <div className="space-y-1 mb-2">
                  {exitTriggers.map((eid, idx) => {
                    const eDef = API_TRIGGERS.find((t) => t.id === eid);
                    return (
                      <div
                        key={eid}
                        className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs"
                        style={{
                          background: 'var(--accent-muted)',
                          border: '1px solid var(--border)',
                        }}
                      >
                        {eDef && <ExecBadge type={eDef.execType} />}
                        <span className="flex-1" style={{ color: 'var(--text-secondary)' }}>
                          {eDef?.name || eid}
                        </span>
                        <button
                          className="text-[10px] hover:opacity-70"
                          style={{ color: 'var(--text-muted)' }}
                          onClick={() => handleRemoveExit(idx)}
                        >
                          x
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
              <div className="mt-1">
                <TriggerGroupedList
                  triggers={API_TRIGGERS.filter((t) => {
                    // Show direction-appropriate exit triggers
                    if (direction === 'LONG') {
                      return !t.name.toLowerCase().includes('up') &&
                             !t.name.toLowerCase().includes('bull') &&
                             !t.name.toLowerCase().includes('buy') &&
                             !t.name.toLowerCase().includes('long') &&
                             !t.name.toLowerCase().includes('oversold');
                    }
                    return !t.name.toLowerCase().includes('down') &&
                           !t.name.toLowerCase().includes('bear') &&
                           !t.name.toLowerCase().includes('sell') &&
                           !t.name.toLowerCase().includes('short') &&
                           !t.name.toLowerCase().includes('overbought');
                  })}
                  searchQuery=""
                  selectedId=""
                  onSelect={(id) => {
                    if (exitTriggers.length < 3 && !exitTriggers.includes(id)) {
                      setExitTriggers((prev) => [...prev, id]);
                    }
                  }}
                />
              </div>
            </div>

            {/* Stop Loss — pack list (limit 1) */}
            <div>
              <SectionLabel>Stop Loss Pack</SectionLabel>
              <div className="space-y-1 overflow-y-auto pr-1" style={{ maxHeight: 240 }}>
                {API_STOP_PACKS.map((p) => (
                  <button
                    key={p.id}
                    className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
                    style={{
                      background: p.id === selectedStopPack ? 'var(--accent-muted)' : 'transparent',
                      color: p.id === selectedStopPack ? 'var(--accent)' : 'var(--text-primary)',
                      border: p.id === selectedStopPack ? '1px solid var(--accent)' : '1px solid transparent',
                    }}
                    onClick={() => setSelectedStopPack(p.id)}
                  >
                    <span className="flex-1 truncate">{p.name} ({p.version})</span>
                    <span className="text-[10px] font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{p.summary}</span>
                  </button>
                ))}
              </div>
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                <a href="/confluence-packs/stop-loss" className="underline" style={{ color: 'var(--accent)' }}>Manage stop loss packs</a>
              </p>
            </div>

            {/* Take Profit — pack list (limit 1) */}
            <div>
              <SectionLabel>Take Profit Pack</SectionLabel>
              <div className="space-y-1 overflow-y-auto pr-1" style={{ maxHeight: 240 }}>
                {API_TARGET_PACKS.map((p) => (
                  <button
                    key={p.id}
                    className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
                    style={{
                      background: p.id === selectedTargetPack ? 'var(--accent-muted)' : 'transparent',
                      color: p.id === selectedTargetPack ? 'var(--accent)' : 'var(--text-primary)',
                      border: p.id === selectedTargetPack ? '1px solid var(--accent)' : '1px solid transparent',
                    }}
                    onClick={() => setSelectedTargetPack(p.id)}
                  >
                    <span className="flex-1 truncate">{p.name} ({p.version})</span>
                    <span className="text-[10px] font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{p.summary}</span>
                  </button>
                ))}
              </div>
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                <a href="/confluence-packs/take-profit" className="underline" style={{ color: 'var(--accent)' }}>Manage take profit packs</a>
              </p>
            </div>
          </div>
        )}
      </Card>

      {/* ================================================================= */}
      {/* PRE-BACKTEST PROMPT                                               */}
      {/* ================================================================= */}
      {!backtestRan && (
        <Card className="text-center py-12">
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Select your settings above, then click <strong>Load Data</strong> to begin analysis.
          </p>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {symbol} | {direction} | {timeframe} | ~{estimatedBars.toLocaleString()} bars
          </p>
        </Card>
      )}

      {/* ================================================================= */}
      {/* POST-BACKTEST: KPIs, Charts, Analysis Tabs                        */}
      {/* ================================================================= */}
      {backtestRan && (
        <>
          {/* Strategy Summary Header */}
          <div className="mb-4">
            <h2 className="text-xl font-bold">{strategyName}</h2>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
              {symbol} | {direction} |{' '}
              {entryDef ? (
                <><ExecBadge type={entryDef.execType} /> {entryDef.name}</>
              ) : '?'}{' '}
              {'\u2192'} {exitDefs.map((e) => e.name).join(' / ') || '?'}
            </p>
          </div>

          {/* Loading progress bar */}
          {isBacktesting && (
            <div className="mb-4">
              <div className="flex items-center gap-3 mb-1">
                <div className="w-4 h-4 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  Loading market data and running backtest...
                </span>
              </div>
              <div className="w-full rounded-full overflow-hidden" style={{ height: 4, background: 'var(--bg-input)' }}>
                <div
                  className="h-full rounded-full animate-pulse"
                  style={{ background: 'var(--accent)', width: '60%', transition: 'width 2s ease' }}
                />
              </div>
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                Fetching {estimatedBars.toLocaleString()} bars from Polygon, computing indicators, running unified engine...
              </p>
            </div>
          )}

          {/* KPIs */}
          <KPIDashboard kpis={backtestResult?.kpis ?? EMPTY_KPIS} backtestRan={backtestRan || !!backtestResult} />

          {/* Optimizable Variables */}
          <div className="mt-4">
            <OptimizableVariables
              entryTrigger={entryTrigger}
              exitTriggers={exitTriggers}
              selectedConditions={selectedConditions}
              generalConditions={selectedGenerals}
              selectedStopPack={selectedStopPack}
              selectedTargetPack={selectedTargetPack}
              allTriggers={API_TRIGGERS}
              allConditions={API_CONFLUENCE_CONDITIONS}
              allGeneralConditions={API_GENERAL_CONDITIONS}
              stopPacks={API_STOP_PACKS}
              targetPacks={API_TARGET_PACKS}
              onRemoveCondition={(id) => handleToggleCondition(id)}
              onRemoveGeneral={(id) => handleToggleGeneral(id)}
              onRemoveExit={handleRemoveExit}
              onReload={handleRunBacktest}
              isLoading={isBacktesting}
            />
          </div>

          {/* Main Content: Charts + Trade History (left) | Analysis Tabs + Advanced Analysis (right) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
            {/* =========================================== */}
            {/* LEFT: Charts + Trade History                */}
            {/* =========================================== */}
            <div className="space-y-4">
              <Card className="flex flex-col">
                <TabBar tabs={['Equity Curve', 'Price Chart']}>
                  {(tab) =>
                    tab === 'Equity Curve' ? (
                      <div className="flex-1">
                        {backtestResult?.equityCurve && backtestResult.equityCurve.length > 0 ? (
                          <EquityCurve
                            data={backtestResult.equityCurve.map((pt: any, i: number) => ({
                              trade_number: i + 1,
                              cumulative_r: pt.cumulative_r ?? 0,
                              timestamp: pt.timestamp,
                            }))}
                            height={400}
                            showZeroLine
                            showHWM
                            xAxis={eqXAxis}
                          />
                        ) : (
                          <ChartPlaceholder label="Run backtest to see equity curve" height={400} />
                        )}
                        <div className="mt-2 flex gap-2 items-center">
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>X-Axis:</span>
                          {(['trade', 'time'] as const).map((mode) => (
                            <button
                              key={mode}
                              className="px-2 py-0.5 rounded text-xs"
                              style={{
                                background: eqXAxis === mode ? 'var(--accent-muted)' : 'transparent',
                                color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)',
                                border: eqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)',
                              }}
                              onClick={() => setEqXAxis(mode)}
                            >
                              {mode === 'trade' ? 'Per Trade' : 'Per Day'}
                            </button>
                          ))}
                        </div>
                        <div className="mt-1 flex gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                          <span>
                            <span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: '#2196F3' }} />
                            Equity
                          </span>
                          <span>
                            <span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'green', opacity: 0.5 }} />
                            High Water Mark
                          </span>
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="mb-2 flex items-center gap-2">
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            {backtestResult?.trades?.length ?? 0} trades on {symbol} ({direction})
                          </span>
                        </div>
                        {backtestResult?.chartData && backtestResult.chartData.length > 0 ? (
                          (() => {
                            const chartBars = backtestResult.chartData;
                            const overlays: string[] = (backtestResult as any)?.overlay_indicators || [];
                            const oscillators: string[] = (backtestResult as any)?.oscillator_indicators || [];
                            const heatmap: any[] = ((backtestResult as any)?.heatmap_conditions || []).filter((c: any) => c.has_data);
                            // Compute tfMs from timeframe for C-type shift
                            const _tfMs = (() => {
                              if (timeframe.includes('Min')) return parseInt(timeframe) * 60 * 1000;
                              if (timeframe.includes('Hour') || timeframe === '1H') return 3600 * 1000;
                              return 60000;
                            })();
                            const panes = buildStrategyChartPanes({
                              bars: chartBars,
                              trades: (backtestResult as any).rawTrades || [],
                              direction,
                              overlayNames: overlays,
                              oscNames: oscillators,
                              heatmapConds: heatmap,
                              tfMs: _tfMs,
                              chartPrefs,
                            });
                            return panes.length > 0 ? (
                              <SyncedChartPane
                                panes={panes}
                                upColor={chartPrefs.candleUp}
                                downColor={chartPrefs.candleDown}
                                upBorderColor={chartPrefs.candleUpBorder}
                                gridLines={chartPrefs.gridLines}
                                rightOffset={chartPrefs.rightOffset}
                              />
                            ) : (
                              <ChartPlaceholder label="No chart data available" height={400} />
                            );
                          })()
                        ) : (
                          <ChartPlaceholder label="Run backtest to see price chart with trade markers" height={400} />
                        )}
                      </div>
                    )
                  }
                </TabBar>
              </Card>

              {/* Trade History */}
              <Card>
                <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                  Trade History
                </h3>
                <TradeHistoryTable trades={backtestResult?.trades ?? EMPTY_TRADES} />
              </Card>
            </div>

            {/* =========================================== */}
            {/* RIGHT: Analysis Tabs + Advanced Analysis    */}
            {/* =========================================== */}
            <div className="space-y-4">
              <Card className="flex flex-col overflow-hidden">
                <div className="min-h-0 overflow-hidden">
                <TabBar tabs={['Entry', 'Exit', 'TF Conditions', 'General', 'Stop Loss', 'Take Profit']}>
                  {(tab) => {
                    // Track active tab for pinned depth selector
                    if (tab !== activeAnalysisTab) {
                      // Use a timeout to avoid setState during render
                      setTimeout(() => setActiveAnalysisTab(tab), 0);
                    }

                    // ---- ENTRY TAB ----
                    if (tab === 'Entry') {
                      return (
                        <div>
                          <AnalysisToolbar search={analyzerSearch} onSearch={setAnalyzerSearch} onAnalyze={() => handleAnalyze('entry')} onFilterClick={() => setFilterModalOpen(true)} placeholder="Search entry triggers..." />
                          {entryDef && (
                            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                              Current: <ExecBadge type={entryDef.execType} />{' '}
                              <strong>{entryDef.name}</strong>
                            </p>
                          )}
                          {isAnalyzing && analyzingMode === 'entry' ? (
                            <AnalyzeProgressBar label="Analyzing entry triggers..." />
                          ) : (analysisResults.entry?.length ?? 0) > 0 ? (
                            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 440 }}>
                              {applyAnalysisFilters(analysisResults.entry || [], filters, analyzerSearch).map((result) => (
                                <AnalyzerResultCard
                                  key={result.trigger_id}
                                  result={{
                                    triggerId: result.trigger_id,
                                    triggerName: result.trigger_name,
                                    execType: result.exec_type,
                                    totalTrades: result.total_trades,
                                    profitFactor: result.profit_factor,
                                    winRate: result.win_rate,
                                    avgR: result.avg_r,
                                    dailyR: result.daily_r,
                                    rSquared: result.r_squared,
                                  }}
                                  isCurrent={result.trigger_id === entryTrigger}
                                  actionLabel="Replace"
                                  onAction={() => setEntryTrigger(result.trigger_id)}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                              Click <strong>Analyze</strong> to compare all available entry triggers using the current strategy config.
                            </p>
                          )}

                        </div>
                      );
                    }

                    // ---- EXIT TAB ----
                    if (tab === 'Exit') {
                      return (
                        <div>
                          <AnalysisToolbar search="" onSearch={() => {}} onAnalyze={() => handleAnalyze('exit')} onFilterClick={() => setFilterModalOpen(true)} placeholder="Search exit triggers..." />
                          {/* Active exit tags */}
                          {exitTriggers.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mb-3">
                              {exitTriggers.map((eid, idx) => {
                                const eDef = API_TRIGGERS.find((t) => t.id === eid);
                                return (
                                  <ConditionPill
                                    key={eid}
                                    label={eDef?.name || eid}
                                    onRemove={() => handleRemoveExit(idx)}
                                  />
                                );
                              })}
                            </div>
                          )}
                          {isAnalyzing && analyzingMode === 'exit' ? (
                            <AnalyzeProgressBar label="Analyzing exit triggers..." />
                          ) : (analysisResults.exit?.length ?? 0) > 0 ? (
                            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 440 }}>
                              {applyAnalysisFilters(analysisResults.exit || [], filters).map((result) => (
                                <AnalyzerResultCard
                                  key={result.trigger_id}
                                  result={{
                                    triggerId: result.trigger_id,
                                    triggerName: result.trigger_name,
                                    execType: result.exec_type,
                                    totalTrades: result.total_trades,
                                    profitFactor: result.profit_factor,
                                    winRate: result.win_rate,
                                    avgR: result.avg_r,
                                    dailyR: result.daily_r,
                                    rSquared: result.r_squared,
                                  }}
                                  isCurrent={exitTriggers.includes(result.trigger_id)}
                                  actionLabel={exitDepth >= 2 ? 'Replace' : 'Add'}
                                  onAction={() => {
                                    if (exitTriggers.length < 3 && !exitTriggers.includes(result.trigger_id)) {
                                      setExitTriggers((prev) => [...prev, result.trigger_id]);
                                    }
                                  }}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                              Click <strong>Analyze</strong> to compare individual exit triggers against the current entry.
                            </p>
                          )}

                        </div>
                      );
                    }

                    // ---- TF CONDITIONS TAB ----
                    if (tab === 'TF Conditions') {
                      const tfSelected = Array.from(selectedConditions).sort();
                      return (
                        <div>
                          <div className="flex gap-2 mb-3">
                            <div className="flex-1">
                              <TextInput value="" onChange={() => {}} placeholder="Search indicators..." />
                            </div>
                            <PrimaryButton onClick={() => handleAnalyze(tfDepth > 1 ? 'combinations' : 'condition')}>{isAnalyzing && (analyzingMode === 'condition' || analyzingMode === 'combinations') ? 'Analyzing...' : 'Analyze'}</PrimaryButton>
                            <button onClick={() => setFilterModalOpen(true)} className="px-2.5 py-2 rounded-lg transition-colors" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)' }} title="Filter & Sort">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6" /><line x1="7" y1="12" x2="17" y2="12" /><line x1="10" y1="18" x2="14" y2="18" /></svg>
                            </button>
                          </div>
                          {/* Active TF condition pills */}
                          {tfSelected.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mb-3">
                              {tfSelected.map((cid) => {
                                const cDef = API_CONFLUENCE_CONDITIONS.find((c) => c.id === cid);
                                return (
                                  <ConditionPill
                                    key={cid}
                                    label={cDef?.label || cid}
                                    onRemove={() => handleToggleCondition(cid)}
                                  />
                                );
                              })}
                              <button
                                className="text-xs px-2 py-1 rounded"
                                style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                                onClick={() => setSelectedConditions(new Set())}
                              >
                                Clear TF
                              </button>
                            </div>
                          )}

                          {/* Confluence drill-down results */}
                          {isAnalyzing && analyzingMode === 'condition' ? (
                            <AnalyzeProgressBar label="Analyzing TF conditions..." />
                          ) : ((analysisResults.condition?.length ?? 0) > 0 || (analysisResults.combinations?.length ?? 0) > 0) ? (
                            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 440 }}>
                              {applyAnalysisFilters(analysisResults.combinations || analysisResults.condition || [], filters).map((result) => (
                                <AnalyzerResultCard
                                  key={result.trigger_id}
                                  result={{ triggerId: result.trigger_id, triggerName: result.trigger_name, execType: 'C', totalTrades: result.total_trades, profitFactor: result.profit_factor, winRate: result.win_rate, avgR: result.avg_r, dailyR: result.daily_r, rSquared: result.r_squared }}
                                  isCurrent={selectedConditions.has(result.trigger_id)}
                                  actionLabel={selectedConditions.has(result.trigger_id) ? 'Remove' : 'Add'}
                                  onAction={() => handleToggleCondition(result.trigger_id)}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                              Click <strong>Analyze</strong> to evaluate how each TF condition affects your strategy's performance.
                            </p>
                          )}

                        </div>
                      );
                    }

                    // ---- GENERAL TAB ----
                    if (tab === 'General') {
                      const genSelected = Array.from(selectedGenerals).sort();
                      return (
                        <div>
                          <div className="flex gap-2 mb-3">
                            <div className="flex-1">
                              <TextInput value="" onChange={() => {}} placeholder="Search general conditions..." />
                            </div>
                            <PrimaryButton onClick={() => handleAnalyze(generalDepth > 1 ? 'general_combinations' : 'general')}>{isAnalyzing && (analyzingMode === 'general' || analyzingMode === 'general_combinations') ? 'Analyzing...' : 'Analyze'}</PrimaryButton>
                            <button onClick={() => setFilterModalOpen(true)} className="px-2.5 py-2 rounded-lg transition-colors" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)' }} title="Filter & Sort">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6" /><line x1="7" y1="12" x2="17" y2="12" /><line x1="10" y1="18" x2="14" y2="18" /></svg>
                            </button>
                          </div>
                          {/* Active general pills */}
                          {genSelected.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mb-3">
                              {genSelected.map((gid) => {
                                const gDef = API_GENERAL_CONDITIONS.find((c) => c.id === gid);
                                return (
                                  <ConditionPill
                                    key={gid}
                                    label={gDef?.label || gid}
                                    onRemove={() => handleToggleGeneral(gid)}
                                  />
                                );
                              })}
                              <button
                                className="text-xs px-2 py-1 rounded"
                                style={{ color: 'var(--red)', background: 'var(--red-muted)' }}
                                onClick={() => setSelectedGenerals(new Set())}
                              >
                                Clear Gen
                              </button>
                            </div>
                          )}

                          {/* General condition results */}
                          {isAnalyzing && analyzingMode === 'general' ? (
                            <AnalyzeProgressBar label="Analyzing general conditions..." />
                          ) : ((analysisResults.general?.length ?? 0) > 0 || (analysisResults.general_combinations?.length ?? 0) > 0) ? (
                            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 440 }}>
                              {applyAnalysisFilters(analysisResults.general_combinations || analysisResults.general || [], filters).map((result) => (
                                <AnalyzerResultCard
                                  key={result.trigger_id}
                                  result={{ triggerId: result.trigger_id, triggerName: result.trigger_name, execType: 'C', totalTrades: result.total_trades, profitFactor: result.profit_factor, winRate: result.win_rate, avgR: result.avg_r, dailyR: result.daily_r, rSquared: result.r_squared }}
                                  isCurrent={selectedGenerals.has(result.trigger_id)}
                                  actionLabel={selectedGenerals.has(result.trigger_id) ? 'Remove' : 'Add'}
                                  onAction={() => handleToggleGeneral(result.trigger_id)}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                              Click <strong>Analyze</strong> to evaluate general conditions (time of day, session, calendar filters).
                            </p>
                          )}

                        </div>
                      );
                    }

                    // ---- STOP LOSS TAB ----
                    if (tab === 'Stop Loss') {
                      const stopResults = analysisResults.stop || [];
                      return (
                        <div>
                          <AnalysisToolbar search="" onSearch={() => {}} onAnalyze={() => handleAnalyze('stop')} onFilterClick={() => setFilterModalOpen(true)} placeholder="Search stop loss packs..." />
                          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                            Current: <strong>{API_STOP_PACKS.find((p) => p.id === selectedStopPack)?.summary || 'None'}</strong>
                          </p>
                          {isAnalyzing && analyzingMode === 'stop' ? (
                            <AnalyzeProgressBar label="Analyzing stop loss packs..." />
                          ) : stopResults.length > 0 ? (
                            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 440 }}>
                              {applyAnalysisFilters(stopResults, filters).map((result) => (
                                <AnalyzerResultCard
                                  key={result.trigger_id}
                                  result={{ triggerId: result.trigger_id, triggerName: result.trigger_name, execType: 'C', totalTrades: result.total_trades, profitFactor: result.profit_factor, winRate: result.win_rate, avgR: result.avg_r, dailyR: result.daily_r, rSquared: result.r_squared }}
                                  isCurrent={result.trigger_id === selectedStopPack}
                                  actionLabel="Replace"
                                  onAction={() => setSelectedStopPack(result.trigger_id)}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                              Click <strong>Analyze</strong> to compare stop loss packs.
                            </p>
                          )}
                        </div>
                      );
                    }

                    // ---- TAKE PROFIT TAB ----
                    if (tab === 'Take Profit') {
                      const tpResults = analysisResults.target || [];
                      return (
                        <div>
                          <AnalysisToolbar search="" onSearch={() => {}} onAnalyze={() => handleAnalyze('target')} onFilterClick={() => setFilterModalOpen(true)} placeholder="Search take profit packs..." />
                          <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                            Current: <strong>{API_TARGET_PACKS.find((p) => p.id === selectedTargetPack)?.summary || 'None'}</strong>
                          </p>
                          {isAnalyzing && analyzingMode === 'target' ? (
                            <AnalyzeProgressBar label="Analyzing take profit packs..." />
                          ) : tpResults.length > 0 ? (
                            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 440 }}>
                              {applyAnalysisFilters(tpResults, filters).map((result) => (
                                <AnalyzerResultCard
                                  key={result.trigger_id}
                                  result={{ triggerId: result.trigger_id, triggerName: result.trigger_name, execType: 'C', totalTrades: result.total_trades, profitFactor: result.profit_factor, winRate: result.win_rate, avgR: result.avg_r, dailyR: result.daily_r, rSquared: result.r_squared }}
                                  isCurrent={result.trigger_id === selectedTargetPack}
                                  actionLabel="Replace"
                                  onAction={() => setSelectedTargetPack(result.trigger_id)}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>
                              Click <strong>Analyze</strong> to compare take profit packs.
                            </p>
                          )}
                        </div>
                      );
                    }

                    return null;
                  }}
                </TabBar>
                </div>
                {/* Pinned depth selector at bottom of card */}
                <div className="flex-shrink-0 -mb-2">
                  {activeAnalysisTab === 'Entry' && <DepthSelector depth={entryDepth} maxDepth={1} onChange={() => {}} />}
                  {activeAnalysisTab === 'Exit' && <DepthSelector depth={1} maxDepth={1} onChange={() => {}} />}
                  {activeAnalysisTab === 'TF Conditions' && <DepthSelector depth={tfDepth} maxDepth={4} onChange={setTfDepth} />}
                  {activeAnalysisTab === 'General' && <DepthSelector depth={generalDepth} maxDepth={4} onChange={setGeneralDepth} />}
                  {activeAnalysisTab === 'Stop Loss' && <DepthSelector depth={stopDepth} maxDepth={1} onChange={() => {}} />}
                  {activeAnalysisTab === 'Take Profit' && <DepthSelector depth={targetDepth} maxDepth={1} onChange={() => {}} />}
                </div>
              </Card>

              {/* Advanced Analysis */}
              <AdvancedAnalysis />
            </div>
          </div>

          {/* ================================================================= */}
          {/* SAVE STRATEGY                                                      */}
          {/* ================================================================= */}
          <div className="mt-6 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="flex justify-center">
              <PrimaryButton size="lg" onClick={() => {
                if (onSave) {
                  onSave({
                    name: strategyName, symbol, direction, timeframe,
                    trading_session: session,
                    data_days: lookbackDays, lookback_mode: lookbackMode,
                    entry_trigger_confluence_id: entryTrigger,
                    exit_trigger_confluence_ids: exitTriggers,
                    confluence: Array.from(selectedConditions),
                    stop_loss_pack_id: selectedStopPack,
                    take_profit_pack_id: selectedTargetPack,
                    kpis: backtestResult?.kpis ?? EMPTY_KPIS,
                    stored_trades: backtestResult?.trades ?? [],
                  });
                }
              }}>
                {onSave ? 'Save Strategy' : 'Save Strategy (Demo)'}
              </PrimaryButton>
            </div>
          </div>
        </>
      )}

      {/* ================================================================= */}
      {/* MODALS                                                             */}
      {/* ================================================================= */}
      <ConfluencePickerModal
        isOpen={confluenceModalOpen}
        onClose={() => setConfluenceModalOpen(false)}
        conditions={API_CONFLUENCE_CONDITIONS}
        selected={selectedConditions}
        onToggle={handleToggleCondition}
      />
      <ConfluencePickerModal
        isOpen={generalModalOpen}
        onClose={() => setGeneralModalOpen(false)}
        conditions={API_GENERAL_CONDITIONS}
        selected={selectedGenerals}
        onToggle={handleToggleGeneral}
      />
      <FilterSortModal
        isOpen={filterModalOpen}
        onClose={() => setFilterModalOpen(false)}
        filters={filters}
        onApply={setFilters}
      />
    </div>
  );
}
