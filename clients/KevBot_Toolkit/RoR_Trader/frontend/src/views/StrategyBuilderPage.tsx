'use client';

/**
 * Strategy Builder — Clean API-first page. V5 design.
 * Strategy method selector, horizontal config bar, expandable config (entry/exit/stop/target packs),
 * confluence pills, symmetric 2-col layout (charts+trades | analysis tabs+advanced),
 * analysis tabs with depth selector, KPI dashboard (2 rows), and save button.
 *
 * QA Fix: Trigger selection now uses confluence group triggers from the API
 * instead of plain text inputs.
 */

import { useState, useCallback, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import MetricCard from '@/components/MetricCard';
import { useRunBacktest, BacktestRequest } from '@/hooks/queries/useBacktest';
import { useCreateStrategy } from '@/hooks/mutations/useStrategyMutations';
import { useConfluenceGroups, useConfluenceTriggers, useRiskManagementPacks } from '@/hooks/queries/usePacks';

const TIMEFRAMES = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1Hour', '4Hour', '1Day'];
const DIRECTIONS: Array<'LONG' | 'SHORT'> = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];
const ASSET_TYPES = ['Equity', 'Crypto'];
const ANALYSIS_TABS = ['Entry', 'Exit', 'TF Conditions', 'General', 'Stop Loss', 'Take Profit'];

const iStyle: React.CSSProperties = { background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '6px 10px', borderRadius: '6px', fontSize: '.8125rem', width: '100%' };
const Label = ({ children }: { children: React.ReactNode }) => <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-muted)' }}>{children}</label>;

function kpi(v: number | undefined, d = 2, s = ''): string {
  if (v == null) return '--';
  return `${v >= 0 ? '+' : ''}${v.toFixed(d)}${s}`;
}

function Pill({ label, onRemove }: { label: string; onRemove?: () => void }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
      {label}
      {onRemove && <button onClick={onRemove} className="ml-0.5 hover:opacity-70" style={{ color: 'var(--text-muted)' }}>x</button>}
    </span>
  );
}

function Depth({ depth, max, onChange }: { depth: number; max: number; onChange: (d: number) => void }) {
  return (
    <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Depth:</span>
      <div className="flex gap-1">
        {Array.from({ length: max }, (_, i) => i + 1).map(n => (
          <button key={n} onClick={() => onChange(n)} className="w-7 h-7 rounded text-xs font-medium" style={{ background: n === depth ? 'var(--accent)' : 'var(--bg-input)', color: n === depth ? 'white' : 'var(--text-muted)', border: n === depth ? 'none' : '1px solid var(--border)' }}>{n}</button>
        ))}
      </div>
      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{depth === 1 ? 'Individual' : `Up to ${depth}`}</span>
    </div>
  );
}

function PackList({ packs, selectedId, onSelect }: { packs: any[]; selectedId: string; onSelect: (id: string) => void }) {
  const items = [{ id: '', label: 'None (default)' }, ...packs.map((p: any) => ({ id: p.id, label: `${p.base_template || 'Pack'} (${p.version || 'v1'})` }))];
  return (
    <div className="space-y-1 overflow-y-auto pr-1" style={{ maxHeight: 240 }}>
      {items.map(p => (
        <button key={p.id} className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors" style={{ background: p.id === selectedId ? 'var(--accent-muted, rgba(0,255,136,.1))' : 'transparent', color: p.id === selectedId ? 'var(--accent)' : 'var(--text-primary)', border: p.id === selectedId ? '1px solid var(--accent)' : '1px solid transparent' }} onClick={() => onSelect(p.id)}>
          <span className="flex-1 truncate">{p.label}</span>
        </button>
      ))}
    </div>
  );
}

/** Grouped trigger list — shows triggers grouped by pack, with search filtering */
function TriggerGroupedList({
  triggers,
  searchQuery,
  selectedId,
  onSelect,
}: {
  triggers: Record<string, string>;
  searchQuery: string;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const filtered = useMemo(() => {
    const entries = Object.entries(triggers);
    if (!searchQuery) return entries;
    const q = searchQuery.toLowerCase();
    return entries.filter(([id, name]) => id.toLowerCase().includes(q) || name.toLowerCase().includes(q));
  }, [triggers, searchQuery]);

  if (filtered.length === 0) {
    return <p className="text-xs p-3" style={{ color: 'var(--text-muted)' }}>No triggers available.</p>;
  }

  return (
    <div className="space-y-0.5 overflow-y-auto pr-1" style={{ maxHeight: 280 }}>
      {filtered.map(([id, name]) => (
        <button
          key={id}
          className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
          style={{
            background: id === selectedId ? 'var(--accent-muted, rgba(0,255,136,.1))' : 'transparent',
            color: id === selectedId ? 'var(--accent)' : 'var(--text-secondary)',
          }}
          onClick={() => onSelect(id)}
        >
          <span className="flex-1 truncate">{name}</span>
          <span className="text-[10px] font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{id}</span>
          {id === selectedId && <span className="text-[10px]" style={{ color: 'var(--accent)' }}>selected</span>}
        </button>
      ))}
    </div>
  );
}

/** Confluence condition picker — checkable list built from confluence groups */
function ConfluenceConditionPicker({
  groups,
  selected,
  onToggle,
}: {
  groups: any[];
  selected: Set<string>;
  onToggle: (id: string) => void;
}) {
  const [search, setSearch] = useState('');

  // Build conditions from confluence groups: "{TF}-{INTERPRETER}-{STATE}" format
  // We show the group base_template + version as label context
  const conditions = useMemo(() => {
    if (!groups || groups.length === 0) return [];
    return groups
      .filter((g: any) => g.enabled !== false)
      .map((g: any) => ({
        id: g.id,
        label: `${g.base_template} (${g.version || 'v1'})`,
        description: g.description || '',
      }));
  }, [groups]);

  const filtered = useMemo(() => {
    if (!search) return conditions;
    const q = search.toLowerCase();
    return conditions.filter(c => c.label.toLowerCase().includes(q) || c.id.toLowerCase().includes(q));
  }, [conditions, search]);

  if (conditions.length === 0) {
    return <p className="text-xs py-4 text-center" style={{ color: 'var(--text-muted)' }}>No confluence groups configured. Create groups in the Confluence Packs page.</p>;
  }

  return (
    <div>
      <input
        type="text"
        placeholder="Search conditions..."
        value={search}
        onChange={e => setSearch(e.target.value)}
        style={{ ...iStyle, marginBottom: 8 }}
      />
      <div className="space-y-0.5 overflow-y-auto pr-1" style={{ maxHeight: 240 }}>
        {filtered.map(c => {
          const active = selected.has(c.id);
          return (
            <button
              key={c.id}
              className="w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition-colors"
              style={{
                background: active ? 'var(--accent-muted, rgba(0,255,136,.1))' : 'transparent',
                color: active ? 'var(--accent)' : 'var(--text-secondary)',
              }}
              onClick={() => onToggle(c.id)}
            >
              <span
                className="w-4 h-4 rounded border flex items-center justify-center text-[10px] flex-shrink-0"
                style={{
                  borderColor: active ? 'var(--accent)' : 'var(--border)',
                  background: active ? 'var(--accent)' : 'transparent',
                  color: active ? '#000' : 'transparent',
                }}
              >
                {active ? '\u2713' : ''}
              </span>
              <span className="flex-1 truncate">{c.label}</span>
              <span className="text-[10px] font-mono flex-shrink-0" style={{ color: 'var(--text-muted)' }}>{c.id}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function StrategyBuilderPage() {
  const router = useRouter();
  const bt = useRunBacktest();
  const createMut = useCreateStrategy();
  const { data: rmPacks } = useRiskManagementPacks();
  const { data: confluenceGroups } = useConfluenceGroups();

  const [method, setMethod] = useState<'standard' | 'webhook' | 'scanner'>('standard');
  const [symbol, setSymbol] = useState('');
  const [asset, setAsset] = useState('Equity');
  const [tf, setTf] = useState('5Min');
  const [dir, setDir] = useState<'LONG' | 'SHORT'>('LONG');
  const [sess, setSess] = useState('RTH');
  const [days, setDays] = useState(30);
  const [name, setName] = useState('');
  const [entry, setEntry] = useState('');
  const [exitTriggers, setExitTriggers] = useState<string[]>([]);
  const [confSelected, setConfSelected] = useState<Set<string>>(new Set());
  const [stopId, setStopId] = useState('');
  const [tpId, setTpId] = useState('');
  const [tab, setTab] = useState('Entry');
  const [exitD, setExitD] = useState(1);
  const [tfD, setTfD] = useState(1);
  const [genD, setGenD] = useState(1);
  const [expanded, setExpanded] = useState(false);
  const [triggerSearch, setTriggerSearch] = useState('');

  // Fetch triggers based on direction
  const { data: entryTriggers } = useConfluenceTriggers(dir);
  const { data: exitTriggerMap } = useConfluenceTriggers('EXIT');

  const allPacks = rmPacks || [];
  const entryTriggerMap = entryTriggers || {};
  const exitTriggersAvailable = exitTriggerMap || {};

  const estBars = useMemo(() => {
    const bpd: Record<string, number> = { '1Min': 390, '2Min': 195, '3Min': 130, '5Min': 78, '10Min': 39, '15Min': 26, '30Min': 13, '1Hour': 7, '4Hour': 2, '1Day': 1 };
    return Math.round(days * (bpd[tf] || 78));
  }, [days, tf]);

  const canRun = symbol.trim() && entry.trim();
  const result = bt.data;
  const kpis = result?.kpis || {};
  const ran = !!result;

  const confPills = useMemo(() => Array.from(confSelected), [confSelected]);

  const run = useCallback(() => {
    if (!canRun) return;
    bt.mutate({
      symbol: symbol.trim().toUpperCase(), timeframe: tf, direction: dir, days, session: sess,
      entry_trigger_confluence_id: entry.trim(),
      exit_trigger_confluence_ids: exitTriggers,
      confluence: confPills,
      stop_loss_pack_id: stopId || undefined, take_profit_pack_id: tpId || undefined, include_chart_data: false,
    } as BacktestRequest);
  }, [canRun, symbol, tf, dir, days, sess, entry, exitTriggers, confPills, stopId, tpId, bt]);

  const save = useCallback(() => {
    if (!result) return;
    createMut.mutate({
      name: name.trim() || `${symbol.toUpperCase()} ${dir} ${tf}`,
      symbol: symbol.trim().toUpperCase(), timeframe: tf, direction: dir, trading_session: sess, data_days: days,
      entry_trigger_confluence_id: entry.trim(),
      exit_trigger_confluence_ids: exitTriggers,
      confluence: confPills,
      stop_loss_pack_id: stopId || undefined, take_profit_pack_id: tpId || undefined, kpis: result.kpis,
    }, { onSuccess: () => router.push('/strategies') });
  }, [result, name, symbol, tf, dir, sess, days, entry, exitTriggers, confPills, stopId, tpId, createMut, router]);

  const handleToggleCondition = useCallback((id: string) => {
    setConfSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }, []);

  const handleRemoveExit = useCallback((idx: number) => {
    setExitTriggers(prev => prev.filter((_, i) => i !== idx));
  }, []);

  // Entry trigger display name
  const entryName = entry ? (entryTriggerMap[entry] || entry) : '';

  return (
    <div>
      <PageHeader title="Strategy Builder" subtitle="Configure, backtest, and save a trading strategy" actions={
        <span className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5" style={{ background: 'rgba(76,175,80,.15)', color: 'var(--green)' }}><span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />Live</span>
      } />

      {/* METHOD SELECTOR */}
      <Card className="mb-4">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>Strategy Method:</span>
          <div className="flex gap-2">
            {([{ id: 'standard' as const, l: 'Standard', ok: true }, { id: 'webhook' as const, l: 'Inbound Webhook', ok: false }, { id: 'scanner' as const, l: 'Scanner', ok: false }]).map(m => (
              <button key={m.id} onClick={() => m.ok && setMethod(m.id)} className="px-4 py-2 rounded-lg text-sm font-medium relative" style={{ background: method === m.id ? 'var(--accent-muted, rgba(0,255,136,.1))' : 'var(--bg-input)', color: method === m.id ? 'var(--accent)' : m.ok ? 'var(--text-primary)' : 'var(--text-muted)', border: method === m.id ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: m.ok ? 'pointer' : 'not-allowed', opacity: m.ok ? 1 : 0.6 }}>
                {m.l}{!m.ok && <span className="ml-1.5 text-[9px] px-1 py-0.5 rounded" style={{ background: 'var(--bg-card)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>Soon</span>}
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* CONFIG BAR */}
      <Card className="mb-4">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-9 gap-3">
          <div><Label>Asset</Label><select value={asset} onChange={e => { setAsset(e.target.value); if (e.target.value === 'Crypto') setSess('24/7'); else if (sess === '24/7') setSess('RTH'); }} style={iStyle}>{ASSET_TYPES.map(a => <option key={a}>{a}</option>)}</select></div>
          <div><Label>Ticker</Label><input type="text" placeholder={asset === 'Crypto' ? 'BTC/USD' : 'NVDA'} value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} style={iStyle} /></div>
          <div><Label>Timeframe</Label><select value={tf} onChange={e => setTf(e.target.value)} style={iStyle}>{TIMEFRAMES.map(t => <option key={t}>{t}</option>)}</select></div>
          <div><Label>Direction</Label><select value={dir} onChange={e => setDir(e.target.value as any)} style={iStyle}>{DIRECTIONS.map(d => <option key={d}>{d}</option>)}</select></div>
          <div><Label>Session</Label>{asset === 'Crypto' ? <input style={{ ...iStyle, color: 'var(--text-muted)' }} value="24/7" disabled /> : <select value={sess} onChange={e => setSess(e.target.value)} style={iStyle}>{SESSIONS.map(s => <option key={s}>{s}</option>)}</select>}</div>
          <div><Label>Days</Label><input type="number" min={1} max={365} value={days} onChange={e => setDays(+e.target.value)} style={iStyle} /></div>
          <div className="xl:col-span-2"><Label>Name</Label><input type="text" placeholder={`${symbol || 'SPY'} ${dir} - 1`} value={name} onChange={e => setName(e.target.value)} style={iStyle} /></div>
          <div className="flex items-end"><button className="w-full py-2 rounded-lg text-sm font-medium" style={{ background: symbol.trim() ? 'var(--accent)' : 'var(--bg-input)', color: symbol.trim() ? '#000' : 'var(--text-muted)', border: 'none', cursor: symbol.trim() ? 'pointer' : 'not-allowed' }} disabled={!symbol.trim() || bt.isPending} onClick={run}>{bt.isPending ? 'Running Backtest...' : ran ? 'Re-run Backtest' : 'Run Backtest'}</button></div>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>~{estBars.toLocaleString()} bars</span>
          {estBars > 200000 && <span className="text-xs" style={{ color: 'var(--red)' }}>Very large dataset</span>}
          {estBars > 50000 && estBars <= 200000 && <span className="text-xs" style={{ color: 'var(--orange, #FF9800)' }}>Large dataset</span>}
        </div>
      </Card>

      {/* CONFIG EXPANDER */}
      <Card className="mb-4">
        <button className="w-full flex items-center justify-between text-sm font-medium" style={{ color: 'var(--text-secondary)' }} onClick={() => setExpanded(!expanded)}>
          <div className="flex items-center gap-3">
            <span>Strategy Config</span>
            {!expanded && entry && (
              <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>
                {entryName} | {exitTriggers.length} exit(s) | {confPills.length} cond | Stop: {stopId || 'None'} | TP: {tpId || 'None'}
              </span>
            )}
          </div>
          <span className="text-xs" style={{ color: 'var(--text-muted)', transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform .2s' }}>v</span>
        </button>
        {expanded && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Entry Trigger — grouped list from API */}
            <div>
              <Label>Entry Trigger</Label>
              <input
                type="text"
                placeholder="Search triggers..."
                value={triggerSearch}
                onChange={e => setTriggerSearch(e.target.value)}
                style={{ ...iStyle, marginBottom: 8 }}
              />
              {Object.keys(entryTriggerMap).length > 0 ? (
                <TriggerGroupedList
                  triggers={entryTriggerMap}
                  searchQuery={triggerSearch}
                  selectedId={entry}
                  onSelect={setEntry}
                />
              ) : (
                <p className="text-xs py-4 text-center" style={{ color: 'var(--text-muted)' }}>
                  No entry triggers available. Create confluence groups first.
                </p>
              )}
            </div>

            {/* Exit Triggers — multi-select list from API */}
            <div>
              <Label>
                Exit Trigger(s){' '}
                <span style={{ color: 'var(--text-muted)' }}>({exitTriggers.length}/3)</span>
              </Label>
              {exitTriggers.length > 0 && (
                <div className="space-y-1 mb-2">
                  {exitTriggers.map((eid, idx) => (
                    <div key={eid} className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs" style={{ background: 'var(--accent-muted, rgba(0,255,136,.1))', border: '1px solid var(--border)' }}>
                      <span className="flex-1 truncate" style={{ color: 'var(--text-secondary)' }}>
                        {exitTriggersAvailable[eid] || eid}
                      </span>
                      <button className="text-[10px] hover:opacity-70" style={{ color: 'var(--text-muted)' }} onClick={() => handleRemoveExit(idx)}>x</button>
                    </div>
                  ))}
                </div>
              )}
              {Object.keys(exitTriggersAvailable).length > 0 ? (
                <TriggerGroupedList
                  triggers={exitTriggersAvailable}
                  searchQuery=""
                  selectedId=""
                  onSelect={(id) => {
                    if (exitTriggers.length < 3 && !exitTriggers.includes(id)) {
                      setExitTriggers(prev => [...prev, id]);
                    }
                  }}
                />
              ) : (
                <p className="text-xs py-4 text-center" style={{ color: 'var(--text-muted)' }}>
                  No exit triggers available.
                </p>
              )}
            </div>

            {/* Stop Loss Pack */}
            <div><Label>Stop Loss Pack</Label><PackList packs={allPacks} selectedId={stopId} onSelect={setStopId} /></div>

            {/* Take Profit Pack */}
            <div><Label>Take Profit Pack</Label><PackList packs={allPacks} selectedId={tpId} onSelect={setTpId} /></div>
          </div>
        )}
      </Card>

      {/* CONFLUENCE CONDITIONS — checkable pills from confluence groups */}
      {expanded && <Card className="mb-4">
        <Label>Confluence Conditions</Label>
        {confPills.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {confPills.map(p => (
              <Pill key={p} label={confluenceGroups?.find((g: any) => g.id === p)?.base_template || p} onRemove={() => handleToggleCondition(p)} />
            ))}
            <button className="text-xs px-2 py-1 rounded" style={{ color: 'var(--red)', background: 'rgba(244,67,54,.15)' }} onClick={() => setConfSelected(new Set())}>Clear All</button>
          </div>
        )}
        <ConfluenceConditionPicker
          groups={confluenceGroups || []}
          selected={confSelected}
          onToggle={handleToggleCondition}
        />
      </Card>}

      {/* PRE-BACKTEST */}
      {!ran && !bt.isError && <Card className="text-center py-12">
        <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Select settings above, then click <strong>Load Data</strong> to begin.</p>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{symbol || 'TICKER'} | {dir} | {tf} | ~{estBars.toLocaleString()} bars</p>
      </Card>}

      {bt.isError && <Card className="mb-4"><div className="text-center py-8" style={{ color: 'var(--red)' }}>Backtest failed.<p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>{(bt.error as any)?.message || 'Unknown error'}</p></div></Card>}

      {/* POST-BACKTEST */}
      {ran && <>
        {/* KPI Dashboard: 2 rows */}
        <div className="space-y-3 mb-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-8 gap-2">
            <MetricCard label="Trades" value={kpis.total_trades != null ? String(kpis.total_trades) : '--'} />
            <MetricCard label="Win Rate" value={kpis.win_rate != null ? `${kpis.win_rate.toFixed(1)}%` : '--'} positive={kpis.win_rate > 50} />
            <MetricCard label="Profit Factor" value={kpis.profit_factor != null ? kpis.profit_factor.toFixed(2) : '--'} positive={kpis.profit_factor > 1} />
            <MetricCard label="Avg R" value={kpi(kpis.avg_r)} positive={kpis.avg_r > 0} />
            <MetricCard label="Total R" value={kpi(kpis.total_r)} positive={kpis.total_r > 0} />
            <MetricCard label="Daily R" value={kpi(kpis.daily_r)} positive={kpis.daily_r > 0} />
            <MetricCard label="R-Squared" value={kpis.r_squared != null ? kpis.r_squared.toFixed(2) : '--'} positive={kpis.r_squared > 0.7} />
            <MetricCard label="Max R DD" value={kpis.max_r_drawdown != null ? `${kpis.max_r_drawdown.toFixed(1)}R` : '--'} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
            <MetricCard label="Avg Win" value={kpi(result?.secondary_kpis?.avg_win_r)} positive />
            <MetricCard label="Avg Loss" value={kpi(result?.secondary_kpis?.avg_loss_r)} />
            <MetricCard label="Max DD" value={result?.secondary_kpis?.max_drawdown != null ? `${result.secondary_kpis.max_drawdown.toFixed(1)}%` : '--'} />
            <MetricCard label="Recovery" value={result?.secondary_kpis?.recovery_factor != null ? result.secondary_kpis.recovery_factor.toFixed(2) : '--'} positive={result?.secondary_kpis?.recovery_factor > 2} />
            <MetricCard label="Sharpe" value={result?.secondary_kpis?.sharpe_ratio != null ? result.secondary_kpis.sharpe_ratio.toFixed(2) : '--'} positive={result?.secondary_kpis?.sharpe_ratio > 1} />
            <MetricCard label="Data" value={result?.data_source || '--'} />
          </div>
        </div>

        {/* SYMMETRIC 2-COL: Charts+Trades | Analysis+Advanced */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* LEFT */}
          <div className="space-y-4">
            <div style={{ height: 620 }}>
              <Card className="h-full flex flex-col">
                <TabBar tabs={['Equity Curve', 'Price Chart']}>
                  {(t) => t === 'Equity Curve' ? (
                    <div className="flex-1">
                      {result?.equity_curve?.length > 0 ? (
                        <div className="rounded-lg overflow-hidden" style={{ background: 'var(--bg-input)', height: 500 }}>
                          <svg width="100%" height="100%" viewBox={`0 0 ${result.equity_curve.length} 100`} preserveAspectRatio="none">
                            {(() => { const d = result.equity_curve.map((p: any) => p.cumulative_r); const mn = Math.min(...d); const mx = Math.max(...d); const r = mx - mn || 1; return <polyline points={d.map((v: number, i: number) => `${i},${100 - ((v - mn) / r) * 90 - 5}`).join(' ')} fill="none" stroke="var(--accent)" strokeWidth="1" />; })()}
                          </svg>
                        </div>
                      ) : <ChartPlaceholder label="Equity curve -- cumulative R over time" height={500} />}
                      <div className="mt-2 flex gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                        <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'var(--accent)' }} />Equity</span>
                        <span><span className="inline-block w-3 h-0.5 mr-1 rounded" style={{ background: 'green', opacity: .5 }} />HWM</span>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <span className="text-xs mb-2 block" style={{ color: 'var(--text-muted)' }}>{result?.trades?.length || 0} trades on {symbol.toUpperCase()} ({dir})</span>
                      <ChartPlaceholder label="OHLC chart with indicators + trade markers" height={500} />
                    </div>
                  )}
                </TabBar>
              </Card>
            </div>
            <Card>
              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Trade History</h3>
              {result?.trades?.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead><tr style={{ borderBottom: '1px solid var(--border)' }}>{['#', 'Entry', 'Exit', 'Dir', 'Entry $', 'Exit $', 'P&L (R)', 'Exec', 'Reason'].map(h => <th key={h} className="px-2 py-2 font-medium text-left whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>{h}</th>)}</tr></thead>
                    <tbody>
                      {result.trades.slice(0, 20).map((t: any, i: number) => (
                        <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td className="px-2 py-2" style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                          <td className="px-2 py-2 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{t.entry_time || '--'}</td>
                          <td className="px-2 py-2 whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{t.exit_time || '--'}</td>
                          <td className="px-2 py-2"><span style={{ color: t.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{t.direction}</span></td>
                          <td className="px-2 py-2 font-mono" style={{ color: 'var(--text-secondary)' }}>{t.entry_price?.toFixed(2) ?? '--'}</td>
                          <td className="px-2 py-2 font-mono" style={{ color: 'var(--text-secondary)' }}>{t.exit_price?.toFixed(2) ?? '--'}</td>
                          <td className="px-2 py-2 font-mono font-medium"><span style={{ color: (t.r_multiple ?? 0) >= 0 ? 'var(--green)' : 'var(--red)' }}>{t.r_multiple != null ? `${t.r_multiple >= 0 ? '+' : ''}${t.r_multiple.toFixed(2)}` : '--'}</span></td>
                          <td className="px-2 py-2"><span className="text-xs font-mono px-1.5 py-0.5 rounded-full" style={{ color: '#2196F3', background: 'rgba(33,150,243,.12)' }}>[{t.exec_type || 'C'}]</span></td>
                          <td className="px-2 py-2" style={{ color: 'var(--text-muted)' }}>{t.exit_reason || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {result.trades.length > 20 && <p className="text-xs text-center py-2" style={{ color: 'var(--text-muted)' }}>Showing 20 of {result.trades.length}</p>}
                </div>
              ) : <p className="text-xs text-center py-4" style={{ color: 'var(--text-muted)' }}>No trades</p>}
            </Card>
          </div>

          {/* RIGHT: Analysis Tabs + Advanced */}
          <div className="space-y-4">
            <div style={{ height: 620 }}>
              <Card className="h-full flex flex-col overflow-hidden">
                <div className="flex-1 min-h-0 overflow-hidden">
                  <TabBar tabs={ANALYSIS_TABS}>
                    {(t) => {
                      if (t !== tab) setTimeout(() => setTab(t), 0);
                      return (
                        <div>
                          <div className="flex gap-2 mb-3">
                            <input type="text" placeholder={`Search ${t.toLowerCase()}...`} style={{ ...iStyle, flex: 1 }} />
                            <button className="px-3 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#000' }}>Analyze</button>
                            <button className="px-2.5 py-2 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)' }} title="Filter & Sort">
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6" /><line x1="7" y1="12" x2="17" y2="12" /><line x1="10" y1="18" x2="14" y2="18" /></svg>
                            </button>
                          </div>
                          {(t === 'TF Conditions' && confPills.length > 0) && <div className="flex flex-wrap gap-1.5 mb-3">{confPills.map((p, i) => <Pill key={p} label={confluenceGroups?.find((g: any) => g.id === p)?.base_template || p} onRemove={() => handleToggleCondition(p)} />)}</div>}
                          <div className="overflow-y-auto" style={{ maxHeight: 440 }}>
                            <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>Click <strong>Analyze</strong> to compare {t.toLowerCase()} options using the current config.</p>
                          </div>
                        </div>
                      );
                    }}
                  </TabBar>
                </div>
                <div className="flex-shrink-0 -mb-2">
                  {tab === 'Entry' && <Depth depth={1} max={1} onChange={() => {}} />}
                  {tab === 'Exit' && <Depth depth={exitD} max={3} onChange={setExitD} />}
                  {tab === 'TF Conditions' && <Depth depth={tfD} max={4} onChange={setTfD} />}
                  {tab === 'General' && <Depth depth={genD} max={4} onChange={setGenD} />}
                  {tab === 'Stop Loss' && <Depth depth={1} max={1} onChange={() => {}} />}
                  {tab === 'Take Profit' && <Depth depth={1} max={1} onChange={() => {}} />}
                </div>
              </Card>
            </div>
            <Card>
              <div className="flex items-center justify-between text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                <span>Advanced Analysis</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Streaks, TOD, DOW, Markov</span>
              </div>
              <p className="text-xs mt-2 py-4 text-center" style={{ color: 'var(--text-muted)' }}>Advanced analysis populates after a backtest with sufficient trades.</p>
            </Card>
          </div>
        </div>

        {/* SAVE */}
        <div className="mt-6 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm">{result?.trades?.length || 0} trades over {result?.total_bars?.toLocaleString() || 0} bars</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Source: {result?.data_source || 'Unknown'}</p>
            </div>
            <div className="flex items-center gap-3">
              <input type="text" placeholder="Strategy name" value={name} onChange={e => setName(e.target.value)} className="w-48" style={iStyle} />
              <button className="px-6 py-3 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: '#000', border: 'none', cursor: 'pointer' }} onClick={save} disabled={createMut.isPending}>{createMut.isPending ? 'Saving...' : 'Save Strategy'}</button>
            </div>
          </div>
        </div>
      </>}
    </div>
  );
}
