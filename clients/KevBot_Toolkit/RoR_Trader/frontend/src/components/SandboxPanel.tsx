'use client';

/**
 * SandboxPanel — Shared backtest + chart environment for testing confluence packs.
 *
 * Used by:
 * - PackBuilderPage Step 5 (layout="sidebar", 3/9 column split)
 * - UserPacksPage detail view (layout="horizontal", full-width)
 */

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import MetricCard from '@/components/MetricCard';
import { useRunBacktest, useBacktestTradeZoom } from '@/hooks/queries/useBacktest';
import { useConfluenceTriggers, useStopLossPacks, useTakeProfitPacks } from '@/hooks/queries/usePacks';
import { useChartPrefs } from '@/hooks/useChartPrefs';
import { buildStrategyChartPanes } from '@/charts/buildStrategyChartPanes';
import EquityCurve from '@/charts/EquityCurve';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });
const TradeZoomModal = dynamic(() => import('@/components/TradeZoomModal'), { ssr: false });

const SB_TIMEFRAMES = ['1Min', '5Min', '15Min', '1H', '1D'] as const;

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)', border: '1px solid var(--border)',
  color: 'var(--text-primary)', padding: '6px 10px', borderRadius: '8px',
  fontSize: '0.8rem', width: '100%',
};
const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)', color: 'white', border: 'none',
  padding: '8px 16px', borderRadius: '8px', fontSize: '0.875rem',
  cursor: 'pointer', fontWeight: 600,
};

function ExecBadge({ tag }: { tag: string }) {
  return <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full" style={{ color: '#2196F3', background: '#2196F320' }}>{tag}</span>;
}

interface SandboxPanelProps {
  packSlug: string;
  layout?: 'horizontal' | 'sidebar';
}

export default function SandboxPanel({ packSlug, layout = 'horizontal' }: SandboxPanelProps) {
  // Config state
  const [sbSymbol, setSbSymbol] = useState('NVDA');
  const [sbTimeframe, setSbTimeframe] = useState('5Min');
  const [sbDirection, setSbDirection] = useState<'LONG' | 'SHORT'>('LONG');
  const [sbDays, setSbDays] = useState(30);
  const [sbEntryTrigger, setSbEntryTrigger] = useState('');
  const [sbExitTrigger, setSbExitTrigger] = useState('');
  const [sbStopPack, setSbStopPack] = useState('');
  const [sbTargetPack, setSbTargetPack] = useState('');
  const [sbHifi, setSbHifi] = useState(false);
  const [sbConfluence, setSbConfluence] = useState('');
  const [sbSecondaryTfs, setSbSecondaryTfs] = useState('');
  const [sbEqXAxis, setSbEqXAxis] = useState<'trade' | 'time'>('trade');
  const [sbZoomTrade, setSbZoomTrade] = useState<{ idx: number; side: 'entry' | 'exit'; trade: any } | null>(null);
  const [sbLastConfig, setSbLastConfig] = useState<any>(null);

  // Hooks (must all be called unconditionally)
  const sbBacktestMut = useRunBacktest();
  const sbTradeZoomMut = useBacktestTradeZoom();
  const chartPrefs = useChartPrefs();
  const { data: sbEntryTriggers } = useConfluenceTriggers(sbDirection);
  const { data: sbExitTriggers } = useConfluenceTriggers('EXIT');
  const { data: sbStopPacks } = useStopLossPacks();
  const { data: sbTargetPacks } = useTakeProfitPacks();

  // Derived: filter triggers to this pack
  const sbPackTriggers = useMemo(() => {
    if (!sbEntryTriggers || !packSlug) return [];
    return Object.entries(sbEntryTriggers)
      .filter(([id]) => id.includes(packSlug))
      .map(([id, name]) => ({ id, name: String(name) }));
  }, [sbEntryTriggers, packSlug]);

  const sbExitTriggerList = useMemo(() => {
    if (!sbExitTriggers) return [];
    return Object.entries(sbExitTriggers).map(([id, name]) => ({ id, name: String(name) }));
  }, [sbExitTriggers]);

  const sbStopPackList = useMemo(() => {
    if (!sbStopPacks) return [];
    return (sbStopPacks as any[]).map((p) => ({ id: p.id, label: `${p.base_template} (${p.version})` }));
  }, [sbStopPacks]);

  const sbTargetPackList = useMemo(() => {
    if (!sbTargetPacks) return [];
    return (sbTargetPacks as any[]).map((p) => ({ id: p.id, label: `${p.base_template} (${p.version})` }));
  }, [sbTargetPacks]);

  const sbResult = useMemo(() => {
    if (!sbBacktestMut.data) return null;
    const d = sbBacktestMut.data;
    return {
      trades: (d.trades || []).map((t: any, i: number) => ({
        id: i + 1,
        entryTime: t.entry_time ? new Date(t.entry_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
        exitTime: t.exit_time ? new Date(t.exit_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
        direction: (t.direction || 'LONG') as 'LONG' | 'SHORT',
        entryPrice: t.entry_price || 0, exitPrice: t.exit_price || 0,
        rMultiple: t.r_multiple || 0,
        execType: ({ HM: 'LC', HL: 'LC', L0: 'L', L1: 'L' } as Record<string, string>)[t.exec_type] || t.exec_type || 'C',
        exitReason: t.exit_reason || '',
      })),
      kpis: {
        winRate: d.kpis?.win_rate ?? 0, profitFactor: d.kpis?.profit_factor ?? 0,
        dailyR: d.kpis?.daily_r ?? 0, totalTrades: d.kpis?.total_trades ?? 0,
        totalR: d.kpis?.total_r ?? 0, avgR: d.kpis?.avg_r ?? 0,
        rSquared: d.kpis?.r_squared ?? 0, maxRDrawdown: d.kpis?.max_r_drawdown ?? 0,
      },
      equityCurve: d.equity_curve || [],
      chartData: d.chart_data,
      rawTrades: d.trades || [],
      overlay_indicators: (d as any).overlay_indicators || [],
      oscillator_indicators: (d as any).oscillator_indicators || [],
      heatmap_conditions: (d as any).heatmap_conditions || [],
    };
  }, [sbBacktestMut.data]);

  const sbChartPanes = useMemo(() => {
    if (!sbResult?.chartData || sbResult.chartData.length === 0) return [];
    const tfMs = (() => {
      const tf = sbTimeframe;
      if (tf.includes('Min')) return parseInt(tf) * 60 * 1000;
      if (tf.includes('H')) return 3600 * 1000;
      if (tf.includes('D')) return 86400 * 1000;
      return 60000;
    })();
    return buildStrategyChartPanes({
      bars: sbResult.chartData,
      trades: sbResult.rawTrades,
      direction: sbDirection,
      overlayNames: sbResult.overlay_indicators,
      oscNames: sbResult.oscillator_indicators,
      heatmapConds: sbResult.heatmap_conditions,
      tfMs,
      chartPrefs,
    });
  }, [sbResult, sbDirection, sbTimeframe, chartPrefs]);

  function handleRunBacktest() {
    if (!sbEntryTrigger) return;
    // Detect bar count exit trigger and pass N as strategy-level param
    const exitIds = sbExitTrigger ? [sbExitTrigger] : [];
    const hasBarCountExit = exitIds.some(t => t.includes('bar_count'));
    const req: any = {
      symbol: sbSymbol, timeframe: sbTimeframe, direction: sbDirection, days: sbDays,
      entry_trigger_confluence_id: sbEntryTrigger,
      exit_trigger_confluence_ids: exitIds,
      stop_loss_pack_id: sbStopPack || undefined,
      take_profit_pack_id: sbTargetPack || undefined,
      hifi_mode: sbHifi, include_chart_data: true,
      ...(hasBarCountExit ? { bar_count_exit: 4 } : {}),
      ...(sbConfluence.trim() ? { confluence: sbConfluence.split(',').map(s => s.trim()).filter(Boolean) } : {}),
      ...(sbSecondaryTfs.trim() ? { secondary_tfs: sbSecondaryTfs.split(',').map(s => s.trim()).filter(Boolean) } : {}),
    };
    setSbLastConfig(req);
    sbBacktestMut.mutate(req);
  }

  // ---- Config Controls ----
  const configControls = (
    <>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Symbol</label>
        <input type="text" value={sbSymbol} onChange={(e) => setSbSymbol(e.target.value.toUpperCase())} style={inputStyle} />
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
        <select value={sbTimeframe} onChange={(e) => setSbTimeframe(e.target.value)} style={inputStyle}>
          {SB_TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
        </select>
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
        <select value={sbDirection} onChange={(e) => setSbDirection(e.target.value as 'LONG' | 'SHORT')} style={inputStyle}>
          <option value="LONG">LONG</option>
          <option value="SHORT">SHORT</option>
        </select>
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Days</label>
        <input type="number" value={sbDays} min={1} max={365} onChange={(e) => setSbDays(parseInt(e.target.value) || 30)} style={inputStyle} />
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Entry Trigger</label>
        <select value={sbEntryTrigger} onChange={(e) => setSbEntryTrigger(e.target.value)} style={inputStyle}>
          <option value="">Select...</option>
          {sbPackTriggers.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
        {sbPackTriggers.length === 0 && <p className="text-[10px] mt-0.5" style={{ color: 'var(--orange)' }}>No triggers found. Try refreshing.</p>}
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Exit Trigger</label>
        <select value={sbExitTrigger} onChange={(e) => setSbExitTrigger(e.target.value)} style={inputStyle}>
          <option value="">None (stop/target only)</option>
          {sbExitTriggerList.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Stop Loss</label>
        <select value={sbStopPack} onChange={(e) => setSbStopPack(e.target.value)} style={inputStyle}>
          <option value="">None</option>
          {sbStopPackList.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Take Profit</label>
        <select value={sbTargetPack} onChange={(e) => setSbTargetPack(e.target.value)} style={inputStyle}>
          <option value="">None</option>
          {sbTargetPackList.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Confluence</label>
        <input type="text" value={sbConfluence} onChange={(e) => setSbConfluence(e.target.value)}
          placeholder="e.g. 5M-EMA_STACK-SML"
          title="Comma-separated confluence records required for entry. Format: TF-INTERPRETER-STATE"
          style={{ ...inputStyle, fontSize: '0.65rem' }} />
      </div>
      <div>
        <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Secondary TFs</label>
        <input type="text" value={sbSecondaryTfs} onChange={(e) => setSbSecondaryTfs(e.target.value)}
          placeholder="e.g. 15Min,1H"
          title="Comma-separated secondary timeframes for cross-TF confluence"
          style={{ ...inputStyle, fontSize: '0.65rem' }} />
      </div>
    </>
  );

  // ---- Results Panel ----
  const resultsPanel = (
    <div className="space-y-4">
      {sbBacktestMut.isPending && (
        <Card>
          <div className="flex items-center gap-3 py-4">
            <div className="w-4 h-4 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Running backtest on {sbSymbol}...</span>
          </div>
        </Card>
      )}

      {sbBacktestMut.isError && (
        <Card>
          <p className="text-sm" style={{ color: 'var(--red)' }}>
            Backtest failed: {sbBacktestMut.error instanceof Error ? sbBacktestMut.error.message : 'Unknown error'}
          </p>
        </Card>
      )}

      {!sbResult && !sbBacktestMut.isPending && !sbBacktestMut.isError && (
        <Card>
          <div className="text-center py-8">
            <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Configure and click <strong>Run Backtest</strong> to test the pack.</p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{sbSymbol} | {sbDirection} | {sbTimeframe} | {sbDays} days</p>
          </div>
        </Card>
      )}

      {sbResult && (
        <>
          {/* KPI Strip */}
          <div className="grid grid-cols-4 xl:grid-cols-8 gap-2">
            <MetricCard label="Trades" value={String(sbResult.kpis.totalTrades)} />
            <MetricCard label="Win Rate" value={`${sbResult.kpis.winRate.toFixed(1)}%`} positive={sbResult.kpis.winRate > 50} />
            <MetricCard label="PF" value={sbResult.kpis.profitFactor.toFixed(2)} positive={sbResult.kpis.profitFactor > 1} />
            <MetricCard label="Avg R" value={`${sbResult.kpis.avgR >= 0 ? '+' : ''}${sbResult.kpis.avgR.toFixed(2)}`} positive={sbResult.kpis.avgR > 0} />
            <MetricCard label="Total R" value={`${sbResult.kpis.totalR >= 0 ? '+' : ''}${sbResult.kpis.totalR.toFixed(1)}`} positive={sbResult.kpis.totalR > 0} />
            <MetricCard label="Daily R" value={`${sbResult.kpis.dailyR >= 0 ? '+' : ''}${sbResult.kpis.dailyR.toFixed(2)}`} positive={sbResult.kpis.dailyR > 0} />
            <MetricCard label="R²" value={sbResult.kpis.rSquared.toFixed(2)} positive={sbResult.kpis.rSquared > 0.7} />
            <MetricCard label="Max DD" value={`${sbResult.kpis.maxRDrawdown.toFixed(1)}R`} positive={false} />
          </div>

          {/* Price Chart */}
          <Card>
            <div style={{ minHeight: 450 }}>
              {sbChartPanes.length > 0 ? (
                <SyncedChartPane
                  panes={sbChartPanes}
                  upColor={chartPrefs?.candleUp}
                  downColor={chartPrefs?.candleDown}
                  upBorderColor={chartPrefs?.candleUpBorder}
                  gridLines={chartPrefs?.gridLines}
                  rightOffset={chartPrefs?.rightOffset}
                />
              ) : (
                <ChartPlaceholder label="No chart data available" height={400} />
              )}
            </div>
          </Card>

          {/* Equity Curve */}
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium">Equity Curve</h4>
              <div className="flex gap-1">
                {(['trade', 'time'] as const).map((mode) => (
                  <button key={mode} className="px-2 py-0.5 rounded text-[10px]"
                    style={{ background: sbEqXAxis === mode ? 'var(--accent-muted)' : 'transparent', color: sbEqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: sbEqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}
                    onClick={() => setSbEqXAxis(mode)}>
                    {mode === 'trade' ? 'Per Trade' : 'Per Day'}
                  </button>
                ))}
              </div>
            </div>
            {sbResult.equityCurve.length > 0 ? (
              <EquityCurve
                data={sbResult.equityCurve.map((pt: any, i: number) => ({
                  trade_number: i + 1, cumulative_r: pt.cumulative_r ?? 0, timestamp: pt.timestamp,
                }))}
                height={250} showZeroLine xAxis={sbEqXAxis}
              />
            ) : (
              <ChartPlaceholder label="No equity data" height={200} />
            )}
          </Card>

          {/* Trade History */}
          <Card>
            <h4 className="text-sm font-medium mb-2">Trade History <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({sbResult.trades.length} trades — click to drill down)</span></h4>
            <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
              <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['#', 'Entry', 'Exit', 'Dir', 'Entry $', 'Exit $', 'P&L (R)', 'Exec', 'Exit Reason'].map((h) => (
                      <th key={h} className="text-left py-1.5 px-2 text-[10px] font-medium sticky top-0" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sbResult.trades.length === 0 ? (
                    <tr><td colSpan={9} className="py-4 px-2 text-center" style={{ color: 'var(--text-muted)' }}>No trades</td></tr>
                  ) : sbResult.trades.map((t: any) => (
                    <tr key={t.id} style={{ borderBottom: '1px solid var(--border)', cursor: sbLastConfig ? 'pointer' : undefined }}
                      onClick={() => {
                        if (!sbLastConfig) return;
                        setSbZoomTrade({ idx: t.id - 1, side: 'entry', trade: t });
                        sbTradeZoomMut.mutate({ ...sbLastConfig, trade_idx: t.id - 1, side: 'entry' });
                      }}>
                      <td className="px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>{t.id}</td>
                      <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-secondary)', fontSize: '0.65rem' }}>{t.entryTime}</td>
                      <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-secondary)', fontSize: '0.65rem' }}>{t.exitTime}</td>
                      <td className="px-2 py-1.5"><span style={{ color: t.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{t.direction}</span></td>
                      <td className="px-2 py-1.5 text-right font-mono">${t.entryPrice.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right font-mono">${t.exitPrice.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right font-mono" style={{ color: t.rMultiple >= 0 ? 'var(--green)' : 'var(--red)' }}>{t.rMultiple >= 0 ? '+' : ''}{t.rMultiple.toFixed(2)}R</td>
                      <td className="px-2 py-1.5"><ExecBadge tag={`[${t.execType}]`} /></td>
                      <td className="px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>{t.exitReason.replace(/_/g, ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* Trade Zoom Modal */}
      {sbZoomTrade && (
        <TradeZoomModal
          isOpen={!!sbZoomTrade}
          onClose={() => { setSbZoomTrade(null); sbTradeZoomMut.reset(); }}
          tradeIdx={sbZoomTrade.idx}
          side={sbZoomTrade.side}
          trade={{
            entry_price: sbZoomTrade.trade.entryPrice,
            exit_price: sbZoomTrade.trade.exitPrice,
            r_multiple: sbZoomTrade.trade.rMultiple,
            exec_type: sbZoomTrade.trade.execType,
            exit_reason: sbZoomTrade.trade.exitReason,
          }}
          zoomData={sbTradeZoomMut.data ?? null}
          isLoading={sbTradeZoomMut.isPending}
          error={sbTradeZoomMut.error ? String(sbTradeZoomMut.error) : null}
        />
      )}
    </div>
  );

  // ---- Layout ----
  if (layout === 'horizontal') {
    return (
      <div className="space-y-4">
        {/* Horizontal config bar */}
        <Card>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-9 gap-3 items-end">
            {configControls}
            <div className="flex flex-col gap-1">
              <label className="flex items-center gap-2 text-[10px] cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                <input type="checkbox" checked={sbHifi} onChange={(e) => setSbHifi(e.target.checked)} style={{ accentColor: 'var(--accent)' }} />
                Hi-Fi
              </label>
              <button style={{ ...btnPrimary, width: '100%', fontSize: '0.75rem', padding: '6px 12px', opacity: (!sbEntryTrigger || sbBacktestMut.isPending) ? 0.5 : 1 }}
                disabled={!sbEntryTrigger || sbBacktestMut.isPending}
                onClick={handleRunBacktest}>
                {sbBacktestMut.isPending ? 'Running...' : 'Run Backtest'}
              </button>
            </div>
          </div>
        </Card>

        {resultsPanel}
      </div>
    );
  }

  // sidebar layout (Pack Builder Step 5)
  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 lg:col-span-3">
        <Card>
          <h4 className="text-sm font-medium mb-3">Backtest Config</h4>
          <div className="space-y-3">
            {configControls}
            <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
              <input type="checkbox" checked={sbHifi} onChange={(e) => setSbHifi(e.target.checked)} style={{ accentColor: 'var(--accent)' }} />
              Hi-Fi Mode
            </label>
            <button style={{ ...btnPrimary, width: '100%', opacity: (!sbEntryTrigger || sbBacktestMut.isPending) ? 0.5 : 1 }}
              disabled={!sbEntryTrigger || sbBacktestMut.isPending}
              onClick={handleRunBacktest}>
              {sbBacktestMut.isPending ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        </Card>
      </div>
      <div className="col-span-12 lg:col-span-9">
        {resultsPanel}
      </div>
    </div>
  );
}
