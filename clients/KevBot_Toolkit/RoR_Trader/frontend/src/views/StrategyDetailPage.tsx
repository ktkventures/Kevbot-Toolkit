'use client';

/**
 * Strategy Detail — Clean API-first page.
 * Visual design from V5, data wired to real API hooks. No mock data.
 */

import { useState, useMemo, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import EquityCurve from '@/charts/EquityCurve';
import DistributionChart from '@/charts/DistributionChart';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import {
  useStrategy, useStrategyTrades, useStrategyForwardTest, useStrategyKPIs,
  type TradeDTO,
} from '@/hooks/queries/useStrategies';
import { useStrategyAlerts } from '@/hooks/queries/useAlerts';
import { useDeleteStrategy, useDuplicateStrategy } from '@/hooks/mutations/useStrategyMutations';

/* ======================================================================= */
/* Constants & styles                                                       */
/* ======================================================================= */

const EXEC_BADGE_COLOR = '#2196F3';
const EQ_COLORS = { bt: '#2196F3', fwd: '#FF9800', live: '#4CAF50' };
const TABS = ['Equity & KPIs', 'Chart & Trades', 'Confluence Analysis', 'Configuration', 'Alerts', 'Alert Analysis'];
const PULSE_CSS = `@keyframes pulse{0%,100%{transform:scale(1);opacity:.5}50%{transform:scale(2.2);opacity:0}}`;

const execColors: Record<string, string> = { '[C]': EXEC_BADGE_COLOR, '[L]': 'var(--green)', '[LC]': '#AB47BC', '[CC]': '#FF9800' };
const exitColors: Record<string, string> = { Signal: 'var(--green)', Target: 'var(--green)', Stop: 'var(--red)', 'Bar Count': '#009688' };

const th: React.CSSProperties = { color: 'var(--text-muted)', background: 'var(--bg-secondary)', textAlign: 'left', padding: '8px 12px', fontSize: '.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.05em' };
const td: React.CSSProperties = { padding: '8px 12px', fontSize: '.8125rem', borderBottom: '1px solid var(--border)', color: 'var(--text-secondary)' };
const btnSec: React.CSSProperties = { background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', padding: '6px 14px', borderRadius: '8px', fontSize: '.875rem', cursor: 'pointer' };
const sel: React.CSSProperties = { background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '6px 12px', borderRadius: '8px', fontSize: '.875rem' };

/* ======================================================================= */
/* Helpers                                                                  */
/* ======================================================================= */

const fmtR = (v?: number | null) => v == null ? '--' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}R`;
const fmtPct = (v?: number | null) => v == null ? '--' : `${v.toFixed(1)}%`;
const fmtNum = (v?: number | null, d = 2) => v == null ? '--' : v.toFixed(d);

function Skeleton({ h = 200 }: { h?: number }) {
  return <div className="rounded-xl animate-pulse" style={{ background: 'var(--bg-input)', height: h }} />;
}

function Badge({ text, color }: { text: string; color: string }) {
  return <span className="text-xs px-2 py-0.5 rounded-full" style={{ color, background: color + '20' }}>{text}</span>;
}

function PillTabs({ tabs, active, onChange }: { tabs: string[]; active: string; onChange: (t: string) => void }) {
  return (
    <div className="flex gap-1 mb-4 flex-wrap">
      {tabs.map((t) => (
        <button key={t} className="text-xs px-3 py-1.5 rounded-full transition-colors" onClick={() => onChange(t)}
          style={{ background: active === t ? 'var(--accent)' : 'var(--bg-input)', color: active === t ? 'white' : 'var(--text-muted)', border: active === t ? 'none' : '1px solid var(--border)', cursor: 'pointer' }}
        >{t}</button>
      ))}
    </div>
  );
}

/* ======================================================================= */
/* Main component                                                           */
/* ======================================================================= */

export default function StrategyDetailPage({ strategyId }: { strategyId: number }) {
  const router = useRouter();

  // Data
  const { data: strategy, isLoading } = useStrategy(strategyId || null);
  const { data: trades } = useStrategyTrades(strategyId || null);
  const { data: forwardTest } = useStrategyForwardTest(strategyId || null);
  const { data: kpiData } = useStrategyKPIs(strategyId || null);
  const { data: alerts } = useStrategyAlerts(strategyId || null);

  // Mutations
  const deleteMut = useDeleteStrategy();
  const dupMut = useDuplicateStrategy();

  // State
  const [kpiMode, setKpiMode] = useState('Overall');
  const [extKpiTab, setExtKpiTab] = useState('Performance');
  const [showExt, setShowExt] = useState(true);
  const [hwm, setHwm] = useState(false);
  const [fwdOpen, setFwdOpen] = useState(true);
  const [btOpen, setBtOpen] = useState(false);
  const [confirmDel, setConfirmDel] = useState(false);

  useEffect(() => {
    if (!document.getElementById('sd-pulse')) {
      const s = document.createElement('style'); s.id = 'sd-pulse'; s.textContent = PULSE_CSS; document.head.appendChild(s);
    }
  }, []);

  // Derived
  const kpis = strategy?.kpis ?? kpiData?.kpis;
  const secKpis = kpiData?.secondary_kpis;
  const eqPts = useMemo(() => {
    const d = strategy?.equity_curve_data;
    if (!d?.cumulative_r) return [];
    return d.cumulative_r.map((r: number, i: number) => ({ trade_number: i + 1, timestamp: d.exit_times?.[i], cumulative_r: r }));
  }, [strategy]);
  const bIdx = strategy?.equity_curve_data?.boundary_index ?? null;
  const btTrades = useMemo(() => forwardTest?.backtest_trades ?? [], [forwardTest]);
  const fwdTrades = useMemo(() => forwardTest?.forward_trades ?? [], [forwardTest]);
  const rVals = useMemo(() => (trades ?? []).map((t: TradeDTO) => t.r_multiple).filter((v): v is number => v != null), [trades]);
  const confluence = strategy?.confluence ?? [];

  // Handlers
  const handleDel = () => { if (!confirmDel) { setConfirmDel(true); return; } deleteMut.mutate(strategyId, { onSuccess: () => router.push('/strategies') }); };
  const handleDup = () => dupMut.mutate(strategyId);

  // Loading
  if (isLoading || !strategy) {
    return <div className="flex flex-col gap-4 p-4"><Skeleton h={60} /><Skeleton h={40} /><Skeleton h={300} /><div className="grid grid-cols-3 gap-3"><Skeleton h={80} /><Skeleton h={80} /><Skeleton h={80} /></div></div>;
  }

  const s = strategy;
  return (
    <div>
      {/* Header */}
      <PageHeader title={s.name} backHref="/strategies" actions={<>
        <button style={btnSec} onClick={handleDup}>{dupMut.isPending ? 'Cloning...' : 'Clone'}</button>
        <button style={{ ...btnSec, background: confirmDel ? 'var(--red)' : 'var(--red-muted)', color: confirmDel ? 'white' : 'var(--red)', border: 'none' }} onClick={handleDel}>{confirmDel ? 'Confirm Delete' : 'Delete'}</button>
      </>} />

      {/* Badges */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <Badge text={s.direction} color={s.direction === 'LONG' ? 'var(--green)' : 'var(--red)'} />
        {s.forward_testing && <Badge text="Forward Testing" color={EQ_COLORS.fwd} />}
        {s.alert_tracking_enabled && (
          <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--green)' }}>
            <span style={{ position: 'relative', display: 'inline-block', width: 8, height: 8 }}>
              <span style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: 'var(--green)', opacity: .5, animation: 'pulse 2s ease-in-out infinite' }} />
              <span style={{ position: 'absolute', inset: '25%', borderRadius: '50%', background: 'var(--green)' }} />
            </span>Live
          </span>
        )}
        {(s.tags ?? []).map((t: string) => (
          <span key={t} className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)', border: '1px solid var(--border)' }}>{t}</span>
        ))}
      </div>

      {/* Meta */}
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        {s.symbol} &middot; {s.timeframe}{s.trading_session ? ` \u00B7 ${s.trading_session}` : ''}{s.data_days ? ` \u00B7 ${s.data_days}d` : ''}{s.created_at ? ` \u00B7 ${new Date(s.created_at).toLocaleDateString()}` : ''}
      </p>

      {/* Strategy Variables (entry/exit/stop/target/confluence) */}
      <Card className="mb-4">
        <div className="flex flex-col gap-2">
          {/* Entry */}
          {s.entry_trigger_confluence_id && (
            <div className="flex items-center gap-2 flex-wrap text-xs">
              <span style={{ color: 'var(--text-muted)', minWidth: 48 }}>entry:</span>
              <span className="px-1.5 py-0.5 rounded-full font-mono font-medium" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20', fontSize: 10 }}>[C]</span>
              <span style={{ color: 'var(--text-secondary)' }}>{s.entry_trigger_confluence_id}</span>
            </div>
          )}
          {/* Exit */}
          {(s.exit_trigger_confluence_ids?.length ?? 0) > 0 && (
            <div className="flex items-center gap-2 flex-wrap text-xs">
              <span style={{ color: 'var(--text-muted)', minWidth: 48 }}>exit:</span>
              {s.exit_trigger_confluence_ids!.map((ex: string, i: number) => (
                <span key={i} className="font-mono" style={{ color: 'var(--text-secondary)' }}>
                  <span className="px-1.5 py-0.5 rounded-full font-medium mr-1" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20', fontSize: 10 }}>[C]</span>
                  {ex}{i < s.exit_trigger_confluence_ids!.length - 1 ? ' , ' : ''}
                </span>
              ))}
            </div>
          )}
          {/* Stop + Target */}
          <div className="flex items-center gap-2 flex-wrap text-xs">
            <span style={{ color: 'var(--text-muted)', minWidth: 48 }}>stop:</span>
            <span className="px-2 py-0.5 rounded font-mono" style={{ color: 'var(--red)', background: 'var(--red)18' }}>
              {s.stop_config?.method ? `${s.stop_config.method}${s.stop_config.atr_mult ? ` ${s.stop_config.atr_mult}x` : ''}` : 'None'}
            </span>
            <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>target:</span>
            <span className="px-2 py-0.5 rounded font-mono" style={{ color: 'var(--green)', background: 'var(--green)18' }}>
              {s.target_config?.method || 'Signal exit only'}
            </span>
          </div>
          {/* Confluence */}
          <div className="flex items-center gap-2 flex-wrap text-xs">
            <span style={{ color: 'var(--text-muted)', minWidth: 48 }}>confluence:</span>
            {confluence.length > 0 ? confluence.map((c: string) => (
              <span key={c} className="flex items-center gap-1 font-mono">
                <span className="px-1 py-0.5 rounded-full font-medium" style={{ color: '#26C6DA', background: '#26C6DA20', fontSize: 9 }}>[PB]</span>
                <span className="px-1.5 py-0.5 rounded" style={{ color: 'var(--accent)', background: 'var(--accent)15' }}>{c}</span>
              </span>
            )) : <span style={{ color: 'var(--text-muted)' }}>none</span>}
          </div>
        </div>
      </Card>

      {/* Tabs */}
      <TabBar tabs={TABS}>{(tab) => (<div>
        {/* TAB 1: Equity & KPIs */}
        {tab === TABS[0] && (<div>
          <div className="flex items-center gap-4 mb-4">
            <label className="text-xs" style={{ color: 'var(--text-muted)' }}>KPI Mode:</label>
            <select style={sel} value={kpiMode} onChange={(e) => setKpiMode(e.target.value)}>
              {['Overall', 'BT vs FWD'].map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
          {kpiMode === 'Overall' && kpis && (
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mb-6">
              {[{ l: 'Win Rate', v: fmtPct(kpis.win_rate) }, { l: 'PF', v: fmtNum(kpis.profit_factor) }, { l: 'Daily R', v: fmtR(kpis.daily_r)?.replace('R', '') }, { l: 'Total R', v: fmtR(kpis.total_r) }, { l: 'Trades', v: String(kpis.total_trades ?? '--') }, { l: 'Max DD', v: fmtR(kpis.max_r_drawdown) }].map((k) => (
                <Card key={k.l}><p className="text-xs" style={{ color: 'var(--text-muted)' }}>{k.l}</p><p className="text-lg font-bold mt-1">{k.v}</p></Card>
              ))}
            </div>
          )}
          {kpiMode === 'BT vs FWD' && forwardTest && (
            <Card className="mb-6"><div style={{ overflowX: 'auto' }}>
              <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                <thead><tr><th style={th}>Metric</th><th style={{ ...th, textAlign: 'right', color: EQ_COLORS.bt }}>BT ({btTrades.length})</th><th style={{ ...th, textAlign: 'right', color: EQ_COLORS.fwd }}>FWD ({fwdTrades.length})</th></tr></thead>
                <tbody>{(() => {
                  const wr = (t: TradeDTO[]) => t.length ? (t.filter((x) => x.win).length / t.length) * 100 : 0;
                  const tot = (t: TradeDTO[]) => t.reduce((s, x) => s + (x.r_multiple ?? 0), 0);
                  return [{ l: 'Win Rate', b: fmtPct(wr(btTrades)), f: fmtPct(wr(fwdTrades)) }, { l: 'Total R', b: fmtR(tot(btTrades)), f: fmtR(tot(fwdTrades)) }, { l: 'Trades', b: String(btTrades.length), f: String(fwdTrades.length) }].map((r) => (
                    <tr key={r.l}><td style={td}>{r.l}</td><td style={{ ...td, textAlign: 'right' }}>{r.b}</td><td style={{ ...td, textAlign: 'right' }}>{r.f}</td></tr>
                  ));
                })()}</tbody>
              </table>
            </div></Card>
          )}
          {/* Extended KPIs */}
          {secKpis && (<div className="mb-6">
            <button className="flex items-center gap-2 text-sm font-medium mb-3" style={{ color: 'var(--text-primary)', cursor: 'pointer', background: 'none', border: 'none', padding: 0 }} onClick={() => setShowExt(!showExt)}>
              <span style={{ color: 'var(--text-muted)', fontSize: '.75rem' }}>{showExt ? '\u25BC' : '\u25B6'}</span> Extended KPIs
            </button>
            {showExt && <>
              <PillTabs tabs={['Performance', 'Risk-Adjusted', 'Drawdown']} active={extKpiTab} onChange={setExtKpiTab} />
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {(extKpiTab === 'Performance' ? [{ l: 'Avg R', v: fmtR(secKpis.avg_r) }, { l: 'Avg Win', v: fmtR(secKpis.avg_win) }, { l: 'Avg Loss', v: fmtR(secKpis.avg_loss) }, { l: 'R\u00B2', v: fmtNum(secKpis.r_squared) }]
                  : extKpiTab === 'Risk-Adjusted' ? [{ l: 'Sharpe', v: fmtNum(secKpis.sharpe) }, { l: 'Sortino', v: fmtNum(secKpis.sortino) }, { l: 'Calmar', v: fmtNum(secKpis.calmar) }, { l: 'Kelly', v: fmtPct(secKpis.kelly ? secKpis.kelly * 100 : null) }]
                  : [{ l: 'Max R DD', v: fmtR(secKpis.max_r_drawdown) }, { l: 'Recovery', v: fmtNum(secKpis.recovery_factor, 1) }, { l: 'Consec Wins', v: String(secKpis.max_consec_wins ?? '--') }, { l: 'Consec Losses', v: String(secKpis.max_consec_losses ?? '--') }]
                ).map((k) => <Card key={k.l}><p className="text-xs" style={{ color: 'var(--text-muted)' }}>{k.l}</p><p className="text-base font-bold mt-1">{k.v}</p></Card>)}
              </div>
            </>}
          </div>)}
          {/* Equity Curve */}
          <Card className="mb-4">
            <h4 className="text-sm font-medium mb-3">Equity Curve</h4>
            {eqPts.length > 0 ? <EquityCurve data={eqPts} boundaryIndex={bIdx} height={300} showHWM={hwm} /> : <ChartPlaceholder label="No equity data" height={300} />}
            <div className="flex items-center gap-6 mt-3 text-xs" style={{ color: 'var(--text-muted)' }}>
              {Object.entries(EQ_COLORS).map(([k, c]) => <span key={k} className="flex items-center gap-1.5"><span style={{ display: 'inline-block', width: 16, height: 2, background: c }} /> {k === 'bt' ? 'Backtest' : k === 'fwd' ? 'Forward' : 'Alerts'}</span>)}
              <label className="flex items-center gap-1.5 cursor-pointer"><input type="checkbox" checked={hwm} onChange={() => setHwm(!hwm)} /> HWM</label>
            </div>
          </Card>
          {/* R-Distribution: BT and Forward side by side */}
          {rVals.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
              <Card>
                <h4 className="text-sm font-medium mb-3" style={{ color: EQ_COLORS.bt }}>Backtest R-Distribution</h4>
                <DistributionChart values={btTrades.length > 0 ? btTrades.map((t: TradeDTO) => t.r_multiple) : rVals} height={200} />
              </Card>
              <Card>
                <h4 className="text-sm font-medium mb-3" style={{ color: EQ_COLORS.fwd }}>Forward Test R-Distribution</h4>
                {fwdTrades.length > 0 ? (
                  <DistributionChart values={fwdTrades.map((t: TradeDTO) => t.r_multiple)} height={200} />
                ) : (
                  <div className="flex items-center justify-center" style={{ height: 200, color: 'var(--text-muted)', fontSize: 13 }}>
                    No forward test trades yet
                  </div>
                )}
              </Card>
            </div>
          )}

          {/* Advanced Analysis */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Advanced Analysis</h4>
              <button className="text-xs" style={{ color: 'var(--text-muted)' }} onClick={() => setShowExt(!showExt)}>
                {showExt ? 'Collapse' : 'Expand'}
              </button>
            </div>
            {showExt && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="p-4 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Rolling Metrics</h5>
                  <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    Rolling win rate, profit factor, and Sharpe ratio over a configurable window.
                    Reveals how strategy performance evolves over time.
                  </p>
                  <div className="mt-3 h-24 rounded flex items-center justify-center" style={{ background: 'var(--bg-secondary)' }}>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Chart — coming soon</span>
                  </div>
                </div>
                <div className="p-4 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Return Distribution</h5>
                  <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    Histogram, box plot, and statistical summary of trade returns.
                    Skewness and kurtosis reveal tail risk characteristics.
                  </p>
                  <div className="mt-3 h-24 rounded flex items-center justify-center" style={{ background: 'var(--bg-secondary)' }}>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Chart — coming soon</span>
                  </div>
                </div>
                <div className="p-4 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Markov Motor</h5>
                  <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    Win/loss transition matrix, edge decay analysis, and market regime detection.
                    Shows if the strategy has persistent edge or is mean-reverting.
                  </p>
                  <div className="mt-3 h-24 rounded flex items-center justify-center" style={{ background: 'var(--bg-secondary)' }}>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Chart — coming soon</span>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>)}

        {/* TAB 2: Chart & Trades */}
        {tab === TABS[1] && (<div>
          <Card className="mb-4"><ChartPlaceholder label="Price chart with indicator overlays and trade markers — coming soon" height={400} /></Card>
          {/* Show FWD/BT split if available, otherwise show all trades */}
          {(fwdTrades.length > 0 || btTrades.length > 0) ? (<>
            <CollapseTrades label="Forward Test Trades" trades={fwdTrades} open={fwdOpen} toggle={() => setFwdOpen(!fwdOpen)} />
            <CollapseTrades label="Backtest Trades" trades={btTrades} open={btOpen} toggle={() => setBtOpen(!btOpen)} />
          </>) : trades && trades.length > 0 ? (
            <Card>
              <h4 className="text-sm font-medium mb-3">Trade History ({trades.length} trades)</h4>
              <TradeTable trades={trades} />
            </Card>
          ) : (
            <Card>
              <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                No trade data available. Trades are computed from stored_trades or via backtest.
              </div>
            </Card>
          )}
        </div>)}

        {/* TAB 3: Confluence Analysis */}
        {tab === TABS[2] && (<Card>
          {confluence.length === 0 ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No confluence conditions.</p> : <>
            <h4 className="text-sm font-medium mb-3">Confluence Conditions</h4>
            <div className="flex items-center gap-2 flex-wrap mb-4">{confluence.map((c: string) => <Badge key={c} text={c} color="var(--accent)" />)}</div>
            <ChartPlaceholder label="Per-group indicator analysis -- next iteration" height={300} />
          </>}
        </Card>)}

        {/* TAB 4: Configuration */}
        {tab === TABS[3] && (<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card><h4 className="text-sm font-medium mb-4">Strategy Setup</h4><div className="flex flex-col gap-3">
            {[{ l: 'Symbol', v: s.symbol }, { l: 'Direction', v: s.direction }, { l: 'Timeframe', v: s.timeframe }, { l: 'Session', v: s.trading_session ?? 'Default' }, { l: 'Data Days', v: s.data_days ? `${s.data_days}d` : '--' }, { l: 'Created', v: s.created_at ? new Date(s.created_at).toLocaleString() : '--' }, { l: 'FW Testing', v: s.forward_testing ? 'Yes' : 'No' }, { l: 'FW Start', v: s.forward_test_start ?? '--' }].map((r) => (
              <div key={r.l} className="flex items-center justify-between"><span className="text-xs" style={{ color: 'var(--text-muted)' }}>{r.l}</span><span className="text-sm font-medium">{r.v}</span></div>
            ))}
          </div></Card>
          <Card><h4 className="text-sm font-medium mb-4">Variables</h4><div className="flex flex-col gap-3">
            <CfgRow label="Entry Trigger" value={s.entry_trigger_confluence_id ?? '--'} />
            <CfgRow label="Exit Triggers" value={s.exit_trigger_confluence_ids?.join(', ') ?? '--'} />
            <CfgRow label="Stop Config" value={s.stop_config ? Object.entries(s.stop_config).map(([k, v]) => `${k}: ${v}`).join(', ') : '--'} />
            <CfgRow label="Target Config" value={s.target_config ? Object.entries(s.target_config).map(([k, v]) => `${k}: ${v}`).join(', ') : '--'} />
            {confluence.length > 0 && <div><span className="text-xs" style={{ color: 'var(--text-muted)' }}>Confluence</span><div className="flex items-center gap-2 flex-wrap mt-1">{confluence.map((c: string) => <Badge key={c} text={c} color="var(--accent)" />)}</div></div>}
          </div></Card>
        </div>)}

        {/* TAB 5: Alerts */}
        {tab === TABS[4] && (<Card>
          {!alerts?.length ? <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No alerts recorded.</p> : <>
            <h4 className="text-sm font-medium mb-3">Alert History ({alerts.length})</h4>
            <div style={{ overflowX: 'auto' }}>
              <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
                <thead><tr>{['Time', 'Type', 'Trigger', 'Price', 'Status'].map((h) => <th key={h} style={th}>{h}</th>)}</tr></thead>
                <tbody>{alerts.map((a: any, i: number) => (
                  <tr key={i}>
                    <td style={{ ...td, fontFamily: 'monospace', fontSize: '.75rem' }}>{a.timestamp ?? a.time ?? '--'}</td>
                    <td style={td}><span className="text-xs font-semibold px-2 py-0.5 rounded-full" style={{ color: a.type === 'ENTRY' ? 'var(--green)' : 'var(--red)', background: a.type === 'ENTRY' ? 'var(--green-muted)' : 'var(--red-muted)' }}>{a.type ?? a.event_type ?? '--'}</span></td>
                    <td style={td}>{a.trigger ?? a.trigger_name ?? '--'}</td>
                    <td style={td}>{a.price != null ? `$${Number(a.price).toFixed(2)}` : '--'}</td>
                    <td style={td}><span style={{ color: a.status === 'Delivered' ? 'var(--green)' : a.status === 'Failed' ? 'var(--red)' : 'var(--text-secondary)' }}>{a.status ?? '--'}</span></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </>}
        </Card>)}

        {/* TAB 6: Alert Analysis */}
        {tab === TABS[5] && (<Card>
          <h4 className="text-sm font-medium mb-3">Alert Analysis</h4>
          <ChartPlaceholder label="Discrepancy detection, position health -- next iteration" height={250} />
        </Card>)}
      </div>)}</TabBar>
    </div>
  );
}

/* ======================================================================= */
/* Sub-components                                                           */
/* ======================================================================= */

function CfgRow({ label, value }: { label: string; value: string }) {
  return <div><span className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</span><p className="text-sm mt-0.5" style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{value}</p></div>;
}

function CollapseTrades({ label, trades, open, toggle }: { label: string; trades: TradeDTO[]; open: boolean; toggle: () => void }) {
  return (
    <Card className="mb-4">
      <button className="flex items-center gap-2 w-full text-left text-sm font-medium mb-3" style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-primary)', padding: 0 }} onClick={toggle}>
        <span style={{ color: 'var(--text-muted)', fontSize: '.75rem' }}>{open ? '\u25BC' : '\u25B6'}</span>
        {label} <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({trades.length})</span>
      </button>
      {open && <TradeTable trades={trades} />}
    </Card>
  );
}

function TradeTable({ trades }: { trades: TradeDTO[] }) {
  if (!trades.length) return <p className="text-xs" style={{ color: 'var(--text-muted)' }}>No trades.</p>;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
        <thead><tr>{['#', 'Entry Time', 'Exit Time', 'Entry $', 'Exit $', 'R', 'Exec', 'Exit Reason'].map((h) => <th key={h} style={th}>{h}</th>)}</tr></thead>
        <tbody>{trades.map((t, i) => {
          const r = t.r_multiple; const tag = t.exec_type ? `[${t.exec_type}]` : '';
          return (
            <tr key={i}>
              <td style={td}>{i + 1}</td>
              <td style={{ ...td, fontFamily: 'monospace', fontSize: '.75rem' }}>{t.entry_time ?? '--'}</td>
              <td style={{ ...td, fontFamily: 'monospace', fontSize: '.75rem' }}>{t.exit_time ?? '--'}</td>
              <td style={td}>{t.entry_price != null ? `$${t.entry_price.toFixed(2)}` : '--'}</td>
              <td style={td}>{t.exit_price != null ? `$${t.exit_price.toFixed(2)}` : '--'}</td>
              <td style={{ ...td, color: r >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>{r >= 0 ? '+' : ''}{r.toFixed(2)}R</td>
              <td style={td}>{tag && <span className="text-xs font-mono px-1.5 py-0.5 rounded-full" style={{ color: execColors[tag] || EXEC_BADGE_COLOR, background: (execColors[tag] || EXEC_BADGE_COLOR) + '20' }}>{tag}</span>}</td>
              <td style={td}><span style={{ color: exitColors[t.exit_reason ?? ''] ?? 'var(--text-secondary)' }}>{t.exit_reason ?? '--'}</span></td>
            </tr>
          );
        })}</tbody>
      </table>
    </div>
  );
}
