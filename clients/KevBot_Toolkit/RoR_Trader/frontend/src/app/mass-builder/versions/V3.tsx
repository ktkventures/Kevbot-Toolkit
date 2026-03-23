'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES                                                                      */
/* ========================================================================= */

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
}

/* ========================================================================= */
/* CONSTANTS                                                                  */
/* ========================================================================= */

const AVAILABLE_TFS = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1Hour'];

const ENTRY_TRIGGERS = [
  'EMA Bull Cross', 'EMA Bear Cross', 'EMA Fan Open Bull',
  'MACD Cross Bull', 'MACD Cross Bear',
  'VWAP Cross Above', 'VWAP Cross Below',
  'UT Bot Buy', 'UT Bot Sell',
];

const TF_CONFLUENCES = [
  'EMA_STACK: SML', 'EMA_STACK: LMS',
  'MACD_LINE: M>S+', 'MACD_LINE: M<S-',
  'VWAP_POSITION: >V', 'VWAP_POSITION: <V',
  'RVOL: HIGH',
];

const SORT_OPTIONS = ['Daily R', 'Win Rate', 'Profit Factor', 'Trades'] as const;

/* ========================================================================= */
/* MOCK RESULTS                                                               */
/* ========================================================================= */

const mockResults: MassResult[] = [
  { rank: 1, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, status: 'active' },
  { rank: 2, ticker: 'SPY', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 58.3, pf: 2.45, dailyR: 1.95, trades: 124, status: 'active' },
  { rank: 3, ticker: 'NVDA', direction: 'LONG', tf: '1Min', trigger: 'UT Bot Buy', winRate: 54.0, pf: 2.05, dailyR: 1.78, trades: 201, status: 'active' },
  { rank: 4, ticker: 'TSLA', direction: 'LONG', tf: '1Min', trigger: 'EMA Bull Cross', winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 178, status: 'active' },
  { rank: 5, ticker: 'AAPL', direction: 'LONG', tf: '5Min', trigger: 'VWAP Cross Above', winRate: 55.0, pf: 1.92, dailyR: 1.45, trades: 95, status: 'active' },
  { rank: 6, ticker: 'MSFT', direction: 'LONG', tf: '5Min', trigger: 'MACD Cross Bull', winRate: 49.5, pf: 1.65, dailyR: 0.98, trades: 210, status: 'active' },
  { rank: 7, ticker: 'GOOG', direction: 'SHORT', tf: '1Min', trigger: 'EMA Bear Cross', winRate: 53.2, pf: 1.71, dailyR: 1.05, trades: 145, status: 'active' },
];

/* ========================================================================= */
/* SUB-COMPONENTS                                                             */
/* ========================================================================= */

function ToggleChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-2.5 py-1 rounded-lg text-xs font-medium transition-colors"
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

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function MassBuilderV3() {
  // Config state
  const [tickerInput, setTickerInput] = useState('');
  const [selectedTickers, setSelectedTickers] = useState<string[]>(['NVDA', 'SPY']);
  const [selectedTFs, setSelectedTFs] = useState<string[]>(['1Min', '5Min']);
  const [direction, setDirection] = useState<'LONG' | 'SHORT' | 'BOTH'>('LONG');
  const [selectedTriggers, setSelectedTriggers] = useState<string[]>(['EMA Bull Cross', 'VWAP Cross Above']);
  const [selectedConf, setSelectedConf] = useState<string[]>(['EMA_STACK: SML']);

  // Criteria
  const [minTrades, setMinTrades] = useState(10);
  const [minWR, setMinWR] = useState(0);
  const [minPF, setMinPF] = useState(0);

  // Results
  const [results, setResults] = useState<MassResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [sortBy, setSortBy] = useState<typeof SORT_OPTIONS[number]>('Daily R');

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

  function toggleConf(conf: string) {
    setSelectedConf((prev) => prev.includes(conf) ? prev.filter((c) => c !== conf) : [...prev, conf]);
  }

  function runAnalysis() {
    setIsRunning(true);
    setTimeout(() => {
      setResults(mockResults);
      setIsRunning(false);
    }, 1500);
  }

  function saveResult(rank: number) {
    setResults((prev) => prev.map((r) => r.rank === rank ? { ...r, status: 'saved' as const } : r));
  }

  const sortedResults = useMemo(() => {
    const sortKeyMap: Record<string, keyof MassResult> = {
      'Daily R': 'dailyR', 'Win Rate': 'winRate', 'Profit Factor': 'pf', 'Trades': 'trades',
    };
    const sk = sortKeyMap[sortBy] || 'dailyR';
    return [...results].sort((a, b) => (b[sk] as number) - (a[sk] as number));
  }, [results, sortBy]);

  const canRun = selectedTickers.length > 0 && selectedTriggers.length > 0;

  return (
    <div>
      <PageHeader title="Mass Strategy Builder" subtitle="Bulk strategy discovery" />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* ====== LEFT: Config ====== */}
        <div className="lg:col-span-1 space-y-4">
          <Card>
            {/* Tickers */}
            <div className="mb-4">
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Tickers</label>
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
                  type="text"
                  value={tickerInput}
                  onChange={(e) => setTickerInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') addTicker(); }}
                  placeholder="Add ticker..."
                  className="flex-1 px-2 py-1.5 rounded text-xs"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
                <button onClick={addTicker} className="px-2 py-1.5 rounded text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                  +
                </button>
              </div>
            </div>

            {/* Timeframes */}
            <div className="mb-4">
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Timeframes</label>
              <div className="flex gap-1.5 flex-wrap">
                {AVAILABLE_TFS.map((tf) => (
                  <ToggleChip key={tf} label={tf} active={selectedTFs.includes(tf)} onClick={() => toggleTF(tf)} />
                ))}
              </div>
            </div>

            {/* Direction */}
            <div className="mb-4">
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
              <div className="flex gap-1.5">
                {(['LONG', 'SHORT', 'BOTH'] as const).map((d) => (
                  <ToggleChip key={d} label={d} active={direction === d} onClick={() => setDirection(d)} />
                ))}
              </div>
            </div>

            {/* Triggers */}
            <div className="mb-4">
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Entry Triggers</label>
              <div className="flex gap-1.5 flex-wrap">
                {ENTRY_TRIGGERS.map((tr) => (
                  <ToggleChip key={tr} label={tr} active={selectedTriggers.includes(tr)} onClick={() => toggleTrigger(tr)} />
                ))}
              </div>
            </div>

            {/* Confluence */}
            <div className="mb-4">
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Confluence</label>
              <div className="flex gap-1.5 flex-wrap">
                {TF_CONFLUENCES.map((c) => (
                  <ToggleChip key={c} label={c} active={selectedConf.includes(c)} onClick={() => toggleConf(c)} />
                ))}
              </div>
            </div>

            {/* Criteria: compact inline row */}
            <div className="border-t pt-3" style={{ borderColor: 'var(--border)' }}>
              <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>Min Criteria</label>
              <div className="flex gap-3">
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Trades</label>
                  <input
                    type="number" min={0} value={minTrades}
                    onChange={(e) => setMinTrades(parseInt(e.target.value) || 0)}
                    className="w-16 px-2 py-1 rounded text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                </div>
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>WR%</label>
                  <input
                    type="number" min={0} max={100} value={minWR}
                    onChange={(e) => setMinWR(parseInt(e.target.value) || 0)}
                    className="w-16 px-2 py-1 rounded text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                </div>
                <div>
                  <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>PF</label>
                  <input
                    type="number" min={0} step={0.1} value={minPF}
                    onChange={(e) => setMinPF(parseFloat(e.target.value) || 0)}
                    className="w-16 px-2 py-1 rounded text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                </div>
              </div>
            </div>

            {/* Run button */}
            <button
              onClick={runAnalysis}
              disabled={!canRun || isRunning}
              className="w-full mt-4 py-2.5 rounded-lg text-sm font-medium transition-opacity"
              style={{
                background: 'var(--accent)', color: 'white',
                opacity: canRun && !isRunning ? 1 : 0.5,
                cursor: canRun && !isRunning ? 'pointer' : 'not-allowed',
              }}
            >
              {isRunning ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </Card>
        </div>

        {/* ====== RIGHT: Results ====== */}
        <div className="lg:col-span-2">
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                Results {results.length > 0 && `(${results.length})`}
              </h3>
              {results.length > 0 && (
                <div className="flex items-center gap-2">
                  <label className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Sort:</label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as typeof SORT_OPTIONS[number])}
                    className="px-2 py-1 rounded text-xs"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  >
                    {SORT_OPTIONS.map((o) => <option key={o}>{o}</option>)}
                  </select>
                </div>
              )}
            </div>

            {results.length === 0 ? (
              <div className="flex items-center justify-center py-16">
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  {isRunning ? 'Running analysis...' : 'Configure and run analysis to see results.'}
                </p>
              </div>
            ) : (
              <>
                {/* Table header */}
                <div
                  className="grid items-center gap-2 py-2 px-2 text-[10px] font-medium border-b"
                  style={{ gridTemplateColumns: '32px 56px 50px 52px 120px 56px 56px 64px 52px 64px', borderColor: 'var(--border)', color: 'var(--text-muted)' }}
                >
                  <span>#</span>
                  <span>Ticker</span>
                  <span>Dir</span>
                  <span>TF</span>
                  <span>Trigger</span>
                  <span className="text-right">WR%</span>
                  <span className="text-right">PF</span>
                  <span className="text-right">Daily R</span>
                  <span className="text-right">Trades</span>
                  <span className="text-right">Action</span>
                </div>

                {/* Table rows */}
                <div>
                  {sortedResults.map((r) => (
                    <div
                      key={r.rank}
                      className="grid items-center gap-2 py-2 px-2 text-xs border-b transition-colors"
                      style={{ gridTemplateColumns: '32px 56px 50px 52px 120px 56px 56px 64px 52px 64px', borderColor: 'var(--border)' }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-input)')}
                      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                    >
                      <span style={{ color: 'var(--text-muted)' }}>{r.rank}</span>
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{r.ticker}</span>
                      <span
                        className="text-[10px] px-1 py-0.5 rounded text-center"
                        style={{
                          background: r.direction === 'LONG' ? 'var(--green-muted)' : 'var(--red-muted)',
                          color: r.direction === 'LONG' ? 'var(--green)' : 'var(--red)',
                        }}
                      >
                        {r.direction}
                      </span>
                      <span style={{ color: 'var(--text-secondary)' }}>{r.tf}</span>
                      <span className="truncate" style={{ color: 'var(--text-secondary)' }}>{r.trigger}</span>
                      <span className="text-right font-medium" style={{ color: r.winRate >= 50 ? 'var(--green)' : 'var(--red)' }}>
                        {r.winRate.toFixed(1)}
                      </span>
                      <span className="text-right font-medium" style={{ color: r.pf >= 1.5 ? 'var(--green)' : 'var(--text-secondary)' }}>
                        {r.pf.toFixed(2)}
                      </span>
                      <span className="text-right font-medium" style={{ color: r.dailyR > 0 ? 'var(--green)' : 'var(--red)' }}>
                        +{r.dailyR.toFixed(2)}
                      </span>
                      <span className="text-right" style={{ color: 'var(--text-secondary)' }}>{r.trades}</span>
                      <span className="text-right">
                        {r.status === 'saved' ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
                            Saved
                          </span>
                        ) : (
                          <button
                            onClick={() => saveResult(r.rank)}
                            className="text-[10px] px-2 py-1 rounded font-medium transition-colors"
                            style={{ background: 'var(--accent)', color: 'white' }}
                          >
                            Save
                          </button>
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
