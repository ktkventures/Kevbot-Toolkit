'use client';
/**
 * Mass Builder — V6 locked design. Search name in titles, 1/2/3 column toggle,
 * My Strategies-style result cards with equity curve + pack-aware variables.
 */
import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import { useRunMassSearch } from '@/hooks/queries/useMassBuilder';

const TIMEFRAMES = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1Hour'];
const DIRECTIONS = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Pre-Market', 'After Hours', 'Extended', '24/7'];

const TICKER_PRESETS: Record<string, string[]> = {
  'Mag 7': ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA'],
  'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA'],
  'Crypto': ['BTC/USD', 'ETH/USD'],
};

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  padding: '8px 14px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  width: '100%',
};

const chipStyle = (selected: boolean): React.CSSProperties => ({
  padding: '4px 12px',
  borderRadius: '6px',
  fontSize: '0.8rem',
  cursor: 'pointer',
  border: selected ? '1px solid var(--accent)' : '1px solid var(--border)',
  background: selected ? 'var(--accent-muted)' : 'var(--bg-input)',
  color: selected ? 'var(--accent)' : 'var(--text-secondary)',
});

interface MassSearchResult {
  rank: number;
  ticker: string;
  direction: string;
  tf: string;
  trigger: string;
  exit_trigger: string;
  stop_desc: string;
  target_desc: string;
  confluence: string[];
  win_rate: number;
  pf: number;
  daily_r: number;
  trades: number;
  max_dd: number;
  total_r: number;
  avg_r: number;
  r_squared: number;
  equity_curve: number[];
  date_range: string;
  status: 'active' | 'saved' | 'passed';
}

function MiniEquityCurve({ data, height = 80 }: { data: number[]; height?: number }) {
  if (!data || data.length < 2) return null;
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
  const fillPoints = `0,${h} ${points} ${w},${h}`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={height} preserveAspectRatio="none">
      <polygon points={fillPoints} fill={fillColor} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

function ResultCard({ result, searchName }: { result: MassSearchResult; searchName: string }) {
  return (
    <Card>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs font-bold px-2 py-0.5 rounded" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>#{result.rank}</span>
        <h3 className="font-semibold text-sm">{searchName ? `${searchName} — ` : ''}{result.ticker} {result.direction}</h3>
        <span className="flex-1" />
        <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{result.tf}</span>
      </div>
      <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
        {result.ticker} {result.direction} &middot; {result.tf}{result.date_range && <> &middot; {result.date_range}</>}
      </p>

      {result.equity_curve && result.equity_curve.length > 1 ? (
        <div className="rounded-lg mb-2 overflow-hidden" style={{ background: 'var(--bg-input)' }}>
          <MiniEquityCurve data={result.equity_curve} height={80} />
        </div>
      ) : (
        <ChartPlaceholder label="Equity curve" height={80} />
      )}

      {/* KPIs grid — two rows of 4 */}
      <div className="grid grid-cols-4 gap-2 mb-2">
        {[
          { label: 'Win Rate', value: result.win_rate != null ? `${result.win_rate.toFixed(1)}%` : '--' },
          { label: 'Profit Factor', value: result.pf != null ? result.pf.toFixed(2) : '--' },
          { label: 'Daily R', value: result.daily_r != null ? `${result.daily_r >= 0 ? '+' : ''}${result.daily_r.toFixed(2)}` : '--' },
          { label: 'R\u00B2', value: result.r_squared != null ? result.r_squared.toFixed(2) : '--' },
          { label: 'Avg R', value: result.avg_r != null ? `${result.avg_r >= 0 ? '+' : ''}${result.avg_r.toFixed(2)}` : '--' },
          { label: 'Total R', value: result.total_r != null ? `${result.total_r >= 0 ? '+' : ''}${result.total_r.toFixed(0)}` : '--' },
          { label: 'Max DD', value: result.max_dd != null ? `${result.max_dd.toFixed(1)}R` : '--' },
          { label: 'Trades', value: result.trades != null ? String(result.trades) : '--' },
        ].map((kpi) => (
          <div key={kpi.label}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
            <p className="text-sm font-medium">{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Strategy variables — pack-aware display */}
      <div className="space-y-1 mb-2">
        {result.trigger && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>entry:</span>
            <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
              {result.trigger}
            </span>
          </div>
        )}
        {result.exit_trigger && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>exit:</span>
            <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>
              {result.exit_trigger}
            </span>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-1.5">
          {result.stop_desc && (
            <>
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>stop:</span>
              <span
                className="text-[10px] font-mono px-2 py-0.5 rounded"
                style={{ color: 'var(--red)', background: 'rgba(244,67,54,0.1)' }}
              >
                {result.stop_desc}
              </span>
            </>
          )}
          {result.target_desc && (
            <>
              <span className="text-[10px] ml-1" style={{ color: 'var(--text-muted)' }}>target:</span>
              <span
                className="text-[10px] font-mono px-2 py-0.5 rounded"
                style={{ color: 'var(--green)', background: 'rgba(76,175,80,0.1)' }}
              >
                {result.target_desc}
              </span>
            </>
          )}
        </div>
        {result.confluence && result.confluence.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>confluence:</span>
            {result.confluence.map((c) => (
              <span
                key={c}
                className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                style={{ color: 'var(--accent)', background: 'rgba(99,102,241,0.1)' }}
              >
                {c}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
        <button className="px-3 py-1 rounded text-xs font-medium" style={{ background: 'var(--accent)', color: 'white', cursor: 'pointer' }}>Save Strategy</button>
        <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-muted)', cursor: 'pointer' }}>Pass</button>
      </div>
    </Card>
  );
}

export default function MassBuilderPage() {
  const runMutation = useRunMassSearch();

  // Config state
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tickerInput, setTickerInput] = useState('');
  const [selectedTFs, setSelectedTFs] = useState<string[]>(['5Min']);
  const [selectedDirs, setSelectedDirs] = useState<string[]>(['LONG']);
  const [session, setSession] = useState('RTH');
  const [lookbackDays, setLookbackDays] = useState(30);
  const [searchName, setSearchName] = useState('');

  // Layout state (V6)
  const [resultColumns, setResultColumns] = useState(2);

  const toggleTF = (tf: string) => {
    setSelectedTFs((prev) =>
      prev.includes(tf) ? prev.filter((t) => t !== tf) : [...prev, tf]
    );
  };

  const toggleDir = (dir: string) => {
    setSelectedDirs((prev) =>
      prev.includes(dir) ? prev.filter((d) => d !== dir) : [...prev, dir]
    );
  };

  const addTickers = (raw: string) => {
    const tickers = raw.split(/[,;\s\n]+/).map((s) => s.trim().toUpperCase()).filter(Boolean);
    setSelectedTickers((prev) => Array.from(new Set([...prev, ...tickers])).sort());
    setTickerInput('');
  };

  const removeTicker = (t: string) => {
    setSelectedTickers((prev) => prev.filter((x) => x !== t));
  };

  // Estimate combinations
  const totalCombinations = Math.max(1, selectedTickers.length) * selectedTFs.length * selectedDirs.length;

  // Extract results from mutation data
  const results: MassSearchResult[] = useMemo(() => {
    if (!runMutation.isSuccess || !runMutation.data) return [];
    const data = runMutation.data as any;
    return data.results ?? [];
  }, [runMutation.isSuccess, runMutation.data]);

  const handleRun = () => {
    if (selectedTickers.length === 0) return;
    if (selectedTFs.length === 0) return;
    if (selectedDirs.length === 0) return;

    runMutation.mutate({
      name: searchName || `Search ${new Date().toISOString().slice(0, 16)}`,
      symbols: selectedTickers,
      timeframes: selectedTFs,
      directions: selectedDirs,
      session,
      lookback_days: lookbackDays,
    });
  };

  const colClass =
    resultColumns === 1
      ? 'grid-cols-1'
      : resultColumns === 2
        ? 'grid-cols-1 lg:grid-cols-2'
        : 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3';

  return (
    <div>
      <PageHeader
        title="Mass Strategy Builder"
        subtitle="Bulk strategy discovery and optimization engine"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
          </div>
        }
      />

      {/* Search name row (V6 style) */}
      <div className="flex items-end gap-4 mb-5 mt-4">
        <div className="flex-1">
          <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>
            Search Name
          </label>
          <input
            type="text"
            placeholder="My Mass Search"
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          />
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
            Flows into strategy titles: &quot;{searchName || 'Search'} -- TICKER DIRECTION&quot;
          </p>
        </div>
        <button
          className="px-5 py-2 rounded-lg text-sm font-medium"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
        >
          Save Search
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Config panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Symbols */}
          <Card>
            <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Symbols
            </label>
            <div className="flex gap-2 mb-2">
              {Object.entries(TICKER_PRESETS).map(([name, tickers]) => (
                <button
                  key={name}
                  onClick={() => {
                    const merged = new Set([...selectedTickers, ...tickers]);
                    setSelectedTickers(Array.from(merged).sort());
                  }}
                  className="px-3 py-1 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer' }}
                >
                  + {name}
                </button>
              ))}
              {selectedTickers.length > 0 && (
                <button
                  onClick={() => setSelectedTickers([])}
                  className="px-3 py-1 rounded-lg text-xs"
                  style={{ color: 'var(--red)', cursor: 'pointer' }}
                >
                  Clear
                </button>
              )}
            </div>
            <div className="flex gap-2 mb-2">
              <input
                type="text"
                placeholder="Enter tickers (comma separated)..."
                value={tickerInput}
                onChange={(e) => setTickerInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') addTickers(tickerInput); }}
                className="flex-1 px-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
              <button
                onClick={() => addTickers(tickerInput)}
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--accent)', color: 'white', cursor: 'pointer' }}
              >
                Add
              </button>
            </div>
            {selectedTickers.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {selectedTickers.map((t) => (
                  <span
                    key={t}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: '1px solid var(--accent)' }}
                  >
                    {t}
                    <button
                      onClick={() => removeTicker(t)}
                      className="ml-0.5 text-xs opacity-60 hover:opacity-100"
                      style={{ cursor: 'pointer' }}
                    >
                      x
                    </button>
                  </span>
                ))}
              </div>
            )}
          </Card>

          {/* Timeframes */}
          <Card>
            <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>
              Timeframes
            </label>
            <div className="flex flex-wrap gap-1.5">
              {TIMEFRAMES.map((tf) => (
                <button key={tf} style={chipStyle(selectedTFs.includes(tf))} onClick={() => toggleTF(tf)}>
                  {tf}
                </button>
              ))}
            </div>
          </Card>

          {/* Directions */}
          <Card>
            <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>
              Directions
            </label>
            <div className="flex gap-2">
              {DIRECTIONS.map((dir) => (
                <button key={dir} style={chipStyle(selectedDirs.includes(dir))} onClick={() => toggleDir(dir)}>
                  {dir}
                </button>
              ))}
            </div>
          </Card>

          {/* Session & Lookback */}
          <Card>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
                  Session
                </label>
                <select
                  value={session}
                  onChange={(e) => setSession(e.target.value)}
                  style={inputStyle}
                >
                  {SESSIONS.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
                  Lookback Days
                </label>
                <input
                  type="number"
                  min={1}
                  max={365}
                  value={lookbackDays}
                  onChange={(e) => setLookbackDays(Number(e.target.value))}
                  style={inputStyle}
                />
              </div>
            </div>
          </Card>

          {/* Combination estimate + Run */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Estimated Combinations</p>
                <p className="text-lg font-bold font-mono">{totalCombinations.toLocaleString()}</p>
              </div>
              <div className="text-right text-xs" style={{ color: 'var(--text-muted)' }}>
                <p>{selectedTickers.length} symbol{selectedTickers.length !== 1 ? 's' : ''}</p>
                <p>{selectedTFs.length} TF{selectedTFs.length !== 1 ? 's' : ''}</p>
                <p>{selectedDirs.length} dir{selectedDirs.length !== 1 ? 's' : ''}</p>
              </div>
            </div>

            {runMutation.isPending && (
              <div className="mb-3">
                <div className="w-full rounded-full h-2" style={{ background: 'var(--bg-input)' }}>
                  <div
                    className="h-2 rounded-full transition-all"
                    style={{
                      background: 'var(--accent)',
                      width: '60%',
                      animation: 'pulse 1.5s ease-in-out infinite',
                    }}
                  />
                </div>
                <p className="text-xs mt-1 text-center" style={{ color: 'var(--text-muted)' }}>
                  Starting search...
                </p>
              </div>
            )}

            <button
              className="w-full py-2.5 rounded-lg text-sm font-medium"
              style={{
                background: selectedTickers.length > 0 ? 'var(--accent)' : 'var(--bg-input)',
                color: selectedTickers.length > 0 ? '#fff' : 'var(--text-muted)',
                border: 'none',
                cursor: selectedTickers.length > 0 ? 'pointer' : 'not-allowed',
              }}
              disabled={selectedTickers.length === 0 || selectedTFs.length === 0 || selectedDirs.length === 0 || runMutation.isPending}
              onClick={handleRun}
            >
              {runMutation.isPending ? 'Starting Search...' : 'Run Mass Search'}
            </button>
          </Card>
        </div>

        {/* Right: Results panel */}
        <div className="lg:col-span-2">
          {/* Column toggle (V6) */}
          {(results.length > 0 || runMutation.isSuccess) && (
            <div className="flex items-center gap-4 mb-4 text-[10px]" style={{ color: 'var(--text-muted)' }}>
              <div className="flex items-center gap-1.5">
                <span>Columns:</span>
                {[1, 2, 3].map((n) => (
                  <button
                    key={n}
                    onClick={() => setResultColumns(n)}
                    className="px-2 py-1 rounded font-medium"
                    style={{
                      background: resultColumns === n ? 'var(--accent-muted)' : 'var(--bg-input)',
                      color: resultColumns === n ? 'var(--accent)' : 'var(--text-muted)',
                      border: resultColumns === n ? '1px solid var(--accent)' : '1px solid var(--border)',
                      cursor: 'pointer',
                    }}
                  >
                    {n}
                  </button>
                ))}
              </div>
              {results.length > 0 && (
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {results.length} result{results.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          )}

          {/* Result cards in My Strategies style (V6) */}
          {results.length > 0 && (
            <div className={`grid gap-4 ${colClass}`}>
              {results.map((result, idx) => (
                <ResultCard key={idx} result={result} searchName={searchName} />
              ))}
            </div>
          )}

          {/* Success state with no inline results (worker-based) */}
          {runMutation.isSuccess && results.length === 0 && (
            <Card>
              <div className="text-center py-8">
                <p className="text-sm font-medium mb-2" style={{ color: 'var(--green)' }}>
                  Search started successfully
                </p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Search ID: {(runMutation.data as any)?.search_id || '--'}
                </p>
                <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                  View progress and results in the Mass Results page.
                </p>
              </div>
            </Card>
          )}

          {runMutation.isError && (
            <Card>
              <div className="text-center py-8" style={{ color: 'var(--red)' }}>
                Failed to start mass search. Check your configuration and try again.
              </div>
            </Card>
          )}

          {!runMutation.isSuccess && !runMutation.isError && !runMutation.isPending && (
            <Card>
              <div className="text-center py-16">
                <p className="text-lg font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                  Configure and run a search
                </p>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                  Set your symbols, timeframes, and parameters on the left, then click Run Mass Search.
                  Results will appear here and be saved to the Mass Results page.
                </p>
              </div>
            </Card>
          )}

          {runMutation.isPending && (
            <div className={`grid gap-4 ${colClass}`}>
              {[1, 2, 3].map((i) => (
                <Card key={i}>
                  <div className="animate-pulse space-y-3">
                    <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                    <div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} />
                    <div className="grid grid-cols-4 gap-2">{[1, 2, 3, 4].map((j) => <div key={j} className="h-8 rounded" style={{ background: 'var(--border)' }} />)}</div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
