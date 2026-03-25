'use client';

/**
 * Mass Builder — Clean API-first page.
 *
 * Visual design derived from V6 (versions/V6.tsx), data layer built
 * around the mass search mutation. No mock data.
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useRunMassSearch } from '@/hooks/queries/useMassBuilder';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MassBuilderPage() {
  const runMutation = useRunMassSearch();

  // Config state
  const [symbols, setSymbols] = useState('');
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tickerInput, setTickerInput] = useState('');
  const [selectedTFs, setSelectedTFs] = useState<string[]>(['5Min']);
  const [selectedDirs, setSelectedDirs] = useState<string[]>(['LONG']);
  const [session, setSession] = useState('RTH');
  const [lookbackDays, setLookbackDays] = useState(30);
  const [searchName, setSearchName] = useState('');

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
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Config panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Symbols — multi-select chip input (V6 style) */}
          <Card>
            <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Symbols
            </label>
            {/* Quick-add presets */}
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
            {/* Text input */}
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
            {/* Selected chip display */}
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

          {/* Timeframes — multi-select chips */}
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

            {/* Progress bar during run */}
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
          {runMutation.isSuccess && (
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

          {/* Placeholder for result cards matching My Strategies style */}
          {runMutation.isPending && (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <Card key={i}>
                  <div className="animate-pulse space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="h-4 rounded w-16" style={{ background: 'var(--border)' }} />
                      <div className="h-4 rounded w-24" style={{ background: 'var(--border)' }} />
                      <div className="h-4 rounded w-12" style={{ background: 'var(--border)' }} />
                    </div>
                    <div className="h-16 rounded" style={{ background: 'var(--bg-input)' }} />
                    <div className="grid grid-cols-4 gap-2">
                      {[1, 2, 3, 4].map((j) => (
                        <div key={j} className="h-8 rounded" style={{ background: 'var(--border)' }} />
                      ))}
                    </div>
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
