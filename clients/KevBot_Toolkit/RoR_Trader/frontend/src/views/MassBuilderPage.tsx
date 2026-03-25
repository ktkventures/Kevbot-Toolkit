'use client';

/**
 * Mass Builder — Clean API-first page.
 *
 * Simplified version of the mass strategy builder. Core config form
 * with Run button that calls the mass search mutation. No mock data.
 */

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useRunMassSearch } from '@/hooks/queries/useMassBuilder';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEFRAMES = ['1Min', '2Min', '3Min', '5Min', '10Min', '15Min', '30Min', '1Hour'];
const DIRECTIONS = ['LONG', 'SHORT'];
const SESSIONS = ['RTH', 'Extended', '24/7'];

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

  // Parse symbols
  const symbolList = symbols
    .split(/[,\s]+/)
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);

  // Estimate combinations
  const totalCombinations = Math.max(1, symbolList.length) * selectedTFs.length * selectedDirs.length;

  const handleRun = () => {
    if (symbolList.length === 0) return;
    if (selectedTFs.length === 0) return;
    if (selectedDirs.length === 0) return;

    runMutation.mutate({
      name: searchName || `Search ${new Date().toISOString().slice(0, 16)}`,
      symbols: symbolList,
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
        subtitle="Bulk backtest strategy combinations across multiple symbols, timeframes, and parameters"
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
        {/* Left: Config panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Search name */}
          <Card>
            <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Search Name
            </label>
            <input
              type="text"
              placeholder="My Mass Search"
              value={searchName}
              onChange={(e) => setSearchName(e.target.value)}
              style={inputStyle}
            />
          </Card>

          {/* Symbols */}
          <Card>
            <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
              Symbols (comma or space separated)
            </label>
            <input
              type="text"
              placeholder="NVDA, SPY, TSLA, AAPL"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              style={inputStyle}
            />
            {symbolList.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {symbolList.map((s) => (
                  <span key={s} className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                    {s}
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
                <p>{symbolList.length} symbol{symbolList.length !== 1 ? 's' : ''}</p>
                <p>{selectedTFs.length} TF{selectedTFs.length !== 1 ? 's' : ''}</p>
                <p>{selectedDirs.length} dir{selectedDirs.length !== 1 ? 's' : ''}</p>
              </div>
            </div>
            <button
              className="w-full py-2.5 rounded-lg text-sm font-medium"
              style={{
                background: symbolList.length > 0 ? 'var(--accent)' : 'var(--bg-input)',
                color: symbolList.length > 0 ? '#fff' : 'var(--text-muted)',
                border: 'none',
                cursor: symbolList.length > 0 ? 'pointer' : 'not-allowed',
              }}
              disabled={symbolList.length === 0 || selectedTFs.length === 0 || selectedDirs.length === 0 || runMutation.isPending}
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

          {!runMutation.isSuccess && !runMutation.isError && (
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
        </div>
      </div>
    </div>
  );
}
