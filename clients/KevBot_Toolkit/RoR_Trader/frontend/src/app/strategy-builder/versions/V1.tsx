import Card from '@/components/Card';

export default function StrategyBuilderV1() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Strategy Builder</h1>

      <div className="grid grid-cols-3 gap-6">
        {/* Left: Configuration */}
        <div className="col-span-1 space-y-4">
          <Card>
            <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
              Configuration
            </h3>
            {[
              { label: 'Ticker', placeholder: 'SPY' },
              { label: 'Direction', placeholder: 'LONG' },
              { label: 'Timeframe', placeholder: '1Min' },
              { label: 'Session', placeholder: 'RTH' },
            ].map((field, i) => (
              <div key={i} className="mb-3">
                <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>
                  {field.label}
                </label>
                <input
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  placeholder={field.placeholder}
                />
              </div>
            ))}
          </Card>

          <Card>
            <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
              Entry Trigger
            </h3>
            <select
              className="w-full px-3 py-2 rounded-lg text-sm mb-3"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            >
              <option>[C] EMA Price Cross Up</option>
              <option>[L] VWAP Cross</option>
              <option>[LC] UT Bot Buy</option>
            </select>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Execution type determines when the entry fires and how it&apos;s confirmed.
            </p>
          </Card>

          <Card>
            <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
              Confluence Conditions
            </h3>
            <div className="space-y-2">
              {['5M-EMA_STACK-BULL (Default) [PB]', '1D-MACD_LINE-BULL (Default) [PB]'].map((cond, i) => (
                <div
                  key={i}
                  className="px-3 py-2 rounded-lg text-xs flex items-center justify-between"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                >
                  <span style={{ color: 'var(--text-secondary)' }}>{cond}</span>
                  <span style={{ color: 'var(--text-muted)' }}>x</span>
                </div>
              ))}
            </div>
            <button
              className="mt-3 text-xs px-3 py-1.5 rounded"
              style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
            >
              + Add Condition
            </button>
          </Card>
        </div>

        {/* Right: Results */}
        <div className="col-span-2 space-y-4">
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
              Price Chart
            </h3>
            <div
              className="rounded-lg flex items-center justify-center"
              style={{ background: 'var(--bg-input)', height: 400 }}
            >
              <span style={{ color: 'var(--text-muted)' }}>TradingView chart — OHLC + indicators + trade markers</span>
            </div>
          </Card>

          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
              Equity Curve & KPIs
            </h3>
            <div className="grid grid-cols-4 gap-3 mb-4">
              {[
                { label: 'Win Rate', value: '62.5%' },
                { label: 'Profit Factor', value: '2.84' },
                { label: 'Daily R', value: '+1.92' },
                { label: 'Trades', value: '147' },
              ].map((kpi, i) => (
                <div key={i} className="text-center">
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                  <p className="text-lg font-semibold">{kpi.value}</p>
                </div>
              ))}
            </div>
            <div
              className="rounded-lg flex items-center justify-center"
              style={{ background: 'var(--bg-input)', height: 200 }}
            >
              <span style={{ color: 'var(--text-muted)' }}>Equity curve placeholder</span>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
