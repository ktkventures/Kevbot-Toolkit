import Link from 'next/link';
import Card from '@/components/Card';

const mockStrategies = [
  { name: 'NVDA LONG - Mass #2', symbol: 'NVDA', direction: 'LONG', winRate: 54.0, pf: 2.05, dailyR: 1.95, trades: 224, status: 'On Track' },
  { name: 'SPY LONG - Mass #1', symbol: 'SPY', direction: 'LONG', winRate: 62.5, pf: 3.12, dailyR: 2.41, trades: 89, status: 'Outperforming' },
  { name: 'AAPL LONG - Mass #5', symbol: 'AAPL', direction: 'LONG', winRate: 48.2, pf: 1.45, dailyR: 0.82, trades: 156, status: 'Insufficient Data' },
  { name: 'TSLA LONG - Mass #5', symbol: 'TSLA', direction: 'LONG', winRate: 51.3, pf: 1.78, dailyR: 1.23, trades: 201, status: 'On Track' },
];

const statusColors: Record<string, string> = {
  'On Track': 'var(--green)',
  'Outperforming': 'var(--blue)',
  'Underperforming': 'var(--red)',
  'Insufficient Data': 'var(--text-muted)',
};

export default function StrategiesV1() {
  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">My Strategies</h1>
        <div className="flex gap-3">
          <button
            className="px-4 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          >
            Update Data
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + New Strategy
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 mb-6">
        {['Ticker', 'Direction', 'Tag'].map((filter) => (
          <select
            key={filter}
            className="px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            <option>{filter}: All</option>
          </select>
        ))}
      </div>

      <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
        {mockStrategies.length} strategies
      </p>

      {/* Strategy cards */}
      <div className="space-y-4">
        {mockStrategies.map((strat, i) => (
          <Card key={i}>
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-1">
                  <h3 className="font-semibold">{strat.name}</h3>
                  <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: statusColors[strat.status], background: statusColors[strat.status] + '20' }}>
                    {strat.status}
                  </span>
                </div>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  {strat.symbol} | {strat.direction} | BT 90d | Fwd 14d
                </p>

                {/* Mini equity curve placeholder */}
                <div
                  className="rounded-lg mb-4"
                  style={{ background: 'var(--bg-input)', height: 60 }}
                />

                {/* KPIs */}
                <div className="grid grid-cols-5 gap-4">
                  {[
                    { label: 'WR', value: `${strat.winRate}%` },
                    { label: 'PF', value: strat.pf.toFixed(2) },
                    { label: 'Daily R', value: `+${strat.dailyR.toFixed(2)}` },
                    { label: 'Trades', value: strat.trades },
                    { label: 'Max DD', value: '-2.5R' },
                  ].map((kpi, j) => (
                    <div key={j}>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
                      <p className="text-sm font-medium">{kpi.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 ml-4">
                <Link href={`/strategies/${i + 1}`} className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                  View
                </Link>
                <button className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                  Edit
                </button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
