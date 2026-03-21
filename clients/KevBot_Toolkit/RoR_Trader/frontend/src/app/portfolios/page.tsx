import Card from '@/components/Card';

export default function Portfolios() {
  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Portfolios</h1>
        <button
          className="px-4 py-2 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          + New Portfolio
        </button>
      </div>

      {/* Portfolio cards */}
      <div className="grid grid-cols-2 gap-6">
        {[
          { name: 'My Portfolio', strategies: 8, balance: 80000, pnl: 2847, dd: -1.8, wr: 58.3 },
          { name: 'Growth Portfolio', strategies: 5, balance: 50000, pnl: 1205, dd: -2.4, wr: 52.1 },
        ].map((port, i) => (
          <Card key={i} className="cursor-pointer hover:border-opacity-50">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="font-semibold text-lg">{port.name}</h3>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {port.strategies} strategies | ${port.balance.toLocaleString()} balance
                </p>
              </div>
              <span
                className="text-xs px-2 py-1 rounded"
                style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
              >
                Active
              </span>
            </div>

            {/* Mini equity curve */}
            <div
              className="rounded-lg mb-4"
              style={{ background: 'var(--bg-input)', height: 80 }}
            />

            {/* KPIs */}
            <div className="grid grid-cols-4 gap-3">
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L</p>
                <p className="text-sm font-medium" style={{ color: 'var(--green)' }}>
                  +${port.pnl.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Max DD</p>
                <p className="text-sm font-medium">{port.dd}%</p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Win Rate</p>
                <p className="text-sm font-medium">{port.wr}%</p>
              </div>
              <div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Strategies</p>
                <p className="text-sm font-medium">{port.strategies}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
