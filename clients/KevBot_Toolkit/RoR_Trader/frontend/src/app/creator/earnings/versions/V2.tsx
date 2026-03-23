'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import TabBar from '@/components/TabBar';

const mockEarnings = {
  totalEarned: 4280,
  pendingPayout: 890,
  nextPayoutDate: '2026-04-01',
  platformFeePaid: 756,
  monthlyBreakdown: [
    { month: 'Mar 2026', gross: 1050, platformFee: 158, net: 892 },
    { month: 'Feb 2026', gross: 850, platformFee: 128, net: 722 },
    { month: 'Jan 2026', gross: 780, platformFee: 117, net: 663 },
    { month: 'Dec 2025', gross: 610, platformFee: 92, net: 518 },
    { month: 'Nov 2025', gross: 520, platformFee: 78, net: 442 },
    { month: 'Oct 2025', gross: 380, platformFee: 57, net: 323 },
  ],
  payouts: [
    { date: '2026-03-01', amount: 722, method: 'PayPal', status: 'Paid' as const },
    { date: '2026-02-01', amount: 663, method: 'PayPal', status: 'Paid' as const },
    { date: '2026-01-01', amount: 581, method: 'PayPal', status: 'Paid' as const },
  ],
};

const mockRevenueByItem = [
  { name: 'Momentum Alpha', type: 'Portfolio', revenue: 490, share: 55 },
  { name: 'VWAP Scalper Pro', type: 'Portfolio', revenue: 230, share: 26 },
  { name: 'EMA Momentum', type: 'Strategy', revenue: 90, share: 10 },
  { name: 'RSI Zones Pack', type: 'Pack', revenue: 80, share: 9 },
];

const mockRevenueShares = [
  { item: 'Momentum Alpha', total: 490, yourShare: 245, strategyCreators: 122.5, packCreators: 49, platform: 73.5 },
  { item: 'VWAP Scalper Pro', total: 230, yourShare: 115, strategyCreators: 57.5, packCreators: 23, platform: 34.5 },
];

const typeColors: Record<string, { color: string; bg: string }> = {
  Portfolio: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Strategy: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Pack: { color: 'var(--purple)', bg: 'var(--purple-muted)' },
};

const statusColors: Record<string, { color: string; bg: string }> = {
  Paid: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Pending: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Processing: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
};

function DonutChart({ segments }: { segments: { label: string; value: number; color: string }[] }) {
  const total = segments.reduce((s, seg) => s + seg.value, 0);
  const size = 140;
  const cx = size / 2;
  const cy = size / 2;
  const r = 50;
  const strokeWidth = 20;
  const circumference = 2 * Math.PI * r;

  let accumulatedOffset = 0;

  return (
    <div className="flex items-center gap-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {segments.map((seg, i) => {
          const pct = total > 0 ? seg.value / total : 0;
          const dashArray = `${pct * circumference} ${circumference}`;
          const rotation = accumulatedOffset * 360 - 90;
          accumulatedOffset += pct;
          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={r}
              fill="none"
              stroke={seg.color}
              strokeWidth={strokeWidth}
              strokeDasharray={dashArray}
              transform={`rotate(${rotation} ${cx} ${cy})`}
            />
          );
        })}
        <text x={cx} y={cy - 6} textAnchor="middle" fontSize="16" fontWeight="bold" fill="var(--text-primary)">
          ${total}
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
          /month
        </text>
      </svg>
      <div className="space-y-2">
        {segments.map((seg, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ background: seg.color }} />
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              {seg.label}: ${seg.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CreatorEarningsV2() {
  const [payoutMethod, setPayoutMethod] = useState('PayPal');
  const [minThreshold, setMinThreshold] = useState('100');
  const [payoutFreq, setPayoutFreq] = useState('Monthly');

  return (
    <div>
      <PageHeader
        title="Earnings & Payouts"
        subtitle="Detailed revenue tracking, payout history, and tax reporting"
        backHref="/creator/dashboard"
      />

      {/* Revenue Summary */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Earned" value={`$${mockEarnings.totalEarned.toLocaleString()}`} delta="All time" />
        <MetricCard label="Pending Payout" value={`$${mockEarnings.pendingPayout}`} delta="Awaiting cycle" positive />
        <MetricCard label="Next Payout Date" value={mockEarnings.nextPayoutDate} delta="Auto-scheduled" />
        <MetricCard label="Platform Fees Paid" value={`$${mockEarnings.platformFeePaid}`} delta="15% of gross" />
      </div>

      <TabBar tabs={['Revenue', 'Revenue Shares', 'Payouts', 'Settings', 'Tax Documents']}>
        {(tab) => (
          <div>
            {tab === 'Revenue' && (
              <div className="grid grid-cols-3 gap-6">
                <div className="col-span-2">
                  {/* Revenue by item donut */}
                  <Card className="mb-6">
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                      Revenue Breakdown by Item
                    </h3>
                    <DonutChart
                      segments={mockRevenueByItem.map((item) => ({
                        label: item.name,
                        value: item.revenue,
                        color: typeColors[item.type]?.color || 'var(--text-muted)',
                      }))}
                    />
                  </Card>

                  {/* Monthly earnings table */}
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                      Monthly Earnings
                    </h3>
                    <table className="w-full text-sm">
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                          {['Month', 'Gross', 'Platform Fee (15%)', 'Net Earnings'].map((h) => (
                            <th key={h} className="text-left py-3 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {mockEarnings.monthlyBreakdown.map((row) => (
                          <tr key={row.month} style={{ borderBottom: '1px solid var(--border)' }}>
                            <td className="py-3 px-3">{row.month}</td>
                            <td className="py-3 px-3">${row.gross.toFixed(2)}</td>
                            <td className="py-3 px-3" style={{ color: 'var(--red)' }}>-${row.platformFee.toFixed(2)}</td>
                            <td className="py-3 px-3 font-medium" style={{ color: 'var(--green)' }}>${row.net.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Card>
                </div>

                {/* Per-item revenue list */}
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                    Revenue by Item
                  </h3>
                  <div className="space-y-4">
                    {mockRevenueByItem.map((item) => (
                      <div key={item.name}>
                        <div className="flex justify-between items-center mb-1">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">{item.name}</span>
                            <span
                              className="text-xs px-1.5 py-0.5 rounded"
                              style={{ color: typeColors[item.type]?.color, background: typeColors[item.type]?.bg }}
                            >
                              {item.type}
                            </span>
                          </div>
                          <span className="text-sm font-medium">${item.revenue}/mo</span>
                        </div>
                        <div className="h-2 rounded-full" style={{ background: 'var(--bg-input)' }}>
                          <div
                            className="h-2 rounded-full"
                            style={{ width: `${item.share}%`, background: typeColors[item.type]?.color }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>
            )}

            {tab === 'Revenue Shares' && (
              <Card>
                <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                  Revenue Share Breakdown
                </h3>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  How revenue splits across the creator chain for each portfolio
                </p>
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      {['Portfolio', 'Total Rev', 'Your Share (50%)', 'Strategy Creators (25%)', 'Pack Creators (10%)', 'Platform (15%)'].map((h) => (
                        <th key={h} className="text-left py-3 px-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {mockRevenueShares.map((row) => (
                      <tr key={row.item} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td className="py-3 px-2 font-medium">{row.item}</td>
                        <td className="py-3 px-2">${row.total.toFixed(2)}</td>
                        <td className="py-3 px-2" style={{ color: 'var(--green)' }}>${row.yourShare.toFixed(2)}</td>
                        <td className="py-3 px-2" style={{ color: 'var(--blue)' }}>${row.strategyCreators.toFixed(2)}</td>
                        <td className="py-3 px-2" style={{ color: 'var(--purple)' }}>${row.packCreators.toFixed(2)}</td>
                        <td className="py-3 px-2" style={{ color: 'var(--text-muted)' }}>${row.platform.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}

            {tab === 'Payouts' && (
              <Card>
                <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
                  Payout History
                </h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                      {['Date', 'Amount', 'Method', 'Status'].map((h) => (
                        <th key={h} className="text-left py-3 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {mockEarnings.payouts.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td className="py-3 px-3">{p.date}</td>
                        <td className="py-3 px-3 font-medium">${p.amount.toFixed(2)}</td>
                        <td className="py-3 px-3">{p.method}</td>
                        <td className="py-3 px-3">
                          <span
                            className="text-xs px-2 py-0.5 rounded"
                            style={{ color: statusColors[p.status]?.color, background: statusColors[p.status]?.bg }}
                          >
                            {p.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            )}

            {tab === 'Settings' && (
              <Card>
                <h3 className="text-sm font-medium mb-6" style={{ color: 'var(--text-secondary)' }}>
                  Payout Settings
                </h3>
                <div className="space-y-6 max-w-lg">
                  <div>
                    <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Payout Method</label>
                    <select
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      value={payoutMethod}
                      onChange={(e) => setPayoutMethod(e.target.value)}
                    >
                      <option>PayPal</option>
                      <option>Stripe</option>
                      <option>Bank Transfer (ACH)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Minimum Payout Threshold</label>
                    <div className="flex items-center gap-2">
                      <span className="text-sm" style={{ color: 'var(--text-muted)' }}>$</span>
                      <input
                        type="number"
                        className="flex-1 px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                        value={minThreshold}
                        onChange={(e) => setMinThreshold(e.target.value)}
                      />
                    </div>
                  </div>
                  <div>
                    <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Payout Frequency</label>
                    <select
                      className="w-full px-3 py-2 rounded-lg text-sm"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      value={payoutFreq}
                      onChange={(e) => setPayoutFreq(e.target.value)}
                    >
                      <option>Weekly</option>
                      <option>Bi-weekly</option>
                      <option>Monthly</option>
                    </select>
                  </div>
                  <button
                    className="px-4 py-2 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--accent)', color: 'white' }}
                  >
                    Save Settings
                  </button>
                </div>
              </Card>
            )}

            {tab === 'Tax Documents' && (
              <Card>
                <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
                  Tax Documents
                </h3>
                <p className="text-xs mb-6" style={{ color: 'var(--text-muted)' }}>
                  Download your tax forms for each reporting year.
                </p>
                <div className="space-y-3">
                  {['2025'].map((year) => (
                    <div
                      key={year}
                      className="flex items-center justify-between p-4 rounded-lg"
                      style={{ background: 'var(--bg-input)' }}
                    >
                      <div>
                        <p className="text-sm font-medium">Form 1099-NEC &mdash; {year}</p>
                        <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                          Non-employee compensation reporting
                        </p>
                      </div>
                      <button
                        className="px-4 py-2 rounded-lg text-xs font-medium"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
                      >
                        Download PDF
                      </button>
                    </div>
                  ))}
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Tax documents are generated annually and available by January 31st of the following year.
                  </p>
                </div>
              </Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
