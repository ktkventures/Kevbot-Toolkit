'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

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
    { date: '2025-12-01', amount: 442, method: 'PayPal', status: 'Paid' as const },
  ],
};

const statusColors: Record<string, { color: string; bg: string }> = {
  Paid: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Pending: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Processing: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
};

export default function CreatorEarningsV1() {
  return (
    <div>
      <PageHeader
        title="Earnings & Payouts"
        subtitle="Revenue tracking and payout history"
        backHref="/creator/dashboard"
      />

      {/* Summary KPIs */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <MetricCard label="Total Earned" value={`$${mockEarnings.totalEarned.toLocaleString()}`} delta="Lifetime" />
        <MetricCard label="Pending Payout" value={`$${mockEarnings.pendingPayout}`} delta="Next cycle" positive />
        <MetricCard label="Next Payout" value={mockEarnings.nextPayoutDate} delta="Scheduled" />
        <MetricCard label="Platform Fees" value={`$${mockEarnings.platformFeePaid}`} delta="15% rate" />
      </div>

      {/* Monthly earnings table */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
          Monthly Earnings
        </h3>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Month', 'Gross', 'Platform Fee', 'Net'].map((h) => (
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

      {/* Payout history */}
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
    </div>
  );
}
