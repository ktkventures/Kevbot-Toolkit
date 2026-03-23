'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

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

const statusColors: Record<string, { color: string; bg: string }> = {
  Paid: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Pending: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Processing: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
};

export default function CreatorEarningsV3() {
  const kpis = [
    { label: 'Earned', value: `$${mockEarnings.totalEarned.toLocaleString()}`, accent: 'var(--accent)' },
    { label: 'Pending', value: `$${mockEarnings.pendingPayout}`, accent: 'var(--green)' },
    { label: 'Next Payout', value: mockEarnings.nextPayoutDate, accent: 'var(--blue)' },
    { label: 'Fees Paid', value: `$${mockEarnings.platformFeePaid}`, accent: 'var(--orange)' },
  ];

  return (
    <div>
      <PageHeader title="Earnings & Payouts" subtitle="Compact view" backHref="/creator/dashboard" />

      {/* KPI strip */}
      <div className="flex gap-0 rounded-xl overflow-hidden border mb-6" style={{ borderColor: 'var(--border)' }}>
        {kpis.map((kpi, i) => (
          <div
            key={kpi.label}
            className="flex-1 text-center py-3 px-2"
            style={{
              background: 'var(--bg-card)',
              borderRight: i < kpis.length - 1 ? '1px solid var(--border)' : 'none',
            }}
          >
            <p className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
            <p className="text-lg font-bold" style={{ color: kpi.accent }}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Earnings table */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Earnings</h3>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Month', 'Gross', 'Fee', 'Net'].map((h) => (
                <th key={h} className="text-left py-2 px-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {mockEarnings.monthlyBreakdown.map((row) => (
              <tr key={row.month} style={{ borderBottom: '1px solid var(--border)' }}>
                <td className="py-2 px-2 text-sm">{row.month}</td>
                <td className="py-2 px-2 text-sm">${row.gross}</td>
                <td className="py-2 px-2 text-sm" style={{ color: 'var(--red)' }}>-${row.platformFee}</td>
                <td className="py-2 px-2 text-sm font-medium" style={{ color: 'var(--green)' }}>${row.net}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Payout list */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Payouts</h3>
        <div className="space-y-2">
          {mockEarnings.payouts.map((p, i) => (
            <div
              key={i}
              className="flex items-center justify-between px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center gap-4">
                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{p.date}</span>
                <span className="text-sm font-medium">${p.amount.toFixed(2)}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.method}</span>
                <span
                  className="text-xs px-2 py-0.5 rounded"
                  style={{ color: statusColors[p.status]?.color, background: statusColors[p.status]?.bg }}
                >
                  {p.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
