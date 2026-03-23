'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface PropFirm {
  id: string;
  name: string;
  evalPrice: string;
  profitTarget: string;
  maxDD: string;
  dailyLoss: string;
  minDays: number;
  compatible: number;
  accounts: string;
}

const mockFirms: PropFirm[] = [
  { id: 'ttp', name: 'Trade The Pool', evalPrice: '$99-$399', profitTarget: '$3K-$12K', maxDD: '6%', dailyLoss: '2%', minDays: 5, compatible: 12, accounts: '25K-200K' },
  { id: 'ftmo', name: 'FTMO', evalPrice: '$155-$1,080', profitTarget: '10%', maxDD: '10%', dailyLoss: '5%', minDays: 4, compatible: 8, accounts: '10K-200K' },
  { id: 'topstep', name: 'Topstep', evalPrice: '$49-$149/mo', profitTarget: '$3K-$9K', maxDD: '$2K-$4.5K', dailyLoss: '$1K-$2K', minDays: 0, compatible: 6, accounts: '50K-150K' },
  { id: 'apex', name: 'Apex Trader', evalPrice: '$37-$187/mo', profitTarget: '$1.5K-$15K', maxDD: '$1.5K-$6.25K', dailyLoss: 'N/A', minDays: 7, compatible: 5, accounts: '25K-250K' },
];

const rules = ['Eval Price', 'Profit Target', 'Max DD', 'Daily Loss', 'Min Days', 'Accounts', 'Portfolios'];

function getFirmValue(firm: PropFirm, rule: string): string {
  switch (rule) {
    case 'Eval Price': return firm.evalPrice;
    case 'Profit Target': return firm.profitTarget;
    case 'Max DD': return firm.maxDD;
    case 'Daily Loss': return firm.dailyLoss;
    case 'Min Days': return firm.minDays === 0 ? 'None' : `${firm.minDays}`;
    case 'Accounts': return firm.accounts;
    case 'Portfolios': return `${firm.compatible}`;
    default: return '-';
  }
}

function getRuleColor(rule: string): string {
  if (rule === 'Max DD' || rule === 'Daily Loss') return 'var(--red)';
  if (rule === 'Portfolios') return 'var(--accent)';
  return 'var(--text-primary)';
}

export default function PropFirmsV3() {
  return (
    <div>
      <PageHeader title="Prop Firm Hub" subtitle="Comparison table" backHref="/marketplace" />

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{ borderColor: 'var(--border)' }}>
                <th className="text-left py-3 px-4 text-xs font-medium w-32" style={{ color: 'var(--text-muted)' }}>Rule</th>
                {mockFirms.map((firm) => (
                  <th key={firm.id} className="text-center py-3 px-4">
                    <div className="flex flex-col items-center gap-1">
                      <span className="text-sm font-bold">{firm.name}</span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rules.map((rule, idx) => (
                <tr key={rule} className="border-b" style={{ borderColor: 'var(--border)', background: idx % 2 === 0 ? 'transparent' : 'var(--bg-input)' }}>
                  <td className="py-3 px-4 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>{rule}</td>
                  {mockFirms.map((firm) => (
                    <td key={firm.id} className="py-3 px-4 text-center text-sm font-medium" style={{ color: getRuleColor(rule) }}>
                      {getFirmValue(firm, rule)}
                    </td>
                  ))}
                </tr>
              ))}
              <tr>
                <td className="py-4 px-4"></td>
                {mockFirms.map((firm) => (
                  <td key={firm.id} className="py-4 px-4 text-center">
                    <button className="px-4 py-2 rounded-lg text-xs font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                      Sign Up
                    </button>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </Card>

      {/* Compatible portfolio counts */}
      <div className="grid grid-cols-4 gap-3 mt-4">
        {mockFirms.map((firm) => (
          <div key={firm.id} className="text-center rounded-lg p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{firm.name}</p>
            <p className="text-lg font-bold" style={{ color: 'var(--accent)' }}>{firm.compatible}</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>compatible portfolios</p>
          </div>
        ))}
      </div>
    </div>
  );
}
