'use client';

import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

const TABS = [
  'Live Dashboard', 'Performance', 'Strategies', 'Prop Firm Check',
  'Account', 'Webhooks', 'Deploy',
];

export default function PortfolioDetail() {
  return (
    <div>
      <PageHeader
        title="My Portfolio"
        subtitle="8 strategies | $80,000 starting balance"
        backHref="/portfolios"
        actions={
          <>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Refresh</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Update Strategies</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Edit</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Clone</button>
          </>
        }
      />

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Live Dashboard' && (
              <div>
                <div className="grid grid-cols-5 gap-3 mb-6">
                  <MetricCard label="Alert Trades" value="17" />
                  <MetricCard label="Win Rate" value="52.9%" />
                  <MetricCard label="Total P&L" value="$161" delta="+$161" positive />
                  <MetricCard label="Expected P&L" value="$305" />
                  <MetricCard label="vs Plan" value="-$144" delta="below plan" positive={false} />
                </div>
                <Card className="mb-6"><ChartPlaceholder label="Performance vs Plan — plan line + confidence bands + actual" height={400} /></Card>
                <Card className="mb-6"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>Open Positions — no open positions</p></Card>
                <Card className="mb-6"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>Buying Power Tracker — chart placeholder</p></Card>
                <Card className="mb-6"><p className="text-sm" style={{ color: 'var(--text-muted)' }}>Anomalies — no anomalies detected</p></Card>
                <Card><p className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Trade History</p><ChartPlaceholder label="Scrollable trade table" height={300} /></Card>
              </div>
            )}
            {tab === 'Performance' && (
              <div>
                <div className="grid grid-cols-6 gap-3 mb-6">
                  <MetricCard label="Trades" value="847" />
                  <MetricCard label="Win Rate" value="55.2%" />
                  <MetricCard label="PF" value="1.89" />
                  <MetricCard label="Total P&L" value="$4,230" />
                  <MetricCard label="Balance" value="$84,230" />
                  <MetricCard label="Max DD" value="-2.1%" />
                </div>
                <Card className="mb-6"><ChartPlaceholder label="Combined Equity Curve — per-strategy lines + combined" height={400} /></Card>
                <Card className="mb-6"><ChartPlaceholder label="Drawdown Analysis" height={250} /></Card>
                <div className="grid grid-cols-2 gap-6">
                  <Card><ChartPlaceholder label="Daily P&L Distribution" height={300} /></Card>
                  <Card><ChartPlaceholder label="Strategy Correlation Heatmap" height={300} /></Card>
                </div>
              </div>
            )}
            {tab === 'Strategies' && (
              <div className="space-y-4">
                {['NVDA LONG - Mass #2', 'SPY LONG - Mass #1', 'AAPL LONG - Mass #5'].map((name, i) => (
                  <Card key={i}>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{name} <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green-muted)' }}>On Track</span></p>
                        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>$100/trade | WR 54% | PF 2.05</p>
                      </div>
                      <div className="flex gap-2">
                        <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>View Strategy</button>
                        <button className="px-3 py-1 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>View Chart</button>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
            {tab === 'Account' && (
              <div>
                <div className="grid grid-cols-4 gap-3 mb-6">
                  <MetricCard label="Current Balance" value="$80,161" />
                  <MetricCard label="Starting Balance" value="$80,000" />
                  <MetricCard label="Net Deposits" value="$0" />
                  <MetricCard label="Trading P&L" value="$161" />
                </div>
                <Card className="mb-6"><ChartPlaceholder label="Balance History" height={250} /></Card>
                <Card><p className="text-sm font-medium mb-3">Ledger</p><ChartPlaceholder label="Daily ledger with Details modal" height={200} /></Card>
              </div>
            )}
            {!['Live Dashboard', 'Performance', 'Strategies', 'Account'].includes(tab) && (
              <Card><p style={{ color: 'var(--text-muted)' }}>{tab} — content placeholder</p></Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
