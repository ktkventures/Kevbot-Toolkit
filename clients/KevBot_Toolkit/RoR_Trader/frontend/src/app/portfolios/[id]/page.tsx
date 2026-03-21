'use client';

import { useState } from 'react';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import Modal from '@/components/Modal';
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
            {tab === 'Account' && <AccountTab />}
            {!['Live Dashboard', 'Performance', 'Strategies', 'Account'].includes(tab) && (
              <Card><p style={{ color: 'var(--text-muted)' }}>{tab} — content placeholder</p></Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}


const mockLedger = [
  { date: '2026-03-20', type: 'Daily Trading P&L', amount: 47.50, summary: '5 trades, 1 change', trades: [
    { note: 'SPY LONG - Mass #3 trade #12', amount: 32.00 },
    { note: 'NVDA LONG - Mass #2 trade #45', amount: -18.50 },
    { note: 'AAPL LONG - Mass #5 trade #22', amount: 15.00 },
    { note: 'SPY LONG - Mass #1 trade #8', amount: 28.00 },
    { note: 'META LONG - Mass #13 trade #3', amount: -9.00 },
  ], changes: ['Risk adjusted: SPY LONG Mass #1 $100 → $115/trade'] },
  { date: '2026-03-19', type: 'Daily Trading P&L', amount: 113.60, summary: '13 trades, has notes', trades: [
    { note: 'Multiple trades across 6 strategies', amount: 113.60 },
  ], changes: [] },
  { date: '2026-03-18', type: 'Deposit', amount: 5000.00, summary: 'Initial funding', trades: [], changes: [] },
];


function AccountTab() {
  const [modalDate, setModalDate] = useState<string | null>(null);
  const [activeModalTab, setActiveModalTab] = useState('Trades');
  const modalEntry = mockLedger.find((e) => e.date === modalDate);

  return (
    <div>
      {/* Balance metrics */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="Current Balance" value="$85,161" />
        <MetricCard label="Starting Balance" value="$80,000" />
        <MetricCard label="Net Deposits" value="$5,000" />
        <MetricCard label="Trading P&L" value="$161" delta="+$161" positive />
      </div>

      {/* Balance History Chart */}
      <Card className="mb-6">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Balance History</h3>
        <div className="rounded-lg p-4" style={{ background: 'var(--bg-input)', height: 200 }}>
          {/* Simple SVG line chart mock */}
          <svg width="100%" height="100%" viewBox="0 0 400 160" preserveAspectRatio="none">
            <defs>
              <linearGradient id="balGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.3" />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d="M0,140 L50,138 L100,130 L150,120 L200,115 L250,100 L300,95 L350,80 L400,60" fill="url(#balGrad)" stroke="none" />
            <path d="M0,140 L50,138 L100,130 L150,120 L200,115 L250,100 L300,95 L350,80 L400,60" fill="none" stroke="var(--accent)" strokeWidth="2" />
          </svg>
        </div>
      </Card>

      {/* Ledger */}
      <Card>
        <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Ledger</h3>

        {/* Header */}
        <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Date</p>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Type</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Amount</p>
          <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Summary</p>
          <p className="col-span-2 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Action</p>
        </div>

        {/* Rows */}
        {mockLedger.map((entry) => (
          <div
            key={entry.date}
            className="grid grid-cols-12 gap-2 py-3 border-b items-center"
            style={{ borderColor: 'var(--border)' }}
          >
            <p className="col-span-2 text-sm">{entry.date}</p>
            <p className="col-span-3 text-sm" style={{ color: 'var(--text-secondary)' }}>{entry.type}</p>
            <p className="col-span-2 text-sm font-medium" style={{ color: entry.amount > 0 ? 'var(--green)' : 'var(--red)' }}>
              {entry.amount > 0 ? '+' : ''}${entry.amount.toFixed(2)}
            </p>
            <p className="col-span-3 text-xs" style={{ color: 'var(--text-muted)' }}>{entry.summary}</p>
            <div className="col-span-2">
              {entry.type.includes('Trading') && (
                <button
                  onClick={() => { setModalDate(entry.date); setActiveModalTab('Trades'); }}
                  className="px-3 py-1 rounded text-xs"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  Details
                </button>
              )}
            </div>
          </div>
        ))}
      </Card>

      {/* Daily Detail Modal */}
      <Modal
        title={`Daily Detail — ${modalDate || ''}`}
        isOpen={!!modalDate}
        onClose={() => setModalDate(null)}
        width="700px"
      >
        {/* Modal tabs */}
        <div className="flex gap-1 border-b mb-5" style={{ borderColor: 'var(--border)' }}>
          {['Trades', 'Portfolio Changes', 'Notes'].map((t) => (
            <button
              key={t}
              onClick={() => setActiveModalTab(t)}
              className="px-4 py-2 text-sm"
              style={{
                color: activeModalTab === t ? 'var(--accent)' : 'var(--text-muted)',
                borderBottom: activeModalTab === t ? '2px solid var(--accent)' : '2px solid transparent',
                marginBottom: '-1px',
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {activeModalTab === 'Trades' && modalEntry && (
          <div>
            <div
              className="rounded-lg p-4 mb-4"
              style={{ background: 'var(--accent-muted)' }}
            >
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Day Total P&L</p>
              <p className="text-2xl font-bold" style={{ color: modalEntry.amount > 0 ? 'var(--green)' : 'var(--red)' }}>
                {modalEntry.amount > 0 ? '+' : ''}${modalEntry.amount.toFixed(2)}
              </p>
            </div>

            {/* Trade rows */}
            <div className="grid grid-cols-12 gap-2 pb-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <p className="col-span-1 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>#</p>
              <p className="col-span-8 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Trade</p>
              <p className="col-span-3 text-xs font-medium" style={{ color: 'var(--text-muted)' }}>P&L</p>
            </div>
            {modalEntry.trades.map((trade, i) => (
              <div key={i} className="grid grid-cols-12 gap-2 py-2.5 border-b" style={{ borderColor: 'var(--border)' }}>
                <p className="col-span-1 text-sm" style={{ color: 'var(--text-muted)' }}>{i + 1}</p>
                <p className="col-span-8 text-sm" style={{ color: 'var(--text-secondary)' }}>{trade.note}</p>
                <p className="col-span-3 text-sm font-medium" style={{ color: trade.amount > 0 ? 'var(--green)' : 'var(--red)' }}>
                  {trade.amount > 0 ? '+' : ''}${trade.amount.toFixed(2)}
                </p>
              </div>
            ))}
          </div>
        )}

        {activeModalTab === 'Portfolio Changes' && modalEntry && (
          <div>
            {modalEntry.changes.length > 0 ? (
              modalEntry.changes.map((change, i) => (
                <div key={i} className="flex items-center gap-3 py-3 border-b" style={{ borderColor: 'var(--border)' }}>
                  <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
                    Risk Adjusted
                  </span>
                  <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{change}</span>
                </div>
              ))
            ) : (
              <p style={{ color: 'var(--text-muted)' }}>No portfolio changes on this date.</p>
            )}
          </div>
        )}

        {activeModalTab === 'Notes' && (
          <div>
            <textarea
              className="w-full rounded-lg p-3 text-sm"
              style={{
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-primary)',
                resize: 'vertical',
              }}
              rows={5}
              placeholder="Add your thoughts, observations, or context for this trading day..."
            />
            <button
              className="mt-3 px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Save Note
            </button>
          </div>
        )}
      </Modal>
    </div>
  );
}
