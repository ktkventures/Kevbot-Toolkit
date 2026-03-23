'use client';

import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Card from '@/components/Card';
import MetricCard from '@/components/MetricCard';
import ChartPlaceholder from '@/components/ChartPlaceholder';

const TABS = [
  'Equity & KPIs', 'Extended', 'Price Chart', 'Live Chart',
  'Trade History', 'Confluence Analysis', 'Configuration', 'Alerts', 'Alert Analysis',
];

export default function StrategyDetailV1() {
  return (
    <div>
      <PageHeader
        title="NVDA LONG - Mass #2"
        subtitle="NVDA | LONG | 1Min | RTH | [C] EMA Price Cross Up"
        backHref="/strategies"
        actions={
          <>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Refresh</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Edit</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Clone</button>
            <button className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>Delete</button>
          </>
        }
      />

      <TabBar tabs={TABS}>
        {(tab) => (
          <div>
            {tab === 'Equity & KPIs' && (
              <div>
                <div className="grid grid-cols-8 gap-3 mb-6">
                  {[
                    { l: 'Trades', v: '224' }, { l: 'Win Rate', v: '54.0%' },
                    { l: 'PF', v: '2.05' }, { l: 'Avg R', v: '+0.42' },
                    { l: 'Total R', v: '+94.1' }, { l: 'Daily R', v: '+1.95' },
                    { l: 'R²', v: '0.91' }, { l: 'Max DD', v: '-4.5R' },
                  ].map((k, i) => <MetricCard key={i} label={k.l} value={k.v} />)}
                </div>
                <Card><ChartPlaceholder label="Equity Curve — BT (blue) + FW (orange) + Alerts (green)" height={300} /></Card>
              </div>
            )}
            {tab === 'Price Chart' && (
              <Card><ChartPlaceholder label="TradingView OHLC + indicators + trade markers (+/x)" height={500} /></Card>
            )}
            {tab === 'Live Chart' && (
              <Card><ChartPlaceholder label="Live Chart — auto-refreshing via Polygon WS" height={500} /></Card>
            )}
            {tab === 'Trade History' && (
              <Card>
                <p className="text-sm mb-3" style={{ color: 'var(--text-secondary)' }}>Trade history with entry/exit details, R-multiples, and execution types.</p>
                <ChartPlaceholder label="Trade table placeholder" height={200} />
              </Card>
            )}
            {tab === 'Confluence Analysis' && (
              <Card><p style={{ color: 'var(--text-muted)' }}>Per-group indicator charts with trade markers and interpreter state timelines.</p></Card>
            )}
            {tab === 'Configuration' && (
              <Card>
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-medium mb-3">Strategy Setup</h4>
                    {['Ticker: NVDA', 'Direction: LONG', 'Timeframe: 1Min', 'Entry: [C] EMA Price Cross Up', 'Exit: 4-bar exit', 'Stop: Swing (5 bars, $0.03 pad)'].map((s, i) => (
                      <p key={i} className="text-sm py-1" style={{ color: 'var(--text-secondary)' }}>{s}</p>
                    ))}
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-3">Confluence Conditions</h4>
                    {['5M-RVOL-HIGH (Default) [PB]', '1D-MACD_LINE-BULL (Default) [PB]'].map((c, i) => (
                      <p key={i} className="text-sm py-1" style={{ color: 'var(--text-secondary)' }}>{c}</p>
                    ))}
                  </div>
                </div>
              </Card>
            )}
            {!['Equity & KPIs', 'Price Chart', 'Live Chart', 'Trade History', 'Confluence Analysis', 'Configuration'].includes(tab) && (
              <Card><p style={{ color: 'var(--text-muted)' }}>{tab} — content placeholder</p></Card>
            )}
          </div>
        )}
      </TabBar>
    </div>
  );
}
