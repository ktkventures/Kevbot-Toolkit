'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsDisplayV2() {
  const [timezone, setTimezone] = useState('US/Mountain');
  const [dateFormat, setDateFormat] = useState('YYYY-MM-DD');
  const [numberFormat, setNumberFormat] = useState('1,000.00');
  const [currencySymbol, setCurrencySymbol] = useState('$');
  const [candleStyle, setCandleStyle] = useState('Candle');
  const [defaultTf, setDefaultTf] = useState('1Min');
  const [gridLines, setGridLines] = useState(true);
  const [tableDensity, setTableDensity] = useState<'compact' | 'comfortable' | 'spacious'>('comfortable');

  function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
    return (
      <div>
        <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>{label}</label>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
        >
          {options.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
        </select>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Display" subtitle="Configure how data is formatted and displayed" />

      <div className="max-w-2xl space-y-6">
        {/* Regional Preferences */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Regional Preferences
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <SelectField label="Timezone" value={timezone} onChange={setTimezone} options={['US/Mountain', 'US/Eastern', 'US/Central', 'US/Pacific', 'UTC', 'Europe/London', 'Asia/Tokyo']} />
            <SelectField label="Date Format" value={dateFormat} onChange={setDateFormat} options={['YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY', 'DD-MMM-YYYY']} />
            <SelectField label="Number Format" value={numberFormat} onChange={setNumberFormat} options={['1,000.00', '1.000,00', '1 000.00', '1 000,00']} />
            <SelectField label="Currency Symbol" value={currencySymbol} onChange={setCurrencySymbol} options={['$', 'USD', 'None']} />
          </div>
        </Card>

        {/* Chart Defaults */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Chart Defaults
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <SelectField label="Candle Style" value={candleStyle} onChange={setCandleStyle} options={['Candle', 'Hollow Candle', 'OHLC Bars', 'Heikin Ashi', 'Line']} />
            <SelectField label="Default Timeframe" value={defaultTf} onChange={setDefaultTf} options={['1Min', '5Min', '15Min', '1H', '4H', '1D']} />
            <div>
              <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Grid Lines</label>
              <label className="flex items-center gap-2 mt-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={gridLines}
                  onChange={(e) => setGridLines(e.target.checked)}
                  className="accent-[var(--accent)]"
                />
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
                  {gridLines ? 'Visible' : 'Hidden'}
                </span>
              </label>
            </div>
          </div>
        </Card>

        {/* Table Density */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Table Density
          </h3>
          <div className="flex gap-3">
            {(['compact', 'comfortable', 'spacious'] as const).map((density) => (
              <button
                key={density}
                onClick={() => setTableDensity(density)}
                className="flex-1 rounded-lg border py-3 px-4 text-sm font-medium capitalize transition-all"
                style={{
                  borderColor: tableDensity === density ? 'var(--accent)' : 'var(--border)',
                  background: tableDensity === density ? 'var(--accent-muted)' : 'var(--bg-input)',
                  color: tableDensity === density ? 'var(--accent)' : 'var(--text-secondary)',
                  boxShadow: tableDensity === density ? '0 0 0 1px var(--accent)' : 'none',
                }}
              >
                {density}
                <p className="text-xs font-normal mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  {density === 'compact' ? 'More rows visible' : density === 'comfortable' ? 'Balanced spacing' : 'Easy to read'}
                </p>
              </button>
            ))}
          </div>

          {/* Preview */}
          <div className="mt-4 rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border)' }}>
            <div
              className="grid grid-cols-4 gap-2 text-xs font-medium px-3"
              style={{
                color: 'var(--text-muted)',
                background: 'var(--bg-input)',
                paddingTop: tableDensity === 'compact' ? 4 : tableDensity === 'comfortable' ? 8 : 12,
                paddingBottom: tableDensity === 'compact' ? 4 : tableDensity === 'comfortable' ? 8 : 12,
              }}
            >
              <span>Symbol</span><span>Price</span><span>Change</span><span>Volume</span>
            </div>
            {['NVDA', 'SPY', 'AAPL'].map((sym) => (
              <div
                key={sym}
                className="grid grid-cols-4 gap-2 text-sm px-3 border-t"
                style={{
                  borderColor: 'var(--border)',
                  paddingTop: tableDensity === 'compact' ? 4 : tableDensity === 'comfortable' ? 10 : 16,
                  paddingBottom: tableDensity === 'compact' ? 4 : tableDensity === 'comfortable' ? 10 : 16,
                }}
              >
                <span className="font-medium">{sym}</span>
                <span>{currencySymbol !== 'None' ? currencySymbol : ''}487.50</span>
                <span style={{ color: 'var(--green)' }}>+2.3%</span>
                <span style={{ color: 'var(--text-muted)' }}>12{numberFormat.charAt(1)}450{numberFormat.charAt(5) || ''}</span>
              </div>
            ))}
          </div>
        </Card>

        <button
          className="w-full px-4 py-2.5 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}
