'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* TYPES & CONSTANTS                                                          */
/* ========================================================================= */

const CANDLE_THEMES: Record<string, { label: string; up: string; down: string; desc: string; upBorder?: string }> = {
  classic: { label: 'Classic', up: '#26a69a', down: '#ef5350', desc: 'Traditional green/red' },
  neutral: { label: 'Neutral', up: '#FFFFFF', down: '#787B86', desc: 'White up, gray down — lets indicators stand out' },
  neutral_hollow: { label: 'Neutral Hollow', up: 'transparent', upBorder: '#FFFFFF', down: '#787B86', desc: 'Hollow up candles, filled down' },
  monochrome: { label: 'Monochrome', up: '#d4d4d8', down: '#52525b', desc: 'Subtle gray tones for minimal distraction' },
  neon: { label: 'Neon', up: '#00ff88', down: '#ff0055', desc: 'High contrast for dark themes' },
};

type CandleThemeKey = string;

const EXEC_STYLES = {
  badges: { label: 'Badges', desc: 'Colored rounded badges' },
  tags: { label: 'Inline Tags', desc: 'Compact monospace tags' },
  dots: { label: 'Color Dots', desc: 'Minimal colored dots with tooltip' },
};

type ExecStyleKey = keyof typeof EXEC_STYLES;

const VARIATION_STYLES = {
  parenthetical: { label: 'Parenthetical', example: 'EMA Stack (Scalping)', desc: 'Name (Version) — classic format' },
  badge: { label: 'Badge', example: 'EMA Stack', badgeText: 'Scalping', desc: 'Name with version as a separate badge' },
  separator: { label: 'Separator', example: 'EMA Stack · Scalping', desc: 'Name · Version with dot separator' },
};

type VariationStyleKey = keyof typeof VARIATION_STYLES;

/* ========================================================================= */
/* HELPER: Execution Type Badge Preview                                       */
/* ========================================================================= */

const EXEC_TYPES = [
  { code: 'C', label: 'Bar Close', color: 'var(--green)', bg: 'var(--green-muted)', desc: 'Triggers on completed candle close' },
  { code: 'L0', label: 'Level Cross', color: 'var(--blue)', bg: 'var(--blue-muted)', desc: 'Triggers on intra-bar level cross' },
  { code: 'L1', label: 'Fixed Level', color: 'var(--teal)', bg: 'rgba(38, 198, 218, 0.15)', desc: 'Previous bar\'s level as fixed reference' },
  { code: 'HM', label: 'Hybrid Market', color: 'var(--orange)', bg: 'var(--orange-muted)', desc: 'L1 cross + bar-close confirmation → market fill' },
  { code: 'HL', label: 'Hybrid Limit', color: 'var(--accent)', bg: 'var(--accent-muted)', desc: 'L1 cross + limit fill at entry price' },
];

function ExecPreview({ style }: { style: ExecStyleKey }) {
  if (style === 'badges') {
    return (
      <div className="flex flex-wrap gap-1.5">
        {EXEC_TYPES.map((et) => (
          <span
            key={et.code}
            className="text-xs font-mono px-2 py-0.5 rounded-full font-medium"
            style={{ color: et.color, background: et.bg }}
          >
            [{et.code}]
          </span>
        ))}
      </div>
    );
  }
  if (style === 'tags') {
    return (
      <div className="flex flex-wrap gap-1.5">
        {EXEC_TYPES.map((et) => (
          <span
            key={et.code}
            className="text-xs font-mono px-1.5 py-0.5 rounded font-medium"
            style={{ color: et.color, background: et.bg, border: `1px solid ${et.color}30` }}
          >
            {et.code}
          </span>
        ))}
      </div>
    );
  }
  // dots
  return (
    <div className="flex flex-wrap gap-3">
      {EXEC_TYPES.map((et) => (
        <div key={et.code} className="flex items-center gap-1.5" title={et.label}>
          <span className="w-2.5 h-2.5 rounded-full" style={{ background: et.color }} />
          <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{et.code}</span>
        </div>
      ))}
    </div>
  );
}

/* ========================================================================= */
/* HELPER: Variation Name Preview                                             */
/* ========================================================================= */

function VariationPreview({ style }: { style: VariationStyleKey }) {
  const packs = [
    { name: 'EMA Stack', version: 'Default' },
    { name: 'EMA Stack', version: 'Scalping' },
    { name: 'MACD Line', version: 'Default' },
  ];

  return (
    <div className="space-y-1.5">
      {packs.map((pack, i) => (
        <div key={i} className="flex items-center gap-2">
          {style === 'parenthetical' && (
            <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
              {pack.name} <span style={{ color: 'var(--text-muted)' }}>({pack.version})</span>
            </span>
          )}
          {style === 'badge' && (
            <>
              <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{pack.name}</span>
              <span
                className="text-xs px-1.5 py-0.5 rounded"
                style={{
                  background: pack.version === 'Default' ? 'var(--bg-input)' : 'var(--accent-muted)',
                  color: pack.version === 'Default' ? 'var(--text-muted)' : 'var(--accent)',
                }}
              >
                {pack.version}
              </span>
            </>
          )}
          {style === 'separator' && (
            <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
              {pack.name} <span style={{ color: 'var(--text-muted)' }}>&middot;</span>{' '}
              <span style={{ color: 'var(--accent)' }}>{pack.version}</span>
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

/* ========================================================================= */
/* HELPER: Mock Candlestick SVG                                               */
/* ========================================================================= */

function CandlestickPreview({
  theme,
  style,
  gridLines,
  visibleCandles,
}: {
  theme: CandleThemeKey;
  style: string;
  gridLines: boolean;
  visibleCandles: number;
}) {
  const t = CANDLE_THEMES[theme];
  // Generate deterministic mock candles
  const candles = [];
  let price = 485;
  for (let i = 0; i < Math.min(visibleCandles, 30); i++) {
    const change = (Math.sin(i * 0.8) + Math.cos(i * 1.3)) * 2;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.abs(change) * 0.5 + 0.5;
    const low = Math.min(open, close) - Math.abs(change) * 0.5 - 0.5;
    candles.push({ open, close, high, low, up: close >= open });
    price = close;
  }

  const allPrices = candles.flatMap((c) => [c.high, c.low]);
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const range = maxP - minP || 1;

  const svgW = 520;
  const svgH = 180;
  const pad = 12;
  const plotW = svgW - pad * 2;
  const plotH = svgH - pad * 2;

  const barW = Math.max(2, Math.min(16, plotW / candles.length - 2));
  const gap = (plotW - barW * candles.length) / (candles.length + 1);

  function yPos(p: number) {
    return pad + plotH - ((p - minP) / range) * plotH;
  }

  const isLine = style === 'Line';
  const isBar = style === 'OHLC Bars';
  const isHollow = style === 'Hollow Candle';

  return (
    <svg width="100%" height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="none">
      {/* Grid */}
      {gridLines && [0.25, 0.5, 0.75].map((frac) => (
        <line
          key={frac}
          x1={pad} y1={pad + plotH * frac}
          x2={svgW - pad} y2={pad + plotH * frac}
          stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4"
        />
      ))}

      {isLine ? (
        // Line chart
        <polyline
          points={candles.map((c, i) => {
            const x = pad + gap + i * (barW + gap) + barW / 2;
            const y = yPos(c.close);
            return `${x},${y}`;
          }).join(' ')}
          fill="none"
          stroke={t.up === 'transparent' ? (t.upBorder || '#fff') : t.up}
          strokeWidth="1.5"
        />
      ) : (
        candles.map((c, i) => {
          const x = pad + gap + i * (barW + gap);
          const cx = x + barW / 2;
          const fillColor = c.up ? t.up : t.down;
          const borderColor = c.up ? (t.upBorder || t.up) : t.down;
          const wickColor = c.up ? (t.upBorder || t.up) : t.down;

          if (isBar) {
            // OHLC bars
            return (
              <g key={i}>
                <line x1={cx} y1={yPos(c.high)} x2={cx} y2={yPos(c.low)} stroke={wickColor} strokeWidth="1" />
                <line x1={cx - barW / 3} y1={yPos(c.open)} x2={cx} y2={yPos(c.open)} stroke={wickColor} strokeWidth="1.5" />
                <line x1={cx} y1={yPos(c.close)} x2={cx + barW / 3} y2={yPos(c.close)} stroke={wickColor} strokeWidth="1.5" />
              </g>
            );
          }

          // Candle or Hollow Candle
          const bodyTop = yPos(Math.max(c.open, c.close));
          const bodyBottom = yPos(Math.min(c.open, c.close));
          const bodyH = Math.max(1, bodyBottom - bodyTop);

          return (
            <g key={i}>
              <line x1={cx} y1={yPos(c.high)} x2={cx} y2={bodyTop} stroke={wickColor} strokeWidth="1" />
              <line x1={cx} y1={bodyBottom} x2={cx} y2={yPos(c.low)} stroke={wickColor} strokeWidth="1" />
              <rect
                x={x} y={bodyTop} width={barW} height={bodyH}
                fill={isHollow && c.up ? 'transparent' : fillColor}
                stroke={borderColor}
                strokeWidth="1"
                rx="0.5"
              />
            </g>
          );
        })
      )}

      {/* Price axis labels */}
      {[0, 0.5, 1].map((frac) => {
        const p = minP + range * (1 - frac);
        return (
          <text
            key={frac}
            x={svgW - pad + 2} y={pad + plotH * frac + 3}
            fill="var(--text-muted)" fontSize="8" textAnchor="start"
          >
            {p.toFixed(1)}
          </text>
        );
      })}
    </svg>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsDisplayV4() {
  // Regional
  const [timezone, setTimezone] = useState('US/Mountain');
  const [dateFormat, setDateFormat] = useState('YYYY-MM-DD');
  const [numberFormat, setNumberFormat] = useState<'comma' | 'dot' | 'space'>('comma');
  const [currencySymbol, setCurrencySymbol] = useState('$');

  // Chart
  const [candleStyle, setCandleStyle] = useState('Candle');
  const [candleTheme, setCandleTheme] = useState<CandleThemeKey>('neutral');
  const [defaultTf, setDefaultTf] = useState('1Min');
  const [visibleCandles, setVisibleCandles] = useState(20);
  const [gridLines, setGridLines] = useState(true);
  const [chartLibrary, setChartLibrary] = useState('lightweight');

  // Table
  const [tableDensity, setTableDensity] = useState<'compact' | 'comfortable' | 'spacious'>('comfortable');

  // Styling
  const [execStyle, setExecStyle] = useState<ExecStyleKey>('badges');
  const [variationStyle, setVariationStyle] = useState<VariationStyleKey>('badge');

  // Format helpers
  function formatDate(d: Date): string {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    if (dateFormat === 'MM/DD/YYYY') return `${m}/${day}/${y}`;
    if (dateFormat === 'DD/MM/YYYY') return `${day}/${m}/${y}`;
    return `${y}-${m}-${day}`;
  }

  function formatNumber(n: number): string {
    const [int, dec] = n.toFixed(2).split('.');
    const thousands = numberFormat === 'comma' ? ',' : numberFormat === 'dot' ? '.' : ' ';
    const decimal = numberFormat === 'dot' ? ',' : '.';
    const formatted = int.replace(/\B(?=(\d{3})+(?!\d))/g, thousands);
    return formatted + decimal + dec;
  }

  function formatCurrency(n: number): string {
    const num = formatNumber(n);
    if (currencySymbol === 'None') return num;
    return currencySymbol + num;
  }

  const now = new Date();
  const densityPadding = { compact: 'py-1', comfortable: 'py-2.5', spacious: 'py-4' };

  return (
    <div>
      <PageHeader
        title="Display Preferences"
        subtitle="Configure formatting, chart appearance, and component styling — see a live preview on the right"
      />

      <div className="grid grid-cols-2 gap-6">
        {/* ============ LEFT: Controls ============ */}
        <div className="space-y-5">

          {/* Regional */}
          <Card>
            <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Regional</h3>
            <div className="space-y-3">
              {[
                { label: 'Timezone', value: timezone, setter: setTimezone, options: ['US/Mountain', 'US/Eastern', 'US/Central', 'US/Pacific', 'UTC', 'Europe/London', 'Asia/Tokyo'] },
                { label: 'Date Format', value: dateFormat, setter: setDateFormat, options: ['YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY'] },
                { label: 'Currency Symbol', value: currencySymbol, setter: setCurrencySymbol, options: ['$', 'USD ', 'None'] },
              ].map((field) => (
                <div key={field.label} className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{field.label}</span>
                  <select
                    value={field.value}
                    onChange={(e) => field.setter(e.target.value)}
                    className="px-3 py-1.5 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  >
                    {field.options.map((o) => <option key={o} value={o}>{o.trim() || 'None'}</option>)}
                  </select>
                </div>
              ))}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Number Format</span>
                <div className="flex gap-1">
                  {([
                    { id: 'comma' as const, label: '1,000.00' },
                    { id: 'dot' as const, label: '1.000,00' },
                    { id: 'space' as const, label: '1 000.00' },
                  ]).map((fmt) => (
                    <button
                      key={fmt.id}
                      onClick={() => setNumberFormat(fmt.id)}
                      className="px-2 py-1 rounded text-xs font-mono"
                      style={{
                        background: numberFormat === fmt.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: numberFormat === fmt.id ? 'var(--accent)' : 'var(--text-muted)',
                        border: `1px solid ${numberFormat === fmt.id ? 'var(--accent)' : 'var(--border)'}`,
                      }}
                    >
                      {fmt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* Chart Settings */}
          <Card>
            <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Chart</h3>
            <div className="space-y-3">
              {/* Chart Library */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Chart Library</span>
                <div className="flex gap-1">
                  {[
                    { id: 'lightweight', label: 'Lightweight Charts' },
                    { id: 'echarts', label: 'ECharts' },
                  ].map((lib) => (
                    <button
                      key={lib.id}
                      onClick={() => setChartLibrary(lib.id)}
                      className="px-2 py-1 rounded text-xs"
                      style={{
                        background: chartLibrary === lib.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: chartLibrary === lib.id ? 'var(--accent)' : 'var(--text-muted)',
                        border: `1px solid ${chartLibrary === lib.id ? 'var(--accent)' : 'var(--border)'}`,
                      }}
                    >
                      {lib.label}
                    </button>
                  ))}
                </div>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {chartLibrary === 'lightweight'
                  ? 'TradingView-style charts with zoom/pan and crosshair. Fast and lightweight.'
                  : 'Rich interactive charts with hover tooltips showing OHLCV values per candle. More features, slightly heavier.'}
              </p>

              {/* Candle Style */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Candle Style</span>
                <select
                  value={candleStyle}
                  onChange={(e) => setCandleStyle(e.target.value)}
                  className="px-3 py-1.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  {['Candle', 'Hollow Candle', 'OHLC Bars', 'Heikin Ashi', 'Line'].map((o) => <option key={o}>{o}</option>)}
                </select>
              </div>

              {/* Candle Color Theme */}
              <div>
                <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Candle Colors</span>
                <div className="grid grid-cols-5 gap-1.5">
                  {(Object.keys(CANDLE_THEMES) as CandleThemeKey[]).map((key) => {
                    const theme = CANDLE_THEMES[key];
                    const isActive = candleTheme === key;
                    return (
                      <button
                        key={key}
                        onClick={() => setCandleTheme(key)}
                        className="rounded-lg p-2 text-center"
                        style={{
                          background: isActive ? 'var(--accent-muted)' : 'var(--bg-input)',
                          border: `1px solid ${isActive ? 'var(--accent)' : 'var(--border)'}`,
                        }}
                        title={theme.desc}
                      >
                        <div className="flex justify-center gap-1 mb-1">
                          <span className="w-2.5 h-5 rounded-sm" style={{ background: theme.up === 'transparent' ? 'none' : theme.up, border: theme.up === 'transparent' ? `1px solid ${theme.upBorder || '#fff'}` : 'none' }} />
                          <span className="w-2.5 h-5 rounded-sm" style={{ background: theme.down }} />
                        </div>
                        <span className="text-xs block" style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}>
                          {theme.label}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Default TF */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Default Timeframe</span>
                <select
                  value={defaultTf}
                  onChange={(e) => setDefaultTf(e.target.value)}
                  className="px-3 py-1.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  {['1Min', '5Min', '15Min', '1H', '4H', '1D'].map((o) => <option key={o}>{o}</option>)}
                </select>
              </div>

              {/* Visible Candles */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Visible Candles</span>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="10"
                    max="30"
                    value={visibleCandles}
                    onChange={(e) => setVisibleCandles(Number(e.target.value))}
                    className="w-24"
                  />
                  <span className="text-xs font-mono w-6 text-center" style={{ color: 'var(--text-muted)' }}>{visibleCandles}</span>
                </div>
              </div>

              {/* Grid Lines */}
              <div className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Grid Lines</span>
                <button
                  onClick={() => setGridLines(!gridLines)}
                  className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                  style={{
                    background: gridLines ? 'var(--accent)' : 'var(--bg-input)',
                    border: gridLines ? 'none' : '1px solid var(--border)',
                  }}
                >
                  <div
                    className="w-4 h-4 rounded-full absolute top-1 transition-all"
                    style={{ background: gridLines ? 'white' : 'var(--text-muted)', left: gridLines ? '22px' : '4px' }}
                  />
                </button>
              </div>
            </div>
          </Card>

          {/* Execution Type Styling */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Execution Types</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              How execution types ([C], [L0], [HM], etc.) are displayed throughout the app.
            </p>
            <div className="flex gap-1.5 mb-3">
              {(Object.keys(EXEC_STYLES) as ExecStyleKey[]).map((key) => (
                <button
                  key={key}
                  onClick={() => setExecStyle(key)}
                  className="flex-1 py-1.5 rounded-lg text-xs font-medium"
                  style={{
                    background: execStyle === key ? 'var(--accent-muted)' : 'var(--bg-input)',
                    color: execStyle === key ? 'var(--accent)' : 'var(--text-muted)',
                    border: `1px solid ${execStyle === key ? 'var(--accent)' : 'var(--border)'}`,
                  }}
                >
                  {EXEC_STYLES[key].label}
                </button>
              ))}
            </div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {EXEC_STYLES[execStyle].desc}
            </p>
          </Card>

          {/* Variation Name Styling */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Pack Variation Names</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              How confluence pack version names are displayed (e.g., Default, Scalping, Custom).
            </p>
            <div className="flex gap-1.5">
              {(Object.keys(VARIATION_STYLES) as VariationStyleKey[]).map((key) => (
                <button
                  key={key}
                  onClick={() => setVariationStyle(key)}
                  className="flex-1 py-1.5 rounded-lg text-xs font-medium"
                  style={{
                    background: variationStyle === key ? 'var(--accent-muted)' : 'var(--bg-input)',
                    color: variationStyle === key ? 'var(--accent)' : 'var(--text-muted)',
                    border: `1px solid ${variationStyle === key ? 'var(--accent)' : 'var(--border)'}`,
                  }}
                >
                  {VARIATION_STYLES[key].label}
                </button>
              ))}
            </div>
          </Card>

          {/* Table Density */}
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Table Density</h3>
            <div className="flex gap-2">
              {(['compact', 'comfortable', 'spacious'] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setTableDensity(d)}
                  className="flex-1 py-2 rounded-lg text-xs font-medium capitalize"
                  style={{
                    background: tableDensity === d ? 'var(--accent-muted)' : 'var(--bg-input)',
                    color: tableDensity === d ? 'var(--accent)' : 'var(--text-muted)',
                    border: `1px solid ${tableDensity === d ? 'var(--accent)' : 'var(--border)'}`,
                  }}
                >
                  {d}
                </button>
              ))}
            </div>
          </Card>
        </div>

        {/* ============ RIGHT: Live Preview ============ */}
        <div className="space-y-4">

          {/* Chart Preview */}
          <Card>
            <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Chart Preview</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              {candleStyle} &middot; {CANDLE_THEMES[candleTheme].label} &middot; {visibleCandles} candles &middot; {gridLines ? 'Grid on' : 'Grid off'}
            </p>
            <div
              className="rounded-lg overflow-hidden"
              style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}
            >
              <CandlestickPreview
                theme={candleTheme}
                style={candleStyle}
                gridLines={gridLines}
                visibleCandles={visibleCandles}
              />
            </div>
          </Card>

          {/* Execution Type Preview */}
          <Card>
            <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Execution Types</h3>
            <ExecPreview style={execStyle} />
            <div className="mt-3 space-y-1.5">
              {EXEC_TYPES.map((et) => (
                <div key={et.code} className="flex items-center gap-2">
                  {execStyle === 'badges' && (
                    <span className="text-xs font-mono px-2 py-0.5 rounded-full" style={{ color: et.color, background: et.bg }}>[{et.code}]</span>
                  )}
                  {execStyle === 'tags' && (
                    <span className="text-xs font-mono px-1.5 py-0.5 rounded" style={{ color: et.color, background: et.bg, border: `1px solid ${et.color}30` }}>{et.code}</span>
                  )}
                  {execStyle === 'dots' && (
                    <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: et.color }} />
                  )}
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{et.label} — {et.desc}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Variation Name Preview */}
          <Card>
            <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Pack Names</h3>
            <VariationPreview style={variationStyle} />
            <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
              {VARIATION_STYLES[variationStyle].desc}
            </p>
          </Card>

          {/* Date/Currency/Table Preview */}
          <Card>
            <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Formatting</h3>

            <div className="rounded-lg p-3 mb-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Date & Time</p>
              <p className="text-sm font-medium">{formatDate(now)} 10:32:15 AM {timezone}</p>
            </div>

            <div className="rounded-lg p-3 mb-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Price & Currency</p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Price</p>
                  <p className="text-base font-semibold">{formatCurrency(487.50)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Volume</p>
                  <p className="text-base font-semibold">{formatNumber(12450000)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L (Win)</p>
                  <p className="text-base font-semibold" style={{ color: 'var(--green)' }}>+{formatCurrency(2847.32)}</p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>P&L (Loss)</p>
                  <p className="text-base font-semibold" style={{ color: 'var(--red)' }}>-{formatCurrency(1203.45)}</p>
                </div>
              </div>
            </div>

            {/* Table Preview */}
            <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
              <p className="text-xs px-3 pt-2 mb-2" style={{ color: 'var(--text-muted)' }}>Table ({tableDensity})</p>
              <div className="grid grid-cols-5 gap-2 text-xs font-medium px-3 pb-1 border-b" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)' }}>
                <span>Symbol</span><span>Entry</span><span>Price</span><span>P&L</span><span>Type</span>
              </div>
              {[
                { sym: 'NVDA', trigger: 'EMA Bull Cross', price: 487.50, pnl: '+1.2R', up: true, exec: 'C' },
                { sym: 'SPY', trigger: 'VWAP Cross', price: 562.10, pnl: '+0.8R', up: true, exec: 'L0' },
                { sym: 'TSLA', trigger: 'UT Bot Buy', price: 178.90, pnl: '-0.5R', up: false, exec: 'HM' },
              ].map((row) => (
                <div
                  key={row.sym}
                  className={`grid grid-cols-5 gap-2 text-sm px-3 border-t items-center ${densityPadding[tableDensity]}`}
                  style={{ borderColor: 'var(--border)' }}
                >
                  <span className="font-medium">{row.sym}</span>
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{row.trigger}</span>
                  <span>{formatCurrency(row.price)}</span>
                  <span style={{ color: row.up ? 'var(--green)' : 'var(--red)' }}>{row.pnl}</span>
                  <span>
                    {execStyle === 'badges' && (
                      <span className="text-xs font-mono px-1.5 py-0.5 rounded-full" style={{
                        color: EXEC_TYPES.find((e) => e.code === row.exec)?.color,
                        background: EXEC_TYPES.find((e) => e.code === row.exec)?.bg,
                      }}>[{row.exec}]</span>
                    )}
                    {execStyle === 'tags' && (
                      <span className="text-xs font-mono px-1 py-0.5 rounded" style={{
                        color: EXEC_TYPES.find((e) => e.code === row.exec)?.color,
                        background: EXEC_TYPES.find((e) => e.code === row.exec)?.bg,
                      }}>{row.exec}</span>
                    )}
                    {execStyle === 'dots' && (
                      <span className="w-2.5 h-2.5 rounded-full inline-block" style={{
                        background: EXEC_TYPES.find((e) => e.code === row.exec)?.color,
                      }} title={row.exec} />
                    )}
                  </span>
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
    </div>
  );
}
