'use client';

/**
 * Display Preferences — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: src/app/settings/display/versions/V5.tsx (1,353 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useEffect, useCallback } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useSettings, useSaveSettings } from '@/hooks/queries/useSettings';

/* ========================================================================= */
/* TYPES & CONSTANTS                                                          */
/* ========================================================================= */

const CANDLE_THEMES: Record<string, { label: string; up: string; down: string; desc: string; upBorder?: string; isTheme?: boolean }> = {
  theme: { label: 'Theme', up: 'var(--accent)', down: 'var(--red)', desc: 'Matches your current app theme', isTheme: true },
  classic: { label: 'Classic', up: '#26a69a', down: '#ef5350', desc: 'Traditional green/red' },
  neutral: { label: 'Neutral', up: '#FFFFFF', down: '#787B86', desc: 'White up, gray down — lets indicators stand out' },
  neutral_hollow: { label: 'Neutral Hollow', up: 'transparent', upBorder: '#FFFFFF', down: '#787B86', desc: 'Hollow up candles, filled down' },
  monochrome: { label: 'Monochrome', up: '#d4d4d8', down: '#52525b', desc: 'Subtle gray tones for minimal distraction' },
  neon: { label: 'Neon', up: '#00ff88', down: '#ff0055', desc: 'High contrast for dark themes' },
};

const CANDLE_STYLES = [
  { id: 'Candle', label: 'Candle', icon: 'filled', desc: 'Standard filled candlesticks' },
  { id: 'Hollow Candle', label: 'Hollow', icon: 'hollow', desc: 'Hollow up candles, filled down' },
  { id: 'OHLC Bars', label: 'OHLC', icon: 'bars', desc: 'Open-high-low-close bar chart' },
  { id: 'Heikin Ashi', label: 'Heikin Ashi', icon: 'ha', desc: 'Smoothed candles for trend clarity' },
  { id: 'Line', label: 'Line', icon: 'line', desc: 'Simple close-price line chart' },
];

type ChartPane = 'confluence' | 'price' | 'oscillators';

const CHART_PANES: { id: ChartPane; label: string; desc: string }[] = [
  { id: 'confluence', label: 'Confluence Heatmap', desc: 'Red/green/yellow blocks showing confluence condition status per bar' },
  { id: 'price', label: 'Price Chart', desc: 'OHLC candlestick chart with indicator overlays and trade markers' },
  { id: 'oscillators', label: 'Oscillators', desc: 'MACD histogram, RSI, and other oscillator indicators' },
];

type BadgeShape = 'rounded' | 'square';
type BracketStyle = 'brackets' | 'none';

const VARIATION_STYLES = {
  parenthetical: { label: 'Parenthetical', desc: 'Name (Version) — classic format' },
  badge: { label: 'Badge', desc: 'Name with version as a separate colored badge' },
  separator: { label: 'Separator', desc: 'Name · Version with dot separator' },
};
type VariationStyleKey = keyof typeof VARIATION_STYLES;

// Execution Types — 31F naming convention (colors applied dynamically from settings)
const EXEC_TYPES = [
  { code: 'C', label: 'Close', desc: 'Enter on bar close — waits for completed candle' },
  { code: 'L', label: 'Level', desc: 'Enter on intra-bar level cross — fastest fill' },
  { code: 'LC', label: 'Level-Close', desc: 'Level cross entry, bar close confirmation — two-stage' },
  { code: 'CC', label: 'Close-Close', desc: 'Bar close entry, next bar close confirmation — most conservative' },
];

// Fidelity Types — confluence condition timing
const FIDELITY_TYPES = [
  { code: 'PB', label: 'Previous Bar', desc: 'Condition checked on previous closed bar — standard, always reliable' },
  { code: 'CB', label: 'Current Bar', desc: 'Condition checked on forming bar using 1-second sub-data — requires HiFi backtest' },
];

const TABS = ['Price Charts', 'Equity Curves', 'Formatting', 'Components', 'Tables'] as const;
type TabKey = typeof TABS[number];

/* ========================================================================= */
/* HELPER: Candlestick SVG Preview                                            */
/* ========================================================================= */

interface MarkerConfig {
  entryColor: string;
  exitWinColor: string;
  exitLossColor: string;
  exitStopColor: string;
  exitBarCountColor: string;
  exitHybridColor: string;
  btEntryMark: string;
  btExitMark: string;
  alertEntryMark: string;
  alertExitMark: string;
  showLabels: boolean;
}

function CandlestickPreview({ theme, style, gridLines, visibleCandles, rightOffset = 0, markers: mCfg }: { theme: string; style: string; gridLines: boolean; visibleCandles: number; rightOffset?: number; markers?: MarkerConfig }) {
  const t = CANDLE_THEMES[theme];
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

  // Compute a simple 9-period EMA from close prices
  const emaValues: number[] = [];
  const emaPeriod = 9;
  const multiplier = 2 / (emaPeriod + 1);
  candles.forEach((c, i) => {
    if (i === 0) { emaValues.push(c.close); }
    else { emaValues.push((c.close - emaValues[i - 1]) * multiplier + emaValues[i - 1]); }
  });

  const allPrices = candles.flatMap((c) => [c.high, c.low]);
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const range = maxP - minP || 1;
  const svgW = 520, svgH = 220, pad = 12;
  const plotW = svgW - pad * 2, plotH = svgH - pad * 2;
  const totalSlots = candles.length + rightOffset;
  const barW = Math.max(2, Math.min(16, plotW / totalSlots - 2));
  const gap = (plotW - barW * totalSlots) / (totalSlots + 1);
  const yPos = (p: number) => pad + plotH - ((p - minP) / range) * plotH;

  const isLine = style === 'Line';
  const isBar = style === 'OHLC Bars';
  const isHollow = style === 'Hollow Candle';

  return (
    <svg width="100%" height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} preserveAspectRatio="none">
      {gridLines && [0.25, 0.5, 0.75].map((frac) => (
        <line key={frac} x1={pad} y1={pad + plotH * frac} x2={svgW - pad} y2={pad + plotH * frac} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
      ))}
      {isLine ? (
        <polyline
          points={candles.map((c, i) => `${pad + gap + i * (barW + gap) + barW / 2},${yPos(c.close)}`).join(' ')}
          fill="none" stroke={t.up === 'transparent' ? (t.upBorder || '#fff') : t.up} strokeWidth="1.5"
        />
      ) : (
        candles.map((c, i) => {
          const x = pad + gap + i * (barW + gap);
          const cx = x + barW / 2;
          const fillColor = c.up ? t.up : t.down;
          const borderColor = c.up ? (t.upBorder || t.up) : t.down;
          const wickColor = c.up ? (t.upBorder || t.up) : t.down;
          if (isBar) {
            return (
              <g key={i}>
                <line x1={cx} y1={yPos(c.high)} x2={cx} y2={yPos(c.low)} stroke={wickColor} strokeWidth="1" />
                <line x1={cx - barW / 3} y1={yPos(c.open)} x2={cx} y2={yPos(c.open)} stroke={wickColor} strokeWidth="1.5" />
                <line x1={cx} y1={yPos(c.close)} x2={cx + barW / 3} y2={yPos(c.close)} stroke={wickColor} strokeWidth="1.5" />
              </g>
            );
          }
          const bodyTop = yPos(Math.max(c.open, c.close));
          const bodyBottom = yPos(Math.min(c.open, c.close));
          const bodyH = Math.max(1, bodyBottom - bodyTop);
          return (
            <g key={i}>
              <line x1={cx} y1={yPos(c.high)} x2={cx} y2={bodyTop} stroke={wickColor} strokeWidth="1" />
              <line x1={cx} y1={bodyBottom} x2={cx} y2={yPos(c.low)} stroke={wickColor} strokeWidth="1" />
              <rect x={x} y={bodyTop} width={barW} height={bodyH} fill={isHollow && c.up ? 'transparent' : fillColor} stroke={borderColor} strokeWidth="1" rx="0.5" />
            </g>
          );
        })
      )}
      {/* EMA overlay line */}
      <polyline
        points={candles.map((_, i) => {
          const cx = pad + gap + i * (barW + gap) + barW / 2;
          return `${cx},${yPos(emaValues[i])}`;
        }).join(' ')}
        fill="none"
        stroke="#22c55e"
        strokeWidth="1.2"
        opacity="0.7"
        strokeDasharray="none"
      />
      {/* EMA label */}
      {candles.length > 0 && (
        <text
          x={pad + gap + (candles.length - 1) * (barW + gap) + barW / 2 + 4}
          y={yPos(emaValues[emaValues.length - 1]) + 3}
          fill="#22c55e"
          fontSize="7"
          opacity="0.7"
        >
          EMA 9
        </text>
      )}

      {/* Position markers — rendered on the price chart */}
      {mCfg && (() => {
        // Place a mock entry at candle 4 and exit at candle 12 (or last-3)
        const entryIdx = Math.min(4, candles.length - 4);
        const exitIdx = Math.min(candles.length - 3, candles.length - 1);
        if (entryIdx < 0 || exitIdx < 0 || entryIdx >= candles.length || exitIdx >= candles.length) return null;

        const entryCx = pad + gap + entryIdx * (barW + gap) + barW / 2;
        const exitCx = pad + gap + exitIdx * (barW + gap) + barW / 2;
        const entryCandle = candles[entryIdx];
        const exitCandle = candles[exitIdx];
        const entryPrice = entryCandle.close;
        const exitPrice = exitCandle.close;
        const isWin = exitPrice > entryPrice;

        // Helper to render a shape at (cx, cy)
        function renderMark(cx: number, cy: number, shape: string, color: string, size: number = 5) {
          if (shape === 'none') return null;
          if (shape === 'cross') return <g><line x1={cx} y1={cy - size} x2={cx} y2={cy + size} stroke={color} strokeWidth="1.5" /><line x1={cx - size} y1={cy} x2={cx + size} y2={cy} stroke={color} strokeWidth="1.5" /></g>;
          if (shape === 'xcross') return <g><line x1={cx - size} y1={cy - size} x2={cx + size} y2={cy + size} stroke={color} strokeWidth="1.5" /><line x1={cx + size} y1={cy - size} x2={cx - size} y2={cy + size} stroke={color} strokeWidth="1.5" /></g>;
          if (shape === 'circle') return <circle cx={cx} cy={cy} r={size} fill="none" stroke={color} strokeWidth="1.5" />;
          if (shape === 'diamond') return <polygon points={`${cx},${cy - size} ${cx + size},${cy} ${cx},${cy + size} ${cx - size},${cy}`} fill={color} opacity="0.8" />;
          return null;
        }

        const exitColor = isWin ? mCfg.exitWinColor : mCfg.exitLossColor;

        return (
          <g>
            {/* Entry arrow */}
            <polygon points={`${entryCx - 4},${yPos(entryCandle.low) + 10} ${entryCx},${yPos(entryCandle.low) + 3} ${entryCx + 4},${yPos(entryCandle.low) + 10}`} fill={mCfg.entryColor} />
            {mCfg.showLabels && <text x={entryCx} y={yPos(entryCandle.low) + 18} fill={mCfg.entryColor} fontSize="7" textAnchor="middle">Entry</text>}

            {/* BT entry price mark */}
            {renderMark(entryCx, yPos(entryPrice), mCfg.btEntryMark, mCfg.entryColor, 4)}

            {/* Alert entry mark (offset slightly right) */}
            {mCfg.alertEntryMark !== 'none' && renderMark(entryCx + barW * 0.8, yPos(entryPrice - 0.3), mCfg.alertEntryMark, mCfg.entryColor + 'cc', 3.5)}

            {/* Exit arrow */}
            <polygon points={`${exitCx - 4},${yPos(exitCandle.high) - 10} ${exitCx},${yPos(exitCandle.high) - 3} ${exitCx + 4},${yPos(exitCandle.high) - 10}`} fill={exitColor} />
            {mCfg.showLabels && <text x={exitCx} y={yPos(exitCandle.high) - 14} fill={exitColor} fontSize="7" textAnchor="middle">{isWin ? '+1.2R' : '-0.5R'}</text>}

            {/* BT exit price mark */}
            {renderMark(exitCx, yPos(exitPrice), mCfg.btExitMark, exitColor, 4)}

            {/* Alert exit mark (offset slightly right) */}
            {mCfg.alertExitMark !== 'none' && renderMark(exitCx + barW * 0.8, yPos(exitPrice + 0.2), mCfg.alertExitMark, exitColor + 'cc', 3.5)}

            {/* Dashed line connecting entry to exit */}
            <line x1={entryCx} y1={yPos(entryPrice)} x2={exitCx} y2={yPos(entryPrice)} stroke={mCfg.entryColor} strokeWidth="0.5" strokeDasharray="3,3" opacity="0.4" />
          </g>
        );
      })()}

      {[0, 0.5, 1].map((frac) => (
        <text key={frac} x={svgW - pad + 2} y={pad + plotH * frac + 3} fill="var(--text-muted)" fontSize="8" textAnchor="start">{(minP + range * (1 - frac)).toFixed(1)}</text>
      ))}
    </svg>
  );
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

function CollapsibleCard({ title, defaultOpen = true, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      className="rounded-xl border"
      style={{ background: 'var(--bg-card)', borderColor: 'var(--border)', boxShadow: 'var(--card-shadow)', backdropFilter: 'var(--card-backdrop)', WebkitBackdropFilter: 'var(--card-backdrop)' }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 text-left"
      >
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{title}</h3>
        <span className="text-xs transition-transform" style={{ color: 'var(--text-muted)', transform: open ? 'rotate(0)' : 'rotate(-90deg)' }}>&#x25BE;</span>
      </button>
      {open && <div className="px-5 pb-5">{children}</div>}
    </div>
  );
}

export default function SettingsDisplayPage() {
  /* ---- API hooks (must be before any early return) ---- */
  const { data: savedSettings, isLoading } = useSettings();
  const saveSettingsMut = useSaveSettings();

  const [activeTab, setActiveTab] = useState<TabKey>('Price Charts');
  const [hydrated, setHydrated] = useState(false);

  // Regional
  const [timezone, setTimezone] = useState('US/Mountain');
  const [dateFormat, setDateFormat] = useState('YYYY-MM-DD');
  const [numberFormat, setNumberFormat] = useState<'comma' | 'dot' | 'space'>('comma');
  const [currencySymbol, setCurrencySymbol] = useState('$');

  // Chart
  const [candleStyle, setCandleStyle] = useState('Candle');
  const [candleTheme, setCandleTheme] = useState('neutral');
  const [defaultTf, setDefaultTf] = useState('1Min');
  const [visibleCandles, setVisibleCandles] = useState(20);
  const [gridLines, setGridLines] = useState(true);
  const [chartLibrary, setChartLibrary] = useState('lightweight');
  const [rightOffset, setRightOffset] = useState(3);
  const [paneOrder, setPaneOrder] = useState<ChartPane[]>(['confluence', 'price', 'oscillators']);

  // Equity Curves — segment colors
  const [eqBacktestColor, setEqBacktestColor] = useState('#2196F3');
  const [eqForwardColor, setEqForwardColor] = useState('#FF9800');
  const [eqLiveColor, setEqLiveColor] = useState('#4CAF50');
  // Equity Curves — style
  const [eqLineStyle, setEqLineStyle] = useState('solid');
  const [eqFillGradient, setEqFillGradient] = useState(true);
  const [eqLiveFill, setEqLiveFill] = useState(false);
  const [eqXAxis, setEqXAxis] = useState('time');
  // Equity Curves — overlays
  const [eqShowZeroLine, setEqShowZeroLine] = useState(true);
  const [eqShowHWM, setEqShowHWM] = useState(true);
  const [eqShowEdgeCheck, setEqShowEdgeCheck] = useState(false);
  const [eqShowConfBands, setEqShowConfBands] = useState(false);

  // Position markers
  const [entryShape, setEntryShape] = useState('arrowUp');
  const [exitWinShape, setExitWinShape] = useState('arrowDown');
  const [exitLossShape, setExitLossShape] = useState('arrowDown');
  const [btEntryMark, setBtEntryMark] = useState('cross');
  const [btExitMark, setBtExitMark] = useState('cross');
  const [alertEntryMark, setAlertEntryMark] = useState('xcross');
  const [alertExitMark, setAlertExitMark] = useState('xcross');
  const [entryColor, setEntryColor] = useState('#2196F3');
  const [exitWinColor, setExitWinColor] = useState('#4CAF50');
  const [exitLossColor, setExitLossColor] = useState('#f44336');
  const [exitStopColor, setExitStopColor] = useState('#f44336');
  const [exitBarCountColor, setExitBarCountColor] = useState('#26A69A');
  const [exitHybridColor, setExitHybridColor] = useState('#FF9800');
  const [showLabels, setShowLabels] = useState(true);
  const [alertSlippage, setAlertSlippage] = useState(5);
  const [dragIdx, setDragIdx] = useState<number | null>(null);

  // Table
  const [tableDensity, setTableDensity] = useState<'compact' | 'comfortable' | 'spacious'>('comfortable');

  // Styling — tag appearance
  const [badgeShape, setBadgeShape] = useState<BadgeShape>('rounded');
  const [bracketStyle, setBracketStyle] = useState<BracketStyle>('brackets');
  const [execColor, setExecColor] = useState('#2196F3');
  const [fidelityColor, setFidelityColor] = useState('#26C6DA');
  const [variationStyle, setVariationStyle] = useState<VariationStyleKey>('badge');

  /* ---- Hydrate local state from API settings (once) ---- */
  useEffect(() => {
    if (!savedSettings || hydrated) return;
    const s = savedSettings as Record<string, any>;
    if (s.timezone) setTimezone(s.timezone);
    if (s.dateFormat) setDateFormat(s.dateFormat);
    if (s.numberFormat) setNumberFormat(s.numberFormat);
    if (s.currencySymbol) setCurrencySymbol(s.currencySymbol);
    if (s.candleStyle) setCandleStyle(s.candleStyle);
    if (s.candleTheme) setCandleTheme(s.candleTheme);
    if (s.defaultTf) setDefaultTf(s.defaultTf);
    if (s.visibleCandles != null) setVisibleCandles(s.visibleCandles);
    if (s.gridLines != null) setGridLines(s.gridLines);
    if (s.chartLibrary) setChartLibrary(s.chartLibrary);
    if (s.rightOffset != null) setRightOffset(s.rightOffset);
    if (s.paneOrder) setPaneOrder(s.paneOrder);
    if (s.eqBacktestColor) setEqBacktestColor(s.eqBacktestColor);
    if (s.eqForwardColor) setEqForwardColor(s.eqForwardColor);
    if (s.eqLiveColor) setEqLiveColor(s.eqLiveColor);
    if (s.eqLineStyle) setEqLineStyle(s.eqLineStyle);
    if (s.eqFillGradient != null) setEqFillGradient(s.eqFillGradient);
    if (s.eqLiveFill != null) setEqLiveFill(s.eqLiveFill);
    if (s.eqXAxis) setEqXAxis(s.eqXAxis);
    if (s.eqShowZeroLine != null) setEqShowZeroLine(s.eqShowZeroLine);
    if (s.eqShowHWM != null) setEqShowHWM(s.eqShowHWM);
    if (s.eqShowEdgeCheck != null) setEqShowEdgeCheck(s.eqShowEdgeCheck);
    if (s.eqShowConfBands != null) setEqShowConfBands(s.eqShowConfBands);
    if (s.entryShape) setEntryShape(s.entryShape);
    if (s.exitWinShape) setExitWinShape(s.exitWinShape);
    if (s.exitLossShape) setExitLossShape(s.exitLossShape);
    if (s.btEntryMark) setBtEntryMark(s.btEntryMark);
    if (s.btExitMark) setBtExitMark(s.btExitMark);
    if (s.alertEntryMark) setAlertEntryMark(s.alertEntryMark);
    if (s.alertExitMark) setAlertExitMark(s.alertExitMark);
    if (s.entryColor) setEntryColor(s.entryColor);
    if (s.exitWinColor) setExitWinColor(s.exitWinColor);
    if (s.exitLossColor) setExitLossColor(s.exitLossColor);
    if (s.exitStopColor) setExitStopColor(s.exitStopColor);
    if (s.exitBarCountColor) setExitBarCountColor(s.exitBarCountColor);
    if (s.exitHybridColor) setExitHybridColor(s.exitHybridColor);
    if (s.showLabels != null) setShowLabels(s.showLabels);
    if (s.alertSlippage != null) setAlertSlippage(s.alertSlippage);
    if (s.tableDensity) setTableDensity(s.tableDensity);
    if (s.badgeShape) setBadgeShape(s.badgeShape);
    if (s.bracketStyle) setBracketStyle(s.bracketStyle);
    if (s.execColor) setExecColor(s.execColor);
    if (s.fidelityColor) setFidelityColor(s.fidelityColor);
    if (s.variationStyle) setVariationStyle(s.variationStyle);
    setHydrated(true);
  }, [savedSettings, hydrated]);

  /* ---- Collect all settings for save ---- */
  const collectSettings = useCallback(() => ({
    timezone, dateFormat, numberFormat, currencySymbol,
    candleStyle, candleTheme, defaultTf, visibleCandles, gridLines, chartLibrary, rightOffset, paneOrder,
    eqBacktestColor, eqForwardColor, eqLiveColor, eqLineStyle, eqFillGradient, eqLiveFill, eqXAxis,
    eqShowZeroLine, eqShowHWM, eqShowEdgeCheck, eqShowConfBands,
    entryShape, exitWinShape, exitLossShape, btEntryMark, btExitMark, alertEntryMark, alertExitMark,
    entryColor, exitWinColor, exitLossColor, exitStopColor, exitBarCountColor, exitHybridColor, showLabels,
    alertSlippage,
    tableDensity, badgeShape, bracketStyle, execColor, fidelityColor, variationStyle,
  }), [
    timezone, dateFormat, numberFormat, currencySymbol,
    candleStyle, candleTheme, defaultTf, visibleCandles, gridLines, chartLibrary, rightOffset, paneOrder,
    eqBacktestColor, eqForwardColor, eqLiveColor, eqLineStyle, eqFillGradient, eqLiveFill, eqXAxis,
    eqShowZeroLine, eqShowHWM, eqShowEdgeCheck, eqShowConfBands,
    entryShape, exitWinShape, exitLossShape, btEntryMark, btExitMark, alertEntryMark, alertExitMark,
    entryColor, exitWinColor, exitLossColor, exitStopColor, exitBarCountColor, exitHybridColor, showLabels,
    alertSlippage,
    tableDensity, badgeShape, bracketStyle, execColor, fidelityColor, variationStyle,
  ]);

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
    return int.replace(/\B(?=(\d{3})+(?!\d))/g, thousands) + decimal + dec;
  }

  function formatCurrency(n: number): string {
    const num = formatNumber(n);
    return currencySymbol === 'None' ? num : currencySymbol + num;
  }

  const now = new Date();
  const densityPadding = { compact: 'py-1', comfortable: 'py-2.5', spacious: 'py-4' };

  // Badge renderer — uses global exec/fidelity colors and shape settings
  function renderBadge(code: string, type: 'exec' | 'fidelity' = 'exec') {
    const color = type === 'exec' ? execColor : fidelityColor;
    const bgAlpha = '20'; // 12% opacity for background
    const bg = color + bgAlpha;
    const radius = badgeShape === 'rounded' ? 'rounded-full' : 'rounded';
    const text = bracketStyle === 'brackets' ? `[${code}]` : code;
    return (
      <span
        className={`text-xs font-mono px-2 py-0.5 ${radius} font-medium`}
        style={{ color, background: bg }}
      >
        {text}
      </span>
    );
  }

  return (
    <div>
      <PageHeader
        title="Display Preferences"
        subtitle="Configure how data is displayed throughout the app"
      />

      {/* Tab Navigation */}
      <div className="flex gap-1 border-b mb-6" style={{ borderColor: 'var(--border)' }}>
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className="px-5 py-2.5 text-sm transition-colors relative"
            style={{
              color: activeTab === tab ? 'var(--accent)' : 'var(--text-muted)',
              borderBottom: activeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
              marginBottom: '-1px',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ================================================================ */}
      {/* CHARTS TAB                                                        */}
      {/* ================================================================ */}
      {activeTab === 'Price Charts' && (
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-5">
            {/* Chart Library */}
            <CollapsibleCard title="Chart Library">
              <div className="space-y-3">
                {[
                  { id: 'lightweight', label: 'Lightweight Charts', desc: 'TradingView-style with zoom, pan, and crosshair. Fast and lightweight.' },
                  { id: 'echarts', label: 'ECharts', desc: 'Rich interactive charts with hover tooltips showing OHLCV values per candle.' },
                ].map((lib) => (
                  <button
                    key={lib.id}
                    onClick={() => setChartLibrary(lib.id)}
                    className="w-full text-left rounded-lg p-3 transition-all"
                    style={{
                      background: chartLibrary === lib.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                      border: `1px solid ${chartLibrary === lib.id ? 'var(--accent)' : 'var(--border)'}`,
                    }}
                  >
                    <span className="text-sm font-medium block" style={{ color: chartLibrary === lib.id ? 'var(--accent)' : 'var(--text-primary)' }}>{lib.label}</span>
                    <span className="text-xs block mt-0.5" style={{ color: 'var(--text-muted)' }}>{lib.desc}</span>
                  </button>
                ))}
              </div>
            </CollapsibleCard>

            {/* Candle Appearance */}
            <CollapsibleCard title="Candle Appearance">
              <div className="space-y-4">
                {/* Style — visual grid matching color selector */}
                <div>
                  <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Style</span>
                  <div className="grid grid-cols-5 gap-1.5">
                    {CANDLE_STYLES.map((s) => {
                      const isActive = candleStyle === s.id;
                      return (
                        <button key={s.id} onClick={() => setCandleStyle(s.id)} className="rounded-lg p-2 text-center" title={s.desc} style={{ background: isActive ? 'var(--accent-muted)' : 'var(--bg-input)', border: `1px solid ${isActive ? 'var(--accent)' : 'var(--border)'}` }}>
                          <div className="flex justify-center mb-1" style={{ height: 20 }}>
                            {s.icon === 'filled' && (
                              <svg width="14" height="20" viewBox="0 0 14 20"><line x1="7" y1="0" x2="7" y2="5" stroke="var(--text-muted)" strokeWidth="1" /><rect x="2" y="5" width="10" height="10" fill="var(--text-secondary)" rx="1" /><line x1="7" y1="15" x2="7" y2="20" stroke="var(--text-muted)" strokeWidth="1" /></svg>
                            )}
                            {s.icon === 'hollow' && (
                              <svg width="14" height="20" viewBox="0 0 14 20"><line x1="7" y1="0" x2="7" y2="5" stroke="var(--text-muted)" strokeWidth="1" /><rect x="2" y="5" width="10" height="10" fill="none" stroke="var(--text-secondary)" strokeWidth="1.5" rx="1" /><line x1="7" y1="15" x2="7" y2="20" stroke="var(--text-muted)" strokeWidth="1" /></svg>
                            )}
                            {s.icon === 'bars' && (
                              <svg width="14" height="20" viewBox="0 0 14 20"><line x1="7" y1="0" x2="7" y2="20" stroke="var(--text-secondary)" strokeWidth="1.5" /><line x1="2" y1="5" x2="7" y2="5" stroke="var(--text-secondary)" strokeWidth="1.5" /><line x1="7" y1="15" x2="12" y2="15" stroke="var(--text-secondary)" strokeWidth="1.5" /></svg>
                            )}
                            {s.icon === 'ha' && (
                              <svg width="14" height="20" viewBox="0 0 14 20"><line x1="7" y1="0" x2="7" y2="4" stroke="var(--text-muted)" strokeWidth="1" /><rect x="2" y="4" width="10" height="12" fill="var(--text-secondary)" rx="2" /><line x1="7" y1="16" x2="7" y2="20" stroke="var(--text-muted)" strokeWidth="1" /></svg>
                            )}
                            {s.icon === 'line' && (
                              <svg width="14" height="20" viewBox="0 0 14 20"><polyline points="0,15 4,8 8,12 14,3" fill="none" stroke="var(--text-secondary)" strokeWidth="1.5" /></svg>
                            )}
                          </div>
                          <span className="text-xs block" style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}>{s.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Colors — visual grid with theme option */}
                <div>
                  <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Colors</span>
                  <div className="grid grid-cols-6 gap-1.5">
                    {Object.keys(CANDLE_THEMES).map((key) => {
                      const theme = CANDLE_THEMES[key];
                      const isActive = candleTheme === key;
                      return (
                        <button key={key} onClick={() => setCandleTheme(key)} className="rounded-lg p-2 text-center" title={theme.desc} style={{ background: isActive ? 'var(--accent-muted)' : 'var(--bg-input)', border: `1px solid ${isActive ? 'var(--accent)' : 'var(--border)'}` }}>
                          <div className="flex justify-center gap-1 mb-1">
                            {theme.isTheme ? (
                              <>
                                <span className="w-2.5 h-5 rounded-sm" style={{ background: 'var(--accent)' }} />
                                <span className="w-2.5 h-5 rounded-sm" style={{ background: 'var(--red)' }} />
                              </>
                            ) : (
                              <>
                                <span className="w-2.5 h-5 rounded-sm" style={{ background: theme.up === 'transparent' ? 'none' : theme.up, border: theme.up === 'transparent' ? `1px solid ${theme.upBorder || '#fff'}` : 'none' }} />
                                <span className="w-2.5 h-5 rounded-sm" style={{ background: theme.down }} />
                              </>
                            )}
                          </div>
                          <span className="text-xs block" style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}>{theme.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            </CollapsibleCard>

            {/* Chart Pane Order */}
            <CollapsibleCard title="Chart Pane Order">
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>Drag to reorder how chart sections are stacked from top to bottom.</p>
              <div className="space-y-1.5">
                {paneOrder.map((paneId, idx) => {
                  const pane = CHART_PANES.find((p) => p.id === paneId)!;
                  return (
                    <div
                      key={paneId}
                      draggable
                      onDragStart={() => setDragIdx(idx)}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={() => {
                        if (dragIdx === null) return;
                        const next = [...paneOrder];
                        const [moved] = next.splice(dragIdx, 1);
                        next.splice(idx, 0, moved);
                        setPaneOrder(next);
                        setDragIdx(null);
                      }}
                      onDragEnd={() => setDragIdx(null)}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-grab active:cursor-grabbing transition-all"
                      style={{
                        background: dragIdx === idx ? 'var(--accent-muted)' : 'var(--bg-input)',
                        border: `1px solid ${dragIdx === idx ? 'var(--accent)' : 'var(--border)'}`,
                        opacity: dragIdx === idx ? 0.7 : 1,
                      }}
                    >
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>&#x2630;</span>
                      <span className="text-xs font-mono w-4 text-center" style={{ color: 'var(--text-muted)' }}>{idx + 1}</span>
                      <div className="flex-1">
                        <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{pane.label}</span>
                        <span className="text-xs block" style={{ color: 'var(--text-muted)' }}>{pane.desc}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CollapsibleCard>

            {/* Chart Behavior */}
            <CollapsibleCard title="Chart Defaults">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Default Timeframe</span>
                  <select value={defaultTf} onChange={(e) => setDefaultTf(e.target.value)} className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                    {['1Min', '5Min', '15Min', '1H', '4H', '1D'].map((o) => <option key={o}>{o}</option>)}
                  </select>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Visible Candles</span>
                  <div className="flex items-center gap-2">
                    <input type="range" min="10" max="30" value={visibleCandles} onChange={(e) => setVisibleCandles(Number(e.target.value))} className="w-28" />
                    <span className="text-xs font-mono w-6 text-center" style={{ color: 'var(--text-muted)' }}>{visibleCandles}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm block" style={{ color: 'var(--text-primary)' }}>Right Offset</span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Empty space after the last candle</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input type="range" min="0" max="15" value={rightOffset} onChange={(e) => setRightOffset(Number(e.target.value))} className="w-28" />
                    <span className="text-xs font-mono w-6 text-center" style={{ color: 'var(--text-muted)' }}>{rightOffset}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Grid Lines</span>
                  <button onClick={() => setGridLines(!gridLines)} className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors" style={{ background: gridLines ? 'var(--accent)' : 'var(--bg-input)', border: gridLines ? 'none' : '1px solid var(--border)' }}>
                    <div className="w-4 h-4 rounded-full absolute top-1 transition-all" style={{ background: gridLines ? 'white' : 'var(--text-muted)', left: gridLines ? '22px' : '4px' }} />
                  </button>
                </div>
              </div>
            </CollapsibleCard>

            {/* Position Markers */}
            <CollapsibleCard title="Position Markers" defaultOpen={false}>
              <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                Customize how trade entries, exits, and alert markers appear on price charts.
              </p>

              {/* Entry/Exit Arrow Shapes */}
              <div className="mb-4">
                <span className="text-xs font-medium block mb-2" style={{ color: 'var(--text-secondary)' }}>Arrow Markers</span>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Arrows shown above/below candles for trade entries and exits.</p>
                <div className="space-y-2">
                  {[
                    { label: 'Entry Arrow', value: entryShape, setter: setEntryShape, color: entryColor, setColor: setEntryColor, colorLabel: 'Entry' },
                    { label: 'Exit (Win)', value: exitWinShape, setter: setExitWinShape, color: exitWinColor, setColor: setExitWinColor, colorLabel: 'Win' },
                    { label: 'Exit (Loss)', value: exitLossShape, setter: setExitLossShape, color: exitLossColor, setColor: setExitLossColor, colorLabel: 'Loss' },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-3">
                      <span className="text-xs w-20" style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                      <select value={item.value} onChange={(e) => item.setter(e.target.value)} className="px-2 py-1 rounded text-xs flex-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                        <option value="arrowUp">Arrow Up</option>
                        <option value="arrowDown">Arrow Down</option>
                        <option value="circle">Circle</option>
                        <option value="square">Square</option>
                      </select>
                      <div className="flex items-center gap-1">
                        <input type="color" value={item.color} onChange={(e) => item.setColor(e.target.value)} className="w-6 h-6 rounded cursor-pointer" style={{ border: 'none', padding: 0 }} />
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.colorLabel}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Price-Level Marks (+ and x) */}
              <div className="mb-4">
                <span className="text-xs font-medium block mb-2" style={{ color: 'var(--text-secondary)' }}>Price-Level Marks</span>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Marks plotted at exact entry/exit price levels on the chart.</p>
                <div className="space-y-2">
                  {[
                    { label: 'BT Entry', desc: 'Backtest entry price', value: btEntryMark, setter: setBtEntryMark },
                    { label: 'BT Exit', desc: 'Backtest exit price', value: btExitMark, setter: setBtExitMark },
                    { label: 'Alert Entry', desc: 'Live alert entry price', value: alertEntryMark, setter: setAlertEntryMark },
                    { label: 'Alert Exit', desc: 'Live alert exit price', value: alertExitMark, setter: setAlertExitMark },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-3">
                      <div className="w-20">
                        <span className="text-xs block" style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                        <span className="text-xs block" style={{ color: 'var(--text-muted)', fontSize: 9 }}>{item.desc}</span>
                      </div>
                      <div className="flex gap-1 flex-1">
                        {[
                          { id: 'cross', label: '+', desc: 'Plus' },
                          { id: 'xcross', label: 'x', desc: 'X Mark' },
                          { id: 'circle', label: 'o', desc: 'Circle' },
                          { id: 'diamond', label: '◆', desc: 'Diamond' },
                          { id: 'none', label: '—', desc: 'Hidden' },
                        ].map((shape) => (
                          <button
                            key={shape.id}
                            onClick={() => item.setter(shape.id)}
                            className="flex-1 py-1 rounded text-xs font-mono"
                            title={shape.desc}
                            style={{
                              background: item.value === shape.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                              color: item.value === shape.id ? 'var(--accent)' : 'var(--text-muted)',
                              border: `1px solid ${item.value === shape.id ? 'var(--accent)' : 'var(--border)'}`,
                            }}
                          >
                            {shape.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Exit Reason Colors */}
              <div className="mb-4">
                <span className="text-xs font-medium block mb-2" style={{ color: 'var(--text-secondary)' }}>Exit Reason Colors</span>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Color coding for exit markers based on why the position closed.</p>
                <div className="space-y-2">
                  {[
                    { label: 'Signal / Target', color: exitWinColor, setter: setExitWinColor },
                    { label: 'Stop Loss', color: exitStopColor, setter: setExitStopColor },
                    { label: 'Bar Count Exit', color: exitBarCountColor, setter: setExitBarCountColor },
                    { label: 'Hybrid Unconfirmed', color: exitHybridColor, setter: setExitHybridColor },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-3">
                      <span className="text-xs flex-1" style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                      <div className="flex items-center gap-2">
                        <input type="color" value={item.color} onChange={(e) => item.setter(e.target.value)} className="w-6 h-6 rounded cursor-pointer" style={{ border: 'none', padding: 0 }} />
                        <span className="text-xs font-mono w-16" style={{ color: 'var(--text-muted)' }}>{item.color}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Labels toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-medium block" style={{ color: 'var(--text-secondary)' }}>Show Labels</span>
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Display R-multiple and Entry/Exit text on markers</span>
                </div>
                <button onClick={() => setShowLabels(!showLabels)} className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors" style={{ background: showLabels ? 'var(--accent)' : 'var(--bg-input)', border: showLabels ? 'none' : '1px solid var(--border)' }}>
                  <div className="w-4 h-4 rounded-full absolute top-1 transition-all" style={{ background: showLabels ? 'white' : 'var(--text-muted)', left: showLabels ? '22px' : '4px' }} />
                </button>
              </div>

              {/* Alert slippage tolerance */}
              <div>
                <span className="text-xs font-medium block mb-1" style={{ color: 'var(--text-secondary)' }}>Alert Slippage Tolerance</span>
                <span className="text-xs block mb-2" style={{ color: 'var(--text-muted)' }}>
                  Max acceptable seconds between algo and alert execution. Deltas within this threshold show green; beyond show red.
                </span>
                <div className="flex items-center gap-2">
                  <input
                    type="number" min={0} max={300} step={0.5}
                    value={alertSlippage}
                    onChange={e => setAlertSlippage(Number(e.target.value))}
                    className="w-20 px-2 py-1 rounded text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                  />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>seconds</span>
                </div>
              </div>
            </CollapsibleCard>
          </div>

          {/* Chart Preview — stacked panes in user's chosen order */}
          <div>
            <Card>
              <h3 className="text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Live Chart Preview</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {candleStyle} &middot; {CANDLE_THEMES[candleTheme].label} &middot; {visibleCandles} candles &middot; {rightOffset > 0 ? `${rightOffset} offset` : 'No offset'} &middot; {gridLines ? 'Grid on' : 'Grid off'} &middot; {chartLibrary === 'lightweight' ? 'Lightweight Charts' : 'ECharts'}
              </p>

              <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                {paneOrder.map((paneId) => (
                  <div key={paneId} style={{ borderBottom: '1px solid var(--border)' }}>
                    {/* Pane label */}
                    <div className="flex items-center gap-2 px-3 py-1.5" style={{ background: 'var(--bg-card)' }}>
                      <span className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>
                        {CHART_PANES.find((p) => p.id === paneId)!.label}
                      </span>
                    </div>

                    {/* Pane content */}
                    {paneId === 'price' && (
                      <div style={{ background: 'var(--bg-primary)' }}>
                        <CandlestickPreview
                          theme={candleTheme}
                          style={candleStyle}
                          gridLines={gridLines}
                          visibleCandles={visibleCandles}
                          rightOffset={rightOffset}
                          markers={{
                            entryColor, exitWinColor, exitLossColor, exitStopColor,
                            exitBarCountColor, exitHybridColor,
                            btEntryMark, btExitMark, alertEntryMark, alertExitMark,
                            showLabels,
                          }}
                        />
                      </div>
                    )}

                    {paneId === 'confluence' && (
                      <div className="px-3 py-2" style={{ background: 'var(--bg-primary)' }}>
                        {/* Mock confluence heatmap — stacked colored blocks */}
                        {['EMA_STACK', 'MACD_LINE', 'VWAP'].map((label, row) => (
                          <div key={label} className="flex items-center gap-1 mb-1">
                            <span className="text-xs font-mono w-20 text-right pr-2" style={{ color: 'var(--text-muted)', fontSize: 9 }}>{label}</span>
                            <div className="flex-1 flex gap-px">
                              {Array.from({ length: Math.min(visibleCandles, 30) }).map((_, i) => {
                                const seed = (row * 31 + i * 7) % 10;
                                const color = seed < 5 ? 'var(--green)' : seed < 8 ? 'var(--red)' : 'var(--orange)';
                                return <div key={i} className="flex-1" style={{ height: 12, background: color, opacity: 0.7, minWidth: 2 }} />;
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {paneId === 'oscillators' && (
                      <div className="px-3 py-2" style={{ background: 'var(--bg-primary)', height: 60 }}>
                        {/* Mock MACD histogram */}
                        <svg width="100%" height="50" viewBox="0 0 520 50" preserveAspectRatio="none">
                          {Array.from({ length: Math.min(visibleCandles, 30) }).map((_, i) => {
                            const val = Math.sin(i * 0.6) * 20 + Math.cos(i * 1.1) * 10;
                            const barWidth = Math.max(2, Math.min(14, 500 / Math.min(visibleCandles, 30) - 2));
                            const gap = (500 - barWidth * Math.min(visibleCandles, 30)) / (Math.min(visibleCandles, 30) + 1);
                            const x = 10 + gap + i * (barWidth + gap);
                            return (
                              <rect
                                key={i}
                                x={x}
                                y={val > 0 ? 25 - val : 25}
                                width={barWidth}
                                height={Math.abs(val)}
                                fill={val > 0 ? 'var(--green)' : 'var(--red)'}
                                opacity={0.6}
                                rx="0.5"
                              />
                            );
                          })}
                          <line x1="10" y1="25" x2="510" y2="25" stroke="var(--border)" strokeWidth="0.5" />
                        </svg>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                NVDA &middot; {defaultTf} &middot; Panes stacked in your chosen order. Drag panes on the left to rearrange.
              </p>
            </Card>

            {/* Marker Legend */}
            <Card className="mt-4">
              <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Chart Legend</h3>
              <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                {[
                  { label: 'EMA 9', color: '#22c55e', shape: '―' },
                  { label: 'Entry', color: entryColor, shape: '▲' },
                  { label: 'Exit (Win)', color: exitWinColor, shape: '▼' },
                  { label: 'Exit (Loss)', color: exitLossColor, shape: '▼' },
                  { label: 'BT Price', color: entryColor, shape: btEntryMark === 'cross' ? '+' : btEntryMark === 'xcross' ? '×' : btEntryMark === 'circle' ? '○' : btEntryMark === 'diamond' ? '◆' : '—' },
                  { label: 'Alert Price', color: entryColor, shape: alertEntryMark === 'xcross' ? '×' : alertEntryMark === 'cross' ? '+' : alertEntryMark === 'circle' ? '○' : alertEntryMark === 'diamond' ? '◆' : '—' },
                  { label: 'Stop', color: exitStopColor, shape: '■' },
                  { label: 'Bar Count', color: exitBarCountColor, shape: '■' },
                  { label: 'Hybrid', color: exitHybridColor, shape: '■' },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-1">
                    <span className="text-xs" style={{ color: item.color }}>{item.shape}</span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* EQUITY CURVES TAB                                                 */}
      {/* ================================================================ */}
      {activeTab === 'Equity Curves' && (
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-5">
            {/* Segment Colors */}
            <CollapsibleCard title="Segment Colors">
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                Equity curves show up to 3 segments: backtest data transitions into forward test, with live alert data overlaid on top.
              </p>
              <div className="space-y-2">
                {[
                  { label: 'Backtest', desc: 'Historical backtest results', color: eqBacktestColor, setter: setEqBacktestColor },
                  { label: 'Forward Test', desc: 'Live forward-testing period', color: eqForwardColor, setter: setEqForwardColor },
                  { label: 'Live Alerts', desc: 'Actual alert execution (shows slippage vs forward test)', color: eqLiveColor, setter: setEqLiveColor },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-3">
                    <div className="flex-1">
                      <span className="text-sm block" style={{ color: 'var(--text-primary)' }}>{item.label}</span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.desc}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="color" value={item.color} onChange={(e) => item.setter(e.target.value)} className="w-6 h-6 rounded cursor-pointer" style={{ border: 'none', padding: 0 }} />
                      <span className="text-xs font-mono w-16" style={{ color: 'var(--text-muted)' }}>{item.color}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CollapsibleCard>

            {/* Curve Style */}
            <CollapsibleCard title="Curve Style">
              <div className="space-y-3">
                <div>
                  <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Line Style</span>
                  <div className="flex gap-1.5">
                    {[{ id: 'solid', label: 'Solid' }, { id: 'smooth', label: 'Smooth' }, { id: 'stepped', label: 'Stepped' }].map((s) => (
                      <button key={s.id} onClick={() => setEqLineStyle(s.id)} className="flex-1 py-1.5 rounded-lg text-xs font-medium" style={{ background: eqLineStyle === s.id ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqLineStyle === s.id ? 'var(--accent)' : 'var(--text-muted)', border: `1px solid ${eqLineStyle === s.id ? 'var(--accent)' : 'var(--border)'}` }}>
                        {s.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Default X-Axis</span>
                  <div className="flex gap-1.5">
                    {[{ id: 'time', label: 'Time & Date' }, { id: 'trade', label: 'Trade Number' }].map((a) => (
                      <button key={a.id} onClick={() => setEqXAxis(a.id)} className="flex-1 py-1.5 rounded-lg text-xs font-medium" style={{ background: eqXAxis === a.id ? 'var(--accent-muted)' : 'var(--bg-input)', color: eqXAxis === a.id ? 'var(--accent)' : 'var(--text-muted)', border: `1px solid ${eqXAxis === a.id ? 'var(--accent)' : 'var(--border)'}` }}>
                        {a.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </CollapsibleCard>

            {/* Display Options */}
            <CollapsibleCard title="Display Options">
              <div className="space-y-3">
                {[
                  { label: 'Gradient Fill', desc: 'Gradient fill under backtest and forward test segments (not live alerts)', value: eqFillGradient, setter: setEqFillGradient },
                  { label: 'Zero Line', desc: 'Dashed horizontal line at starting value (0R or $0)', value: eqShowZeroLine, setter: setEqShowZeroLine },
                  { label: 'High Water Mark', desc: 'Dotted line tracking the equity peak', value: eqShowHWM, setter: setEqShowHWM },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between">
                    <div>
                      <span className="text-sm block" style={{ color: 'var(--text-primary)' }}>{item.label}</span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.desc}</span>
                    </div>
                    <button onClick={() => item.setter(!item.value)} className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors" style={{ background: item.value ? 'var(--accent)' : 'var(--bg-input)', border: item.value ? 'none' : '1px solid var(--border)' }}>
                      <div className="w-4 h-4 rounded-full absolute top-1 transition-all" style={{ background: item.value ? 'white' : 'var(--text-muted)', left: item.value ? '22px' : '4px' }} />
                    </button>
                  </div>
                ))}
              </div>
            </CollapsibleCard>

            {/* Advanced Overlays */}
            <CollapsibleCard title="Advanced Overlays" defaultOpen={false}>
              <div className="space-y-3">
                {[
                  { label: 'Edge Check', desc: '21-period MA + Bollinger Bands on equity curve — detects edge degradation', value: eqShowEdgeCheck, setter: setEqShowEdgeCheck },
                  { label: 'Confidence Bands', desc: '1SD/2SD bands from backtest distribution (portfolio context only)', value: eqShowConfBands, setter: setEqShowConfBands },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between">
                    <div>
                      <span className="text-sm block" style={{ color: 'var(--text-primary)' }}>{item.label}</span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.desc}</span>
                    </div>
                    <button onClick={() => item.setter(!item.value)} className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors" style={{ background: item.value ? 'var(--accent)' : 'var(--bg-input)', border: item.value ? 'none' : '1px solid var(--border)' }}>
                      <div className="w-4 h-4 rounded-full absolute top-1 transition-all" style={{ background: item.value ? 'white' : 'var(--text-muted)', left: item.value ? '22px' : '4px' }} />
                    </button>
                  </div>
                ))}
              </div>
            </CollapsibleCard>
          </div>

          {/* Equity Curve Preview */}
          <div>
            <Card>
              <h3 className="text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>Strategy Equity Curve Preview</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {eqLineStyle} &middot; {eqXAxis === 'time' ? 'Time axis' : 'Trade # axis'} &middot; {eqFillGradient ? 'Gradient' : 'No fill'} &middot; {eqShowHWM ? 'HWM on' : 'HWM off'} &middot; {eqShowEdgeCheck ? 'Edge Check on' : ''}
              </p>
              <div className="rounded-lg overflow-hidden p-2" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                <svg width="100%" height="220" viewBox="0 0 520 220" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="btGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={eqBacktestColor} stopOpacity="0.2" />
                      <stop offset="100%" stopColor={eqBacktestColor} stopOpacity="0" />
                    </linearGradient>
                    <linearGradient id="fwGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={eqForwardColor} stopOpacity="0.15" />
                      <stop offset="100%" stopColor={eqForwardColor} stopOpacity="0" />
                    </linearGradient>
                  </defs>

                  {/* Grid */}
                  {[0.25, 0.5, 0.75].map((f) => (
                    <line key={f} x1="20" y1={f * 190 + 10} x2="500" y2={f * 190 + 10} stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" />
                  ))}

                  {/* Zero line */}
                  {eqShowZeroLine && (
                    <line x1="20" y1="155" x2="500" y2="155" stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="6,4" opacity="0.5" />
                  )}

                  {(() => {
                    // Generate 3-segment equity curve
                    const allPts: { x: number; y: number; eq: number }[] = [];
                    let equity = 0;
                    for (let i = 0; i <= 30; i++) {
                      equity += (Math.sin(i * 0.45) * 0.7 + 0.25) * (1 + Math.cos(i * 0.25) * 0.4);
                      allPts.push({ x: 20 + (i / 30) * 480, y: 155 - equity * 3.5, eq: equity });
                    }

                    const boundaryIdx = 18; // ~60% through = forward test start
                    const btPts = allPts.slice(0, boundaryIdx + 1);
                    const fwPts = allPts.slice(boundaryIdx);
                    // Live alert line — slightly offset from forward test to show slippage
                    const livePts = fwPts.map((p) => ({ ...p, y: p.y + (Math.sin(p.x * 0.1) * 2 + 1.5) }));

                    const boundaryX = allPts[boundaryIdx].x;

                    const btLine = btPts.map((p) => `${p.x},${p.y}`).join(' ');
                    const fwLine = fwPts.map((p) => `${p.x},${p.y}`).join(' ');
                    const liveLine = livePts.map((p) => `${p.x},${p.y}`).join(' ');

                    const btFill = `${btPts[0].x},155 ${btLine} ${btPts[btPts.length - 1].x},155`;
                    const fwFill = `${fwPts[0].x},155 ${fwLine} ${fwPts[fwPts.length - 1].x},155`;

                    // High Water Mark
                    let hwmPeak = 0;
                    const hwmLine = allPts.map((p) => {
                      hwmPeak = Math.max(hwmPeak, p.eq);
                      return `${p.x},${155 - hwmPeak * 3.5}`;
                    }).join(' ');

                    // Edge Check: 21-MA approximation (simple moving average over equity)
                    const maLine = allPts.map((p, i) => {
                      const window = allPts.slice(Math.max(0, i - 6), i + 1);
                      const avg = window.reduce((s, w) => s + w.eq, 0) / window.length;
                      return `${p.x},${155 - avg * 3.5}`;
                    }).join(' ');

                    return (
                      <>
                        {/* Gradient fills (BT + FW, not live) */}
                        {eqFillGradient && <>
                          <polygon points={btFill} fill="url(#btGrad)" />
                          <polygon points={fwFill} fill="url(#fwGrad)" />
                        </>}

                        {/* Edge Check MA */}
                        {eqShowEdgeCheck && (
                          <polyline points={maLine} fill="none" stroke="#808000" strokeWidth="1" strokeDasharray="3,2" opacity="0.6" />
                        )}

                        {/* Confidence Bands (simplified) */}
                        {eqShowConfBands && <>
                          <polyline points={fwPts.map((p) => `${p.x},${p.y - 15}`).join(' ')} fill="none" stroke={eqBacktestColor} strokeWidth="0.5" strokeDasharray="2,2" opacity="0.3" />
                          <polyline points={fwPts.map((p) => `${p.x},${p.y + 15}`).join(' ')} fill="none" stroke={eqBacktestColor} strokeWidth="0.5" strokeDasharray="2,2" opacity="0.3" />
                          <rect x={boundaryX} y={fwPts[0].y - 15} width={500 - boundaryX} height="30" fill={eqBacktestColor} opacity="0.05" rx="2" />
                        </>}

                        {/* High Water Mark */}
                        {eqShowHWM && (
                          <polyline points={hwmLine} fill="none" stroke="var(--green)" strokeWidth="0.8" strokeDasharray="3,3" opacity="0.5" />
                        )}

                        {/* Backtest segment */}
                        <polyline points={btLine} fill="none" stroke={eqBacktestColor} strokeWidth="2" strokeLinejoin={eqLineStyle === 'smooth' ? 'round' : 'miter'} />

                        {/* Forward test segment */}
                        <polyline points={fwLine} fill="none" stroke={eqForwardColor} strokeWidth="2" strokeLinejoin={eqLineStyle === 'smooth' ? 'round' : 'miter'} />

                        {/* Live alert segment — no fill, overlaid */}
                        <polyline points={liveLine} fill="none" stroke={eqLiveColor} strokeWidth="2" strokeLinejoin={eqLineStyle === 'smooth' ? 'round' : 'miter'} />

                        {/* Forward test boundary line */}
                        <line x1={boundaryX} y1="10" x2={boundaryX} y2="205" stroke={eqForwardColor} strokeWidth="1" strokeDasharray="4,3" opacity="0.6" />
                        <text x={boundaryX + 3} y="18" fill={eqForwardColor} fontSize="7" opacity="0.7">FW Start</text>
                      </>
                    );
                  })()}

                  {/* Y-axis */}
                  <text x="505" y="14" fill="var(--text-muted)" fontSize="7">+12R</text>
                  <text x="505" y="158" fill="var(--text-muted)" fontSize="7">0R</text>
                  {/* X-axis label */}
                  <text x="260" y="215" fill="var(--text-muted)" fontSize="7" textAnchor="middle">{eqXAxis === 'time' ? 'Time' : 'Trade #'}</text>
                </svg>
              </div>

              {/* Legend */}
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3">
                {[
                  { label: 'Backtest', color: eqBacktestColor, style: '―' },
                  { label: 'Forward Test', color: eqForwardColor, style: '―' },
                  { label: 'Live Alerts', color: eqLiveColor, style: '―' },
                  ...(eqShowHWM ? [{ label: 'High Water Mark', color: 'var(--green)', style: '---' }] : []),
                  ...(eqShowEdgeCheck ? [{ label: 'Edge MA', color: '#808000', style: '---' }] : []),
                  ...(eqShowConfBands ? [{ label: '1SD Band', color: eqBacktestColor, style: '···' }] : []),
                  ...(eqShowZeroLine ? [{ label: 'Zero', color: 'var(--text-muted)', style: '---' }] : []),
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-1">
                    <span className="text-xs font-mono" style={{ color: item.color }}>{item.style}</span>
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                  </div>
                ))}
              </div>

              <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                The live alert line (green) overlays the forward test line to show execution slippage. In an ideal scenario, the green line sits directly on top of the orange line.
              </p>
            </Card>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* FORMATTING TAB                                                    */}
      {/* ================================================================ */}
      {activeTab === 'Formatting' && (
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-5">
            <CollapsibleCard title="Regional">
              <div className="space-y-3">
                {[
                  { label: 'Timezone', value: timezone, setter: setTimezone, options: ['US/Mountain', 'US/Eastern', 'US/Central', 'US/Pacific', 'UTC', 'Europe/London', 'Asia/Tokyo'] },
                  { label: 'Date Format', value: dateFormat, setter: setDateFormat, options: ['YYYY-MM-DD', 'MM/DD/YYYY', 'DD/MM/YYYY'] },
                  { label: 'Currency Symbol', value: currencySymbol, setter: setCurrencySymbol, options: ['$', 'USD ', 'None'] },
                ].map((field) => (
                  <div key={field.label} className="flex items-center justify-between">
                    <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{field.label}</span>
                    <select value={field.value} onChange={(e) => field.setter(e.target.value)} className="px-3 py-1.5 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                      {field.options.map((o) => <option key={o} value={o}>{o.trim() || 'None'}</option>)}
                    </select>
                  </div>
                ))}
                <div className="flex items-center justify-between">
                  <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Number Format</span>
                  <div className="flex gap-1">
                    {([{ id: 'comma' as const, label: '1,000.00' }, { id: 'dot' as const, label: '1.000,00' }, { id: 'space' as const, label: '1 000.00' }]).map((fmt) => (
                      <button key={fmt.id} onClick={() => setNumberFormat(fmt.id)} className="px-2 py-1 rounded text-xs font-mono" style={{ background: numberFormat === fmt.id ? 'var(--accent-muted)' : 'var(--bg-input)', color: numberFormat === fmt.id ? 'var(--accent)' : 'var(--text-muted)', border: `1px solid ${numberFormat === fmt.id ? 'var(--accent)' : 'var(--border)'}` }}>
                        {fmt.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </CollapsibleCard>
          </div>

          {/* Formatting Preview */}
          <div>
            <Card>
              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Live Preview</h3>

              <div className="rounded-lg p-3 mb-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Date & Time</p>
                <p className="text-sm font-medium">{formatDate(now)} 10:32:15 AM {timezone}</p>
              </div>

              <div className="rounded-lg p-3 mb-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Price & Currency</p>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Price', value: formatCurrency(487.50) },
                    { label: 'Volume', value: formatNumber(12450000) },
                    { label: 'P&L (Win)', value: '+' + formatCurrency(2847.32), color: 'var(--green)' },
                    { label: 'P&L (Loss)', value: '-' + formatCurrency(1203.45), color: 'var(--red)' },
                  ].map((item) => (
                    <div key={item.label}>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</p>
                      <p className="text-lg font-semibold" style={{ color: item.color }}>{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Large Numbers</p>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'Account Balance', value: formatCurrency(84230.00) },
                    { label: 'Market Cap', value: formatNumber(2145000000) },
                  ].map((item) => (
                    <div key={item.label}>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</p>
                      <p className="text-base font-semibold">{item.value}</p>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* COMPONENTS TAB                                                    */}
      {/* ================================================================ */}
      {activeTab === 'Components' && (
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-5">
            {/* Tag Appearance */}
            <CollapsibleCard title="Tag Appearance">
              <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                Controls how execution type and fidelity badges look throughout the app.
              </p>

              {/* Shape */}
              <div className="mb-4">
                <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Badge Shape</span>
                <div className="flex gap-1.5">
                  {([{ id: 'rounded' as BadgeShape, label: 'Rounded' }, { id: 'square' as BadgeShape, label: 'Square' }]).map((s) => (
                    <button key={s.id} onClick={() => setBadgeShape(s.id)} className="flex-1 py-2 rounded-lg text-xs font-medium" style={{ background: badgeShape === s.id ? 'var(--accent-muted)' : 'var(--bg-input)', color: badgeShape === s.id ? 'var(--accent)' : 'var(--text-muted)', border: `1px solid ${badgeShape === s.id ? 'var(--accent)' : 'var(--border)'}` }}>
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Brackets */}
              <div className="mb-4">
                <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Brackets</span>
                <div className="flex gap-1.5">
                  {([{ id: 'brackets' as BracketStyle, label: '[C]  [PB]', desc: 'With brackets' }, { id: 'none' as BracketStyle, label: 'C  PB', desc: 'Without brackets' }]).map((s) => (
                    <button key={s.id} onClick={() => setBracketStyle(s.id)} className="flex-1 py-2 rounded-lg text-xs font-mono" style={{ background: bracketStyle === s.id ? 'var(--accent-muted)' : 'var(--bg-input)', color: bracketStyle === s.id ? 'var(--accent)' : 'var(--text-muted)', border: `1px solid ${bracketStyle === s.id ? 'var(--accent)' : 'var(--border)'}` }}>
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Colors */}
              <div>
                <span className="text-sm block mb-2" style={{ color: 'var(--text-primary)' }}>Colors</span>
                <div className="space-y-2">
                  {[
                    { label: 'Execution Types', desc: '[C] [L] [LC] [CC]', color: execColor, setter: setExecColor },
                    { label: 'Fidelity Types', desc: '[PB] [CB]', color: fidelityColor, setter: setFidelityColor },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center gap-3">
                      <div className="flex-1">
                        <span className="text-sm block" style={{ color: 'var(--text-primary)' }}>{item.label}</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.desc}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <input type="color" value={item.color} onChange={(e) => item.setter(e.target.value)} className="w-6 h-6 rounded cursor-pointer" style={{ border: 'none', padding: 0 }} />
                        <span className="text-xs font-mono w-16" style={{ color: 'var(--text-muted)' }}>{item.color}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CollapsibleCard>

            {/* Variation Name Styling */}
            <CollapsibleCard title="Pack Version Display">
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                How confluence pack variation names (Default, Scalping, Custom) appear throughout the app.
              </p>
              <div className="space-y-2">
                {(Object.keys(VARIATION_STYLES) as VariationStyleKey[]).map((key) => (
                  <button
                    key={key}
                    onClick={() => setVariationStyle(key)}
                    className="w-full text-left rounded-lg p-3 transition-all"
                    style={{
                      background: variationStyle === key ? 'var(--accent-muted)' : 'var(--bg-input)',
                      border: `1px solid ${variationStyle === key ? 'var(--accent)' : 'var(--border)'}`,
                    }}
                  >
                    <span className="text-sm font-medium" style={{ color: variationStyle === key ? 'var(--accent)' : 'var(--text-primary)' }}>{VARIATION_STYLES[key].label}</span>
                    <span className="text-xs block mt-0.5" style={{ color: 'var(--text-muted)' }}>{VARIATION_STYLES[key].desc}</span>
                  </button>
                ))}
              </div>
            </CollapsibleCard>
          </div>

          {/* Components Preview */}
          <div>
            <Card>
              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Execution Types</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>How you enter a position — determines fill timing and confirmation.</p>
              <div className="space-y-2 mb-6">
                {EXEC_TYPES.map((et) => (
                  <div key={et.code} className="flex items-center gap-3 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    {renderBadge(et.code, 'exec')}
                    <div>
                      <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{et.label}</span>
                      <span className="text-xs block" style={{ color: 'var(--text-muted)' }}>{et.desc}</span>
                    </div>
                  </div>
                ))}
              </div>

              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Fidelity Types</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>When confluence conditions are evaluated — determines data reliability.</p>
              <div className="space-y-2 mb-6">
                {FIDELITY_TYPES.map((ft) => (
                  <div key={ft.code} className="flex items-center gap-3 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    {renderBadge(ft.code, 'fidelity')}
                    <div>
                      <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{ft.label}</span>
                      <span className="text-xs block" style={{ color: 'var(--text-muted)' }}>{ft.desc}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Combined example: how they appear together in context */}
              <h3 className="text-sm font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>In Context</h3>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>How execution and fidelity tags appear together on triggers and conditions.</p>
              <div className="space-y-2 mb-6">
                {[
                  { trigger: 'EMA Bull Cross', exec: 'C', fidelity: 'PB' },
                  { trigger: 'VWAP Cross Above', exec: 'L', fidelity: 'PB' },
                  { trigger: 'UT Bot Buy (Confirmed)', exec: 'LC', fidelity: 'PB' },
                  { trigger: 'EMA Stack > SML', exec: null, fidelity: 'PB' },
                  { trigger: 'MACD > Signal (forming)', exec: null, fidelity: 'CB' },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    {item.exec && renderBadge(item.exec, 'exec')}
                    {renderBadge(item.fidelity, 'fidelity')}
                    <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{item.trigger}</span>
                  </div>
                ))}
              </div>

              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Pack Names</h3>
              <div className="space-y-2">
                {[
                  { name: 'EMA Stack', version: 'Default' },
                  { name: 'EMA Stack', version: 'Scalping' },
                  { name: 'MACD Line', version: 'Default' },
                  { name: 'VWAP', version: 'Conservative' },
                ].map((pack, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                    {variationStyle === 'parenthetical' && (
                      <span className="text-sm">{pack.name} <span style={{ color: 'var(--text-muted)' }}>({pack.version})</span></span>
                    )}
                    {variationStyle === 'badge' && (
                      <>
                        <span className="text-sm font-medium">{pack.name}</span>
                        <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: pack.version === 'Default' ? 'var(--bg-card)' : 'var(--accent-muted)', color: pack.version === 'Default' ? 'var(--text-muted)' : 'var(--accent)' }}>{pack.version}</span>
                      </>
                    )}
                    {variationStyle === 'separator' && (
                      <span className="text-sm">{pack.name} <span style={{ color: 'var(--text-muted)' }}>&middot;</span> <span style={{ color: 'var(--accent)' }}>{pack.version}</span></span>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* TABLES TAB                                                        */}
      {/* ================================================================ */}
      {activeTab === 'Tables' && (
        <div className="grid grid-cols-2 gap-6">
          <div>
            <CollapsibleCard title="Table Density">
              <div className="space-y-2">
                {(['compact', 'comfortable', 'spacious'] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setTableDensity(d)}
                    className="w-full text-left rounded-lg p-3 capitalize transition-all"
                    style={{
                      background: tableDensity === d ? 'var(--accent-muted)' : 'var(--bg-input)',
                      border: `1px solid ${tableDensity === d ? 'var(--accent)' : 'var(--border)'}`,
                    }}
                  >
                    <span className="text-sm font-medium" style={{ color: tableDensity === d ? 'var(--accent)' : 'var(--text-primary)' }}>{d}</span>
                    <span className="text-xs block mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      {d === 'compact' && 'Maximum data density — fits more rows on screen'}
                      {d === 'comfortable' && 'Balanced spacing — easy to read, good density'}
                      {d === 'spacious' && 'Generous padding — easier on the eyes, fewer rows visible'}
                    </span>
                  </button>
                ))}
              </div>
            </CollapsibleCard>
          </div>

          {/* Table Preview */}
          <div>
            <Card>
              <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Trade History Preview</h3>
              <div className="rounded-lg overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                <div className="grid grid-cols-6 gap-2 text-xs font-medium px-3 py-2 border-b" style={{ color: 'var(--text-muted)', borderColor: 'var(--border)', background: 'var(--bg-input)' }}>
                  <span>#</span><span>Symbol</span><span>Entry</span><span>Price</span><span>P&L</span><span>Type</span>
                </div>
                {[
                  { id: 1, sym: 'NVDA', trigger: 'EMA Bull Cross', price: 487.50, pnl: '+1.2R', up: true, exec: 'C' },
                  { id: 2, sym: 'SPY', trigger: 'VWAP Cross', price: 562.10, pnl: '+0.8R', up: true, exec: 'L0' },
                  { id: 3, sym: 'TSLA', trigger: 'UT Bot Buy', price: 178.90, pnl: '-0.5R', up: false, exec: 'HM' },
                  { id: 4, sym: 'AAPL', trigger: 'EMA Bull Cross', price: 192.30, pnl: '+2.1R', up: true, exec: 'C' },
                  { id: 5, sym: 'NVDA', trigger: 'VWAP Cross', price: 489.20, pnl: '-0.3R', up: false, exec: 'HL' },
                  { id: 6, sym: 'META', trigger: 'UT Bot Buy', price: 512.80, pnl: '+0.6R', up: true, exec: 'L1' },
                ].map((row) => (
                  <div key={row.id} className={`grid grid-cols-6 gap-2 text-sm px-3 border-t items-center ${densityPadding[tableDensity]}`} style={{ borderColor: 'var(--border)' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{row.id}</span>
                    <span className="font-medium">{row.sym}</span>
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{row.trigger}</span>
                    <span>{formatCurrency(row.price)}</span>
                    <span style={{ color: row.up ? 'var(--green)' : 'var(--red)' }}>{row.pnl}</span>
                    <span>{renderBadge(row.exec, 'exec')}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
                Showing {tableDensity} density with {badgeShape} {bracketStyle === 'brackets' ? 'bracketed' : 'unbracketed'} badges.
                Table also reflects your formatting preferences from the Formatting tab.
              </p>
            </Card>
          </div>
        </div>
      )}

      {/* Save button — always visible */}
      <div className="mt-6">
        <button
          onClick={() => saveSettingsMut.mutate(collectSettings())}
          disabled={saveSettingsMut.isPending}
          className="px-6 py-2.5 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white', opacity: saveSettingsMut.isPending ? 0.6 : 1 }}
        >
          {saveSettingsMut.isPending ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
