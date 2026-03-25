'use client';

/**
 * Settings Display — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around the settings API endpoint. No mock data.
 */

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import { useSettings, useSaveSettings } from '@/hooks/queries/useSettings';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEZONES = [
  'America/New_York', 'America/Chicago', 'America/Denver',
  'America/Los_Angeles', 'UTC', 'Europe/London', 'Europe/Berlin',
  'Asia/Tokyo', 'Asia/Shanghai', 'Australia/Sydney',
];

const DATE_FORMATS = ['MM/DD/YYYY', 'DD/MM/YYYY', 'YYYY-MM-DD', 'MMM DD, YYYY'];

const CANDLE_THEMES = [
  { id: 'theme', label: 'Theme', up: 'var(--accent)', down: 'var(--red)', desc: 'Matches your current app theme' },
  { id: 'classic', label: 'Classic', up: '#26a69a', down: '#ef5350', desc: 'Traditional green/red' },
  { id: 'neutral', label: 'Neutral', up: '#FFFFFF', down: '#787B86', desc: 'White up, gray down' },
  { id: 'monochrome', label: 'Monochrome', up: '#d4d4d8', down: '#52525b', desc: 'Subtle gray tones' },
  { id: 'neon', label: 'Neon', up: '#00ff88', down: '#ff0055', desc: 'High contrast for dark themes' },
];

const CANDLE_STYLES = [
  { id: 'Candle', label: 'Candle', desc: 'Standard filled candlesticks' },
  { id: 'Hollow Candle', label: 'Hollow', desc: 'Hollow up candles, filled down' },
  { id: 'OHLC Bars', label: 'OHLC', desc: 'Open-high-low-close bar chart' },
  { id: 'Heikin Ashi', label: 'Heikin Ashi', desc: 'Smoothed candles for trend clarity' },
  { id: 'Line', label: 'Line', desc: 'Simple close-price line chart' },
];

const CHART_LIBRARIES = [
  { id: 'lightweight', label: 'Lightweight Charts', desc: 'TradingView-style with zoom, pan, and crosshair' },
  { id: 'echarts', label: 'ECharts', desc: 'Rich interactive charts with hover tooltips showing OHLCV' },
];

const CHART_HEIGHTS = [
  { value: 300, label: 'Small' },
  { value: 450, label: 'Medium' },
  { value: 600, label: 'Large' },
  { value: 800, label: 'Extra Large' },
];

const TABLE_DENSITIES = [
  { id: 'compact', label: 'Compact' },
  { id: 'comfortable', label: 'Comfortable' },
  { id: 'spacious', label: 'Spacious' },
];

const TABS = ['Price Charts', 'Equity Curves', 'Formatting', 'Components', 'Tables'] as const;

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)', border: '1px solid var(--border)',
  color: 'var(--text-primary)', padding: '8px 14px', borderRadius: '8px',
  fontSize: '0.875rem', width: '100%',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SettingsDisplayPage() {
  const { data: settings, isLoading, error } = useSettings();
  const saveMutation = useSaveSettings();

  // Local state — initialized from API
  const [timezone, setTimezone] = useState('America/New_York');
  const [dateFormat, setDateFormat] = useState('MM/DD/YYYY');
  const [candleTheme, setCandleTheme] = useState('classic');
  const [candleStyle, setCandleStyle] = useState('Candle');
  const [chartHeight, setChartHeight] = useState(450);
  const [chartLibrary, setChartLibrary] = useState('lightweight');
  const [showGrid, setShowGrid] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [tableDensity, setTableDensity] = useState('comfortable');
  // Equity curve settings
  const [eqShowHWM, setEqShowHWM] = useState(true);
  const [eqShowConfBands, setEqShowConfBands] = useState(false);
  const [eqXAxis, setEqXAxis] = useState('time');

  // Sync local state when settings load
  useEffect(() => {
    if (settings) {
      const display = settings.display || {};
      if (display.timezone) setTimezone(display.timezone);
      if (display.date_format) setDateFormat(display.date_format);
      if (display.candle_theme) setCandleTheme(display.candle_theme);
      if (display.candle_style) setCandleStyle(display.candle_style);
      if (display.chart_height) setChartHeight(display.chart_height);
      if (display.chart_library) setChartLibrary(display.chart_library);
      if (display.show_grid !== undefined) setShowGrid(display.show_grid);
      if (display.show_volume !== undefined) setShowVolume(display.show_volume);
      if (display.table_density) setTableDensity(display.table_density);
      if (display.eq_show_hwm !== undefined) setEqShowHWM(display.eq_show_hwm);
      if (display.eq_show_conf_bands !== undefined) setEqShowConfBands(display.eq_show_conf_bands);
      if (display.eq_x_axis) setEqXAxis(display.eq_x_axis);
    }
  }, [settings]);

  const handleSave = () => {
    saveMutation.mutate({
      ...settings,
      display: {
        ...(settings?.display || {}),
        timezone, date_format: dateFormat, candle_theme: candleTheme,
        candle_style: candleStyle, chart_height: chartHeight,
        chart_library: chartLibrary, show_grid: showGrid,
        show_volume: showVolume, table_density: tableDensity,
        eq_show_hwm: eqShowHWM, eq_show_conf_bands: eqShowConfBands,
        eq_x_axis: eqXAxis,
      },
    });
  };

  const hasChanges = settings && (
    timezone !== (settings.display?.timezone || 'America/New_York') ||
    dateFormat !== (settings.display?.date_format || 'MM/DD/YYYY') ||
    candleTheme !== (settings.display?.candle_theme || 'classic') ||
    candleStyle !== (settings.display?.candle_style || 'Candle') ||
    chartHeight !== (settings.display?.chart_height || 450) ||
    chartLibrary !== (settings.display?.chart_library || 'lightweight') ||
    showGrid !== (settings.display?.show_grid ?? true) ||
    showVolume !== (settings.display?.show_volume ?? true) ||
    tableDensity !== (settings.display?.table_density || 'comfortable') ||
    eqShowHWM !== (settings.display?.eq_show_hwm ?? true) ||
    eqShowConfBands !== (settings.display?.eq_show_conf_bands ?? false) ||
    eqXAxis !== (settings.display?.eq_x_axis || 'time')
  );

  // ---------------------------------------------------------------------------
  // Loading / Error
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Display Settings" subtitle="Loading..." />
        <div className="space-y-4 mt-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <div className="animate-pulse space-y-3">
                <div className="h-4 rounded w-1/4" style={{ background: 'var(--border)' }} />
                <div className="h-10 rounded" style={{ background: 'var(--border)' }} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="Display Settings" subtitle="Error" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load settings. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Display Preferences"
        subtitle="Configure how data is displayed throughout the app"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{
                background: hasChanges ? 'var(--accent)' : 'var(--bg-input)',
                color: hasChanges ? '#fff' : 'var(--text-muted)',
                border: 'none',
                cursor: hasChanges ? 'pointer' : 'not-allowed',
              }}
              disabled={!hasChanges || saveMutation.isPending}
              onClick={handleSave}
            >
              {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        }
      />

      {/* V5 5-tab layout */}
      <TabBar tabs={[...TABS]}>
        {(tab) => (
          <div>
            {/* ============================================================ */}
            {/* PRICE CHARTS TAB                                              */}
            {/* ============================================================ */}
            {tab === 'Price Charts' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-5">
                  {/* Chart Library Selector */}
                  <Card>
                    <h3 className="text-sm font-semibold mb-3">Chart Library</h3>
                    <div className="space-y-2">
                      {CHART_LIBRARIES.map((lib) => (
                        <button
                          key={lib.id}
                          onClick={() => setChartLibrary(lib.id)}
                          className="w-full text-left rounded-lg p-3 transition-all"
                          style={{
                            background: chartLibrary === lib.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                            border: `1px solid ${chartLibrary === lib.id ? 'var(--accent)' : 'var(--border)'}`,
                            cursor: 'pointer',
                          }}
                        >
                          <span className="text-sm font-medium block" style={{ color: chartLibrary === lib.id ? 'var(--accent)' : 'var(--text-primary)' }}>{lib.label}</span>
                          <span className="text-xs block mt-0.5" style={{ color: 'var(--text-muted)' }}>{lib.desc}</span>
                        </button>
                      ))}
                    </div>
                  </Card>

                  {/* Candle Style — V5 visual grid */}
                  <Card>
                    <h3 className="text-sm font-semibold mb-3">Candle Style</h3>
                    <div className="grid grid-cols-5 gap-1.5">
                      {CANDLE_STYLES.map((s) => {
                        const isActive = candleStyle === s.id;
                        return (
                          <button
                            key={s.id}
                            onClick={() => setCandleStyle(s.id)}
                            className="rounded-lg p-2.5 text-center"
                            title={s.desc}
                            style={{
                              background: isActive ? 'var(--accent-muted)' : 'var(--bg-input)',
                              border: `1px solid ${isActive ? 'var(--accent)' : 'var(--border)'}`,
                              cursor: 'pointer',
                            }}
                          >
                            <span className="text-xs block" style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}>{s.label}</span>
                          </button>
                        );
                      })}
                    </div>
                  </Card>

                  {/* Candle Theme */}
                  <Card>
                    <h3 className="text-sm font-semibold mb-3">Candle Colors</h3>
                    <div className="grid grid-cols-5 gap-1.5">
                      {CANDLE_THEMES.map((theme) => {
                        const isActive = candleTheme === theme.id;
                        return (
                          <button
                            key={theme.id}
                            onClick={() => setCandleTheme(theme.id)}
                            className="flex flex-col items-center p-3 rounded-lg"
                            title={theme.desc}
                            style={{
                              background: isActive ? 'var(--accent-muted)' : 'var(--bg-input)',
                              border: `1px solid ${isActive ? 'var(--accent)' : 'var(--border)'}`,
                              cursor: 'pointer',
                            }}
                          >
                            <div className="flex gap-1 mb-1.5">
                              <div style={{ width: 8, height: 20, background: theme.up, borderRadius: 2 }} />
                              <div style={{ width: 8, height: 14, background: theme.down, borderRadius: 2 }} />
                              <div style={{ width: 8, height: 18, background: theme.up, borderRadius: 2 }} />
                            </div>
                            <span className="text-xs" style={{ color: isActive ? 'var(--accent)' : 'var(--text-muted)' }}>
                              {theme.label}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </Card>

                  {/* Chart Height + Toggles */}
                  <Card>
                    <h3 className="text-sm font-semibold mb-3">Chart Options</h3>
                    <div className="mb-3">
                      <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>Default Height</label>
                      <div className="flex gap-2">
                        {CHART_HEIGHTS.map((h) => (
                          <button
                            key={h.value}
                            onClick={() => setChartHeight(h.value)}
                            className="px-3 py-1.5 rounded text-xs font-medium"
                            style={{
                              background: chartHeight === h.value ? 'var(--accent-muted)' : 'var(--bg-input)',
                              color: chartHeight === h.value ? 'var(--accent)' : 'var(--text-muted)',
                              border: `1px solid ${chartHeight === h.value ? 'var(--accent)' : 'var(--border)'}`,
                              cursor: 'pointer',
                            }}
                          >
                            {h.label} ({h.value}px)
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-6">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} className="w-4 h-4 rounded" style={{ accentColor: 'var(--accent)' }} />
                        <span className="text-sm">Grid Lines</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input type="checkbox" checked={showVolume} onChange={(e) => setShowVolume(e.target.checked)} className="w-4 h-4 rounded" style={{ accentColor: 'var(--accent)' }} />
                        <span className="text-sm">Volume</span>
                      </label>
                    </div>
                  </Card>
                </div>

                {/* Right-side preview panel */}
                <div>
                  <Card>
                    <h3 className="text-sm font-semibold mb-3">Preview</h3>
                    <div
                      className="rounded-lg flex items-center justify-center"
                      style={{ background: 'var(--bg-input)', height: 220 }}
                    >
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Chart preview ({candleStyle} / {CANDLE_THEMES.find((t) => t.id === candleTheme)?.label || candleTheme})
                      </span>
                    </div>
                    <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                      Library: {CHART_LIBRARIES.find((l) => l.id === chartLibrary)?.label || chartLibrary} | Height: {chartHeight}px | Grid: {showGrid ? 'On' : 'Off'} | Volume: {showVolume ? 'On' : 'Off'}
                    </p>
                  </Card>
                </div>
              </div>
            )}

            {/* ============================================================ */}
            {/* EQUITY CURVES TAB                                             */}
            {/* ============================================================ */}
            {tab === 'Equity Curves' && (
              <Card>
                <h3 className="text-sm font-semibold mb-4">Equity Curve Settings</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>X-Axis</label>
                    <div className="flex gap-2">
                      {['time', 'trade'].map((mode) => (
                        <button
                          key={mode}
                          onClick={() => setEqXAxis(mode)}
                          className="px-3 py-1.5 rounded text-xs font-medium"
                          style={{
                            background: eqXAxis === mode ? 'var(--accent-muted)' : 'var(--bg-input)',
                            color: eqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)',
                            border: `1px solid ${eqXAxis === mode ? 'var(--accent)' : 'var(--border)'}`,
                            cursor: 'pointer',
                          }}
                        >
                          {mode === 'time' ? 'Time' : 'Trade #'}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-col gap-3">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={eqShowHWM} onChange={(e) => setEqShowHWM(e.target.checked)} className="w-4 h-4 rounded" style={{ accentColor: 'var(--accent)' }} />
                      <span className="text-sm">High Water Mark</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" checked={eqShowConfBands} onChange={(e) => setEqShowConfBands(e.target.checked)} className="w-4 h-4 rounded" style={{ accentColor: 'var(--accent)' }} />
                      <span className="text-sm">Confidence Bands</span>
                    </label>
                  </div>
                </div>
              </Card>
            )}

            {/* ============================================================ */}
            {/* FORMATTING TAB                                                */}
            {/* ============================================================ */}
            {tab === 'Formatting' && (
              <Card>
                <h3 className="text-sm font-semibold mb-4">Formatting</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Timezone</label>
                    <select value={timezone} onChange={(e) => setTimezone(e.target.value)} style={inputStyle}>
                      {TIMEZONES.map((tz) => <option key={tz} value={tz}>{tz}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>Date Format</label>
                    <select value={dateFormat} onChange={(e) => setDateFormat(e.target.value)} style={inputStyle}>
                      {DATE_FORMATS.map((fmt) => <option key={fmt} value={fmt}>{fmt}</option>)}
                    </select>
                  </div>
                </div>
              </Card>
            )}

            {/* ============================================================ */}
            {/* COMPONENTS TAB                                                */}
            {/* ============================================================ */}
            {tab === 'Components' && (
              <Card>
                <h3 className="text-sm font-semibold mb-4">Component Preferences</h3>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Configure badge shapes, execution type colors, fidelity badge colors, and variation display styles.
                  These settings will be available once the component library is connected.
                </p>
              </Card>
            )}

            {/* ============================================================ */}
            {/* TABLES TAB                                                    */}
            {/* ============================================================ */}
            {tab === 'Tables' && (
              <Card>
                <h3 className="text-sm font-semibold mb-4">Table Density</h3>
                <div className="flex gap-2">
                  {TABLE_DENSITIES.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => setTableDensity(d.id)}
                      className="px-4 py-2 rounded-lg text-sm font-medium"
                      style={{
                        background: tableDensity === d.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: tableDensity === d.id ? 'var(--accent)' : 'var(--text-muted)',
                        border: `1px solid ${tableDensity === d.id ? 'var(--accent)' : 'var(--border)'}`,
                        cursor: 'pointer',
                      }}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}
      </TabBar>

      {/* Save status */}
      {saveMutation.isSuccess && (
        <p className="text-sm mt-4" style={{ color: 'var(--green)' }}>Settings saved successfully.</p>
      )}
      {saveMutation.isError && (
        <p className="text-sm mt-4" style={{ color: 'var(--red)' }}>Failed to save settings. Please try again.</p>
      )}
    </div>
  );
}
