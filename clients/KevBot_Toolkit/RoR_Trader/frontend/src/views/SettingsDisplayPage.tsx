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
import { useSettings, useSaveSettings } from '@/hooks/queries/useSettings';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEZONES = [
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'UTC',
  'Europe/London',
  'Europe/Berlin',
  'Asia/Tokyo',
  'Asia/Shanghai',
  'Australia/Sydney',
];

const DATE_FORMATS = [
  'MM/DD/YYYY',
  'DD/MM/YYYY',
  'YYYY-MM-DD',
  'MMM DD, YYYY',
];

const CANDLE_THEMES = [
  { id: 'classic', label: 'Classic', up: '#26a69a', down: '#ef5350' },
  { id: 'neutral', label: 'Neutral', up: '#FFFFFF', down: '#787B86' },
  { id: 'monochrome', label: 'Monochrome', up: '#d4d4d8', down: '#52525b' },
  { id: 'neon', label: 'Neon', up: '#00ff88', down: '#ff0055' },
  { id: 'theme', label: 'Theme', up: 'var(--accent)', down: 'var(--red)' },
];

const CHART_HEIGHTS = [
  { value: 300, label: 'Small' },
  { value: 450, label: 'Medium' },
  { value: 600, label: 'Large' },
  { value: 800, label: 'Extra Large' },
];

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
  padding: '8px 14px',
  borderRadius: '8px',
  fontSize: '0.875rem',
  width: '100%',
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
  const [chartHeight, setChartHeight] = useState(450);
  const [showGrid, setShowGrid] = useState(true);
  const [showVolume, setShowVolume] = useState(true);

  // Sync local state when settings load
  useEffect(() => {
    if (settings) {
      const display = settings.display || {};
      if (display.timezone) setTimezone(display.timezone);
      if (display.date_format) setDateFormat(display.date_format);
      if (display.candle_theme) setCandleTheme(display.candle_theme);
      if (display.chart_height) setChartHeight(display.chart_height);
      if (display.show_grid !== undefined) setShowGrid(display.show_grid);
      if (display.show_volume !== undefined) setShowVolume(display.show_volume);
    }
  }, [settings]);

  const handleSave = () => {
    saveMutation.mutate({
      ...settings,
      display: {
        ...(settings?.display || {}),
        timezone,
        date_format: dateFormat,
        candle_theme: candleTheme,
        chart_height: chartHeight,
        show_grid: showGrid,
        show_volume: showVolume,
      },
    });
  };

  const hasChanges = settings && (
    timezone !== (settings.display?.timezone || 'America/New_York') ||
    dateFormat !== (settings.display?.date_format || 'MM/DD/YYYY') ||
    candleTheme !== (settings.display?.candle_theme || 'classic') ||
    chartHeight !== (settings.display?.chart_height || 450) ||
    showGrid !== (settings.display?.show_grid ?? true) ||
    showVolume !== (settings.display?.show_volume ?? true)
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
        title="Display Settings"
        subtitle="Customize how data is displayed across the application"
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

      <div className="space-y-4 mt-4">
        {/* Formatting section */}
        <Card>
          <h3 className="text-sm font-semibold mb-4">Formatting</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
                Timezone
              </label>
              <select value={timezone} onChange={(e) => setTimezone(e.target.value)} style={inputStyle}>
                {TIMEZONES.map((tz) => <option key={tz} value={tz}>{tz}</option>)}
              </select>
            </div>
            <div>
              <label className="text-xs font-medium block mb-1.5" style={{ color: 'var(--text-muted)' }}>
                Date Format
              </label>
              <select value={dateFormat} onChange={(e) => setDateFormat(e.target.value)} style={inputStyle}>
                {DATE_FORMATS.map((fmt) => <option key={fmt} value={fmt}>{fmt}</option>)}
              </select>
            </div>
          </div>
        </Card>

        {/* Chart section */}
        <Card>
          <h3 className="text-sm font-semibold mb-4">Chart Preferences</h3>

          {/* Candle theme */}
          <div className="mb-4">
            <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>
              Candle Theme
            </label>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {CANDLE_THEMES.map((theme) => (
                <button
                  key={theme.id}
                  onClick={() => setCandleTheme(theme.id)}
                  className="flex flex-col items-center p-3 rounded-lg"
                  style={{
                    background: candleTheme === theme.id ? 'var(--accent-muted)' : 'var(--bg-input)',
                    border: candleTheme === theme.id ? '1px solid var(--accent)' : '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  <div className="flex gap-1 mb-1.5">
                    <div style={{ width: 8, height: 20, background: theme.up, borderRadius: 2 }} />
                    <div style={{ width: 8, height: 14, background: theme.down, borderRadius: 2 }} />
                    <div style={{ width: 8, height: 18, background: theme.up, borderRadius: 2 }} />
                  </div>
                  <span className="text-xs" style={{ color: candleTheme === theme.id ? 'var(--accent)' : 'var(--text-muted)' }}>
                    {theme.label}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Chart height */}
          <div className="mb-4">
            <label className="text-xs font-medium block mb-2" style={{ color: 'var(--text-muted)' }}>
              Default Chart Height
            </label>
            <div className="flex gap-2">
              {CHART_HEIGHTS.map((h) => (
                <button
                  key={h.value}
                  onClick={() => setChartHeight(h.value)}
                  className="px-3 py-1.5 rounded text-xs font-medium"
                  style={{
                    background: chartHeight === h.value ? 'var(--accent-muted)' : 'var(--bg-input)',
                    color: chartHeight === h.value ? 'var(--accent)' : 'var(--text-muted)',
                    border: chartHeight === h.value ? '1px solid var(--accent)' : '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                >
                  {h.label} ({h.value}px)
                </button>
              ))}
            </div>
          </div>

          {/* Toggles */}
          <div className="flex gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showGrid}
                onChange={(e) => setShowGrid(e.target.checked)}
                className="w-4 h-4 rounded"
                style={{ accentColor: 'var(--accent)' }}
              />
              <span className="text-sm">Show Grid Lines</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showVolume}
                onChange={(e) => setShowVolume(e.target.checked)}
                className="w-4 h-4 rounded"
                style={{ accentColor: 'var(--accent)' }}
              />
              <span className="text-sm">Show Volume</span>
            </label>
          </div>
        </Card>

        {/* Save status */}
        {saveMutation.isSuccess && (
          <p className="text-sm" style={{ color: 'var(--green)' }}>Settings saved successfully.</p>
        )}
        {saveMutation.isError && (
          <p className="text-sm" style={{ color: 'var(--red)' }}>Failed to save settings. Please try again.</p>
        )}
      </div>
    </div>
  );
}
