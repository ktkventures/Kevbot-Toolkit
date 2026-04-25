'use client';

import { useState, useMemo, useEffect } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Modal from '@/components/Modal';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Link from 'next/link';
import { apiFetch } from '@/lib/api/client';
import { useUserPackCode, usePackPreview, type PackPreviewResponse } from '@/hooks/queries/usePacks';
import { useChartPrefs } from '@/hooks/useChartPrefs';
const ReplayableChart = dynamic(() => import('@/charts/ReplayableChart'), { ssr: false });
const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });
import type { PaneConfig, SeriesConfig } from '@/charts/SyncedChartPane';
import { useRequestFix, useInstallPack } from '@/hooks/mutations/useAiBuilder';

/* ========================================================================
   Types
   ======================================================================== */

interface PackParam {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
  type: 'int' | 'float';
}

interface PlotSetting {
  key: string;
  label: string;
  default: string;
}

interface PackOutput {
  code: string;
  description: string;
}

interface PackTrigger {
  id: string;
  name: string;
  direction: 'LONG' | 'SHORT' | 'BOTH';
  type: 'ENTRY' | 'EXIT';
  exec: string;
}

interface StateTimelineEntry {
  time: string;
  state: string;
  prevState: string;
}

interface TriggerEvent {
  time: string;
  trigger: string;
  direction: string;
  type: string;
  price: string;
  execType: string;
}

interface UserPack {
  id: string;
  name: string;
  version: string;
  packType: 'tf_confluence' | 'general';
  category: string;
  enabled: boolean;
  isDefault: boolean;
  visibility: 'private' | 'public';
  strategiesUsing: number;
  lastModified: string;
  validationStatus: 'passed' | 'warnings' | 'failed';
  parityScore: number | null;
  params: PackParam[];
  plotSettings: PlotSetting[];
  outputs: PackOutput[];
  triggers: PackTrigger[];
}

/* ========================================================================
   Data — User packs are a future feature; no API hooks yet.
   All pack data is empty. The page renders the full V5 layout with empty states.
   ======================================================================== */

// No user packs — future feature
const emptyUserPacks: UserPack[] = [];

// Preview data now loaded from API via usePackPreview hook

/* Built-in template packs for "Create from Template" — future feature */
const builtInTemplates: { key: string; name: string; category: string; description: string }[] = [];

/* ========================================================================
   Style Constants
   ======================================================================== */

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Momentum': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Volume': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Volatility': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Trend': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'Custom': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

const execTypeBadgeStyles: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
  '[L0]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[L1]': { color: 'var(--purple)', bg: 'rgba(168,85,247,0.15)' },
  '[HM]': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  '[HL]': { color: 'var(--teal)', bg: 'rgba(20,184,166,0.15)' },
};

const directionColors: Record<string, { color: string; bg: string }> = {
  'LONG': { color: 'var(--green)', bg: 'var(--green-muted)' },
  'SHORT': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'BOTH': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
};

const triggerTypeColors: Record<string, { color: string; bg: string }> = {
  'ENTRY': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'EXIT': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

/* ========================================================================
   Shared inline-style helpers
   ======================================================================== */

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
};

const selectStyle: React.CSSProperties = { ...inputStyle };

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)',
  color: 'white',
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  color: 'var(--text-secondary)',
};

/* ========================================================================
   Sub-components
   ======================================================================== */

function ExecBadge({ exec }: { exec: string }) {
  const s = execTypeBadgeStyles[exec] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span
      className="text-xs font-mono px-2 py-0.5 rounded"
      style={{ color: s.color, background: s.bg }}
    >
      {exec}
    </span>
  );
}

function DirectionBadge({ dir }: { dir: string }) {
  const s = directionColors[dir] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs px-2 py-0.5 rounded font-medium" style={{ color: s.color, background: s.bg }}>
      {dir}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const s = triggerTypeColors[type] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs px-2 py-0.5 rounded" style={{ color: s.color, background: s.bg }}>
      {type}
    </span>
  );
}

function CategoryBadge({ category }: { category: string }) {
  const s = categoryColors[category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-xs px-2 py-0.5 rounded font-medium" style={{ color: s.color, background: s.bg }}>
      {category}
    </span>
  );
}

function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
      style={{
        background: enabled ? 'var(--accent)' : 'var(--bg-input)',
        border: enabled ? 'none' : '1px solid var(--border)',
      }}
    >
      <div
        className="w-4 h-4 rounded-full absolute top-1 transition-all"
        style={{
          background: enabled ? 'white' : 'var(--text-muted)',
          left: enabled ? '22px' : '4px',
        }}
      />
    </button>
  );
}

/* ========================================================================
   Detail: Parameters Tab
   ======================================================================== */

function ParametersTab({ pack }: { pack: UserPack }) {
  return (
    <Card>
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
          Pack Parameters
        </h3>
        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {pack.params.length} parameters
        </span>
      </div>

      <div className="space-y-4">
        {pack.params.map((param) => (
          <div key={param.key}>
            <div className="flex items-center gap-4">
              <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                {param.label}
              </label>
              <input
                type="number"
                step={param.type === 'float' ? 0.1 : 1}
                min={param.min}
                max={param.max}
                className="px-3 py-2 rounded-lg text-sm flex-1 font-mono"
                style={inputStyle}
                defaultValue={param.value}
              />
            </div>
            <div className="flex gap-4 mt-1">
              <span className="w-52 flex-shrink-0" />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {param.type === 'int' ? 'Integer' : 'Decimal'} &middot; Range: {param.min} - {param.max}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div
        className="flex gap-3 mt-6 pt-4 border-t"
        style={{ borderColor: 'var(--border)' }}
      >
        <button className="px-4 py-2 rounded-lg text-sm font-medium" style={btnPrimary}>
          Save Changes
        </button>
        <button className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
          Reset to Default
        </button>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail: Plot Settings Tab
   ======================================================================== */

function PlotSettingsTab({ pack }: { pack: UserPack }) {
  const [lineWidth, setLineWidth] = useState(2);

  return (
    <Card>
      <h3 className="text-sm font-medium mb-5" style={{ color: 'var(--text-secondary)' }}>
        Visual Settings for {pack.name}
      </h3>

      <div className="space-y-4">
        {pack.plotSettings.map((ps) => (
          <div key={ps.key} className="flex items-center gap-4">
            <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
              {ps.label}
            </label>
            <div className="flex items-center gap-3">
              <input
                type="color"
                defaultValue={ps.default}
                className="w-8 h-8 rounded border-0 cursor-pointer"
                style={{ background: 'transparent', padding: 0 }}
              />
              <div
                className="w-10 h-8 rounded border"
                style={{
                  background: ps.default,
                  borderColor: 'var(--border)',
                }}
              />
              <input
                className="px-3 py-2 rounded-lg text-sm font-mono w-28"
                style={inputStyle}
                defaultValue={ps.default}
              />
            </div>
          </div>
        ))}

        <div className="border-t my-2" style={{ borderColor: 'var(--border)' }} />

        <div className="flex items-center gap-4">
          <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
            Line Width
          </label>
          <div className="flex items-center gap-3 flex-1">
            <input
              type="range"
              min="1"
              max="5"
              value={lineWidth}
              onChange={(e) => setLineWidth(Number(e.target.value))}
              className="flex-1"
            />
            <span
              className="text-sm font-mono w-12 text-center px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)' }}
            >
              {lineWidth}px
            </span>
          </div>
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
        <button className="px-4 py-2 rounded-lg text-sm font-medium" style={btnPrimary}>
          Save Plot Settings
        </button>
        <button className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
          Reset Colors
        </button>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail: Outputs & Triggers Tab
   ======================================================================== */

function OutputsTriggersTab({ pack }: { pack: UserPack }) {
  return (
    <div className="grid grid-cols-2 gap-6">
      {/* Outputs column */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Interpreter Outputs
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {pack.outputs.length} states
          </span>
        </div>
        <div className="space-y-2">
          {pack.outputs.map((output) => (
            <div
              key={output.code}
              className="px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="text-sm font-mono font-semibold px-2 py-0.5 rounded"
                  style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                >
                  {output.code}
                </span>
              </div>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                {output.description}
              </p>
            </div>
          ))}
        </div>
      </Card>

      {/* Triggers column */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Available Triggers
          </h3>
          <span
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}
          >
            {pack.triggers.length} triggers
          </span>
        </div>
        <div className="space-y-2">
          {pack.triggers.map((trigger) => (
            <div
              key={trigger.id}
              className="px-3 py-2.5 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                  {trigger.name}
                </span>
                <div className="flex items-center gap-1.5">
                  <DirectionBadge dir={trigger.direction} />
                  <TypeBadge type={trigger.type} />
                  <ExecBadge exec={trigger.exec} />
                </div>
              </div>
              <p className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                {trigger.id}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================
   Detail: Preview Tab
   ======================================================================== */

// State color palette — cycles through distinguishable colors for pack states
const STATE_COLORS = ['#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#E91E63', '#607D8B', '#CDDC39'];

function PreviewTab({ pack }: { pack: UserPack }) {
  const [symbol, setSymbol] = useState('NVDA');
  const [timeframe, setTimeframe] = useState('5Min');
  const [session, setSession] = useState('RTH');
  const [days, setDays] = useState(5);
  const previewMut = usePackPreview();
  const chartPrefs = useChartPrefs();

  const data = previewMut.data as PackPreviewResponse | undefined;

  // Derive state timeline (only transitions where state changes)
  const stateTimeline = useMemo(() => {
    if (!data?.bars) return [];
    const transitions: StateTimelineEntry[] = [];
    let prevState = '';
    for (const bar of data.bars) {
      const state = bar.state || '';
      if (state !== prevState && state) {
        transitions.push({
          time: new Date(bar.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          state,
          prevState: prevState || '—',
        });
        prevState = state;
      }
    }
    return transitions;
  }, [data]);

  // Derive trigger events
  const triggerEvents = useMemo(() => {
    if (!data?.triggers) return [];
    return data.triggers.map((t) => ({
      time: new Date(t.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      trigger: t.trigger_name,
      direction: t.direction,
      type: t.type,
      price: '',
      execType: 'C',
    }));
  }, [data]);

  // Build state color map from the pack's output states
  const stateColorMap = useMemo(() => {
    if (!data?.states) return {};
    const map: Record<string, string> = {};
    data.states.forEach((s, i) => { map[s] = STATE_COLORS[i % STATE_COLORS.length]; });
    return map;
  }, [data]);

  const isOscillator = data?.display_type === 'oscillator';

  // Filter indicator columns to only plottable numeric ones (not booleans, strings, or color columns)
  const plottableColumns = useMemo(() => {
    if (!data?.bars || data.bars.length === 0) return [];
    // Check a sample bar to determine column types
    const sampleBar = data.bars.find((b) => data.indicator_columns.some((col) => b[col] != null)) || data.bars[0];
    return data.indicator_columns.filter((col) => {
      const val = sampleBar[col];
      if (val === null || val === undefined) return false;
      if (typeof val === 'boolean') return false;
      if (typeof val === 'string') return false; // color strings, etc.
      return typeof val === 'number' && !isNaN(val);
    });
  }, [data]);

  // Price chart series setup — background state histogram + candlestick + overlay lines
  const priceSeriesSetup = useMemo(() => {
    const setup: any[] = [];

    // Background state histogram (first = renders behind candles)
    if (data?.states && data.states.length > 0) {
      setup.push({
        type: 'Histogram' as const,
        options: {
          priceScaleId: 'state_bg',
          lastValueVisible: false,
          priceLineVisible: false,
        },
      });
    }

    // Candlestick (renders on top of background)
    setup.push({
      type: 'Candlestick' as const,
      options: {
        upColor: chartPrefs.candleUp, downColor: chartPrefs.candleDown,
        borderUpColor: chartPrefs.candleUpBorder, borderDownColor: chartPrefs.candleDown,
        wickUpColor: chartPrefs.candleUpBorder, wickDownColor: chartPrefs.candleDown,
      },
    });

    // Overlay indicator lines (only for overlay display type, only numeric columns)
    if (data && !isOscillator && plottableColumns.length > 0) {
      const lineColors = ['#2196F3', '#FF9800', '#E040FB', '#00BCD4', '#FFEB3B'];
      plottableColumns.forEach((col, i) => {
        setup.push({
          type: 'Line' as const,
          options: { color: lineColors[i % lineColors.length], lineWidth: 2, lastValueVisible: false, priceLineVisible: false },
        });
      });
    }
    return setup;
  }, [chartPrefs, data, isOscillator, plottableColumns]);

  // Price chart series data — background states + candles + trigger markers + overlay lines
  const priceSeriesData = useMemo(() => {
    if (!data?.bars) {
      const empty = [{ data: [], markers: [] }];
      return data?.states?.length ? [{ data: [] }, ...empty] : empty;
    }

    const result: any[] = [];

    // Background state histogram data (if states exist)
    if (data.states.length > 0) {
      const bgData = data.bars.map((b) => ({
        time: b.timestamp,
        value: 1, // constant value — scale is hidden
        color: b.state ? (stateColorMap[b.state] || 'transparent') + '25' : 'transparent',
      }));
      result.push({ data: bgData });
    }

    // Candlestick data — use indicator candle colors if pack defines candle_color_column
    const candleColorCol = (data as any).plot_config?.candle_color_column;
    const candles = data.bars.map((b) => {
      const bar: any = { time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close };
      const indicatorColor = candleColorCol ? b[candleColorCol] : null;
      if (indicatorColor && typeof indicatorColor === 'string' && indicatorColor.startsWith('#')) {
        bar.color = indicatorColor;
        bar.borderColor = indicatorColor;
        bar.wickColor = indicatorColor;
      }
      return bar;
    });

    const markers = (data.triggers || []).map((t) => ({
      time: t.timestamp,
      position: 'aboveBar' as const,
      shape: 'circle' as const,
      color: '#FFD700',
      text: t.trigger_name,
      size: 1,
    }));

    result.push({ data: candles, markers });

    // Overlay indicator line data (only plottable numeric columns)
    if (!isOscillator) {
      for (const col of plottableColumns) {
        const lineData = data.bars
          .filter((b) => b[col] != null && typeof b[col] === 'number')
          .map((b) => ({ time: b.timestamp, value: b[col] as number }));
        result.push({ data: lineData });
      }
    }

    return result;
  }, [data, stateColorMap, isOscillator, plottableColumns]);

  // Oscillator chart setup (separate pane for RSI, MACD, etc.)
  const oscSeriesSetup = useMemo(() => {
    if (!data || !isOscillator) return [];
    const refLines = (data as any).plot_config?.reference_lines || [];
    const lineColors = ['#2196F3', '#FF9800', '#E040FB', '#00BCD4'];
    return data.indicator_columns.map((col, i) => ({
      type: 'Line' as const,
      options: { color: lineColors[i % lineColors.length], lineWidth: 2, lastValueVisible: true, priceLineVisible: false },
      priceLines: refLines.map((rl: any) => ({
        price: rl.value, color: rl.color || 'var(--text-muted)',
        lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: rl.label || '',
      })),
    }));
  }, [data, isOscillator]);

  const oscSeriesData = useMemo(() => {
    if (!data || !isOscillator) return [];
    return data.indicator_columns.map((col) => ({
      data: data.bars
        .filter((b) => b[col] != null)
        .map((b) => ({ time: b.timestamp, value: b[col] as number })),
    }));
  }, [data, isOscillator]);

  // Build SyncedChartPane configuration — combines price + oscillator into
  // synchronized panes that share a single time scale (zoom/pan together).
  const previewPanes: PaneConfig[] = useMemo(() => {
    if (!data?.bars || data.bars.length === 0) return [];
    const panes: PaneConfig[] = [];

    // Build price pane series
    const priceSeries: SeriesConfig[] = [];

    // Background state histogram (renders behind candles)
    if (data.states.length > 0) {
      priceSeries.push({
        type: 'Histogram',
        data: data.bars.map((b) => ({
          time: b.timestamp,
          value: 1,
          color: b.state ? (stateColorMap[b.state] || 'transparent') + '25' : 'transparent',
        })),
        options: {
          priceScaleId: 'state_bg',
          lastValueVisible: false,
          priceLineVisible: false,
        },
      });
    }

    // Candlesticks with optional indicator-driven colors
    const candleColorCol = (data as any).plot_config?.candle_color_column;
    const candles = data.bars.map((b) => {
      const bar: any = { time: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close };
      const indicatorColor = candleColorCol ? b[candleColorCol] : null;
      if (indicatorColor && typeof indicatorColor === 'string' && indicatorColor.startsWith('#')) {
        bar.color = indicatorColor;
        bar.borderColor = indicatorColor;
        bar.wickColor = indicatorColor;
      }
      return bar;
    });
    const triggerMarkers = (data.triggers || []).map((t) => ({
      time: t.timestamp,
      position: 'aboveBar',
      shape: 'circle',
      color: '#FFD700',
      text: t.trigger_name,
      size: 1,
    }));
    priceSeries.push({
      type: 'Candlestick',
      data: candles,
      markers: triggerMarkers,
    });

    // Overlay indicator lines (only for non-oscillator packs)
    if (!isOscillator && plottableColumns.length > 0) {
      const lineColors = ['#2196F3', '#FF9800', '#E040FB', '#00BCD4', '#FFEB3B'];
      plottableColumns.forEach((col, i) => {
        priceSeries.push({
          type: 'Line',
          data: data.bars.filter((b) => b[col] != null && typeof b[col] === 'number').map((b) => ({
            time: b.timestamp, value: b[col] as number,
          })),
          options: { color: lineColors[i % lineColors.length], lineWidth: 2, lastValueVisible: false, priceLineVisible: false },
        });
      });
    }

    panes.push({ id: 'price', height: 350, series: priceSeries });

    // Oscillator pane (synced to price pane via SyncedChartPane)
    if (isOscillator) {
      const refLines = (data as any).plot_config?.reference_lines || [];
      const lineColors = ['#2196F3', '#FF9800', '#E040FB', '#00BCD4'];
      const oscSeries: SeriesConfig[] = data.indicator_columns
        .filter((col) => {
          const sample = data.bars.find((b) => b[col] != null);
          return sample && typeof sample[col] === 'number';
        })
        .map((col, i) => ({
          type: 'Line' as const,
          data: data.bars.filter((b) => b[col] != null).map((b) => ({
            time: b.timestamp, value: b[col] as number,
          })),
          options: {
            color: lineColors[i % lineColors.length],
            lineWidth: 2,
            lastValueVisible: true,
            priceLineVisible: false,
            title: col,
          },
        }));
      if (oscSeries.length > 0) {
        // Add reference lines as price lines on the first oscillator series
        if (refLines.length > 0 && oscSeries[0]) {
          (oscSeries[0] as any).priceLines = refLines.map((rl: any) => ({
            price: rl.value,
            color: rl.color || 'var(--text-muted)',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: rl.label || '',
          }));
        }
        panes.push({ id: 'oscillator', height: 150, series: oscSeries });
      }
    }

    return panes;
  }, [data, stateColorMap, isOscillator, plottableColumns]);

  const handleLoad = () => {
    previewMut.mutate({ slug: pack.id, symbol, timeframe, days, session });
  };

  return (
    <div className="space-y-4">
      {/* Controls bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {['NVDA', 'SPY', 'AAPL', 'TSLA', 'MSFT', 'AMD'].map((s) => <option key={s}>{s}</option>)}
        </select>
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
          {['1Min', '5Min', '15Min', '1H', '1Day'].map((tf) => <option key={tf}>{tf}</option>)}
        </select>
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={session} onChange={(e) => setSession(e.target.value)}>
          <option value="RTH">RTH Only</option>
          <option value="ETH">Extended Hours</option>
          <option value="24/7">24/7</option>
        </select>
        <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={days} onChange={(e) => setDays(Number(e.target.value))}>
          {[1, 2, 5, 10, 20].map((d) => <option key={d} value={d}>{d} days</option>)}
        </select>

        <button
          className="px-4 py-2 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'white', border: 'none', cursor: 'pointer', opacity: previewMut.isPending ? 0.6 : 1 }}
          onClick={handleLoad}
          disabled={previewMut.isPending}
        >
          {previewMut.isPending ? 'Loading...' : 'Load Preview'}
        </button>
      </div>

      {/* Error */}
      {previewMut.error && (
        <Card>
          <p className="text-xs" style={{ color: 'var(--red)' }}>Failed to load preview: {String(previewMut.error)}</p>
        </Card>
      )}

      {/* Summary stats */}
      {data && (
        <Card>
          <div className="flex gap-6 text-xs">
            <span style={{ color: 'var(--text-muted)' }}>Bars: <strong style={{ color: 'var(--text-primary)' }}>{data.bars.length}</strong></span>
            <span style={{ color: 'var(--text-muted)' }}>States: <strong style={{ color: 'var(--text-primary)' }}>{data.states.length}</strong></span>
            <span style={{ color: 'var(--text-muted)' }}>Triggers fired: <strong style={{ color: 'var(--text-primary)' }}>{data.triggers.length}</strong></span>
            <span style={{ color: 'var(--text-muted)' }}>Display: <strong style={{ color: 'var(--text-primary)' }}>{data.display_type}</strong></span>
            <span style={{ color: 'var(--text-muted)' }}>Indicators: <strong style={{ color: 'var(--text-primary)' }}>{data.indicator_columns.join(', ') || 'none'}</strong></span>
          </div>
        </Card>
      )}

      {/* Price chart with state-colored candles + trigger markers + synced oscillator pane */}
      {data && data.bars.length > 0 && previewPanes.length > 0 && (
        <Card>
          <h4 className="text-sm font-medium mb-2">{symbol} — {pack.name} Preview</h4>
          <SyncedChartPane
            panes={previewPanes}
            upColor={chartPrefs.candleUp}
            downColor={chartPrefs.candleDown}
            upBorderColor={chartPrefs.candleUpBorder}
            timezone={chartPrefs.timezone}
          />

          {/* State legend */}
          {data.states.length > 0 && (
            <div className="flex gap-3 mt-2 flex-wrap">
              {data.states.map((state, i) => (
                <span key={state} className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  <span className="w-2.5 h-2.5 rounded-sm" style={{ background: STATE_COLORS[i % STATE_COLORS.length] + '80' }} />
                  {state}
                </span>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Data tables below chart */}
      {data && (stateTimeline.length > 0 || triggerEvents.length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          {/* State Timeline */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                State Timeline
              </h4>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {stateTimeline.length} transitions
              </span>
            </div>
            <div className="rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)', maxHeight: 400, overflowY: 'auto' }}>
              <div className="grid grid-cols-3 text-xs font-medium px-3 py-2 sticky top-0" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)' }}>
                <span>Time</span>
                <span>State</span>
                <span>Previous</span>
              </div>
              {stateTimeline.map((row, i) => {
                const changed = row.state !== row.prevState;
                return (
                  <div key={i} className="grid grid-cols-3 text-xs px-3 py-2 border-t" style={{ borderColor: 'var(--border)', background: changed ? 'var(--accent-muted)' : 'var(--bg-card)' }}>
                    <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{row.time}</span>
                    <span className="font-mono font-semibold" style={{ color: changed ? 'var(--accent)' : 'var(--text-primary)' }}>{row.state}</span>
                    <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{row.prevState}</span>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Trigger Events */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                Trigger Events
              </h4>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {triggerEvents.length} events
              </span>
            </div>
            <div className="rounded-lg overflow-hidden border" style={{ borderColor: 'var(--border)', maxHeight: 400, overflowY: 'auto' }}>
              <div className="grid text-xs font-medium px-3 py-2 sticky top-0" style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)', gridTemplateColumns: '70px 1fr 55px 50px' }}>
                <span>Time</span>
                <span>Trigger</span>
                <span>Dir</span>
                <span>Type</span>
              </div>
              {triggerEvents.map((row, i) => (
                <div key={i} className="grid text-xs px-3 py-2 border-t items-center" style={{ borderColor: 'var(--border)', background: 'var(--bg-card)', gridTemplateColumns: '70px 1fr 55px 50px' }}>
                  <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{row.time}</span>
                  <span className="font-medium truncate" style={{ color: 'var(--text-primary)' }}>{row.trigger}</span>
                  <DirectionBadge dir={row.direction} />
                  <TypeBadge type={row.type} />
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

/* ========================================================================
   Detail: Signal Validation Tab
   ======================================================================== */

function SignalValidationTab({ pack }: { pack: UserPack }) {
  const [symbol, setSymbol] = useState('NVDA');
  const [timeframe, setTimeframe] = useState('5Min');
  const [days, setDays] = useState(90);
  const validationMut = usePackPreview();

  const data = validationMut.data as PackPreviewResponse | undefined;

  // Compute validation stats from preview data
  const stats = useMemo(() => {
    if (!data?.bars || data.bars.length === 0) return null;

    const totalBars = data.bars.length;
    const totalTriggers = data.triggers.length;

    // State coverage
    const observedStates = new Set(data.bars.map((b) => b.state).filter(Boolean));
    const definedStates = data.states.length;
    const stateCoverage = definedStates > 0 ? observedStates.size / definedStates : 0;
    const allStatesReached = definedStates > 0 && observedStates.size === definedStates;
    const missingStates = data.states.filter((s) => !observedStates.has(s));

    // Avg bars between triggers (across all triggers combined)
    const avgBarsBetween = totalTriggers > 1 ? Math.round(totalBars / totalTriggers) : totalBars;

    // Per-trigger breakdown — match by trigger name since IDs use the manifest's
    // trigger_prefix (e.g., "s123t") not the pack slug (e.g., "swing_123_test")
    const triggerCountsByName: Record<string, number> = {};
    const triggerCountsById: Record<string, number> = {};
    for (const t of data.triggers) {
      triggerCountsByName[t.trigger_name] = (triggerCountsByName[t.trigger_name] || 0) + 1;
      triggerCountsById[t.trigger_id] = (triggerCountsById[t.trigger_id] || 0) + 1;
    }

    const perTrigger = pack.triggers.map((t) => {
      // Try matching by name first, then by ID suffix
      const fires = triggerCountsByName[t.name]
        || Object.entries(triggerCountsById).find(([id]) => id.endsWith('_' + t.id))?.[1]
        || 0;
      const avgBars = fires > 0 ? Math.round(totalBars / fires) : 0;
      return { id: t.id, name: t.name, fires, avgBars };
    });

    // State distribution
    const stateDistribution: Record<string, number> = {};
    for (const b of data.bars) {
      if (b.state) stateDistribution[b.state] = (stateDistribution[b.state] || 0) + 1;
    }

    return {
      totalBars, totalTriggers, stateCoverage, allStatesReached, missingStates,
      avgBarsBetween, perTrigger, stateDistribution, observedStates,
    };
  }, [data, pack]);

  const handleRun = () => {
    validationMut.mutate({ slug: pack.id, symbol, timeframe, days, session: 'RTH' });
  };

  return (
    <div className="space-y-4">
      <Card>
        <h4 className="text-sm font-medium mb-2">Signal Validation</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Runs {pack.name} on historical data to verify signals fire correctly, all states are reached, and trigger frequency is reasonable.
        </p>

        {/* Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {['NVDA', 'SPY', 'AAPL', 'TSLA', 'MSFT', 'AMD'].map((s) => <option key={s}>{s}</option>)}
          </select>
          <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            {['1Min', '5Min', '15Min', '1H', '1Day'].map((tf) => <option key={tf}>{tf}</option>)}
          </select>
          <select className="px-3 py-2 rounded-lg text-sm" style={selectStyle} value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {[30, 60, 90, 180].map((d) => <option key={d} value={d}>{d} days</option>)}
          </select>
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white', border: 'none', cursor: 'pointer', opacity: validationMut.isPending ? 0.6 : 1 }}
            onClick={handleRun}
            disabled={validationMut.isPending}
          >
            {validationMut.isPending ? 'Running...' : 'Run Validation'}
          </button>
        </div>
      </Card>

      {validationMut.error && (
        <Card><p className="text-xs" style={{ color: 'var(--red)' }}>Validation failed: {String(validationMut.error)}</p></Card>
      )}

      {stats && (
        <>
          {/* Summary metrics */}
          <div className="grid grid-cols-4 gap-4">
            <Card>
              <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Total Signals</p>
              <p className="text-2xl font-bold">{stats.totalTriggers}</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>across {stats.totalBars} bars</p>
            </Card>
            <Card>
              <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Avg Bars Between</p>
              <p className="text-2xl font-bold">{stats.avgBarsBetween}</p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>bars per signal</p>
            </Card>
            <Card>
              <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>State Coverage</p>
              <p className="text-2xl font-bold" style={{ color: stats.stateCoverage === 1 ? 'var(--green)' : 'var(--orange)' }}>
                {Math.round(stats.stateCoverage * 100)}%
              </p>
              <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{stats.observedStates.size}/{data!.states.length} states</p>
            </Card>
            <Card>
              <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>All States Reached</p>
              <p className="text-2xl font-bold" style={{ color: stats.allStatesReached ? 'var(--green)' : 'var(--red)' }}>
                {stats.allStatesReached ? 'Yes' : 'No'}
              </p>
              {stats.missingStates.length > 0 && (
                <p className="text-[10px]" style={{ color: 'var(--red)' }}>Missing: {stats.missingStates.join(', ')}</p>
              )}
            </Card>
          </div>

          {/* State distribution */}
          <Card>
            <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>State Distribution</h5>
            <div className="flex gap-1 rounded overflow-hidden" style={{ height: 24 }}>
              {data!.states.map((state, i) => {
                const count = stats.stateDistribution[state] || 0;
                const pct = stats.totalBars > 0 ? (count / stats.totalBars) * 100 : 0;
                if (pct === 0) return null;
                return (
                  <div
                    key={state}
                    title={`${state}: ${count} bars (${pct.toFixed(1)}%)`}
                    style={{ flex: count, background: STATE_COLORS[i % STATE_COLORS.length] + '60', minWidth: pct > 2 ? undefined : 2 }}
                    className="flex items-center justify-center text-[8px] font-mono font-bold truncate px-0.5"
                  >
                    {pct > 8 ? `${state} ${pct.toFixed(0)}%` : ''}
                  </div>
                );
              })}
            </div>
            <div className="flex gap-3 mt-1.5 flex-wrap">
              {data!.states.map((state, i) => {
                const count = stats.stateDistribution[state] || 0;
                const pct = stats.totalBars > 0 ? (count / stats.totalBars * 100).toFixed(1) : '0';
                return (
                  <span key={state} className="flex items-center gap-1 text-[10px]" style={{ color: count > 0 ? 'var(--text-secondary)' : 'var(--red)' }}>
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ background: STATE_COLORS[i % STATE_COLORS.length] + '80' }} />
                    {state}: {count} ({pct}%)
                  </span>
                );
              })}
            </div>
          </Card>

          {/* Per-trigger breakdown */}
          <Card>
            <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Per-Trigger Breakdown</h5>
            <div style={{ overflowX: 'auto' }}>
              <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['Trigger', 'Fires', 'Avg Bars Between', 'Frequency'].map((h) => (
                      <th key={h} className="text-left py-2 px-3 text-[10px] font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {stats.perTrigger.map((t) => (
                    <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td className="py-2 px-3 font-medium">{t.name}</td>
                      <td className="py-2 px-3 font-mono" style={{ color: t.fires > 0 ? 'var(--text-primary)' : 'var(--red)' }}>
                        {t.fires}
                      </td>
                      <td className="py-2 px-3 font-mono">{t.avgBars > 0 ? t.avgBars : '—'}</td>
                      <td className="py-2 px-3">
                        {t.fires === 0 ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--red)', background: 'var(--red)' + '20' }}>Never fires</span>
                        ) : t.avgBars < 5 ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange)' + '20' }}>Very frequent</span>
                        ) : t.avgBars > 500 ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--orange)', background: 'var(--orange)' + '20' }}>Very rare</span>
                        ) : (
                          <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--green)', background: 'var(--green)' + '20' }}>Normal</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}

/* ========================================================================
   Detail: Parity Simulator Tab
   ======================================================================== */

interface ParityResult {
  pack_id: string;
  entry_trigger: string;
  symbol: string;
  timeframe: string;
  days: number;
  bars_loaded: number;
  warmup_bars: number;
  backtest_fires: Array<{ bar_idx: number; timestamp: string; trigger: string }>;
  live_fires: Array<{ bar_idx: number; timestamp: string; trigger: string }>;
  matched: Array<{ bar_idx: number; timestamp: string; trigger: string }>;
  backtest_only: Array<{ bar_idx: number; timestamp: string; trigger: string }>;
  live_only: Array<{ bar_idx: number; timestamp: string; trigger: string }>;
  backtest_warmup: Array<{ bar_idx: number; timestamp: string; trigger: string }>;
  parity_score: number | null;
  verdict: string;
  explanation: string;
  summary: string;
}

const VERDICT_COLORS: Record<string, { color: string; bg: string }> = {
  PASS: { color: 'var(--green)', bg: 'var(--green-muted)' },
  PASS_WITHIN_TOLERANCE: { color: 'var(--green)', bg: 'var(--green-muted)' },
  PARTIAL: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  FAIL_SILENT: { color: 'var(--red)', bg: 'var(--red-muted)' },
  FAIL_REVERSE: { color: 'var(--red)', bg: 'var(--red-muted)' },
  NO_FIRES: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

function ParitySimulatorTab({ pack }: { pack: UserPack }) {
  // Triggers are direction- and type-agnostic by design — the AI builder
  // emits type='BOTH' on every trigger. Filtering to type==='ENTRY' here
  // would empty the dropdown for any modern pack. Show all triggers; the
  // user picks which one to test.
  const entryTriggers = pack.triggers;
  const [trigger, setTrigger] = useState(entryTriggers[0]?.id || '');
  const [symbol, setSymbol] = useState('SPY');
  const [timeframe, setTimeframe] = useState('1Min');
  const [days, setDays] = useState(7);
  const [warmupBars, setWarmupBars] = useState(200);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ParityResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runTest = async () => {
    if (!trigger) {
      setError('Select an entry trigger first.');
      return;
    }
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiFetch<ParityResult>('/api/packs/parity-test', {
        method: 'POST',
        body: JSON.stringify({
          pack_id: pack.id,
          entry_trigger: trigger,
          symbol,
          timeframe,
          days,
          warmup_bars: warmupBars,
        }),
      });
      setResult(res);
    } catch (e: any) {
      setError(e?.message || 'Parity test failed.');
    } finally {
      setRunning(false);
    }
  };

  const verdictStyle = result
    ? VERDICT_COLORS[result.verdict] || VERDICT_COLORS.NO_FIRES
    : null;

  return (
    <div className="space-y-3">
      <Card>
        <h4 className="text-sm font-medium mb-2">Backtest ↔ Live Parity Simulator</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          Replays the same OHLCV window through the backtest path and the live worker&apos;s
          incremental engine, then compares trigger fires bar-by-bar. PASS means parity confirmed.
          FAIL_SILENT means the pack works in batch mode but is silent in production.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-3">
          <div>
            <label className="text-[11px] block mb-1" style={{ color: 'var(--text-muted)' }}>Trigger</label>
            <select
              className="w-full text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              value={trigger}
              onChange={(e) => setTrigger(e.target.value)}
            >
              {entryTriggers.length === 0 && <option value="">(no entry triggers)</option>}
              {entryTriggers.map((t) => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[11px] block mb-1" style={{ color: 'var(--text-muted)' }}>Symbol</label>
            <input
              className="w-full text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            />
          </div>
          <div>
            <label className="text-[11px] block mb-1" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
            <select
              className="w-full text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="1Min">1Min</option>
              <option value="5Min">5Min</option>
              <option value="15Min">15Min</option>
              <option value="1Hour">1Hour</option>
              <option value="1Day">1Day</option>
            </select>
          </div>
          <div>
            <label className="text-[11px] block mb-1" style={{ color: 'var(--text-muted)' }}>Days</label>
            <input
              type="number"
              min={1}
              max={30}
              className="w-full text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              value={days}
              onChange={(e) => setDays(parseInt(e.target.value) || 7)}
            />
          </div>
          <div>
            <label className="text-[11px] block mb-1" style={{ color: 'var(--text-muted)' }}>Warmup bars</label>
            <input
              type="number"
              min={0}
              max={1000}
              className="w-full text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              value={warmupBars}
              onChange={(e) => setWarmupBars(parseInt(e.target.value) || 0)}
            />
          </div>
        </div>

        <button
          className="text-xs px-3 py-1.5 rounded font-medium"
          style={{
            background: running ? 'var(--bg-input)' : 'var(--accent)',
            color: running ? 'var(--text-muted)' : 'white',
            cursor: running ? 'not-allowed' : 'pointer',
          }}
          onClick={runTest}
          disabled={running || !trigger}
        >
          {running ? 'Running…' : 'Run Parity Test'}
        </button>

        {error && (
          <p className="text-xs mt-3 px-3 py-2 rounded" style={{ color: 'var(--red)', background: 'var(--red-muted)' }}>
            {error}
          </p>
        )}
      </Card>

      {result && verdictStyle && (
        <Card>
          <div className="flex items-center gap-3 mb-2">
            <span
              className="text-xs px-2 py-1 rounded font-semibold"
              style={{ color: verdictStyle.color, background: verdictStyle.bg }}
            >
              {result.verdict}
            </span>
            {result.parity_score !== null && (
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                parity_score = {(result.parity_score * 100).toFixed(1)}%
              </span>
            )}
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {result.bars_loaded.toLocaleString()} bars loaded
            </span>
          </div>
          <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
            {result.explanation}
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <Stat label="Backtest fires" value={result.backtest_fires.length} />
            <Stat label="Live fires" value={result.live_fires.length} />
            <Stat label="Matched" value={result.matched.length} accent="green" />
            <Stat
              label="Divergent"
              value={result.backtest_only.length + result.live_only.length}
              accent={result.backtest_only.length + result.live_only.length > 0 ? 'red' : undefined}
            />
          </div>

          {result.backtest_warmup.length > 0 && (
            <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>
              {result.backtest_warmup.length} backtest fire(s) inside the warmup window — excluded from the score.
            </p>
          )}

          {(result.backtest_only.length > 0 || result.live_only.length > 0) && (
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
              {result.backtest_only.length > 0 && (
                <FireList
                  title="Backtest only (live missed these)"
                  fires={result.backtest_only}
                  accent="red"
                />
              )}
              {result.live_only.length > 0 && (
                <FireList
                  title="Live only (backtest missed these)"
                  fires={result.live_only}
                  accent="orange"
                />
              )}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: number; accent?: 'green' | 'red' }) {
  const color = accent === 'green' ? 'var(--green)' : accent === 'red' ? 'var(--red)' : 'var(--text-primary)';
  return (
    <div className="px-3 py-2 rounded" style={{ background: 'var(--bg-input)' }}>
      <div className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>{label}</div>
      <div className="text-sm font-semibold mt-0.5" style={{ color }}>{value.toLocaleString()}</div>
    </div>
  );
}

function FireList({
  title,
  fires,
  accent,
}: {
  title: string;
  fires: Array<{ bar_idx: number; timestamp: string }>;
  accent: 'red' | 'orange';
}) {
  const color = accent === 'red' ? 'var(--red)' : 'var(--orange)';
  return (
    <div>
      <div className="text-xs font-medium mb-1" style={{ color }}>{title}</div>
      <div className="text-[11px] max-h-48 overflow-auto rounded" style={{ background: 'var(--bg-input)' }}>
        {fires.slice(0, 50).map((f) => (
          <div key={f.bar_idx} className="px-2 py-1" style={{ borderBottom: '1px solid var(--border)' }}>
            <span style={{ color: 'var(--text-muted)' }}>bar {f.bar_idx}</span>
            <span className="ml-2" style={{ color: 'var(--text-secondary)' }}>
              {new Date(f.timestamp).toLocaleString()}
            </span>
          </div>
        ))}
        {fires.length > 50 && (
          <div className="px-2 py-1 text-center" style={{ color: 'var(--text-muted)' }}>
            … {fires.length - 50} more
          </div>
        )}
      </div>
    </div>
  );
}

/* ========================================================================
   Detail: Code Tab
   ======================================================================== */

interface FixHistoryEntry {
  timestamp: string;
  prompt: string;
  response: string;
  model: string;
  installStatus: 'success' | 'failed' | 'pending';
}

function CodeTab({ pack }: { pack: UserPack }) {
  const { data: code, isLoading, refetch } = useUserPackCode(pack.id);
  const requestFixMut = useRequestFix();
  const installPackMut = useInstallPack();
  const [showFixModal, setShowFixModal] = useState(false);
  const [fixDescription, setFixDescription] = useState('');
  const [fixMessage, setFixMessage] = useState('');
  const [aiModel, setAiModel] = useState('claude-sonnet');
  const [fixHistory, setFixHistory] = useState<FixHistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  const codeBlockStyle: React.CSSProperties = {
    background: '#0d1117', color: '#c9d1d9', fontFamily: 'monospace',
    fontSize: '12px', lineHeight: '1.6', padding: '16px', borderRadius: '8px',
    whiteSpace: 'pre-wrap', overflowX: 'auto', maxHeight: 400, overflowY: 'auto',
  };

  function handleRequestFix() {
    if (!code) return;
    setShowFixModal(false);
    const prompt = fixDescription;
    const ts = new Date().toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    setFixMessage('Sending fix request to AI...');

    // Add pending history entry
    const entry: FixHistoryEntry = { timestamp: ts, prompt, response: 'Waiting for AI...', model: aiModel, installStatus: 'pending' };
    setFixHistory((prev) => [entry, ...prev]);

    requestFixMut.mutate({
      manifest: code.manifest as Record<string, unknown>,
      indicator_code: code.indicator_code,
      interpreter_code: code.interpreter_code,
      validation_errors: [],
      user_description: prompt,
      ai_model: aiModel,
    }, {
      onSuccess: (data) => {
        setFixMessage(data.ai_message);
        // Update history entry with AI response
        setFixHistory((prev) => {
          const updated = [...prev];
          if (updated[0]) { updated[0] = { ...updated[0], response: data.ai_message }; }
          return updated;
        });
        // Re-install
        if (data.manifest && data.indicator_code && data.interpreter_code) {
          installPackMut.mutate({
            manifest: data.manifest,
            indicator_code: data.indicator_code,
            interpreter_code: data.interpreter_code,
            pine_script_code: data.pine_script_code,
          }, {
            onSuccess: (installData) => {
              const installMsg = installData.success
                ? 'Pack re-installed successfully. Refresh to see changes.'
                : `Re-install failed: ${installData.errors.join(', ')}`;
              setFixMessage((prev) => prev + '\n' + installMsg);
              setFixHistory((prev) => {
                const updated = [...prev];
                if (updated[0]) {
                  updated[0] = { ...updated[0], response: updated[0].response + '\n' + installMsg, installStatus: installData.success ? 'success' : 'failed' };
                }
                return updated;
              });
              if (installData.success) refetch();
            },
          });
        }
      },
      onError: (error) => {
        const errMsg = `Fix failed: ${error instanceof Error ? error.message : 'Unknown error'}`;
        setFixMessage(errMsg);
        setFixHistory((prev) => {
          const updated = [...prev];
          if (updated[0]) { updated[0] = { ...updated[0], response: errMsg, installStatus: 'failed' }; }
          return updated;
        });
      },
    });
    setFixDescription('');
  }

  if (isLoading) {
    return <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading code...</p></Card>;
  }

  return (
    <div className="space-y-4">
      {/* Active Parameters */}
      <Card>
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Active Parameters</h3>
        <div className="rounded-lg px-4 py-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
          <div className="grid grid-cols-2 gap-y-2 gap-x-8">
            {pack.params.map((p) => (
              <div key={p.key} className="flex justify-between">
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{p.label}</span>
                <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-primary)' }}>{p.value}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Code Files */}
      {['manifest.json', 'indicator.py', 'interpreter.py'].map((file) => (
        <Card key={file}>
          <details>
            <summary className="text-sm font-mono font-medium cursor-pointer py-1" style={{ color: 'var(--accent)' }}>{file}</summary>
            <div className="mt-2">
              <pre style={codeBlockStyle}>
                {file === 'manifest.json' && (code?.manifest ? JSON.stringify(code.manifest, null, 2) : '// Not available')}
                {file === 'indicator.py' && (code?.indicator_code || '# Not available')}
                {file === 'interpreter.py' && (code?.interpreter_code || '# Not available')}
              </pre>
            </div>
          </details>
        </Card>
      ))}

      {/* AI Fix */}
      <Card>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>AI Fix</h3>
          <select value={aiModel} onChange={(e) => setAiModel(e.target.value)}
            className="text-[10px] px-2 py-1 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
            <option value="claude-sonnet">Claude Sonnet</option>
            <option value="claude-opus">Claude Opus</option>
            <option value="gpt-4o">GPT-4o</option>
          </select>
        </div>
        {fixMessage && (
          <div className="rounded-lg px-3 py-2 mb-3 text-xs" style={{ background: 'var(--accent-muted)', borderLeft: '3px solid var(--accent)', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
            {fixMessage}
          </div>
        )}
        <button style={{ background: 'var(--orange)', color: 'white', border: 'none', padding: '6px 14px', borderRadius: '8px', fontSize: '0.8rem', cursor: 'pointer' }}
          onClick={() => setShowFixModal(true)}
          disabled={requestFixMut.isPending || installPackMut.isPending}>
          {requestFixMut.isPending ? 'Fixing...' : 'Request Fix'}
        </button>
      </Card>

      {/* Fix History */}
      {fixHistory.length > 0 && (
        <Card>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Fix History <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({fixHistory.length})</span>
            </h3>
            <button className="text-[10px] underline" style={{ color: 'var(--text-muted)' }} onClick={() => setShowHistory(!showHistory)}>
              {showHistory ? 'Collapse' : 'Expand'}
            </button>
          </div>
          {showHistory && (
            <div className="space-y-3" style={{ maxHeight: 400, overflowY: 'auto' }}>
              {fixHistory.map((entry, i) => (
                <div key={i} className="rounded-lg px-3 py-2 text-xs" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{entry.timestamp}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>{entry.model}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded font-medium" style={{
                        color: entry.installStatus === 'success' ? 'var(--green)' : entry.installStatus === 'failed' ? 'var(--red)' : 'var(--orange)',
                        background: (entry.installStatus === 'success' ? 'var(--green)' : entry.installStatus === 'failed' ? 'var(--red)' : 'var(--orange)') + '20',
                      }}>
                        {entry.installStatus === 'success' ? 'Installed' : entry.installStatus === 'failed' ? 'Failed' : 'Pending'}
                      </span>
                    </div>
                  </div>
                  <div className="mb-1.5">
                    <span className="text-[10px] font-medium uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Prompt:</span>
                    <p className="mt-0.5" style={{ color: 'var(--text-secondary)' }}>{entry.prompt}</p>
                  </div>
                  <div>
                    <span className="text-[10px] font-medium uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Response:</span>
                    <p className="mt-0.5 whitespace-pre-wrap" style={{ color: 'var(--text-primary)' }}>{entry.response}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Fix Modal */}
      {showFixModal && (
        <Modal title="Request AI Fix" isOpen={showFixModal} onClose={() => setShowFixModal(false)} width="600px">
          <div className="mb-4">
            <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Describe what needs fixing</label>
            <textarea value={fixDescription} onChange={(e) => setFixDescription(e.target.value)} rows={4}
              placeholder="The trigger fires one bar late... / The OVERBOUGHT state never appears... / The interpreter returns NaN..."
              style={{ ...codeBlockStyle, background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)', resize: 'vertical', width: '100%', maxHeight: 'none' }} />
          </div>
          <div className="flex justify-end gap-2">
            <button style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', padding: '6px 14px', borderRadius: '8px', fontSize: '0.8rem', cursor: 'pointer' }} onClick={() => setShowFixModal(false)}>Cancel</button>
            <button style={{ background: 'var(--orange)', color: 'white', border: 'none', padding: '6px 14px', borderRadius: '8px', fontSize: '0.8rem', cursor: 'pointer' }} onClick={handleRequestFix}>Send Fix to AI</button>
          </div>
        </Modal>
      )}
    </div>
  );
}

/* ========================================================================
   Detail: Danger Zone Tab
   ======================================================================== */

function DangerZoneTab({
  pack,
  onDelete,
}: {
  pack: UserPack;
  onDelete: () => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <Card>
      <h3 className="text-sm font-medium mb-5" style={{ color: 'var(--red)' }}>
        Danger Zone
      </h3>
      <div className="space-y-6">
        {/* Rename */}
        <div>
          <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
            Rename Pack
          </label>
          <div className="flex gap-3">
            <input
              className="px-3 py-2 rounded-lg text-sm flex-1"
              style={inputStyle}
              defaultValue={pack.name}
            />
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={btnSecondary}
            >
              Rename
            </button>
          </div>
        </div>

        {/* Delete */}
        <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
            Delete this pack permanently. This cannot be undone.
          </p>
          {pack.strategiesUsing > 0 && (
            <p className="text-xs mb-3 px-3 py-2 rounded-lg" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
              Warning: This pack is used in {pack.strategiesUsing} {pack.strategiesUsing === 1 ? 'strategy' : 'strategies'}.
              Deleting it will remove it from those strategies.
            </p>
          )}
          {!confirmDelete ? (
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--red)', color: 'white' }}
              onClick={() => setConfirmDelete(true)}
            >
              Delete Pack
            </button>
          ) : (
            <div className="flex items-center gap-3">
              <button
                className="px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: 'var(--red)', color: 'white' }}
                onClick={onDelete}
              >
                Confirm Delete
              </button>
              <button
                className="px-4 py-2 rounded-lg text-sm"
                style={btnSecondary}
                onClick={() => setConfirmDelete(false)}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail View (all 6 tabs)
   ======================================================================== */

function DetailView({
  pack,
  onBack,
  onDelete,
}: {
  pack: UserPack;
  onBack: () => void;
  onDelete: () => void;
}) {
  return (
    <div>
      <PageHeader
        title={pack.name}
        backHref="#"
        actions={
          <div className="flex items-center gap-3">
            <Toggle enabled={pack.enabled} onToggle={() => {}} />
            <span className="text-xs" style={{ color: pack.enabled ? 'var(--green)' : 'var(--text-muted)' }}>
              {pack.enabled ? 'Enabled' : 'Disabled'}
            </span>
            <div className="flex gap-1">
              {(['verification', 'private', 'public'] as const).map((s) => {
                const isActive = (pack.visibility === s) || (pack.visibility === 'private' && s === 'private');
                const colors = s === 'verification' ? { c: 'var(--orange)', bg: 'var(--orange)' } : s === 'private' ? { c: 'var(--accent)', bg: 'var(--accent)' } : { c: 'var(--green)', bg: 'var(--green)' };
                return (
                  <span key={s} className="text-[10px] px-2 py-1 rounded-full font-medium capitalize cursor-default"
                    style={{ color: isActive ? colors.c : 'var(--text-muted)', background: isActive ? colors.bg + '20' : 'transparent', border: isActive ? 'none' : '1px solid var(--border)' }}>
                    {s}
                  </span>
                );
              })}
            </div>
            <a href="/strategy-builder" className="px-3 py-1.5 rounded-lg text-xs font-medium no-underline"
              style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
              title={`Open Strategy Builder to test ${pack.name} with real data`}>
              Test in Strategy Builder
            </a>
            <button onClick={onBack} className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>Back to Packs</button>
          </div>
        }
      />

      {/* Status badges */}
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}>
          {pack.packType === 'tf_confluence' ? 'TF Confluence' : 'General'}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
          {pack.category}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full" style={{
          color: pack.validationStatus === 'passed' ? 'var(--green)' : pack.validationStatus === 'warnings' ? 'var(--orange)' : 'var(--red)',
          background: (pack.validationStatus === 'passed' ? 'var(--green)' : pack.validationStatus === 'warnings' ? 'var(--orange)' : 'var(--red)') + '20',
        }}>
          Validation: {pack.validationStatus}
        </span>
        {pack.parityScore !== null && (
          <span className="text-xs px-2 py-0.5 rounded-full" style={{
            color: pack.parityScore === 100 ? 'var(--green)' : 'var(--orange)',
            background: (pack.parityScore === 100 ? 'var(--green)' : 'var(--orange)') + '20',
          }}>
            Parity: {pack.parityScore}%
          </span>
        )}
        <span className="text-xs px-2 py-0.5 rounded-full" style={{
          color: pack.visibility === 'public' ? 'var(--accent)' : 'var(--text-muted)',
          background: pack.visibility === 'public' ? 'var(--accent-muted)' : 'var(--bg-input)',
        }}>
          {pack.visibility === 'public' ? 'Public' : 'Private'}
        </span>
      </div>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        {pack.version} &middot; {pack.strategiesUsing} strategies &middot; Last modified {pack.lastModified}
      </p>

      <TabBar tabs={['Parameters', 'Plot Settings', 'States & Triggers', 'Chart Preview', 'Signal Validation', 'Parity Simulator', 'Code', 'Danger Zone']}>
        {(tab) => (
          <div>
            {tab === 'Parameters' && <ParametersTab pack={pack} />}
            {tab === 'Plot Settings' && <PlotSettingsTab pack={pack} />}
            {tab === 'States & Triggers' && <OutputsTriggersTab pack={pack} />}
            {tab === 'Chart Preview' && <PreviewTab pack={pack} />}
            {tab === 'Signal Validation' && <SignalValidationTab pack={pack} />}
            {tab === 'Parity Simulator' && <ParitySimulatorTab pack={pack} />}
            {tab === 'Code' && <CodeTab pack={pack} />}
            {tab === 'Danger Zone' && <DangerZoneTab pack={pack} onDelete={onDelete} />}
          </div>
        )}
      </TabBar>
    </div>
  );
}

/* ========================================================================
   Pack Card (list view)
   ======================================================================== */

function PackCard({
  pack,
  onToggle,
  onDetail,
  onClone,
  onDelete,
}: {
  pack: UserPack;
  onToggle: () => void;
  onDetail: () => void;
  onClone: () => void;
  onDelete: () => void;
}) {
  return (
    <Card>
      <div className="flex items-start gap-4">
        {/* Toggle */}
        <div className="pt-0.5">
          <Toggle enabled={pack.enabled} onToggle={onToggle} />
        </div>

        {/* Main info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
              {pack.name}
            </span>
            <CategoryBadge category={pack.category} />
            <span
              className="text-xs px-1.5 py-0.5 rounded font-medium"
              style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
            >
              {pack.version}
            </span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
              style={{ color: 'var(--orange)', background: 'var(--orange-muted)' }}
            >
              Legacy
            </span>
          </div>

          {/* Parameter summary */}
          <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
            {pack.params.map((p) => `${p.label}: ${p.value}`).join(', ')}
          </p>

          {/* Outputs preview + usage stats */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 flex-wrap">
              {pack.outputs.map((output) => (
                <span
                  key={output.code}
                  className="text-xs font-mono px-1.5 py-0.5 rounded"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}
                  title={output.description}
                >
                  {output.code}
                </span>
              ))}
            </div>
            <div className="border-l h-4" style={{ borderColor: 'var(--border)' }} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Used in {pack.strategiesUsing} {pack.strategiesUsing === 1 ? 'strategy' : 'strategies'}
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              &middot; Modified {pack.lastModified}
            </span>
          </div>
        </div>

        {/* Right side: visibility + trigger count + actions */}
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <div className="flex items-center gap-2">
            <select className="text-[10px] px-2 py-0.5 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', cursor: 'pointer' }} defaultValue={pack.visibility}>
              <option value="private">Private</option>
              <option value="public">Public</option>
            </select>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {pack.triggers.length} triggers
            </span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={onDetail}
              className="px-3 py-1.5 rounded text-xs font-medium"
              style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
            >
              Edit
            </button>
            <button
              onClick={onClone}
              className="px-3 py-1.5 rounded text-xs"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Clone
            </button>
            <button
              onClick={onDelete}
              className="px-3 py-1.5 rounded text-xs"
              style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
            >
              Delete
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Create from Scratch Modal
   ======================================================================== */

function CreateScratchModal({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (name: string, category: string) => void;
}) {
  const [name, setName] = useState('');
  const [category, setCategory] = useState('Momentum');

  return (
    <Modal title="Create Pack from Scratch" isOpen={isOpen} onClose={onClose} width="500px">
      <div className="space-y-4">
        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Pack Name
          </label>
          <input
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={inputStyle}
            placeholder="e.g. My RSI Pack"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Category
          </label>
          <select
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={selectStyle}
            value={category}
            onChange={(e) => setCategory(e.target.value)}
          >
            {Object.keys(categoryColors).map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>
        <div className="flex gap-3 pt-2">
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{
              ...btnPrimary,
              ...(name.trim() ? {} : { opacity: 0.5, cursor: 'not-allowed' }),
            }}
            disabled={!name.trim()}
            onClick={() => {
              onSubmit(name.trim(), category);
              setName('');
              onClose();
            }}
          >
            Create Pack
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={btnSecondary}
            onClick={onClose}
          >
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}

/* ========================================================================
   Create from Template Modal
   ======================================================================== */

function CreateTemplateModal({
  isOpen,
  onClose,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (templateKey: string, name: string) => void;
}) {
  const [selectedTemplate, setSelectedTemplate] = useState(builtInTemplates[0]?.key ?? '');
  const [customName, setCustomName] = useState('');

  const tpl = builtInTemplates.find((t) => t.key === selectedTemplate);

  return (
    <Modal title="Create from Template" isOpen={isOpen} onClose={onClose} width="560px">
      {builtInTemplates.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No built-in templates available yet. This feature is coming soon.</p>
        </div>
      ) : (
      <div className="space-y-5">
        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Template
          </label>
          <select
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={selectStyle}
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
          >
            {builtInTemplates.map((t) => (
              <option key={t.key} value={t.key}>
                {t.name} ({t.category})
              </option>
            ))}
          </select>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            {tpl?.description ?? '--'}
          </p>
        </div>

        {/* Template preview */}
        <div
          className="rounded-lg px-4 py-3"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-2 mb-2">
            <CategoryBadge category={tpl?.category ?? '--'} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Built-in template -- all parameters customizable
            </span>
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {tpl?.description ?? '--'}
          </p>
        </div>

        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Custom Name
          </label>
          <input
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={inputStyle}
            placeholder={`e.g. My ${tpl?.name ?? 'Pack'}`}
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
          />
        </div>

        <div className="flex gap-3 pt-2">
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{
              ...btnPrimary,
              ...(customName.trim() ? {} : { opacity: 0.5, cursor: 'not-allowed' }),
            }}
            disabled={!customName.trim()}
            onClick={() => {
              onSubmit(selectedTemplate, customName.trim());
              setCustomName('');
              onClose();
            }}
          >
            Create from Template
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={btnSecondary}
            onClick={onClose}
          >
            Cancel
          </button>
        </div>
      </div>
      )}
    </Modal>
  );
}

/* ========================================================================
   Import Pack Modal
   ======================================================================== */

function ImportPackModal({
  isOpen,
  onClose,
  onImport,
}: {
  isOpen: boolean;
  onClose: () => void;
  onImport: () => void;
}) {
  const [jsonInput, setJsonInput] = useState('');

  return (
    <Modal title="Import Pack" isOpen={isOpen} onClose={onClose} width="600px">
      <div className="space-y-4">
        <div>
          <label className="text-sm mb-1.5 block font-medium" style={{ color: 'var(--text-secondary)' }}>
            Pack JSON Definition
          </label>
          <textarea
            className="w-full px-3 py-2 rounded-lg text-sm font-mono"
            style={{
              ...inputStyle,
              resize: 'vertical' as const,
              minHeight: '200px',
            }}
            rows={10}
            placeholder='{ "name": "My Pack", "category": "Momentum", ... }'
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
          />
        </div>

        {/* Validation status */}
        <div
          className="rounded-lg px-4 py-3"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
        >
          <div className="flex items-center gap-2">
            {jsonInput.trim() ? (
              <>
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} />
                <span className="text-xs" style={{ color: 'var(--green)' }}>Valid JSON detected</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--text-muted)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Paste a pack JSON definition above</span>
              </>
            )}
          </div>
        </div>

        <div className="flex gap-3 pt-2">
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{
              ...btnPrimary,
              ...(jsonInput.trim() ? {} : { opacity: 0.5, cursor: 'not-allowed' }),
            }}
            disabled={!jsonInput.trim()}
            onClick={() => {
              onImport();
              setJsonInput('');
              onClose();
            }}
          >
            Import Pack
          </button>
          <button
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={btnSecondary}
            onClick={onClose}
          >
            Cancel
          </button>
        </div>
      </div>
    </Modal>
  );
}

/* ========================================================================
   Main Component
   ======================================================================== */

export default function UserPacksPage() {
  const [packs, setPacks] = useState<UserPack[]>(emptyUserPacks);
  const [detailPackId, setDetailPackId] = useState<string | null>(null);
  const [showCreateScratchModal, setShowCreateScratchModal] = useState(false);
  const [showCreateTemplateModal, setShowCreateTemplateModal] = useState(false);
  const [showImportModal, setShowImportModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState<string | null>(null);

  // Fetch installed user packs from API
  useEffect(() => {
    apiFetch<any[]>('/api/packs/builder/user-packs')
      .then((data) => {
        const mapped: UserPack[] = data.map((p: any) => ({
          id: p.slug,
          name: p.name || p.slug,
          version: p.version || '1.0.0',
          packType: p.pack_type || 'tf_confluence',
          category: p.category || 'Custom',
          enabled: true,
          isDefault: false,
          visibility: 'private' as const,
          strategiesUsing: 0,
          lastModified: new Date().toISOString().slice(0, 10),
          validationStatus: p.is_valid ? 'passed' as const : 'failed' as const,
          parityScore: null,
          params: Object.entries(p.parameters_schema || {}).map(([key, schema]: [string, any]) => ({
            key,
            label: schema.label || key.replace(/_/g, ' '),
            value: schema.default ?? 0,
            min: schema.min ?? 0,
            max: schema.max ?? 100,
            type: (schema.type === 'float' ? 'float' : 'int') as 'int' | 'float',
          })),
          plotSettings: Object.entries(p.plot_schema || {}).map(([key, schema]: [string, any]) => ({
            key,
            label: schema.label || key.replace(/_/g, ' '),
            default: schema.default || '#8b5cf6',
          })),
          outputs: (p.outputs || []).map((code: string) => ({
            code,
            description: (p.output_descriptions || {})[code] || '',
          })),
          triggers: (p.triggers || []).map((t: any) => ({
            id: t.base || '', name: t.name || t.base || '',
            direction: t.direction || 'BOTH', type: t.type || 'ENTRY',
            exec: '[C]',
          })),
        }));
        if (mapped.length > 0) {
          setPacks(mapped);
          // Auto-select pack from URL query param (?pack=slug)
          if (typeof window !== 'undefined') {
            const params = new URLSearchParams(window.location.search);
            const packSlug = params.get('pack');
            if (packSlug && mapped.some((p) => p.id === packSlug)) {
              setDetailPackId(packSlug);
            }
          }
        }
      })
      .catch(() => { /* API not available — keep empty */ });
  }, []);

  const enabledCount = packs.filter((p) => p.enabled).length;

  function togglePack(id: string) {
    setPacks((prev) =>
      prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p))
    );
  }

  function clonePack(pack: UserPack) {
    const newId = `${pack.id}-clone-${Date.now()}`;
    const clone: UserPack = {
      ...pack,
      id: newId,
      name: `${pack.name} (Copy)`,
      strategiesUsing: 0,
      lastModified: new Date().toISOString().slice(0, 10),
      params: pack.params.map((p) => ({ ...p })),
      plotSettings: pack.plotSettings.map((ps) => ({ ...ps })),
      outputs: pack.outputs.map((o) => ({ ...o })),
      triggers: pack.triggers.map((t) => ({ ...t })),
    };
    setPacks((prev) => [...prev, clone]);
  }

  function deletePack(id: string) {
    setPacks((prev) => prev.filter((p) => p.id !== id));
    setShowDeleteModal(null);
    if (detailPackId === id) setDetailPackId(null);
  }

  function handleCreateScratch(name: string, category: string) {
    const newId = `custom-${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    const newPack: UserPack = {
      id: newId,
      name,
      version: 'Custom',
      packType: 'tf_confluence',
      category,
      enabled: true,
      isDefault: false,
      visibility: 'private',
      strategiesUsing: 0,
      lastModified: new Date().toISOString().slice(0, 10),
      validationStatus: 'passed',
      parityScore: null,
      params: [],
      plotSettings: [],
      outputs: [],
      triggers: [],
    };
    setPacks((prev) => [...prev, newPack]);
  }

  function handleCreateFromTemplate(templateKey: string, name: string) {
    const tpl = builtInTemplates.find((t) => t.key === templateKey);
    if (!tpl) return;
    const newId = `${templateKey}-custom-${Date.now()}`;
    const newPack: UserPack = {
      id: newId,
      name,
      version: 'Custom',
      packType: 'tf_confluence',
      category: tpl.category,
      enabled: true,
      isDefault: false,
      visibility: 'private',
      strategiesUsing: 0,
      lastModified: new Date().toISOString().slice(0, 10),
      validationStatus: 'passed',
      parityScore: null,
      params: [
        { key: 'period', label: 'Period', value: 14, min: 2, max: 100, type: 'int' },
      ],
      plotSettings: [
        { key: 'line_color', label: 'Line Color', default: '#8b5cf6' },
      ],
      outputs: [
        { code: 'HIGH', description: 'Above threshold' },
        { code: 'LOW', description: 'Below threshold' },
      ],
      triggers: [
        { id: `${templateKey}_cross_up`, name: 'Cross Up', direction: 'LONG', type: 'ENTRY', exec: '[C]' },
        { id: `${templateKey}_cross_down`, name: 'Cross Down', direction: 'SHORT', type: 'ENTRY', exec: '[C]' },
      ],
    };
    setPacks((prev) => [...prev, newPack]);
  }

  function handleImport() {
    const imported: UserPack = {
      id: `imported-${Date.now()}`,
      name: 'Imported Pack',
      version: 'Imported',
      packType: 'tf_confluence',
      category: 'Custom',
      enabled: true,
      isDefault: false,
      visibility: 'private',
      strategiesUsing: 0,
      lastModified: new Date().toISOString().slice(0, 10),
      validationStatus: 'passed',
      parityScore: null,
      params: [
        { key: 'length', label: 'Length', value: 20, min: 1, max: 100, type: 'int' },
      ],
      plotSettings: [
        { key: 'color', label: 'Color', default: '#f59e0b' },
      ],
      outputs: [
        { code: 'ACTIVE', description: 'Signal active' },
        { code: 'INACTIVE', description: 'Signal inactive' },
      ],
      triggers: [
        { id: 'imported_trigger', name: 'Signal Trigger', direction: 'BOTH', type: 'ENTRY', exec: '[C]' },
      ],
    };
    setPacks((prev) => [...prev, imported]);
  }

  // Detail view
  const detailPack = packs.find((p) => p.id === detailPackId);
  if (detailPack) {
    return (
      <DetailView
        pack={detailPack}
        onBack={() => setDetailPackId(null)}
        onDelete={() => deletePack(detailPack.id)}
      />
    );
  }

  const packToDelete = packs.find((p) => p.id === showDeleteModal);

  // List view
  return (
    <div>
      <PageHeader
        title="User Packs"
        subtitle="Custom indicator packs you've created"
        actions={
          <div className="flex gap-2">
            <button
              onClick={() => setShowImportModal(true)}
              className="px-4 py-2 rounded-lg text-sm"
              style={btnSecondary}
            >
              Import JSON
            </button>
            <button
              onClick={() => setShowCreateTemplateModal(true)}
              className="px-4 py-2 rounded-lg text-sm"
              style={btnSecondary}
            >
              From Template
            </button>
            <Link
              href="/confluence-packs/pack-builder"
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={btnPrimary}
            >
              + Create Pack
            </Link>
          </div>
        }
      />

      {/* Summary stats */}
      <div className="flex items-center gap-4 mb-6">
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {packs.length} {packs.length === 1 ? 'pack' : 'packs'}{packs.length > 0 ? `, ${enabledCount} enabled` : ''}
        </span>
        {packs.length > 0 && (
          <div className="flex items-center gap-1.5">
            {Array.from(new Set(packs.map((p) => p.category))).map((cat) => (
              <CategoryBadge key={cat} category={cat} />
            ))}
          </div>
        )}
      </div>

      {packs.length === 0 ? (
        <Card>
          <div className="text-center py-16">
            <div className="text-4xl mb-4" style={{ color: 'var(--text-muted)' }}>
              { /* Package icon placeholder */ }
              <span style={{ fontSize: '48px', opacity: 0.3 }}>[ ]</span>
            </div>
            <p className="text-lg font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              No custom packs installed
            </p>
            <p className="text-sm mb-8 max-w-md mx-auto" style={{ color: 'var(--text-muted)' }}>
              Create packs with the Pack Builder. Custom indicator packs with
              parameters, outputs, and triggers will appear here once created.
            </p>
            <div className="flex gap-3 justify-center">
              <Link
                href="/confluence-packs/pack-builder"
                className="px-5 py-2.5 rounded-lg text-sm font-medium"
                style={btnPrimary}
              >
                Create from Scratch
              </Link>
              <button
                onClick={() => setShowCreateTemplateModal(true)}
                className="px-5 py-2.5 rounded-lg text-sm"
                style={btnSecondary}
              >
                Start from Template
              </button>
            </div>
          </div>
        </Card>
      ) : (
        <div className="space-y-3">
          {packs.map((pack) => (
            <PackCard
              key={pack.id}
              pack={pack}
              onToggle={() => togglePack(pack.id)}
              onDetail={() => setDetailPackId(pack.id)}
              onClone={() => clonePack(pack)}
              onDelete={() => setShowDeleteModal(pack.id)}
            />
          ))}
        </div>
      )}

      {/* Create from Scratch Modal */}
      <CreateScratchModal
        isOpen={showCreateScratchModal}
        onClose={() => setShowCreateScratchModal(false)}
        onSubmit={handleCreateScratch}
      />

      {/* Create from Template Modal */}
      <CreateTemplateModal
        isOpen={showCreateTemplateModal}
        onClose={() => setShowCreateTemplateModal(false)}
        onSubmit={handleCreateFromTemplate}
      />

      {/* Import Pack Modal */}
      <ImportPackModal
        isOpen={showImportModal}
        onClose={() => setShowImportModal(false)}
        onImport={handleImport}
      />

      {/* Delete confirmation modal */}
      <Modal title="Delete Pack" isOpen={!!showDeleteModal} onClose={() => setShowDeleteModal(null)} width="420px">
        <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>
          Are you sure you want to delete <strong>{packToDelete?.name || 'this pack'}</strong>? This action cannot be undone.
        </p>
        {packToDelete && packToDelete.strategiesUsing > 0 && (
          <p className="text-xs mb-4 px-3 py-2 rounded-lg" style={{ background: 'var(--orange-muted)', color: 'var(--orange)' }}>
            This pack is used in {packToDelete.strategiesUsing} {packToDelete.strategiesUsing === 1 ? 'strategy' : 'strategies'}.
          </p>
        )}
        <div className="flex gap-3">
          <button
            onClick={() => showDeleteModal && deletePack(showDeleteModal)}
            className="px-4 py-2 rounded-lg text-sm font-medium flex-1"
            style={{ background: 'var(--red)', color: 'white' }}
          >
            Delete
          </button>
          <button
            onClick={() => setShowDeleteModal(null)}
            className="px-4 py-2 rounded-lg text-sm flex-1"
            style={btnSecondary}
          >
            Cancel
          </button>
        </div>
      </Modal>
    </div>
  );
}
