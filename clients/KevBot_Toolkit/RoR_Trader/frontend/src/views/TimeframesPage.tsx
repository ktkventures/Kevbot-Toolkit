'use client';

/**
 * Timeframes — Clean API-first page.
 *
 * Visual design derived from V5 (versions/V5.tsx), data layer built
 * around the settings API endpoint. No mock data.
 */

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useSettings, useSaveSettings } from '@/hooks/queries/useSettings';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TimeframeDef {
  id: string;
  label: string;
  seconds: number;
  bars_per_day: number;
  primary_enabled: boolean;
  confluence_enabled: boolean;
  mass_builder_enabled: boolean;
  chart_display_enabled: boolean;
  is_sub_minute: boolean;
}

const USE_CASES = [
  { key: 'primary_enabled', label: 'Strategy Primary' },
  { key: 'confluence_enabled', label: 'TF Confluence' },
  { key: 'mass_builder_enabled', label: 'Mass Builder' },
  { key: 'chart_display_enabled', label: 'Chart Display' },
];

// Fallback timeframes if settings don't include them yet
const DEFAULT_TIMEFRAMES: TimeframeDef[] = [
  { id: '1Min', label: '1m', seconds: 60, bars_per_day: 390, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '2Min', label: '2m', seconds: 120, bars_per_day: 195, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '3Min', label: '3m', seconds: 180, bars_per_day: 130, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '5Min', label: '5m', seconds: 300, bars_per_day: 78, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '10Min', label: '10m', seconds: 600, bars_per_day: 39, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '15Min', label: '15m', seconds: 900, bars_per_day: 26, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '30Min', label: '30m', seconds: 1800, bars_per_day: 13, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '1Hour', label: '1h', seconds: 3600, bars_per_day: 7, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: true, chart_display_enabled: true, is_sub_minute: false },
  { id: '4Hour', label: '4h', seconds: 14400, bars_per_day: 2, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: false, chart_display_enabled: true, is_sub_minute: false },
  { id: '1Day', label: '1d', seconds: 86400, bars_per_day: 1, primary_enabled: true, confluence_enabled: true, mass_builder_enabled: false, chart_display_enabled: true, is_sub_minute: false },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TimeframesPage() {
  const { data: settings, isLoading, error } = useSettings();
  const saveMutation = useSaveSettings();

  const [tfs, setTfs] = useState<TimeframeDef[]>(DEFAULT_TIMEFRAMES);
  const [defaultTf, setDefaultTf] = useState('1Min');
  const [hasChanges, setHasChanges] = useState(false);

  // Sync from API
  useEffect(() => {
    if (settings?.timeframes) {
      setTfs(settings.timeframes);
      setHasChanges(false);
    }
    if (settings?.default_timeframe) {
      setDefaultTf(settings.default_timeframe);
    }
  }, [settings]);

  const toggleUseCase = (tfId: string, key: string) => {
    setTfs((prev) => prev.map((tf) =>
      tf.id === tfId ? { ...tf, [key]: !(tf as unknown as Record<string, unknown>)[key] } : tf
    ));
    setHasChanges(true);
  };

  const enabledCount = (key: string) => tfs.filter((tf) => (tf as unknown as Record<string, unknown>)[key]).length;

  const handleSave = () => {
    saveMutation.mutate({
      ...settings,
      timeframes: tfs,
      default_timeframe: defaultTf,
    });
    setHasChanges(false);
  };

  // ---------------------------------------------------------------------------
  // Loading / Error
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Timeframes" subtitle="Loading..." />
        <Card className="mt-4">
          <div className="animate-pulse space-y-3">
            <div className="h-4 rounded w-1/4" style={{ background: 'var(--border)' }} />
            <div className="h-40 rounded" style={{ background: 'var(--bg-input)' }} />
          </div>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <PageHeader title="Timeframes" subtitle="Error" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load timeframe settings. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Timeframes"
        subtitle="Configure which timeframes are available across the application"
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

      {/* Default TF selector */}
      <Card className="mb-4 mt-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Default Timeframe</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              The primary timeframe that loads by default in Strategy Builder, charts, and data views.
            </p>
          </div>
          <select
            value={defaultTf}
            onChange={(e) => { setDefaultTf(e.target.value); setHasChanges(true); }}
            className="px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          >
            {tfs.filter((tf) => tf.primary_enabled).map((tf) => (
              <option key={tf.id} value={tf.id}>{tf.id} ({tf.label})</option>
            ))}
          </select>
        </div>
      </Card>

      {/* Use case legend */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {USE_CASES.map((uc) => (
          <div key={uc.key} className="rounded-lg p-3" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs font-medium">{uc.label}</p>
              <span className="text-xs font-bold" style={{ color: 'var(--accent)' }}>{enabledCount(uc.key)}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Main grid */}
      <Card>
        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                  Timeframe
                </th>
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                  Label
                </th>
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                  Bars/Day
                </th>
                {USE_CASES.map((uc) => (
                  <th key={uc.key} className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                    {uc.label}
                  </th>
                ))}
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                  Default
                </th>
              </tr>
            </thead>
            <tbody>
              {tfs.map((tf) => (
                <tr key={tf.id} style={{ borderBottom: '1px solid var(--border)', opacity: tf.is_sub_minute ? 0.6 : 1 }}>
                  <td className="py-2.5 px-3 font-medium">
                    <div className="flex items-center gap-2">
                      {tf.id}
                      {tf.is_sub_minute && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--orange, #FF9800)20', color: 'var(--orange, #FF9800)' }}>
                          sub-min
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="text-center py-2.5 px-3 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
                    {tf.label}
                  </td>
                  <td className="text-center py-2.5 px-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                    {tf.bars_per_day >= 1 ? Math.round(tf.bars_per_day) : tf.bars_per_day.toFixed(2)}
                  </td>
                  {USE_CASES.map((uc) => (
                    <td key={uc.key} className="text-center py-2.5 px-3">
                      <input
                        type="checkbox"
                        checked={(tf as unknown as Record<string, unknown>)[uc.key] as boolean}
                        onChange={() => toggleUseCase(tf.id, uc.key)}
                        className="w-4 h-4 rounded cursor-pointer"
                        style={{ accentColor: 'var(--accent)' }}
                      />
                    </td>
                  ))}
                  <td className="text-center py-2.5 px-3">
                    <input
                      type="radio"
                      name="defaultTf"
                      checked={defaultTf === tf.id}
                      onChange={() => { setDefaultTf(tf.id); setHasChanges(true); }}
                      disabled={!tf.primary_enabled}
                      className="w-4 h-4 cursor-pointer"
                      style={{ accentColor: 'var(--accent)' }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Save status */}
      {saveMutation.isSuccess && (
        <p className="text-xs mt-3" style={{ color: 'var(--green)' }}>Timeframe settings saved.</p>
      )}
    </div>
  );
}
