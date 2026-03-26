'use client';

/**
 * Timeframes — Faithful copy of V5 design with mock data replaced by API hooks.
 * Source: src/app/confluence-packs/timeframes/versions/V5.tsx (192 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useSettings, useSaveSettings } from '@/hooks/queries/useSettings';

/* ========================================================================= */
/* DATA                                                                        */
/* ========================================================================= */

interface TimeframeDef {
  id: string;
  label: string;
  seconds: number;
  barsPerDay: number;
  primaryEnabled: boolean;
  confluenceEnabled: boolean;
  massBuilderEnabled: boolean;
  chartDisplayEnabled: boolean;
  isDefault: boolean;
  isSubMinute: boolean;
  providerSupport: 'polygon' | 'both' | 'streaming_only';
}

const TF_DEFINITIONS: TimeframeDef[] = [
  { id: '5Sec', label: '5s', seconds: 5, barsPerDay: 4680, primaryEnabled: false, confluenceEnabled: false, massBuilderEnabled: false, chartDisplayEnabled: false, isDefault: false, isSubMinute: true, providerSupport: 'polygon' },
  { id: '10Sec', label: '10s', seconds: 10, barsPerDay: 2340, primaryEnabled: false, confluenceEnabled: false, massBuilderEnabled: false, chartDisplayEnabled: false, isDefault: false, isSubMinute: true, providerSupport: 'polygon' },
  { id: '15Sec', label: '15s', seconds: 15, barsPerDay: 1560, primaryEnabled: false, confluenceEnabled: false, massBuilderEnabled: false, chartDisplayEnabled: false, isDefault: false, isSubMinute: true, providerSupport: 'polygon' },
  { id: '30Sec', label: '30s', seconds: 30, barsPerDay: 780, primaryEnabled: false, confluenceEnabled: false, massBuilderEnabled: false, chartDisplayEnabled: false, isDefault: false, isSubMinute: true, providerSupport: 'polygon' },
  { id: '1Min', label: '1m', seconds: 60, barsPerDay: 390, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: true, isSubMinute: false, providerSupport: 'both' },
  { id: '2Min', label: '2m', seconds: 120, barsPerDay: 195, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '3Min', label: '3m', seconds: 180, barsPerDay: 130, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '5Min', label: '5m', seconds: 300, barsPerDay: 78, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '10Min', label: '10m', seconds: 600, barsPerDay: 39, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '15Min', label: '15m', seconds: 900, barsPerDay: 26, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '30Min', label: '30m', seconds: 1800, barsPerDay: 13, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '1Hour', label: '1h', seconds: 3600, barsPerDay: 7, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: true, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '2Hour', label: '2h', seconds: 7200, barsPerDay: 4, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: false, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '4Hour', label: '4h', seconds: 14400, barsPerDay: 2, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: false, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '1Day', label: '1d', seconds: 86400, barsPerDay: 1, primaryEnabled: true, confluenceEnabled: true, massBuilderEnabled: false, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '1Week', label: '1w', seconds: 604800, barsPerDay: 0.2, primaryEnabled: false, confluenceEnabled: false, massBuilderEnabled: false, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
  { id: '1Month', label: '1mo', seconds: 2592000, barsPerDay: 0.05, primaryEnabled: false, confluenceEnabled: false, massBuilderEnabled: false, chartDisplayEnabled: true, isDefault: false, isSubMinute: false, providerSupport: 'both' },
];

const USE_CASES = [
  { key: 'primaryEnabled', label: 'Strategy Primary', desc: 'Available as the main trading timeframe in Strategy Builder' },
  { key: 'confluenceEnabled', label: 'TF Confluence', desc: 'Available as a secondary timeframe for confluence conditions' },
  { key: 'massBuilderEnabled', label: 'Mass Builder', desc: 'Available in Mass Strategy Builder timeframe selection' },
  { key: 'chartDisplayEnabled', label: 'Chart Display', desc: 'Available for chart viewing and data display' },
];

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function TimeframesPage() {
  // --- API hooks (MUST come before any early returns) ---
  const { data: settings, isLoading, error } = useSettings();
  const saveMutation = useSaveSettings();

  const [tfs, setTfs] = useState(TF_DEFINITIONS);
  const [defaultTf, setDefaultTf] = useState('1Min');
  const [initialized, setInitialized] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Initialize from settings API
  useEffect(() => {
    if (settings && !initialized) {
      const enabledTfs: Record<string, Record<string, boolean>> = settings.enabled_timeframes ?? {};
      const newDefaultTf = settings.default_timeframe ?? '1Min';
      setDefaultTf(newDefaultTf);
      if (Object.keys(enabledTfs).length > 0) {
        setTfs(TF_DEFINITIONS.map((tf) => {
          const overrides = enabledTfs[tf.id];
          if (!overrides) return tf;
          return {
            ...tf,
            primaryEnabled: overrides.primaryEnabled ?? tf.primaryEnabled,
            confluenceEnabled: overrides.confluenceEnabled ?? tf.confluenceEnabled,
            massBuilderEnabled: overrides.massBuilderEnabled ?? tf.massBuilderEnabled,
            chartDisplayEnabled: overrides.chartDisplayEnabled ?? tf.chartDisplayEnabled,
          };
        }));
      }
      setInitialized(true);
    }
  }, [settings, initialized]);

  const toggleUseCase = (tfId: string, key: keyof TimeframeDef) => {
    setTfs((prev) => prev.map((tf) =>
      tf.id === tfId ? { ...tf, [key]: !tf[key] } : tf
    ));
    setHasChanges(true);
  };

  const enabledCount = (key: keyof TimeframeDef) => tfs.filter((tf) => tf[key]).length;

  const handleSave = () => {
    const enabledTimeframes: Record<string, Record<string, boolean>> = {};
    for (const tf of tfs) {
      enabledTimeframes[tf.id] = {
        primaryEnabled: tf.primaryEnabled,
        confluenceEnabled: tf.confluenceEnabled,
        massBuilderEnabled: tf.massBuilderEnabled,
        chartDisplayEnabled: tf.chartDisplayEnabled,
      };
    }
    saveMutation.mutate({ ...settings, enabled_timeframes: enabledTimeframes, default_timeframe: defaultTf });
    setHasChanges(false);
  };

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Timeframes" subtitle="Loading..." />
        <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading timeframe settings...</p></Card>
      </div>
    );
  }
  if (error) {
    return (
      <div>
        <PageHeader title="Timeframes" subtitle="Error loading data" />
        <Card><p className="text-sm py-8 text-center" style={{ color: 'var(--red)' }}>Failed to load settings: {String(error)}</p></Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Timeframes"
        subtitle="Configure which timeframes are available across the application"
        actions={
          <button
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white', border: 'none', cursor: hasChanges ? 'pointer' : 'default', opacity: hasChanges ? 1 : 0.5 }}
            onClick={handleSave}
            disabled={!hasChanges || saveMutation.isPending}
          >
            {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
          </button>
        }
      />

      {/* Default TF selector */}
      <Card className="mb-4">
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
            {tfs.filter((tf) => tf.primaryEnabled).map((tf) => (
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
              <span className="text-xs font-bold" style={{ color: 'var(--accent)' }}>{enabledCount(uc.key as keyof TimeframeDef)}</span>
            </div>
            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{uc.desc}</p>
          </div>
        ))}
      </div>

      {/* Main grid */}
      <Card>
        <div style={{ overflowX: 'auto' }}>
          <table className="w-full text-sm" style={{ borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th className="text-left py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)', position: 'sticky', left: 0, zIndex: 1 }}>Timeframe</th>
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>Label</th>
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>Bars/Day</th>
                {USE_CASES.map((uc) => (
                  <th key={uc.key} className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                    {uc.label}
                  </th>
                ))}
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>Default</th>
                <th className="text-center py-2 px-3 text-xs font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>Provider</th>
              </tr>
            </thead>
            <tbody>
              {tfs.map((tf) => (
                <tr key={tf.id} style={{ borderBottom: '1px solid var(--border)', opacity: tf.isSubMinute ? 0.6 : 1 }}>
                  <td className="py-2.5 px-3 font-medium" style={{ position: 'sticky', left: 0, background: 'var(--bg-card)', zIndex: 1 }}>
                    <div className="flex items-center gap-2">
                      {tf.id}
                      {tf.isSubMinute && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>sub-min</span>
                      )}
                    </div>
                  </td>
                  <td className="text-center py-2.5 px-3 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{tf.label}</td>
                  <td className="text-center py-2.5 px-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                    {tf.barsPerDay >= 1 ? Math.round(tf.barsPerDay) : tf.barsPerDay.toFixed(2)}
                  </td>
                  {USE_CASES.map((uc) => (
                    <td key={uc.key} className="text-center py-2.5 px-3">
                      <input
                        type="checkbox"
                        checked={tf[uc.key as keyof TimeframeDef] as boolean}
                        onChange={() => toggleUseCase(tf.id, uc.key as keyof TimeframeDef)}
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
                      disabled={!tf.primaryEnabled}
                      className="w-4 h-4 cursor-pointer"
                      style={{ accentColor: 'var(--accent)' }}
                    />
                  </td>
                  <td className="text-center py-2.5 px-3">
                    <span className="text-[10px] px-1.5 py-0.5 rounded" style={{
                      background: tf.providerSupport === 'both' ? 'var(--green-muted)' : tf.providerSupport === 'polygon' ? 'var(--accent-muted)' : 'var(--bg-input)',
                      color: tf.providerSupport === 'both' ? 'var(--green)' : tf.providerSupport === 'polygon' ? 'var(--accent)' : 'var(--text-muted)',
                    }}>
                      {tf.providerSupport === 'both' ? 'All' : tf.providerSupport === 'polygon' ? 'Polygon' : 'Stream'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Info note */}
      <p className="text-xs mt-3" style={{ color: 'var(--text-muted)' }}>
        Sub-minute timeframes (5s-30s) require Polygon.io data feed and are not available for confluence resampling from REST data.
        Weekly and monthly timeframes are display-only and cannot be used as primary strategy timeframes.
      </p>
    </div>
  );
}
