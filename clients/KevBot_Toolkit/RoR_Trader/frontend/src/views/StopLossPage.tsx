'use client';

/**
 * Stop Loss Packs — Clean API-first page.
 *
 * Visual design adapted from V1 (stop-loss/versions/V1.tsx), data layer built
 * around actual Supabase API response shapes. No mock data, no fallbacks.
 *
 * Filters risk management packs to show only stop-related templates
 * (those with stop_method on the template).
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useRiskManagementPacks, useRiskManagementTemplates } from '@/hooks/queries/usePacks';
import { useSaveRiskManagementPacks } from '@/hooks/mutations/usePackMutations';
import type { RiskManagementPackDTO, RMTemplateDTO } from '@/hooks/queries/usePacks';

// ---------------------------------------------------------------------------
// Style constants (from V1)
// ---------------------------------------------------------------------------

const tagColors: Record<string, { color: string; bg: string }> = {
  Volatility: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Fixed: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Structure: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Trailing: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function TagBadge({ tag }: { tag: string }) {
  const style = tagColors[tag] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span
      className="text-[10px] font-medium px-1.5 py-0.5 rounded"
      style={{ color: style.color, background: style.bg }}
    >
      {tag}
    </span>
  );
}

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onChange(); }}
      className="w-9 h-5 rounded-full relative flex-shrink-0 transition-colors"
      style={{
        background: enabled ? 'var(--accent)' : 'var(--bg-input)',
        border: enabled ? 'none' : '1px solid var(--border)',
      }}
    >
      <div
        className="w-3.5 h-3.5 rounded-full absolute transition-all"
        style={{
          background: enabled ? 'white' : 'var(--text-muted)',
          top: '3px',
          left: enabled ? '19px' : '3px',
        }}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Known stop-related template keys */
const STOP_TEMPLATES = new Set([
  'atr', 'fixed_dollar', 'percentage', 'swing', 'atr_trailing', 'breakeven',
  'atr_stop', 'fixed_dollar_stop', 'percentage_stop', 'swing_stop', 'breakeven_stop',
]);

function isStopTemplate(templateKey: string, template?: RMTemplateDTO): boolean {
  // Primary check: template has stop_method
  if (template?.stop_method) return true;
  // Fallback: known stop template names
  if (STOP_TEMPLATES.has(templateKey)) return true;
  // Exclude templates that are target-only
  if (template?.target_method && !template?.stop_method) return false;
  return false;
}

function formatParamSummary(
  parameters: Record<string, unknown>,
  schema: Record<string, { type: string; default: unknown; label: string }> | undefined,
): string {
  if (!schema) return '';
  return Object.entries(schema)
    .map(([key, s]) => {
      const val = parameters[key] ?? s.default;
      return `${s.label}: ${val}`;
    })
    .join(', ');
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function StopLossPage() {
  const { data: allPacks, isLoading: packsLoading, error: packsError } = useRiskManagementPacks();
  const { data: templates, isLoading: templatesLoading, error: templatesError } = useRiskManagementTemplates();
  const saveMutation = useSaveRiskManagementPacks();

  const isLoading = packsLoading || templatesLoading;
  const error = packsError || templatesError;

  // Local state for toggles (optimistic UI)
  const [localToggles, setLocalToggles] = useState<Record<string, boolean>>({});
  const [search, setSearch] = useState('');

  // Filter to stop-only packs
  const stopPacks = useMemo(() => {
    if (!allPacks || !templates) return [];
    return allPacks.filter((p) => isStopTemplate(p.base_template, templates[p.base_template]));
  }, [allPacks, templates]);

  // Search filter
  const filtered = useMemo(() => {
    if (!search.trim()) return stopPacks;
    const q = search.toLowerCase();
    return stopPacks.filter((p) => {
      const tmpl = templates?.[p.base_template];
      const name = tmpl?.name || p.base_template;
      return (
        name.toLowerCase().includes(q) ||
        p.version.toLowerCase().includes(q) ||
        p.base_template.toLowerCase().includes(q) ||
        (p.description || '').toLowerCase().includes(q)
      );
    });
  }, [stopPacks, templates, search]);

  // Group by base_template
  const groupedPacks = useMemo(() => {
    const map = new Map<string, RiskManagementPackDTO[]>();
    for (const p of filtered) {
      const key = p.base_template;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(p);
    }
    Array.from(map.values()).forEach((arr) => {
      arr.sort((a: RiskManagementPackDTO, b: RiskManagementPackDTO) => (b.is_default ? 1 : 0) - (a.is_default ? 1 : 0));
    });
    return map;
  }, [filtered]);

  const [expandedTemplates, setExpandedTemplates] = useState<Set<string>>(new Set());

  function handleToggle(pack: RiskManagementPackDTO) {
    const currentEnabled = localToggles[pack.id] ?? pack.enabled;
    const newEnabled = !currentEnabled;
    setLocalToggles((prev) => ({ ...prev, [pack.id]: newEnabled }));

    // Save ALL risk management packs (not just stop packs) so we don't lose targets
    if (allPacks) {
      const updated = allPacks.map((p) =>
        p.id === pack.id ? { ...p, enabled: newEnabled } : p,
      );
      saveMutation.mutate(updated);
    }
  }

  function isEnabled(pack: RiskManagementPackDTO): boolean {
    return localToggles[pack.id] ?? pack.enabled;
  }

  // ---------------------------------------------------------------------------
  // Loading state
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Stop Loss Packs" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <div className="animate-pulse flex items-center gap-3">
                <div className="w-9 h-5 rounded-full" style={{ background: 'var(--border)' }} />
                <div className="flex-1 space-y-2">
                  <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                  <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Error state
  // ---------------------------------------------------------------------------

  if (error) {
    return (
      <div>
        <PageHeader title="Stop Loss Packs" subtitle="Error loading data" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load stop loss packs. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const enabledCount = stopPacks.filter((p) => isEnabled(p)).length;
  const totalCount = stopPacks.length;

  return (
    <div>
      <PageHeader
        title="Stop Loss Packs"
        subtitle={`${enabledCount} of ${totalCount} enabled`}
        actions={
          <span
            className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
            style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
            Live
          </span>
        }
      />

      {/* Search */}
      <div className="flex flex-wrap gap-2 mb-4 mt-4">
        <input
          type="text"
          placeholder="Search stop loss packs..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 rounded-lg text-sm flex-1 min-w-[180px]"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
          }}
        />
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {totalCount === 0
                ? 'No stop loss packs configured yet.'
                : 'No packs match the current search.'}
            </p>
          </div>
        </Card>
      )}

      {/* Pack cards */}
      <div className="space-y-2">
        {Array.from(groupedPacks.entries()).map(([templateKey, packList]) => {
          const tmpl = templates?.[templateKey];
          const defaultPack = packList.find((p) => p.is_default);
          const variations = packList.filter((p) => !p.is_default);
          const hasVariations = variations.length > 0;
          const isExpanded = expandedTemplates.has(templateKey);

          return (
            <div key={templateKey}>
              {defaultPack && (
                <StopPackCard
                  pack={defaultPack}
                  template={tmpl}
                  enabled={isEnabled(defaultPack)}
                  onToggle={() => handleToggle(defaultPack)}
                  hasVariations={hasVariations}
                  isExpanded={isExpanded}
                  onToggleExpand={() => {
                    setExpandedTemplates((prev) => {
                      const next = new Set(prev);
                      if (next.has(templateKey)) next.delete(templateKey);
                      else next.add(templateKey);
                      return next;
                    });
                  }}
                />
              )}

              {isExpanded && variations.map((v) => (
                <StopPackCard
                  key={v.id}
                  pack={v}
                  template={tmpl}
                  enabled={isEnabled(v)}
                  onToggle={() => handleToggle(v)}
                  isVariation
                />
              ))}
            </div>
          );
        })}
      </div>

      {saveMutation.isPending && (
        <div
          className="fixed bottom-4 right-4 px-4 py-2 rounded-lg text-sm"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          Saving...
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pack Card
// ---------------------------------------------------------------------------

function StopPackCard({
  pack,
  template,
  enabled,
  onToggle,
  hasVariations,
  isExpanded,
  onToggleExpand,
  isVariation,
}: {
  pack: RiskManagementPackDTO;
  template?: RMTemplateDTO;
  enabled: boolean;
  onToggle: () => void;
  hasVariations?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  isVariation?: boolean;
}) {
  const name = template?.name || pack.base_template;
  const category = template?.category || 'Other';
  const paramSummary = formatParamSummary(pack.parameters, template?.parameters_schema);
  const description = template?.description || pack.description;

  return (
    <div
      className="flex items-center gap-3 px-4 py-3 rounded-xl border transition-colors mb-2"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
        boxShadow: 'var(--card-shadow)',
        backdropFilter: 'var(--card-backdrop)',
        WebkitBackdropFilter: 'var(--card-backdrop)',
        opacity: enabled ? 1 : 0.6,
        marginLeft: isVariation ? 24 : 0,
      }}
    >
      {hasVariations ? (
        <button
          onClick={(e) => { e.stopPropagation(); onToggleExpand?.(); }}
          className="w-5 h-5 rounded flex items-center justify-center text-xs flex-shrink-0 transition-transform"
          style={{ color: 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
        >
          {'\u25B6'}
        </button>
      ) : isVariation ? (
        <div className="w-5 flex-shrink-0" />
      ) : null}

      <Toggle enabled={enabled} onChange={onToggle} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
            {name}
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{
              color: pack.is_default ? 'var(--text-muted)' : 'var(--text-secondary)',
              background: 'var(--bg-input)',
            }}
          >
            {pack.version}
          </span>
          <TagBadge tag={category} />
        </div>
        <p className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
          {paramSummary}
        </p>
        {description && (
          <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
            {description}
          </p>
        )}
      </div>
    </div>
  );
}
