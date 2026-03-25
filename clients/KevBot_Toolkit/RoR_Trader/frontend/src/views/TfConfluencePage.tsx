'use client';

/**
 * TF Confluence Groups — Clean API-first page.
 *
 * Visual design adapted from V5 (versions/V5.tsx), data layer built
 * around actual Supabase API response shapes. No mock data, no fallbacks.
 *
 * V5 features:
 * - Nested variations indent under parent with expand/collapse chevron
 * - Parameter lock icon on saved packs
 * - "Save as Variation" button in header for drafts
 * - Card row 1: name, version, tags, state count (tooltip), trigger count (tooltip)
 * - Card row 2: locked params in monospace shorthand | exec type settings
 * - Execution type badges [C]/[L]/[LC]/[CC]
 * - "Create New Template" button linking to Pack Builder
 */

import { useState, useMemo } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import { useConfluenceGroups, useConfluenceTemplates } from '@/hooks/queries/usePacks';
import { useSaveConfluenceGroups } from '@/hooks/mutations/usePackMutations';
import type { ConfluenceGroupDTO, ConfluenceTemplateDTO } from '@/hooks/queries/usePacks';

// ---------------------------------------------------------------------------
// Style constants (from V5)
// ---------------------------------------------------------------------------

const EXEC_BADGE_COLOR = '#2196F3';

const tagColors: Record<string, { color: string; bg: string }> = {
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Momentum: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Volume: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Volatility: { color: 'var(--red)', bg: 'var(--red-muted)' },
  Trend: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Exit: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
};

const selectStyle: React.CSSProperties = {
  padding: '6px 10px',
  borderRadius: '6px',
  border: '1px solid var(--border)',
  background: 'var(--bg-input)',
  color: 'var(--text-primary)',
  fontSize: '12px',
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

function ExecBadge({ exec }: { exec: string }) {
  return (
    <span
      className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded-full"
      style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}
    >
      [{exec}]
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

function LockIcon({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ color: 'var(--orange)' }}>
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0110 0v4" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build monospace shorthand for params: "Short:9 Mid:21 Long:200" */
function formatParamShorthand(
  parameters: Record<string, unknown>,
  schema: Record<string, { type: string; default: number | string; label: string }> | undefined,
): { key: string; short: string; val: string }[] {
  if (!schema) return [];
  return Object.entries(schema).map(([key, s]) => {
    const val = parameters[key] ?? s.default;
    // Shorten labels for compact display
    const short = s.label
      .replace(' Period', '')
      .replace(' Threshold', '')
      .replace(' Multiplier', 'x')
      .replace('Inner Band ', '')
      .replace('Outer Band ', '')
      .replace('Bar Count', 'Bars')
      .trim();
    return { key, short, val: String(val) };
  });
}

/** Build tooltip text listing all output states */
function buildOutputsTooltip(outputs: Record<string, string> | undefined): string {
  if (!outputs) return 'No states';
  return Object.entries(outputs)
    .map(([code, desc]) => `${code} -- ${desc}`)
    .join('\n');
}

/** Build tooltip text listing all triggers */
function buildTriggersTooltip(triggers: ConfluenceTemplateDTO['triggers'] | undefined): string {
  if (!triggers || triggers.length === 0) return 'No triggers';
  return triggers.map((t) => `${t.name} (${t.direction})`).join('\n');
}

/** Derive which exec types are represented in triggers */
function getExecTypes(triggers: ConfluenceTemplateDTO['triggers'] | undefined): string[] {
  if (!triggers) return [];
  const types = new Set<string>();
  for (const t of triggers) {
    if (t.execution) {
      // execution field might be "C", "L", etc.
      types.add(t.execution.toUpperCase());
    }
  }
  // If no execution field, default to [C]
  return types.size > 0 ? Array.from(types).sort() : ['C'];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TfConfluencePage() {
  const { data: groups, isLoading: groupsLoading, error: groupsError } = useConfluenceGroups();
  const { data: templates, isLoading: templatesLoading, error: templatesError } = useConfluenceTemplates();
  const saveMutation = useSaveConfluenceGroups();

  const isLoading = groupsLoading || templatesLoading;
  const error = groupsError || templatesError;

  // Local state for toggles (optimistic UI)
  const [localToggles, setLocalToggles] = useState<Record<string, boolean>>({});

  // Filter / search state
  const [search, setSearch] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('All');

  // Expand/collapse for variation nesting
  const [expandedTemplates, setExpandedTemplates] = useState<Set<string>>(new Set());

  // Categories derived from templates
  const allCategories = useMemo(() => {
    if (!templates) return [];
    return Array.from(new Set(Object.values(templates).map((t) => t.category))).sort();
  }, [templates]);

  // Filtered groups
  const filtered = useMemo(() => {
    if (!groups || !templates) return [];
    let result = [...groups];

    // Category filter
    if (categoryFilter !== 'All') {
      result = result.filter((g) => {
        const tmpl = templates[g.base_template];
        return tmpl && tmpl.category === categoryFilter;
      });
    }

    // Search filter
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter((g) => {
        const tmpl = templates[g.base_template];
        const name = tmpl?.name || g.base_template;
        return (
          name.toLowerCase().includes(q) ||
          g.version.toLowerCase().includes(q) ||
          g.base_template.toLowerCase().includes(q) ||
          (g.description || '').toLowerCase().includes(q)
        );
      });
    }

    return result;
  }, [groups, templates, categoryFilter, search]);

  // Group by base_template to show defaults with their variations
  const groupedPacks = useMemo(() => {
    const map = new Map<string, ConfluenceGroupDTO[]>();
    for (const g of filtered) {
      const key = g.base_template;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(g);
    }
    // Sort: defaults first within each group
    Array.from(map.values()).forEach((arr) => {
      arr.sort((a: ConfluenceGroupDTO, b: ConfluenceGroupDTO) => (b.is_default ? 1 : 0) - (a.is_default ? 1 : 0));
    });
    return map;
  }, [filtered]);

  // Detect if any group is a draft (not saved yet / not is_default and not yet saved)
  // For API data, all groups from the server are saved. Drafts would only exist
  // if we add local creation UI. For now we expose the button pattern.

  // Toggle handler
  function handleToggle(group: ConfluenceGroupDTO) {
    const currentEnabled = localToggles[group.id] ?? group.enabled;
    const newEnabled = !currentEnabled;
    setLocalToggles((prev) => ({ ...prev, [group.id]: newEnabled }));

    // Save to API
    if (groups) {
      const updated = groups.map((g) =>
        g.id === group.id ? { ...g, enabled: newEnabled } : g,
      );
      saveMutation.mutate(updated);
    }
  }

  function isEnabled(group: ConfluenceGroupDTO): boolean {
    return localToggles[group.id] ?? group.enabled;
  }

  // ---------------------------------------------------------------------------
  // Loading state
  // ---------------------------------------------------------------------------

  if (isLoading) {
    return (
      <div>
        <PageHeader title="TF Confluence Groups" subtitle="Loading..." />
        <div className="space-y-3 mt-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <Card key={i}>
              <div className="animate-pulse flex items-center gap-3">
                <div className="w-9 h-5 rounded-full" style={{ background: 'var(--border)' }} />
                <div className="flex-1 space-y-2">
                  <div className="h-4 rounded w-1/3" style={{ background: 'var(--border)' }} />
                  <div className="h-3 rounded w-2/3" style={{ background: 'var(--border)' }} />
                </div>
                <div className="h-7 w-16 rounded" style={{ background: 'var(--border)' }} />
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
        <PageHeader title="TF Confluence Groups" subtitle="Error loading data" />
        <Card>
          <div className="text-center py-8" style={{ color: 'var(--red)' }}>
            Failed to load confluence groups. Check your connection and try again.
          </div>
        </Card>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const enabledCount = groups?.filter((g) => isEnabled(g)).length ?? 0;
  const totalCount = groups?.length ?? 0;

  return (
    <div>
      <PageHeader
        title="TF Confluence Groups"
        subtitle={`${enabledCount} of ${totalCount} enabled`}
        actions={
          <div className="flex items-center gap-2">
            <Link
              href="/confluence-packs/tf-confluence/builder"
              className="px-3 py-1.5 rounded-lg text-xs font-medium"
              style={{
                background: 'var(--accent)',
                color: 'white',
                textDecoration: 'none',
                whiteSpace: 'nowrap',
              }}
            >
              Create New Template
            </Link>
            <span
              className="text-xs px-2 py-1 rounded-full flex items-center gap-1.5"
              style={{ background: 'var(--green)' + '15', color: 'var(--green)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--green)' }} />
              Live
            </span>
          </div>
        }
      />

      {/* Filter row */}
      <div className="flex flex-wrap gap-2 mb-4 mt-4">
        <input
          type="text"
          placeholder="Search packs..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 rounded-lg text-sm flex-1 min-w-[180px]"
          style={{
            background: 'var(--bg-input)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
          }}
        />
        <select
          style={selectStyle}
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value)}
        >
          <option value="All">All Categories</option>
          {allCategories.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <Card>
          <div className="text-center py-12">
            <p className="text-lg mb-2" style={{ color: 'var(--text-secondary)' }}>
              {totalCount === 0
                ? 'No confluence groups configured yet.'
                : 'No groups match the current filters.'}
            </p>
            {totalCount === 0 && (
              <Link
                href="/confluence-packs/tf-confluence/builder"
                className="text-sm"
                style={{ color: 'var(--accent)', textDecoration: 'none' }}
              >
                Create your first template
              </Link>
            )}
          </div>
        </Card>
      )}

      {/* Pack cards */}
      <div className="space-y-2">
        {Array.from(groupedPacks.entries()).map(([templateKey, packs]) => {
          const tmpl = templates?.[templateKey];
          const defaultPack = packs.find((p) => p.is_default);
          const variations = packs.filter((p) => !p.is_default);
          const hasVariations = variations.length > 0;
          const isExpanded = expandedTemplates.has(templateKey);

          return (
            <div key={templateKey}>
              {/* Default pack card */}
              {defaultPack && (
                <PackCard
                  group={defaultPack}
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

              {/* Variation cards (indented, only when expanded) */}
              {isExpanded && variations.map((v) => (
                <PackCard
                  key={v.id}
                  group={v}
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

      {/* Save indicator */}
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

function PackCard({
  group,
  template,
  enabled,
  onToggle,
  hasVariations,
  isExpanded,
  onToggleExpand,
  isVariation,
}: {
  group: ConfluenceGroupDTO;
  template?: ConfluenceTemplateDTO;
  enabled: boolean;
  onToggle: () => void;
  hasVariations?: boolean;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
  isVariation?: boolean;
}) {
  const name = template?.name || group.base_template;
  const category = template?.category || 'Other';
  const outputCount = template?.outputs ? Object.keys(template.outputs).length : 0;
  const triggerCount = template?.triggers?.length || 0;
  const paramItems = formatParamShorthand(group.parameters, template?.parameters_schema);
  const execTypes = getExecTypes(template?.triggers);

  // All packs from the API are saved; is_default packs are system-saved.
  // Parameters are locked for all saved packs.
  const isSaved = true; // All API packs are persisted

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
      {/* Expand chevron for defaults with variations */}
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

      {/* Enable/disable toggle */}
      <Toggle enabled={enabled} onChange={onToggle} />

      {/* Pack info */}
      <div className="flex-1 min-w-0">
        {/* Row 1: Name, version, tags, state count (tooltip), trigger count (tooltip) */}
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
            {name}
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded"
            style={{
              color: group.is_default ? 'var(--text-muted)' : 'var(--text-secondary)',
              background: 'var(--bg-input)',
            }}
          >
            {group.version}
          </span>
          <TagBadge tag={category} />
          {isSaved && <LockIcon size={10} />}
          <span
            className="text-[10px] cursor-default"
            style={{ color: 'var(--text-muted)' }}
            title={buildOutputsTooltip(template?.outputs)}
          >
            {outputCount} states
          </span>
          <span
            className="text-[10px] cursor-default"
            style={{ color: 'var(--text-muted)' }}
            title={buildTriggersTooltip(template?.triggers)}
          >
            {triggerCount} trigger{triggerCount !== 1 ? 's' : ''}
          </span>
        </div>

        {/* Row 2: Locked params in monospace shorthand | exec type settings */}
        <div className="flex items-center gap-x-3 gap-y-0.5 flex-wrap text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
          {paramItems.map((p) => (
            <span key={p.key}>
              {p.short}:<span style={{ color: 'var(--text-secondary)' }}>{p.val}</span>
            </span>
          ))}
          {paramItems.length > 0 && execTypes.length > 0 && (
            <span style={{ color: 'var(--border)' }}>|</span>
          )}
          {execTypes.map((et) => (
            <ExecBadge key={et} exec={et} />
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 flex-shrink-0">
        {!group.is_default && (
          <button
            className="px-3 py-1.5 rounded text-xs font-medium"
            style={{ background: 'var(--accent-muted)', color: 'var(--accent)', whiteSpace: 'nowrap' }}
            title="Save as a new variation with these parameters"
          >
            Save as Variation
          </button>
        )}
      </div>
    </div>
  );
}
