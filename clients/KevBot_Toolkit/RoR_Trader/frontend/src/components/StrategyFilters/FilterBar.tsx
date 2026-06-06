/**
 * Filter UI for Strategy Health views.
 *
 * Inline chips show active filters. Click "Filters" to open a popover
 * with all the multi-select pickers. Preset chips at the right.
 */
'use client';

import { useMemo, useState } from 'react';
import {
  type FilterState,
  type StrategyMeta,
  type StrategyType,
  PRESETS,
  EMPTY_FILTERS,
  isFilterEmpty,
} from './types';
import { useFilterState } from './useFilterState';

// Friendly display names for the strategy_origin dropdown. Raw DB values
// are kept as-is to remain URL-bookmarkable; only the visible label changes.
// See docs / memory for the full origin taxonomy.
const ORIGIN_LABELS: Record<string, string> = {
  standard_mass_builder: 'Mass Builder',
  standard_mass_builder_legacy: 'Mass Builder (legacy)',
  standard_pack_canary: 'Pack Canary',
  standard_ai_sop: 'AI / SOP',
  migration: 'Migration (copy)',
  webhook_inbound: 'Webhook inbound',
  standard: 'Standard (uncategorized)',
};
const originLabel = (v: string): string => ORIGIN_LABELS[v] ?? v;

const PILL_STYLE: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '4px 10px',
  borderRadius: 14,
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  fontSize: 11,
  whiteSpace: 'nowrap',
};

const ACTIVE_PILL_STYLE: React.CSSProperties = {
  ...PILL_STYLE,
  background: 'var(--blue-muted, #1e40af33)',
  color: 'var(--blue, #3b82f6)',
  borderColor: 'var(--blue, #3b82f6)',
};

const CLEAR_BTN_STYLE: React.CSSProperties = {
  marginLeft: 4,
  cursor: 'pointer',
  color: 'var(--text-muted)',
  fontSize: 14,
  lineHeight: 1,
};

interface MultiSelectProps {
  label: string;
  options: { value: string; label: string }[];
  selected: string[];
  onChange: (next: string[]) => void;
}

function MultiSelect({ label, options, selected, onChange }: MultiSelectProps) {
  const [query, setQuery] = useState('');
  const filtered = useMemo(() => {
    if (!query) return options;
    const q = query.toLowerCase();
    return options.filter(
      (o) =>
        o.label.toLowerCase().includes(q) ||
        o.value.toLowerCase().includes(q),
    );
  }, [options, query]);

  const toggle = (val: string) => {
    if (selected.includes(val)) {
      onChange(selected.filter((s) => s !== val));
    } else {
      onChange([...selected, val]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ fontSize: 12, fontWeight: 600 }}>
          {label}{' '}
          {selected.length > 0 && (
            <span style={{ color: 'var(--blue, #3b82f6)' }}>
              ({selected.length})
            </span>
          )}
        </div>
        {selected.length > 0 && (
          <button
            onClick={() => onChange([])}
            style={{
              fontSize: 11,
              color: 'var(--text-muted)',
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            clear
          </button>
        )}
      </div>
      {options.length > 8 && (
        <input
          type="text"
          placeholder="Search…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{
            padding: '4px 8px',
            fontSize: 11,
            border: '1px solid var(--border)',
            borderRadius: 4,
            background: 'var(--bg-input)',
            color: 'var(--text)',
          }}
        />
      )}
      <div
        style={{
          maxHeight: 180,
          overflowY: 'auto',
          border: '1px solid var(--border)',
          borderRadius: 4,
          padding: 4,
        }}
      >
        {filtered.length === 0 && (
          <div
            style={{
              padding: 8,
              color: 'var(--text-muted)',
              fontSize: 11,
            }}
          >
            No matches
          </div>
        )}
        {filtered.map((opt) => {
          const checked = selected.includes(opt.value);
          return (
            <label
              key={opt.value}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 6px',
                fontSize: 11,
                cursor: 'pointer',
                background: checked
                  ? 'var(--blue-muted, #1e40af22)'
                  : undefined,
                borderRadius: 3,
              }}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggle(opt.value)}
                style={{ cursor: 'pointer' }}
              />
              <span>{opt.label}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}

interface FilterBarProps {
  cohort: StrategyMeta[];
}

export function FilterBar({ cohort }: FilterBarProps) {
  const { filters, setFilters, clearFilters } = useFilterState();
  const [open, setOpen] = useState(false);

  const options = useMemo(() => {
    const sidOpts = cohort.map((m) => ({
      value: String(m.strategy_id),
      label: `${m.strategy_id} · ${m.name || '(unknown)'}`,
    }));
    const symbolOpts = Array.from(
      new Set(cohort.map((m) => m.symbol).filter((s): s is string => !!s)),
    )
      .sort()
      .map((s) => ({ value: s, label: s }));
    const tfOpts = Array.from(
      new Set(
        cohort.map((m) => m.timeframe).filter((s): s is string => !!s),
      ),
    )
      .sort()
      .map((s) => ({ value: s, label: s }));
    const sessionOpts = Array.from(
      new Set(cohort.map((m) => m.session).filter((s): s is string => !!s)),
    )
      .sort()
      .map((s) => ({ value: s, label: s }));
    const originOpts = Array.from(
      new Set(cohort.map((m) => m.origin).filter((s): s is string => !!s)),
    )
      .sort()
      .map((s) => ({ value: s, label: originLabel(s) }));
    const trigOpts = Array.from(
      new Set(
        cohort
          .map((m) => m.trigger_pack)
          .filter((s): s is string => !!s),
      ),
    )
      .sort()
      .map((s) => ({ value: s, label: s }));
    const gateOpts = Array.from(
      new Set(cohort.flatMap((m) => m.gate_packs)),
    )
      .sort()
      .map((s) => ({ value: s, label: s }));
    return {
      sidOpts,
      symbolOpts,
      tfOpts,
      sessionOpts,
      originOpts,
      trigOpts,
      gateOpts,
    };
  }, [cohort]);

  const setType = (t: StrategyType | null) => {
    setFilters({ ...filters, type: t });
  };

  const empty = isFilterEmpty(filters);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setOpen((v) => !v)}
          style={{
            padding: '6px 12px',
            fontSize: 12,
            fontWeight: 600,
            borderRadius: 6,
            border: '1px solid var(--border)',
            background: open ? 'var(--blue-muted, #1e40af44)' : 'var(--bg-input)',
            color: open ? 'var(--blue, #3b82f6)' : 'var(--text)',
            cursor: 'pointer',
          }}
        >
          {empty ? 'Filters' : `Filters · active`}
        </button>
        {!empty && (
          <button
            onClick={clearFilters}
            style={{
              padding: '6px 10px',
              fontSize: 11,
              borderRadius: 6,
              border: '1px solid var(--border)',
              background: 'var(--bg-input)',
              color: 'var(--text-muted)',
              cursor: 'pointer',
            }}
          >
            Clear all
          </button>
        )}
        <div
          style={{
            display: 'flex',
            gap: 4,
            marginLeft: 'auto',
            flexWrap: 'wrap',
          }}
        >
          <span
            style={{
              alignSelf: 'center',
              fontSize: 11,
              color: 'var(--text-muted)',
              marginRight: 4,
            }}
          >
            Presets:
          </span>
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => setFilters(p.build(cohort))}
              title={p.description}
              style={{
                padding: '4px 10px',
                fontSize: 11,
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: 'var(--bg-input)',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Active chips */}
      {!empty && (
        <div
          style={{
            display: 'flex',
            gap: 6,
            flexWrap: 'wrap',
            fontFamily: 'monospace',
          }}
        >
          {filters.type && (
            <span style={ACTIVE_PILL_STYLE}>
              type: {filters.type}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setType(null)}
              >
                ×
              </span>
            </span>
          )}
          {filters.sids.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              sids: {filters.sids.length}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setFilters({ ...filters, sids: [] })}
              >
                ×
              </span>
            </span>
          )}
          {filters.symbols.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              symbols: {filters.symbols.join(', ')}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setFilters({ ...filters, symbols: [] })}
              >
                ×
              </span>
            </span>
          )}
          {filters.timeframes.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              tfs: {filters.timeframes.join(', ')}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setFilters({ ...filters, timeframes: [] })}
              >
                ×
              </span>
            </span>
          )}
          {filters.sessions.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              sessions: {filters.sessions.join(', ')}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setFilters({ ...filters, sessions: [] })}
              >
                ×
              </span>
            </span>
          )}
          {filters.origins.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              origin: {filters.origins.map(originLabel).join(', ')}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setFilters({ ...filters, origins: [] })}
              >
                ×
              </span>
            </span>
          )}
          {filters.triggerPacks.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              trigger: {filters.triggerPacks.join(', ')}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() =>
                  setFilters({ ...filters, triggerPacks: [] })
                }
              >
                ×
              </span>
            </span>
          )}
          {filters.gatePacks.length > 0 && (
            <span style={ACTIVE_PILL_STYLE}>
              gate: {filters.gatePacks.join(', ')}
              <span
                style={CLEAR_BTN_STYLE}
                onClick={() => setFilters({ ...filters, gatePacks: [] })}
              >
                ×
              </span>
            </span>
          )}
        </div>
      )}

      {/* Popover (renders inline as expanding panel — keeps it simple) */}
      {open && (
        <div
          style={{
            padding: 12,
            border: '1px solid var(--border)',
            borderRadius: 6,
            background: 'var(--bg-card, var(--bg))',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: 12,
          }}
        >
          {/* Type single-select */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              Strategy type
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['trigger', 'gate', null] as const).map((t) => (
                <button
                  key={String(t)}
                  onClick={() => setType(t)}
                  style={{
                    flex: 1,
                    padding: '6px 8px',
                    fontSize: 11,
                    borderRadius: 4,
                    border: '1px solid var(--border)',
                    background:
                      filters.type === t
                        ? 'var(--blue-muted, #1e40af44)'
                        : 'var(--bg-input)',
                    color:
                      filters.type === t
                        ? 'var(--blue, #3b82f6)'
                        : 'var(--text)',
                    cursor: 'pointer',
                    fontWeight: filters.type === t ? 600 : 400,
                  }}
                >
                  {t === null ? 'both' : t}
                </button>
              ))}
            </div>
          </div>
          <MultiSelect
            label="Strategy ID"
            options={options.sidOpts}
            selected={filters.sids.map(String)}
            onChange={(vals) =>
              setFilters({
                ...filters,
                sids: vals.map((v) => parseInt(v, 10)),
              })
            }
          />
          <MultiSelect
            label="Symbol"
            options={options.symbolOpts}
            selected={filters.symbols}
            onChange={(vals) => setFilters({ ...filters, symbols: vals })}
          />
          <MultiSelect
            label="Timeframe"
            options={options.tfOpts}
            selected={filters.timeframes}
            onChange={(vals) =>
              setFilters({ ...filters, timeframes: vals })
            }
          />
          <MultiSelect
            label="Session"
            options={options.sessionOpts}
            selected={filters.sessions}
            onChange={(vals) => setFilters({ ...filters, sessions: vals })}
          />
          <MultiSelect
            label="Origin"
            options={options.originOpts}
            selected={filters.origins}
            onChange={(vals) => setFilters({ ...filters, origins: vals })}
          />
          <MultiSelect
            label="Trigger pack"
            options={options.trigOpts}
            selected={filters.triggerPacks}
            onChange={(vals) =>
              setFilters({ ...filters, triggerPacks: vals })
            }
          />
          <MultiSelect
            label="Gate pack"
            options={options.gateOpts}
            selected={filters.gatePacks}
            onChange={(vals) =>
              setFilters({ ...filters, gatePacks: vals })
            }
          />
        </div>
      )}
    </div>
  );
}
