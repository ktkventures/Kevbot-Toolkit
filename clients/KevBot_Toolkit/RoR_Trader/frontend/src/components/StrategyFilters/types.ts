/**
 * Shared types for the Strategy Health filter system.
 *
 * Used by both the By Hour and By Deploy tabs. Filter state lives in
 * the URL via search params so views are bookmarkable / shareable.
 */

export type StrategyType = 'trigger' | 'gate';

export interface StrategyMeta {
  strategy_id: number;
  name: string | null;
  symbol: string | null;
  timeframe: string | null;
  session: string | null;
  trigger_pack: string | null;
  gate_packs: string[];
  strategy_type: StrategyType;
}

export interface FilterState {
  sids: number[];
  symbols: string[];
  timeframes: string[];
  sessions: string[];
  triggerPacks: string[];
  gatePacks: string[];
  type: StrategyType | null;
}

export const EMPTY_FILTERS: FilterState = {
  sids: [],
  symbols: [],
  timeframes: [],
  sessions: [],
  triggerPacks: [],
  gatePacks: [],
  type: null,
};

export function isFilterEmpty(f: FilterState): boolean {
  return (
    f.sids.length === 0 &&
    f.symbols.length === 0 &&
    f.timeframes.length === 0 &&
    f.sessions.length === 0 &&
    f.triggerPacks.length === 0 &&
    f.gatePacks.length === 0 &&
    f.type === null
  );
}

/**
 * Does this strategy pass the active filters? All filters AND together.
 * Multi-select dimensions are OR-within (e.g., symbols: SPY OR TSLA).
 */
export function passesFilters(m: StrategyMeta, f: FilterState): boolean {
  if (f.sids.length > 0 && !f.sids.includes(m.strategy_id)) return false;
  if (f.symbols.length > 0 && (!m.symbol || !f.symbols.includes(m.symbol)))
    return false;
  if (
    f.timeframes.length > 0 &&
    (!m.timeframe || !f.timeframes.includes(m.timeframe))
  )
    return false;
  if (
    f.sessions.length > 0 &&
    (!m.session || !f.sessions.includes(m.session))
  )
    return false;
  if (
    f.triggerPacks.length > 0 &&
    (!m.trigger_pack || !f.triggerPacks.includes(m.trigger_pack))
  )
    return false;
  if (f.gatePacks.length > 0) {
    // Strategy must have AT LEAST one of the selected gate packs
    const has = m.gate_packs.some((g) => f.gatePacks.includes(g));
    if (!has) return false;
  }
  if (f.type !== null && m.strategy_type !== f.type) return false;
  return true;
}

/**
 * Pre-built presets that combine common filter combinations.
 * Picking a preset replaces the current filter state.
 */
export interface FilterPreset {
  id: string;
  label: string;
  description: string;
  build: (cohort: StrategyMeta[]) => FilterState;
}

export const PRESETS: FilterPreset[] = [
  {
    id: 'all-gates',
    label: 'All gates',
    description: 'Only gate-style strategies',
    build: () => ({ ...EMPTY_FILTERS, type: 'gate' }),
  },
  {
    id: 'all-triggers',
    label: 'All triggers',
    description: 'Only trigger-style strategies',
    build: () => ({ ...EMPTY_FILTERS, type: 'trigger' }),
  },
  {
    id: 'ut-bot',
    label: 'UT Bot anywhere',
    description: 'Strategies with ut_bot_v4 as trigger OR gate',
    build: (cohort) => ({
      ...EMPTY_FILTERS,
      sids: cohort
        .filter(
          (m) =>
            m.trigger_pack === 'ut_bot_v4' ||
            m.gate_packs.includes('ut_bot_v4'),
        )
        .map((m) => m.strategy_id),
    }),
  },
  {
    id: 'tsla',
    label: 'TSLA',
    description: 'TSLA symbol only',
    build: () => ({ ...EMPTY_FILTERS, symbols: ['TSLA'] }),
  },
  {
    id: '10sec',
    label: '10 Second',
    description: '10Sec primary timeframe',
    build: () => ({ ...EMPTY_FILTERS, timeframes: ['10Sec'] }),
  },
];
