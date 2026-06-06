/**
 * URL-synced filter state for Strategy Health views.
 *
 * Reads from search params on mount and writes back when filters
 * change. Each filter dimension uses a comma-separated param.
 * Empty filters omit the param entirely so the URL stays clean.
 */
'use client';

import { useCallback } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import {
  type FilterState,
  type StrategyType,
  EMPTY_FILTERS,
} from './types';

function parseCsv(v: string | null): string[] {
  if (!v) return [];
  return v
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

function parseSidCsv(v: string | null): number[] {
  if (!v) return [];
  return v
    .split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isInteger(n) && n > 0);
}

function parseType(v: string | null): StrategyType | null {
  if (v === 'trigger' || v === 'gate') return v;
  return null;
}

export function useFilterState(): {
  filters: FilterState;
  setFilters: (f: FilterState) => void;
  clearFilters: () => void;
} {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const filters: FilterState = {
    sids: parseSidCsv(searchParams.get('sids')),
    symbols: parseCsv(searchParams.get('symbols')),
    timeframes: parseCsv(searchParams.get('tfs')),
    sessions: parseCsv(searchParams.get('sessions')),
    origins: parseCsv(searchParams.get('origins')),
    triggerPacks: parseCsv(searchParams.get('trigger_packs')),
    gatePacks: parseCsv(searchParams.get('gate_packs')),
    type: parseType(searchParams.get('type')),
  };

  const setFilters = useCallback(
    (f: FilterState) => {
      const params = new URLSearchParams(searchParams.toString());
      // Preserve non-filter params (like tab if it's ever encoded
      // in URL); replace only the filter keys.
      const setOrDelete = (k: string, v: string) => {
        if (v) params.set(k, v);
        else params.delete(k);
      };
      setOrDelete('sids', f.sids.join(','));
      setOrDelete('symbols', f.symbols.join(','));
      setOrDelete('tfs', f.timeframes.join(','));
      setOrDelete('sessions', f.sessions.join(','));
      setOrDelete('origins', f.origins.join(','));
      setOrDelete('trigger_packs', f.triggerPacks.join(','));
      setOrDelete('gate_packs', f.gatePacks.join(','));
      setOrDelete('type', f.type ?? '');
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [searchParams, router, pathname],
  );

  const clearFilters = useCallback(() => {
    setFilters(EMPTY_FILTERS);
  }, [setFilters]);

  return { filters, setFilters, clearFilters };
}
