/**
 * Hook for the /api/admin/data-health endpoint.  Polls every 30s so the
 * dashboard reflects fresh writes without manual refresh — but stale-time
 * is set so multiple subscribers in the same session share the result.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface DataHealthWindow {
  expected: number;
  actual: number;
  coverage: number;        // 0.0–1.0
  ws: number;
  rest_backfill: number;
  other: number;
}

/** Rolling mode populates 1h/4h/rth/24h; custom mode populates `custom`. */
export type DataHealthWindows = Partial<Record<
  '1h' | '4h' | 'rth' | '24h' | 'custom',
  DataHealthWindow
>>;

export interface DataHealthRow {
  symbol: string;
  timeframe_seconds: number;
  subscribers: number;
  windows: DataHealthWindows;
  latest_bar: string | null;
  latest_bar_age_sec: number | null;
  gap_events_4h: number;
  bars_missing_4h: number;
}

export interface DataHealthResponse {
  now: string;
  scan_minutes: number;
  mode: 'rolling' | 'custom';
  window_start?: string;   // present iff mode === 'custom'
  window_end?: string;
  rows: DataHealthRow[];
}

export interface DataHealthQueryArgs {
  /** Both must be set to switch to custom mode. ISO 8601 or Unix-sec. */
  start?: string | null;
  end?: string | null;
}

export function useDataHealth(args: DataHealthQueryArgs = {}) {
  const { start, end } = args;
  const isCustom = !!(start && end);
  const qs = isCustom
    ? `?start=${encodeURIComponent(start!)}&end=${encodeURIComponent(end!)}`
    : '';
  return useQuery<DataHealthResponse>({
    queryKey: ['admin', 'data-health', start ?? null, end ?? null],
    queryFn: () => apiFetch<DataHealthResponse>(`/api/admin/data-health${qs}`),
    staleTime: 25_000,
    // In custom mode, the window is fixed — don't poll.  Rolling mode polls
    // every 30s so freshness reflects new writes.
    refetchInterval: isCustom ? false : 30_000,
  });
}
