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

export interface DataHealthRow {
  symbol: string;
  timeframe_seconds: number;
  subscribers: number;
  windows: {
    '1h': DataHealthWindow;
    '4h': DataHealthWindow;
    rth: DataHealthWindow;
    '24h': DataHealthWindow;
  };
  latest_bar: string | null;
  latest_bar_age_sec: number | null;
  gap_events_4h: number;
  bars_missing_4h: number;
}

export interface DataHealthResponse {
  now: string;
  scan_minutes: number;
  rows: DataHealthRow[];
}

export function useDataHealth() {
  return useQuery<DataHealthResponse>({
    queryKey: ['admin', 'data-health'],
    queryFn: () => apiFetch<DataHealthResponse>('/api/admin/data-health'),
    staleTime: 25_000,
    refetchInterval: 30_000,
  });
}
