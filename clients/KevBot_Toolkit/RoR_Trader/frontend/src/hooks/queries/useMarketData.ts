/**
 * Market data query hooks — OHLCV bars from Polygon via backend.
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface BarData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export function useBars(
  symbol: string | null,
  timeframe = '1Min',
  days = 30,
) {
  return useQuery({
    queryKey: ['bars', symbol, timeframe, days],
    queryFn: () =>
      apiFetch<BarData[]>(
        `/api/data/bars/${symbol}?timeframe=${timeframe}&days=${days}`
      ),
    enabled: !!symbol,
    staleTime: 60000, // 1 minute
  });
}
