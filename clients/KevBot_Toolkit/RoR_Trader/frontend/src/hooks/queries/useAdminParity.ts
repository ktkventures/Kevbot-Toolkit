/**
 * useAdminParity — orchestrates existing data hooks for the Admin > Parity
 * page. No new endpoint needed for Phase B (Bars Comparison) — just combines
 * useStrategyCacheBars + useBars and exposes a unified loading state.
 *
 * Phase C will add a per-snapshot endpoint for entry-overlay data
 * (live alerts + algo trades + bt trades aligned to bars).
 */
import { useMemo } from 'react';
import { useBars, type BarData } from './useMarketData';
import { useStrategyCacheBars, type CacheBar } from './useStrategies';

export interface BarComparisonRow {
  timestamp: string;
  cache_close: number | null;
  rest_close: number | null;
  close_diff: number | null;
  cache_vol: number | null;
  rest_vol: number | null;
  vol_ratio: number | null;
  /** Convenience flag: significant divergence (>$0.01 close or >5% vol). */
  flagged: boolean;
}

export interface UseAdminParityBarsArgs {
  strategyId: number | null;
  symbol: string | null;
  timeframe: string | null;
  /** REST-side lookback. Defaults to 2 days. Cache-side window is
   *  controlled by the cache-bars endpoint itself. */
  days?: number;
}

export interface UseAdminParityBarsResult {
  cacheBars: CacheBar[];
  restBars: BarData[];
  rows: BarComparisonRow[];
  isLoading: boolean;
  isError: boolean;
  cacheNotes: string[];
  cacheValueType: string | null;
}

/**
 * Compute the per-timestamp comparison between cache OHLCV and REST OHLCV.
 *
 * Match key = `timestamp` truncated to the minute. The cache returns
 * bars at the strategy's primary TF (e.g., 10Sec), while REST is
 * always 1Min via the existing useBars hook. The page should select
 * a TF that's commensurable on both sides — for now we display whatever
 * each side returns and just join by minute, leaving the granularity
 * mismatch to be a Phase B finding the user can SEE if it bites.
 */
function buildRows(
  cacheBars: CacheBar[],
  restBars: BarData[],
): BarComparisonRow[] {
  const restByMin = new Map<string, BarData>();
  for (const b of restBars) {
    const minute = b.timestamp.slice(0, 16); // YYYY-MM-DDTHH:MM
    restByMin.set(minute, b);
  }

  const cacheByMin = new Map<string, CacheBar[]>();
  for (const c of cacheBars) {
    const minute = c.timestamp.slice(0, 16);
    const arr = cacheByMin.get(minute) || [];
    arr.push(c);
    cacheByMin.set(minute, arr);
  }

  const allMinutes = new Set<string>();
  restByMin.forEach((_v, k) => allMinutes.add(k));
  cacheByMin.forEach((_v, k) => allMinutes.add(k));
  const sortedMinutes = Array.from(allMinutes).sort();

  return sortedMinutes.map<BarComparisonRow>((minute) => {
    const rest = restByMin.get(minute) || null;
    const cacheList = cacheByMin.get(minute) || [];

    // Cache may have multiple sub-minute bars per minute; aggregate to
    // a single OHLCV for fair comparison vs the REST 1Min bar.
    let cacheClose: number | null = null;
    let cacheVol: number | null = null;
    if (cacheList.length > 0) {
      cacheClose = cacheList[cacheList.length - 1].close;
      cacheVol = cacheList.reduce((sum, b) => sum + (b.volume || 0), 0);
    }

    const restClose = rest?.close ?? null;
    const restVol = rest?.volume ?? null;

    const close_diff = cacheClose != null && restClose != null
      ? cacheClose - restClose
      : null;
    const vol_ratio = cacheVol != null && restVol != null && restVol > 0
      ? cacheVol / restVol
      : null;

    const closeFlag = close_diff != null && Math.abs(close_diff) > 0.01;
    const volFlag = vol_ratio != null && (vol_ratio < 0.95 || vol_ratio > 1.05);

    return {
      timestamp: minute + ':00Z',
      cache_close: cacheClose,
      rest_close: restClose,
      close_diff,
      cache_vol: cacheVol,
      rest_vol: restVol,
      vol_ratio,
      flagged: closeFlag || volFlag,
    };
  });
}

export function useAdminParityBars(
  args: UseAdminParityBarsArgs,
): UseAdminParityBarsResult {
  const { strategyId, symbol, timeframe, days = 2 } = args;

  const cacheQuery = useStrategyCacheBars(strategyId, 'first', !!strategyId);
  const restQuery = useBars(symbol, timeframe || '1Min', days);

  const rows = useMemo(() => {
    const cacheBars = cacheQuery.data?.chart_data || [];
    const restBars = restQuery.data || [];
    return buildRows(cacheBars, restBars);
  }, [cacheQuery.data, restQuery.data]);

  return {
    cacheBars: cacheQuery.data?.chart_data || [],
    restBars: restQuery.data || [],
    rows,
    isLoading: cacheQuery.isLoading || restQuery.isLoading,
    isError: cacheQuery.isError || restQuery.isError,
    cacheNotes: cacheQuery.data?.notes || [],
    cacheValueType: cacheQuery.data?.value_type || null,
  };
}
