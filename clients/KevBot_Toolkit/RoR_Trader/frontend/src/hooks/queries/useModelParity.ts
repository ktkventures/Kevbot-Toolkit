/**
 * Hooks + fetchers for /api/admin/model-parity (W2-6c, 2026-07-07).
 *
 * Two endpoints back the Strategy Health "Model Parity" tab:
 *
 *  - GET /models          — cheap, whole fleet in one call. One row per
 *    strategy with the RUNTIME-EFFECTIVE backtest/algo/live model. The
 *    server mirrors the actual engine resolution sites (ralph_engine
 *    live_model fallback, forward_test_service backtest 'rest_hifi'
 *    literal, PR #26 _require_algo_model tripwire) — NOT the admin
 *    registry defaults.
 *
 *  - GET /parity?sids=&lens= — HEAVY, max 8 sids per request. Per-bar
 *    parity summary computed server-side with the exact logic of the
 *    detail page's "Per-Bar Parity Drift · v2" ribbon (last 300 common
 *    bars, same tolerances), so the numbers reconcile 1:1 with what the
 *    Strategy Detail page shows. Server caches ~10 min per (sid, lens).
 *    The tab walks the fleet in small batches — never one big request.
 *
 * Wire format is snake_case; these helpers map to camelCase per the
 * repo convention (CLAUDE.md "API Data Conventions").
 */

import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

// ── Domain types (camelCase) ──────────────────────────────────────────

/** How the runtime arrived at the effective model for a lane. */
export type ModelSource =
  | 'explicit'            // set in strategy config (green)
  | 'default'             // unset — runtime resolves the lane default (amber)
  | 'fallback_backtest'   // algo only: unset → falls back to config.backtest_model (amber)
  | 'invalid_fallback'    // live only: set but invalid id → engine uses default (red)
  | 'error';              // algo only: ALL unset → runtime raises (PR #26 tripwire) (red)

export interface ResolvedModel {
  /** Raw config value ('' → null). */
  declared: string | null;
  /** What the runtime actually uses. Null only for algo source='error'. */
  effective: string | null;
  source: ModelSource;
}

export interface FleetModelRow {
  strategyId: number;
  userId: string | null;
  name: string;
  symbol: string | null;
  timeframe: string | null;
  forwardTesting: boolean;
  backtestModel: ResolvedModel;
  algoModel: ResolvedModel;
  liveModel: ResolvedModel;
}

export interface ParityMetric {
  key: string;                       // 'exists' | 'open' | ... | 'ind:<col>'
  label: string;
  kind: 'exists' | 'price' | 'num';
  match: number;
  minor: number;
  major: number;
  comparable: number;
  /** match / comparable × 100 — same definition as the detail ribbon. */
  pct: number | null;
}

export interface ParityBarsSummary {
  common: number;
  backtestOnly: number;
  liveOnly: number;
  windowBars: number;
  totalOverlapBars: number;
  truncated: boolean;
  windowStart: string | null;
  windowEnd: string | null;
}

export interface StrategyParityRow {
  strategyId: number;
  symbol: string | null;
  timeframe: string | null;
  lens: 'first' | 'latest';
  computedAt: string | null;
  cached: boolean;
  metrics: ParityMetric[];
  bars: ParityBarsSummary | null;
  note: string | null;
  error: string | null;
}

// ── Wire types (snake_case, as returned by the API) ──────────────────

interface WireResolvedModel {
  declared: string | null;
  effective: string | null;
  source: ModelSource;
}

interface WireFleetModelRow {
  strategy_id: number;
  user_id: string | null;
  name: string;
  symbol: string | null;
  timeframe: string | null;
  forward_testing: boolean;
  backtest_model: WireResolvedModel;
  algo_model: WireResolvedModel;
  live_model: WireResolvedModel;
}

interface WireFleetModelsResponse {
  now: string;
  rows: WireFleetModelRow[];
}

interface WireParityMetric {
  key: string;
  label: string;
  kind: 'exists' | 'price' | 'num';
  match: number;
  minor: number;
  major: number;
  comparable: number;
  pct: number | null;
}

interface WireParityRow {
  strategy_id: number;
  symbol?: string | null;
  timeframe?: string | null;
  lens: 'first' | 'latest';
  computed_at?: string;
  cached?: boolean;
  metrics?: WireParityMetric[];
  bars?: {
    common: number;
    backtest_only: number;
    live_only: number;
    window_bars: number;
    total_overlap_bars: number;
    truncated: boolean;
    window_start: string | null;
    window_end: string | null;
  } | null;
  note?: string;
  error?: string;
}

interface WireParityResponse {
  now: string;
  lens: 'first' | 'latest';
  cache_ttl_seconds: number;
  max_sids_per_request: number;
  rows: WireParityRow[];
}

// ── Mapping ───────────────────────────────────────────────────────────

function mapModelRow(w: WireFleetModelRow): FleetModelRow {
  return {
    strategyId: w.strategy_id,
    userId: w.user_id,
    name: w.name,
    symbol: w.symbol,
    timeframe: w.timeframe,
    forwardTesting: w.forward_testing,
    backtestModel: w.backtest_model,
    algoModel: w.algo_model,
    liveModel: w.live_model,
  };
}

function mapParityRow(w: WireParityRow): StrategyParityRow {
  return {
    strategyId: w.strategy_id,
    symbol: w.symbol ?? null,
    timeframe: w.timeframe ?? null,
    lens: w.lens,
    computedAt: w.computed_at ?? null,
    cached: w.cached ?? false,
    metrics: (w.metrics ?? []).map(m => ({ ...m })),
    bars: w.bars
      ? {
          common: w.bars.common,
          backtestOnly: w.bars.backtest_only,
          liveOnly: w.bars.live_only,
          windowBars: w.bars.window_bars,
          totalOverlapBars: w.bars.total_overlap_bars,
          truncated: w.bars.truncated,
          windowStart: w.bars.window_start,
          windowEnd: w.bars.window_end,
        }
      : null,
    note: w.note ?? null,
    error: w.error ?? null,
  };
}

// ── Hooks / fetchers ──────────────────────────────────────────────────

export function useFleetModels() {
  return useQuery<FleetModelRow[]>({
    queryKey: ['admin', 'model-parity', 'models'],
    queryFn: async () => {
      const resp = await apiFetch<WireFleetModelsResponse>(
        '/api/admin/model-parity/models');
      return (resp.rows ?? []).map(mapModelRow);
    },
    staleTime: 60_000,
  });
}

/** Batch parity fetch — call with ≤8 sids (server rejects more). Used
 *  imperatively by the tab's progressive loader, not as a hook, because
 *  batches are sequenced against a client-side cache map. */
export async function fetchParityBatch(
  sids: number[],
  lens: 'first' | 'latest',
): Promise<StrategyParityRow[]> {
  const resp = await apiFetch<WireParityResponse>(
    `/api/admin/model-parity/parity?sids=${sids.join(',')}&lens=${lens}`);
  return (resp.rows ?? []).map(mapParityRow);
}
