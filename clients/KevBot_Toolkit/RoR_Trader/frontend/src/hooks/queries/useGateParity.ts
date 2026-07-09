/**
 * useGateParity — fetches the theoretical Gate Parity view from the engine.
 *
 * Thin renderer: the backend (`gate_parity_harness.build_gate_parity_view`)
 * runs the REAL engine pipeline and returns engine-truth PB/CB gate ribbons,
 * theoretical backtest entries (fresh/current logic), and per-live-entry gate
 * classification. The frontend only displays this.
 */
import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

/** Per-gate PB/CB state for a single live entry (keyed by gate string). */
export interface GateParityGateState {
  pb: string | null;
  cb: string | null;
  pb_pass: boolean;
  cb_pass: boolean;
}

export interface GateParityLiveRow {
  ts: string;
  // Top-level fields describe the FIRST gate (backward compatible).
  pb: string | null;
  cb: string | null;
  pb_pass: boolean;
  cb_pass: boolean;
  paired_bt: boolean;
  // Per-gate breakdown across ALL confluence gates (added).
  gates?: Record<string, GateParityGateState>;
}

/** Per-confluence-gate parity summary. */
export interface GateParityPerGate {
  gate: string;
  tf: string;
  interp: string;
  want_state: string;
  resolved: boolean;        // false = ribbon not computable in the backtest lens
  pb_dist: Record<string, number>;
  cb_dist: Record<string, number>;
  pb_open_pct: number;      // NOTE: coarse-gate PB is NaN-shallow over short windows
  cb_open_pct: number;
  live_n: number;
  live_pb_pass: number;
  live_cb_pass: number;
  phantom_pb_fail: number;  // phantoms this gate blocks under PB (backtest lens)
  phantom_cb_fail: number;  // phantoms this gate blocks under CB (reliable signal)
}

export interface GateParityResponse {
  meta: {
    sid: number;
    symbol: string;
    primary_tf: string;
    gate: string | null;              // first gate (null when ungated)
    want_state: string | null;
    entry_trigger: string | null;     // base trigger id (matches alerts.trigger_id)
    entry_trigger_confluence_id?: string | null;
    gates?: string[];                 // every confluence gate
    ungated?: boolean;
    divergent_gate?: string | null;   // gate that most explains the phantoms
    live_model: string;
    backtest_model: string;
    session: string;
    window: [string, string];
    bars: number;
  };
  ribbon: {
    pb_dist: Record<string, number>;
    cb_dist: Record<string, number>;
  };
  per_gate?: GateParityPerGate[];
  entries: { theoretical_bt: number; live_actual: number };
  live_rows: GateParityLiveRow[];
}

export function useGateParity(strategyId: number | null, windowHours: number, enabled = true) {
  return useQuery({
    queryKey: ['gate-parity', strategyId, windowHours],
    queryFn: () =>
      apiFetch<GateParityResponse>(
        `/api/strategies/${strategyId}/gate-parity?window_hours=${windowHours}`
      ),
    enabled: enabled && strategyId !== null,
    staleTime: 60_000,
  });
}

export interface Window1sBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
export interface Window1sResponse {
  symbol: string;
  end: string;
  lookback: number;
  bars_1s: Window1sBar[];
}

/** 1-second bars for a right-aligned window ending at `endIso` (the scrub head). */
export function useWindow1s(
  strategyId: number | null,
  endIso: string | null,
  lookback = 120,
  enabled = true,
) {
  return useQuery({
    queryKey: ['window-1s', strategyId, endIso, lookback],
    queryFn: () =>
      apiFetch<Window1sResponse>(
        `/api/strategies/${strategyId}/window-1s?end=${encodeURIComponent(endIso || '')}&lookback=${lookback}`
      ),
    enabled: enabled && strategyId !== null && !!endIso,
    staleTime: 60_000,
  });
}

/**
 * useLiveGateTelemetry — per-bar LIVE gate-state records the engine actually
 * emitted (bar_diagnostics source='live_gate'). Ground truth for "what the
 * live gates saw", as opposed to inferring gate state from the bar cache.
 * Each row: { bar_ts, tf, records:[confluence strings], written_at }.
 */
export interface LiveGateTelemetryRow {
  bar_ts: string;
  tf: number | null;
  records: string[];
  written_at: string | null;
}
export interface LiveGateTelemetryResponse {
  strategy_id: number;
  start: string;
  end: string;
  timeframes: number[];
  count: number;
  rows: LiveGateTelemetryRow[];
}

export function useLiveGateTelemetry(
  strategyId: number | null,
  startIso: string | null,
  endIso: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ['live-gate-telemetry', strategyId, startIso, endIso],
    queryFn: () =>
      apiFetch<LiveGateTelemetryResponse>(
        `/api/strategies/${strategyId}/live-gate-telemetry` +
          `?start=${encodeURIComponent(startIso || '')}` +
          `&end=${encodeURIComponent(endIso || '')}`
      ),
    enabled: enabled && strategyId !== null && !!startIso && !!endIso,
    staleTime: 60_000,
  });
}

/**
 * usePineExport — generate a TradingView Pine v6 strategy() for this strategy.
 * Faithful port (RTH trade parity ~98.6% vs our backtest on the sid 303 ref).
 * ok=false with `error` when a pack has no Pine emitter yet.
 */
export interface PineExportResponse {
  ok: boolean;
  pine: string | null;
  error: string | null;
  caveats: string[];
  symbol?: string;
  timeframe?: string;
}

export function usePineExport(strategyId: number | null, enabled = true) {
  return useQuery({
    queryKey: ['pine-export', strategyId],
    queryFn: () =>
      apiFetch<PineExportResponse>(`/api/strategies/${strategyId}/pine-export`),
    enabled: enabled && strategyId !== null,
    staleTime: 300_000,
  });
}

/**
 * usePineReadiness — structured TV-export readiness for the badge. Returns EVERY
 * blocking gap (not just the first like pine-export's `error`), so the UI can
 * list exactly what a strategy needs to be exportable + trade-ready via TV.
 */
export interface PineReadinessGap {
  kind: string; // 'trigger' | 'gate' | 'stop' | 'time_exit'
  detail: string;
  role?: string; // entry | exit (triggers)
  interp?: string;
  state?: string;
  method?: string;
  value?: string | null;
}
export interface PineReadinessResponse {
  ready: boolean;
  missing: PineReadinessGap[];
  caveats: string[];
}
export function usePineReadiness(strategyId: number | null, enabled = true) {
  return useQuery({
    queryKey: ['pine-readiness', strategyId],
    queryFn: () =>
      apiFetch<PineReadinessResponse>(
        `/api/strategies/${strategyId}/pine-readiness`,
      ),
    enabled: enabled && strategyId !== null,
    staleTime: 300_000,
  });
}
