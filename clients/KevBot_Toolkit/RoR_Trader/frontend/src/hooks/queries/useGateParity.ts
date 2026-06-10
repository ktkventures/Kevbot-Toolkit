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

export interface GateParityLiveRow {
  ts: string;
  pb: string | null;
  cb: string | null;
  pb_pass: boolean;
  cb_pass: boolean;
  paired_bt: boolean;
}

export interface GateParityResponse {
  meta: {
    sid: number;
    symbol: string;
    primary_tf: string;
    gate: string;
    want_state: string;
    entry_trigger: string;
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
