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
