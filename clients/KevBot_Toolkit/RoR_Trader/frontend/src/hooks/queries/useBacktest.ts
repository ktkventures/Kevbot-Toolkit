/**
 * Backtest hooks — run backtests and compute KPIs.
 */

import { useMutation } from '@tanstack/react-query';
import { apiFetch } from '@/lib/api/client';

export interface BacktestRequest {
  symbol: string;
  timeframe?: string;
  direction: 'LONG' | 'SHORT';
  days?: number;
  lookback_mode?: string;
  lookback_start_date?: string;
  lookback_end_date?: string;
  session?: string;
  data_feed?: string;
  entry_trigger_confluence_id: string;
  exit_trigger_confluence_ids?: string[];
  confluence?: string[];
  stop_loss_pack_id?: string;
  take_profit_pack_id?: string;
  stop_config?: Record<string, any>;
  target_config?: Record<string, any>;
  stop_exec_type?: 'C' | 'L' | 'LC' | 'CC';
  target_exec_type?: 'C' | 'L' | 'LC' | 'CC';
  stop_atr_mult?: number;
  risk_per_trade?: number;
  bar_count_exit?: number;
  secondary_tfs?: string[];
  hifi_mode?: boolean;
  cb_conditions?: string[];
  include_chart_data?: boolean;
  chart_indicators?: string[];
}

export interface BacktestResponse {
  trades: Record<string, any>[];
  kpis: Record<string, number>;
  secondary_kpis: Record<string, any>;
  equity_curve: { trade_number: number; timestamp: string; cumulative_r: number }[];
  total_bars: number;
  total_trading_days: number;
  data_source: string;
  chart_data?: Record<string, any>[];
}

export function useRunBacktest() {
  return useMutation({
    mutationFn: (req: BacktestRequest) =>
      apiFetch<BacktestResponse>('/api/backtest/run', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  });
}

/* ── Trade Zoom (for unsaved backtests — no strategy ID needed) ── */

export interface BacktestTradeZoomRequest extends BacktestRequest {
  trade_idx: number;
  side: 'entry' | 'exit';
  padding_seconds?: number;
}

export interface TradeZoomResponse {
  bars_1s: { time: string; open: number; high: number; low: number; close: number; volume: number }[];
  trade: Record<string, any>;
  indicators?: Record<string, { time: string; value: number }[]>;
  cb_confluence_timeline?: Record<string, { time: string; states: Record<string, string> }[]>;
  pb_states?: Record<string, string>;
  side: 'entry' | 'exit';
  timeframe: string;
  symbol: string;
  error?: string;
}

export function useBacktestTradeZoom() {
  return useMutation({
    mutationFn: (req: BacktestTradeZoomRequest) =>
      apiFetch<TradeZoomResponse>('/api/backtest/trade-zoom', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  });
}

/* ── Trade Replay (scenario data for a single trade) ── */

export interface BacktestTradeReplayRequest extends BacktestRequest {
  trade_idx: number;
}

export function useBacktestTradeReplay() {
  return useMutation({
    mutationFn: (req: BacktestTradeReplayRequest) =>
      apiFetch<any>('/api/backtest/trade-replay', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  });
}

/* ── Analyze ── */

export interface AnalyzeResult {
  trigger_id: string;
  trigger_name: string;
  exec_type: string;
  fidelity_type?: 'PB' | 'CB';
  total_trades: number;
  profit_factor: number;
  win_rate: number;
  avg_r: number;
  daily_r: number;
  r_squared: number;
}

export function useAnalyzeTriggers() {
  return useMutation({
    mutationFn: ({ mode, depth, ...req }: BacktestRequest & { mode?: string; depth?: number }) => {
      const params = new URLSearchParams({ mode: mode || 'entry' });
      if (depth && depth > 1) params.set('depth', String(depth));
      return apiFetch<{ results: AnalyzeResult[] }>(`/api/backtest/analyze?${params}`, {
        method: 'POST',
        body: JSON.stringify(req),
      });
    },
  });
}
