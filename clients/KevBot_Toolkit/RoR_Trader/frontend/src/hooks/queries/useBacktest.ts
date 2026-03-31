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
  stop_atr_mult?: number;
  risk_per_trade?: number;
  bar_count_exit?: number;
  secondary_tfs?: string[];
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

export interface AnalyzeResult {
  trigger_id: string;
  trigger_name: string;
  exec_type: string;
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
