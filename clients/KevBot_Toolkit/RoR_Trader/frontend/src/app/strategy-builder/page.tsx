'use client';

import { useCallback } from 'react';
import { useRouter } from 'next/navigation';
import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import type { BacktestResult } from './versions/V5';
import { useRunBacktest } from '@/hooks/queries/useBacktest';
import { useCreateStrategy } from '@/hooks/mutations/useStrategyMutations';
import { useRiskManagementPacks } from '@/hooks/queries/usePacks';

/**
 * Transform API backtest response to V5's BacktestResult shape.
 */
function transformBacktestResult(apiResult: any): BacktestResult {
  return {
    trades: (apiResult.trades || []).map((t: any, i: number) => ({
      id: i + 1,
      entryTime: t.entry_time ? new Date(t.entry_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
      exitTime: t.exit_time ? new Date(t.exit_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
      direction: t.direction || 'LONG',
      entryPrice: t.entry_price || 0,
      exitPrice: t.exit_price || 0,
      rMultiple: t.r_multiple || 0,
      execType: t.exec_type || 'C',
      exitReason: t.exit_reason || '',
      confluences: '',
    })),
    kpis: {
      winRate: apiResult.kpis?.win_rate ?? 0,
      profitFactor: apiResult.kpis?.profit_factor ?? 0,
      dailyR: apiResult.kpis?.daily_r ?? 0,
      totalTrades: apiResult.kpis?.total_trades ?? 0,
      avgWin: apiResult.secondary_kpis?.avg_win_r ?? 0,
      avgLoss: apiResult.secondary_kpis?.avg_loss_r ?? 0,
      maxDrawdown: apiResult.kpis?.max_r_drawdown ?? 0,
      recoveryFactor: apiResult.secondary_kpis?.recovery_factor ?? 0,
      sharpeRatio: apiResult.secondary_kpis?.sharpe_ratio ?? 0,
      expectancy: apiResult.kpis?.avg_r ?? 0,
      totalR: apiResult.kpis?.total_r ?? 0,
      avgR: apiResult.kpis?.avg_r ?? 0,
      rSquared: apiResult.kpis?.r_squared ?? 0,
      maxRDrawdown: apiResult.kpis?.max_r_drawdown ?? 0,
    },
    equityCurve: apiResult.equity_curve || [],
    chartData: apiResult.chart_data,
    totalBars: apiResult.total_bars || 0,
    dataSource: apiResult.data_source || 'Unknown',
  };
}

/**
 * Transform RM packs from API to the RiskPack shape V5 expects.
 */
function transformRMPacks(packs: any[], type: 'stop' | 'target'): { id: string; name: string; version: string; summary: string }[] {
  if (!packs) return [];
  return packs.map((p) => ({
    id: p.id,
    name: `${p.base_template} (${p.version})`,
    version: p.version,
    summary: Object.entries(p.parameters || {}).map(([k, v]) => `${k}: ${v}`).join(', ') || p.version,
  }));
}

function WiredV5() {
  const router = useRouter();
  const backtestMutation = useRunBacktest();
  const createMutation = useCreateStrategy();
  const { data: rmPacks } = useRiskManagementPacks();

  const handleRunBacktest = useCallback((config: Record<string, any>) => {
    backtestMutation.mutate({
      symbol: config.symbol,
      timeframe: config.timeframe,
      direction: config.direction,
      days: config.days,
      lookback_mode: config.lookback_mode,
      session: config.session,
      entry_trigger_confluence_id: config.entry_trigger_confluence_id,
      exit_trigger_confluence_ids: config.exit_trigger_confluence_ids || [],
      confluence: config.confluence || [],
      stop_loss_pack_id: config.stop_loss_pack_id,
      take_profit_pack_id: config.take_profit_pack_id,
      include_chart_data: true,
    });
  }, [backtestMutation]);

  const handleSave = useCallback((strategy: Record<string, any>) => {
    createMutation.mutate(strategy, {
      onSuccess: () => {
        router.push('/strategies');
      },
    });
  }, [createMutation, router]);

  // Transform backtest result for V5
  const backtestResult = backtestMutation.data
    ? transformBacktestResult(backtestMutation.data)
    : null;

  // Transform RM packs
  const stopPacks = rmPacks ? transformRMPacks(rmPacks, 'stop') : undefined;
  const targetPacks = rmPacks ? transformRMPacks(rmPacks, 'target') : undefined;

  return (
    <V5
      backtestResult={backtestResult}
      onRunBacktest={handleRunBacktest}
      onSave={handleSave}
      isBacktesting={backtestMutation.isPending}
      stopPacks={stopPacks}
      targetPacks={targetPacks}
    />
  );
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Three-column layout with config inputs.',
      rationale: 'First pass focused on layout.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete Streamlit feature set.',
      rationale: 'Direct port of every feature.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Single scrollable page.',
      rationale: 'Removes build-time friction.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Visual pipeline builder.',
      rationale: 'The wow-factor version.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Live)',
      description: 'Pack-integrated builder with real backtest API — run backtests, see real KPIs, save strategies.',
      rationale: 'Production version wired to FastAPI backtest endpoint.',
    },
    component: WiredV5,
  },
];

export default function StrategyBuilderPage() {
  return <VersionedPage pageKey="strategy-builder" versions={versions} />;
}
