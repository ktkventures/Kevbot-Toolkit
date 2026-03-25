'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V5 from './versions/V5';
import { usePortfolios } from '@/hooks/queries/usePortfolios';
import { useDeletePortfolio, useDuplicatePortfolio } from '@/hooks/mutations/usePortfolioMutations';

/**
 * Transform API portfolio to V5 Portfolio interface shape.
 */
function apiToV5Portfolio(p: any): any {
  const kpis = p.kpis || {};
  return {
    id: String(p.id),
    name: p.name || 'Untitled',
    status: p.enabled ? 'On Track' : 'Insufficient Data',
    enabled: p.enabled ?? true,
    strategies: (p.strategies || []).map((s: any) => ({ symbol: s.symbol || '???', direction: s.direction || 'LONG' })),
    startingBalance: p.account?.starting_balance ?? 10000,
    compoundRate: p.account?.compound_rate ?? 0,
    avgRiskPerTrade: p.account?.avg_risk_per_trade ?? 250,
    webhookTemplate: p.webhook_template ?? null,
    requirementSet: p.requirement_set_id ? String(p.requirement_set_id) : null,
    reqPassing: null,
    reqTotal: null,
    totalPnl: kpis.total_pnl ?? 0,
    finalBalance: kpis.final_balance ?? (p.account?.starting_balance ?? 10000),
    winRate: kpis.win_rate ?? 0,
    pf: kpis.profit_factor ?? 0,
    maxDDPct: kpis.max_dd_pct ?? 0,
    avgDailyPnl: kpis.avg_daily_pnl ?? 0,
    trades: kpis.total_trades ?? 0,
    tradesPerDay: kpis.trades_per_day ?? 0,
    fwdDays: 0,
    fwdTotalPnl: null,
    fwdWinRate: null,
    fwdPF: null,
    fwdMaxDDPct: null,
    fwdAvgDailyPnl: null,
    alertTotalPnl: null,
    alertWinRate: null,
    alertPF: null,
    alertMaxDDPct: null,
    alertAvgDailyPnl: null,
    fwdSD: 0,
    alertSD: 0,
    tags: p.tags || [],
    createdAt: p.created_at || '',
    btDays: p.bt_days ?? 30,
  };
}

function WiredV5() {
  const { data: apiPortfolios, isLoading } = usePortfolios();
  const deleteMutation = useDeletePortfolio();
  const duplicateMutation = useDuplicatePortfolio();

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading portfolios...
      </div>
    );
  }

  const v5Portfolios = apiPortfolios
    ? apiPortfolios.map(apiToV5Portfolio)
    : undefined;

  return (
    <V5
      apiPortfolios={v5Portfolios}
      onDelete={(id) => deleteMutation.mutate(Number(id))}
      onDuplicate={(id) => duplicateMutation.mutate(Number(id))}
    />
  );
}

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Two-column grid of portfolio cards with mini equity curves, P&L/DD/WR/strategy count KPIs, and active status badges.',
      rationale: 'First pass establishing the portfolio list with clickable cards that link to detail views.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Rich portfolio cards with deploy status, compliance checks, requirement set badges, and full KPIs.',
      rationale: 'Every Streamlit portfolio list feature: richer cards with live/paper/paused status, deployment indicator, requirement set badge, compliance check (passing/violations), full 6-KPI row, mini equity curves, and Clone/Delete actions alongside View/Edit.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v5-refined',
      name: 'Refined (My Strategies Style)',
      description: 'Portfolio cards matching My Strategies V5 style: pulsing enabled dot, status badge, final balance, meta line (strategies/balance/scaling/risk/trades per day/webhook template), strategy pills, 3-segment equity curve with HWM/Edge Check/Confidence Bands, 6-KPI row or comparison table (Overall/BT vs FWD/FWD vs Alerts/BT vs Alerts), requirement set compliance badge, action row with View/Edit/Clone/Delete + enabled toggle + bulk select checkbox.',
      rationale: 'Consistent visual language across strategies and portfolios. Same equity curve controls, KPI comparison modes, bulk actions, and card structure. Portfolio-specific KPIs: P&L ($), Win Rate, PF, Max DD (%), Avg Daily P&L ($), Trades. Enabled toggle replaces alert tracking toggle.',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v5-wired',
      name: 'Production (Live)',
      description: 'Portfolio list wired to real API data from FastAPI portfolios endpoint.',
      rationale: 'Production version with live data — same UI as V5 but fetches portfolios from the API.',
    },
    component: WiredV5,
  },
];

export default function PortfoliosPage() {
  return <VersionedPage pageKey="portfolios" versions={versions} />;
}
