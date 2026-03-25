'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { useStrategies } from '@/hooks/queries/useStrategies';
import { useDeleteStrategy, useDuplicateStrategy, useBulkDeleteStrategies } from '@/hooks/mutations/useStrategyMutations';

/**
 * Transform API strategy to V5 Strategy interface shape.
 */
function apiToV5Strategy(s: any): any {
  const kpis = s.kpis || {};
  const eqData = s.equity_curve_data || {};
  return {
    id: s.id,
    name: s.name || 'Untitled',
    symbol: s.symbol || '???',
    direction: s.direction || 'LONG',
    timeframe: s.timeframe || '1Min',
    status: s.forward_testing ? 'Forward Testing' : 'Backtest Only',
    tags: s.tags || [],
    monitored: s.alert_tracking_enabled || false,
    winRate: kpis.win_rate ?? 0,
    pf: kpis.profit_factor ?? 0,
    dailyR: kpis.daily_r ?? 0,
    dailyROI: 0,
    tpd: kpis.total_trades ? kpis.total_trades / Math.max(1, Math.round((kpis.total_trades || 0) / 5)) : 0,
    maxDD: kpis.max_r_drawdown ?? 0,
    totalR: kpis.total_r ?? 0,
    avgR: kpis.avg_r ?? 0,
    rSquared: kpis.r_squared ?? 0,
    btDays: s.data_days || 30,
    fwdSigma: 0,
    alertSigma: 0,
    alertAccuracy: null,
    equityCurve: eqData.cumulative_r || [],
    boundaryIndex: eqData.boundary_index ?? null,
    entry: s.entry_trigger_confluence_id || '',
    exit: (s.exit_trigger_confluence_ids || []).join(', '),
    stopSummary: '',
    targetSummary: '',
    confluences: (s.confluence || []).join(', '),
  };
}

function WiredV5() {
  const { data: apiStrategies, isLoading } = useStrategies();
  const deleteMutation = useDeleteStrategy();
  const duplicateMutation = useDuplicateStrategy();
  const bulkDeleteMutation = useBulkDeleteStrategies();

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading strategies...
      </div>
    );
  }

  const v5Strategies = apiStrategies
    ? apiStrategies.map(apiToV5Strategy)
    : undefined;

  return (
    <V5
      apiStrategies={v5Strategies}
      onDelete={(id) => deleteMutation.mutate(id)}
      onDuplicate={(id) => duplicateMutation.mutate(id)}
      onBulkDelete={(ids) => bulkDeleteMutation.mutate(ids)}
    />
  );
}

const versions = [
  { meta: { id: 'v1-initial', name: 'Initial Scaffold', description: 'Strategy list with cards.', rationale: 'First pass.' }, component: V1 },
  { meta: { id: 'v2-full-parity', name: 'Full Parity', description: 'All Streamlit features.', rationale: 'Direct port.' }, component: V2 },
  { meta: { id: 'v3-streamlined', name: 'Streamlined', description: 'Compact table view.', rationale: 'Maximum density.' }, component: V3 },
  { meta: { id: 'v4-creative', name: 'Creative', description: 'Command center.', rationale: 'Wow factor.' }, component: V4 },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Live)',
      description: 'Refined strategy list with real API data — live KPIs, equity curves, bulk actions.',
      rationale: 'Production version wired to FastAPI strategies endpoint.',
    },
    component: WiredV5,
  },
];

export default function StrategiesPage() {
  return <VersionedPage pageKey="strategies" versions={versions} />;
}
