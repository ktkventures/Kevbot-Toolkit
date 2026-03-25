'use client';

import { useParams } from 'next/navigation';
import PageSwitch from '@/components/PageSwitch';
import PortfolioDetailPage from '@/views/PortfolioDetailPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';
import V5 from './versions/V5';
import { usePortfolio, usePortfolioAccount } from '@/hooks/queries/usePortfolios';

function WiredV5() {
  const params = useParams();
  const portfolioId = params?.id ? Number(params.id) : null;
  const { data: portfolio, isLoading: portfolioLoading } = usePortfolio(portfolioId);
  const { data: account, isLoading: accountLoading } = usePortfolioAccount(portfolioId);

  const isLoading = portfolioLoading || accountLoading;

  if (isLoading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-secondary)' }}>
        Loading portfolio...
      </div>
    );
  }

  return <V5 apiPortfolio={portfolio || undefined} apiAccount={account || undefined} />;
}

function LiveDetail() {
  const params = useParams();
  const id = params?.id ? Number(params.id) : 0;
  return <PortfolioDetailPage portfolioId={id} />;
}

const designVersions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: '7-tab portfolio detail with live dashboard, performance charts, strategy list, prop firm check, account ledger with daily detail modal, webhooks, and deploy.',
      rationale: 'First pass covering all portfolio detail tabs including the Account ledger with interactive daily detail modal.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'All 7 tabs with full content: Live Dashboard with benchmark, Performance analytics, Strategy health, Prop Firm compliance, Account ledger, Webhooks, Deploy.',
      rationale: 'Complete Streamlit portfolio detail.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: '4 tabs: Dashboard (live + performance), Strategies (health cards), Compliance (rules + balance), Settings (webhooks + deploy).',
      rationale: 'Consolidated 7 tabs to 4.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Command center layout: P&L contribution donut, risk allocation treemap, daily P&L calendar heatmap.',
      rationale: 'Visual-first portfolio dashboard.',
    },
    component: V4,
  },
  {
    meta: {
      id: 'v5-production',
      name: 'Production (Design Ref)',
      description: 'V2 Full Parity as baseline with Phase 39 webhook template integration.',
      rationale: 'Design reference with mock data.',
    },
    component: V5,
  },
  {
    meta: {
      id: 'v5-wired',
      name: 'Production (Wired)',
      description: 'Portfolio detail wired to real API data — fetches portfolio and account from FastAPI.',
      rationale: 'Production version with live data.',
    },
    component: WiredV5,
  },
];

export default function Page() {
  return <PageSwitch live={LiveDetail} versions={designVersions} pageKey="portfolio-detail" />;
}
