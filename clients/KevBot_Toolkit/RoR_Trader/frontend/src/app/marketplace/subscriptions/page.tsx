'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Subscription list with name, status, cost, performance, webhook indicator, pause/resume toggle, and configure button.',
      rationale: 'First pass — establish the core subscription management layout. Each card shows what matters: is it active, is the webhook working, and how is performance. Pause/Resume toggle and Configure button provide the primary actions.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Full subscription detail with trades, WR, P&L, alert count, inline risk-per-trade editing, recent alert history, and 4-step webhook configuration wizard (broker selection, URL input, connection test, risk setting).',
      rationale: 'Everything a subscriber needs in one place. The webhook wizard walks users through setup step by step — no guessing. Inline risk editing lets users adjust on the fly. Recent alerts give confidence the system is working. Full P&L and trade stats show whether the subscription is paying for itself.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Compact table view with status/webhook dots, inline webhook URL field (paste-and-test), toggle switches, total cost at top.',
      rationale: 'No frills. Each subscription is one row with all the info: status, webhook health, cost, alert count. Webhook URL is edited inline — click the field, paste, test, done. The toggle switch for pause/resume is right there. Total cost at top. A trader can manage 10 subscriptions in 30 seconds.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Subscription dashboard with health gauges, combined P&L chart, webhook pulse monitor, cost-vs-revenue analysis, and smart recommendations.',
      rationale: 'Reimagined as a portfolio health dashboard. Health gauges give an instant visual read on each subscription. Combined P&L shows the aggregate value. Webhook pulse monitor feels alive — green dots pulse when connected. Cost vs revenue analysis answers the key question: am I making money? Smart recommendations drive additional subscriptions based on what the user already has.',
    },
    component: V4,
  },
];

export default function SubscriptionsPage() {
  return <VersionedPage pageKey="subscriptions" versions={versions} />;
}
