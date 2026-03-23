'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const currentPlan = {
  name: 'Free Subscriber',
  price: 0,
  period: 'forever',
  role: 'Subscriber',
  nextTier: 'Strategy Builder',
  nextPrice: 29,
};

export default function SettingsProfileV1() {
  return (
    <div>
      <PageHeader
        title="Profile & Plan"
        subtitle="Your account and subscription"
        backHref="/settings"
      />

      <Card className="mb-6">
        <div className="flex items-center gap-4 mb-4">
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center text-xl font-bold"
            style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
          >
            K
          </div>
          <div>
            <p className="text-lg font-semibold">Kevin</p>
            <span
              className="px-2 py-0.5 rounded-full text-xs font-medium"
              style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
            >
              {currentPlan.role}
            </span>
          </div>
        </div>

        <div className="p-4 rounded-lg mb-4" style={{ background: 'var(--bg-input)' }}>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Current Plan</p>
              <p className="text-lg font-bold">{currentPlan.name}</p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {currentPlan.price === 0 ? 'Free forever' : `$${currentPlan.price}/${currentPlan.period}`}
              </p>
            </div>
            <span
              className="px-3 py-1 rounded-full text-xs font-medium"
              style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
            >
              Active
            </span>
          </div>
        </div>

        <button
          className="w-full py-2.5 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
        >
          Upgrade to {currentPlan.nextTier} — ${currentPlan.nextPrice}/mo
        </button>
      </Card>
    </div>
  );
}
