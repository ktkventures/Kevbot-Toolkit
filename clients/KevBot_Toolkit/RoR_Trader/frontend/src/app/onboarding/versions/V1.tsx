'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function OnboardingV1() {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <div className="max-w-2xl mx-auto">
      <PageHeader
        title="Welcome to RoR Trader"
        subtitle="Let's get you started. What brings you here?"
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
        <Card
          className={`cursor-pointer transition-all ${selected === 'trade' ? 'ring-2' : ''}`}
        >
          <div
            onClick={() => setSelected('trade')}
            className="text-center py-4"
          >
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center text-3xl mx-auto mb-4"
              style={{
                background: selected === 'trade' ? 'var(--green-muted)' : 'var(--bg-input)',
                border: `2px solid ${selected === 'trade' ? 'var(--green)' : 'var(--border)'}`,
              }}
            >
              $
            </div>
            <h3 className="text-lg font-semibold mb-2">I want to trade</h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Subscribe to proven portfolios and receive automated trade alerts via webhook.
            </p>
            <p className="text-xs mt-2 font-medium" style={{ color: 'var(--green)' }}>Free to start</p>
          </div>
        </Card>

        <Card
          className={`cursor-pointer transition-all ${selected === 'create' ? 'ring-2' : ''}`}
        >
          <div
            onClick={() => setSelected('create')}
            className="text-center py-4"
          >
            <div
              className="w-16 h-16 rounded-full flex items-center justify-center text-3xl mx-auto mb-4"
              style={{
                background: selected === 'create' ? 'var(--accent-muted)' : 'var(--bg-input)',
                border: `2px solid ${selected === 'create' ? 'var(--accent)' : 'var(--border)'}`,
              }}
            >
              +
            </div>
            <h3 className="text-lg font-semibold mb-2">I want to create</h3>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Build strategies, curate portfolios, develop indicator packs, and earn revenue.
            </p>
            <p className="text-xs mt-2 font-medium" style={{ color: 'var(--accent)' }}>From $29/mo</p>
          </div>
        </Card>
      </div>

      <button
        className="w-full py-3 rounded-lg text-sm font-medium transition-colors"
        style={{
          background: selected ? 'var(--accent)' : 'var(--bg-input)',
          color: selected ? 'var(--bg-primary)' : 'var(--text-muted)',
          border: `1px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
        }}
        disabled={!selected}
      >
        {selected ? 'Continue' : 'Choose a path to continue'}
      </button>
    </div>
  );
}
