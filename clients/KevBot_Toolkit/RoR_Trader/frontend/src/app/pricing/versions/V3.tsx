'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const tiers = [
  { name: 'Free Subscriber', price: 0, period: 'forever', features: ['Browse marketplace', 'Free portfolio rotation', 'Basic webhook setup', '1 active subscription', 'Community forum access'], cta: 'Get Started', popular: false },
  { name: 'Strategy Builder', price: 29, period: 'month', features: ['Strategy Builder access', 'Mass Backtesting & forward testing', 'Up to 10 subscriptions', 'License & publish strategies', 'Marketplace pack access'], cta: 'Start Building', popular: false },
  { name: 'Portfolio Builder', price: 49, period: 'month', features: ['Portfolio construction tools', 'Prop firm compliance checker', 'Publish & sell portfolios', 'Revenue sharing on sales', 'Unlimited subscriptions'], cta: 'Start Creating', popular: true },
  { name: 'Pack Creator', price: 79, period: 'month', features: ['Indicator dev tools', 'Code editor & testing', 'Publish confluence packs', 'Pack royalties on sales', 'API access & priority support'], cta: 'Go Pro', popular: false },
];

export default function PricingV3() {
  const [activePlan] = useState('Free Subscriber');

  return (
    <div>
      <PageHeader
        title="Pricing & Plans"
        subtitle="Simple, transparent pricing for every trader"
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {tiers.map((tier) => {
          const isActive = tier.name === activePlan;

          return (
            <Card key={tier.name} className={`relative flex flex-col ${tier.popular ? 'ring-2' : ''}`}>
              {tier.popular && (
                <div
                  className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-semibold"
                  style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                >
                  Popular
                </div>
              )}

              <h3 className="text-sm font-semibold mb-1" style={{ color: 'var(--text-secondary)' }}>{tier.name}</h3>
              <div className="flex items-baseline gap-1 mb-4">
                <span className="text-2xl font-bold">{tier.price === 0 ? 'Free' : `$${tier.price}`}</span>
                {tier.price > 0 && (
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/{tier.period}</span>
                )}
              </div>

              <ul className="space-y-2 mb-5 flex-1">
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-2 text-xs">
                    <span style={{ color: 'var(--green)' }}>&#10003;</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                className="w-full py-2 rounded-lg text-sm font-medium"
                style={{
                  background: isActive ? 'transparent' : tier.popular ? 'var(--accent)' : 'var(--bg-input)',
                  color: isActive ? 'var(--text-muted)' : tier.popular ? 'var(--bg-primary)' : 'var(--text-primary)',
                  border: `1px solid ${isActive ? 'var(--border)' : tier.popular ? 'var(--accent)' : 'var(--border)'}`,
                }}
                disabled={isActive}
              >
                {isActive ? 'Current Plan' : tier.cta}
              </button>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
