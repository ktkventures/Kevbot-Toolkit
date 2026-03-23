'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const tiers = [
  {
    name: 'Free',
    price: 0,
    period: 'forever',
    features: ['Browse marketplace', 'Free portfolio rotation', 'Basic webhook setup', '1 active subscription', 'Community forum access'],
    cta: 'Get Started',
    popular: false,
  },
  {
    name: 'Creator',
    price: 29,
    period: 'month',
    features: ['Everything in Free', 'Strategy Builder access', 'Mass Backtesting', 'Forward testing', 'Up to 10 active subscriptions', 'License marketplace packs', 'Publish strategies'],
    cta: 'Start Building',
    popular: false,
  },
  {
    name: 'Pro Creator',
    price: 79,
    period: 'month',
    features: ['Everything in Creator', 'Portfolio construction tools', 'Prop firm compliance checker', 'Publish portfolios & packs', 'Revenue sharing on sales', 'API access', 'Priority support'],
    cta: 'Go Pro',
    popular: true,
  },
];

export default function PricingV1() {
  const [activePlan] = useState('Free');

  return (
    <div>
      <PageHeader
        title="Pricing & Plans"
        subtitle="Choose the plan that fits your trading journey"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {tiers.map((tier) => (
          <Card key={tier.name} className={tier.popular ? 'relative' : ''}>
            {tier.popular && (
              <div
                className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-semibold"
                style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
              >
                Most Popular
              </div>
            )}

            <div className="text-center mb-6">
              <h3 className="text-lg font-semibold mb-2">{tier.name}</h3>
              <div className="flex items-baseline justify-center gap-1">
                <span className="text-3xl font-bold">
                  {tier.price === 0 ? 'Free' : `$${tier.price}`}
                </span>
                {tier.price > 0 && (
                  <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    /{tier.period}
                  </span>
                )}
              </div>
            </div>

            <ul className="space-y-3 mb-6">
              {tier.features.map((feature) => (
                <li key={feature} className="flex items-start gap-2 text-sm">
                  <span style={{ color: 'var(--green)' }}>&#10003;</span>
                  <span style={{ color: 'var(--text-secondary)' }}>{feature}</span>
                </li>
              ))}
            </ul>

            <button
              className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors"
              style={{
                background: tier.name === activePlan ? 'var(--bg-input)' : tier.popular ? 'var(--accent)' : 'var(--bg-input)',
                color: tier.name === activePlan ? 'var(--text-muted)' : tier.popular ? 'var(--bg-primary)' : 'var(--text-primary)',
                border: '1px solid var(--border)',
              }}
              disabled={tier.name === activePlan}
            >
              {tier.name === activePlan ? 'Current Plan' : tier.cta}
            </button>
          </Card>
        ))}
      </div>
    </div>
  );
}
