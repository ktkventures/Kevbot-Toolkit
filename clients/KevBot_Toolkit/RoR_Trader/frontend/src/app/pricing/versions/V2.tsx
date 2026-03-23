'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const tiers = [
  { name: 'Free Subscriber', price: 0, annual: 0, period: 'forever', features: ['Browse marketplace', 'Free portfolio rotation', 'Basic webhook setup', '1 active subscription', 'Community forum access'], cta: 'Get Started', popular: false },
  { name: 'Strategy Builder', price: 29, annual: 23, period: 'month', features: ['Everything in Free', 'Strategy Builder access', 'Mass Backtesting', 'Forward testing', 'Up to 10 active subscriptions', 'License marketplace packs', 'Publish strategies'], cta: 'Start Building', popular: false },
  { name: 'Portfolio Builder', price: 49, annual: 39, period: 'month', features: ['Everything in Strategy Builder', 'Portfolio construction tools', 'Prop firm compliance checker', 'Publish portfolios', 'Revenue sharing on sales', 'Priority webhook delivery', 'Unlimited subscriptions'], cta: 'Start Creating', popular: true },
  { name: 'Pack Creator', price: 79, annual: 63, period: 'month', features: ['Everything in Portfolio Builder', 'Indicator development tools', 'Code editor & testing', 'Publish confluence packs', 'Pack royalties on sales', 'API access', 'Priority support'], cta: 'Go Pro', popular: false },
];

const featureMatrix: { feature: string; tiers: boolean[] }[] = [
  { feature: 'Browse marketplace', tiers: [true, true, true, true] },
  { feature: 'Free portfolio rotation', tiers: [true, true, true, true] },
  { feature: 'Basic webhook setup', tiers: [true, true, true, true] },
  { feature: 'Community forum access', tiers: [true, true, true, true] },
  { feature: 'Strategy Builder', tiers: [false, true, true, true] },
  { feature: 'Mass Backtesting', tiers: [false, true, true, true] },
  { feature: 'Forward testing', tiers: [false, true, true, true] },
  { feature: 'License marketplace packs', tiers: [false, true, true, true] },
  { feature: 'Publish strategies', tiers: [false, true, true, true] },
  { feature: 'Portfolio construction', tiers: [false, false, true, true] },
  { feature: 'Prop firm compliance', tiers: [false, false, true, true] },
  { feature: 'Publish portfolios', tiers: [false, false, true, true] },
  { feature: 'Revenue sharing', tiers: [false, false, true, true] },
  { feature: 'Priority webhooks', tiers: [false, false, true, true] },
  { feature: 'Unlimited subscriptions', tiers: [false, false, true, true] },
  { feature: 'Indicator development', tiers: [false, false, false, true] },
  { feature: 'Code editor & testing', tiers: [false, false, false, true] },
  { feature: 'Publish packs', tiers: [false, false, false, true] },
  { feature: 'Pack royalties', tiers: [false, false, false, true] },
  { feature: 'API access', tiers: [false, false, false, true] },
  { feature: 'Priority support', tiers: [false, false, false, true] },
];

const faqs = [
  { q: "What's included in the free tier?", a: 'Free users can browse the marketplace, access admin-curated free portfolio rotations, set up a basic webhook, and participate in the community forum. You get real value from day one.' },
  { q: 'Can I upgrade or downgrade anytime?', a: 'Yes. Upgrades take effect immediately with prorated billing. Downgrades take effect at the end of your current billing period. No long-term contracts.' },
  { q: 'How does revenue sharing work?', a: 'When subscribers pay for a portfolio, revenue flows through the creator chain: platform fee (15%), portfolio creator (50% of remainder), strategy creators (25%), pack creators (10%), and a community reserve.' },
  { q: 'Is there a money-back guarantee?', a: 'Yes. All paid plans come with a 14-day money-back guarantee. If you are not satisfied, contact support for a full refund.' },
  { q: 'What payment methods are accepted?', a: 'We accept all major credit cards, debit cards, and PayPal. Annual billing is also available for a 20% discount.' },
];

export default function PricingV2() {
  const [billing, setBilling] = useState<'monthly' | 'annual'>('monthly');
  const [activePlan] = useState('Free Subscriber');
  const [showComparison, setShowComparison] = useState(false);
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  return (
    <div>
      <PageHeader
        title="Pricing & Plans"
        subtitle="Choose the plan that fits your trading journey"
      />

      {/* Billing Toggle */}
      <div className="flex items-center justify-center gap-4 mb-8">
        <span
          className="text-sm font-medium"
          style={{ color: billing === 'monthly' ? 'var(--text-primary)' : 'var(--text-muted)' }}
        >
          Monthly
        </span>
        <button
          onClick={() => setBilling(billing === 'monthly' ? 'annual' : 'monthly')}
          className="relative w-12 h-6 rounded-full transition-colors"
          style={{ background: billing === 'annual' ? 'var(--accent)' : 'var(--bg-input)' }}
        >
          <div
            className="absolute top-1 w-4 h-4 rounded-full transition-transform"
            style={{
              background: 'var(--text-primary)',
              left: billing === 'annual' ? '28px' : '4px',
            }}
          />
        </button>
        <span
          className="text-sm font-medium"
          style={{ color: billing === 'annual' ? 'var(--text-primary)' : 'var(--text-muted)' }}
        >
          Annual
        </span>
        {billing === 'annual' && (
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
            Save 20%
          </span>
        )}
      </div>

      {/* Tier Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        {tiers.map((tier) => {
          const price = billing === 'annual' ? tier.annual : tier.price;
          const isActive = tier.name === activePlan;

          return (
            <Card key={tier.name} className={`relative ${tier.popular ? 'ring-2' : ''}`}>
              {tier.popular && (
                <div
                  className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-semibold"
                  style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                >
                  Most Popular
                </div>
              )}
              {isActive && (
                <div
                  className="absolute -top-3 right-4 px-3 py-1 rounded-full text-xs font-semibold"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  Current Plan
                </div>
              )}

              <div className="text-center mb-5">
                <h3 className="text-base font-semibold mb-2">{tier.name}</h3>
                <div className="flex items-baseline justify-center gap-1">
                  <span className="text-3xl font-bold">
                    {price === 0 ? 'Free' : `$${price}`}
                  </span>
                  {price > 0 && (
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      /{billing === 'annual' ? 'mo' : tier.period}
                    </span>
                  )}
                </div>
                {billing === 'annual' && tier.price > 0 && (
                  <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                    ${price * 12}/year (billed annually)
                  </p>
                )}
              </div>

              <ul className="space-y-2.5 mb-5">
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
                  background: isActive ? 'var(--bg-input)' : tier.popular ? 'var(--accent)' : 'var(--bg-input)',
                  color: isActive ? 'var(--text-muted)' : tier.popular ? 'var(--bg-primary)' : 'var(--text-primary)',
                  border: `1px solid ${tier.popular ? 'var(--accent)' : 'var(--border)'}`,
                }}
                disabled={isActive}
              >
                {isActive ? 'Current Plan' : tier.cta}
              </button>
            </Card>
          );
        })}
      </div>

      {/* Money-back guarantee */}
      <div className="text-center mb-8">
        <span
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <span style={{ color: 'var(--green)' }}>&#9679;</span>
          14-day money-back guarantee on all paid plans
        </span>
      </div>

      {/* Compare Plans */}
      <Card className="mb-8">
        <button
          onClick={() => setShowComparison(!showComparison)}
          className="w-full flex items-center justify-between text-left"
        >
          <h3 className="text-lg font-semibold">Compare Plans</h3>
          <span style={{ color: 'var(--text-muted)' }}>{showComparison ? '−' : '+'}</span>
        </button>

        {showComparison && (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  <th className="text-left py-3 pr-4" style={{ color: 'var(--text-muted)' }}>Feature</th>
                  {tiers.map((t) => (
                    <th key={t.name} className="text-center py-3 px-3" style={{ color: 'var(--text-secondary)' }}>
                      {t.name.split(' ')[0]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {featureMatrix.map((row) => (
                  <tr key={row.feature} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td className="py-2.5 pr-4" style={{ color: 'var(--text-secondary)' }}>{row.feature}</td>
                    {row.tiers.map((available, i) => (
                      <td key={i} className="text-center py-2.5 px-3">
                        <span style={{ color: available ? 'var(--green)' : 'var(--text-muted)' }}>
                          {available ? '&#10003;' : '&#10007;'}
                        </span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* FAQ */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Frequently Asked Questions</h3>
        <div className="space-y-2">
          {faqs.map((faq, i) => (
            <div key={i} className="rounded-lg" style={{ border: '1px solid var(--border)' }}>
              <button
                onClick={() => setExpandedFaq(expandedFaq === i ? null : i)}
                className="w-full flex items-center justify-between px-4 py-3 text-sm text-left font-medium"
              >
                {faq.q}
                <span style={{ color: 'var(--text-muted)' }}>{expandedFaq === i ? '−' : '+'}</span>
              </button>
              {expandedFaq === i && (
                <div className="px-4 pb-3 text-sm" style={{ color: 'var(--text-secondary)' }}>
                  {faq.a}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
