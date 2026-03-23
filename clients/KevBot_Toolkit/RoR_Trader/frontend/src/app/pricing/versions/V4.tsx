'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const tiers = [
  { name: 'Free Subscriber', price: 0, annual: 0, period: 'forever', color: 'var(--text-muted)', features: ['Browse marketplace', 'Free portfolio rotation', 'Basic webhook setup', '1 active subscription', 'Community forum access'], cta: 'Get Started', popular: false },
  { name: 'Strategy Builder', price: 29, annual: 23, period: 'month', color: 'var(--blue)', features: ['Everything in Free', 'Strategy Builder access', 'Mass Backtesting', 'Forward testing', 'Up to 10 active subscriptions', 'License marketplace packs', 'Publish strategies'], cta: 'Start Building', popular: false },
  { name: 'Portfolio Builder', price: 49, annual: 39, period: 'month', color: 'var(--accent)', features: ['Everything in Strategy Builder', 'Portfolio construction tools', 'Prop firm compliance checker', 'Publish portfolios', 'Revenue sharing on sales', 'Priority webhook delivery', 'Unlimited subscriptions'], cta: 'Start Creating', popular: true },
  { name: 'Pack Creator', price: 79, annual: 63, period: 'month', color: 'var(--purple)', features: ['Everything in Portfolio Builder', 'Indicator development tools', 'Code editor & testing', 'Publish confluence packs', 'Pack royalties on sales', 'API access', 'Priority support'], cta: 'Go Pro', popular: false },
];

const testimonials = [
  { name: 'Alex K.', role: 'Portfolio Builder', quote: 'I replaced my day job income within 6 months of publishing my first portfolio.', earnings: '$4,200/mo' },
  { name: 'Sarah M.', role: 'Pack Creator', quote: 'My custom EMA pack is used in 40+ strategies. Royalties add up fast.', earnings: '$1,800/mo' },
  { name: 'James T.', role: 'Subscriber', quote: 'Hands-free trading. I subscribe to two portfolios and let the webhooks do the work.', earnings: 'N/A' },
];

export default function PricingV4() {
  const [billing, setBilling] = useState<'monthly' | 'annual'>('monthly');
  const [hoveredTier, setHoveredTier] = useState<string | null>(null);
  const [activePlan] = useState('Free Subscriber');

  // Revenue calculator
  const [subscribers, setSubscribers] = useState(50);
  const [subPrice, setSubPrice] = useState(30);

  const earnings = useMemo(() => {
    const gross = subscribers * subPrice;
    const platformFee = gross * 0.15;
    const creatorShare = (gross - platformFee) * 0.50;
    return { gross, platformFee, creatorShare, net: creatorShare };
  }, [subscribers, subPrice]);

  return (
    <div>
      <PageHeader
        title="Pricing & Plans"
        subtitle="Build, create, and earn on the platform designed for systematic traders"
      />

      {/* Billing Toggle */}
      <div className="flex items-center justify-center gap-4 mb-8">
        <span className="text-sm" style={{ color: billing === 'monthly' ? 'var(--text-primary)' : 'var(--text-muted)' }}>Monthly</span>
        <button
          onClick={() => setBilling(billing === 'monthly' ? 'annual' : 'monthly')}
          className="relative w-12 h-6 rounded-full transition-colors"
          style={{ background: billing === 'annual' ? 'var(--accent)' : 'var(--bg-input)' }}
        >
          <div
            className="absolute top-1 w-4 h-4 rounded-full transition-all duration-200"
            style={{ background: 'var(--text-primary)', left: billing === 'annual' ? '28px' : '4px' }}
          />
        </button>
        <span className="text-sm" style={{ color: billing === 'annual' ? 'var(--text-primary)' : 'var(--text-muted)' }}>Annual</span>
        {billing === 'annual' && (
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>Save 20%</span>
        )}
      </div>

      {/* Interactive Tier Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-10">
        {tiers.map((tier) => {
          const isActive = tier.name === activePlan;
          const isHovered = hoveredTier === tier.name;
          const price = billing === 'annual' ? tier.annual : tier.price;

          return (
            <div
              key={tier.name}
              className="relative rounded-xl border p-5 transition-all duration-300"
              style={{
                background: 'var(--bg-card)',
                borderColor: isHovered ? tier.color : 'var(--border)',
                boxShadow: isHovered ? `0 0 20px ${tier.color}33` : 'var(--card-shadow)',
                transform: isHovered ? 'translateY(-4px)' : 'translateY(0)',
              }}
              onMouseEnter={() => setHoveredTier(tier.name)}
              onMouseLeave={() => setHoveredTier(null)}
            >
              {tier.popular && (
                <div
                  className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-semibold"
                  style={{ background: tier.color, color: 'var(--bg-primary)' }}
                >
                  Most Popular
                </div>
              )}

              {/* Tier color bar */}
              <div className="h-1 rounded-full mb-4" style={{ background: tier.color, opacity: isHovered ? 1 : 0.4, transition: 'opacity 0.3s' }} />

              <h3 className="text-base font-semibold mb-1">{tier.name}</h3>
              <div className="flex items-baseline gap-1 mb-4">
                <span className="text-3xl font-bold" style={{ color: isHovered ? tier.color : 'var(--text-primary)', transition: 'color 0.3s' }}>
                  {price === 0 ? 'Free' : `$${price}`}
                </span>
                {price > 0 && <span className="text-sm" style={{ color: 'var(--text-muted)' }}>/mo</span>}
              </div>

              <ul className="space-y-2.5 mb-5">
                {tier.features.map((feature, fi) => (
                  <li
                    key={feature}
                    className="flex items-start gap-2 text-sm transition-all duration-200"
                    style={{
                      opacity: isHovered ? 1 : 0.7,
                      transform: isHovered ? `translateX(${Math.min(fi * 1, 4)}px)` : 'translateX(0)',
                      transitionDelay: `${fi * 30}ms`,
                    }}
                  >
                    <span style={{ color: tier.color }}>&#10003;</span>
                    <span style={{ color: 'var(--text-secondary)' }}>{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                className="w-full py-2.5 rounded-lg text-sm font-medium transition-all duration-200"
                style={{
                  background: isActive ? 'transparent' : isHovered ? tier.color : 'var(--bg-input)',
                  color: isActive ? 'var(--text-muted)' : isHovered ? 'var(--bg-primary)' : 'var(--text-primary)',
                  border: `1px solid ${isActive ? 'var(--border)' : isHovered ? tier.color : 'var(--border)'}`,
                }}
                disabled={isActive}
              >
                {isActive ? 'Current Plan' : tier.cta}
              </button>
            </div>
          );
        })}
      </div>

      {/* Revenue Calculator */}
      <Card className="mb-8">
        <h3 className="text-lg font-semibold mb-1">Revenue Calculator</h3>
        <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
          See how much you could earn as a portfolio creator
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-5">
            <div>
              <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
                Subscribers: <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{subscribers}</span>
              </label>
              <input
                type="range" min={1} max={500} value={subscribers}
                onChange={(e) => setSubscribers(Number(e.target.value))}
                className="w-full"
                style={{ accentColor: 'var(--accent)' }}
              />
              <div className="flex justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
                <span>1</span><span>250</span><span>500</span>
              </div>
            </div>

            <div>
              <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>
                Subscription Price: <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>${subPrice}/mo</span>
              </label>
              <input
                type="range" min={5} max={200} step={5} value={subPrice}
                onChange={(e) => setSubPrice(Number(e.target.value))}
                className="w-full"
                style={{ accentColor: 'var(--accent)' }}
              />
              <div className="flex justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
                <span>$5</span><span>$100</span><span>$200</span>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between text-sm py-2" style={{ borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Gross Revenue</span>
              <span className="font-semibold">${earnings.gross.toLocaleString()}/mo</span>
            </div>
            <div className="flex justify-between text-sm py-2" style={{ borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Platform Fee (15%)</span>
              <span style={{ color: 'var(--red)' }}>-${earnings.platformFee.toLocaleString()}/mo</span>
            </div>
            <div className="flex justify-between text-sm py-2" style={{ borderBottom: '1px solid var(--border)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Your Share (50%)</span>
              <span className="font-semibold" style={{ color: 'var(--green)' }}>${earnings.net.toLocaleString()}/mo</span>
            </div>
            <div className="flex justify-between text-base pt-2">
              <span className="font-semibold">Annual Earnings</span>
              <span className="font-bold text-lg" style={{ color: 'var(--green)' }}>${(earnings.net * 12).toLocaleString()}/yr</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Testimonials */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
        {testimonials.map((t) => (
          <Card key={t.name}>
            <div className="flex items-center gap-3 mb-3">
              <div
                className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
              >
                {t.name.charAt(0)}
              </div>
              <div>
                <p className="text-sm font-semibold">{t.name}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{t.role}</p>
              </div>
              {t.earnings !== 'N/A' && (
                <span className="ml-auto text-sm font-semibold" style={{ color: 'var(--green)' }}>{t.earnings}</span>
              )}
            </div>
            <p className="text-sm italic" style={{ color: 'var(--text-secondary)' }}>
              &ldquo;{t.quote}&rdquo;
            </p>
          </Card>
        ))}
      </div>

      {/* Money-back guarantee */}
      <div className="text-center">
        <span
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full text-sm"
          style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
        >
          <span style={{ color: 'var(--green)' }}>&#9679;</span>
          14-day money-back guarantee &middot; Cancel anytime &middot; No long-term contracts
        </span>
      </div>
    </div>
  );
}
