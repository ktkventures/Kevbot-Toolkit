'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

type Screen = 'choose' | 'webhook' | 'done';

export default function OnboardingV3() {
  const [screen, setScreen] = useState<Screen>('choose');
  const [path, setPath] = useState<'trade' | 'create' | null>(null);
  const [webhookUrl, setWebhookUrl] = useState('');

  if (screen === 'choose') {
    return (
      <div className="max-w-lg mx-auto">
        <PageHeader
          title="Welcome to RoR Trader"
          subtitle="What would you like to do?"
        />

        <div className="space-y-3 mb-6">
          <button
            onClick={() => setPath('trade')}
            className="w-full p-4 rounded-xl text-left transition-all"
            style={{
              background: path === 'trade' ? 'var(--green-muted)' : 'var(--bg-card)',
              border: `2px solid ${path === 'trade' ? 'var(--green)' : 'var(--border)'}`,
            }}
          >
            <p className="font-semibold mb-0.5">Subscribe & Trade</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Follow proven portfolios with automated alerts &middot; Free</p>
          </button>

          <button
            onClick={() => setPath('create')}
            className="w-full p-4 rounded-xl text-left transition-all"
            style={{
              background: path === 'create' ? 'var(--accent-muted)' : 'var(--bg-card)',
              border: `2px solid ${path === 'create' ? 'var(--accent)' : 'var(--border)'}`,
            }}
          >
            <p className="font-semibold mb-0.5">Create & Earn</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Build strategies, portfolios, or packs &middot; From $29/mo</p>
          </button>
        </div>

        <button
          onClick={() => path && setScreen('webhook')}
          className="w-full py-3 rounded-lg text-sm font-medium"
          style={{
            background: path ? 'var(--accent)' : 'var(--bg-input)',
            color: path ? 'var(--bg-primary)' : 'var(--text-muted)',
          }}
          disabled={!path}
        >
          Next
        </button>
      </div>
    );
  }

  if (screen === 'webhook') {
    return (
      <div className="max-w-lg mx-auto">
        <PageHeader
          title="Set up your webhook"
          subtitle="Connect your broker to receive trade alerts"
        />

        <Card className="mb-6">
          <div className="space-y-4">
            <div>
              <label className="text-sm block mb-1" style={{ color: 'var(--text-secondary)' }}>Webhook URL</label>
              <input
                type="url"
                value={webhookUrl}
                onChange={(e) => setWebhookUrl(e.target.value)}
                placeholder="https://your-webhook-url.com/..."
                className="w-full px-3 py-2.5 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              />
            </div>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Don&apos;t have one? You can set this up later in Settings.
            </p>
          </div>
        </Card>

        <div className="flex gap-3">
          <button
            onClick={() => setScreen('choose')}
            className="px-4 py-3 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          >
            Back
          </button>
          <button
            onClick={() => setScreen('done')}
            className="flex-1 py-3 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
          >
            {webhookUrl ? 'Save & Continue' : 'Skip for Now'}
          </button>
        </div>
      </div>
    );
  }

  // Done screen
  return (
    <div className="max-w-lg mx-auto text-center py-12">
      <div
        className="w-16 h-16 rounded-full flex items-center justify-center text-3xl mx-auto mb-4"
        style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
      >
        &#10003;
      </div>
      <h2 className="text-xl font-bold mb-2">All done!</h2>
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
        {path === 'trade'
          ? 'Head to the marketplace to find a portfolio.'
          : 'Open the Strategy Builder to create your first strategy.'}
      </p>
      <button
        className="px-8 py-3 rounded-lg text-sm font-medium"
        style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
      >
        {path === 'trade' ? 'Browse Marketplace' : 'Open Strategy Builder'}
      </button>
    </div>
  );
}
