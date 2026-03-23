'use client';

import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

export default function SettingsProfileV3() {
  return (
    <div>
      <PageHeader
        title="Profile"
        subtitle="Account overview"
        backHref="/settings"
      />

      <Card>
        <div className="flex items-center gap-4 mb-5">
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold"
            style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
          >
            K
          </div>
          <div className="flex-1">
            <p className="font-semibold">Kevin</p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>kevin@rortrader.com</p>
          </div>
          <span
            className="px-2 py-0.5 rounded-full text-xs font-medium"
            style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
          >
            Free Subscriber
          </span>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Plan</span>
            <span className="text-sm font-medium">Free</span>
          </div>
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Member Since</span>
            <span className="text-sm font-medium">Oct 2025</span>
          </div>
          <div className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--border)' }}>
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Subscriptions</span>
            <span className="text-sm font-medium">1 active</span>
          </div>
          <div className="flex items-center justify-between py-2">
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>Alerts Received</span>
            <span className="text-sm font-medium">42</span>
          </div>
        </div>

        <div className="mt-5 flex gap-3">
          <button
            className="flex-1 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
          >
            Upgrade Plan
          </button>
          <button
            className="flex-1 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
          >
            Edit Profile
          </button>
        </div>
      </Card>
    </div>
  );
}
