'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsAccountV2() {
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [twoFaEnabled, setTwoFaEnabled] = useState(false);

  return (
    <div>
      <PageHeader title="Account" subtitle="Manage your profile, security, and usage" />

      <div className="max-w-3xl space-y-6">
        {/* Profile Card */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Profile
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Email</span>
              <span className="text-sm font-medium">kevin@example.com</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Member Since</span>
              <span className="text-sm">March 2026</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Subscription</span>
              <span
                className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
              >
                Pro
              </span>
            </div>
          </div>
        </Card>

        {/* Usage Stats */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Usage Statistics
          </h3>
          <div className="grid grid-cols-4 gap-4">
            <MetricCard label="Strategies Created" value="24" />
            <MetricCard label="Backtests Run" value="147" />
            <MetricCard label="Alerts Fired" value="892" />
            <MetricCard label="Portfolios" value="3" />
          </div>
        </Card>

        {/* API Usage */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            API Usage (This Month)
          </h3>
          <div className="space-y-3">
            {[
              { label: 'Polygon.io REST Calls', used: 45200, limit: 100000 },
              { label: 'Polygon.io WebSocket Hours', used: 312, limit: 744 },
              { label: 'Supabase DB Operations', used: 8400, limit: 50000 },
            ].map((item) => {
              const pct = Math.round((item.used / item.limit) * 100);
              return (
                <div key={item.label}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                    <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {item.used.toLocaleString()} / {item.limit.toLocaleString()}
                    </span>
                  </div>
                  <div className="h-2 rounded-full" style={{ background: 'var(--bg-input)' }}>
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${pct}%`,
                        background: pct > 80 ? 'var(--red)' : pct > 60 ? 'var(--yellow, #e5a813)' : 'var(--accent)',
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        {/* Security */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Security
          </h3>
          <div className="space-y-4">
            {/* Change Password */}
            <div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">Password</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Last changed 30 days ago</p>
                </div>
                <button
                  onClick={() => setShowPasswordForm(!showPasswordForm)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
                >
                  {showPasswordForm ? 'Cancel' : 'Change Password'}
                </button>
              </div>
              {showPasswordForm && (
                <div className="mt-3 space-y-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  {['Current Password', 'New Password', 'Confirm New Password'].map((label) => (
                    <div key={label}>
                      <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>{label}</label>
                      <input
                        type="password"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                    </div>
                  ))}
                  <button
                    className="px-4 py-2 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--accent)', color: 'white' }}
                  >
                    Update Password
                  </button>
                </div>
              )}
            </div>

            <div className="border-t" style={{ borderColor: 'var(--border)' }} />

            {/* 2FA */}
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Two-Factor Authentication</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {twoFaEnabled ? 'Enabled — authenticator app' : 'Not enabled'}
                </p>
              </div>
              <button
                onClick={() => setTwoFaEnabled(!twoFaEnabled)}
                className="px-3 py-1.5 rounded-lg text-xs font-medium"
                style={{
                  background: twoFaEnabled ? 'var(--red-muted)' : 'var(--green-muted)',
                  color: twoFaEnabled ? 'var(--red)' : 'var(--green)',
                }}
              >
                {twoFaEnabled ? 'Disable' : 'Enable'}
              </button>
            </div>
          </div>
        </Card>

        {/* Data Export & Actions */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            Data & Account Actions
          </h3>
          <div className="flex gap-3">
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
            >
              Export All Data
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
            >
              Sign Out
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}
