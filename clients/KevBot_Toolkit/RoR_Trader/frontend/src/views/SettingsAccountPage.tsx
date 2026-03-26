'use client';

/**
 * Account — Faithful copy of V2 design with mock data replaced by API hooks.
 * Source: src/app/settings/account/versions/V2.tsx (185 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import MetricCard from '@/components/MetricCard';
import { useAuth } from '@/providers/AuthProvider';
import { useSettings } from '@/hooks/queries/useSettings';

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsAccountPage() {
  /* ---- API hooks (must be before any early return) ---- */
  const { user, logout } = useAuth();
  const { data: settings } = useSettings();

  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [twoFaEnabled, setTwoFaEnabled] = useState(false);

  /* ---- Derived values ---- */
  const email = user?.email ?? '--';
  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
    : '--';

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
              <span className="text-sm font-medium">{email}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Member Since</span>
              <span className="text-sm">{memberSince}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Subscription</span>
              <span
                className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
              >
                {'{{tier}}'}
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
            <MetricCard label="Strategies Created" value={'{{strategies_created}}'} />
            <MetricCard label="Backtests Run" value={'{{backtests_run}}'} />
            <MetricCard label="Alerts Fired" value={'{{alerts_fired}}'} />
            <MetricCard label="Portfolios" value={'{{portfolios_count}}'} />
          </div>
        </Card>

        {/* API Usage */}
        <Card>
          <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>
            API Usage (This Month)
          </h3>
          <div className="space-y-3">
            {[
              { label: 'Polygon.io REST Calls', used: '{{polygon_rest_used}}', limit: '{{polygon_rest_limit}}' },
              { label: 'Polygon.io WebSocket Hours', used: '{{polygon_ws_used}}', limit: '{{polygon_ws_limit}}' },
              { label: 'Supabase DB Operations', used: '{{supabase_ops_used}}', limit: '{{supabase_ops_limit}}' },
            ].map((item) => (
              <div key={item.label}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {item.used} / {item.limit}
                  </span>
                </div>
                <div className="h-2 rounded-full" style={{ background: 'var(--bg-input)' }}>
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: '0%',
                      background: 'var(--accent)',
                    }}
                  />
                </div>
              </div>
            ))}
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
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{'{{password_last_changed}}'}</p>
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
              onClick={() => logout()}
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
