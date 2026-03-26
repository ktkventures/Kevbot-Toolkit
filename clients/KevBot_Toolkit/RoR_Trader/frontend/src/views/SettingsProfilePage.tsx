'use client';

/**
 * Profile & Plan — Faithful copy of V2 design with mock data replaced by API hooks.
 * Source: src/app/settings/profile/versions/V2.tsx (274 lines)
 *
 * Data convention:
 *   Real value = live from API
 *   -- = wired but no data yet
 *   {{field}} = not wired, needs backend work
 */

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import MetricCard from '@/components/MetricCard';
import { useAuth } from '@/providers/AuthProvider';

/* ========================================================================= */
/* CONSTANTS — plan/feature/achievement data is structural, not mock          */
/* ========================================================================= */

const featureAccess = [
  { feature: 'Browse marketplace', has: true },
  { feature: 'Free portfolio rotation', has: true },
  { feature: 'Basic webhook setup', has: true },
  { feature: '1 active subscription', has: true },
  { feature: 'Strategy Builder', has: false },
  { feature: 'Mass Backtesting', has: false },
  { feature: 'Portfolio construction', has: false },
  { feature: 'Publish content', has: false },
];

const achievements = [
  { name: 'First Login', earned: true, description: 'Logged in for the first time' },
  { name: 'Week Streak', earned: true, description: '7 consecutive days active' },
  { name: 'First Alert', earned: true, description: 'Received your first trading alert' },
  { name: 'Subscriber', earned: true, description: 'Subscribed to a portfolio' },
  { name: 'Strategy Creator', earned: false, description: 'Create your first strategy' },
  { name: 'Power User', earned: false, description: '30 consecutive days active' },
  { name: 'Social Butterfly', earned: false, description: 'Share a portfolio on social media' },
  { name: 'Top Performer', earned: false, description: 'Achieve 60%+ win rate on live trades' },
];

export default function SettingsProfilePage() {
  /* ---- API hooks (must be before any early return) ---- */
  const { user } = useAuth();

  const [bio, setBio] = useState('');
  const [socialTwitter, setSocialTwitter] = useState('');
  const [socialDiscord, setSocialDiscord] = useState('');

  /* ---- Derived values from auth ---- */
  const displayName = user?.user_metadata?.display_name ?? user?.email?.split('@')[0] ?? '--';
  const email = user?.email ?? '--';
  const memberSince = user?.created_at
    ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
    : '--';
  const avatarInitial = displayName !== '--' ? displayName.charAt(0).toUpperCase() : '?';

  return (
    <div>
      <PageHeader
        title="Profile & Plan"
        subtitle="Manage your account, plan, and public profile"
        backHref="/settings"
      />

      <TabBar tabs={['Profile', 'Plan & Features', 'Achievements', 'Security']}>
        {(tab) => {
          if (tab === 'Profile') return (
            <div className="space-y-6">
              {/* Profile Card */}
              <Card>
                <div className="flex items-start gap-5">
                  <div
                    className="w-20 h-20 rounded-full flex items-center justify-center text-2xl font-bold shrink-0"
                    style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                  >
                    {avatarInitial}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h2 className="text-xl font-bold">{displayName}</h2>
                      <span className="px-2 py-0.5 rounded-full text-xs font-medium" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                        {'{{role}}'}
                      </span>
                    </div>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{email}</p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Member since {memberSince}</p>
                  </div>
                </div>
              </Card>

              {/* Usage Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard label="Strategies Created" value={'{{strategies_created}}'} />
                <MetricCard label="Backtests Run" value={'{{backtests_run}}'} />
                <MetricCard label="Alerts Received" value={'{{alerts_received}}'} />
                <MetricCard label="Login Streak" value={'{{login_streak}}'} />
              </div>

              {/* Public Creator Profile */}
              <Card>
                <h3 className="text-sm font-semibold mb-4">Public Creator Profile</h3>
                <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                  Visible when you publish content to the marketplace (requires Creator tier)
                </p>
                <div className="space-y-4">
                  <div>
                    <label className="text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>Bio</label>
                    <textarea
                      value={bio}
                      onChange={(e) => setBio(e.target.value)}
                      placeholder="Tell subscribers about your trading style..."
                      rows={3}
                      className="w-full px-3 py-2 rounded-lg text-sm resize-none"
                      style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>Twitter / X</label>
                      <input
                        type="text" value={socialTwitter} onChange={(e) => setSocialTwitter(e.target.value)}
                        placeholder="@handle"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                    </div>
                    <div>
                      <label className="text-xs block mb-1" style={{ color: 'var(--text-secondary)' }}>Discord</label>
                      <input
                        type="text" value={socialDiscord} onChange={(e) => setSocialDiscord(e.target.value)}
                        placeholder="username#1234"
                        className="w-full px-3 py-2 rounded-lg text-sm"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                      />
                    </div>
                  </div>
                  <button
                    className="px-4 py-2 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                  >
                    Save Profile
                  </button>
                </div>
              </Card>
            </div>
          );

          if (tab === 'Plan & Features') return (
            <div className="space-y-6">
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Current Plan</p>
                    <p className="text-xl font-bold">{'{{plan_name}}'}</p>
                  </div>
                  <span className="px-3 py-1 rounded-full text-xs font-medium" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>Active</span>
                </div>

                <h4 className="text-sm font-semibold mb-3">Feature Access</h4>
                <div className="space-y-2 mb-5">
                  {featureAccess.map((f) => (
                    <div key={f.feature} className="flex items-center gap-2">
                      <span style={{ color: f.has ? 'var(--green)' : 'var(--text-muted)' }}>
                        {f.has ? '\u2713' : '\u2717'}
                      </span>
                      <span className="text-sm" style={{ color: f.has ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                        {f.feature}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Upgrade CTA */}
                <div className="p-4 rounded-lg" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)' }}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-semibold">Upgrade to {'{{next_tier_name}}'}</p>
                    <p className="font-bold" style={{ color: 'var(--accent)' }}>{'{{next_tier_price}}'}/mo</p>
                  </div>
                  <ul className="space-y-1 mb-3">
                    {['Strategy Builder access', 'Mass Backtesting', 'Forward testing', 'Up to 10 subscriptions'].map((f) => (
                      <li key={f} className="text-xs flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
                        <span style={{ color: 'var(--accent)' }}>+</span> {f}
                      </li>
                    ))}
                  </ul>
                  <button
                    className="w-full py-2.5 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                  >
                    Upgrade Now
                  </button>
                </div>
              </Card>
            </div>
          );

          if (tab === 'Achievements') return (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {achievements.map((a) => (
                <Card key={a.name} className={a.earned ? '' : 'opacity-40'}>
                  <div className="text-center">
                    <div
                      className="w-12 h-12 rounded-full flex items-center justify-center text-lg mx-auto mb-2"
                      style={{
                        background: a.earned ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: a.earned ? 'var(--accent)' : 'var(--text-muted)',
                      }}
                    >
                      {a.earned ? '\u2605' : '\u2606'}
                    </div>
                    <p className="text-sm font-semibold">{a.name}</p>
                    <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>{a.description}</p>
                  </div>
                </Card>
              ))}
            </div>
          );

          if (tab === 'Security') return (
            <div className="space-y-5">
              <Card>
                <h3 className="text-sm font-semibold mb-3">Two-Factor Authentication</h3>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm">2FA is currently <span className="font-medium" style={{ color: 'var(--red)' }}>disabled</span></p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Add an extra layer of security to your account</p>
                  </div>
                  <button
                    className="px-3 py-1.5 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                  >
                    Enable 2FA
                  </button>
                </div>
              </Card>

              <Card>
                <h3 className="text-sm font-semibold mb-3">Active Sessions</h3>
                <div className="space-y-3">
                  {[
                    { device: '{{device}}', location: '{{location}}', current: true, lastSeen: 'Now' },
                  ].map((session, i) => (
                    <div key={i} className="flex items-center justify-between p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                      <div>
                        <p className="text-sm font-medium">{session.device}</p>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{session.location} &middot; {session.lastSeen}</p>
                      </div>
                      {session.current ? (
                        <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>Current</span>
                      ) : (
                        <button className="text-xs px-2 py-1 rounded" style={{ color: 'var(--red)', background: 'var(--red-muted)' }}>Revoke</button>
                      )}
                    </div>
                  ))}
                </div>
              </Card>

              <Card>
                <h3 className="text-sm font-semibold mb-3">Change Password</h3>
                <div className="space-y-3">
                  <input type="password" placeholder="Current password" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                  <input type="password" placeholder="New password" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                  <input type="password" placeholder="Confirm new password" className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} />
                  <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}>
                    Update Password
                  </button>
                </div>
              </Card>
            </div>
          );

          return null;
        }}
      </TabBar>
    </div>
  );
}
