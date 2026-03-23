'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const profileData = {
  displayName: 'Kevin',
  email: 'kevin@rortrader.com',
  memberSince: '2025-10-01',
  role: 'Free Subscriber',
  level: 4,
  xp: 720,
  xpToNext: 1000,
  plan: 'Free Subscriber',
};

const badges = [
  { name: 'First Login', icon: '\u26A1', earned: true, rarity: 'Common' },
  { name: 'Week Warrior', icon: '\u{1F525}', earned: true, rarity: 'Common' },
  { name: 'Alert Receiver', icon: '\u{1F514}', earned: true, rarity: 'Common' },
  { name: 'Subscriber', icon: '\u2B50', earned: true, rarity: 'Uncommon' },
  { name: 'Strategy Creator', icon: '\u{1F3AF}', earned: false, rarity: 'Rare' },
  { name: 'Portfolio Builder', icon: '\u{1F4BC}', earned: false, rarity: 'Rare' },
  { name: 'Pack Developer', icon: '\u{1F9E9}', earned: false, rarity: 'Epic' },
  { name: 'Top Performer', icon: '\u{1F3C6}', earned: false, rarity: 'Legendary' },
];

const rarityColors: Record<string, string> = {
  Common: 'var(--text-muted)',
  Uncommon: 'var(--green)',
  Rare: 'var(--blue)',
  Epic: 'var(--purple)',
  Legendary: 'var(--orange)',
};

const activityTimeline = [
  { date: 'Today', action: 'Received 3 trading alerts', type: 'alert' },
  { date: 'Yesterday', action: 'Logged in - 7 day streak!', type: 'streak' },
  { date: 'Mar 19', action: 'Subscribed to Momentum Alpha', type: 'subscription' },
  { date: 'Mar 15', action: 'Configured webhook URL', type: 'setup' },
  { date: 'Mar 12', action: 'Joined RoR Trader', type: 'join' },
];

const typeColors: Record<string, string> = {
  alert: 'var(--accent)',
  streak: 'var(--orange)',
  subscription: 'var(--green)',
  setup: 'var(--blue)',
  join: 'var(--purple)',
};

function AnimatedXPBar({ current, max }: { current: number; max: number }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const timer = setTimeout(() => setWidth((current / max) * 100), 100);
    return () => clearTimeout(timer);
  }, [current, max]);

  return (
    <div className="h-3 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
      <div
        className="h-full rounded-full transition-all duration-1000 ease-out"
        style={{ width: `${width}%`, background: 'linear-gradient(90deg, var(--accent), var(--green))' }}
      />
    </div>
  );
}

export default function SettingsProfileV4() {
  const [hoveredBadge, setHoveredBadge] = useState<string | null>(null);

  return (
    <div>
      <PageHeader
        title="Profile"
        subtitle="Your journey on RoR Trader"
        backHref="/settings"
      />

      {/* Visual Profile Card */}
      <div
        className="rounded-xl border p-6 mb-6 relative overflow-hidden"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
      >
        {/* Decorative gradient */}
        <div
          className="absolute top-0 left-0 right-0 h-24 opacity-20"
          style={{ background: 'linear-gradient(135deg, var(--accent), var(--purple), var(--blue))' }}
        />

        <div className="relative flex items-start gap-5">
          <div
            className="w-20 h-20 rounded-full flex items-center justify-center text-2xl font-bold shrink-0"
            style={{
              background: 'var(--accent)',
              color: 'var(--bg-primary)',
              boxShadow: '0 0 20px var(--accent-muted)',
            }}
          >
            K
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-xl font-bold">{profileData.displayName}</h2>
              <span className="px-2 py-0.5 rounded-full text-xs font-medium" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                {profileData.role}
              </span>
            </div>
            <p className="text-sm mb-3" style={{ color: 'var(--text-muted)' }}>{profileData.email}</p>

            {/* Level & XP */}
            <div className="flex items-center gap-3 mb-1">
              <span
                className="px-2 py-0.5 rounded text-xs font-bold"
                style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
              >
                LVL {profileData.level}
              </span>
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                {profileData.xp} / {profileData.xpToNext} XP
              </span>
            </div>
            <AnimatedXPBar current={profileData.xp} max={profileData.xpToNext} />
          </div>
        </div>
      </div>

      {/* Gamification Badges */}
      <Card className="mb-6">
        <h3 className="text-sm font-semibold mb-4">
          Badges ({badges.filter((b) => b.earned).length}/{badges.length})
        </h3>
        <div className="grid grid-cols-4 md:grid-cols-8 gap-3">
          {badges.map((badge) => (
            <div
              key={badge.name}
              className="relative text-center"
              onMouseEnter={() => setHoveredBadge(badge.name)}
              onMouseLeave={() => setHoveredBadge(null)}
            >
              <div
                className="w-14 h-14 rounded-xl flex items-center justify-center text-2xl mx-auto mb-1 transition-all duration-200"
                style={{
                  background: badge.earned ? `${rarityColors[badge.rarity]}22` : 'var(--bg-input)',
                  border: `2px solid ${badge.earned ? rarityColors[badge.rarity] : 'var(--border)'}`,
                  opacity: badge.earned ? 1 : 0.3,
                  transform: hoveredBadge === badge.name ? 'scale(1.1)' : 'scale(1)',
                  filter: badge.earned ? 'none' : 'grayscale(1)',
                }}
              >
                {badge.icon}
              </div>
              <p className="text-xs font-medium truncate">{badge.name}</p>
              {hoveredBadge === badge.name && (
                <div
                  className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded text-xs whitespace-nowrap z-10"
                  style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', color: rarityColors[badge.rarity] }}
                >
                  {badge.rarity}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Activity Timeline */}
      <Card className="mb-6">
        <h3 className="text-sm font-semibold mb-4">Activity Timeline</h3>
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-3 top-0 bottom-0 w-px" style={{ background: 'var(--border)' }} />

          <div className="space-y-4">
            {activityTimeline.map((event, i) => (
              <div key={i} className="flex items-start gap-4 relative">
                <div
                  className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 z-10"
                  style={{ background: typeColors[event.type] + '33', border: `2px solid ${typeColors[event.type]}` }}
                >
                  <div className="w-2 h-2 rounded-full" style={{ background: typeColors[event.type] }} />
                </div>
                <div className="flex-1 pb-1">
                  <p className="text-sm">{event.action}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{event.date}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Upgrade CTA */}
      <div
        className="rounded-xl p-5 text-center"
        style={{
          background: 'linear-gradient(135deg, var(--accent-muted), var(--bg-card))',
          border: '1px solid var(--accent)',
        }}
      >
        <p className="text-lg font-bold mb-1">Ready to level up?</p>
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Upgrade to Strategy Builder to unlock creation tools and earn XP faster
        </p>
        <button
          className="px-6 py-2.5 rounded-lg text-sm font-medium"
          style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
        >
          Upgrade to Strategy Builder — $29/mo
        </button>
      </div>
    </div>
  );
}
