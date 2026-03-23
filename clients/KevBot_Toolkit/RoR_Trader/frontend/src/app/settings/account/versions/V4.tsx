'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

/* ========================================================================= */
/* MOCK DATA                                                                  */
/* ========================================================================= */

// Activity heatmap data: 52 weeks x 7 days, value 0-4 representing intensity
const generateHeatmapData = (): number[][] => {
  const weeks: number[][] = [];
  for (let w = 0; w < 26; w++) {
    const week: number[] = [];
    for (let d = 0; d < 7; d++) {
      // More activity on weekdays
      if (d >= 5) {
        week.push(Math.random() > 0.7 ? 1 : 0);
      } else {
        const r = Math.random();
        week.push(r > 0.8 ? 4 : r > 0.6 ? 3 : r > 0.35 ? 2 : r > 0.15 ? 1 : 0);
      }
    }
    weeks.push(week);
  }
  return weeks;
};

const heatmapData = generateHeatmapData();

interface Badge {
  name: string;
  description: string;
  earned: boolean;
  icon: string;
}

const badges: Badge[] = [
  { name: 'First Strategy', description: 'Created your first strategy', earned: true, icon: '1' },
  { name: '100 Backtests', description: 'Ran 100 backtests', earned: true, icon: 'B' },
  { name: 'First Profit', description: 'First profitable strategy', earned: true, icon: 'P' },
  { name: 'Portfolio Pro', description: 'Created 3 portfolios', earned: true, icon: 'F' },
  { name: 'Alert Master', description: 'Fired 500 alerts', earned: true, icon: 'A' },
  { name: 'Night Trader', description: 'Active at 3 AM', earned: true, icon: 'N' },
  { name: 'Multi-Asset', description: 'Traded stocks and crypto', earned: false, icon: 'M' },
  { name: '1000 Trades', description: 'Generated 1000 trade signals', earned: false, icon: 'T' },
];

/* ========================================================================= */
/* HELPERS                                                                    */
/* ========================================================================= */

function heatmapColor(value: number): string {
  if (value === 0) return 'var(--bg-input)';
  if (value === 1) return 'var(--accent)30';
  if (value === 2) return 'var(--accent)60';
  if (value === 3) return 'var(--accent)90';
  return 'var(--accent)';
}

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsAccountV4() {
  const [showAllBadges, setShowAllBadges] = useState(false);

  const displayBadges = showAllBadges ? badges : badges.filter((b) => b.earned);

  return (
    <div>
      <PageHeader
        title="Account"
        subtitle="Your profile, achievements, and activity overview"
      />

      <div className="max-w-3xl space-y-6">
        {/* Profile Card */}
        <Card>
          <div className="flex items-center gap-5">
            {/* Avatar placeholder */}
            <div
              className="w-20 h-20 rounded-full flex items-center justify-center text-2xl font-bold flex-shrink-0"
              style={{
                background: 'linear-gradient(135deg, var(--accent), var(--accent)80)',
                color: 'white',
                boxShadow: '0 0 20px var(--accent)30',
              }}
            >
              K
            </div>
            <div className="flex-1">
              <h2 className="text-lg font-semibold mb-0.5">kevin@example.com</h2>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>
                Member since March 2026 -- Pro tier
              </p>
              <div className="flex gap-4 text-xs" style={{ color: 'var(--text-secondary)' }}>
                <span><strong>24</strong> strategies</span>
                <span><strong>147</strong> backtests</span>
                <span><strong>892</strong> alerts</span>
                <span><strong>3</strong> portfolios</span>
              </div>
            </div>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium flex-shrink-0"
              style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
            >
              Sign Out
            </button>
          </div>
        </Card>

        {/* Activity Heatmap */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">Activity</h3>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              Last 6 months
            </span>
          </div>
          <div className="overflow-x-auto">
            <div className="flex gap-0.5" style={{ minWidth: 'fit-content' }}>
              {/* Day labels */}
              <div className="flex flex-col gap-0.5 mr-1 justify-between" style={{ width: 20 }}>
                {['', 'Mon', '', 'Wed', '', 'Fri', ''].map((d, i) => (
                  <span key={i} className="text-xs leading-none" style={{ color: 'var(--text-muted)', fontSize: '0.55rem', height: 11 }}>
                    {d}
                  </span>
                ))}
              </div>
              {/* Heatmap grid */}
              {heatmapData.map((week, wi) => (
                <div key={wi} className="flex flex-col gap-0.5">
                  {week.map((val, di) => (
                    <div
                      key={di}
                      className="rounded-sm"
                      style={{
                        width: 11,
                        height: 11,
                        background: heatmapColor(val),
                        border: val === 0 ? '1px solid var(--border)' : 'none',
                      }}
                      title={`Week ${wi + 1}, Day ${di + 1}: ${val} activities`}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
          {/* Legend */}
          <div className="flex items-center gap-2 mt-3 justify-end">
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Less</span>
            {[0, 1, 2, 3, 4].map((v) => (
              <div
                key={v}
                className="rounded-sm"
                style={{
                  width: 11,
                  height: 11,
                  background: heatmapColor(v),
                  border: v === 0 ? '1px solid var(--border)' : 'none',
                }}
              />
            ))}
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>More</span>
          </div>
        </Card>

        {/* Achievement Badges */}
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold">
              Achievements ({badges.filter((b) => b.earned).length}/{badges.length})
            </h3>
            <button
              onClick={() => setShowAllBadges(!showAllBadges)}
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
            >
              {showAllBadges ? 'Earned Only' : 'Show All'}
            </button>
          </div>
          <div className="grid grid-cols-4 gap-3">
            {displayBadges.map((badge) => (
              <div
                key={badge.name}
                className="rounded-xl border p-3 text-center transition-all"
                style={{
                  borderColor: badge.earned ? 'var(--accent)' : 'var(--border)',
                  background: badge.earned ? 'var(--accent-muted)' : 'var(--bg-input)',
                  opacity: badge.earned ? 1 : 0.5,
                }}
              >
                {/* Badge icon */}
                <div
                  className="w-10 h-10 rounded-full mx-auto mb-2 flex items-center justify-center text-sm font-bold"
                  style={{
                    background: badge.earned
                      ? 'linear-gradient(135deg, var(--accent), var(--accent)80)'
                      : 'var(--bg-card)',
                    color: badge.earned ? 'white' : 'var(--text-muted)',
                    border: badge.earned ? 'none' : '2px dashed var(--border)',
                  }}
                >
                  {badge.icon}
                </div>
                <p className="text-xs font-medium mb-0.5">{badge.name}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                  {badge.description}
                </p>
              </div>
            ))}
          </div>
        </Card>

        {/* Usage Analytics */}
        <Card>
          <h3 className="text-sm font-semibold mb-4">Usage Analytics</h3>
          <div className="grid grid-cols-2 gap-4">
            {/* Backtests over time - simple bar chart */}
            <div>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Backtests per Week</p>
              <div className="flex items-end gap-1 h-16">
                {[8, 12, 5, 15, 22, 18, 10, 14, 9, 20, 16, 11].map((v, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t-sm"
                    style={{
                      height: `${(v / 22) * 100}%`,
                      background: i === 11 ? 'var(--accent)' : 'var(--accent)40',
                    }}
                  />
                ))}
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-xs" style={{ color: 'var(--text-muted)', fontSize: '0.55rem' }}>12w ago</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)', fontSize: '0.55rem' }}>Now</span>
              </div>
            </div>

            {/* Alerts by type - horizontal bars */}
            <div>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Alerts by Type</p>
              <div className="space-y-2">
                {[
                  { label: 'Entry [C]', value: 312, color: 'var(--green)' },
                  { label: 'Exit [C]', value: 280, color: 'var(--red)' },
                  { label: 'Entry [L0]', value: 145, color: 'var(--green)' },
                  { label: 'Exit (Stop)', value: 98, color: 'var(--red)' },
                  { label: 'Entry [HM]', value: 57, color: 'var(--green)' },
                ].map((item) => (
                  <div key={item.label} className="flex items-center gap-2">
                    <span className="text-xs w-20 shrink-0" style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                      {item.label}
                    </span>
                    <div className="flex-1 h-2.5 rounded-full" style={{ background: 'var(--bg-input)' }}>
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${(item.value / 312) * 100}%`, background: item.color, opacity: 0.7 }}
                      />
                    </div>
                    <span className="text-xs w-8 text-right" style={{ color: 'var(--text-muted)', fontSize: '0.65rem' }}>
                      {item.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
