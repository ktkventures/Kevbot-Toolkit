'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  joinDate: string;
  status: 'active' | 'inactive' | 'suspended';
  segment: string;
  retentionDays: number;
  ltv: number;
}

const mockUsers: User[] = [
  { id: 'u1', name: 'Alex Kim', email: 'alex.k@email.com', role: 'Portfolio Builder', joinDate: '2025-11-15', status: 'active', segment: 'Power Creator', retentionDays: 127, ltv: 4620 },
  { id: 'u2', name: 'Sarah Mitchell', email: 'sarah.m@email.com', role: 'Pack Creator', joinDate: '2025-12-02', status: 'active', segment: 'Power Creator', retentionDays: 110, ltv: 3480 },
  { id: 'u3', name: 'James Torres', email: 'james.t@email.com', role: 'Subscriber', joinDate: '2026-01-10', status: 'active', segment: 'Engaged Subscriber', retentionDays: 71, ltv: 210 },
  { id: 'u4', name: 'Maria Lopez', email: 'maria.l@email.com', role: 'Strategy Builder', joinDate: '2026-01-22', status: 'inactive', segment: 'At Risk', retentionDays: 45, ltv: 580 },
  { id: 'u5', name: 'David Reed', email: 'david.r@email.com', role: 'Portfolio Builder', joinDate: '2026-02-05', status: 'active', segment: 'Growing Creator', retentionDays: 44, ltv: 1960 },
  { id: 'u6', name: 'Emily Chen', email: 'emily.c@email.com', role: 'Subscriber', joinDate: '2026-02-18', status: 'suspended', segment: 'Churned', retentionDays: 12, ltv: 0 },
  { id: 'u7', name: 'Michael Brown', email: 'michael.b@email.com', role: 'Subscriber', joinDate: '2026-03-01', status: 'active', segment: 'New User', retentionDays: 20, ltv: 60 },
  { id: 'u8', name: 'Lisa Park', email: 'lisa.p@email.com', role: 'Strategy Builder', joinDate: '2026-03-08', status: 'active', segment: 'New User', retentionDays: 13, ltv: 290 },
];

const segments = [
  { name: 'Power Creator', count: 2, color: 'var(--accent)', description: 'High revenue, daily active' },
  { name: 'Growing Creator', count: 1, color: 'var(--green)', description: 'Building content, growing subscribers' },
  { name: 'Engaged Subscriber', count: 1, color: 'var(--blue)', description: 'Active subscriber, regular logins' },
  { name: 'New User', count: 2, color: 'var(--teal)', description: 'Joined in last 30 days' },
  { name: 'At Risk', count: 1, color: 'var(--orange)', description: 'Declining activity' },
  { name: 'Churned', count: 1, color: 'var(--red)', description: 'No activity in 30+ days' },
];

const cohortRetention = [
  { cohort: 'Nov 2025', month1: 100, month2: 82, month3: 71, month4: 65 },
  { cohort: 'Dec 2025', month1: 100, month2: 78, month3: 68, month4: 0 },
  { cohort: 'Jan 2026', month1: 100, month2: 85, month3: 0, month4: 0 },
  { cohort: 'Feb 2026', month1: 100, month2: 0, month3: 0, month4: 0 },
];

const segmentColors: Record<string, string> = {
  'Power Creator': 'var(--accent)',
  'Growing Creator': 'var(--green)',
  'Engaged Subscriber': 'var(--blue)',
  'New User': 'var(--teal)',
  'At Risk': 'var(--orange)',
  'Churned': 'var(--red)',
};

function getRetentionColor(pct: number): string {
  if (pct === 0) return 'var(--bg-input)';
  if (pct >= 80) return 'var(--green)';
  if (pct >= 60) return 'var(--blue)';
  if (pct >= 40) return 'var(--orange)';
  return 'var(--red)';
}

export default function AdminUsersV4() {
  const [selectedSegment, setSelectedSegment] = useState<string | null>(null);

  const filteredUsers = useMemo(() => {
    if (!selectedSegment) return mockUsers;
    return mockUsers.filter((u) => u.segment === selectedSegment);
  }, [selectedSegment]);

  return (
    <div>
      <PageHeader
        title="User Management"
        subtitle="Segments, cohorts, and retention analysis"
        backHref="/admin"
      />

      {/* Segment Visualization */}
      <Card className="mb-6">
        <h3 className="text-sm font-semibold mb-4">User Segments</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {segments.map((seg) => (
            <button
              key={seg.name}
              onClick={() => setSelectedSegment(selectedSegment === seg.name ? null : seg.name)}
              className="p-3 rounded-lg text-left transition-all"
              style={{
                background: selectedSegment === seg.name ? seg.color + '22' : 'var(--bg-input)',
                border: `1px solid ${selectedSegment === seg.name ? seg.color : 'var(--border)'}`,
              }}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: seg.color }} />
                <span className="text-lg font-bold">{seg.count}</span>
              </div>
              <p className="text-xs font-medium truncate">{seg.name}</p>
              <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>{seg.description}</p>
            </button>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
        {/* Cohort Retention */}
        <Card>
          <h3 className="text-sm font-semibold mb-4">Cohort Retention</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="text-left py-2 pr-3 text-xs" style={{ color: 'var(--text-muted)' }}>Cohort</th>
                  <th className="text-center py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>M1</th>
                  <th className="text-center py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>M2</th>
                  <th className="text-center py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>M3</th>
                  <th className="text-center py-2 px-2 text-xs" style={{ color: 'var(--text-muted)' }}>M4</th>
                </tr>
              </thead>
              <tbody>
                {cohortRetention.map((row) => (
                  <tr key={row.cohort}>
                    <td className="py-1.5 pr-3 text-xs font-medium">{row.cohort}</td>
                    {[row.month1, row.month2, row.month3, row.month4].map((pct, i) => (
                      <td key={i} className="py-1.5 px-1">
                        <div
                          className="h-8 rounded flex items-center justify-center text-xs font-mono"
                          style={{
                            background: pct > 0 ? getRetentionColor(pct) + '33' : 'var(--bg-input)',
                            color: pct > 0 ? getRetentionColor(pct) : 'var(--text-muted)',
                          }}
                        >
                          {pct > 0 ? `${pct}%` : '--'}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Retention Curve */}
        <Card>
          <h3 className="text-sm font-semibold mb-4">Retention Curves</h3>
          <ChartPlaceholder label="Retention curve by cohort" height={200} />
        </Card>
      </div>

      {/* Filtered User List */}
      <Card>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold">
            {selectedSegment ? `${selectedSegment} Users` : 'All Users'}
          </h3>
          {selectedSegment && (
            <button
              onClick={() => setSelectedSegment(null)}
              className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--bg-input)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
            >
              Clear filter
            </button>
          )}
        </div>
        <div className="space-y-2">
          {filteredUsers.map((user) => (
            <div
              key={user.id}
              className="flex items-center gap-3 p-3 rounded-lg"
              style={{ background: 'var(--bg-input)' }}
            >
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
              >
                {user.name.charAt(0)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium">{user.name}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{user.role}</p>
              </div>
              <span
                className="px-2 py-0.5 rounded-full text-xs shrink-0"
                style={{ background: (segmentColors[user.segment] || 'var(--text-muted)') + '22', color: segmentColors[user.segment] || 'var(--text-muted)' }}
              >
                {user.segment}
              </span>
              <div className="text-right shrink-0">
                <p className="text-sm font-medium">${user.ltv.toLocaleString()}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>LTV</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
