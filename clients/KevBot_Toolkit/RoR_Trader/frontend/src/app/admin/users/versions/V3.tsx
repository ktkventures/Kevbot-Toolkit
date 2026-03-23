'use client';

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  joinDate: string;
  status: 'active' | 'inactive' | 'suspended';
}

const mockUsers: User[] = [
  { id: 'u1', name: 'Alex Kim', email: 'alex.k@email.com', role: 'Portfolio Builder', joinDate: '2025-11-15', status: 'active' },
  { id: 'u2', name: 'Sarah Mitchell', email: 'sarah.m@email.com', role: 'Pack Creator', joinDate: '2025-12-02', status: 'active' },
  { id: 'u3', name: 'James Torres', email: 'james.t@email.com', role: 'Subscriber', joinDate: '2026-01-10', status: 'active' },
  { id: 'u4', name: 'Maria Lopez', email: 'maria.l@email.com', role: 'Strategy Builder', joinDate: '2026-01-22', status: 'inactive' },
  { id: 'u5', name: 'David Reed', email: 'david.r@email.com', role: 'Portfolio Builder', joinDate: '2026-02-05', status: 'active' },
  { id: 'u6', name: 'Emily Chen', email: 'emily.c@email.com', role: 'Subscriber', joinDate: '2026-02-18', status: 'suspended' },
  { id: 'u7', name: 'Michael Brown', email: 'michael.b@email.com', role: 'Subscriber', joinDate: '2026-03-01', status: 'active' },
  { id: 'u8', name: 'Lisa Park', email: 'lisa.p@email.com', role: 'Strategy Builder', joinDate: '2026-03-08', status: 'active' },
];

const statusColors: Record<string, { color: string; bg: string }> = {
  active: { color: 'var(--green)', bg: 'var(--green-muted)' },
  inactive: { color: 'var(--text-muted)', bg: 'var(--bg-input)' },
  suspended: { color: 'var(--red)', bg: 'var(--red-muted)' },
};

export default function AdminUsersV3() {
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    if (!search) return mockUsers;
    const q = search.toLowerCase();
    return mockUsers.filter((u) => u.name.toLowerCase().includes(q) || u.email.toLowerCase().includes(q) || u.role.toLowerCase().includes(q));
  }, [search]);

  return (
    <div>
      <PageHeader
        title="User Management"
        subtitle={`${mockUsers.length} users`}
        backHref="/admin"
      />

      <Card>
        <div className="mb-4">
          <input
            type="text"
            placeholder="Search users..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          />
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                <th className="text-left py-2 px-2 font-medium" style={{ color: 'var(--text-muted)' }}>Name</th>
                <th className="text-left py-2 px-2 font-medium" style={{ color: 'var(--text-muted)' }}>Email</th>
                <th className="text-left py-2 px-2 font-medium" style={{ color: 'var(--text-muted)' }}>Role</th>
                <th className="text-left py-2 px-2 font-medium" style={{ color: 'var(--text-muted)' }}>Joined</th>
                <th className="text-left py-2 px-2 font-medium" style={{ color: 'var(--text-muted)' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((user) => (
                <tr key={user.id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td className="py-2 px-2 font-medium">{user.name}</td>
                  <td className="py-2 px-2" style={{ color: 'var(--text-muted)' }}>{user.email}</td>
                  <td className="py-2 px-2" style={{ color: 'var(--text-secondary)' }}>{user.role}</td>
                  <td className="py-2 px-2" style={{ color: 'var(--text-muted)' }}>{user.joinDate}</td>
                  <td className="py-2 px-2">
                    <span className="px-2 py-0.5 rounded-full text-xs" style={{ background: statusColors[user.status].bg, color: statusColors[user.status].color }}>
                      {user.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && (
            <p className="text-center py-6 text-sm" style={{ color: 'var(--text-muted)' }}>No users found</p>
          )}
        </div>
      </Card>
    </div>
  );
}
